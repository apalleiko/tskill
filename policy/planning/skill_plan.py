import torch
from torch import nn
import numpy as np

import IPython
e = IPython.embed
import dill as pickle
from policy.skill.skill_vae import TSkillCVAE
from policy.dataset.masking_utils import get_dec_ar_masks, get_plan_ar_masks, get_skill_pad_from_seq_pad


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class TSkillPlan(nn.Module):
    """ Transformer Skill CVAE Module for encoding/decoding skill sequences"""
    def __init__(self, transformer: torch.nn.Transformer, 
                 vae: TSkillCVAE, 
                 conditional_plan,
                 device, **kwargs):
        """ Initializes the model.
        Parameters:
            transformer: pytorch transformer to use in prediction
            vae: TSkillVAE to use to decode
            device: device to operate on
        """
        super().__init__()
        ### General args
        self.transformer = transformer
        self.conditional_plan = conditional_plan
        self.hidden_dim = transformer.d_model
        self.vae = vae
        self.z_dim = self.vae.z_dim
        self.state_dim = self.vae.state_dim
        self.max_skill_len = self.vae.max_skill_len
        self._device = device
        self.norm = nn.LayerNorm

        ### Get a sinusoidal position encoding table for a given sequence size
        self.get_pos_table = lambda x: get_sinusoid_encoding_table(x, self.hidden_dim).to(self._device) # (1, x, hidden_dim)
        # Create pe scaling factors
        self.src_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.001)
        self.tgt_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.img_pe_scale_factor = nn.Parameter(torch.ones(1) * 0.001)

        # learned embeddings for the 4 input types (state, image, goal)
        self.input_embed = nn.Embedding(3, self.hidden_dim) 
        self.input_embed_scale_factor = nn.Parameter(torch.ones(3, 1) * 0.01)

        ### Backbone
        self.stt_encoder = self.vae.stt_encoder
        self.num_img_feats = 16 # Hardcoded
        self.image_proj = nn.Linear(self.stt_encoder.num_channels, self.hidden_dim)
        self.image_feat_norm = self.norm([self.num_img_feats, self.hidden_dim])

        ### Inputs
        self.src_state_proj = nn.Linear(self.state_dim, self.hidden_dim)  # project qpos -> hidden
        self.src_state_norm = self.norm(self.hidden_dim)
        self.src_norm = self.norm(self.hidden_dim)

        ### Outputs
        self.tgt_norm = self.norm(self.hidden_dim)
        self.z_proj = nn.Linear(self.hidden_dim, self.z_dim) # project hidden -> latent
        self.tgt_z_proj = nn.Linear(self.z_dim, self.hidden_dim) # project latent -> hidden
        self.tgt_start_token = nn.Parameter(torch.zeros(1,1,self.z_dim))
        # self.tgt_start_token = torch.zeros(1,1,self.z_dim)

        # Other
        self.metrics = dict()
        self.to(self._device)
        self.apply(self.init_weights)
    

    def forward(self, data, **kwargs):
        """
        data:
            goal: (bs, 1, num_cam, 3, h, w) | (bs, 1, num_cam, h*w, c & hidden)
            qpos: (bs, seq, state_dim)
            actions: (bs, seq, action_dim)
            rgb | (img_feat, img_pe): (bs, seq, num_cam, 3, h, w) | (bs, seq, num_cam, h*w, c & hidden)
            seq_pad_mask: Padding mask for input sequence (bs, seq)
            skill_pad_mask: Padding mask for skill sequence (bs, max_num_skills)
            *autoregressive masks
            *z_tgt: (bs, tgt_seq, z_dim)
        """
        qpos = data["state"].to(self._device)
        qpos_plan = qpos[:,:,:self.state_dim]
        actions = data["actions"].to(self._device) if data["actions"] is not None else None
        is_training = actions is not None
        if is_training:
            seq_pad_mask = data["seq_pad_mask"].to(self._device, torch.bool)
        skill_pad_mask = data["skill_pad_mask"].to(self._device, torch.bool)
        BS, MNS = skill_pad_mask.shape

        ### Image calculation or features
        use_precalc = kwargs.get("use_precalc",False)
        if use_precalc and all(x in data.keys() for x in ("img_feat","img_pe")):
            img_src = data["img_feat"].transpose(0,1).to(self._device) # (seq, bs, num_cam, h*w, c)
            img_pe = data["img_pe"].transpose(0,1).to(self._device) # (1, bs, num_cam, h*w, hidden)
        else:
            images = data["rgb"].to(self._device)
            img_src, img_pe = self.stt_encoder(images) # (seq, bs, num_cam, h*w, c)

        ### Goal image or features
        if "goal_feat" in data.keys():
            goal_src = data["goal_feat"].transpose(0,1).to(self._device) # (1, bs, num_cam, h*w, c)
            goal_pe = data["goal_pe"].transpose(0,1).to(self._device) # (1, bs, num_cam, h*w, hidden)
        else:
            goal = data["goal"].to(self._device)
            goal_src, goal_pe = self.stt_encoder(goal)

        num_cam = img_src.shape[2]

        ### Get autoregressive masks, if applicable
        if self.vae.autoregressive_decode and is_training:
            dec_src_mask, dec_mem_mask, dec_tgt_mask = get_dec_ar_masks(self.num_img_feats*num_cam, self.max_skill_len, device=self._device)
        else:
            dec_tgt_mask = dec_src_mask = dec_mem_mask = None

        ### Get conditional planning masks if applicable
        if self.conditional_plan:
            plan_src_mask, plan_mem_mask, plan_tgt_mask = get_plan_ar_masks(self.num_img_feats*num_cam, MNS, device=self._device)
            if is_training: # Get the qpos and images where the model will make next skill prediciton (every max_skill_len)
                qpos_plan = qpos_plan[:,::self.max_skill_len,:] # (bs, MNS, state_dim)
                img_info_plan = (img_src[::self.max_skill_len,...], img_pe[::self.max_skill_len,...]) # (MNS, bs, ...)
            goal_info = (goal_src, goal_pe)
        else:
            plan_src_mask = plan_mem_mask = None
            _, _, plan_tgt_mask = get_plan_ar_masks(self.num_img_feats*num_cam, MNS, device=self._device)
            qpos_plan = qpos_plan[:,:1,:] # (bs, 1, state_dim)
            img_info_plan = (img_src[:1,...], img_pe[:1,...]) # (1, bs, ...)
            goal_info = (goal_src, goal_pe)

        ### Autoregressively plan skills
        if is_training:
            # Whether to seperate planning of skills and actions from reconstruction in computation graph
            sep_vae_grad = kwargs.get("sep_vae_grad",False)

            # Get target skills from vae
            vae_out = self.vae(data, use_precalc=use_precalc)
            
            # Create tgt shifted right 1 token
            z_tgt0 = self.tgt_start_token.repeat(1,BS,1)
            z_tgt = vae_out["mu"].permute(1,0,2) # (skill_seq, bs, z_dim)
            if sep_vae_grad: # Turn off vae grad calculation for planning z
                z_tgt = z_tgt.clone().detach()
            z_tgt = torch.vstack((z_tgt0, z_tgt[:-1,...]))
            
            # Plan skills
            z_hat = self.forward_plan(goal_info, 
                                      qpos_plan, 
                                      img_info_plan,
                                      z_tgt, skill_pad_mask,
                                      plan_src_mask, plan_mem_mask, plan_tgt_mask) # (skill_seq, bs, latent_dim)
            
            ### Decode skills
            if sep_vae_grad: # Turn off vae grad calculation for sequence decoding on pred z
                self.vae.requires_grad_(False)
            a_hat = self.vae.sequence_decode(z_hat, qpos, actions, (img_src, img_pe),
                                             seq_pad_mask, skill_pad_mask,
                                             dec_src_mask, dec_mem_mask, dec_tgt_mask)
            if sep_vae_grad:
                self.vae.requires_grad_(True)

            a_hat = a_hat.permute(1,0,2) # (bs, seq, act_dim)
        else:
            z_tgt = data["z_tgt"].permute(1,0,2) # (tgt_seq, bs, z_dim)
            z_hat = self.forward_plan(goal_info,
                                      qpos,
                                      (img_src, img_pe),
                                      z_tgt, skill_pad_mask,
                                      plan_src_mask, plan_mem_mask, plan_tgt_mask) # (skill_seq, bs, latent_dim)

            a_hat = vae_out = None

        z_hat = z_hat.permute(1,0,2) # (bs, skill_seq, latent_dim)
        
        return dict(a_hat=a_hat, z_hat=z_hat, vae_out=vae_out)
        

    def forward_plan(self, goal_info, qpos, img_info, 
                     tgt, tgt_pad_mask, 
                     src_mask=None, mem_mask=None, tgt_mask=None):
        """Plan skill sequence based on current robot state and image input with goal state
        args:
            - goal_info: tuple of
                - goal_src: (1 | MNS|<, bs, num_cam, h*w, c)
                - goal_pe: (1 | MNS|<, bs, num_cam, h*w, hidden)
            - qpos: state information (bs, 1 | MNS|<, state_dim)
            - img_info: tuple of
                - img_src: (1 | MNS|<, bs, num_cam, h*w, c)
                - img_pe (1 | MNS|<, bs, num_cam, h*w, hidden)
            - tgt: skill sequence (MNS|<, bs, z_dim)
            - tgt_pad_mask: square subsequent mask (MNS|<, MNS|<)
        """
        bs, _, _ = qpos.shape # Use bs for masking
        img_src, img_pe = img_info # (1, bs, num_cam, h*w, c&hidden)
        goal_src, goal_pe = goal_info # (1, bs, num_cam, h*w, c&hidden)

        ### type embeddings
        type_embed = self.input_embed.weight * self.input_embed_scale_factor
        type_embed = type_embed.unsqueeze(1).repeat(1, bs, 1) # (3, bs, hidden_dim)

        # position
        state_src = self.src_state_proj(qpos) # (bs, 1|MNS, hidden_dim)
        state_src = self.src_state_norm(state_src).permute(1, 0, 2) # (1|MNS, bs, hidden_dim)
        state_src = state_src + type_embed[0, :, :] # add type 1 embedding
        # state_pe = torch.zeros_like(state_src, device=self._device) # No pe for current qpos
        state_pe = self.get_pos_table(state_src.shape[0]).permute(1, 0, 2) * self.src_pos_scale_factor
        state_pe = state_pe.repeat(1,bs,1)

        # image
        img_src = self.image_proj(img_src) # (1|MNS, bs, num_cam, h*w, hidden)
        img_src = self.image_feat_norm(img_src)
        img_pe = img_pe * self.img_pe_scale_factor
        img_src = img_src.flatten(2,3) # (1|MNS, bs, num_cam*h*w, hidden)
        img_pe = img_pe.flatten(2,3)
        img_src = img_src.permute(0, 2, 1, 3) # (1|MNS, h*num_cam*w, bs, hidden)
        img_pe = img_pe.permute(0, 2, 1, 3) # sinusoidal skill pe
        img_pos = self.get_pos_table(img_src.shape[0]).permute(1, 0, 2) * self.src_pos_scale_factor # (1|MSL, bs, hidden)
        img_pos = img_pos.unsqueeze(1) # (1|MSL, 1, bs, hidden)
        img_src = img_src + img_pos # Add temporal positional encoding
        img_src = img_src.flatten(0,1) # (num_cam*h*w|*MNS, bs, hidden)
        img_pe = img_pe.flatten(0,1)
        img_src = img_src + type_embed[1, :, :] # add type 2 embedding

        # goal
        goal_src = self.image_proj(goal_src) # (1, bs, num_cam, h*w, hidden)
        goal_src = self.image_feat_norm(goal_src)
        goal_pe = goal_pe * self.img_pe_scale_factor
        goal_src = goal_src.flatten(2,3) # (1, bs, num_cam*h*w, hidden)
        goal_pe = goal_pe.flatten(2,3)
        goal_src = goal_src.permute(0, 2, 1, 3) # (1, h*num_cam*w, bs, hidden)
        goal_pe = goal_pe.permute(0, 2, 1, 3) # sinusoidal skill pe
        goal_src = goal_src.flatten(0,1) # (num_cam*h*w, bs, hidden)
        goal_pe = goal_pe.flatten(0,1)
        goal_src = goal_src + type_embed[2, :, :] # add type 3 embedding

        # src
        src = torch.cat([state_src, img_src, goal_src], axis=0) # (1 + 2*num_cam*num_feats|*MNS, bs, hidden_dim)
        src_pe = torch.cat([state_pe, img_pe, goal_pe], axis=0)
        # Add and norm
        src = src + src_pe
        src = self.src_norm(src)

        # tgt
        tgt = self.tgt_z_proj(tgt) # (MNS|<, bs, hidden_dim)
        # TODO ONLY PASSING IN ZEROS FOR TGT
        # tgt = torch.zeros_like(tgt, device=self._device) # (MNS|<, bs, hidden_dim)
        tgt_pe = self.get_pos_table(tgt.shape[0]).permute(1,0,2).repeat(1, bs, 1) * self.tgt_pos_scale_factor # (MNS|<, bs, hidden_dim)
        # tgt_pe = torch.zeros_like(tgt, device=self._device)
        tgt_pad_mask = None # With isolated time steps, padding a skill leads to NaNs #TODO for all cases?
        tgt = tgt + tgt_pe
        tgt = self.tgt_norm(tgt)

        # query encoder model
        z_output = self.transformer(src=src, 
                                    src_key_padding_mask=None, 
                                    src_is_causal=False,
                                    src_mask = src_mask,
                                    memory_key_padding_mask=None,
                                    memory_mask=mem_mask,
                                    memory_is_causal=False, 
                                    tgt=tgt,
                                    tgt_key_padding_mask=tgt_pad_mask,
                                    tgt_is_causal=True,
                                    tgt_mask=tgt_mask) # (skill_seq, bs, hidden_dim)

        z = self.z_proj(z_output) # (skill_seq, bs, 2*latent_dim)

        return z
    

    def to(self, device):
        model = super().to(device)
        model._device = device
        model.vae._device = device
        return model
    
    
    def log_metric(self, metric, name, type=None): 
        if type is None:
            computed = metric
        elif type=="mean":
            computed = torch.mean(metric)
        elif type=="nonzero_mean":
            computed = torch.sum(metric) / torch.nonzero(metric).shape[0]
        elif type=="mean_along_seq":
            computed = torch.mean(metric, dim=0)

        self.metrics[name] = computed


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
