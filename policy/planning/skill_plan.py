import torch
from torch import nn
import numpy as np

import IPython
e = IPython.embed
import dill as pickle
from policy.skill.skill_vae import TSkillCVAE


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class TSkillPlan(nn.Module):
    """ Transformer Skill CVAE Module for encoding/decoding skill sequences"""
    def __init__(self, transformer, vae: TSkillCVAE,
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
        num_img_feats = 16 # Hardcoded
        self.image_proj = nn.Linear(self.stt_encoder.num_channels, self.hidden_dim)
        self.image_feat_norm = self.norm([num_img_feats, self.hidden_dim])

        ### Inputs
        self.src_state_proj = nn.Linear(self.state_dim, self.hidden_dim)  # project qpos -> hidden
        self.src_state_norm = self.norm(self.hidden_dim)
        self.src_norm = self.norm(self.hidden_dim)

        ### Outputs
        self.tgt_norm = self.norm(self.hidden_dim)
        self.z_proj = nn.Linear(self.hidden_dim, self.z_dim) # project hidden -> latent
        self.tgt_z_proj = nn.Linear(self.z_dim, self.hidden_dim) # project latent -> hidden

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
            images: (bs, seq, num_cam, 3, h, w) | (bs, seq, num_cam, h*w, c & hidden)
            seq_pad_mask: Padding mask for input sequence (bs, seq)
            skill_pad_mask: Padding mask for skill sequence (bs, max_num_skills)
        """
        qpos = data["state"].to(self._device)
        actions = data["actions"].to(self._device) if data["actions"] is not None else None
        is_training = actions is not None
        if is_training:
            seq_pad_mask = data["seq_pad_mask"].to(self._device, torch.bool)
        skill_pad_mask = data["skill_pad_mask"].to(self._device, torch.bool)

        ### Get autoregressive masks, if applicable
        if self.vae.autoregressive_decode and is_training:
            if self.vae.conditional_decode: # Use full decoder mask from padded dataset
                dec_src_mask = data["dec_src_mask"][0,...].to(self._device)
                dec_mem_mask = data["dec_mem_mask"][0,...].to(self._device)
            else: # Only allow attention on one skill at a time
                dec_src_mask = ~torch.diag(torch.ones(self.max_skill_len)).to(self._device, torch.bool)
                dec_mem_mask = data["dec_mem_mask"][0,:,-self.max_skill_len:].to(self._device)
            dec_tgt_mask = data["dec_tgt_mask"][0,...].to(self._device)
        else:
            dec_tgt_mask = dec_src_mask = dec_mem_mask = None

        ### Calculate image features or use precalculated from dataset
        use_precalc = kwargs.get("use_precalc",False)
        if use_precalc and all(x in data.keys() for x in ("img_feat","img_pe")):
            img_src = data["img_feat"].transpose(0,1).to(self._device) # (seq, bs, num_cam, h*w, c)
            img_pe = data["img_pe"].transpose(0,1).to(self._device)
            goal_src = data["goal_feat"].transpose(0,1).to(self._device) # (1, bs, num_cam, h*w, c)
            goal_pe = data["goal_pe"].transpose(0,1).to(self._device)
        else:
            images = data["rgb"].to(self._device)
            img_src, img_pe = self.stt_encoder(images) # (seq, bs, num_cam, h*w, c)
            goal = data["goal"].to(self._device)
            goal_src, goal_pe = self.stt_encoder(goal)

        ### Autoregressively plan skills
        bs, MNS = skill_pad_mask.shape
        if is_training:
            vae_out = self.vae(data, use_precalc=use_precalc)
            z_tgt0 = torch.zeros(1,bs,self.z_dim, device=self._device)
            z_tgt = vae_out["mu"].permute(1,0,2) # (skill_seq, bs, z_dim)
            z_tgt = torch.vstack((z_tgt0, z_tgt[:-1,...]))
            plan_tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(z_tgt.shape[0], device=self._device)
            z_hat = self.forward_plan((goal_src, goal_pe), 
                                       qpos[:,:1,...], 
                                       (img_src[:1,...], img_pe[:1,...]),
                                       z_tgt, skill_pad_mask,
                                       None, None, plan_tgt_mask) # (skill_seq, bs, latent_dim)
            
            ### Decode skills one at a time, possibly with conditional state & image info from current state
            a_hat = self.vae.sequence_decode(z_hat, qpos, actions, (img_src, img_pe),
                                            seq_pad_mask, skill_pad_mask,
                                            dec_src_mask, dec_mem_mask, dec_tgt_mask)
            
            a_hat = a_hat.permute(1,0,2) # (bs, seq, act_dim)
        else:
            z_tgt = data["z_tgt"] # (tgt_seq, bs, z_dim)
            plan_tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(z_tgt.shape[0], device=self._device)
            plan_skill_pad_mask = torch.zeros(z_tgt.shape[1], z_tgt.shape[0])
            z_hat = self.forward_plan((goal_src, goal_pe),
                                    qpos[:,:1,...],
                                    (img_src[:1,...], img_pe[:1,...]),
                                    z_tgt, plan_skill_pad_mask,
                                    None, None, plan_tgt_mask) # (skill_seq, bs, latent_dim)
        
            a_hat = vae_out = None

        z_hat = z_hat.permute(1,0,2) # (bs, skill_seq, latent_dim)
        
        return dict(a_hat=a_hat, z_hat=z_hat, vae_out=vae_out)
        

    def forward_plan(self, goal_info, qpos, img_info, 
                     tgt, tgt_pad_mask, 
                     src_mask=None, mem_mask=None, tgt_mask=None):
        """Plan skill sequence based on current robot state and image input with goal state"""
        bs, _, _ = qpos.shape # Use bs for masking
        img_src, img_pe = img_info # (1, bs, num_cam, h*w, c&hidden)
        goal_src, goal_pe = goal_info # (1, bs, num_cam, h*w, c&hidden)

        ### type embeddings
        type_embed = self.input_embed.weight * self.input_embed_scale_factor
        type_embed = type_embed.unsqueeze(1).repeat(1, bs, 1) # (3, bs, hidden_dim)

        # position
        qpos_src = self.src_state_proj(qpos) # (bs, 1, hidden_dim)
        qpos_src = self.src_state_norm(qpos_src).permute(1, 0, 2) # (1, bs, hidden_dim)
        qpos_src = qpos_src + type_embed[0, :, :] # add type 1 embedding
        qpos_pe = torch.zeros_like(qpos_src, device=self._device) # No pe for current qpos

        # image
        img_src = self.image_proj(img_src) # (1, bs, num_cam, h*w, hidden)
        img_src = self.image_feat_norm(img_src)
        img_pe = img_pe * self.img_pe_scale_factor
        img_src = img_src.flatten(2,3) # (1, bs, num_cam*h*w, hidden)
        img_pe = img_pe.flatten(2,3)
        img_src = img_src.permute(0, 2, 1, 3) # (1, h*num_cam*w, bs, hidden)
        img_pe = img_pe.permute(0, 2, 1, 3) # sinusoidal skill pe
        img_src = img_src.flatten(0,1) # (num_cam*h*w, bs, hidden)
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
        src = torch.cat([qpos_src, img_src, goal_src], axis=0) # (1 + 2*num_cam*num_feats, bs, hidden_dim)
        src_pe = torch.cat([qpos_pe, img_pe, goal_pe], axis=0) * self.src_pos_scale_factor
        # Add and norm
        src = src + src_pe
        src = self.src_norm(src)

        # tgt
        tgt = self.tgt_z_proj(tgt) # (MNS|<, bs, hidden_dim)
        tgt_pe = self.get_pos_table(tgt.shape[0]).permute(1,0,2) * self.tgt_pos_scale_factor # (MNS|<, 1, hidden_dim)
        tgt_pe = tgt_pe.repeat(1, bs, 1)  # (MNS|<, bs, hidden_dim)
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
