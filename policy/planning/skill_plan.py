import torch
from torch import nn
import numpy as np

import dill as pickle
from policy.skill.skill_vae import TSkillCVAE, top_k_sampling
from policy.dataset.masking_utils import get_dec_ar_masks, get_plan_ar_masks


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
                 goal_mode,
                 obs_history,
                 stt_encoder, 
                 device, 
                 **kwargs):
        """ Initializes the model.
        Parameters:
            transformer: pytorch transformer to use in prediction
            vae: TSkillVAE to use to decode
            conditional_plan: whether to plan each skill with current conditional image/state info
            goal_mode: goal mode to use, currently image or one-hot
            device: device to operate on
            stt_encoder: custom state encoder if desired. Will use VAE by default.
        """
        super().__init__()
        ### General args
        self.transformer = transformer
        self.conditional_plan = conditional_plan
        self.hidden_dim = transformer.d_model
        self.goal_mode = goal_mode
        self.num_obs = obs_history
        self.vae = vae
        self.z_dim = self.vae.z_dim
        self.num_skills = self.vae.num_skills
        self.state_dim = self.vae.state_dim
        self.max_skill_len = self.vae.max_skill_len
        self._device = device
        self.norm = nn.LayerNorm

        ### Get a sinusoidal position encoding table for a given sequence size
        self.get_pos_table = lambda x: get_sinusoid_encoding_table(x, self.hidden_dim).to(self._device) # (1, x, hidden_dim)
        # Create pe scaling factors
        self.src_qpos_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.src_img_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.tgt_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.img_pe_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.goal_pe_scale_factor = nn.Parameter(torch.ones(1) * 0.01)

        # learned embeddings for the 4 input types (state, image, goal)
        self.input_embed = nn.Embedding(self.num_skills+1, self.hidden_dim) 

        ### Backbone
        self.stt_encoder = stt_encoder
        self.num_img_feats = 16 # Hardcoded
        self.image_proj = nn.Sequential(nn.ReLU(),
                                        nn.Linear(self.stt_encoder.num_channels, int(self.hidden_dim*3/8)))
        self.image_feat_norm = self.norm([self.num_img_feats, self.hidden_dim])

        ### Inputs
        if self.goal_mode == "one-hot":
            self.goal_input_size = 90 # Hardcoded
        else:
            self.goal_input_size = 512
        self.goal_proj = nn.Sequential(nn.Linear(512, 1024),
                                        nn.GELU(),
                                        nn.Linear(1024,2048),
                                        nn.GELU(),
                                        nn.Linear(2048,self.hidden_dim)) # Hardcoded
        self.goal_feat_norm = self.norm(self.hidden_dim)
        self.src_state_proj = nn.Linear(self.state_dim, int(self.hidden_dim/4))  # project qpos -> hidden
        self.src_img_proj = nn.Linear(self.num_img_feats,1)
        self.src_state_norm = self.norm(self.hidden_dim)
        self.src_norm = self.norm(self.hidden_dim)

        ### Outputs
        self.tgt_norm = self.norm(self.hidden_dim)
        self.z_proj = nn.Linear(self.hidden_dim, self.num_skills) # project hidden -> latent
        self.tgt_start_token = self.num_skills

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
        actions = data["actions"].to(self._device) if data["actions"] is not None else None
        is_training = actions is not None
        seq_pad_mask = data["seq_pad_mask"].to(self._device, torch.bool)
        skill_pad_mask = data["skill_pad_mask"].to(self._device, torch.bool)
        BS, MNS = skill_pad_mask.shape

        if is_training: # Get the qpos and images where the model will make next skill prediciton (every max_skill_len)
            sc = self.max_skill_len
            qpos_plan = qpos[:,::sc,:] # (bs, MNS, state_dim)
        else: # only pass in the relevant timesteps
            sc = 1 # Only pass in obs at each planning timestep
            qpos_plan = qpos

        ### Image calculation or features
        use_precalc = kwargs.get("use_precalc",False)
        if use_precalc:
            img_src = data["img_feat"][:,::sc,...].transpose(0,1).to(self._device) # (seq, bs, num_cam, h*w, c)
            img_pe = data["img_pe"][:,::sc,...].transpose(0,1).to(self._device) # (1, bs, num_cam, h*w, hidden)
        else:
            images = data["rgb"][:,::sc,...].to(self._device)
            img_src, img_pe = self.stt_encoder(images) # (seq, bs, num_cam, h*w, c)
            # data["img_feat"] = img_src.transpose(0,1) # VAE expects (bs, seq, ...)
            # data["img_pe"] = img_pe.transpose(0,1)
        img_info_plan = (img_src, img_pe)
        num_cam = img_src.shape[2]

        goal_src = data["goal"].to(self._device)

        ### Get conditional planning masks if applicable
        plan_src_mask, plan_mem_mask, plan_tgt_mask = get_plan_ar_masks(self.num_img_feats*num_cam, MNS, self.goal_mode, self.num_obs, device=self._device)

        if is_training:
            # Get target skills from vae
            vae_out = self.vae.forward(data, use_precalc=use_precalc)

            # Create tgt shifted right 1 token
            z_tgt0 = self.tgt_start_token * torch.ones(1,BS,device=self._device) # (1, bs)
            z_tgt = vae_out["idx"].clone().detach().permute(1,0) # (skill_seq, bs)
            z_tgt = torch.vstack((z_tgt0, z_tgt[:-1,...])).to(torch.int) # (skill_seq, bs)
        else:
            z_tgt = data["z_tgt"].permute(1,0) # (tgt_seq, bs)
            vae_out = None

        ### Autoregressively plan skills
        z_hat = self.forward_plan(goal_src, qpos_plan, img_info_plan,
                                    z_tgt, skill_pad_mask,
                                    plan_mem_mask, plan_tgt_mask) # (skill_seq, bs, num_skills)
        
        if torch.any(torch.isnan(z_hat)):
            with open("ERROR_DATA.pickle",'+wb') as f:
                pickle.dump(data, f)
            raise ValueError(f"NaNs encountered during planning! Data saved to ERROR_DATA.pickle")
        
        z_idx = top_k_sampling(z_hat.clone().detach(), 5, 1, device=self._device).squeeze(-1) # (skill_seq, bs)
        z_code = self.vae.vq.indices_to_codes(z_idx)
        
        ### Get decoder masks
        if self.vae.autoregressive_decode:
            dec_mem_mask, dec_tgt_mask = get_dec_ar_masks(MNS, self.max_skill_len, self.vae.decoder_obs, device=self._device)
        else:
            dec_mem_mask = dec_tgt_mask = None
        ### Decoder skills
        a_hat = self.vae.skill_decode(z_code,
                                      seq_pad_mask, skill_pad_mask,
                                      dec_mem_mask, dec_tgt_mask)

        a_hat = a_hat.permute(1,0,2) # (bs, seq, act_dim)

        if not is_training:
            z_hat = z_idx # (skill_seq, bs)
            z_hat = z_hat.transpose(0,1) # (skill_seq, bs)
        else:
            z_hat = z_hat.permute(1,0,2) # (bs, skill_seq, num_skills)
        
        return dict(z_hat=z_hat ,a_hat=a_hat, vae_out=vae_out)
        

    def forward_plan(self, goal_info, qpos, img_info, 
                     tgt, tgt_pad_mask, 
                     mem_mask=None, tgt_mask=None,
                     **kwargs):
        """Plan skill sequence based on current robot state and image input with goal state
        args:
            - goal_info: tuple of
                - goal_src: (1 | MNS|<, bs, num_cam, h*w, c)
            - qpos: state information (bs, 1 | MNS|<, state_dim)
            - img_info: tuple of
                - img_src: (1 | MNS|<, bs, num_cam, h*w, c)
                - img_pe (1 | MNS|<, bs, num_cam, h*w, hidden)
            - tgt: skill sequence indices (MNS|<, bs)
            - tgt_pad_mask: square subsequent mask (MNS|<, MNS|<)
        """
        bs, MNS, _ = qpos.shape # Use bs for masking
        img_src, img_pe = img_info # (1, bs, num_cam, h*w, c&hidden)
        goal_src = goal_info # (1, bs, num_cam, h*w, c&hidden) | (bs, num_tasks)/None

        # position
        state_src = self.src_state_proj(qpos).permute(1, 0, 2) # (bs, 1|MNS, hidden_dim/4)

        # image
        img_pe = img_pe * self.img_pe_scale_factor
        img_src = img_src + img_pe
        img_src = self.image_proj(img_src) # (1|MNS, bs, num_cam, h*w, 3*hidden/8)
        img_src = img_src.transpose(3,4) # (1|MNS, bs, num_cam, 3*hidden/8, h*w)
        img_src = self.src_img_proj(img_src).squeeze(-1) # (1|MNS, bs, num_cam, 3*hidden/8)
        img_src = img_src.flatten(2,3) # (1|MNS, bs, 3*hidden/4)

        context = torch.cat((state_src, img_src), dim=-1) # (1|MNS, bs, hidden)
        context_pe = self.get_pos_table(context.shape[0]).permute(1, 0, 2) * self.src_qpos_pos_scale_factor
        context = self.src_norm(context)
        context = context + context_pe

        # goal
        goal_src = self.goal_proj(goal_src) # (bs, hidden)
        goal_src = self.goal_feat_norm(goal_src)
        goal_src = goal_src.unsqueeze(0) # (1, bs, hidden)

        # src
        src = torch.cat([context, goal_src], axis=0) # (1+MNS, bs, hidden_dim)
        
        # tgt
        tgt = self.input_embed(tgt) # (MNS|<, bs, hidden_dim)
        tgt_pe = self.get_pos_table(tgt.shape[0]).permute(1,0,2).repeat(1, bs, 1) * self.tgt_pos_scale_factor # (MNS|<, bs, hidden_dim)
        tgt = self.tgt_norm(tgt)
        tgt = tgt + tgt_pe

        # query encoder model
        z_output = self.transformer(tgt, src, tgt_mask=tgt_mask, memory_mask=mem_mask,
                                    tgt_key_padding_mask=None,
                                    memory_key_padding_mask=None)

        z = self.z_proj(z_output) # (skill_seq, bs, num_skills)

        return z
    

    def get_action(self, data, t, replan=False):
        # Get image features from state encoder
        img_src, img_pe = self.stt_encoder(data["rgb"]) # (seq, bs, num_cam, h*w, c)
        img_src = img_src.transpose(0,1) # (bs, seq, ...)
        img_pe = img_pe.transpose(0,1)
        bs = img_src.shape[0]

        if t==0 or replan:
            self.execution_data = dict()
            self.execution_data["t_plan"] = 0

        t_act = self.execution_data["t_plan"] % self.max_skill_len
        # Get next skill prediction if appropriate
        if t_act == 0:
            # Set or update data related to current planning segment
            if self.execution_data["t_plan"] == 0:
                self.execution_data["z_tgt"] = self.tgt_start_token * torch.ones(bs,1,device=self._device).to(torch.int) # (bs,1)
                self.execution_data["actions"] = None
                self.execution_data["img_feat"] = img_src # (bs, seq, ...)
                self.execution_data["img_pe"] = img_pe
                self.execution_data["state"] = data["state"] # (bs, seq, state_dim)
            else:
                self.execution_data["img_feat"] = torch.cat((self.execution_data["img_feat"], img_src), dim=1)
                self.execution_data["img_pe"] = torch.cat((self.execution_data["img_pe"], img_pe), dim=1)
                self.execution_data["state"] = torch.cat((self.execution_data["state"], data["state"]), dim=1)

            # Set/update data required for model call
            self.execution_data["skill_pad_mask"] = torch.zeros(bs,self.execution_data["z_tgt"].shape[1])
            self.execution_data["seq_pad_mask"] = torch.zeros(bs,self.execution_data["z_tgt"].shape[1]*self.max_skill_len)
            self.execution_data["goal"] = data["goal"]

            out = self.forward(self.execution_data, use_precalc=True)

            z_hat = out["z_hat"]
            self.execution_data["z_hat"] =  z_hat # (bs, sk)
            self.execution_data["z_tgt"] = torch.cat((self.execution_data["z_tgt"], z_hat[:,-1:]), dim=1).to(torch.int)
            self.execution_data["a_hat"] = out["a_hat"] # (bs, seq, act_dim)

        a_t = self.execution_data["a_hat"][:,t_act-self.max_skill_len,:]
        self.execution_data["t_plan"] += 1
        
        return a_t


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
