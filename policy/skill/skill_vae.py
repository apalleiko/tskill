# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class TSkillCVAE(nn.Module):
    """ Transformer Skill CVAE Module for encoding/decoding skill sequences"""
    def __init__(self, stt_encoder, encoder, decoder, state_dim, action_dim, max_skill_len, z_dim, device, **kwargs):
        """ Initializes the model.
        Parameters:
            stt_encoder: torch module of the state encoder to be used. See perception
            encoder: transformer encoder module
            decoder: transformer decoder module
            state_dim: robot state dimension. Size of input robot position (delta_pos)
            action_dim: robot action dimenstion.
            max_skill_len: Max number of actions that can be predicted from a skill
            z_dim: dimension of latent skill vectors
            device: device to operate on
        kwargs:
            autoregressive: Whether skill generation and action decoding is autoregressive TODO
            single_skill: Whether only one skill is generated per input sequence TODO
        """
        super().__init__()
        ### General args
        self.decoder = decoder
        self.encoder = encoder
        self.hidden_dim = decoder.d_model
        self.z_dim = z_dim
        self.max_skill_len = max_skill_len
        self._device = device
        self.autoregressive = kwargs.get("autoregressive",False)
        self.single_skill = kwargs.get("single_skill",False)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.norm = nn.LayerNorm

        ### Get a sinusoidal position encoding table for a given sequence size
        self.get_pos_table = lambda x: get_sinusoid_encoding_table(x, self.hidden_dim).to(self._device) # (1, x, hidden_dim)
        # Create pe scaling factors
        self.enc_src_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.001)
        self.enc_tgt_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.dec_z_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.dec_tgt_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.img_pe_scale_factor = nn.Parameter(torch.ones(1) * 0.001)

        # learned embeddings for the 4 input types (action, state, image, skill)
        self.input_embed = nn.Embedding(4, self.hidden_dim) 
        self.input_embed_scale_factor = nn.Parameter(torch.ones(4, 1) * 0.01)

        ### Backbone
        self.stt_encoder = stt_encoder
        self.image_feat_norm = self.norm([32, self.hidden_dim]) # Hardcoded

        ### Encoder
        self.enc_action_proj = nn.Linear(action_dim, self.hidden_dim) # project action -> hidden
        self.enc_action_norm = self.norm(self.hidden_dim)
        self.enc_state_proj = nn.Linear(state_dim, self.hidden_dim)  # project qpos -> hidden
        self.enc_state_norm = self.norm(self.hidden_dim)
        self.enc_image_proj = nn.Linear(32,1) #TODO Hardcoded
        self.enc_src_norm = self.norm(self.hidden_dim)
        self.enc_tgt_norm = self.norm(self.hidden_dim)

        ### Latent
        self.enc_z = nn.Linear(self.hidden_dim, self.z_dim*2) # project hidden -> latent [std; var]
        self.dec_z = nn.Linear(self.z_dim, self.hidden_dim) # project latent -> hidden

        ### Decoder
        self.dec_input_state_proj = nn.Linear(state_dim, self.hidden_dim)
        self.dec_input_state_norm = self.norm(self.hidden_dim)
        self.dec_input_z_norm = self.norm(self.hidden_dim)
        self.dec_src_norm = self.norm(self.hidden_dim)
        self.dec_tgt_norm = self.norm(self.hidden_dim)

        # self.dec_action_proj = nn.Linear(self.hidden_dim, action_dim)

        nl = self.hidden_dim
        action_head = []  # TODO make this an mlp?
        for n in (128, 64, 32):
            action_head.append(nn.Linear(nl, n)) 
            action_head.append(self.norm(n))
            action_head.append(nn.LeakyReLU())
            nl = n
        action_head.append(nn.Linear(nl, action_dim))
        self.dec_action_proj = nn.Sequential(*action_head)

        # Other
        self.metrics = dict()
        self.to(self._device)
        self.apply(self.init_weights)
    

    def forward(self, data, **kwargs):
        """
        data:
            qpos: bs, seq, qpos_dim
            images: bs, seq, num_cam, channel, height, width
            actions: bs, seq, action_dim
            seq_pad_mask: Padding mask for input sequence (bs, seq)
            skill_pad_mask: Padding mask for skill sequence (bs, max_num_skills)
            seq_mask: bs, seq, seq
            skill_mask: bs, max_num_skills, max_num_skills
        """
        qpos = data["state"].to(self._device)
        images = data["rgb"].to(self._device)
        actions = data["actions"].to(self._device)
        seq_pad_mask = data["seq_pad_mask"].to(self._device)
        skill_pad_mask = data["skill_pad_mask"].to(self._device)
        seq_mask = None # TODO data["seq_mask"][0,:,:].to(self._device)
        skill_mask = None # data["skill_mask"][0,:,:].to(self._device)

        # Calculate image features
        img_src, img_pe = self.stt_encoder(images) # (seq, bs, h*num_cam*w, c)
        img_src = self.image_feat_norm(img_src)

        max_num_skills = skill_pad_mask.shape[1]
        mu, logvar, z = self.forward_encode(qpos, actions, (img_src, img_pe), max_num_skills, 
                                            seq_pad_mask, skill_pad_mask, seq_mask, skill_mask)
        
        # Decode skills conditioned with state & image from sequence start ("current" state)
        run_subseq = kwargs.get("run_subseq", 0)
        if run_subseq > 0: # TODO Decode skills for subsequences of the predictions for training
            assert max_num_skills % run_subseq == 0, "max_num_skills not divisible by run_subseq"
            for i in range(int(max_num_skills / run_subseq)):
                seq0 = i*self.max_skill_len*run_subseq
                raise NotImplementedError
                
        else:
            a_hat = self.skill_decode(z, qpos[:,0,:], (img_src[0,...], img_pe[0,...]),
                                      skill_pad_mask, seq_pad_mask, skill_mask, seq_mask)

            a_hat = a_hat.permute(1,0,2) # Shift back to (bs, seq, act_dim)
            mu = mu.permute(1,0,2) # (bs, seq, latent_dim)
            logvar = logvar.permute(1,0,2)
            return dict(a_hat=a_hat, mu=mu, logvar=logvar)
        

    def forward_encode(self, qpos, actions, img_info, max_num_skills, 
                       src_pad_mask=None, tgt_pad_mask=None, 
                       src_mask=None, tgt_mask=None):
        """Encode skill sequence based on robot state, action, and image input sequence"""
        is_training = actions is not None # train or val
        bs, seq, _ = qpos.shape # Use bs for masking
        img_feat, img_pe = img_info # (seq, bs, h*num_cam*w, c)

        ### Obtain latent z from state,action,image sequence
        if is_training:
            enc_type_embed = self.input_embed.weight * self.input_embed_scale_factor
            enc_type_embed = enc_type_embed.unsqueeze(1).repeat(1, bs, 1) # (4, bs, hidden_dim)
            self.log_metric(enc_type_embed[0, 0, :], "action_type_vector", None) # TEMP METRIC
            self.log_metric(enc_type_embed[1, 0, :], "state_type_vector", None) # TEMP METRIC
            self.log_metric(enc_type_embed[2, 0, :], "img_type_vector", None) # TEMP METRIC
            self.log_metric(enc_type_embed[3, 0, :], "z_type_vector", None) # TEMP METRIC

            # project action sequences to hidden dim (bs, seq, action_dim)
            action_src = self.enc_action_proj(actions) # (bs, seq, hidden_dim)
            action_src = self.enc_action_norm(action_src).permute(1, 0, 2) # (seq, bs, hidden_dim)
            self.log_metric(action_src[:,0,:], "enc_action_vector_along_seq", "mean_along_seq") # TEMP METRIC
            self.log_metric(enc_type_embed[0, :, :], "mean_enc_action_type", "mean") # METRIC
            self.log_metric(action_src, "mean_enc_action_src", "nonzero_mean") # METRIC
            action_src = action_src + enc_type_embed[0, :, :] # add type 1 embedding

            # project position sequences to hidden dim (bs, seq, state_dim)
            qpos_src = self.enc_state_proj(qpos) # (bs, seq, hidden_dim)
            qpos_src = self.enc_state_norm(qpos_src).permute(1, 0, 2) # (seq, bs, hidden_dim)
            self.log_metric(qpos_src[:,0,:], "enc_state_vector_along_seq", "mean_along_seq") # TEMP METRIC
            self.log_metric(enc_type_embed[1, :, :], "mean_enc_qpos_type", "mean") # METRIC
            self.log_metric(qpos_src,"mean_enc_qpos_src","nonzero_mean") # METRIC
            qpos_src = qpos_src + enc_type_embed[1, :, :] # add type 2 embedding

            # project img with local pe to correct size
            img_pe = img_pe * self.img_pe_scale_factor
            self.log_metric(img_pe, "enc_img_pe_vector", "mean_along_seq") # TEMP METRIC
            img_src = (img_feat + img_pe) # TODO should this PE be scaled or changed?
            img_src = img_src.permute(0, 1, 3, 2) # (seq, bs, c, h*num_cam*w)
            img_src = self.enc_image_proj(img_src).squeeze(-1) # (seq, bs, c=hidden)
            self.log_metric(img_src[:,0,:], "enc_img_vector_along_seq", "mean_along_seq") # TEMP METRIC
            self.log_metric(enc_type_embed[2, :, :], "mean_enc_img_type", "mean") # METRIC
            self.log_metric(img_src, "mean_enc_img_src", "nonzero_mean") # METRIC
            img_src = img_src + enc_type_embed[2, :, :] # add type 3 embedding

            # Concatenate enc_src
            enc_src = torch.cat([qpos_src, action_src, img_src], axis=0) # (3*seq, bs, hidden_dim)
            # obtain seq position embedding for input
            enc_src_pe = self.get_pos_table(seq) * self.enc_src_pos_scale_factor
            enc_src_pe = enc_src_pe.permute(1, 0, 2)  # (seq, 1, hidden_dim)
            enc_src_pe = enc_src_pe.repeat(3, bs, 1)  # (3*seq, bs, hidden_dim)
            self.log_metric(enc_src_pe, "mean_enc_src_pe", "mean" ) # METRIC
            self.log_metric(enc_src_pe, "enc_src_pe_vector", "mean_along_seq") # TEMP METRIC
            # Add and norm
            enc_src = enc_src + enc_src_pe
            enc_src = self.enc_src_norm(enc_src)

            # encoder target
            enc_tgt = torch.zeros(max_num_skills, bs, self.hidden_dim).to(self._device) # (skill_seq, bs, hidden_dim)
            enc_tgt_pe = self.get_pos_table(max_num_skills).permute(1, 0, 2).repeat(1, bs, 1) * self.enc_tgt_pos_scale_factor
            self.log_metric(img_pe, "enc_tgt_pe_vector", "mean_along_seq") # TEMP METRIC
            self.log_metric(enc_tgt_pe, "mean_enc_tgt_pe", "mean") # METRIC
            # Add and norm
            enc_tgt = enc_tgt + enc_tgt_pe
            enc_tgt = self.enc_tgt_norm(enc_tgt)

            # TODO Keep causal mask? + enable combining with other masks
            # causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(max_num_skills).to(self._device)

            # Repeat same padding mask for new seq length (same pattern)
            src_pad_mask = src_pad_mask.repeat(1,3) 
            
            # rehsape src_mask and tgt_mask
            # TODO

            # query encoder model
            enc_output = self.encoder(src=enc_src, 
                                          src_key_padding_mask=src_pad_mask, 
                                          src_is_causal=False,
                                          src_mask = src_mask,
                                          tgt=enc_tgt,
                                          tgt_key_padding_mask=tgt_pad_mask,
                                          tgt_is_causal=False,
                                          tgt_mask = tgt_mask,
                                          memory_is_causal=False) # (skill_seq, bs, hidden_dim)
            self.log_metric(enc_output[:,0,:], "enc_output_vector_along_seq", "mean_along_seq") # TEMP METRIC
            self.log_metric(enc_output, "mean_enc_output", "nonzero_mean") # METRIC

            latent_info = self.enc_z(enc_output) # (skill_seq, bs, 2*latent_dim)
            mu = latent_info[:, :, :self.z_dim]
            logvar = latent_info[:, :, self.z_dim:]
            latent_sample = reparametrize(mu, logvar) # (skill_seq, bs, latent_dim)
            self.log_metric(latent_sample, "mean_latent_sample", "nonzero_mean") # METRIC
        else:
            mu = logvar = None
            latent_sample = torch.zeros([max_num_skills, bs, self.z_dim], dtype=torch.float32).to(qpos.device)

        return mu, logvar, latent_sample


    def skill_decode(self, z_sample, qpos, img_info, 
                     src_pad_mask=None, tgt_pad_mask=None, 
                     src_mask=None, tgt_mask=None):
        """Decode a sequence of skills into actions based on current image state and robot position
        Currently is not autoregressive and has fixed skill mapping size"""
        img_src, img_pe = img_info # (bs, h*num_cam*w, hidden)
        assert len(img_src.shape) == 3, "img_src should be size (bs, h*num_cam*w, hidden)"
        
        z = self.dec_z(z_sample) # (skill_seq, bs, hidden_dim)
        skill_seq, bs, _ = z.shape
        seq = tgt_pad_mask.shape[-1]

        # obtain learned postition embedding for z, state, & img inputs
        dec_type_embed = self.input_embed.weight * self.input_embed_scale_factor
        dec_type_embed = dec_type_embed.unsqueeze(1).repeat(1, bs, 1) # (4, bs, hidden_dim)

        # proprioception features
        state_src = self.dec_input_state_proj(qpos) # (bs, hidden)
        state_src = self.dec_input_state_norm(state_src).unsqueeze(0) # (1, bs, hidden)
        self.log_metric(state_src[0,0,:], "dec_state_vector", None) # TEMP METRIC
        self.log_metric(dec_type_embed[1, :, :], "mean_dec_state_type", "mean") # METRIC
        self.log_metric(state_src, "mean_dec_state_src", "nonzero_mean") # METRIC
        state_src = state_src + dec_type_embed[1, :, :] # add type 2 embedding
        state_pe = torch.zeros_like(state_src) # no pe

        # image, only use one image to decode rest of the skills
        img_src = img_src.permute(1, 0, 2) # (h*num_cam*w, bs, hidden)
        self.log_metric(img_src[:,0,:], "dec_img_vector_along_seq", "mean_along_seq") # TEMP METRIC
        img_pe = img_pe.permute(1, 0, 2) * self.img_pe_scale_factor # sinusoidal skill pe
        self.log_metric(img_pe, "dec_img_pe_vector", "mean_along_seq") # TEMP METRIC
        self.log_metric(dec_type_embed[2, :, :], "mean_dec_img_type", "mean") # METRIC
        self.log_metric(img_pe, "mean_dec_img_pe", "mean") # METRIC
        self.log_metric(img_src, "mean_dec_img_src", "nonzero_mean") # METRIC
        img_src = img_src + dec_type_embed[2, :, :] # add type 3 embedding

        # skills
        z_src = self.dec_input_z_norm(z) # (skill_seq, bs, hidden_dim)
        self.log_metric(z_src[:,0,:], "dec_z_vector_along_seq", "mean_along_seq") # TEMP METRIC
        z_pe = self.get_pos_table(skill_seq).permute(1, 0, 2).repeat(1, bs, 1) * self.dec_z_pos_scale_factor
        self.log_metric(z_pe, "enc_z_pe_vector", "mean_along_seq") # TEMP METRIC
        self.log_metric(dec_type_embed[3, :, :], "mean_dec_z_type", "mean") # METRIC
        self.log_metric(z_pe, "mean_dec_z_pe", "mean") # METRIC
        self.log_metric(z_src, "mean_dec_z_src", "nonzero_mean") # METRIC
        z_src = z_src + dec_type_embed[3, :, :].repeat(skill_seq, 1, 1) # add type 4 embedding (skill_seq, bs, hidden_dim)

        # Concatenate full decoder src
        dec_src = torch.cat([state_src, img_src, z_src], axis=0) # (state + img + z, bs, hidden)
        dec_src_pe = torch.cat([state_pe, img_pe, z_pe], axis=0) 
        # Add and norm
        dec_src = dec_src + dec_src_pe
        dec_src = self.dec_src_norm(dec_src)

        # position encoding for output sequence
        dec_tgt_pe = self.get_pos_table(seq).permute(1, 0, 2) * self.dec_tgt_pos_scale_factor # (seq, 1, hidden_dim)
        dec_tgt_pe = dec_tgt_pe.repeat(1, bs, 1)  # (seq, bs, hidden_dim)
        self.log_metric(img_pe, "dec_tgt_pe_vector", "mean_along_seq") # TEMP METRIC
        self.log_metric(dec_tgt_pe, "mean_dec_tgt_pe", "mean") # METRIC
        dec_tgt  = torch.zeros_like(dec_tgt_pe)
        # Add and norm
        dec_tgt = dec_tgt + dec_tgt_pe
        dec_tgt = self.dec_tgt_norm(dec_tgt)

        # src padding mask should pad unused skills but not other inputs TODO keep this calculation here?
        src_pad_mask = torch.cat([torch.zeros(bs, dec_src.shape[0]-src_pad_mask.shape[1]).to(self._device),
                                  src_pad_mask], dim=1)

        dec_output = self.decoder(src=dec_src,
                          src_key_padding_mask=src_pad_mask, 
                          src_is_causal=False,
                          src_mask=src_mask,
                          tgt=dec_tgt,
                          tgt_key_padding_mask=tgt_pad_mask,
                          tgt_is_causal=False, # Right now, generate entire action sequence at once
                          tgt_mask=tgt_mask,
                          memory_is_causal=False) # (seq, bs, hidden_dim)
        self.log_metric(dec_output[:,0,:], "dec_out_vector_along_seq", "mean_along_seq") # TEMP METRIC
        self.log_metric(dec_output, "mean_dec_output", "nonzero_mean") # METRIC
        
        a_hat = self.dec_action_proj(dec_output) # (bs, seq, act_dim)

        return a_hat
    

    def to(self, device):
        model = super().to(device)
        model._device = device
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
