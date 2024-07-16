# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

import IPython
e = IPython.embed
import dill as pickle


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
    def __init__(self, stt_encoder, 
                 encoder, decoder, 
                 state_dim, action_dim, 
                 max_skill_len, z_dim,
                 single_skill, conditional_decode,
                 device, **kwargs):
        """ Initializes the model.
        Parameters:
            stt_encoder: torch module of the state encoder to be used. See perception
            encoder: transformer encoder module
            decoder: transformer decoder module
            state_dim: robot state dimension. Size of input robot position (delta_pos)
            action_dim: robot action dimenstion.
            max_skill_len: Max number of actions that can be predicted from a skill
            z_dim: dimension of latent skill vectors
            single_skill: bool whether to decode only 1 skill at a time
            conditional_decode: bool whether to decode with image/qpos conditional info
            device: device to operate on
        kwargs:
            autoregressive: Whether skill generation and action decoding is autoregressive TODO
        """
        super().__init__()
        ### General args
        self.decoder = decoder
        self.encoder = encoder
        self.hidden_dim = decoder.d_model
        self.z_dim = z_dim
        self.max_skill_len = max_skill_len
        self._device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.single_skill = single_skill
        self.look_ahead = 1 # TODO
        self.norm = nn.LayerNorm
        self.conditional_decode = conditional_decode
        self.autoregressive = kwargs.get("autoregressive",False)

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
        self.by_cam = self.stt_encoder.by_cam
        if self.by_cam:
            num_img_feats = 16 # HARDCODED
        else:
            num_img_feats = 32
        self.image_feat_norm = self.norm([num_img_feats, self.hidden_dim]) # TODO???

        ### Encoder
        self.enc_action_proj = nn.Linear(action_dim, self.hidden_dim) # project action -> hidden
        self.enc_action_norm = self.norm(self.hidden_dim)
        self.enc_state_proj = nn.Linear(state_dim, self.hidden_dim)  # project qpos -> hidden
        self.enc_state_norm = self.norm(self.hidden_dim)
        self.enc_image_proj = nn.Linear(num_img_feats,1) #TODO Hardcoded
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

        # Action heads
        nl = self.hidden_dim
        action_head = []  # TODO make this an mlp?
        for n in (128, 64, 32):
            action_head.append(nn.Linear(nl, n)) 
            action_head.append(self.norm(n))
            action_head.append(nn.GELU())
            nl = n
        self.dec_action_head = nn.Sequential(*action_head)
        self.dec_action_joint_proj = nn.Linear(nl, action_dim - 1) # Decode joint actions
        # Decode gripper actions with a tanh to restrict between -1 and 1
        self.dec_action_gripper_proj = nn.Sequential(nn.Linear(nl, 1), nn.Tanh()) 

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
            
            enc_mask: bs, (2+num_cam)*seq, "
            dec_mask: bs, 2+16*num_cam, "
        """
        qpos = data["state"].to(self._device)
        images = data["rgb"].to(self._device)
        actions = data["actions"].to(self._device) if data["actions"] is not None else None
        seq_pad_mask = data["seq_pad_mask"].to(self._device, torch.bool)
        skill_pad_mask = data["skill_pad_mask"].to(self._device, torch.bool)
        enc_mask = data.get("enc_mask", None)
        dec_mask = data.get("dec_mask", None)
        if enc_mask is not None:
            enc_mask = enc_mask.to(self._device, torch.bool)
            # Reshape mask from (N,S,S) to (N*nhead,S,S) as expected by transformer
            enc_mask = torch.cat([enc_mask[i:i+1,:,:].repeat(self.encoder.nhead,1,1) 
                                  for i in range(enc_mask.shape[0])])
        if dec_mask is not None:
            dec_mask = dec_mask.to(self._device, torch.bool)
            # Reshape mask from (N,S,S) to (N*nhead,S,S) as expected by transformer
            dec_mask = torch.cat([dec_mask[i:i+1,:,:].repeat(self.decoder.nhead,1,1) 
                                  for i in range(dec_mask.shape[0])])

        # Calculate image features
        img_src, img_pe = self.stt_encoder(images) # (seq, bs, h*num_cam*w, c) | (seq, bs, num_cam, h*w, c) 

        mu, logvar, z = self.forward_encode(qpos, actions, (img_src, img_pe), 
                                            seq_pad_mask, skill_pad_mask, 
                                            enc_mask, None) # (skill_seq, bs, latent_dim)
        
        # Decode skills conditioned with conditional state & image info from current state
        if not self.single_skill:
            a_hat = self.skill_decode(z, qpos[:,0,:], (img_src[0,...], img_pe[0,...]),
                                        skill_pad_mask, seq_pad_mask, 
                                        dec_mask, None)
        else:
            bs, _ = seq_pad_mask.shape
            a_hat = torch.zeros(0, bs, self.action_dim, device=self._device) # (0, bs, act_dim)
            for sk in range(z.shape[0]):
                # Get current time step and skill
                t = sk*self.max_skill_len
                tf = self.max_skill_len*(sk+self.look_ahead)
                sk_z = z[sk:sk + self.look_ahead, :, :] # (skill_seq <= decode_num, bs, latent_dim)
                sk_skill_pad_mask = skill_pad_mask[:, sk:sk+self.look_ahead]
                sk_seq_pad_mask = seq_pad_mask[:, t:tf]
                # Decode current skill with conditional info
                a_pred = self.skill_decode(sk_z, qpos[:,t,:], (img_src[t,...], img_pe[t,...]),
                                           sk_skill_pad_mask, sk_seq_pad_mask, 
                                           dec_mask, None) # (seq, bs, act_dim)
                a_hat = torch.vstack((a_hat, a_pred))
                if torch.any(torch.isnan(a_hat)):
                    with open("ERROR_DATA.pickle",'+wb') as f:
                        pickle.dump(data, f)
                    raise ValueError(f"NaNs encountered during decoding! {a_hat.shape}")

        a_hat = a_hat.permute(1,0,2) # (bs, seq, act_dim)
        if mu is not None:
            mu = mu.permute(1,0,2) # (bs, seq, latent_dim)
            logvar = logvar.permute(1,0,2)
        return dict(a_hat=a_hat, mu=mu, logvar=logvar, latent=z)
        

    def forward_encode(self, qpos, actions, img_info, 
                       src_pad_mask, tgt_pad_mask, 
                       src_mask=None, tgt_mask=None):
        """Encode skill sequence based on robot state, action, and image input sequence"""
        is_training = actions is not None # train or val
        bs, seq, _ = qpos.shape # Use bs for masking
        img_feat, img_pe = img_info # (seq, bs, h*num_cam*w, c) | (seq, bs, num_cam, h*w, c)
        img_feat = self.image_feat_norm(img_feat)
        max_num_skills = tgt_pad_mask.shape[1]
        if self.by_cam:
            num_cam = img_feat.shape[2]
        else:
            num_cam = 1

        # Get a batch mask for eliminating fully padded inputs during training
        batch_mask = torch.all(src_pad_mask, dim=1) 
        if torch.any(batch_mask) > 0:
            print("Encountered fully padded input!")

        ### Obtain latent z from state, action, image sequence
        if is_training:
            enc_type_embed = self.input_embed.weight * self.input_embed_scale_factor
            enc_type_embed = enc_type_embed.unsqueeze(1).repeat(1, bs, 1) # (4, bs, hidden_dim)

            # project action sequences to hidden dim (bs, seq, action_dim)
            action_src = self.enc_action_proj(actions) # (bs, seq, hidden_dim)
            action_src = self.enc_action_norm(action_src).permute(1, 0, 2) # (seq, bs, hidden_dim)
            action_src = action_src + enc_type_embed[0, :, :] # add type 1 embedding

            # project position sequences to hidden dim (bs, seq, state_dim)
            qpos_src = self.enc_state_proj(qpos) # (bs, seq, hidden_dim)
            qpos_src = self.enc_state_norm(qpos_src).permute(1, 0, 2) # (seq, bs, hidden_dim)
            qpos_src = qpos_src + enc_type_embed[1, :, :] # add type 2 embedding

            # project img with local pe to correct size
            img_pe = img_pe * self.img_pe_scale_factor
            img_src = (img_feat + img_pe) # (seq, bs, h*num_cam*w, c) | (seq, bs, num_cam, h*w, c)
            img_src = torch.transpose(img_src, -1, -2) # (seq, bs, c, h*num_cam*w) | (seq, bs, num_cam, c, h*w)
            img_src = self.enc_image_proj(img_src).squeeze(-1) # (seq, bs, c=hidden) | (seq, bs, num_cam, c=hidden)
            if self.by_cam:
                img_src = img_src.permute(0, 2, 1, 3) # (seq, num_cam, bs, hidden)
                img_src = torch.vstack([img_src[:,m,...] for m in range(num_cam)]) # (seq*num_cam, bs, hidden)
            img_src = img_src + enc_type_embed[2, :, :] # add type 3 embedding

            # encoder src
            enc_src = torch.cat([qpos_src, action_src, img_src], axis=0) # (3*seq, bs, hidden_dim) | ((2+num_cam)*seq, bs, hidden_dim)
            # obtain seq position embedding for input
            enc_src_pe = self.get_pos_table(seq) * self.enc_src_pos_scale_factor
            enc_src_pe = enc_src_pe.permute(1, 0, 2)  # (seq, 1, hidden_dim)
            enc_src_pe = enc_src_pe.repeat((2+num_cam), bs, 1)  # ((2+num_cam)*seq, bs, hidden_dim) | ((2+num_cam)*seq, bs, hidden_dim)
            # Add and norm
            enc_src = enc_src + enc_src_pe
            enc_src = self.enc_src_norm(enc_src)

            # encoder tgt
            enc_tgt = torch.zeros(max_num_skills, bs, self.hidden_dim).to(self._device) # (skill_seq, bs, hidden_dim)
            enc_tgt_pe = self.get_pos_table(max_num_skills).permute(1, 0, 2).repeat(1, bs, 1) * self.enc_tgt_pos_scale_factor
            # Add and norm
            enc_tgt = enc_tgt + enc_tgt_pe
            enc_tgt = self.enc_tgt_norm(enc_tgt)

            # Repeat same padding mask for new seq length (same pattern)
            src_pad_mask = src_pad_mask.repeat(1,(2+num_cam)) 
            
            # reverse batch mask for transformer calls to fully padded inputs to avoid NaNs
            src_pad_mask[batch_mask, :] = False
            tgt_pad_mask[batch_mask, :] = False

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

            latent_info = self.enc_z(enc_output) # (skill_seq, bs, 2*latent_dim)
            mu = latent_info[:, :, :self.z_dim]
            logvar = latent_info[:, :, self.z_dim:]
            latent_sample = reparametrize(mu, logvar) # (skill_seq, bs, latent_dim)

            # replace with filler for fully padded batches
            mu[:, batch_mask, :] = 0
            logvar[:, batch_mask, :] = 0
            latent_sample[:, batch_mask, :] = 0
        else:
            mu = logvar = None
            latent_sample = torch.zeros([max_num_skills, bs, self.z_dim], dtype=torch.float32).to(qpos.device)

        return mu, logvar, latent_sample


    def skill_decode(self, z_sample, qpos, img_info, 
                     src_pad_mask=None, tgt_pad_mask=None, 
                     src_mask=None, tgt_mask=None):
        """Decode a sequence of skills into actions based on current image state and robot position
        Currently is not autoregressive and has fixed skill mapping size"""
        img_src, img_pe = img_info # (bs, h*num_cam*w, hidden) | (bs, num_cam, h*w, hidden)
        img_src = self.image_feat_norm(img_src)
        if len(img_src.shape) == 4:
            img_src = img_src.flatten(1,2)
            img_pe = img_pe.flatten(1,2)
        
        z = self.dec_z(z_sample) # (skill_seq, bs, hidden_dim)
        skill_seq, bs, _ = z.shape
        if not self.single_skill:
            seq = tgt_mask_seq = tgt_pad_mask.shape[1]
        else:
            seq = self.max_skill_len
            tgt_mask_seq = tgt_pad_mask.shape[1]

        # Handle case where incomplete tgt (seq) mask is passed in during evaluation of unpadded inputs
        if tgt_mask_seq < seq: # pad with True (unattended)
            tgt_pad_mask = torch.hstack((tgt_pad_mask, torch.ones(bs, seq - tgt_mask_seq, device=self._device).to(torch.bool)))

        # Get a batch mask for eliminating fully padded inputs during training
        batch_mask = torch.all(src_pad_mask, dim=1)
        if torch.all(batch_mask):
            return torch.zeros(seq, bs, self.action_dim, device=self._device)

        # obtain learned postition embedding for z, state, & img inputs
        dec_type_embed = self.input_embed.weight * self.input_embed_scale_factor
        dec_type_embed = dec_type_embed.unsqueeze(1).repeat(1, bs, 1) # (4, bs, hidden_dim)

        # proprioception features
        state_src = self.dec_input_state_proj(qpos) # (bs, hidden)
        state_src = self.dec_input_state_norm(state_src).unsqueeze(0) # (1, bs, hidden)
        state_src = state_src + dec_type_embed[1, :, :] # add type 2 embedding
        state_pe = torch.zeros_like(state_src) # no pe

        # image, only use one image to decode rest of the skills
        img_src = img_src.permute(1, 0, 2) # (h*num_cam*w, bs, hidden)
        img_pe = img_pe.permute(1, 0, 2) * self.img_pe_scale_factor # sinusoidal skill pe
        img_src = img_src + dec_type_embed[2, :, :] # add type 3 embedding

        # skills
        z_src = self.dec_input_z_norm(z) # (skill_seq, bs, hidden_dim)
        z_pe = self.get_pos_table(skill_seq).permute(1, 0, 2).repeat(1, bs, 1) * self.dec_z_pos_scale_factor
        z_src = z_src + dec_type_embed[3, :, :].repeat(skill_seq, 1, 1) # add type 4 embedding (skill_seq, bs, hidden_dim)

        # Concatenate full decoder src
        if self.conditional_decode:
            dec_src = torch.cat([state_src, img_src, z_src], axis=0) # (state + img + z, bs, hidden)
            dec_src_pe = torch.cat([state_pe, img_pe, z_pe], axis=0) 
        else:
            dec_src = z_src # (z, bs, hidden)
            dec_src_pe = z_pe
        # Add and norm
        dec_src = dec_src + dec_src_pe
        dec_src = self.dec_src_norm(dec_src)

        # position encoding for output sequence
        dec_tgt_pe = self.get_pos_table(seq).permute(1, 0, 2) * self.dec_tgt_pos_scale_factor # (seq, 1, hidden_dim)
        dec_tgt_pe = dec_tgt_pe.repeat(1, bs, 1)  # (seq, bs, hidden_dim)
        dec_tgt  = torch.zeros_like(dec_tgt_pe)
        # Add and norm
        dec_tgt = dec_tgt + dec_tgt_pe
        dec_tgt = self.dec_tgt_norm(dec_tgt)

        # src padding mask should pad unused skills but not other inputs
        src_pad_mask = torch.cat([torch.zeros(bs, dec_src.shape[0]-src_pad_mask.shape[1]).to(self._device),
                                  src_pad_mask], dim=1)
        
        # reverse batch mask for transformer calls to fully padded inputs to avoid NaNs
        src_pad_mask[batch_mask, :] = False
        tgt_pad_mask[batch_mask, :] = False

        dec_output = self.decoder(src=dec_src,
                          src_key_padding_mask=src_pad_mask, 
                          src_is_causal=False,
                          src_mask=src_mask,
                          tgt=dec_tgt,
                          tgt_key_padding_mask=tgt_pad_mask,
                          tgt_is_causal=False, # Generate entire action sequence at once for now
                          tgt_mask=tgt_mask,
                          memory_is_causal=False) # (seq, bs-fp, hidden_dim)
        
        # Send decoder output through MLP and project to action dimensions
        a_head = self.dec_action_head(dec_output) # (seq, bs-fp, 32)
        a_joint = self.dec_action_joint_proj(a_head) # (seq, bs-fp, action_dim - 1)
        a_grip = self.dec_action_gripper_proj(a_head) # (seq, bs-fp, 1)
        a_hat = torch.cat((a_joint, a_grip), -1) # (seq, bs-fp, action_dim)
        
        # return zeros for fully padded inputs (seq, bs, action_dim)
        a_hat[:, batch_mask, :] = 0

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
