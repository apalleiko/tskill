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
    def __init__(self, backbone, encoder, decoder, state_dim, action_dim, max_skill_len, z_dim, device, **kwargs):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
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

        # Get a sinusoidal position encoding table for a given sequence size
        self.get_pos_table = lambda x: get_sinusoid_encoding_table(x, self.hidden_dim).to(self._device) # (1, x, hidden_dim)

        ### Backbone
        self.image_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.backbone = backbone

        ### Encoder TODO add batch norms to projs?
        self.enc_action_proj = nn.Linear(action_dim, self.hidden_dim) # project action to hidden
        self.enc_state_proj = nn.Linear(state_dim, self.hidden_dim)  # project qpos to hidden
        # self.enc_image_proj = nn.LazyLinear(1) # project image vector from (seq, bs, c, m) to (seq, bs, c, 1) TODO Try with transformer or MLP instead?
        self.enc_image_proj = nn.Linear(32,1)
        # self.enc_image_proj = nn.Sequential([nn.Flatten(-2), nn.LazyLinear(self.hidden_dim)])

        # learned embeddings for the 3 encoder input types (action, state, image)
        self.enc_input_emb = nn.Embedding(3, self.hidden_dim) 

        ### Latent
        self.enc_z = nn.Linear(self.hidden_dim, self.z_dim*2) # project hidden state to latent [std; var]
        self.dec_z = nn.Linear(self.z_dim, self.hidden_dim) # project latent sample to hidden

        ### Decoder
        self.dec_input_state_proj = nn.Linear(state_dim, self.hidden_dim)
        self.dec_action_proj = nn.Linear(self.hidden_dim, action_dim) # TODO make this an mlp?

        # learned embeddings for the 3 decoder input types (proprio, skill, image)
        self.dec_inputs_pe = nn.Embedding(3, self.hidden_dim) 


    def image_encode(self, image):
        # Image observation features and position embeddings
        # Images are (bs, num_cam, channel, h, w)
        all_cam_features = []
        all_cam_pos = []
        for cam in range(image.shape[1]):
            features, pos = self.backbone(image[:, cam, ...])
            features = features[0] # take the last layer feature (bs, c, h, w)
            pos = pos[0]
            all_cam_features.append(self.image_proj(features))
            all_cam_pos.append(pos)

            # fold camera dimension into width dimension
            feat = torch.cat(all_cam_features, axis=3) # (bs, c, h, num_cam*w)
            pos = torch.cat(all_cam_pos, axis=3) 

        # flatten
        feat = feat.flatten(2) # (bs, c, h*num_cam*w)
        pos = pos.flatten(2) # (1, c, h*num_cam*w)
        
        return feat, pos
    

    def image_seq_encode(self, images):
        img_seq_features = []
        img_seq_pos = []

        # Images are (bs, seq, num_cam, channel, h, w)
        for t in range(images.shape[1]):
            t_feat, t_pos = self.image_encode(images[:, t, ...])
            img_seq_features.append(t_feat)
            img_seq_pos.append(t_pos)

        img_src = torch.stack(img_seq_features, axis=0) # (seq, bs, c, h*num_cam*w)
        img_pos = torch.stack(img_seq_pos, axis=0) # (seq, 1, c, h*num_cam*w)

        seq, bs, c, hw = img_src.shape
        img_src = img_src.permute(0, 1, 3, 2) # (seq, bs, h*num_cam*w, c)
        img_pe = img_pos.permute(0, 1, 3, 2).repeat(1, bs, 1, 1)
        
        return img_src, img_pe
        

    def forward_encode(self, qpos, actions, img_info, max_num_skills, src_pad_mask=None, tgt_pad_mask=None):
        """Encode skill sequence based on robot state, action, and image input sequence"""
        is_training = actions is not None # train or val
        bs, seq, _ = qpos.shape # Use bs for masking
        img_feat, img_pe = img_info # (seq, bs, h*num_cam*w, c)

        ### Obtain latent z from state,action,image sequence
        if is_training:
            type_embed = self.enc_input_emb.weight.unsqueeze(1).repeat(1, bs, 1) # (3, bs, hidden_dim)
            
            # project action and position sequences to hidden dim 
            action_src = self.enc_action_proj(actions).permute(1, 0, 2) # (seq, bs, hidden_dim)
            action_src = action_src + type_embed[0, :, :] # add type 1 embedding
            qpos_src = self.enc_state_proj(qpos).permute(1, 0, 2)  # (seq, bs, hidden_dim)
            qpos_src = qpos_src + type_embed[1, :, :] # add type 2 embedding

            # project img with local pe to correct size TODO add local pe? This generally doesn't feel right
            img_src = (img_feat + img_pe)
            img_src = img_src.permute(0, 1, 3, 2) # (seq, bs, c, h*num_cam*w)
            img_src = self.enc_image_proj(img_src).squeeze(-1) # (seq, bs, c=hidden)
            img_src = img_src + type_embed[2, :, :] # add type 3 embedding

            # Concatenate enc_src
            enc_src = torch.cat([qpos_src, action_src, img_src], axis=0) # (3*seq, bs, hidden_dim)

            # obtain seq position embedding for input
            enc_src_pe = self.get_pos_table(seq)
            enc_src_pe = enc_src_pe.permute(1, 0, 2)  # (seq, 1, hidden_dim)
            enc_src_pe = enc_src_pe.repeat(3, bs, 1)  # (3*seq, bs, hidden_dim)

            # query encoder model
            enc_tgt = torch.zeros(max_num_skills, bs, self.hidden_dim).to(self._device)
            enc_tgt_pe = self.get_pos_table(max_num_skills).permute(1, 0, 2).repeat(1, bs, 1)

            # TODO Keep causal mask? + enable combining with other masks
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(max_num_skills).to(self._device)

            # Repeat same padding mask for new seq length (same pattern)
            src_pad_mask = src_pad_mask.repeat(1,3) 

            enc_output = self.encoder(src=enc_src + enc_src_pe, 
                                          src_key_padding_mask=src_pad_mask, 
                                          src_is_causal=False,
                                          src_mask = None,
                                          tgt=enc_tgt + enc_tgt_pe,
                                          tgt_key_padding_mask=tgt_pad_mask,
                                          tgt_is_causal=True,
                                          tgt_mask = causal_mask,
                                          memory_is_causal=False) # (skill_seq, bs, hidden_dim)

            latent_info = self.enc_z(enc_output) # (skill_seq, bs, 2*latent_dim)
            mu = latent_info[:, :, :self.z_dim]
            logvar = latent_info[:, :, self.z_dim:]
            latent_sample = reparametrize(mu, logvar) # (skill_seq, bs, latent_dim)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([max_num_skills, bs, self.z_dim], dtype=torch.float32).to(qpos.device)

        return mu, logvar, latent_sample


    def skill_decode(self, z_sample, qpos, img_info, src_pad_mask=None, tgt_pad_mask=None):
        """Decode a sequence of skills into actions based on current image state and robot position
        Currently is not autoregressive and has fixed skill mapping size"""
        img_src, img_pe = img_info
        assert len(img_src.shape) == 3, "img_src should be size (h*num_cam*w, bs, hidden)"
        
        z = self.dec_z(z_sample) # (skill_seq, bs, hidden_dim)
        skill_seq, bs, _ = z.shape
        seq = skill_seq*self.max_skill_len

        # obtain learned postition embedding for z, state, & img inputs
        dec_type_embed = self.dec_inputs_pe.weight.unsqueeze(1).repeat(1, bs, 1) # (3, bs, hidden)

        # proprioception features
        state_src = self.dec_input_state_proj(qpos).unsqueeze(0) # (1, bs, hidden)
        state_src = state_src + dec_type_embed[0, :, :] # add type 1 embedding
        state_pe = torch.zeros_like(state_src) # no pe

        # image, only use one image to decode rest of the skills
        img_src = img_src.permute(1, 0, 2) # (h*num_cam*w, bs, hidden) 
        img_src = img_src + dec_type_embed[1, :, :] # add type 2 embedding
        img_pe = img_pe.permute(1, 0, 2) # sinusoidal skill pe

        # skills
        z_src = z + dec_type_embed[2, :, :].repeat(skill_seq, 1, 1) # add type 3 embedding (skill_seq, bs, hidden_dim)
        z_pe = self.get_pos_table(skill_seq).permute(1, 0, 2).repeat(1, bs, 1)

        dec_src = torch.cat([state_src, img_src, z_src], axis=0) # (state + img + z, bs, hidden)
        dec_src_pe = torch.cat([state_pe, img_pe, z_pe], axis=0) 

        # position encoding for output sequence
        dec_tgt_pe = self.get_pos_table(seq).permute(1, 0, 2)  # (seq, 1, hidden_dim)
        dec_tgt_pe = dec_tgt_pe.repeat(1, bs, 1)  # (seq, bs, hidden_dim)
        dec_tgt  = torch.zeros_like(dec_tgt_pe)

        # src padding mask should pad unused skills but not other inputs TODO keep this calculation here?
        src_pad_mask = torch.cat([torch.zeros(bs, dec_src.shape[0]-src_pad_mask.shape[1]).to(self._device),
                                  src_pad_mask], dim=1)

        dec_output = self.decoder(src=dec_src + dec_src_pe, 
                          src_key_padding_mask=src_pad_mask, 
                          src_is_causal=False,
                          src_mask=None,
                          tgt=dec_tgt + dec_tgt_pe,
                          tgt_key_padding_mask=tgt_pad_mask,
                          tgt_is_causal=False, # Right now, generate entire action sequence at once
                          tgt_mask=None, #TODO
                          memory_is_causal=False) # (seq, bs, hidden_dim)

        a_hat = self.dec_action_proj(dec_output)

        return a_hat


    def forward(self, data, **kwargs):
        """
        data:
            qpos: batch, seq, qpos_dim
            images: batch, seq, num_cam, channel, height, width
            actions: batch, seq, action_dim
            seq_pad_mask: Padding mask for input sequence (bs, seq)
        """
        qpos = data["state"].to(self._device)
        images = data["rgb"].to(self._device)
        actions = data["actions"].to(self._device)
        seq_pad_mask = data["seq_mask"].to(self._device)

        bs, seq, _ = qpos.shape

        run_backbone = kwargs.get("run_backbone",True)
        if run_backbone:
            img_src, img_pe = self.image_seq_encode(images)
        else:
            raise NotImplementedError # TODO, precalculate all image features?

        max_num_skills = int(seq / self.max_skill_len) # Sequence input needs to be divisible by skill_len for now
        
        # Infer skill mask from input sequence mask
        num_unpad_seq = torch.sum(torch.logical_not(seq_pad_mask.to(torch.int16)), dim=1)
        num_unpad_skills = torch.ceil(num_unpad_seq / self.max_skill_len) # number of skills with relevant outputs TODO handle non divisible skill lengths
        skill_pad_mask = torch.zeros(bs, max_num_skills) # True is unattended
        for b in range(bs):
            skill_pad_mask[b,num_unpad_skills[b].to(torch.int16):] = 1
        skill_pad_mask = skill_pad_mask.to(torch.bool).to(self._device)

        mu, logvar, z = self.forward_encode(qpos, actions, (img_src, img_pe), max_num_skills, seq_pad_mask, skill_pad_mask)
        
        # Decode skills conditioned with state & image from sequence start
        run_subseq = kwargs.get("run_subseq", False)
        if run_subseq: # TODO Decode skills for subsequences of the predictions for training
            pass
        else:
            a_hat = self.skill_decode(z, 
                                      qpos[:,0,:], 
                                      (img_src[0,...], img_pe[0,...]),
                                      skill_pad_mask,
                                      seq_pad_mask)

        return a_hat, [mu, logvar]
    

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model
