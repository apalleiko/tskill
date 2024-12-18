import time
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

import IPython
e = IPython.embed
import dill as pickle
from policy.dataset.masking_utils import get_dec_ar_masks, get_enc_causal_masks, get_skill_pad_from_seq_pad


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
                 conditional_decode,
                 autoregressive_decode,
                 encode_state,
                 encoder_is_causal,
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
            conditional_decode: bool whether to decode with image/qpos conditional info
            device: device to operate on
            autoregressive: Whether action decoding is autoregressive
        kwargs:
            encode_state: whether to encode image and qpos to skills
            encoder_is_causal: whether causal masks are applied to skill generation
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
        self.norm = nn.LayerNorm
        self.conditional_decode = conditional_decode
        self.autoregressive_decode = autoregressive_decode
        self.encode_state = encode_state
        self.encoder_is_causal = encoder_is_causal

        ### Get a sinusoidal position encoding table for a given sequence size
        self.get_pos_table = lambda x: get_sinusoid_encoding_table(x, self.hidden_dim).to(self._device) # (1, x, hidden_dim)
        # Create pe scaling factors
        self.enc_src_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.001)
        self.enc_tgt_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.dec_src_qpos_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.dec_src_img_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.dec_tgt_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.img_pe_scale_factor = nn.Parameter(torch.ones(1) * 0.001)

        # learned embeddings for the 4 input types (action, state, image, skill)
        self.input_embed = nn.Embedding(4, self.hidden_dim) 
        self.input_embed_scale_factor = nn.Parameter(torch.ones(4, 1) * 0.01)

        ### Backbone
        self.stt_encoder = stt_encoder
        self.num_img_feats = 16 # Hardcoded
        self.image_proj = nn.Linear(self.stt_encoder.num_channels, self.hidden_dim)
        self.image_feat_norm = self.norm([self.num_img_feats, self.hidden_dim])

        ### Encoder
        self.enc_action_proj = nn.Linear(action_dim, self.hidden_dim) # project action -> hidden
        self.enc_action_norm = self.norm(self.hidden_dim)
        self.enc_state_proj = nn.Linear(state_dim, self.hidden_dim)  # project qpos -> hidden
        self.enc_state_norm = self.norm(self.hidden_dim)
        self.enc_image_proj = nn.Linear(self.num_img_feats, 1)
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
        self.dec_tgt_start_token = nn.Parameter(torch.zeros(1,1,self.action_dim))
        # self.dec_tgt_start_token = torch.zeros(1,1,self.action_dim)

        # Action heads
        self.dec_action_joint_proj = nn.Linear(self.hidden_dim, action_dim - 1) # Decode joint actions
        # Decode gripper actions with a tanh to restrict between -1 and 1
        self.dec_action_gripper_proj = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Tanh())

        # Other
        self.metrics = dict()
        self.to(self._device)
        self.apply(self.init_weights)
    

    def forward(self, data, **kwargs):
        """
        data:
            qpos: bs, seq, qpos_dim
            images: bs, seq, num_cam, 3, h, w
            /img_feat: bs, seq, num_cam, c, h, w
            /img_pe: bs, seq, num_cam, hidden, h, w
            actions: bs, seq, action_dim
            seq_pad_mask: Padding mask for input sequence (bs, seq)
            skill_pad_mask: Padding mask for skill sequence (bs, max_num_skills)
            enc_mask: Encoder input mask:L (bs, (2+num_cam)*seq)
            dec_mask: (bs, MSL*(2+num_features*num_cam), ")
            mem_mask: (bs, MSL, MSL*(2+num_features*num_cam))
            tgt_mask: (bs, MSL, MSL)
        """
        qpos = data["state"].to(self._device)
        actions = data["actions"].to(self._device)
        seq_pad_mask = data["seq_pad_mask"].to(self._device, torch.bool)
        skill_pad_mask = data["skill_pad_mask"].to(self._device, torch.bool)

        BS, SEQ = seq_pad_mask.shape
        _, MNS = skill_pad_mask.shape

        ### Calculate image features or use precalculated from dataset
        use_precalc = kwargs.get("use_precalc",False)
        if use_precalc and all(x in data.keys() for x in ("img_feat","img_pe")):
            img_src = data["img_feat"].transpose(0,1).to(self._device) # (seq, bs, num_cam, h*w, c)
            img_pe = data["img_pe"].transpose(0,1).to(self._device)
        else:
            images = data["rgb"].to(self._device)
            img_src, img_pe = self.stt_encoder(images) # (seq, bs, num_cam, h*w, c)
        
        num_cam = img_src.shape[2]

        ### Get causal/other masks, if applicable
        # Encoder causal masks
        if self.encoder_is_causal:
            enc_src_mask, enc_mem_mask, enc_tgt_mask = get_enc_causal_masks(SEQ, MNS, self.max_skill_len, device=self._device)
        else:
            enc_src_mask, enc_mem_mask = enc_tgt_mask = None
        # Decoder ar masks
        if self.autoregressive_decode:
            dec_src_mask, dec_mem_mask, dec_tgt_mask = get_dec_ar_masks(self.num_img_feats*num_cam, self.max_skill_len, device=self._device)
        else:
            dec_tgt_mask = dec_src_mask = dec_mem_mask = None

        ### Encode inputs to latent
        mu, logvar, z = self.forward_encode(qpos, actions, (img_src, img_pe), 
                                            seq_pad_mask, skill_pad_mask, 
                                            enc_src_mask, enc_mem_mask, enc_tgt_mask) # (skill_seq, bs, latent_dim)
        
        # Check for NaNs
        if torch.any(torch.isnan(z)):
            with open("ERROR_DATA.pickle",'+wb') as f:
                pickle.dump(data, f)
            raise ValueError(f"NaNs encountered during encoding! Data saved to ERROR_DATA.pickle")
        
        ### Decode skill sequence
        a_hat = self.sequence_decode(z, qpos, actions, (img_src, img_pe),
                                     seq_pad_mask, skill_pad_mask,
                                     dec_src_mask, dec_mem_mask, dec_tgt_mask)

        # Check for NaNs
        if torch.any(torch.isnan(a_hat)):
            with open("ERROR_DATA.pickle",'+wb') as f:
                pickle.dump(data, f)
            raise ValueError(f"NaNs encountered during decoding! Data saved to ERROR_DATA.pickle")

        # Reorder outputs to match inputs
        a_hat = a_hat.permute(1,0,2) # (bs, seq, act_dim)
        mu = mu.permute(1,0,2) # (bs, seq, latent_dim)
        logvar = logvar.permute(1,0,2) # (bs, seq, latent_dim)

        return dict(a_hat=a_hat, mu=mu, logvar=logvar, latent=z)
        

    def forward_encode(self, qpos, actions, img_info, 
                       src_pad_mask, tgt_pad_mask, 
                       src_mask=None, mem_mask=None, tgt_mask=None):
        """Encode skill sequence based on robot state, action, and image input sequence"""
        bs, seq, _ = qpos.shape # Use bs for masking
        img_feat, img_pe = img_info # (seq, bs, num_cam, h*w, hidden)
        max_num_skills = tgt_pad_mask.shape[1]
        num_cam = img_feat.shape[2]

        # Get a batch mask for eliminating fully padded inputs during training
        batch_mask = torch.all(src_pad_mask, dim=1) 
        if torch.any(batch_mask) > 0:
            print("Encountered fully padded encoder input!")

        ### Obtain latent z from state, action, image sequence
        enc_type_embed = self.input_embed.weight * self.input_embed_scale_factor
        enc_type_embed = enc_type_embed.unsqueeze(1).repeat(1, bs, 1) # (4, bs, hidden_dim)

        # project action sequences to hidden dim (bs, seq, action_dim)
        action_src = self.enc_action_proj(actions) # (bs, seq, hidden_dim)
        action_src = self.enc_action_norm(action_src).permute(1, 0, 2) # (seq, bs, hidden_dim)
        action_src = action_src + enc_type_embed[0, :, :] # add type 1 embedding

        if self.encode_state:
            # project position sequences to hidden dim (bs, seq, state_dim)
            qpos_src = self.enc_state_proj(qpos) # (bs, seq, hidden_dim)
            qpos_src = self.enc_state_norm(qpos_src).permute(1, 0, 2) # (seq, bs, hidden_dim)
            qpos_src = qpos_src + enc_type_embed[1, :, :] # add type 2 embedding

            # project img with local pe to correct size
            img_src = self.image_proj(img_feat) # (seq, bs, num_cam, h*w, hidden)
            img_src = self.image_feat_norm(img_src)
            img_pe = img_pe * self.img_pe_scale_factor
            img_src = (img_src + img_pe) # (seq, bs, num_cam, h*w, hidden)
            img_src = torch.transpose(img_src, -1, -2) # (seq, bs, num_cam, hidden, h*w)
            img_src = self.enc_image_proj(img_src).squeeze(-1) # (seq, bs, num_cam, hidden)
            img_src = img_src.permute(0, 2, 1, 3) # (seq, num_cam, bs, hidden)
            img_src = torch.vstack([img_src[:,m,...] for m in range(num_cam)]) # (seq*num_cam, bs, hidden)
            img_src = img_src + enc_type_embed[2, :, :] # add type 3 embedding

            # encoder src
            enc_src = torch.cat([qpos_src, action_src, img_src], axis=0) # ((2+num_cam)*seq, bs, hidden_dim)
            # obtain seq position embedding for input
            enc_src_pe = self.get_pos_table(seq) * self.enc_src_pos_scale_factor
            enc_src_pe = enc_src_pe.permute(1, 0, 2)  # (seq, 1, hidden_dim)
            enc_src_pe = enc_src_pe.repeat((2+num_cam), bs, 1)  # ((2+num_cam)*seq, bs, hidden_dim)
            enc_src = enc_src + enc_src_pe
            enc_src = self.enc_src_norm(enc_src)

            # Repeat same padding mask for new seq length (same pattern)
            src_pad_mask = src_pad_mask.repeat(1,(2+num_cam)) # (bs, (2+num_cam)*seq)
            if src_mask is not None:
                src_mask = src_mask.repeat((2+num_cam), (2+num_cam)) # (bs*nhead, (2+num_cam)*seq, (2+num_cam)*seq)
            if mem_mask is not None:
                mem_mask = mem_mask.repeat(1, (2+num_cam)) # (skill_seq, (2+num_cam)*seq)
        else:
            enc_src = action_src # (seq, bs, hidden_dim)
            # obtain seq position embedding for input
            enc_src_pe = self.get_pos_table(seq) * self.enc_src_pos_scale_factor
            enc_src_pe = enc_src_pe.permute(1, 0, 2)  # (seq, 1, hidden_dim)
            enc_src = enc_src + enc_src_pe
            enc_src = self.enc_src_norm(enc_src)

        # encoder tgt
        enc_tgt = torch.zeros(max_num_skills, bs, self.hidden_dim).to(self._device) # (skill_seq, bs, hidden_dim)
        enc_tgt_pe = self.get_pos_table(max_num_skills).permute(1, 0, 2).repeat(1, bs, 1) * self.enc_tgt_pos_scale_factor
        enc_tgt = enc_tgt + enc_tgt_pe
        enc_tgt = self.enc_tgt_norm(enc_tgt)

        if self.encoder_is_causal:
            src_pad_mask = tgt_pad_mask = None # Padded outputs are ignored in loss downstream

        # # query encoder model
        # enc_output = self.encoder(src=enc_src,
        #                           src_key_padding_mask=src_pad_mask, 
        #                           src_is_causal=self.encoder_is_causal,
        #                           src_mask=src_mask,
        #                           memory_key_padding_mask=src_pad_mask,
        #                           memory_mask=mem_mask,
        #                           memory_is_causal=self.encoder_is_causal, 
        #                           tgt=enc_tgt,
        #                           tgt_key_padding_mask=tgt_pad_mask,
        #                           tgt_is_causal=self.encoder_is_causal,
        #                           tgt_mask=tgt_mask) # (skill_seq, bs, hidden_dim)
        
        enc_output = self.encoder(enc_tgt, enc_src, tgt_mask=tgt_mask, memory_mask=mem_mask,
                                    tgt_key_padding_mask=tgt_pad_mask,
                                    memory_key_padding_mask=src_pad_mask,
                                    tgt_is_causal=self.encoder_is_causal, memory_is_causal=self.encoder_is_causal) # (skill_seq, bs, hidden_dim)

        latent_info = self.enc_z(enc_output) # (skill_seq, bs, 2*latent_dim)
        mu = latent_info[:, :, :self.z_dim] # (skill_seq, bs, latent_dim)
        logvar = latent_info[:, :, self.z_dim:] # (skill_seq, bs, latent_dim)
        latent_sample = reparametrize(mu, logvar) # (skill_seq, bs, latent_dim)
        
        return mu, logvar, latent_sample


    def skill_decode(self, z, qpos, img_info,
                     src_pad_mask, tgt_pad_mask,
                     src_mask=None, mem_mask=None, tgt_mask=None,
                     tgt=None):
        """
        Decode an individual skill into actions, possibly based on current 
        image and state depending on self.conditional_decode
        args: 
            z: (1, bs, latent_dim)
            qpos: (MSL|1, bs, state_dim)
            img_info: (img_src, img_pe): (MSL|1, bs, num_cam, h*w, c & hidden)
            src_pad_mask: (bs, 1)
            tgt_pad_mask: (bs, MSL|<)
            src_mask: (bs, MSL|<*(2+num_features*num_cam), ")
            mem_mask: (bs, MSL|<, MSL|<*(2+num_features*num_cam))
            tgt_mask: (bs, MSL|<, MSL|<)
            tgt: (bs, MSL|<, act_dim)
        returns:
            a_hat: (seq, bs, act_dim)
        """
        ### Move inputs to device (needed for evaluation)
        if tgt is not None:
            tgt = tgt.to(self._device)
        if src_mask is not None:
            src_mask = src_mask.to(self._device)
        if mem_mask is not None:
            mem_mask = mem_mask.to(self._device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(self._device)

        if not self.autoregressive_decode:
            src_pad_mask = src_pad_mask.to(self._device)
            tgt_pad_mask = tgt_pad_mask.to(self._device)
            # Get a batch mask for handling fully padded inputs during training
            batch_mask = torch.all(src_pad_mask, dim=1)
            if torch.all(batch_mask): # Unmask if encounter fully padded inputs, which could happen during training.
                return torch.zeros(self.max_skill_len, bs, self.action_dim, device=self._device)
        else:
            src_pad_mask = None # padded items ignored downstream
            tgt_pad_mask = None # padded items ignored downstream

        bs = z.shape[1]

        # obtain learned postition embedding for z, state, & img inputs
        dec_type_embed = self.input_embed.weight * self.input_embed_scale_factor
        dec_type_embed = dec_type_embed.unsqueeze(1).repeat(1, bs, 1) # (4, bs, hidden_dim)

        # skills
        z_src = self.dec_z(z) # (1, bs, hidden_dim)
        z_src = self.dec_input_z_norm(z_src) # (1, bs, hidden_dim)
        z_src = z_src + dec_type_embed[3, :, :] # add type 4 embedding
        z_pe = torch.zeros_like(z_src) # no pe, only one skill per decoding step

        # tgt
        if self.autoregressive_decode:
            dec_tgt = self.enc_action_proj(tgt).permute(1,0,2) # (MSL|<, bs, hidden_dim)
            # dec_tgt = torch.zeros_like(dec_tgt, device=self._device) # (MSL|<, bs, hidden_dim)
            dec_tgt_pe = self.get_pos_table(dec_tgt.shape[0]).permute(1, 0, 2).repeat(1, bs, 1) * self.dec_tgt_pos_scale_factor # (MSL|<, bs, hidden_dim)
            # dec_tgt_pe = torch.zeros_like(dec_tgt)
        else:
            dec_tgt_pe = self.get_pos_table(self.max_skill_len).permute(1, 0, 2).repeat(1, bs, 1) * self.dec_tgt_pos_scale_factor # (MSL, bs, hidden_dim)
            dec_tgt  = torch.zeros_like(dec_tgt_pe) # (MSL, bs, hidden_dim)
            tgt_pad_mask[batch_mask, :] = False # Unmask fully padded tgts to avoid NaNs
        dec_tgt = dec_tgt + dec_tgt_pe
        dec_tgt = self.dec_tgt_norm(dec_tgt)

        # Concatenate full decoder src
        if self.conditional_decode:
            # proprioception features
            state_src = self.dec_input_state_proj(qpos) # (bs, 1|MSL, hidden)
            state_src = self.dec_input_state_norm(state_src).permute(1,0,2) # (1|MSL, bs, hidden)
            state_src = state_src + dec_type_embed[1, :, :] # add type 2 embedding
            state_pe = self.get_pos_table(state_src.shape[0]).permute(1, 0, 2).repeat(1, bs, 1) * self.dec_src_qpos_pos_scale_factor # (1|MSL, bs, hidden)

            # image, only use one image to decode rest of the skills
            img_src, img_pe = img_info # (1|MSL, bs, num_cam, h*w, c&hidden)
            img_src = self.image_proj(img_src) # (1|MSL, bs, num_cam, h*w, hidden)
            img_src = self.image_feat_norm(img_src)
            img_pe = img_pe * self.img_pe_scale_factor
            img_src = img_src.flatten(2,3) # (1|MSL, bs, num_cam*h*w, hidden)
            img_pe = img_pe.flatten(2,3)
            img_src = img_src.permute(0, 2, 1, 3) # (1|MSL, h*num_cam*w, bs, hidden)
            img_pe = img_pe.permute(0, 2, 1, 3)
            img_pos = self.get_pos_table(img_src.shape[0]).permute(1, 0, 2) * self.dec_src_img_pos_scale_factor # (1|MSL, bs, hidden)
            img_pos = img_pos.unsqueeze(1) # (1|MSL, 1, bs, hidden)
            img_src = img_src + img_pos # Add temporal positional encoding
            img_src = img_src.flatten(0,1) # (1|MSL*num_cam*h*w, bs, hidden)
            img_pe = img_pe.flatten(0,1)
            img_src = img_src + dec_type_embed[2, :, :] # add type 3 embedding

            dec_src = torch.cat([state_src, img_src, z_src], axis=0) # (state + img + z|MSL*(state + img + z), bs, hidden)
            dec_src_pe = torch.cat([state_pe, img_pe, z_pe], axis=0)
        else:
            dec_src = z_src # (z, bs, hidden)
            dec_src_pe = z_pe
            # only get skill sections of src and mem masks with no conditional decoding
            src_mask = src_mask[-1:, -1:]
            mem_mask = mem_mask[:,-1:]

        dec_src = dec_src + dec_src_pe
        dec_src = self.dec_src_norm(dec_src)

        # dec_output = self.decoder(src=dec_src,
        #                           src_key_padding_mask=None, 
        #                           src_is_causal=self.autoregressive_decode, # pseudo-causal
        #                           src_mask=src_mask,
        #                           memory_key_padding_mask=None,
        #                           memory_mask=mem_mask,
        #                           memory_is_causal=self.autoregressive_decode, 
        #                           tgt=dec_tgt,
        #                           tgt_key_padding_mask=tgt_pad_mask,
        #                           tgt_is_causal=self.autoregressive_decode,
        #                           tgt_mask=tgt_mask) # (MSL|<, bs, hidden_dim)
        
        dec_output = self.decoder(dec_tgt, dec_src, tgt_mask=tgt_mask, memory_mask=mem_mask,
                                    tgt_key_padding_mask=tgt_pad_mask,
                                    memory_key_padding_mask=None,
                                    tgt_is_causal=self.autoregressive_decode, memory_is_causal=self.autoregressive_decode)

        # Send project output to action dimensions
        a_joint = self.dec_action_joint_proj(dec_output) # (MSL, bs, action_dim - 1)
        a_grip = self.dec_action_gripper_proj(dec_output) # (MSL, bs, 1)
        a_hat = torch.cat((a_joint, a_grip), -1) # (MSL, bs, action_dim)

        return a_hat
    
    
    def sequence_decode(self, z, qpos, actions, img_info,
                        seq_pad_mask, skill_pad_mask,
                        dec_mask, mem_mask, tgt_mask):
        """
        Decode a sequence of skills into actions given a demo trajectory
        args:
            z: (skill_seq, bs, latent_dim)
            qpos: (bs, seq, state_dim)
            actions: (bs, seq, act_dim)
            img_info: (img_src, img_pe): (seq, bs, num_cam, h*w, c & hidden)
            seq_pad_mask: (bs, seq)
            skill_pad_mask: (bs, skill_seq)
            dec_mask: (bs, MSL*(2+num_features*num_cam), ")
            mem_mask: (bs, MSL, MSL*(2+num_features*num_cam))
            tgt_mask: (bs, MSL, MSL)
        returns:
            a_hat: (seq, bs, act_dim)
        """
        bs = seq_pad_mask.shape[0]
        img_src, img_pe = img_info

        ### Decode skills one at a time
        a_hat = torch.zeros(0, bs, self.action_dim, device=self._device) # (0, bs, act_dim)
        for sk in range(z.shape[0]):
            # Get current time step and skill
            t = sk*self.max_skill_len
            tf = (sk+1)*self.max_skill_len
            sk_z = z[sk:sk + 1, :, :] # (1, bs, latent_dim)
            sk_skill_pad_mask = skill_pad_mask[:, sk:sk+1]
            sk_seq_pad_mask = seq_pad_mask[:, t:tf]

            # Get target actions for autoregressive decoding
            if self.autoregressive_decode:
                # Always set first input as zeros, as "start" token, and shift other tgt actions right
                tgt_0 = self.dec_tgt_start_token.repeat(bs, 1, 1) # (bs, 1, act_dim)
                tgt = torch.cat((tgt_0, actions[:,t:tf-1,:]), dim=1) # (bs, MSL, act_dim)
                # If decoding autoregressively, need to pass in all states and img_infos that will be encountered
                qpos_t = qpos[:,t:tf,:] # (bs, MSL, state_dim)
                img_info_t = (img_src[t:tf,...], img_pe[t:tf,...]) # (MSL, bs, num_cam, h*w, c)
            else:
                tgt = None
                qpos_t = qpos[:,t:t+1,:] # (bs, 1, state_dim)
                img_info_t = (img_src[t:t+1,...], img_pe[t:t+1,...]) # (1, bs, num_cam, h*w, c)
                if (sk_unpad := sk_seq_pad_mask.shape[1]) < self.max_skill_len: # Handle case where input is < MSL during training
                    sk_seq_pad_mask = torch.hstack((sk_seq_pad_mask, torch.ones(bs, self.max_skill_len - sk_unpad)))

            # Decode current skill with conditional info
            a_pred = self.skill_decode(sk_z, qpos_t, img_info_t,
                                        sk_skill_pad_mask, sk_seq_pad_mask, 
                                        dec_mask, mem_mask, tgt_mask,
                                        tgt) # (seq, bs, act_dim)
            
            a_hat = torch.vstack((a_hat, a_pred))
        
        return a_hat
    

    def get_action(self, data, t):
        # Define current info
        t_act = t % self.max_skill_len
        dec_skill_pad_mask = torch.zeros(1,1)
        latent = data["latent"]
        img_src, img_pe = data["img_feat"], data["img_pe"]
        _,_,num_cam,num_feats,_ = img_src.shape
        state = data["state"]

        if self.autoregressive_decode:
            if t_act == 0: # Decode new sequence
                self.execution_data = dict()
                self.execution_data["img_feat"] = img_src # (bs, seq, ...)
                self.execution_data["img_pe"] = img_pe
                self.execution_data["state"] = state # (bs, seq, state_dim)
                self.execution_data["dec_tgt"] = self.dec_tgt_start_token # (bs, 1, act_dim)
            else:
                self.execution_data["img_feat"] = torch.cat((self.execution_data["img_feat"], img_src), dim=1)
                self.execution_data["img_pe"] = torch.cat((self.execution_data["img_pe"], img_pe), dim=1)
                self.execution_data["state"] = torch.cat((self.execution_data["state"], state), dim=1)

            dec_tgt = self.execution_data["dec_tgt"]
            img_srcs = self.execution_data["img_feat"].transpose(0,1)
            img_pes = self.execution_data["img_pe"].transpose(0,1)
            states = self.execution_data["state"]

            num_actions = dec_tgt.shape[1]
            dec_src_mask, dec_mem_mask, dec_tgt_mask = get_dec_ar_masks(num_feats*num_cam, num_actions)
            seq_pad_mask = torch.zeros(1,num_actions) # (bs, MSL|<)

            with torch.no_grad():
                a_pred = self.skill_decode(latent, states, (img_srcs,img_pes),
                                            dec_skill_pad_mask, seq_pad_mask,
                                            dec_src_mask, dec_mem_mask, dec_tgt_mask,
                                            tgt=dec_tgt) # (MSL|<, bs, action_dim)
            
            self.execution_data["dec_tgt"] = torch.cat((dec_tgt, a_pred[-1:,...].permute(1,0,2)), dim=1) # (1, seq + 1, act_dim)
            a_t = a_pred.detach()[-1,...] # Take most recent action
        else:
            if t_act == 0:
                seq_pad_mask = torch.zeros(1,self.max_skill_len)
                with torch.no_grad():
                    a_hat = self.skill_decode(latent, state, (img_src,img_pe), 
                                            dec_skill_pad_mask, seq_pad_mask, 
                                            None, None, None, None) # (MSL, bs, action_dim)
                self.execution_data = dict()
                self.execution_data["a_hat"] = a_hat.detach() 
                
            a_t = self.execution_data["a_hat"][t_act,...]

        return a_t


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
