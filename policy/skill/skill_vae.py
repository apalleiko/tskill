import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import dill as pickle
from policy.dataset.masking_utils import get_dec_ar_masks, get_enc_causal_masks


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
            x = STEFunction.apply(x)
            return x


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class Quantization(nn.Module):
    def __init__(self, alpha, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.z_dim = self.alpha.shape[0]
        self._device = device
        self.round_ste = StraightThroughEstimator()

        self.skill_dict = dict()
        self.reverse_skill_dict = dict()
        indices = torch.ones(tuple(self.alpha.to(torch.int).tolist())).nonzero().to(self._device)
        for i in range(indices.shape[0]):
            skill = torch.floor(indices[i] - torch.floor(self.alpha / 2))
            self.skill_dict[tuple(skill.tolist())] = i
            self.reverse_skill_dict[i] = skill

    def forward(self, x):
        z = torch.floor(self.alpha / 2) * torch.tanh(x)
        z = self.round_ste(z)
        return z

    def indices_to_codes(self, indices):
        """Turn a tensor of indices (bs, seq) into a tensor of codes (bs, seq, z_dim)"""
        z = torch.zeros(indices.shape[0],indices.shape[1],self.z_dim, device=self._device)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                z[i,j,:] = self.reverse_skill_dict[indices[i,j].item()]
        return z

    def codes_to_indices(self, codes):
        """Turn a tensor of codes (bs, seq, z_dim) into a tensor of codes (bs, seq)"""
        indices = torch.zeros(codes.shape[0],codes.shape[1], device=self._device)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                indices[i,j] = self.skill_dict[tuple(codes[i,j,:].tolist())]
        return indices

    def top_k_sampling(self, logits, k=5, temperature=1.0):
        # Apply temperature scaling
        scaled_logits = logits / temperature
        # Find the top k values and indices
        top_values, top_indices = torch.topk(scaled_logits, k, dim=-1)
        # Compute probabilities from top values
        top_probs = torch.softmax(top_values, dim=-1)
        # Sample token index from the filtered probabilities
        sampled_indices = torch.zeros(top_probs.shape[0], top_probs.shape[1], 1, device=self._device)
        for i in range(top_probs.shape[0]):
            sampled_indices[i,:,:] = torch.multinomial(top_probs[i,:,:], num_samples=1, replacement=True)
        sampled_indices = sampled_indices.to(torch.int64)
        # Map the sampled index back to the original logits tensor
        original_indices = top_indices.gather(-1, sampled_indices)
        return original_indices
    
    def top_k(self, logits, k=5, temperature=1.0):
        # Apply temperature scaling
        scaled_logits = logits / temperature
        # Find the top k values and indices
        top_values, top_indices = torch.topk(scaled_logits, k, dim=-1)
        # Compute probabilities from top values
        top_probs = torch.softmax(top_values, dim=-1)
        # Get deterministic max value
        sampled_indices = torch.max(top_probs, dim=-1)
        # Map the sampled index back to the original logits tensor
        original_indices = top_indices.gather(-1, sampled_indices)
        return original_indices


class TSkillCVAE(nn.Module):
    """ Transformer Skill CVAE Module for encoding/decoding skill sequences"""
    def __init__(self, stt_encoder, 
                 encoder, decoder, 
                 state_dim, action_dim,
                 max_skill_len, alpha,
                 autoregressive_decode,
                 decoder_obs,
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
            device: device to operate on
            autoregressive: Whether action decoding is autoregressive
        kwargs:
            encode_state: whether to encode image and qpos to skills
            encoder_is_causal: whether causal masks are applied to skill generation
        """
        super().__init__()
        ### General args
        self._device = device
        self.decoder = decoder
        self.encoder = encoder
        self.hidden_dim = decoder.d_model
        self.alpha = torch.tensor(alpha).to(self._device, torch.float)
        self.num_skills = torch.prod(self.alpha).to(self._device, torch.int).item()
        self.z_dim = self.alpha.shape[0]
        self.max_skill_len = max_skill_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.norm = nn.LayerNorm
        self.decoder_obs = decoder_obs
        self.autoregressive_decode = autoregressive_decode
        self.encode_state = encode_state
        self.encoder_is_causal = encoder_is_causal

        self.vq = Quantization(self.alpha, device)

        ### Get a sinusoidal position encoding table for a given sequence size
        self.get_pos_table = lambda x: get_sinusoid_encoding_table(x, self.hidden_dim).to(self._device) # (1, x, hidden_dim)
        # Create pe scaling factors
        self.enc_src_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.001)
        self.enc_tgt_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
        self.dec_src_pos_scale_factor = nn.Parameter(torch.ones(1) * 0.01)
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
        self.enc_z = nn.Linear(self.hidden_dim, self.z_dim) # project hidden -> latent [std; var]
        self.dec_z = nn.Linear(self.z_dim, self.hidden_dim) # project latent -> hidden

        ### Decoder
        self.dec_input_state_proj = nn.Linear(state_dim, self.hidden_dim)
        self.dec_input_state_norm = self.norm(self.hidden_dim)
        self.dec_input_z_norm = self.norm(self.hidden_dim)
        self.dec_src_norm = self.norm(self.hidden_dim)
        self.dec_tgt_norm = self.norm(self.hidden_dim)
        self.dec_tgt_start_token = nn.Parameter(torch.zeros(1,1,self.action_dim))

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
            /img_feat: bs, seq, num_cam, h*w, c
            /img_pe: bs, seq, num_cam, hidden, h, w
            actions: bs, seq, action_dim
            seq_pad_mask: Padding mask for input sequence (bs, seq)
            skill_pad_mask: Padding mask for skill sequence (bs, max_num_skills)
        """
        qpos = data["state"].to(self._device)
        actions = data["actions"].to(self._device)
        seq_pad_mask = data["seq_pad_mask"].to(self._device, torch.bool)
        skill_pad_mask = data["skill_pad_mask"].to(self._device, torch.bool)

        BS, SEQ = seq_pad_mask.shape
        _, MNS = skill_pad_mask.shape

        ### Which image input to use.
        if not self.encode_state:
            img_src, img_pe = torch.zeros(SEQ, BS, 1, 1, 1, device=self._device), torch.zeros(SEQ, BS, 1, 1, 1, device=self._device)
        elif kwargs.get("use_precalc",False):
            img_src = data["img_feat"].transpose(0,1).to(self._device) # (seq, bs, num_cam, h*w, c)
            img_pe = data["img_pe"].transpose(0,1).to(self._device)
        else:
            images = data["rgb"].to(self._device)
            img_src, img_pe = self.stt_encoder(images) # (seq, bs, num_cam, h*w, c)

        ### Get causal/other masks, if applicable
        # Encoder causal masks
        if self.encoder_is_causal:
            enc_src_mask, enc_mem_mask, enc_tgt_mask = get_enc_causal_masks(SEQ, MNS, self.max_skill_len, self.decoder_obs, device=self._device)
        else:
            enc_src_mask, enc_mem_mask = enc_tgt_mask = None
        # Decoder ar masks
        if self.autoregressive_decode:
            dec_mem_mask, dec_tgt_mask = get_dec_ar_masks(MNS, self.max_skill_len, self.decoder_obs, device=self._device)
        else:
            dec_mem_mask = dec_tgt_mask = None

        ### Encode inputs to latent
        z = self.forward_encode(qpos, actions, (img_src, img_pe),
                                seq_pad_mask, skill_pad_mask,
                                enc_src_mask, enc_mem_mask, enc_tgt_mask) # (skill_seq, bs, latent_dim)
        
        # Check for NaNs
        if torch.any(torch.isnan(z)):
            with open("ERROR_DATA.pickle",'+wb') as f:
                pickle.dump(data, f)
            raise ValueError(f"NaNs encountered during encoding! Data saved to ERROR_DATA.pickle")
        
        ### Decode skill sequence
        a_hat = self.skill_decode(z, 
                                  seq_pad_mask, skill_pad_mask,
                                  dec_mem_mask, dec_tgt_mask)
    
        # Check for NaNs
        if torch.any(torch.isnan(a_hat)):
            with open("ERROR_DATA.pickle",'+wb') as f:
                pickle.dump(data, f)
            raise ValueError(f"NaNs encountered during decoding! Data saved to ERROR_DATA.pickle")

        # Reorder outputs to match inputs
        a_hat = a_hat.permute(1,0,2) # (bs, seq, act_dim)
        z = z.permute(1,0,2) # (bs, seq, z_dim)

        return dict(a_hat=a_hat, z=z)
        

    def forward_encode(self, qpos, actions, img_info, 
                       src_pad_mask, tgt_pad_mask, 
                       src_mask=None, mem_mask=None, tgt_mask=None):
        """Encode skill sequence based on robot state, action, and image input sequence"""
        img_feat, img_pe = img_info # (seq, bs, num_cam, h*w, hidden)
        BS, SEQ, _ = qpos.shape # Use bs for masking
        NUM_CAM = img_feat.shape[2]
        MNS = tgt_pad_mask.shape[1]

        ### Obtain latent z from state, action, image sequence
        enc_type_embed = self.input_embed.weight * self.input_embed_scale_factor
        enc_type_embed = enc_type_embed.unsqueeze(1).repeat(1, BS, 1) # (4, bs, hidden_dim)

        # project action sequences to hidden dim (bs, seq, action_dim)
        action_src = self.enc_action_proj(actions) # (bs, seq, hidden_dim)
        action_src = self.enc_action_norm(action_src).permute(1, 0, 2) # (seq, bs, hidden_dim)
        
        if self.encoder_is_causal:
            for i in range(MNS):
                src_pad_mask[:,i*self.max_skill_len] = False # Prevents NANs by having fully padded memory for a skill

        if self.encode_state:
            # Add in action src type
            action_src = action_src + enc_type_embed[0, :, :] # add type 1 embedding
            
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
            img_src = torch.vstack([img_src[:,m,...] for m in range(NUM_CAM)]) # (seq*num_cam, bs, hidden)
            img_src = img_src + enc_type_embed[2, :, :] # add type 3 embedding

            # encoder src
            enc_src = torch.cat([qpos_src, action_src, img_src], axis=0) # ((2+num_cam)*seq, bs, hidden_dim)
            # obtain seq position embedding for input
            enc_src_pe = self.get_pos_table(SEQ).permute(1, 0, 2) * self.enc_src_pos_scale_factor # (seq, 1, hidden_dim)
            enc_src_pe = enc_src_pe.repeat((2+NUM_CAM), BS, 1)  # ((2+num_cam)*seq, bs, hidden_dim)
            enc_src = self.enc_src_norm(enc_src)
            enc_src = enc_src + enc_src_pe

            # Repeat same padding mask for new seq length (same pattern)
            src_pad_mask = src_pad_mask.repeat(1,(2+NUM_CAM)) # (bs, (2+num_cam)*seq)
            if src_mask is not None:
                src_mask = src_mask.repeat((2+NUM_CAM), (2+NUM_CAM)) # ((2+num_cam)*seq, (2+num_cam)*seq)
            if mem_mask is not None:
                mem_mask = mem_mask.repeat(1, (2+NUM_CAM)) # (skill_seq, (2+num_cam)*seq)
        else:
            enc_src = action_src # (seq, bs, hidden_dim)
            # obtain seq position embedding for input
            enc_src_pe = self.get_pos_table(SEQ).permute(1, 0, 2) * self.enc_src_pos_scale_factor # (seq, bs, hidden_dim)
            enc_src = self.enc_src_norm(enc_src)
            enc_src = enc_src + enc_src_pe

        # encoder tgt
        enc_tgt = torch.zeros(MNS, BS, self.hidden_dim).to(self._device) 
        enc_tgt_pe = self.get_pos_table(MNS).permute(1, 0, 2).repeat(1, BS, 1) * self.enc_tgt_pos_scale_factor # (skill_seq, bs, hidden_dim)
        enc_tgt = self.enc_tgt_norm(enc_tgt)
        enc_tgt = enc_tgt + enc_tgt_pe

        # query encoder model
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
                                    tgt_key_padding_mask=None,
                                    memory_key_padding_mask=src_pad_mask,
                                    tgt_is_causal=False, memory_is_causal=False) # (skill_seq, bs, hidden_dim)
        
        e = self.enc_z(enc_output)
        z = self.vq(e)

        return z

    def skill_decode(self, z,
                     src_pad_mask, tgt_pad_mask,
                     mem_mask=None, tgt_mask=None):
        """
        Decode an individual skill into actions, possibly based on current 
        image and state depending on self.conditional_decode
        args: 
            z: (bs, skill_seq, latent_dim)
            src_pad_mask: (bs, skill_seq)
            tgt_pad_mask: (bs, MSL|<)
            mem_mask: (skill_seq*MSL, skill_seq)
            tgt_mask: (MSL|<, MSL|<)
        returns:
            a_hat: (MSL*skill_seq, bs, act_dim)
        """

        ### Move inputs to device (needed for evaluation)
        if mem_mask is not None:
            mem_mask = mem_mask.to(self._device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(self._device)

        num_z, bs,_ = z.shape

        # skills
        z_src = self.dec_z(z) # (num_z, bs, hidden_dim)
        z_src = self.dec_input_z_norm(z_src) # (num_z, bs, hidden_dim)
        z_pe = self.get_pos_table(num_z).permute(1, 0, 2).repeat(1, bs, 1) * self.dec_src_pos_scale_factor # (num_z, bs, hidden_dim)

        dec_tgt = torch.zeros(self.max_skill_len*num_z, bs, self.hidden_dim, device=self._device) # (MSL*num_z, bs, hidden_dim)
        dec_tgt_pe = self.get_pos_table(dec_tgt.shape[0]).permute(1, 0, 2).repeat(1, bs, 1) * self.dec_tgt_pos_scale_factor # (MSL|<, bs, hidden_dim)
        dec_tgt = self.dec_tgt_norm(dec_tgt)
        dec_tgt = dec_tgt + dec_tgt_pe

        dec_src = z_src # (z, bs, hidden)
        dec_src_pe = z_pe
        dec_src = self.dec_src_norm(dec_src)
        dec_src = dec_src + dec_src_pe

        src_mask = None
        if self.autoregressive_decode:
            src_pad_mask = None
            tgt_pad_mask = None

        # dec_output = self.decoder(src=dec_src,
        #                           src_key_padding_mask=None, 
        #                           src_is_causal=False,
        #                           src_mask=src_mask,
        #                           memory_key_padding_mask=None,
        #                           memory_mask=mem_mask,
        #                           memory_is_causal=False, 
        #                           tgt=dec_tgt,
        #                           tgt_key_padding_mask=tgt_pad_mask,
        #                           tgt_is_causal=self.autoregressive_decode,
        #                           tgt_mask=tgt_mask) # (MSL|<, bs, hidden_dim)
        
        dec_output = self.decoder(dec_tgt, dec_src, tgt_mask=tgt_mask, memory_mask=mem_mask,
                                    tgt_key_padding_mask=tgt_pad_mask,
                                    memory_key_padding_mask=src_pad_mask,
                                    tgt_is_causal=self.autoregressive_decode)

        # Send project output to action dimensions
        a_joint = self.dec_action_joint_proj(dec_output) # (MSL, bs, action_dim - 1)
        a_grip = self.dec_action_gripper_proj(dec_output) # (MSL, bs, 1)
        a_hat = torch.cat((a_joint, a_grip), -1) # (MSL, bs, action_dim)

        return a_hat
    
    def get_action(self, data):
        # Define current info
        latent = data["latent"] # (bs, num_z, z_dim)
        bs, num_z, _ = latent.shape
        skill_pad_mask = torch.zeros(bs,num_z)
        seq_pad_mask = torch.zeros(bs,num_z*self.max_skill_len)

        if self.autoregressive_decode:
            dec_mem_mask, dec_tgt_mask = get_dec_ar_masks(num_z, self.max_skill_len, self.decoder_obs)
        else:
            dec_mem_mask, dec_tgt_mask = None

        z = latent.transpose(0,1)
        with torch.no_grad():
            a_hat = self.skill_decode(z, 
                                      skill_pad_mask, seq_pad_mask,
                                      dec_mem_mask, dec_tgt_mask) # (MSL, bs, action_dim)
        self.execution_data = dict()
        self.execution_data["a_hat"] = a_hat.detach()

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
