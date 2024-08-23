import torch
from torch import nn
import numpy as np

import IPython
e = IPython.embed

class StateDecoder(nn.Module):
    """ Transformer Skill CVAE Module for encoding/decoding skill sequences"""
    def __init__(self,
                 device, **kwargs):
        """ Initializes the model.
        Parameters:
            
        """
        super().__init__()
        ### General args
        self._device = device
        self.norm = nn.LayerNorm

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

        
        return dict(a_hat=a_hat, z_hat=z_hat, vae_out=vae_out)
    

    def to(self, device):
        model = super().to(device)
        model._device = device
        model.vae._device = device
        return model


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
