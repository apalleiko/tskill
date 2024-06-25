# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Resnet backbone module. Modified from Facebook DETR code.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from .util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters(): # only train later layers
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos
    

class ResnetStateEncoder(nn.Module):
    def __init__(self, joiner, hidden_dim, **kwargs) -> None:
        super().__init__(**kwargs)
        self.backbone = joiner
        self.num_channels = joiner.num_channels
        self.hidden_dim = hidden_dim
        self.image_proj = nn.Conv2d(self.num_channels, self.hidden_dim, kernel_size=1)
    

    def forward(self, images):
        """Get image observation features and position embeddings
        - images: (bs, seq num_cam, channel, h, w)"""
        img_seq_features = []
        img_seq_pos = []

        for t in range(images.shape[1]):
            t_feat, t_pos = self.image_encode(images[:, t, ...])
            img_seq_features.append(t_feat)
            img_seq_pos.append(t_pos)

        img_src = torch.stack(img_seq_features, axis=0) # (seq, bs, c, h*num_cam*w)
        img_pos = torch.stack(img_seq_pos, axis=0) # (seq, 1, c, h*num_cam*w)

        seq, bs, c, hw = img_src.shape
        img_src = img_src.permute(0, 1, 3, 2) # (seq, bs, h*num_cam*w, c) TODO Is this the right feature order?
        img_pe = img_pos.permute(0, 1, 3, 2).repeat(1, bs, 1, 1)
        
        return img_src, img_pe


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
            feat = torch.cat(all_cam_features, axis=3) # (bs, hidden, h, num_cam*w)
            pos = torch.cat(all_cam_pos, axis=3) 

        # flatten
        feat = feat.flatten(2) # (bs, hidden, h*num_cam*w)
        pos = pos.flatten(2) # (1, hidden, h*num_cam*w)
        
        return feat, pos        