from .backbone import Backbone, Joiner
from .position_encoding import build_position_encoding

def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False


def get_model(cfg, device=None):
    position_embedding = build_position_encoding(cfg)
    return_interm_layers = cfg["masks"]
    backbone = Backbone(cfg["backbone_name"], False, return_interm_layers, cfg["dilation"])
    model = Joiner(backbone, position_embedding).float()
    model.num_channels = backbone.num_channels
    model.to(device)

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    """Not implemented for resnet state encoder"""
    raise NotImplementedError