from .backbone import Backbone, Joiner, ResnetStateEncoder
from .position_encoding import build_position_encoding

def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False


def get_model(cfg, device=None):
    position_embedding = build_position_encoding(cfg)
    return_interm_layers = cfg["masks"]
    backbone = Backbone(cfg["backbone_name"], True, return_interm_layers, cfg["dilation"])
    joiner = Joiner(backbone, position_embedding).float()
    joiner.num_channels = backbone.num_channels
    joiner.to(device)
    model = ResnetStateEncoder(joiner, cfg["hidden_dim"])

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    """Not implemented for resnet state encoder"""
    raise NotImplementedError