from .backbone import Backbone, Joiner
from .position_encoding import build_position_encoding

def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False


def get_model(cfg, device=None):
    cfg_backbone = cfg["backbone"]
    train_backbone = cfg_backbone["lr_backbone"] > 0

    position_embedding = build_position_encoding(cfg_backbone)
    return_interm_layers = cfg_backbone["masks"]
    backbone = Backbone(cfg_backbone["backbone_name"], train_backbone, return_interm_layers, cfg_backbone["dilation"])
    model = Joiner(backbone, position_embedding).float()
    model.num_channels = backbone.num_channels

    if not train_backbone:
        print("freezing vae network!")
        freeze_network(model)

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    """Not implemented for resnet state encoder"""
    raise NotImplementedError