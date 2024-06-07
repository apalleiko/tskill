import os.path

from .skill_vae import TSkillCVAE
# from .training import Trainer
import torch
from torch import nn
from policy.perception.resnet.config import get_model as get_backbone
from policy.config import load_config


def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False


def build_transformer(args):#
    # DETR Transformer
    # return Transformer(
    #     d_model=args.hidden_dim,
    #     dropout=args.dropout,
    #     nhead=args.nheads,
    #     dim_feedforward=args.dim_feedforward,
    #     num_encoder_layers=args.enc_layers,
    #     num_decoder_layers=args.dec_layers,
    #     normalize_before=args.pre_norm,
    #     return_intermediate_dec=True,
    # )

    return nn.Transformer(
        d_model=args["hidden_dim"],
        dropout=args["dropout"],
        nhead=args["nheads"],
        dim_feedforward=args["dim_feedforward"],
        num_encoder_layers=args["enc_layers"],
        num_decoder_layers=args["dec_layers"],
        norm_first=args["pre_norm"],
    )


def get_model(cfg, device=None):
    # From image
    cfg = cfg["model"]
    for name in ["backbone", "encoder", "decoder"]:
        cfg[name].update({"hidden_dim": cfg["hidden_dim"]})

    backbone = get_backbone(cfg)
    encoder = build_transformer(cfg["encoder"])
    decoder = build_transformer(cfg["decoder"])

    if device is None:
        is_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if is_cuda else "cpu")

    model = TSkillCVAE(
        backbone,
        encoder,
        decoder,
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        max_skill_len=cfg["max_skill_len"],
        z_dim=cfg["z_dim"],
        device=device
    )

    n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: %.2fM" % (n_trainable_parameters/1e6,))
    n_parameters = sum(p.numel() for p in model.parameters())
    print("Number of total parameters: %.2fM" % (n_parameters/1e6,))

    return model


def get_trainer(model, optimizer, cfg, device, **kwcfg):
    raise NotImplementedError
    # trainer = Trainer(cfg, model, optimizer, device=device)

    # return trainer
