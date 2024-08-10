import os
from .state_decode import TSkillPlan
from .training import Trainer
import torch
from torch import nn
from policy.skill.config import get_model as get_vae
from policy import config
from policy.checkpoints import CheckpointIO


def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False


def build_transformer(args):

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
    cfg_model = cfg["model"]
    # for name in ["state_encoder"]:
    #     # TODO As is, since reusing the vae state encoder the hidden dim has to match the vaes
    #     cfg_model[name].update({"hidden_dim": cfg_model["hidden_dim"]})

    if device is None:
        is_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if is_cuda else "cpu")

    train_stt_encoder = cfg["training"].get("lr_state_encoder", 0)

    transformer = build_transformer(cfg_model)

    if cfg.get("vae_cfg",None) is None:
        vae_cfg = config.load_config(os.path.join(cfg_model["vae_path"],"config.yaml"))
        cfg["vae_cfg"] = vae_cfg
    else:
        vae_cfg = cfg["vae_cfg"]
    vae = get_vae(vae_cfg, device=device)
    checkpoint_io = CheckpointIO(cfg_model["vae_path"], model=vae)
    load_dict = checkpoint_io.load("model_best.pt")
    vae.conditional_decode = cfg_model["conditional_decode"]

    stt_encoder = vae.stt_encoder

    if not train_stt_encoder:
        print("Freezing state encoder network!")
        freeze_network(stt_encoder)
    
    if not cfg["training"].get("train_vae",False):
        print("Freezing CVAE network!")
        freeze_network(vae)

    model = TSkillPlan(
        transformer,
        vae,
        device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, scheduler, **kwcfg):
    trainer = Trainer(cfg, model, optimizer, device=device, scheduler=scheduler)
    return trainer
