import os
from .skill_plan import TSkillPlan
from .training import Trainer
import torch
from torch import nn
from policy.skill.config import get_model as get_vae
from policy.perception.resnet.config import get_model as get_stt_encoder
from policy import config

def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False


def build_transformer(args):

    decoder_layer = nn.TransformerDecoderLayer(args["hidden_dim"], args["nheads"], args["dim_feedforward"], args["dropout"],
                                               norm_first=args["pre_norm"])
    decoder_norm = nn.LayerNorm(args["hidden_dim"])
    decoder = nn.TransformerDecoder(decoder_layer, args["dec_layers"], decoder_norm)
    decoder.d_model = args["hidden_dim"]
    return decoder

    # return nn.Transformer(
    #     d_model=args["hidden_dim"],
    #     dropout=args["dropout"],
    #     nhead=args["nheads"],
    #     dim_feedforward=args["dim_feedforward"],
    #     num_encoder_layers=args["enc_layers"],
    #     num_decoder_layers=args["dec_layers"],
    #     norm_first=args["pre_norm"],
    # )


def get_model(cfg, device=None):
    cfg_model = cfg["model"]

    if device is None:
        is_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if is_cuda else "cpu")

    train_stt_encoder = cfg["training"].get("lr_state_encoder", 0)
    cond_plan = cfg_model.get("conditional_plan",False)
    goal_mode = cfg_model.get("goal_mode","image")

    transformer = build_transformer(cfg_model)


    if cfg.get("vae_cfg",None) is None:
        vae_cfg = config.load_config(os.path.join(cfg_model["vae_path"],"config.yaml"))
        cfg["vae_cfg"] = vae_cfg
    else:
        vae_cfg = cfg["vae_cfg"]
    vae = get_vae(vae_cfg, device=device)
    
    if not train_stt_encoder:
        print("Freezing state encoder network!")
        stt_encoder = vae.stt_encoder
        freeze_network(stt_encoder)
    else: # Train separate VAE for the model
        stt_encoder = get_stt_encoder(vae_cfg["model"]["state_encoder"])

    if not cfg["training"].get("train_vae",False):
        print("Freezing CVAE network!")
        freeze_network(vae)
    
    model = TSkillPlan(
        transformer,
        vae,
        cond_plan,
        goal_mode,
        stt_encoder,
        device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, scheduler, **kwcfg):
    trainer = Trainer(cfg, model, optimizer, device=device, scheduler=scheduler)
    return trainer
