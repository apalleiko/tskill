from .skill_vae import TSkillCVAE
from .training import Trainer
import torch
from torch import nn
from policy.perception.resnet.config import get_model as get_stt_encoder


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
    cfg_model = cfg["model"]
    for name in ["state_encoder", "encoder", "decoder"]:
        cfg_model[name].update({"hidden_dim": cfg_model["hidden_dim"]})

    train_stt_encoder = cfg["training"].get("lr_state_encoder", 0)
    cfg_model["state_encoder"]["train"] = True

    stt_encoder = get_stt_encoder(cfg_model["state_encoder"])
    encoder = build_transformer(cfg_model["encoder"])
    decoder = build_transformer(cfg_model["decoder"])

    if device is None:
        is_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if is_cuda else "cpu")

    if not train_stt_encoder:
        print("freezing state encoder network!")
        freeze_network(stt_encoder)

    model = TSkillCVAE(
        stt_encoder,
        encoder,
        decoder,
        state_dim=cfg_model["state_dim"],
        action_dim=cfg_model["action_dim"],
        max_skill_len=cfg_model["max_skill_len"],
        z_dim=cfg_model["z_dim"],
        device=device
    )

    # n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Number of trainable parameters: %.2fM" % (n_trainable_parameters/1e6,))
    # n_parameters = sum(p.numel() for p in model.parameters())
    # print("Number of total parameters: %.2fM" % (n_parameters/1e6,))

    return model


def get_trainer(model, optimizer, cfg, device, scheduler, **kwcfg):
    trainer = Trainer(cfg, model, optimizer, device=device, scheduler=scheduler)
    return trainer
