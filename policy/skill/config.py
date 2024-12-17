from .skill_vae import TSkillCVAE
from .training import Trainer
import torch
from torch import nn
from policy.perception.resnet.config import get_model as get_stt_encoder


def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False


def build_transformer(args):

    decoder_layer = nn.TransformerDecoderLayer(args["hidden_dim"], args["nheads"], args["dim_feedforward"], args["dropout"],
                                               norm_first=args["pre_norm"])
    decoder_norm = nn.LayerNorm(args["hidden_dim"])
    decoder =  nn.TransformerDecoder(decoder_layer, args["dec_layers"], decoder_norm)
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
    for name in ["state_encoder", "encoder", "decoder"]:
        cfg_model[name].update({"hidden_dim": cfg_model["hidden_dim"]})

    stt_encoder = get_stt_encoder(cfg_model["state_encoder"])
    encoder = build_transformer(cfg_model["encoder"])
    decoder = build_transformer(cfg_model["decoder"])

    if device is None:
        is_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if is_cuda else "cpu")

    train_stt_encoder = cfg["training"].get("lr_state_encoder", 0)
    if not train_stt_encoder:
        print("freezing state encoder network!")
        freeze_network(stt_encoder)

    cond_dec = cfg_model.get("conditional_decode",False)
    ar_dec = cfg_model.get("autoregressive_decode",False)
    encode_state = cfg_model.get("encode_state",True)
    encoder_is_causal = cfg_model.get("encoder_is_causal",False)

    model = TSkillCVAE(
        stt_encoder,
        encoder,
        decoder,
        state_dim=cfg_model["state_dim"],
        action_dim=cfg_model["action_dim"],
        max_skill_len=cfg_model["max_skill_len"],
        z_dim=cfg_model["z_dim"],
        conditional_decode=cond_dec,
        autoregressive_decode=ar_dec,
        encode_state=encode_state,
        encoder_is_causal=encoder_is_causal,
        device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, scheduler, **kwcfg):
    trainer = Trainer(cfg, model, optimizer, device=device, scheduler=scheduler)
    return trainer
