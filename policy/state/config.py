import os
from .state_decode import StateDecoder
from .training import Trainer
import torch
from policy import config

def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False


def get_model(cfg, device=None):
    cfg_model = cfg["model"]

    if device is None:
        is_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if is_cuda else "cpu")

    model = StateDecoder(
        device=device,
    )

    return model


def get_trainer(model, optimizer, cfg, device, scheduler, **kwcfg):
    trainer = Trainer(cfg, model, optimizer, device=device, scheduler=scheduler)
    return trainer
