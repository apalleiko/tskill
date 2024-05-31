# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .skill_vae import build as build_vae

def build_skill_model(args):
    return build_vae(args)