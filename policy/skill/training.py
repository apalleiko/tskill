from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from policy.training import BaseTrainer

from .skill_vae import TSkillCVAE


def normal_kl(a, b=None):
    """Computes KL divergence based on base normal dist."""
    if b is None:
        mean = torch.zeros_like(a.mean)
        std = torch.ones_like(a.mean)
        b = torch.distributions.Normal(mean, std)

    return torch.distributions.kl.kl_divergence(a, b)


class Trainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        model: TSkillCVAE,
        optimizer,
        device=None,
    ):
        self.epoch_it = 0
        self.step_it = 0

        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.kl_weights = cfg["loss"]["kl_weights"]

    def epoch_step(self):
        self.epoch_it += 1

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(data)

        loss = 0.0
        _dict = {}
        for k, v in loss_dict.items():
            loss += v
            _dict[k] = v.item()
        loss_dict = _dict

        loss.backward()
        self.optimizer.step()
        self.step_it += 1

        return loss_dict

    def evaluate(self, val_loader):
        self.model.eval()

        eval_list = defaultdict(list)

        for data in tqdm(val_loader, desc="[Validating]"):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}

        return eval_dict

    def eval_step(self, data):
        with torch.no_grad():
            loss_dict = self.compute_loss(data)
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        return loss_dict

    def compute_loss(self, data):
        loss_dict = dict()
        bs, seq, dim = data["actions"].shape
        a_targ = data["actions"].to(self.device)

        out = self.model(data)
        a_hat = out["a_hat"]

        mu, logvar = out["mu"], out["logvar"]
        std = (logvar / 2).exp()
        dist = torch.distributions.Normal(mu, std)

        loss_dict["kldiv_loss"] = self.kl_weights * normal_kl(dist, None).sum()

        loss_dict["act_loss"] = F.mse_loss(a_hat, a_targ, reduction="sum") / (
            bs * seq * dim
        )

        return loss_dict