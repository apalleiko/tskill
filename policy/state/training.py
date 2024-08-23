from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from policy.training import BaseTrainer
from policy.state.config import get_trainer

from .state_decode import StateDecoder


class Trainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        model,
        optimizer,
        device=None,
        scheduler=None
    ):
        self.epoch_it = 0
        self.step_it = 0

        self.model: StateDecoder = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def epoch_step(self):
        if self.scheduler is not None and self.epoch_it > 0:
            self.scheduler.step()
        self.epoch_it += 1

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        loss_dict, metric_dict = self.compute_loss(data)

        loss = 0.0
        _dict = {}
        for k, v in loss_dict.items():
            loss += v
            _dict[k] = v.item()
        loss_dict = _dict
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1.0)
        self.optimizer.step()

        {k: v.item() for k, v in metric_dict.items() if "vector" not in k}
        self.step_it += 1

        return loss_dict, metric_dict

    def evaluate(self, val_loader):
        self.model.eval()

        eval_list = defaultdict(list)
        eval_metric_list = defaultdict(list)

        for data in tqdm(val_loader, desc="[Validating]"):
            eval_step_dict, eval_metric_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

            for k, v in eval_metric_dict.items():
                eval_metric_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        eval_metric_dict = {k: np.mean(v) for k, v in eval_metric_list.items()}

        return eval_dict, eval_metric_dict

    def eval_step(self, data):
        with torch.no_grad():
            loss_dict, metric_dict = self.compute_loss(data, alt=self.val_alt)
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        metric_dict = {k: v.item() for k, v in metric_dict.items()}
        return loss_dict, metric_dict

    def compute_loss(self, data, **kwargs):
        loss_dict = dict()
        metric_dict = dict()
        
        out = self.model(data)

        loss_dict["act_plan_loss"] = self.act_weight * F.mse_loss(a_hat_l, a_targ_l, reduction="sum") / num_actions
        loss_dict["z_loss"] = self.z_weight * F.mse_loss(z_hat_l, z_targ_l, reduction="sum") / num_latent

        for k,v in self.model.metrics.items():
            metric_dict[k] = v.detach()

        return loss_dict, metric_dict