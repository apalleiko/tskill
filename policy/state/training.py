from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from policy.training import BaseTrainer
from policy.skill.config import get_trainer

from .state_decode import TSkillPlan


class Trainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        model: TSkillPlan,
        optimizer,
        device=None,
        scheduler=None
    ):
        self.epoch_it = 0
        self.step_it = 0
        self.zero_grad = True

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.gradient_accumulation = cfg["training"].get("gradient_accumulation",1)

    def epoch_step(self):
        if self.scheduler is not None and self.epoch_it > 0:
            self.scheduler.step()
        self.epoch_it += 1

    def train_step(self, data):
        self.model.train()
        if self.zero_grad:
            self.optimizer.zero_grad()
            self.zero_grad = False
        loss_dict, metric_dict = self.compute_loss(data)
        metric_dict = {k: v for k, v in metric_dict.items()}

        loss = 0.0
        _dict = {}
        for k, v in loss_dict.items():
            loss += v
            _dict[k] = v.item()
        loss_dict = _dict

        loss = loss / self.gradient_accumulation
        loss.backward()
        if (self.step_it + 1) % self.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1.0)
            self.optimizer.step()
            self.zero_grad = True

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
            loss_dict, metric_dict = self.compute_loss(data)
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        metric_dict = {k: v.item() for k, v in metric_dict.items() if "vector" not in k}
        return loss_dict, metric_dict

    def compute_loss(self, data):
        loss_dict = dict()
        metric_dict = dict()
        a_targ = data["actions"].to(self.device) # (bs, seq, act_dim)
        seq_pad_mask = data["seq_pad_mask"].to(self.device) # (bs, seq)
        skill_pad_mask = data["skill_pad_mask"].to(self.device) # (bs, skill_seq)
        bs, seq, act_dim = a_targ.shape
        latent_dim = self.model.z_dim

        # Correct mask size and convert to (0,1) where 1 is attended
        action_loss_mask = seq_pad_mask.unsqueeze(-1).repeat(1, 1, act_dim)
        action_loss_mask = torch.logical_not(action_loss_mask)
        num_actions = torch.sum(action_loss_mask.to(torch.int16))

        # Correct mask size and convert to (0,1) where 1 is attended
        latent_loss_mask = skill_pad_mask.unsqueeze(-1).repeat(1, 1, latent_dim)
        latent_loss_mask = torch.logical_not(latent_loss_mask)
        num_latent = torch.sum(latent_loss_mask.to(torch.int16))
        
        # Get model outputs
        out = self.model(data, use_precalc=self.use_precalc)
        a_hat = out["a_hat"]
        z_hat = out["z_hat"]
        z_targ = out["vae_out"]["mu"].to(self.device) # (bs, skill_seq, z_dim)

        # Set target and pred actions to 0 for padded sequence outputs
        a_hat_l = a_hat[action_loss_mask]
        a_targ_l = a_targ[action_loss_mask]
        loss_dict["act_plan_loss"] = self.act_weight * F.mse_loss(a_hat_l, a_targ_l, reduction="sum") / num_actions

        # Set target and pred latents to 0 for padded skill outputs
        z_hat_l = z_hat[latent_loss_mask]
        z_targ_l = z_targ[latent_loss_mask]
        loss_dict["z_loss"] = self.z_weight * F.mse_loss(z_hat_l, z_targ_l, reduction="sum") / num_latent

        metric_dict["ahat_vector_traj"] = a_hat

        for k,v in self.model.metrics.items():
            metric_dict[k] = v

        if self.train_vae:
            vae_loss_dict, vae_metric_dict = self.vae_trainer.raw_loss(out["vae_out"], data)
            for k,v in vae_loss_dict.items():
                loss_dict[f"{k}"] = v
            for k,v in vae_metric_dict.items():
                metric_dict[f"{k}"] = v

        return loss_dict, metric_dict