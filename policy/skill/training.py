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
        scheduler=None
    ):
        self.epoch_it = 0
        self.step_it = 0
        self.zero_grad = True

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.kl_weights = cfg["loss"]["kl_weights"]
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
        action_loss_mask = torch.logical_not(action_loss_mask).to(torch.int16)
        num_actions = torch.sum(action_loss_mask)
        
        kl_loss_mask_not = skill_pad_mask.unsqueeze(-1).repeat(1,1,latent_dim).to(torch.int16)
        kl_loss_mask = torch.logical_not(kl_loss_mask_not).to(torch.int16)
        num_dist = torch.sum(kl_loss_mask)

        # Get model outputs
        out = self.model(data)
        a_hat = out["a_hat"]
        mu, logvar = out["mu"], out["logvar"]

        # Set (mu,std) to (0,1) for padded skill outputs
        mu = mu * kl_loss_mask
        std = (logvar / 2).exp() * kl_loss_mask + kl_loss_mask_not
        dist = torch.distributions.Normal(mu, std)
        loss_dict["kldiv_loss"] = self.kl_weights * normal_kl(dist, None).sum() / num_dist

        # Set target and pred actions to 0 for padded sequence outputs
        a_hat = a_hat * action_loss_mask
        loss_dict["act_loss"] = F.mse_loss(a_hat, a_targ, reduction="sum") / num_actions

        # Compute some metrics
        metric_dict["batch_mean_seq_len"] = num_actions / bs / act_dim
        mean_targ_acts = torch.sum(a_targ, (0,1))/torch.sum(action_loss_mask, (0,1))
        mean_pred_acts = torch.sum(a_hat, (0,1))/torch.sum(action_loss_mask, (0,1))
        
        for i in range(act_dim):
            metric_dict[f"batch_mean_targ_acts_{i}"] = mean_targ_acts[i]
            metric_dict[f"batch_mean_pred_acts_{i}"] = mean_pred_acts[i]

        for i in [1,5,10,20,50,100,150]: # TODO GRIPPER FIX
            metric_dict[f"batch_mean_joint_error_til_t{i}"] = F.l1_loss(a_hat[:,:i,:], a_targ[:,:i,:], reduction="sum") / torch.sum(action_loss_mask[:,:i,:])
            # metric_dict[f"batch_mean_joint_error_til_t{i}"] = F.l1_loss(a_hat[:,:i,:-1], a_targ[:,:i,:-1], reduction="sum") / torch.sum(action_loss_mask[:,:i,:-1])
            # metric_dict[f"batch_mean_grip_error_til_t{i}"] = F.l1_loss(a_hat[:,:i,-1], a_targ[:,:i,-1], reduction="sum") / torch.sum(action_loss_mask[:,:i,-1])

        metric_dict["batch_mean_mu"] = torch.sum(mu) / num_dist
        metric_dict["batch_mean_std"] = torch.sum((logvar / 2).exp() * kl_loss_mask) / num_dist
        
        metric_dict["ahat_vector_traj"] = a_hat

        for k,v in self.model.metrics.items():
            metric_dict[k] = v

        return loss_dict, metric_dict