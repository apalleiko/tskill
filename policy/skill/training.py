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
        aug_cond = not cfg["data"]["augmentation"].get("image_aug",0) or not cfg["data"]["augment"]
        stt_cond = cfg["model"]["state_encoder"].get("backbone_name",None) == "resnet18" # and not cfg["training"]["lr_state_encoder"]
        self.use_precalc = cfg["training"].get("use_precalc",False)
        if self.use_precalc:
            assert aug_cond and stt_cond

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

    def compute_loss(self, data, out=None):
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
        
        kl_loss_mask_not = skill_pad_mask.unsqueeze(-1).repeat(1,1,latent_dim).to(torch.int16)
        kl_loss_mask = torch.logical_not(kl_loss_mask_not)
        num_dist = torch.sum(kl_loss_mask.to(torch.int16))

        # Get model outputs
        if out is None:
            out = self.model(data, use_precalc=self.use_precalc)
        a_hat = out["a_hat"]
        mu, logvar = out["mu"], out["logvar"]

        # Set (mu,std) to (0,1) for padded skill outputs
        mu = mu[kl_loss_mask]
        std = (logvar[kl_loss_mask] / 2).exp()
        dist = torch.distributions.Normal(mu, std)
        loss_dict["kldiv_loss"] = self.kl_weights * normal_kl(dist, None).sum() / num_dist

        # Set target and pred actions to 0 for padded sequence outputs
        a_hat_l = a_hat[action_loss_mask]
        a_targ_l = a_targ[action_loss_mask]
        loss_dict["act_loss"] = F.mse_loss(a_hat_l, a_targ_l, reduction="sum") / num_actions

        # Compute some metrics
        for i in [1,10,50,100]:
            metric_dict[f"batch_mean_joint_error_til_t{i}"] = F.l1_loss(a_hat[:,:i,:-1], a_targ[:,:i,:-1], reduction="sum") / torch.sum(action_loss_mask[:,:i,:-1]).detach()
            metric_dict[f"batch_mean_grip_error_til_t{i}"] = F.l1_loss(a_hat[:,:i,-1], a_targ[:,:i,-1], reduction="sum") / torch.sum(action_loss_mask[:,:i,-1]).detach()

        metric_dict["batch_mean_mu"] = (torch.sum(mu) / num_dist).detach()
        metric_dict["batch_mean_std"] = (torch.sum(std) / num_dist).detach()
        metric_dict["ahat_vector_traj"] = a_hat.detach()

        return loss_dict, metric_dict