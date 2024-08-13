from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from policy.training import BaseTrainer
from policy.skill.config import get_trainer

from .skill_plan import TSkillPlan


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

        self.model: TSkillPlan = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        # Whether to train VAE
        self.train_vae = cfg["training"]["train_vae"]
        if self.train_vae:
            self.vae_trainer = get_trainer(self.model.vae, None, cfg["vae_cfg"], self.device, None)

        self.z_weight = cfg["loss"]["z_weight"]
        self.act_weight = cfg["loss"]["act_weight"]

        # Precalc featuer availability
        aug_cond = not cfg["data"]["augmentation"].get("image_aug",0) or not cfg["data"]["augment"]
        stt_cond = not cfg["training"]["lr_state_encoder"]
        self.use_precalc = cfg["training"].get("use_precalc",False)
        if self.use_precalc:
            assert aug_cond and stt_cond

        # Get alternative training configs
        self.alt_ratio = cfg["training"].get("fraction_alt",0)
        if self.alt_ratio > 0:
            self.val_alt = cfg["training"].get("val_alt",False)
            self.batch_size_alt = cfg["training"].get("batch_size_alt",1)
            self.alt_batch_num = int(cfg["training"]["batch_size"] / self.batch_size_alt)
        else:
            self.val_alt = False

    def epoch_step(self):
        if self.scheduler is not None and self.epoch_it > 0:
            self.scheduler.step()
        self.epoch_it += 1

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        if self.alt_ratio > 0 and (self.step_it + 1) % int(1 / self.alt_ratio) == 0:
            mb_losses, mb_metrics = [], []
            for n in range(self.alt_batch_num):
                mb_s = self.batch_size_alt*n
                mb_e = mb_s + self.batch_size_alt
                mb_data = {k: v[mb_s:mb_e,...] for k,v in data.items()}
                mb_loss_dict, mb_metric_dict = self.compute_loss(mb_data, alt=True)
                mb_metrics.append(mb_metric_dict)

                mb_loss = 0.0
                _dict = {}
                for k, v in mb_loss_dict.items():
                    mb_loss += v
                    _dict[k] = v.item()
                mb_loss_dict = _dict
                mb_losses.append(mb_loss_dict)
                mb_loss.backward()
    
                del mb_loss
                del mb_data

            loss_dict = {k: np.mean([l[k] for l in mb_losses]) for k in mb_losses[0].keys()}
            metric_dict = {k: torch.mean(torch.stack([m[k] for m in mb_metrics])) if "traj" not in k else mb_metrics[0][k] for k in mb_metrics[0].keys()}
        else:
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
        metric_dict = {k: v.item() for k, v in metric_dict.items() if "vector" not in k}
        return loss_dict, metric_dict

    def compute_loss(self, data, **kwargs):
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
        if alt := kwargs.get("alt",False):
            self.model.conditional_plan = True
        out = self.model(data, use_precalc=self.use_precalc, sep_vae_grad=True)
        if alt:
            self.model.conditional_plan = False

        a_hat = out["a_hat"]
        z_hat = out["z_hat"]
        z_targ = out["vae_out"]["mu"].to(self.device).clone().detach() # (bs, skill_seq, z_dim)

        # Set target and pred actions to 0 for padded sequence outputs
        a_hat_l = a_hat[action_loss_mask]
        a_targ_l = a_targ[action_loss_mask]
        loss_dict["act_plan_loss"] = self.act_weight * F.mse_loss(a_hat_l, a_targ_l, reduction="sum") / num_actions

        # Set target and pred latents to 0 for padded skill outputs
        z_hat_l = z_hat[latent_loss_mask]
        z_targ_l = z_targ[latent_loss_mask]
        loss_dict["z_loss"] = self.z_weight * F.mse_loss(z_hat_l, z_targ_l, reduction="sum") / num_latent

        metric_dict["aplan_vector_traj"] = a_hat.detach()

        # Compute some metrics
        for i in [1,10,50,100]:
            metric_dict[f"batch_mean_plan_joint_error_til_t{i}"] = F.l1_loss(a_hat[:,:i,:-1], a_targ[:,:i,:-1], reduction="sum") / torch.sum(action_loss_mask[:,:i,:-1]).detach()
            metric_dict[f"batch_mean_plan_grip_error_til_t{i}"] = F.l1_loss(a_hat[:,:i,-1], a_targ[:,:i,-1], reduction="sum") / torch.sum(action_loss_mask[:,:i,-1]).detach()

        for k,v in self.model.metrics.items():
            metric_dict[k] = v.detach()

        if self.train_vae:
            vae_loss_dict, vae_metric_dict = self.vae_trainer.raw_loss(out["vae_out"], data)
            for k,v in vae_loss_dict.items():
                loss_dict[f"{k}"] = v
            for k,v in vae_metric_dict.items():
                metric_dict[f"{k}"] = v.detach()

        return loss_dict, metric_dict