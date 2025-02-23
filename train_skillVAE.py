import argparse
import datetime
import os
import shutil
import sys
import time

import matplotlib
import numpy as np
import torch
import wandb
import yaml

from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from policy import config
from policy.checkpoints import CheckpointIO
from policy.dataset.dataset_loaders import dataset_loader
from policy.training import BaseTrainer as Trainer
# from policy.simulation_loss import SimLoss

# matplotlib.use("Agg")
torch.backends.cuda.matmul.allow_tf32 = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Turn on the debugging mode",
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Removing old output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="custom config file",
    )
    parser.add_argument(
        "--max_it",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="model_init.pt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--skip_train_stt_encoder",
        action="store_true",
    )
    parser.add_argument(
        "--log_inputs",
        action="store_true",
    )
    parser.add_argument(
        "--method",
        type=str
    )
    args = parser.parse_args()
    return args


def main(args):
    default_vae = "/home/mrl/Documents/Projects/tskill/assets/skill/default.yaml"
    default_plan = "/home/mrl/Documents/Projects/tskill/assets/planning/default.yaml"

    if args.method == "skill":
        default_cfg_path = default_vae
    elif args.method == "plan":
        default_cfg_path = default_plan

    if len(args.config) > 0:
        user_cfg_path = args.config
        assert os.path.exists(user_cfg_path)
    else:
        user_cfg_path = default_cfg_path

    assert all(os.path.exists(elem) for elem in (default_cfg_path, user_cfg_path))

    cfg = config.load_config(user_cfg_path, default_cfg_path)
    if cfg["method"] == "plan":
        cfg["vae_cfg"] = config.load_config(os.path.join(cfg["model"]["vae_path"],"config.yaml"))

    if args.skip_train_stt_encoder:
        cfg["training"]["lr_state_encoder"] = 0

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    print(f"Using device: {device}")

    # Shorthands
    if args.bootstrap:
        assert len(args.out_dir) > 0 and len(args.ckpt_file) > 0
        assert os.path.exists(args.out_dir)
        out_dir = args.out_dir
    elif len(args.out_dir) > 0:
        out_dir = args.out_dir
    else:
        out_dir = cfg["training"]["out_dir"]

    # cfg stuff
    if args.debug:
        cfg["training"]["batch_size"] = 5
        cfg["training"]["batch_size_val"] = 12
        cfg["training"]["batch_size_alt"] = 0
        cfg["training"]["visualize_every"] = 1
        cfg["training"]["print_every"] = 1
        cfg["training"]["backup_every"] = 1000
        cfg["training"]["validate_every"] = 1000
        cfg["training"]["checkpoint_every"] = 1000
        cfg["training"]["max_it"] = 5

    # Shorthands
    lr = cfg["training"].get("lr", 1e-3)
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    print_every = cfg["training"]["print_every"]
    checkpoint_every = cfg["training"]["checkpoint_every"]
    validate_every = cfg["training"]["validate_every"]
    visualize_every = cfg["training"]["visualize_every"]
    backup_every = cfg["training"]["backup_every"]
    max_it = cfg["training"]["max_it"]
    if args.max_it > 0:
        max_it = args.max_it
    model_selection_metric = cfg["training"]["model_selection_metric"]
    if cfg["training"]["model_selection_mode"] == "maximize":
        model_selection_sign = 1
    elif cfg["training"]["model_selection_mode"] == "minimize":
        model_selection_sign = -1
    else:
        raise ValueError("model_selection_mode must be " "either maximize or minimize.")
    
    # Initialize wandb
    if not args.debug:
        project_name = "tskill"
        run_name = "peg_insertion"
        if args.bootstrap:
            project_name += "-bootstrapping"
        wandb.init(
            project=project_name, config=cfg, settings=wandb.Settings(start_method="fork")
        )
        wandb.run.name = "-".join([run_name, wandb.run.name.split("-")[-1]])
        wandb.run.save()
    yaml.dump(cfg, sys.stdout)

    # make output dir
    if os.path.exists(out_dir) and args.fresh_start:
        print(f"Removing existing directory {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    # copy config to output directory
    if not len(args.config) > 0:
        shutil.copyfile(user_cfg_path, os.path.join(out_dir, "config.yaml"))

    # Dataset
    train_loader, val_loader = dataset_loader(cfg)
    train_dataset, val_dataset = train_loader.dataset, val_loader.dataset
    print("Train Size: ",len(train_loader),"\n","Val Size: ",len(val_loader))

    # Model
    model = config.get_model(cfg, device=device)

    # Intialize training
    param_dicts = [{"params": [p for n, p in model.named_parameters() if "stt_encoder" not in n and p.requires_grad]}]
    if cfg["training"]["lr_state_encoder"] > 0:
        param_dicts.append({
            "params": [p for n, p in model.named_parameters() if "stt_encoder" in n and p.requires_grad],
            "lr": cfg["training"]["lr_state_encoder"],
        })

    optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                  weight_decay=weight_decay)
    
    lr_decay = cfg["training"].get("lr_decay",1)
    if lr_decay < 1 and lr_decay!=0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        if we := cfg["training"].get("lr_warmup_epochs",0) > 0:
            scheduler_wu = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.01, total_iters=we)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler_wu, scheduler],[we])
    else:
        scheduler = None
    
    trainer: Trainer = config.get_trainer(model, optimizer, cfg, device=device, scheduler=scheduler)
    checkpoint_io = CheckpointIO(out_dir, model=model)

    # Load VAE for new planning models    
    if cfg["method"] == "plan":
        checkpoint_io2 = CheckpointIO(cfg["model"]["vae_path"], model=model.vae)
        load_dict_vae = checkpoint_io2.load("model_best.pt")
        vae_val_best = load_dict_vae.get("loss_val_best", -model_selection_sign * np.inf)
        print("Current best VAE validation metric: %.8f"
        % (vae_val_best))

    try:
        load_dict = checkpoint_io.load("model.pt")
    except FileExistsError:
        load_dict = dict()

    epoch_it = load_dict.get("epoch_it", 0)
    it = load_dict.get("it", 0)

    metric_val_best = load_dict.get("loss_val_best", -model_selection_sign * np.inf)

    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf
    print("Current best validation metric (%s): %.8f"
          % (model_selection_metric, metric_val_best))

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: %.2fM" % (n_trainable_parameters/1e6,))
    print("Number of total parameters: %.2fM" % (nparameters/1e6,))
    print("output path: ", cfg["training"]["out_dir"])

    tb_out_dir = os.path.join(out_dir, "tb_logs")
    if not os.path.exists(tb_out_dir):
        os.makedirs(tb_out_dir)
    
    # Set t0
    t0 = time.time()
    while True:
        epoch_it += 1
        trainer.epoch_step()
        for batch in train_loader:
            it += 1

            batch_losses, batch_metrics = trainer.train_step(batch)

            # Tensorboard model graph
            # if args.debug:
            #     writer = SummaryWriter(tb_out_dir)
            #     trace_batch = dict()
            #     for k,v in batch.items():
            #         if "skill" not in k:
            #             trace_batch[k] = v[:,:5,...]
            #         else:
            #             trace_batch[k] = v[:,0:1,...]
            #     writer.add_graph(model, batch, use_strict_trace=False)

            metrics = {f"train/{k}": v for k, v in batch_losses.items()}
            metrics.update({f"train/metrics/{k}": v for k, v in batch_metrics.items() if "vector" not in k})
            if "ahat_vector" in batch_metrics.keys():
                a_hat = train_dataset.action_scaling(batch_metrics["ahat_vector"].cpu(),"inverse")
                a_targ = train_dataset.action_scaling(batch_metrics["atarg_vector"].cpu(),"inverse")
                metrics[f"train/metrics/mean_unscaled_plan_joint_error"] = F.l1_loss(a_hat[:,:-1], a_targ[:,:-1], reduction="sum") / (a_hat.shape[0]*(a_hat.shape[1]-1))
                metrics[f"train/metrics/mean_unscaled_plan_grip_error"] = F.l1_loss(a_hat[:,-1], a_targ[:,-1], reduction="sum") / (a_hat.shape[0]*1)
            if scheduler is not None:
                metrics.update({"train/metrics/lr": scheduler.get_last_lr()[0]})
            
            if not args.debug:
                wandb.log(metrics)
            
            # Log tensorboard input histograms
            if epoch_it == 1 and args.log_inputs:
                writer = SummaryWriter(tb_out_dir)
                acts = batch["actions"]
                qpos = batch["state"]
                bs, seq, _ = acts.shape
                for b in range(bs):
                    for i in range(seq):
                        acts_i = acts[b,i,:]
                        qpos_i = qpos[b,i,:]
                        if torch.nonzero(acts_i).shape[0] > 0:
                            writer.add_histogram(f'ep_{it*b + b}_all_acts', acts_i, i)
                            writer.add_histogram(f'ep_{it*b + b}_all_qpos', qpos_i, i)
                        else:
                            continue
                writer.close()

            # Print output
            if it == 1 or (it % print_every) == 0:
                t_elapsed = time.time() - t0
                t_eta = (t_elapsed * (max_it / trainer.step_it)) - t_elapsed
                t_eta = datetime.timedelta(seconds=t_eta)
                t_eta = str(t_eta).split(".")[0]
                t_elapsed = datetime.timedelta(seconds=t_elapsed)
                t_elapsed = str(t_elapsed).split(".")[0]
                print_str = (
                    f"[Epoch {epoch_it:04d}] it={it:04d}, "
                )
                print_str += f"time: {t_elapsed}, "
                print_str += f"eta: {t_eta}, "

                for k, v in batch_losses.items():
                    if v < 0.001:
                        print_str += f"{k}:{10000*v:.2f}e-4, "
                    else:
                        print_str += f"{k}:{v:.4f}, "
                print(print_str)

            # Save checkpoint
            if checkpoint_every > 0 and (it % checkpoint_every) == 0:
                print("Saving checkpoint")
                checkpoint_io.save(
                    "model.pt", epoch_it=epoch_it, it=it, loss_val_best=metric_val_best
                )

            # Backup if necessary
            if backup_every > 0 and (it % backup_every) == 0:
                print("Backup checkpoint")
                checkpoint_io.save(
                    "model_%d.pt" % it,
                    epoch_it=epoch_it,
                    it=it,
                    loss_val_best=metric_val_best,
                )

            # Plot gradient histograms
            if visualize_every > 0 and ((it % visualize_every) == 0 or it == 1):
                acts = batch["actions"]
                writer = SummaryWriter(tb_out_dir)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'{name}', param, epoch_it)
                        writer.add_histogram(f'{name}.grad', param.grad, epoch_it)
                        # if "goal_proj" in name or "image_proj" in name or "state_proj" in name:
                        #     print(name, param.grad)
                for k,v in batch_metrics.items():
                    if "traj" in k:
                        _,seq,_ = v.shape
                        for i in range(seq):
                            v_i = v[0,i,:].detach().cpu()
                            a_i = acts[0,i,:].detach().cpu()
                            if torch.nonzero(a_i).shape[0] > 0:
                                writer.add_histogram(f'it_{it}_{k}', v_i, i)
                                writer.add_histogram(f'it_{it}_atrue_vector_traj', a_i, i)
                                a_it = train_dataset.action_scaling(a_i.unsqueeze(0),"inverse")
                                writer.add_histogram(f'it_{it}_atrue_vector_traj_unscaled', a_it, i)
                                if "ahat" in k or "aplan" in k:
                                    v_it = train_dataset.action_scaling(v_i.unsqueeze(0),"inverse")
                                    writer.add_histogram(f'it_{it}_{k}_unscaled', v_it, i)
                    elif "vector" in k:
                        writer.add_histogram(f'{k}', v, it)
                writer.close()

            # Run validation
            if validate_every > 0 and (it % validate_every) == 0:
                eval_dict, eval_metric_dict = trainer.evaluate(val_loader)
                metric_val = eval_dict[model_selection_metric]
                print(
                    "Validation metric (%s): %.4f"
                    % (model_selection_metric, metric_val)
                )

                metrics = {f"val/{k}": v for k, v in eval_dict.items()}
                metrics.update({f"val/metrics/{k}": v for k, v in eval_metric_dict.items()})
                # metrics.update({"val/sim_act_loss": sim_eval["sim_act_loss"]})
                metrics.update({"it": it})

                if not args.debug:
                    wandb.log(metrics)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    print("New best model (loss %.4f)" % metric_val_best)
                    checkpoint_io.save(
                        "model_best.pt",
                        epoch_it=epoch_it,
                        it=it,
                        loss_val_best=metric_val_best,
                    )

            # Exit if necessary
            if trainer.step_it >= max_it:
                exit(0)


if __name__ == "__main__":
    args = get_args()
    main(args)
