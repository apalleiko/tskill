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

from policy import config
from policy.checkpoints import CheckpointIO
from policy.dataset.ms2dataset import get_MS_loaders
from policy.skill.training import Trainer

matplotlib.use("Agg")
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

    args = parser.parse_args()
    return args


def main(args):
    
    default_cfg_path = "/home/mrl/Documents/Projects/tskill/assets/skill/default.yaml"
    if len(args.config) > 0:
        user_cfg_path = args.config
        assert os.path.exists(user_cfg_path)
    else:
        user_cfg_path = default_cfg_path # TODO

    assert all(os.path.exists(elem) for elem in (default_cfg_path, user_cfg_path))

    cfg = config.load_config(user_cfg_path, default_cfg_path)
    if args.skip_train_stt_encoder:
        cfg["training"]["lr_state_encoder"] = 0

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    # Shorthands
    if args.bootstrap:
        assert len(args.out_dir) > 0 and len(args.ckpt_file) > 0
        assert os.path.exists(args.out_dir)
        out_dir = args.out_dir
    elif len(args.out_dir) > 0:
        out_dir = args.out_dir
    else:
        out_dir = cfg["training"]["out_dir"]
    print(f"output dir: {out_dir}")

    if args.debug:
        cfg["training"]["batch_size"] = 2
        cfg["training"]["visualize_every"] = 1
        cfg["training"]["print_every"] = 1
        cfg["training"]["backup_every"] = 1
        cfg["training"]["validate_every"] = 1
        cfg["training"]["checkpoint_every"] = 1
        cfg["training"]["visualize_total"] = 1
        cfg["training"]["max_it"] = 1

    lr = cfg["training"].get("lr", 1e-3)
    weight_decay = cfg["training"].get("weight_decay", 1e-4)

    project_name = "tskill"
    run_name = "peg_test"
    if args.bootstrap:
        project_name += "-bootstrapping"
    wandb.init(
        project=project_name, config=cfg, settings=wandb.Settings(start_method="fork")
    )
    wandb.run.name = "-".join([run_name, wandb.run.name.split("-")[-1]])
    wandb.run.save()
    yaml.dump(cfg, sys.stdout)

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

    if os.path.exists(out_dir) and args.fresh_start:
        print(f"Removing existing directory {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    # copy config to output directory
    shutil.copyfile(user_cfg_path, os.path.join(out_dir, "config.yaml"))

    # Dataset
    train_loader, val_loader = get_MS_loaders(cfg)
    print("Train Size: ",len(train_loader),"\n","Val Size: ",len(val_loader))

    # Model
    model = config.get_model(cfg, device=device)
    print(model)

    # Intialize training #TODO replace backbone
    param_dicts = [{"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]}]
    if cfg["training"]["lr_state_encoder"] > 0:
        param_dicts.append({
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg["training"]["lr_state_encoder"],
        })

    optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                  weight_decay=weight_decay)    
    
    trainer: Trainer = config.get_trainer(model, optimizer, cfg, device=device)
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)

    if args.bootstrap:
        checkpoint_io.load_model_only(args.ckpt_file)
        load_dict = dict()
    else:
        try:
            load_dict = checkpoint_io.load("model.pt")
        except FileExistsError:
            load_dict = dict()

    epoch_it = load_dict.get("epoch_it", 0)
    it = load_dict.get("it", 0)

    metric_val_best = load_dict.get("loss_val_best", -model_selection_sign * np.inf)

    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf
    print(
        "Current best validation metric (%s): %.8f"
        % (model_selection_metric, metric_val_best)
    )

    # Shorthands
    print_every = cfg["training"]["print_every"]
    checkpoint_every = cfg["training"]["checkpoint_every"]
    validate_every = cfg["training"]["validate_every"]

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: %.2fM" % (n_trainable_parameters/1e6,))
    print("Number of total parameters: %.2fM" % (nparameters/1e6,))
    print("output path: ", cfg["training"]["out_dir"])

    # Set t0
    t0 = time.time()
    while True:
        epoch_it += 1
        trainer.epoch_step()

        for batch in train_loader:
            it += 1
            
            losses, met = trainer.train_step(batch)

            metrics = {f"train/{k}": v for k, v in losses.items()}
            metrics.update({f"train/metrics/{k}": v for k, v in met.items()})
            wandb.log(metrics)

            # Print output
            if (it % print_every) == 0:
                t_elapsed = time.time() - t0
                t_eta = (t_elapsed * (max_it / trainer.step_it)) - t_elapsed
                t_eta = datetime.timedelta(seconds=t_eta)
                t_eta = str(t_eta).split(".")[0]

                print_str = (
                    f"[Epoch {epoch_it:04d}] it={it:04d}, time: {t_elapsed:.3f}, "
                )
                print_str += f"eta: {t_eta}, "

                for k, v in losses.items():
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

            if args.debug:
                exit(0)

            # Exit if necessary
        if trainer.step_it >= max_it:
            exit(0)


if __name__ == "__main__":
    args = get_args()
    main(args)
