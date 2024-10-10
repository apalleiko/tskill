"""Imitate episodes for a model given a certain dataset.
"""

import argparse
import multiprocessing as mp
import os
from copy import deepcopy
import shutil
from typing import Union
import pickle

import gymnasium as gym
import h5py
import numpy as np
import sapien.core as sapien
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transforms3d.quaternions import quat2axangle
import matplotlib.pyplot as plt

from mani_skill2.agents.controllers import *
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.trajectory.merge_trajectory import merge_h5
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.utils.visualization.misc import images_to_video

from policy import config
from policy.dataset.dataset_loaders import dataset_loader
from policy.checkpoints import CheckpointIO
from policy.skill.skill_vae import TSkillCVAE
from policy.planning.skill_plan import TSkillPlan


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str, help="target observation mode")
    parser.add_argument(
        "-c", "--target-control-mode", type=str, help="target control mode"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--save-traj", action="store_true", help="whether to save trajectories"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="whether to save videos"
    )
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--max-retry", type=int, default=0)
    parser.add_argument(
        "--discard-timeout",
        action="store_true",
        help="whether to discard timeout episodes",
    )
    parser.add_argument(
        "--allow-failure", action="store_true", help="whether to allow failure episodes"
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "--use-env-states",
        action="store_true",
        help="whether to replay by env states instead of actions",
    )
    parser.add_argument(
        "--bg-name",
        type=str,
        default=None,
        help="background scene to use",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="number of demonstrations to replay before exiting. By default will replay all demonstrations",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="whether to use training dataset or not. By default will use val.",
    )
    parser.add_argument(
        "--true",
        action="store_true",
        help="whether to execute true actions",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="encode data or not. By default pass in demos, otherwise latents will be 0.",
    )
    parser.add_argument(
        "--full-seq",
        action="store_true",
        help="whether to run at each time step of the sequence",
    )
    parser.add_argument(
        "--save-obs",
        action="store_true",
        help="whether to save the image observations seen during a full sequence run",
    )
    parser.add_argument(
        "--cond-dec",
        type=int,
        default=None,
        help="override conditional decoding",
    )
    parser.add_argument(
        "--cond-plan",
        type=int,
        default=None,
        help="override conditional planning",
    )
    parser.add_argument(
        "--replan-rate",
        type=int,
        default=200,
        help="rate at which to replan latent vector for planning model",
    )    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="max episode steps",
    )
    parser.add_argument(
        "--vae",
        action="store_true",
        default=False,
        help="whether to show vae output for planning model",
    )

    return parser.parse_args(args)


def _main(args, proc_id: int = 0, num_procs=1, pbar=None):
    
    cfg_path = os.path.join(args.model_dir, "config.yaml")
    cfg = config.load_config(cfg_path, None)
    method = cfg["method"]
    if method == "plan":
        cfg["vae_cfg"] = config.load_config(os.path.join(cfg["model"]["vae_path"],"config.yaml"))
    
    index_path = os.path.join(args.model_dir, "data_info.pickle")
    with open(index_path, 'rb') as f:
        data_info = pickle.load(f)

    # Dataset
    cfg["data"]["pad"] = True
    cfg["data"]["augment"] = False
    cfg["data"]["dataset"] = "/home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.rgbd.pd_joint_delta_pos_c256.h5"
    cfg["training"]["use_precalc"] = True
    use_precalc = True

    # Load only the full episode version of the dataset
    if "train_ep_indices" not in data_info.keys():
        train_idx, val_idx = data_info["train_indices"], data_info["val_indices"]
    else:
        train_idx, val_idx = data_info["train_ep_indices"], data_info["val_ep_indices"]
    train_dataset, val_dataset = dataset_loader(cfg, return_datasets=True, 
                                                fullseq_override=True)
    
    if not args.train:
        dataset = val_dataset
        idxs = val_idx[:args.count]
        print("Using Validation Dataset")
    else:
        dataset = train_dataset
        idxs = train_idx[:args.count]
        print("Using Training Dataset")

    # Model
    model: TSkillCVAE | TSkillPlan = config.get_model(cfg, device="cpu")
    checkpoint_io = CheckpointIO(args.model_dir, model=model)
    load_dict = checkpoint_io.load("model_best.pt")
    model.to(model._device)
    if method == "plan":
        vae = model.vae
    else:
        vae = model
    model.eval()
    if args.cond_dec is not None:
        if args.cond_dec:
            print("ENABLING CONDITIONAL DECODING")
            vae.conditional_decode = True
        else:
            print("DISABLING CONDITIONAL DECODING")
            vae.conditional_decode = False

    if args.cond_plan is not None:
        if args.cond_plan:
            print("ENABLING CONDITIONAL PLANNING")
            model.conditional_plan = True
        else:
            print("DISABLING CONDITIONAL PLANNING")
            model.conditional_plan = False

    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    traj_path = cfg["data"]["dataset"]
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]
    if args.max_steps is not None:
        max_episode_steps = args.max_steps
    else:
        max_episode_steps = env_info["max_episode_steps"]

    # Create a twin env with the original kwargs
    if args.target_control_mode is not None:
        ori_env = gym.make(env_id, **ori_env_kwargs)
    else:
        ori_env = None

    # Create a main env for replay
    target_obs_mode = args.obs_mode
    target_control_mode = args.target_control_mode
    env_kwargs = ori_env_kwargs.copy()
    if target_obs_mode is not None:
        env_kwargs["obs_mode"] = target_obs_mode
    if target_control_mode is not None:
        env_kwargs["control_mode"] = target_control_mode
    env_kwargs["bg_name"] = args.bg_name
    env_kwargs[
        "render_mode"
    ] = "rgb_array"  # note this only affects the videos saved as RecordEpisode wrapper calls env.render
    env = gym.make(env_id, **env_kwargs)

    # Prepare for recording
    output_dir = args.model_dir
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    suffix = "{}.{}".format(env.obs_mode, env.control_mode)
    new_traj_name = ori_traj_name + "." + suffix
    if num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    env = RecordEpisode(
        env,
        output_dir,
        save_on_reset=False,
        save_trajectory=args.save_traj,
        trajectory_name=new_traj_name,
        save_video=args.save_video,
    )

    if env.save_trajectory:
        output_h5_path = env._h5_file.filename
        assert not os.path.samefile(output_h5_path, traj_path)
    else:
        output_h5_path = None

    episodes = json_data["episodes"]
    n_ep = len(episodes)
    inds = np.arange(n_ep)
    inds = np.array_split(inds, num_procs)[proc_id]

    # Replay
    for i in range(len(idxs)):
        ind = idxs[i]
        ep = episodes[ind]
        episode_id = ep["episode_id"]
        traj_id = f"traj_{episode_id}"

        data = dataset[i]
        if args.eval:
            data["actions"] = None
        
        print("Doing model forward pass...")
        with torch.no_grad():
            out = model(data, use_precalc=use_precalc)

        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")

        if traj_id not in ori_h5_file:
            tqdm.write(f"{traj_id} does not exist in {traj_path}")
            continue

        reset_kwargs = ep["reset_kwargs"].copy()
        if "seed" in reset_kwargs:
            assert reset_kwargs["seed"] == ep["episode_seed"]
        else:
            reset_kwargs["seed"] = ep["episode_seed"]
        seed = reset_kwargs.pop("seed")

        env.reset(seed=seed, options=reset_kwargs)
        if ori_env is not None:
            ori_env.reset(seed=seed, options=reset_kwargs)

        ori_env_state = ori_h5_file[traj_id]["env_states"][1]
        env.set_state(ori_env_state)

        if args.vis:
            env.render_human()

        true_actions = ori_h5_file[traj_id]["actions"]
        ori_actions = []
        for k in range(true_actions.shape[0]):
                ori_actions.append(true_actions[k,:])

        # Run model to reconstruct actions
        if not args.full_seq or args.true or args.vae:
            if args.vae and method == "plan":
                if pbar is not None:
                    pbar.set_postfix(
                        {"mode": "VAE Single", "cond_dec": vae.conditional_decode})
                a_hat = out["vae_out"]["a_hat"].detach().cpu().squeeze()
            else:
                if pbar is not None:
                    pbar.set_postfix(
                        {"mode": "Planned Single", "cond_plan": model.conditional_plan, "cond_dec": vae.conditional_decode})
                a_hat = out["a_hat"].detach().cpu().squeeze()
            
            # Invert scaling on the actions
            a_hat = dataset.action_scaling(a_hat, "inverse").numpy()
            pred_actions = []
            for i in range(a_hat.shape[0]):
                pred_actions.append(a_hat[i,:])

            info = {}

            n = len(pred_actions)
            if pbar is not None:
                pbar.reset(total=n)

            if args.true:
                if pbar is not None:
                    pbar.set_postfix(
                        {"mode": "True"})
                actions = ori_actions
            else:
                actions = pred_actions

            for t, a in enumerate(actions):
                if pbar is not None:
                    pbar.update()
                _, _, _, _, info = env.step(a)

                if args.vis:
                    env.render_human()

        # Run model to predict actions based on current obs
        else:
            info = {}
            img_obs = []

            # Take a zero step to get initial observations
            obs, _, _, _, info = env.step(ori_actions[0])
            pbar.reset(total=max_episode_steps)
            for t in range(max_episode_steps):
                pbar.update()
                current_data = dataset.from_obs(obs)
                if "goal_feat" in data.keys():
                    current_data["goal_feat"] = data["goal_feat"]
                    current_data["goal_pe"] = data["goal_pe"]
                else:
                    current_data["goal"] = data["goal"]

                # Get current skill
                if method == "plan" and not args.vae:
                    pbar.set_postfix(
                        {"mode": "Planned Fullseq", "cond_plan": model.conditional_plan, "cond_dec": vae.conditional_decode})
                
                    a_hat = model.get_action(current_data, t)
                    a_t = dataset.action_scaling(a_hat, "inverse").numpy()[0,:]

                obs, _, _, _, info = env.step(a_t)

                if args.vis:
                    env.render_human()

        success = info.get("success", False)
        if args.discard_timeout:
            timeout = "TimeLimit.truncated" in info
            success = success and (not timeout)

        if success or args.allow_failure:
            env.flush_trajectory()
            env.flush_video()
            if args.save_obs:
                images_to_video(img_obs, output_dir, f"ep_{ind}_obs", 20, 10)
            break
        else:
            # Rollback episode id for failed attempts
            env._episode_id -= 1
            if args.verbose:
                print("info", info)
    else:
        tqdm.write(f"Episode {episode_id} is not replayed successfully.")

    # Cleanup
    env.close()
    if ori_env is not None:
        ori_env.close()
    ori_h5_file.close()

    if pbar is not None:
        pbar.close()

    return output_h5_path


def main(args):
    if args.num_procs > 1:
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, args.num_procs) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        if args.save_traj:
            # A hack to find the path
            output_path = res[0][: -len("0.h5")] + "h5"
            merge_h5(output_path, res)
            for h5_path in res:
                tqdm.write(f"Remove {h5_path}")
                os.remove(h5_path)
                json_path = h5_path.replace(".h5", ".json")
                tqdm.write(f"Remove {json_path}")
                os.remove(json_path)
    else:
        _main(args)


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main(parse_args())
