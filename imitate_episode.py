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
from policy.dataset.ms2dataset import get_MS_loaders, convert_observation, rescale_rgbd, get_skill_pad_from_seq_pad
from policy.checkpoints import CheckpointIO
from policy.skill.skill_vae import TSkillCVAE


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
        help="whether to run at each time step of the sequence",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="encode data or not. By default pass in demos.",
    )
    parser.add_argument(
        "--full-seq",
        action="store_true",
        help="whether to run at each time step of the sequence",
    )
    parser.add_argument(
        "--by-skill",
        action="store_true",
        help="whether to fully execute each skill at a time before taking a new obs",
        )
    parser.add_argument(
        "--save-obs",
        action="store_true",
        help="whether to save the image observations seen during a full sequence run",
    )

    return parser.parse_args(args)


def _main(args, proc_id: int = 0, num_procs=1, pbar=None):
    
    cfg_path = os.path.join(args.model_dir, "config.yaml")
    cfg = config.load_config(cfg_path, None)
    
    index_path = os.path.join(args.model_dir, "data_info.pickle")
    with open(index_path, 'rb') as f:
        data_info = pickle.load(f)

    # Dataset
    cfg["data"]["pad_train"] = False
    cfg["data"]["pad_val"] = False
    cfg["data"]["augment"] = False
    cfg["data"]["full_seq"] = False

    # Load only the full episode version of the dataset
    if "train_ep_indices" not in data_info.keys():
        train_idx, val_idx = data_info["train_indices"], data_info["val_indices"]
    else:
        train_idx, val_idx = data_info["train_ep_indices"], data_info["val_ep_indices"]
    train_dataset, val_dataset = get_MS_loaders(cfg, return_datasets=True, 
                                                indices=(train_idx, val_idx))
    
    if not args.train:
        dataset = val_dataset
        print("Using Validation Dataset")
    else:
        dataset = train_dataset
        print("Using Training Dataset")

    # Model
    model: TSkillCVAE = config.get_model(cfg, device="cpu")
    checkpoint_io = CheckpointIO(args.model_dir, model=model)
    load_dict = checkpoint_io.load("model_best.pt")
    model.eval()
    # print(model)

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
    if pbar is not None:
        pbar.set_postfix(
            {
                "control_mode": env_kwargs.get("control_mode"),
                "obs_mode": env_kwargs.get("obs_mode"),
            }
        )

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

    if not args.train:
        idxs = val_idx[:args.count]
    else:
        idxs = train_idx[:args.count]
    
    tb_out_dir = os.path.join("out/PegInsertion/imitation_logs", "tb_logs")
    if os.path.exists(tb_out_dir):
        print(f"Removing existing directory {tb_out_dir}")
        shutil.rmtree(tb_out_dir, ignore_errors=True)
    os.makedirs(tb_out_dir, exist_ok=True)

    # Replay
    for i in range(len(idxs)):
        ind = idxs[i]
        ep = episodes[ind]
        episode_id = ep["episode_id"]
        traj_id = f"traj_{episode_id}"

        data = dataset[i]
        if args.eval:
            data["actions"] = None
        
        with torch.no_grad():
            out = model(data)

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

        for _ in range(args.max_retry + 1):
            env.reset(seed=seed, options=reset_kwargs)
            if ori_env is not None:
                ori_env.reset(seed=seed, options=reset_kwargs)

            if args.vis:
                env.render_human()

            true_actions = ori_h5_file[traj_id]["actions"]
            ori_actions = []
            for k in range(true_actions.shape[0]):
                    ori_actions.append(true_actions[k,:])

            # Run model to reconstruct actions
            if not args.full_seq or args.true:
                a_hat = out["a_hat"].detach().cpu().squeeze()
                a_hat = dataset.action_scaling(a_hat, "inverse").numpy()
                pred_actions = []
                for i in range(a_hat.shape[0]):
                    pred_actions.append(a_hat[i,:])

                info = {}
                ori_env_state = ori_h5_file[traj_id]["env_states"][1]
                env.set_state(ori_env_state)

                # Without conversion between control modes
                n = len(pred_actions)
                if pbar is not None:
                    pbar.reset(total=n)

                if args.true:
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
                if args.eval:
                    out["latent"] = torch.zeros(int(max_episode_steps/model.max_skill_len), 1, model.z_dim)

                info = {}
                img_obs = []
                ori_env_state = ori_h5_file[traj_id]["env_states"][1]
                env.set_state(ori_env_state)
                obs, _, _, _, info = env.step(np.zeros_like(ori_actions[0]))

                if pbar is not None:
                    pbar.reset(total=max_episode_steps)
                for t in range(max_episode_steps):

                    if pbar is not None:
                        pbar.update()
                    
                    if model.single_skill:
                        num_steps_remaining = model.max_skill_len
                    else:
                        num_steps_remaining = max_episode_steps - t

                    seq_pad_mask = torch.zeros(num_steps_remaining)
                    skill_pad_mask = get_skill_pad_from_seq_pad(seq_pad_mask, model.max_skill_len)
                    seq_pad_mask = seq_pad_mask.unsqueeze(0)
                    skill_pad_mask = skill_pad_mask.unsqueeze(0)
                    
                    o = convert_observation(obs, robot_state_only=True, pos_only=True)
                    qpos = torch.from_numpy(o["state"]).float().unsqueeze(0).to(model._device)
                    rgbd = o["rgbd"]
                    rgb = rescale_rgbd(rgbd, discard_depth=True, separate_cams=True)
                    img_obs.append(np.hstack((rgb[...,0],rgb[...,1])) * 255)
                    rgb = torch.from_numpy(rgb).float().unsqueeze(0).permute((0, 4, 3, 1, 2)).unsqueeze(0) # (bs, seq, num_cams, channels, img_h, img_w)                         print(f"==>> rgb: {rgb}")
                    img_src, img_pe = model.stt_encoder(rgb)
                    
                    
                    t_sk = torch.floor(torch.tensor(t) / model.max_skill_len).to(torch.int)
                    if model.single_skill:
                        latent = out["latent"][t_sk:t_sk+model.decode_num,...]
                    else:
                        latent = out["latent"][t_sk:,...]

                    if latent.shape[0] == 0:
                        break
                    
                    if args.by_skill and (t % model.max_skill_len != 0):
                        obs, _, _, _, info = env.step(a_hat[t % model.max_skill_len,:])
                    else:
                        with torch.no_grad():
                            a_hat = model.skill_decode(latent, qpos, (img_src[0,...],img_pe[0,...]), 
                                                    skill_pad_mask, seq_pad_mask, None, None)
                        a_hat = a_hat.detach().cpu().squeeze(1)
                        a_hat = dataset.action_scaling(a_hat, "inverse").numpy()

                        obs, _, _, _, info = env.step(a_hat[0,:])

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
