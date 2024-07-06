"""Replay the trajectory stored in HDF5.
The replayed trajectory can use different observation modes and control modes.
We support translating actions from certain controllers to a limited number of controllers.
The script is only tested for Panda, and may include some Panda-specific hardcode.
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

import mani_skill2.envs
from mani_skill2.agents.base_controller import CombinedController
from mani_skill2.agents.controllers import *
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.trajectory.merge_trajectory import merge_h5
from mani_skill2.utils.common import clip_and_scale_action, inv_scale_action
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import get_entity_by_name
from mani_skill2.utils.wrappers import RecordEpisode

from policy import config
from policy.dataset.ms2dataset import get_MS_loaders


def qpos_to_pd_joint_delta_pos(controller: PDJointPosController, qpos):
    assert type(controller) == PDJointPosController
    assert controller.config.use_delta
    assert controller.config.normalize_action
    delta_qpos = qpos - controller.qpos
    low, high = controller.config.lower, controller.config.upper
    return inv_scale_action(delta_qpos, low, high)


def qpos_to_pd_joint_target_delta_pos(controller: PDJointPosController, qpos):
    assert type(controller) == PDJointPosController
    assert controller.config.use_delta
    assert controller.config.use_target
    assert controller.config.normalize_action
    delta_qpos = qpos - controller._target_qpos
    low, high = controller.config.lower, controller.config.upper
    return inv_scale_action(delta_qpos, low, high)


def qpos_to_pd_joint_vel(controller: PDJointVelController, qpos):
    assert type(controller) == PDJointVelController
    assert controller.config.normalize_action
    delta_qpos = qpos - controller.qpos
    qvel = delta_qpos * controller._control_freq
    low, high = controller.config.lower, controller.config.upper
    return inv_scale_action(qvel, low, high)


def compact_axis_angle_from_quaternion(quat: np.ndarray) -> np.ndarray:
    theta, omega = quat2axangle(quat)
    # - 2 * np.pi to make the angle symmetrical around 0
    if omega > np.pi:
        omega = omega - 2 * np.pi
    return omega * theta


def delta_pose_to_pd_ee_delta(
    controller: Union[PDEEPoseController, PDEEPosController],
    delta_pose: sapien.Pose,
    pos_only=False,
):
    assert isinstance(controller, PDEEPosController)
    assert controller.config.use_delta
    assert controller.config.normalize_action
    low, high = controller._action_space.low, controller._action_space.high
    if pos_only:
        return inv_scale_action(delta_pose.p, low, high)
    delta_pose = np.r_[
        delta_pose.p,
        compact_axis_angle_from_quaternion(delta_pose.q),
    ]
    return inv_scale_action(delta_pose, low, high)


def from_pd_joint_pos_to_ee(
    output_mode,
    ori_actions,
    ori_env: BaseEnv,
    env: BaseEnv,
    render=False,
    pbar=None,
    verbose=False,
):
    n = len(ori_actions)
    if pbar is not None:
        pbar.reset(total=n)

    pos_only = not ("pose" in output_mode)
    target_mode = "target" in output_mode

    ori_controller: CombinedController = ori_env.agent.controller
    controller: CombinedController = env.agent.controller

    # NOTE(jigu): We need to track the end-effector pose in the original env,
    # given target joint positions instead of current joint positions.
    # Thus, we need to compute forward kinematics
    pin_model = ori_controller.articulation.create_pinocchio_model()
    ori_arm_controller: PDJointPosController = ori_controller.controllers["arm"]
    arm_controller: PDEEPoseController = controller.controllers["arm"]
    assert arm_controller.config.frame == "ee"
    ee_link: sapien.Link = arm_controller.ee_link

    info = {}

    for t in range(n):
        if pbar is not None:
            pbar.update()

        ori_action = ori_actions[t]
        ori_action_dict = ori_controller.to_action_dict(ori_action)
        output_action_dict = ori_action_dict.copy()  # do not in-place modify

        # Keep the joint positions with all DoF
        full_qpos = ori_controller.articulation.get_qpos()

        ori_env.step(ori_action)

        # Use target joint positions for arm only
        full_qpos[ori_arm_controller.joint_indices] = ori_arm_controller._target_qpos
        pin_model.compute_forward_kinematics(full_qpos)
        target_ee_pose = pin_model.get_link_pose(arm_controller.ee_link_idx)

        flag = True

        for _ in range(2):
            if target_mode:
                prev_ee_pose_at_base = arm_controller._target_pose
            else:
                base_pose = arm_controller.articulation.pose
                prev_ee_pose_at_base = base_pose.inv() * ee_link.pose

            ee_pose_at_ee = prev_ee_pose_at_base.inv() * target_ee_pose
            arm_action = delta_pose_to_pd_ee_delta(
                arm_controller, ee_pose_at_ee, pos_only=pos_only
            )

            if (np.abs(arm_action[:3])).max() > 1:  # position clipping
                if verbose:
                    tqdm.write(f"Position action is clipped: {arm_action[:3]}")
                arm_action[:3] = np.clip(arm_action[:3], -1, 1)
                flag = False
            if not pos_only:
                if np.linalg.norm(arm_action[3:]) > 1:  # rotation clipping
                    if verbose:
                        tqdm.write(f"Rotation action is clipped: {arm_action[3:]}")
                    arm_action[3:] = arm_action[3:] / np.linalg.norm(arm_action[3:])
                    flag = False

            output_action_dict["arm"] = arm_action
            output_action = controller.from_action_dict(output_action_dict)

            _, _, _, _, info = env.step(output_action)
            if render:
                env.render_human()

            if flag:
                break

    return info


def from_pd_joint_pos(
    output_mode,
    ori_actions,
    ori_env: BaseEnv,
    env: BaseEnv,
    render=False,
    pbar=None,
    verbose=False,
):
    if "ee" in output_mode:
        return from_pd_joint_pos_to_ee(**locals())

    n = len(ori_actions)
    if pbar is not None:
        pbar.reset(total=n)

    ori_controller: CombinedController = ori_env.agent.controller
    controller: CombinedController = env.agent.controller

    info = {}

    for t in range(n):
        if pbar is not None:
            pbar.update()

        ori_action = ori_actions[t]
        ori_action_dict = ori_controller.to_action_dict(ori_action)
        output_action_dict = ori_action_dict.copy()  # do not in-place modify

        ori_env.step(ori_action)
        flag = True

        for _ in range(2):
            if output_mode == "pd_joint_delta_pos":
                arm_action = qpos_to_pd_joint_delta_pos(
                    controller.controllers["arm"], ori_action_dict["arm"]
                )
            elif output_mode == "pd_joint_target_delta_pos":
                arm_action = qpos_to_pd_joint_target_delta_pos(
                    controller.controllers["arm"], ori_action_dict["arm"]
                )
            elif output_mode == "pd_joint_vel":
                arm_action = qpos_to_pd_joint_vel(
                    controller.controllers["arm"], ori_action_dict["arm"]
                )
            else:
                raise NotImplementedError(
                    f"Does not support converting pd_joint_pos to {output_mode}"
                )

            # Assume normalized action
            if np.max(np.abs(arm_action)) > 1 + 1e-3:
                if verbose:
                    tqdm.write(f"Arm action is clipped: {arm_action}")
                flag = False
            arm_action = np.clip(arm_action, -1, 1)
            output_action_dict["arm"] = arm_action

            output_action = controller.from_action_dict(output_action_dict)
            _, _, _, _, info = env.step(output_action)
            if render:
                env.render_human()

            if flag:
                break

    return info


def from_pd_joint_delta_pos(
    output_mode,
    ori_actions,
    ori_env: BaseEnv,
    env: BaseEnv,
    render=False,
    pbar=None,
    verbose=False,
):
    n = len(ori_actions)
    if pbar is not None:
        pbar.reset(total=n)

    ori_controller: CombinedController = ori_env.agent.controller
    controller: CombinedController = env.agent.controller
    ori_arm_controller: PDJointPosController = ori_controller.controllers["arm"]

    assert output_mode == "pd_joint_pos", output_mode
    assert ori_arm_controller.config.normalize_action
    low, high = ori_arm_controller.config.lower, ori_arm_controller.config.upper

    info = {}

    for t in range(n):
        if pbar is not None:
            pbar.update()

        ori_action = ori_actions[t]
        ori_action_dict = ori_controller.to_action_dict(ori_action)
        output_action_dict = ori_action_dict.copy()  # do not in-place modify

        prev_arm_qpos = ori_arm_controller.qpos
        delta_qpos = clip_and_scale_action(ori_action_dict["arm"], low, high)
        arm_action = prev_arm_qpos + delta_qpos

        ori_env.step(ori_action)

        output_action_dict["arm"] = arm_action
        output_action = controller.from_action_dict(output_action_dict)
        _, _, _, _, info = env.step(output_action)

        if render:
            env.render_human()

    return info


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
        help="whether to use training dataset of not. By default will use val.",
    )
    return parser.parse_args(args)


def _main(args, proc_id: int = 0, num_procs=1, pbar=None):
    
    cfg_path = os.path.join(args.model_dir, "config.yaml")
    cfg = config.load_config(cfg_path, None)
    
    index_path = os.path.join(args.model_dir, "train_val_indices.pickle")
    with open(index_path, 'rb') as f:
        train_idx, val_idx = pickle.load(f)

    # Dataset
    cfg["data"]["pad_train"] = False
    cfg["data"]["pad_val"] = False
    cfg["data"]["augment"] = False
    cfg["data"]["action_scaling"] = "normal"
    cfg["data"]["state_scaling"] = 1
    train_dataset, val_dataset = get_MS_loaders(cfg, 
                                                indices=(train_idx, val_idx),
                                                return_datasets=True
                                                )
    if not args.train:
        dataset = val_dataset
        print("Using Validation Dataset")
    else:
        dataset = train_dataset
        print("Using Training Dataset")

    # Model
    model = config.get_model(cfg, device="cpu")
    model.eval()
    print(model)

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

        ori_control_mode = ep["control_mode"]

        for _ in range(args.max_retry + 1):
            env.reset(seed=seed, options=reset_kwargs)
            if ori_env is not None:
                ori_env.reset(seed=seed, options=reset_kwargs)

            if args.vis:
                env.render_human()

            # Run model to get actions to replay
            data = dataset[i]
            with torch.no_grad():
                out = model(data)
            true_actions = ori_h5_file[traj_id]["actions"]
            a_hat = out["a_hat"].detach().cpu().squeeze()
            a_hat = dataset.action_scaling(a_hat, "inverse").numpy()
            ori_actions = []
            for i in range(a_hat.shape[0]):
                ori_actions.append(a_hat[i,:])

            # Log action histograms
            writer = SummaryWriter(tb_out_dir)
            seq,_ = a_hat.shape
            for i in range(seq):
                v_i = a_hat[i,:]
                a_i = true_actions[i,:]
                if torch.nonzero(torch.from_numpy(a_i)).shape[0] > 0:
                    writer.add_histogram(f'ep_{ind}_ahat', v_i, i)
                    writer.add_histogram(f'ep_{ind}_atrue', a_i, i)
            writer.close()

            # Plot image observations
            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # img_idx = 55
            # imgs = data["rgb"]
            # ax1.imshow(np.transpose(imgs[img_idx,0,:,:,:],(1,2,0)))
            # ax2.imshow(np.transpose(imgs[img_idx,1,:,:,:],(1,2,0)))
            # plt.show()
            # input()
            # assert 1==0

            info = {}
            num_unpad_seq = torch.sum(torch.logical_not(data["seq_pad_mask"]).to(torch.int16))
            t0 = len(true_actions) - num_unpad_seq
            ori_env_state = ori_h5_file[traj_id]["env_states"][1+t0]
            env.set_state(ori_env_state)

            # Without conversion between control modes
            if target_control_mode is None:
                n = len(ori_actions)
                if pbar is not None:
                    pbar.reset(total=n)
                for t, a in enumerate(ori_actions):
                    if pbar is not None:
                        pbar.update()
                    _, _, _, _, info = env.step(a)
                    if args.vis:
                        env.render_human()

            # From joint position to others
            elif ori_control_mode == "pd_joint_pos":
                info = from_pd_joint_pos(
                    target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=args.vis,
                    pbar=pbar,
                    verbose=args.verbose,
                )

            # From joint delta position to others
            elif ori_control_mode == "pd_joint_delta_pos":
                info = from_pd_joint_delta_pos(
                    target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=args.vis,
                    pbar=pbar,
                    verbose=args.verbose,
                )

            success = info.get("success", False)
            if args.discard_timeout:
                timeout = "TimeLimit.truncated" in info
                success = success and (not timeout)

            if success or args.allow_failure:
                env.flush_trajectory()
                env.flush_video()
                break
            else:
                # Rollback episode id for failed attempts
                env._episode_id -= 1
                if args.verbose:
                    print("info", info)
        else:
            tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")

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
