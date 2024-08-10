"""Gets loss from a dataset in simulation
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
from policy.dataset.ms2dataset import convert_observation, rescale_rgbd
from policy.skill.skill_vae import TSkillCVAE
from policy.planning.skill_plan import TSkillPlan


class SimLoss():
    def __init__(self, cfg, dataset) -> None:
        self.cfg = cfg
        self.dataset = dataset

    def sim_acts(self, model:TSkillPlan):
        """Returns model outputs from simulation in it's environment
        args:
            - model: current model to use
        returns:
            - acts: simulated output actions from the dataset
        """

        method = self.cfg["method"]
        if method == "plan":
            vae: TSkillCVAE = model.vae
        else:
            vae = model

        MSL = model.max_skill_len
        pbar = tqdm(position=0, leave=None, unit="step", dynamic_ncols=True)

        index_path = os.path.join(self.cfg["training"]["out_dir"], "data_info.pickle")
        with open(index_path, 'rb') as f:
            data_info = pickle.load(f)

        use_precalc = self.cfg["training"].get("use_precalc",False)

        # Load only the full episode version of the dataset
        if "train_ep_indices" not in data_info.keys():
            train_idx, val_idx = data_info["train_indices"], data_info["val_indices"]
        else:
            train_idx, val_idx = data_info["train_ep_indices"], data_info["val_ep_indices"]

        # Load HDF5 containing trajectories
        traj_path = self.cfg["data"]["dataset"]
        ori_h5_file = h5py.File(traj_path, "r")
        output_dir = self.cfg["training"]["out_dir"]

        # Load associated json
        json_path = traj_path.replace(".h5", ".json")
        json_data = load_json(json_path)

        env_info = json_data["env_info"]
        env_id = env_info["env_id"]
        ori_env_kwargs = env_info["env_kwargs"]
        max_episode_steps = env_info["max_episode_steps"]

        # Create a main env for replay
        env_kwargs = ori_env_kwargs.copy()
        env_kwargs[
            "render_mode"
        ] = "rgb_array"  # note this only affects the videos saved as RecordEpisode wrapper calls env.render
        env = gym.make(env_id, **env_kwargs)

        # Prepare for recording
        ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
        suffix = "{}.{}".format(env.obs_mode, env.control_mode)
        new_traj_name = ori_traj_name + "." + suffix
        env = RecordEpisode(
            env,
            output_dir,
            save_on_reset=False,
            save_trajectory=False,
            trajectory_name=new_traj_name,
            save_video=False,
        )

        episodes = json_data["episodes"]        
        idxs = val_idx
        by_skill = False
        outputs = dict()

        # Replay
        for i in range(len(idxs)):
            a_hats = torch.zeros(0,1,vae.action_dim, device=model._device) # (0, bs, act_dim)
            data = {k: v.unsqueeze(0) for k,v in self.dataset[i].items()}
            outputs[i] = dict()

            with torch.no_grad():
                out = model(data, use_precalc=use_precalc)

            ind = idxs[i]
            ep = episodes[ind]
            episode_id = ep["episode_id"]
            traj_id = f"traj_{episode_id}"

            if pbar is not None:
                pbar.set_description(f"Replaying {traj_id}")

            reset_kwargs = ep["reset_kwargs"].copy()
            if "seed" in reset_kwargs:
                assert reset_kwargs["seed"] == ep["episode_seed"]
            else:
                reset_kwargs["seed"] = ep["episode_seed"]
            seed = reset_kwargs.pop("seed")

            env.reset(seed=seed, options=reset_kwargs)
            env.render_human()
        
            if method == "plan":
                vae = model.vae
            else:
                vae = model

            info = {}
            img_obs = []
            ori_env_state = ori_h5_file[traj_id]["env_states"][1]
            env.set_state(ori_env_state)
            obs, _, _, _, info = env.step(np.zeros(vae.action_dim))

            if pbar is not None:
                pbar.reset(total=max_episode_steps)
            for t in range(max_episode_steps):
                if pbar is not None:
                    pbar.update()
                
                # Obtain data in the proper form
                o = convert_observation(obs, robot_state_only=True, pos_only=False)
                qpos = torch.from_numpy(o["state"]).float().unsqueeze(0).unsqueeze(0).to(model._device)
                rgbd = o["rgbd"]
                rgb = rescale_rgbd(rgbd, discard_depth=True, separate_cams=True)
                img_obs.append(np.hstack((np.vstack((rgb[...,0],rgb[...,1])),
                                        np.vstack((rgb[...,2],rgb[...,3])))) * 255)
                rgb = torch.from_numpy(rgb).float().unsqueeze(0).permute((0, 4, 3, 1, 2)).unsqueeze(0) # (bs, seq, num_cams, channels, img_h, img_w)
                img_src, img_pe = model.stt_encoder(rgb.to(model._device))
                
                z_tgt0 = torch.zeros(1, 1, model.z_dim, device=model._device)
                dec_skill_pad_mask = torch.zeros(1,1)

                # Get current skill
                if method == "plan":
                    if t == 0:
                        z_tgt = z_tgt0
                        current_data = dict(state=qpos, actions=None, 
                                            img_feat=img_src[0:1,...], img_pe=img_pe[0:1,...],
                                            z_tgt=z_tgt0)
                        
                        # Check for precalc features
                        if "goal_feat" in data.keys():
                            current_data["goal_feat"] = data["goal_feat"]
                            current_data["goal_pe"] = data["goal_pe"]
                        else:
                            current_data["goal"] = data["goal"]
                        
                        # Get current z pred
                        MNS = np.ceil((max_episode_steps-t) / MSL).astype(np.int16)
                        current_data["skill_pad_mask"] = torch.zeros(1,MNS)
                        for s in range(MNS):
                            out = model(current_data, use_precalc=True)
                            z_hat = out["z_hat"].permute(1,0,2)
                            z_tgt = torch.vstack((z_tgt, z_hat[-1:,...]))
                            current_data["z_tgt"] = z_tgt
                        # Keep track of number of time steps since last replanned
                        t_plan = 0
                    else:
                        t_plan += 1
                    t_sk = torch.floor(torch.tensor(t_plan) / MSL).to(torch.int)
                    latent = z_hat[t_sk:t_sk+1,...]
                else: # Preplanned by vae
                    t_plan = t
                    t_sk = torch.floor(torch.tensor(t_plan) / MSL).to(torch.int)
                    latent = out["latent"] if method != "plan" else out["vae_out"]["latent"]
                    latent = latent[t_sk:t_sk+1,...]

                if latent.shape[0] == 0:
                    break
                
                # Decode latent into actions
                if vae.autoregressive_decode:
                    if t_plan % MSL == 0: # Decode new sequence
                        tgt = torch.zeros(1,1,vae.action_dim, device=model._device)
                    # tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
                    tgt_mask = ~(torch.eye(tgt.shape[1]).to(torch.bool))
                    seq_pad_mask = torch.zeros(1,tgt.shape[1])
                    with torch.no_grad():
                        a_hat = vae.skill_decode(latent, qpos, (img_src[0:1,...],img_pe[0:1,...]), 
                                                    dec_skill_pad_mask, seq_pad_mask,
                                                    tgt_mask=tgt_mask, tgt=tgt) # (MSL|<, bs, action_dim)
                    a_hats = torch.vstack((a_hats, a_hat[-1:,...]))
                    tgt = torch.cat((tgt, a_hat[-1:,...].permute(1,0,2)), dim=1) # (1, seq + 1, act_dim)
                    a_hat = a_hat.detach().cpu()[0,-1:,...] # Take most recent action
                    a_hat = self.dataset.action_scaling(a_hat, "inverse").numpy()
                    a_t = a_hat[0,:]
                elif by_skill and (t_plan % MSL != 0):
                    a_t = a_hat[t_plan % MSL,:]
                else:
                    seq_pad_mask = torch.zeros(1,MSL)
                    with torch.no_grad():
                        a_hat = vae.skill_decode(latent, qpos, (img_src[0:1,...],img_pe[0:1,...]), 
                                                dec_skill_pad_mask, seq_pad_mask, None, None)
                    a_hats = torch.vstack((a_hats, a_hat[-1:,...]))
                    a_hat = a_hat.detach().cpu().squeeze(1)
                    a_hat = self.dataset.action_scaling(a_hat, "inverse").numpy()
                    a_t = a_hat[0,:]
                
                obs, _, _, _, info = env.step(a_t)

                env.render_human()

            success = info.get("success", False)
            a_hats = a_hats.permute(1,0,2) # (bs, seq, act_dim)

            outputs[i]["a_hat"] = a_hats
            outputs[i]["success"] = success
            a_targ = data["actions"].cpu() # (bs, seq, act_dim)
            seq_pad_mask = data["seq_pad_mask"].cpu() # (bs, seq)
            bs, seq, act_dim = a_targ.shape

            # Correct mask size and convert to (0,1) where 1 is attended
            action_loss_mask = seq_pad_mask.unsqueeze(-1).repeat(1, 1, act_dim)
            action_loss_mask = torch.logical_not(action_loss_mask)
            num_actions = torch.sum(action_loss_mask.to(torch.int16))

            # Set target and pred actions to 0 for padded sequence outputs
            a_hat_l = a_hats[action_loss_mask].cpu()
            a_targ_l = a_targ[action_loss_mask].cpu()
            outputs[i]["sim_act_loss"] = torch.nn.functional.mse_loss(a_hat_l, a_targ_l, reduction="sum") / num_actions

        # Cleanup
        env.close()
        del env
        ori_h5_file.close()

        if pbar is not None:
            pbar.close()

        output = dict(sim_act_loss=np.mean([v["sim_act_loss"].item() for k,v in outputs.items()]).item(),
                      success_rate=np.mean([v["success"].astype(np.int16).item() for k,v in outputs.items()]).item())
        
        print("sim_act_loss: ", output["sim_act_loss"])

        return output