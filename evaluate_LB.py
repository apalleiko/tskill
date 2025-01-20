import argparse
import sys
import os

sys.path.append('./LIBERO/')
from matplotlib import pyplot as plt

import json
import numpy as np
import torch
import yaml
from transformers import AutoModel, pipeline, AutoTokenizer, logging
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt

from LIBERO.libero.libero import get_libero_path
from LIBERO.libero.libero.benchmark import get_benchmark
from LIBERO.libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, ControlEnv, DemoRenderEnv
from LIBERO.libero.libero.utils.video_utils import VideoWriter
from LIBERO.libero.lifelong.algos import *
from LIBERO.libero.lifelong.metric import (
    raw_obs_to_tensor_obs)

from LIBERO.libero.lifelong.main import get_task_embs

import time
import dill as pickle

from policy import config
from policy.dataset.dataset_loaders import dataset_loader
from policy.dataset.LIBEROdataset import LiberoDataset
from policy.dataset.multitask_dataset import MultitaskDataset
from policy.checkpoints import CheckpointIO
from policy.skill.skill_vae import TSkillCVAE
from policy.planning.skill_plan import TSkillPlan

benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "libero_90": "LIBERO_90"
}


def evalute(cfg, model, dataset, seed, env_num, batch_size, save_videos=True):
    model.eval()
    if isinstance(seed, int):
        seed = torch.ones(1, dtype=torch.int16) * seed
    elif isinstance(seed, list) and len(seed) == 1:
        seed = torch.randint(0,5000,(seed[0],))

    save_dir = f"{cfg['training']['out_dir']}/training_evals"

    cfg["libero_cfg"] = dict()
    cfg["libero_cfg"]["folder"] = get_libero_path("datasets")
    cfg["libero_cfg"]["bddl_folder"] = get_libero_path("bddl_files")
    cfg["libero_cfg"]["init_states_folder"] = get_libero_path("init_states")

    task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(benchmark_map["libero_90"])(task_order_index)
    # descriptions = [benchmark.get_task(i).language for i in range(10)]
    # task_embs = get_task_embs(cfg, descriptions)
    # benchmark.set_task_embs(task_embs)

    if isinstance(dataset, MultitaskDataset):
        tasks = list(range(90))
    else:
        tasks = [-1]

    eval_stats = dict()
    
    for task_id in tasks:
        eval_stats[task_id] = dict()
        for s in seed:
            s = s.item()

            if task_id == -1:
                demo_name = dataset.dataset_file.split('/')[-1].split('.')[0][:-5]
                task = [t for t in benchmark.tasks if t.name == demo_name][0]
            else:
                task = benchmark.get_task(task_id)
            print("RUNNING ON TASK: ",task.name)

            save_folder = os.path.join(
                save_dir,
                f"{benchmark}_{s}_on{task_id}.stats",
            )

            video_folder = os.path.join(
                save_dir,
                f"{benchmark}_{s}_on{task_id}_videos",
            )

            env_args = {
                "bddl_file_name": os.path.join(
                    cfg["libero_cfg"]["bddl_folder"], task.problem_folder, task.bddl_file
                ),
                "camera_heights": 128,
                "camera_widths": 128,
            }


            init_states_path = os.path.join(
                cfg["libero_cfg"]["init_states_folder"], task.problem_folder, task.init_states_file
            )
            init_states = torch.load(init_states_path)
            indices = np.arange(env_num) % init_states.shape[0]
            print(f"==>> indices: {indices}")
            init_states_ = init_states[indices]

            dones = [False] * env_num
            num_batches = env_num // batch_size

            for i in range(num_batches):
                print("Creating environments...")
                env_i = [OffScreenRenderEnv(**env_args) for i in range(batch_size)]
                obs = []
                for j in range(len(env_i)):
                    idx = i*batch_size + j
                    env = env_i[j]
                    env.reset()
                    env.seed(s)
                    env.set_init_state(init_states_[idx])
                    for _ in range(5):  # simulate the physics without any actions
                        obs_j, _, done_j, _ = env.step(np.zeros(7))
                    obs.append(obs_j)

                steps = 0
                num_success = 0
                max_steps = 250

                pbar = tqdm(position=0, leave=None, unit="step", dynamic_ncols=True)
                pbar.reset(total=max_steps)
                pbar.set_postfix({"Batch": i})

                with torch.no_grad(), VideoWriter(video_folder + f"_{i}", save_videos) as video_writer:
                    while steps < max_steps:
                        data = [dataset.from_obs(o) for o in obs]
                        current_data = dict()
                        for k in data[0].keys():
                            current_data[k] = torch.cat([d[k] for d in data], dim=0)

                        actions = model.get_action(current_data, steps)
                        actions = dataset.action_scaling(actions,"inverse").numpy()
                        obs = []
                        for j in range(len(env_i)):
                            env = env_i[j]
                            obs_j, _, done_j, _ = env.step(actions[j,:])
                            if done_j:
                                idx = i*batch_size + j
                                dones[idx] = True
                            obs.append(obs_j)
                        
                        batch_dones = dones[i*batch_size:(i+1)*batch_size]
                        video_writer.append_vector_obs(
                        obs, batch_dones, camera_name="agentview_image"
                        )
                            
                        steps += 1
                        pbar.update()
                    
                    print(batch_dones)

                if all(dones):
                    break
                
                for env in env_i:
                    env.close()

            for k in range(env_num):
                num_success += int(dones[k])

            success_rate = num_success / env_num
            print(f"Task {task_id}, seed {s} success rate: {success_rate}")
            
            eval_stats[task_id][s] = success_rate

    # os.system(f"mkdir -p {save_dir}")
    # torch.save(eval_stats, save_folder)
    pbar.close()

    return eval_stats


if __name__ == "__main__":

    model_dir = "/home/mrl/Documents/Projects/tskill/out/Plan/009"
    cfg_path = os.path.join(model_dir, "config.yaml")
    cfg = config.load_config(cfg_path, None)
    method = cfg["method"]
    if method == "plan":
        cfg["vae_cfg"] = config.load_config(os.path.join(cfg["model"]["vae_path"],"config.yaml"))

    # Load only the full episode version of the dataset
    train_dataset, val_dataset = dataset_loader(cfg, return_datasets=True, 
                                                save_override=True,
                                                fullseq_override=True,
                                                preshuffle=False,
                                                pad2msl=True)
    # print(len(train_dataset), len(val_dataset))
    
    # Model
    model = config.get_model(cfg, device="cpu")
    checkpoint_io = CheckpointIO(model_dir, model=model)
    load_dict = checkpoint_io.load("model_best.pt")
    print(evalute(cfg, model, train_dataset, [2], 40, 8))
