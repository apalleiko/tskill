import argparse
import sys
import os

sys.path.append('./LIBERO/')
from matplotlib import pyplot as plt

# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from mani_skill2.utils.visualization.misc import images_to_video


benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "libero_90": "LIBERO_90"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--model-dir", type=str)
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", nargs="*", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save-videos", action="store_true")
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
        "--full-seq",
        action="store_true",
        help="whether to run at each time step of the sequence",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="number of demonstrations to replay before exiting. By default will replay all demonstrations",
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
    args = parser.parse_args()
    args.save_dir = f"{args.model_dir}/evals"
    return args


def main():
    args = parse_args()
    # e.g., experiments/LIBERO_SPATIAL/Multitask/BCRNNPolicy_seed100/

    cfg_path = os.path.join(args.model_dir, "config.yaml")
    cfg = config.load_config(cfg_path, None)
    method = cfg["method"]
    if method == "plan":
        cfg["vae_cfg"] = config.load_config(os.path.join(cfg["model"]["vae_path"],"config.yaml"))
    
    index_path = os.path.join(args.model_dir, "data_info.pickle")
    with open(index_path, 'rb') as f:
        data_info_all = pickle.load(f)

    # Dataset
    cfg["data"]["pad"] = True
    cfg["data"]["augment"] = False
    use_precalc = True

    train_dataset, val_dataset = dataset_loader(cfg, return_datasets=True, 
                                                fullseq_override=True)
    
    if not args.train:
        dataset: MultitaskDataset = val_dataset
        print("Using Validation Dataset")
    else:
        dataset: MultitaskDataset = train_dataset
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

    cfg["libero_cfg"] = dict()
    cfg["libero_cfg"]["folder"] = get_libero_path("datasets")
    cfg["libero_cfg"]["bddl_folder"] = get_libero_path("bddl_files")
    cfg["libero_cfg"]["init_states_folder"] = get_libero_path("init_states")

    task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(benchmark_map[args.benchmark])(task_order_index)
    # descriptions = [benchmark.get_task(i).language for i in range(10)]
    # task_embs = get_task_embs(cfg, descriptions)
    # benchmark.set_task_embs(task_embs)

    if len(args.task_id) == 2:
        tasks = list(range(args.task_id[0], args.task_id[1]))
    else:
        tasks = args.task_id
    
    for task_id in tasks:
        eval_stats = dict()

        if isinstance(dataset, MultitaskDataset):
            task = benchmark.get_task(task_id)
            task_dataset: LiberoDataset = [d for d in dataset.sequence_datasets if task.name in d.dataset_file][0]
            data_info = data_info_all["datasets"][task_dataset.dataset_file]
        else:
            task_dataset: LiberoDataset = dataset
            demo_name = task_dataset.dataset_file.split('/')[-1].split('.')[0][:-5]
            task = [t for t in benchmark.tasks if t.name == demo_name][0]
            data_info = data_info_all

        print("RUNNING ON TASK: ",task.name)
        # Have to toggle dataset padding, because is done in collate function for training
        task_dataset.pad = True
        task_dataset.pad2msl = True

        # Load only the full episode version of the dataset
        if "train_ep_indices" not in data_info.keys():
            train_idx, val_idx = data_info["train_indices"], data_info["val_indices"]
        else:
            train_idx, val_idx = data_info["train_ep_indices"], data_info["val_ep_indices"]

        if not args.train:
            idxs = val_idx[:args.count]
        else:
            idxs = train_idx[:args.count]

        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.seed}_on{task_id}.stats",
        )

        video_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.seed}_on{task_id}_videos",
        )

        env_args = {
            "bddl_file_name": os.path.join(
                cfg["libero_cfg"]["bddl_folder"], task.problem_folder, task.bddl_file
            ),
            "camera_heights": 128,
            "camera_widths": 128,
        }

        num_success = 0
        with torch.no_grad():
            for i in range(len(idxs)):
                ind = task_dataset.owned_indices[i]
                data = task_dataset[i]
                true_actions = [dataset.action_scaling(data["actions"][0,j:j+1,:],"inverse")[0].numpy() for j in range(data["actions"].shape[1])]

                env = DemoRenderEnv(**env_args)
                env.reset()
                env.seed(args.seed)

                pbar = tqdm(position=0, leave=None, unit="step", dynamic_ncols=True)

                init_states_path = os.path.join(
                    cfg["libero_cfg"]["init_states_folder"], task.problem_folder, task.init_states_file
                )
                init_states = torch.load(init_states_path)
                # indices = np.arange(env_num) % init_states.shape[0]
                init_states_ = init_states[ind]

                steps = 0
                obs = env.set_init_state(init_states_)

                for _ in range(5):  # simulate the physics without any actions
                    env.step(np.zeros(7))

                # task_emb = benchmark.get_task_emb(args.task_id)

                pbar.set_description(f"Replaying {ind}")
                img_obs = []

                if args.true or not args.full_seq:
                    print("Doing model forward pass...")
                    out = model(data, use_precalc=use_precalc)

                    if args.vae and method == "plan":
                        pbar.set_postfix(
                            {"mode": "VAE Single", "cond_dec": vae.conditional_decode})
                        a_hat = out["vae_out"]["a_hat"].detach().cpu().squeeze()
                    elif method == "plan":
                        pbar.set_postfix(
                            {"mode": "Planned Single", "cond_plan": model.conditional_plan, "cond_dec": vae.conditional_decode})
                        a_hat = out["a_hat"].detach().cpu().squeeze()
                    else:
                        pbar.set_postfix(
                            {"mode": "VAE Single", "cond_dec": vae.conditional_decode})
                        a_hat = out["a_hat"].detach().cpu().squeeze()
                    # Invert scaling on the actions
                    a_hat = dataset.action_scaling(a_hat, "inverse").numpy()
                    pred_actions = [a_hat[i,:] for i in range(a_hat.shape[0])]

                    if args.true:
                        pbar.set_postfix(
                            {"mode": "True"})
                        actions = pred_actions
                    else:
                        actions = pred_actions

                    n = len(actions)
                    pbar.reset(total=n)

                    fig = plt.figure(1, figsize=(10,10))
                    (ax1,ax2),(ax3, ax4),(ax5,ax6),(ax7,ax8),(ax9,ax10),(ax11,ax12) = fig.subplots(6,2)
                    ax7.set_xlim(0,n)
                    ax8.set_xlim(0,n)
                    states = torch.zeros(1,0,vae.state_dim)
                    acts = torch.zeros(0, vae.action_dim)
                    tacts = torch.zeros(0, vae.action_dim)

                    for t, a in enumerate(actions):
                        
                        fig.canvas.flush_events()
                        ax1.clear()
                        ax2.clear()
                        ax3.clear()
                        ax4.clear()
                        ax5.clear()
                        ax6.clear()
                        ax7.clear()
                        ax8.clear()
                        cd = task_dataset.from_obs(obs)
                        ax1.imshow(data["rgb"][0,t,0,...].permute(1,2,0))
                        ax1.set_ylabel("True RGB")
                        ax2.imshow(data["rgb"][0,t,1,...].permute(1,2,0))
                        ax3.imshow(cd["rgb"][0,0,0,...].permute(1,2,0))
                        ax3.set_ylabel("VAE RGB")
                        ax4.imshow(cd["rgb"][0,0,1,...].permute(1,2,0))
                        ax5.imshow(data["rgb"][0,t,0,...].permute(1,2,0), alpha=0.5)
                        ax6.imshow(data["rgb"][0,t,1,...].permute(1,2,0), alpha=0.5)
                        ax5.imshow(cd["rgb"][0,0,0,...].permute(1,2,0), alpha=0.5)
                        ax6.imshow(cd["rgb"][0,0,1,...].permute(1,2,0), alpha=0.5)
                        ax5.set_ylabel("Overlay RGB")
                        states = torch.cat((states, cd["state"]), dim=1)
                        acts = torch.cat((acts, torch.from_numpy(a).unsqueeze(0)), dim=0)
                        tacts = torch.cat((tacts, torch.from_numpy(true_actions[t]).unsqueeze(0)), dim=0)
                        for q in range(model.state_dim):
                            ax7.plot(data["state"][0,:t,q], "r--")
                            ax7.plot(states[0,:t,q], "b:")
                        for q in range(model.state_dim):
                            ax8.plot(data["state"][0,:t,q] - states[0,:t,q])
                        ax7.set_ylabel("True & VAE states")
                        ax8.set_ylabel("State Errors")
                        for d in [0,1,2]:
                            ax9.plot(tacts[:t,d], "r--")
                            ax9.plot(acts[:t,d], "b:")
                        for d in [0,1,2]:
                            ax10.plot(tacts[:t,d] - acts[:t,d])
                        ax9.set_ylabel("True & VAE Locs")
                        ax10.set_ylabel("Action Loc Errors")
                        # ax10.set_ylim(-.2,.2)
                        for d in [3,4,5]:
                            ax11.plot(tacts[:t,d], "r--")
                            ax11.plot(acts[:t,d], "b:")
                        for d in [3,4,5]:
                            ax12.plot(tacts[:t,d] - acts[:t,d])
                        ax11.set_ylabel("True & VAE Rots")
                        ax12.set_ylabel("Action Rot Errors")
                        # ax12.set_ylim()

                        ### FIGURE OUT WHAT IS GOING ON WITH THE FULL SEQUENCE EXECUTION. WITHOUT CONDITIONAL PLANNING IT SHOULD LOOK THE GODDAMN SAME.
                        ### WHAT CAN BE DONE ABOUT THE SPIKES AT THE BEGINNING OF SKILLS? ARTIFACT OF LEARNED START TOKEN?
                        
                        fig.canvas.draw()
                        plt.pause(0.05)

                        pbar.update()
                        obs, reward, done, info = env.step(a)
                        img = env.sim.render(512,512,camera_name="frontview")[::-1,...]
                        img_obs.append(img)
                        
                        if done:
                            print("Success!")
                            break

                else:
                    pbar.reset(total=args.max_steps)
                    if method == "plan" and not args.vae:
                            pbar.set_postfix(
                                {"mode": "Planned Fullseq", "cond_plan": model.conditional_plan, "cond_dec": vae.conditional_decode})
                            
                    fig = plt.figure(1, figsize=(10,10))
                    (ax1,ax2),(ax3, ax4),(ax5,ax6),(ax7,ax8),(ax9,ax10),(ax11,ax12) = fig.subplots(6,2)
                    ax7.set_xlim(0,args.max_steps)
                    ax8.set_xlim(0,args.max_steps)
                    states = torch.zeros(1,0,vae.state_dim)
                    acts = torch.zeros(0, vae.action_dim)
                    tacts = torch.zeros(0, vae.action_dim)

                    while steps < args.max_steps:
                        current_data = task_dataset.from_obs(obs)

                        if model.goal_mode == "image":
                            if "goal_feat" in data.keys():
                                current_data["goal_feat"] = data["goal_feat"]
                                current_data["goal_pe"] = data["goal_pe"]
                            else:
                                current_data["goal"] = data["goal"]
                        elif model.goal_mode == "one-hot":
                            pass
                        else:
                            raise ValueError

                        actions = model.get_action(current_data, steps)
                        actions = dataset.action_scaling(actions,"inverse").numpy()[0,:]
                        obs, reward, done, info = env.step(actions)
                        img = env.sim.render(512,512,camera_name="frontview")[::-1,...]
                        # img = obs["agentview_image"][::-1,...]
                        img_obs.append(img)

                        t=steps
                        fig.canvas.flush_events()
                        ax1.clear()
                        ax2.clear()
                        ax3.clear()
                        ax4.clear()
                        ax5.clear()
                        ax6.clear()
                        ax7.clear()
                        ax8.clear()
                        cd = task_dataset.from_obs(obs)
                        ax1.imshow(data["rgb"][0,t,0,...].permute(1,2,0))
                        ax1.set_ylabel("True RGB")
                        ax2.imshow(data["rgb"][0,t,1,...].permute(1,2,0))
                        ax3.imshow(current_data["rgb"][0,0,0,...].permute(1,2,0))
                        ax3.set_ylabel("VAE RGB")
                        ax4.imshow(current_data["rgb"][0,0,1,...].permute(1,2,0))
                        ax5.imshow(data["rgb"][0,steps,0,...].permute(1,2,0), alpha=0.5)
                        ax6.imshow(data["rgb"][0,t,1,...].permute(1,2,0), alpha=0.5)
                        ax5.imshow(current_data["rgb"][0,0,0,...].permute(1,2,0), alpha=0.5)
                        ax6.imshow(current_data["rgb"][0,0,1,...].permute(1,2,0), alpha=0.5)
                        ax5.set_ylabel("Overlay RGB")
                        states = torch.cat((states, cd["state"]), dim=1)
                        acts = torch.cat((acts, torch.from_numpy(actions).unsqueeze(0)), dim=0)
                        tacts = torch.cat((tacts, torch.from_numpy(true_actions[t]).unsqueeze(0)), dim=0)
                        for q in range(model.state_dim):
                            ax7.plot(data["state"][0,:t,q], "r--")
                            ax7.plot(states[0,:t,q], "b:")
                        for q in range(model.state_dim):
                            ax8.plot(data["state"][0,:t,q] - states[0,:t,q])
                        ax7.set_ylabel("True & VAE states")
                        ax8.set_ylabel("State Errors")
                        for d in [0,1,2]:
                            ax9.plot(tacts[:t,d], "r--")
                            ax9.plot(acts[:t,d], "b:")
                        for d in [0,1,2]:
                            ax10.plot(tacts[:t,d] - acts[:t,d])
                        ax9.set_ylabel("True & VAE Locs")
                        ax10.set_ylabel("Action Loc Errors")
                        ax10.set_ylim(-.2,.2)
                        for d in [3,4,5]:
                            ax11.plot(tacts[:t,d], "r--")
                            ax11.plot(acts[:t,d], "b:")
                        for d in [3,4,5]:
                            ax12.plot(tacts[:t,d] - acts[:t,d])
                        ax11.set_ylabel("True & VAE Rots")
                        ax12.set_ylabel("Action Rot Errors")
                        # ax12.set_ylim()
                        
                        fig.canvas.draw()
                        plt.pause(0.05)

                        steps += 1
                        pbar.update()

                        if done:
                            print("Success!")
                            num_success += 1
                            break

                os.system(f"mkdir -p {args.save_dir}")
                if args.save_videos:
                    images_to_video(img_obs, video_folder, f"ep_{ind}_obs", 20, 10)

            success_rate = num_success / len(idxs)

            eval_stats[task_id] = {"success_rate": success_rate}

            env.close()
            # torch.save(eval_stats, save_folder)
            pbar.close()

        # print(f"Results are saved at {save_folder}")
        print(eval_stats)


if __name__ == "__main__":
    main()
