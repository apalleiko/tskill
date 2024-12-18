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

        for i in range(len(idxs)):
            ind = task_dataset.owned_indices[i]
            data = task_dataset[i]

            print("Doing model forward pass...")
            with torch.no_grad():
                out = model(data, use_precalc=use_precalc)

            # from mpl_toolkits.axes_grid1 import make_axes_locatable
            # plt.close()
            # # true_mu = out["vae_out"]["mu"]
            # # true_std = (out["vae_out"]["logvar"] / 2).exp()
            # # pred_mu = out["z_hat"]
            # # mse = torch.square(true_mu[0,:,:]-pred_mu[0,:,:]).numpy()
            # # mehal = torch.divide(torch.square(true_mu[0,:,:]-pred_mu[0,:,:]),true_std[0,...]).numpy()
            # true_mu = out["mu"]
            # true_std = (out["logvar"] / 2).exp()
            # pred_mu = true_mu
            # print(true_mu.shape)
            # fig, axs = plt.subplots(1, 5, sharey=True)
            # plt.axis("off")
            # pcm0 = axs[0].pcolormesh(true_std[0,:,:])
            # divider = make_axes_locatable(axs[0])
            # cax = divider.new_vertical(size = '5%', pad = 0.1, pack_start=True)
            # fig.add_axes(cax)
            # fig.colorbar(pcm0, cax = cax, orientation = 'horizontal')

            # pcm1 = axs[1].pcolormesh(true_mu[0,:,:], vmin=-3,vmax=3)
            # divider = make_axes_locatable(axs[1])
            # cax = divider.new_vertical(size = '5%', pad = 0.1, pack_start=True)
            # fig.add_axes(cax)
            # fig.colorbar(pcm1, cax = cax, orientation = 'horizontal')

            # pcm2 = axs[2].pcolormesh(pred_mu[0,:,:], vmin=-3,vmax=3)
            # divider = make_axes_locatable(axs[2])
            # cax = divider.new_vertical(size = '5%', pad = 0.1, pack_start=True)
            # fig.add_axes(cax)
            # fig.colorbar(pcm2, cax = cax, orientation = 'horizontal')

            # # pcm3 = axs[3].pcolormesh(mse)
            # # divider = make_axes_locatable(axs[3])
            # # cax = divider.new_vertical(size = '5%', pad = 0.1, pack_start=True)
            # # fig.add_axes(cax)
            # # fig.colorbar(pcm3, cax = cax, orientation = 'horizontal')

            # # pcm4 = axs[4].pcolormesh(mehal)
            # # divider = make_axes_locatable(axs[4])
            # # cax = divider.new_vertical(size = '5%', pad = 0.1, pack_start=True)
            # # fig.add_axes(cax)
            # # fig.colorbar(pcm4, cax = cax, orientation = 'horizontal')

            # axs[0].set_ylabel("Skill Number")
            # axs[1].set_yticks([])
            # axs[2].set_yticks([])
            # axs[3].set_yticks([])
            # axs[4].set_yticks([])
            # axs[0].set_title("True Latent std")
            # axs[1].set_title("True Latent mu")
            # axs[2].set_title("Pred Latent z")
            # axs[3].set_title("MSE mu")
            # axs[4].set_title("MHNB mu")
            # axs[0].set_xticks([])
            # axs[1].set_xticks([])
            # axs[2].set_xticks([])
            # axs[3].set_xticks([])
            # axs[4].set_xticks([])
            # # plt.tight_layout()
            # plt.show()
            # continue

            env_num = 1
            # env = SubprocVectorEnv(
            #     [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
            # )
            # env = OffScreenRenderEnv(**env_args)
            # env = OffScreenRenderEnv(**env_args)
            env = DemoRenderEnv(**env_args)
            # env = ControlEnv(**env_args)
            env.reset()
            env.seed(args.seed)

            pbar = tqdm(position=0, leave=None, unit="step", dynamic_ncols=True)

            init_states_path = os.path.join(
                cfg["libero_cfg"]["init_states_folder"], task.problem_folder, task.init_states_file
            )
            init_states = torch.load(init_states_path)
            # indices = np.arange(env_num) % init_states.shape[0]
            init_states_ = init_states[ind]

            dones = [False] * env_num
            steps = 0
            obs = env.set_init_state(init_states_)

            for _ in range(10):  # simulate the physics without any actions
                env.step(np.zeros(7))

            # task_emb = benchmark.get_task_emb(args.task_id)

            pbar.set_description(f"Replaying {ind}")
            img_obs = []

            if args.true or not args.full_seq:
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
                    actions = [dataset.action_scaling(data["actions"][0,j:j+1,:],"inverse")[0].numpy() for j in range(data["actions"].shape[1])]
                else:
                    actions = pred_actions
                
                n = len(actions)
                pbar.reset(total=n)
                for t, a in enumerate(actions):
                    pbar.update()
                    obs, reward, done, info = env.step(a)
                    img = env.sim.render(512,512,camera_name="frontview")[::-1,...]
                    # img = obs["agentview_image"][::-1,...]
                    img_obs.append(img)
                    
                    if done:
                        print("Success!")
                        break

                success_rate = 0
            else:
                num_success = 0

                if args.max_steps is None:
                    args.max_steps = out["a_hat"].shape[1] + 50

                pbar.reset(total=args.max_steps)
                if method == "plan" and not args.vae:
                        pbar.set_postfix(
                            {"mode": "Planned Fullseq", "cond_plan": model.conditional_plan, "cond_dec": vae.conditional_decode})
                        
                while steps < args.max_steps:
                    current_data = task_dataset.from_obs(obs)
                    # current_data["rgb"] = data["rgb"][:,steps:steps+1,...]
                    # current_data["state"] = data["state"][:,steps:steps+1,...]

                    if model.goal_mode == "image":
                        if "goal_feat" in data.keys():
                            current_data["goal_feat"] = data["goal_feat"]
                            current_data["goal_pe"] = data["goal_pe"]
                        else:
                            current_data["goal"] = data["goal"]
                    elif model.goal_mode == "one-hot":
                        current_data["goal"] = data_info["task_vec"].unsqueeze(0)
                    else:
                        raise ValueError

                    actions = model.get_action(current_data, steps)
                    actions = dataset.action_scaling(actions,"inverse").numpy()[0,:]
                    obs, reward, done, info = env.step(actions)
                    img = env.sim.render(512,512,camera_name="frontview")[::-1,...]
                    # img = obs["agentview_image"][::-1,...]
                    img_obs.append(img)

                    steps += 1
                    pbar.update()
                    # check whether succeed
                    # for k in range(env_num):
                    #     dones[k] = dones[k] or done[k]
                    if done:
                        print("Success!")
                        break

                for k in range(env_num):
                    num_success += int(dones[k])

                success_rate = num_success / env_num

            eval_stats = {
                "success_rate": success_rate,
            }
            env.close()
            os.system(f"mkdir -p {args.save_dir}")
            # torch.save(eval_stats, save_folder)
            pbar.close()
            if args.save_videos:
                images_to_video(img_obs, video_folder, f"ep_{ind}_obs", 20, 10)

        # print(f"Results are saved at {save_folder}")
        # print(success_rate)


if __name__ == "__main__":
    main()
