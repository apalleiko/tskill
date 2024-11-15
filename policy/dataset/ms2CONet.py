# Import required packages
# Unless otherwise stated, copy/pasted from ms2dataset

import os
import os.path
import h5py
import numpy as np
import torch
# import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pylab as plt
import dill as pickle
# sklearn is deprecrated
# if causing issues, check this
import sklearn.preprocessing as skp

from policy.dataset.helpers import load_json, pixelToCoordinate
# Below are commented out mani skill functions that I can't load
# from mani_skill2.utils.io_utils import load_json


def tensor_to_numpy(x):
    # moves all tensors to numpy. This is just for SB3 as SB3 does not optimize for observations stored on the GPU.
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return x


def convert_observation(observation):
    # combines the rgb and depth images
    # image data is not scaled here and is kept as uint16 to save space
    image_obs = observation["image"]
    rgb = image_obs["base_camera"]["rgb"]
    depth = image_obs["base_camera"]["depth"]
    rgb2 = image_obs["hand_camera"]["rgb"]
    depth2 = image_obs["hand_camera"]["depth"]

    # combine the RGB and depth images
    rgbd = np.concatenate([rgb, depth, rgb2, depth2], axis=-1)
    obs = dict(rgbd=rgbd)
    return obs


def rescale_rgbd(rgbd, scale_rgb_only=False, discard_depth=False,
                 separate_cams=False):
    # rescales rgbd data and changes them to floats
    rgb1 = rgbd[..., 0:3] / 255.0
    rgb2 = rgbd[..., 4:7] / 255.0
    depth1 = rgbd[..., 3:4]
    depth2 = rgbd[..., 7:8]
    if not scale_rgb_only:
        depth1 = rgbd[..., 3:4] / (2**10)
        depth2 = rgbd[..., 7:8] / (2**10)
    
    if discard_depth:
        if separate_cams:
            rgbd = np.stack([rgb1, rgb2], axis=-1)
        else:
            rgbd = np.concatenate([rgb1, rgb2], axis=-1)
    else:
        if separate_cams:
            rgbd = np.stack([np.concatenate([rgb1, depth1], axis=-1), 
                             np.concatenate([rgb2, depth2], axis=-1)], axis=-1)
        else:
            rgbd = np.concatenate([rgb1, depth1, rgb2, depth2], axis=-1)

    return rgbd


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def generate_input_vector(cam):
    """
    Takes rgbd data from one camera and returns all points in list of 1x6 data vectors
    """
    cam = np.array(cam)

    if cam.shape != (4, 128, 128):
        raise Exception("Camera Data has improper dimensions")
    
    img = cam[0:3, :, :].transpose((1, 2, 0))
    depth_map = cam[3, :, :]
    h, w = depth_map.shape
    input_vec = []

    for u in range(h):
        for v in range(w):
            # x, y, z = pixelToCoordinate([u, v, depth_map[u, v]])
            z = depth_map[u, v]
            R = img[u, v, 0]
            G = img[u, v, 1]
            B = img[u, v, 2]

            input_vec.append([u, v, z, R, G, B])
    
    return input_vec


class ManiSkillDataset(Dataset):
    def __init__(self, dataset_file: str, indices: list = None) -> None:
        self.dataset_file = dataset_file
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.owned_indices = indices if indices is not None else list(range(len(self.episodes)))
    
    def __len__(self):
        return len(self.owned_indices)
    
    def __getitem__(self, idx):
        eps = self.episodes[self.owned_indices[idx]]
        trajectory = self.data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)
        return trajectory


class ConvONetDataset(ManiSkillDataset):
    def __init__(self, dataset_file: str, indices: list = None, max_seq_len: int = 0) -> None:
        self.dataset_file = dataset_file
        data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.owned_indices = indices if indices is not None else list(range(len(self.episodes)))

        # create datamap
        self.datamap = dict()

        # process input data into 
        for i, idx in enumerate(indices):
            eps = self.episodes[self.owned_indices[i]]
            trajectory = data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)

            # convert the original raw observation with our batch-aware function
            obs = convert_observation(trajectory["obs"])
            
            # we use :-1 to ignore the last obs as terminal observations are included
            rgbd = obs["rgbd"][:-1]
            rgbd = rescale_rgbd(rgbd, separate_cams=True)
            rgbd = rgbd.transpose((0, 4, 3, 2, 1)) # (seq, num_cams, channels, img_h, img_w)
            for seq, seq_data in tqdm(enumerate(rgbd)):
                for cam, cam_data in enumerate(seq_data):

                    # need to change to x,y,d, RGB
                    inputs = torch.tensor(generate_input_vector(cam_data)).float()
                    points = inputs[..., 0:3]
                    occ = torch.where(points[..., 2] > 0, 1, 0).float()
                    
                    # needs points, occ, inputs
                    self.datamap[(idx, seq, cam)] = {"inputs": inputs, 
                                                    "points" : points,
                                                    "occ": occ,
                                                    }
                    
    def __len__(self):
        return len(self.datamap.keys())
    
    def __getitem__(self, idx):
        ep, seq, cam = list(self.datamap.keys())[idx]
        return self.datamap[(ep, seq, cam)]
    

def get_MS_loaders(cfg,  **kwargs) -> None:
        cfg_data = cfg["data"]
        dataset_file: str = cfg_data["dataset"]
        val_split: float = cfg_data.get("val_split", 0.5)
        preshuffle: bool = cfg_data.get("preshuffle", True)
        count = cfg_data.get("max_count", None) # Dataset count limitations

        assert os.path.exists(dataset_file)
        
        json_path = dataset_file.replace(".h5", ".json")
        json_data = load_json(json_path)
        episodes = json_data["episodes"]
        
        # Limit number of loaded episodes if needed
        num_episodes = len(episodes)
        if count is not None:
            assert count <= num_episodes
            num_episodes = count           

        # Try loading exiting train/val split indices
        existing_indices = kwargs.get("indices", None)
        path = os.path.join(cfg["training"]["out_dir"],'train_val_indices.pickle')
        
        if existing_indices is None:
            indices = list(range(num_episodes))
            
            # Shuffle the index list for train/val split
            if preshuffle:
                np.random.shuffle(indices)

            # Train/Val split
            split_idx = int(np.floor(num_episodes*val_split))
            train_idx = indices[:num_episodes-split_idx]
            val_idx = indices[num_episodes-split_idx:]

        print("Training Indices: " + str(train_idx))
        print("Validation Indices: " + str(val_idx))

        # Create datasets
        train_dataset = ConvONetDataset(dataset_file, train_idx)
        val_dataset = ConvONetDataset(dataset_file, val_idx)

        if kwargs.get("return_datasets", False):
            return train_dataset, val_dataset

        # Create loaders
        shuffle = kwargs.get("shuffle", True)
        print(f"Shuffling: {shuffle}")
        train_loader =  DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], 
                                   num_workers=cfg["training"]["n_workers"], 
                                   pin_memory=True, drop_last=False, shuffle=shuffle)
        val_loader =  DataLoader(val_dataset, batch_size=cfg["training"]["batch_size_val"], 
                                 num_workers=cfg["training"]["n_workers_val"], 
                                 pin_memory=True, drop_last=False, shuffle=shuffle)
        
        return train_loader, val_loader

if __name__ == "__main__":
    None
    ### Commands for trajectory replay ###
    # python -m mani_skill2.trajectory.replay_trajectory   --traj-path /home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.h5   --save-traj --target-control-mode pd_joint_delta_pos --obs-mode rgbd --num-procs 10
    # python -m policy/dataset/replay_trajectory.py --traj-path data/demos/v0/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_joint_delta_pos --num-procs 2 --cam-res 480