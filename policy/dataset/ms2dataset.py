# Import required packages
import os.path as osp
import h5py
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pylab as plt
import time
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.common import flatten_state_dict
import h5py


def tensor_to_numpy(x):
    # moves all tensors to numpy. This is just for SB3 as SB3 does not optimize for observations stored on the GPU.
    if th.is_tensor(x):
        return x.cpu().numpy()
    return x


def convert_observation(observation, robot_state_only):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images

    # image data is not scaled here and is kept as uint16 to save space
    image_obs = observation["image"]
    rgb = image_obs["base_camera"]["rgb"]
    depth = image_obs["base_camera"]["depth"]
    rgb2 = image_obs["hand_camera"]["rgb"]
    depth2 = image_obs["hand_camera"]["depth"]

    # we provide a simple tool to flatten dictionaries with state data
    if robot_state_only:
        state = flatten_state_dict(observation["agent"])
    else:
        state = np.hstack(
            [
                flatten_state_dict(observation["agent"]),
                flatten_state_dict(observation["extra"]),
            ]
        )

    # combine the RGB and depth images
    rgbd = np.concatenate([rgb, depth, rgb2, depth2], axis=-1)
    obs = dict(rgbd=rgbd, state=state)
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


class ManiSkillDataset(Dataset):
    def __init__(self, dataset_file: str, indices: list) -> None:
        self.dataset_file = dataset_file
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.owned_indices = indices
    
    def __len__(self):
        raise(NotImplementedError)
    
    def __getitem__(self, idx):
        raise(NotImplementedError)

class ManiSkillrgbdDataset(ManiSkillDataset):
    def __init__(self, dataset_file: str, load_count=-1) -> None:
        self.dataset_file = dataset_file
        super().__init__(dataset_file)

        self.obs_state = []
        self.obs_rgbd = []
        self.actions = []
        self.total_frames = 0
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)

            # convert the original raw observation with our batch-aware function
            obs = convert_observation(trajectory["obs"])
            # we use :-1 to ignore the last obs as terminal observations are included
            # and they don't have actions
            self.obs_rgbd.append(obs["rgbd"][:-1])
            self.obs_state.append(obs["state"][:-1])
            self.actions.append(trajectory["actions"])
        self.obs_rgbd = np.vstack(self.obs_rgbd)
        self.obs_state = np.vstack(self.obs_state)
        self.actions = np.vstack(self.actions)

    def __len__(self):
        return len(self.obs_rgbd)

    def __getitem__(self, idx):
        action = th.from_numpy(self.actions[idx]).float()
        rgbd = self.obs_rgbd[idx]
        # note that we rescale data on demand as opposed to storing the rescaled data directly
        # so we can save a ton of space at the cost of a little extra compute
        rgbd = rescale_rgbd(rgbd)
        # permute data so that channels are the first dimension as PyTorch expects this
        rgbd = th.from_numpy(rgbd).float().permute((2, 0, 1))
        state = th.from_numpy(self.obs_state[idx]).float()
        return dict(rgbd=rgbd, state=state), action
    

class ManiSkillstateDataset(ManiSkillDataset):
    def __init__(self, dataset_file: str, load_count=-1) -> None:
        raise(NotImplementedError)
    
    
class ManiSkillrgbSeqDataset(ManiSkillDataset):
    """Class that organizes maniskill demo dataset into distinct rgb sequences
    for each episode"""
    def __init__(self, dataset_file: str, indices: list, max_seq_len: int = 0, pad=True) -> None:
        self.dataset_file = dataset_file
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.owned_indices = indices if indices is not None else list(range(len(self.episodes)))
        self.max_seq_len = max_seq_len
        self.pad = pad

    def __len__(self):
        return len(self.owned_indices)

    def __getitem__(self, idx):
        eps = self.episodes[self.owned_indices[idx]]
        trajectory = self.data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)

        # convert the original raw observation with our batch-aware function
        obs = convert_observation(trajectory["obs"], robot_state_only=True)
        
        # we use :-1 to ignore the last obs as terminal observations are included
        # and they don't have actions
        action = th.from_numpy(trajectory["actions"]).float()
        rgbd = obs["rgbd"][:-1]

        rgb = rescale_rgbd(rgbd, discard_depth=True, separate_cams=True)
        
        # permute data so that channels are the first dimension as PyTorch expects this
        # (seq, num_cams, channels, img_w, img_h)
        rgb = th.from_numpy(rgb).float().permute((0, 4, 3, 1, 2))
        state = th.from_numpy(obs["state"][:-1]).float()

        # Add padding to sequences to match max possible length and generate padding masks
        if self.pad:
            pad = self.max_seq_len - rgb.shape[0]
            seq_mask = np.concatenate((np.zeros(rgb.shape[0]), np.ones(pad)), axis=0).astype(np.bool_)

            rgb_pad = np.zeros([pad] + list(rgb.shape[1:]))
            # rgb_mask = np.concatenate((np.zeros_like(rgb), np.ones_like(rgb_pad)), axis=0).astype(np.bool_)

            state_pad = np.zeros([pad] + list(state.shape[1:]))
            # state_mask = np.concatenate((np.zeros_like(state), np.ones_like(state_pad)), axis=0).astype(np.bool_)
            
            act_pad = np.zeros([pad] + list(action.shape[1:]))
            # act_mask = np.concatenate((np.zeros_like(action), np.ones_like(act_pad)), axis=0).astype(np.bool_)
            
            rgb = np.concatenate((rgb, rgb_pad), axis=0)
            action = np.concatenate((action, act_pad), axis=0)
            state = np.concatenate((state, state_pad), axis=0)
        else:
            # rgb_mask = None
            # act_mask = None
            seq_mask = None

        return dict(rgb=rgb, state=state, seq_mask=seq_mask), action
    

def get_MS_loaders(cfg, dataset_file: str, val_split: float = 0.1, preshuffle: bool = True, **kwargs) -> None:
        assert osp.exists(path)
        data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        json_data = load_json(json_path)
        episodes = json_data["episodes"]
        num_episodes = len(episodes)

        # If a max sequence length isn't passed, need to find it from the data
        # This is needed to pads sequences and allow batching
        max_seq_len = kwargs.get("max_seq_len", 0)
        if not max_seq_len:
            for idx in tqdm(range(num_episodes)):
                eps = episodes[idx]
                trajectory = data[f"traj_{eps['episode_id']}"]
                trajectory = load_h5_data(trajectory)
                seq_size = trajectory["obs"]["image"]["base_camera"]["rgb"].shape[0] - 1
                if seq_size > max_seq_len:
                    max_seq_len = seq_size
            print("Max Sequence Length: ",max_seq_len)

        indices = list(range(num_episodes))
        if preshuffle:
            np.random.shuffle(indices)

        split_idx = int(np.floor(num_episodes*val_split))
        train_idx = indices[:num_episodes-split_idx]
        val_idx = indices[num_episodes-split_idx:]

        train_dataset = ManiSkillrgbSeqDataset(dataset_file, train_idx, max_seq_len)
        val_dataset = ManiSkillrgbSeqDataset(dataset_file, val_idx, max_seq_len)

        train_loader =  DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], 
                                   num_workers=cfg["training"]["n_workers"], 
                                   pin_memory=True, drop_last=True, shuffle=True)
        val_loader =  DataLoader(val_dataset, batch_size=cfg["training"]["batch_size_val"], 
                                 num_workers=cfg["training"]["n_workers_val"], 
                                 pin_memory=True, drop_last=True, shuffle=True)
        
        return train_loader, val_loader


if __name__ == "__main__":
    path = "/home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.rgbd.pd_joint_delta_pos.h5"
    cfg = {"training": {"batch_size": 5, "batch_size_val": 5,
                        "n_workers": 0, "n_workers_val": 0}}

    dataset = ManiSkillrgbSeqDataset(path, None, pad=False)
    print("Length of Dataset: ",len(dataset))
    
    train_loader, val_loader = get_MS_loaders(cfg, path, preshuffle=False, max_seq_len=200)
    print("Train Size: ",len(train_loader),"\n","Val Size: ",len(val_loader))

    obs, action = dataset[13]    
    # Plot image observations
    fig, (ax1, ax2) = plt.subplots(1, 2)
    img_idx = -22
    imgs = obs["rgb"]
    ax1.imshow(np.transpose(imgs[img_idx,0,:,:,:],(1,2,0)))
    ax2.imshow(np.transpose(imgs[img_idx,1,:,:,:],(1,2,0)))
    plt.show()
    
    # Sequence load time benchmark
    dl = iter(train_loader)
    t0 = time.time()
    for i in range(1):
        next(dl)
    tf = time.time()
    print("Load Time: ",(tf-t0)/1)

    ### Commands for trajectory replay ###
    # python -m mani_skill2.trajectory.replay_trajectory   --traj-path /home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.h5   --save-traj --target-control-mode pd_joint_delta_pos --obs-mode rgbd --num-procs 10
    # python -m policy/dataset/replay_trajectory.py --traj-path data/demos/v0/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_joint_delta_pos --num-procs 2 --cam-res 480