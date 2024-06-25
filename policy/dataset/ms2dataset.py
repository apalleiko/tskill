# Import required packages
import os
import os.path as osp
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pylab as plt
import time
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.common import flatten_state_dict
import h5py
import pickle


def tensor_to_numpy(x):
    # moves all tensors to numpy. This is just for SB3 as SB3 does not optimize for observations stored on the GPU.
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return x


def convert_observation(observation, robot_state_only, pos_only=True):
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
        if pos_only:
            state = observation["agent"]["qpos"]
        else:
            state = np.hstack(
            [
                flatten_state_dict(observation["agent"]["qpos"]),
                flatten_state_dict(observation["agent"]["qvel"]),
            ]
        )
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
        action = torch.from_numpy(self.actions[idx]).float()
        rgbd = self.obs_rgbd[idx]
        # note that we rescale data on demand as opposed to storing the rescaled data directly
        # so we can save a ton of space at the cost of a little extra compute
        rgbd = rescale_rgbd(rgbd)
        # permute data so that channels are the first dimension as PyTorch expects this
        rgbd = torch.from_numpy(rgbd).float().permute((2, 0, 1))
        state = torch.from_numpy(self.obs_state[idx]).float()
        return dict(rgbd=rgbd, state=state), action
    

class ManiSkillstateDataset(ManiSkillDataset):
    def __init__(self, dataset_file: str, load_count=-1) -> None:
        raise(NotImplementedError)
    
    
class ManiSkillrgbSeqDataset(ManiSkillDataset):
    """Class that organizes maniskill demo dataset into distinct rgb sequences
    for each episode"""
    def __init__(self, dataset_file: str, indices: list, 
                 max_seq_len: int = 0, max_skill_len: int = 10, 
                 pad: bool =True, augmentation=None,
                 action_scaling=1) -> None:
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
        self.max_skill_len = max_skill_len
        self.max_num_skills = int(max_seq_len/max_skill_len)
        self.pad = pad
        self.augmentation = augmentation
        self.action_scaling = action_scaling

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
        actions = self.action_scaling(torch.from_numpy(trajectory["actions"]).float())
        rgbd = obs["rgbd"][:-1]
        rgb = rescale_rgbd(rgbd, discard_depth=True, separate_cams=True)
        
        # permute data so that channels are the first dimension as PyTorch expects this
        # (seq, num_cams, channels, img_h, img_w)
        rgb = torch.from_numpy(rgb).float().permute((0, 4, 3, 1, 2))
        state = torch.from_numpy(obs["state"][:-1]).float()

        # Add padding to sequences to match max possible length and generate padding masks
        if self.pad:
            num_unpad_seq = rgb.shape[0]
            pad = self.max_seq_len - num_unpad_seq
            seq_pad_mask = torch.cat((torch.zeros(rgb.shape[0]), torch.ones(pad)), axis=0).to(torch.bool)

            rgb_pad = torch.zeros([pad] + list(rgb.shape[1:]))
            state_pad = torch.zeros([pad] + list(state.shape[1:]))
            act_pad = torch.zeros([pad] + list(actions.shape[1:]))
            
            rgb = torch.cat((rgb, rgb_pad), axis=0).to(torch.float32)
            actions = torch.cat((actions, act_pad), axis=0).to(torch.float32)
            state = torch.cat((state, state_pad), axis=0).to(torch.float32)

            # TODO Keep this here? Infer skill padding mask from input sequence mask
            num_unpad_skills = torch.ceil(torch.tensor(num_unpad_seq / self.max_skill_len)) # get skills with relevant outputs TODO handle non divisible skill lengths
            skill_pad_mask = torch.zeros(self.max_num_skills) # True is unattended
            skill_pad_mask[num_unpad_skills.to(torch.int16):] = 1
            skill_pad_mask = skill_pad_mask.to(torch.bool)
        else: # If not padding, this is being passed directly to model. 
            seq_pad_mask = torch.zeros(rgb.shape[0]).to(torch.bool)
            num_unpad_skills = torch.ceil(torch.tensor(rgb.shape[0] / self.max_skill_len)).to(torch.int16)
            skill_pad_mask = torch.zeros(num_unpad_skills).to(torch.bool)
            
        # Assume no masking for now
        seq_mask = torch.zeros(seq_pad_mask.shape[-1], seq_pad_mask.shape[-1]).to(torch.bool)
        skill_mask = torch.zeros(skill_pad_mask.shape[-1], skill_pad_mask.shape[-1]).to(torch.bool)

        data = dict(rgb=rgb, state=state, 
                    seq_pad_mask=seq_pad_mask, skill_pad_mask=skill_pad_mask, 
                    seq_mask=seq_mask, skill_mask=skill_mask,
                    actions=actions)
        
        if self.augmentation is not None:
            data = self.augmentation(data)

        if not self.pad: # Add extra dimension for "batch", so model runs properly when testing.
            for k,v in data.items():
                data[k] = v.unsqueeze(0)

        return data
    

def get_MS_loaders(cfg,  **kwargs) -> None:
        cfg_data = cfg["data"]
        dataset_file: str = cfg_data["dataset"]
        val_split: float = cfg_data.get("val_split", 0.1)
        preshuffle: bool = cfg_data.get("preshuffle", True)
        augment: bool = cfg["data"].get("augment", False) # Augmentation
        max_skill_len = cfg["model"]["max_skill_len"] # Max skill length
        count = cfg_data.get("max_count",None) # Dataset count limitations

        assert osp.exists(dataset_file)
        data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        json_data = load_json(json_path)
        episodes = json_data["episodes"]
        
        # Limit number of loaded episodes if needed
        num_episodes = len(episodes)
        if count is not None:
            assert count <= num_episodes
            num_episodes = count

        # If a max sequence length isn't passed need to find it from the data to allow batching
        max_seq_len = cfg_data.get("max_seq_len", 0)
        if not max_seq_len:
            print("Computing max sequence length...")
            for idx in tqdm(range(num_episodes)):
                eps = episodes[idx]
                trajectory = data[f"traj_{eps['episode_id']}"]
                trajectory = load_h5_data(trajectory)
                seq_size = trajectory["obs"]["image"]["base_camera"]["rgb"].shape[0] - 1
                if seq_size > max_seq_len:
                    max_seq_len = seq_size
            print("Max Sequence Length: ", max_seq_len)

        # Scale actions, or compute action normalization for the dataset
        action_scaling = cfg_data.get("action_scaling",1)
        if action_scaling=="norm": 
            all_acts = []
            print("Computing action norm")
            for idx in tqdm(range(num_episodes)):
                eps = episodes[idx]
                trajectory = data[f"traj_{eps['episode_id']}"]
                trajectory = load_h5_data(trajectory)
                actions = torch.from_numpy(trajectory["actions"]).float()
                all_acts.append(actions)
            all_acts = torch.vstack(all_acts)
            act_mu = torch.mean(all_acts, 0)
            act_std = torch.std(all_acts, 0)
            print("Action mean: ", act_mu)
            print("Action std: ", act_std)
            print(torch.nonzero(all_acts).shape[0])
            action_scaling = lambda x: (x - act_mu) / torch.sqrt(act_std)

            # Save action scaling values to pickle file
            path = os.path.join(cfg["training"]["out_dir"],'action_norm_stats.pickle')            
            if os.path.exists(path):
                print("Replacing existing action norm file")
                with open(path,'wb') as f:
                    pickle.dump((act_mu, act_std), f)
            else:
                print("Creating new action norm file")
                with open(path,'xb') as f:
                    pickle.dump((act_mu, act_std), f)
        elif isinstance(action_scaling, (int, float)):
            action_scaling = lambda x: action_scaling*x
        else:
            raise ValueError(f"Unsupported action scaling given: {action_scaling}")

        # Try loading exiting train/val split indices
        existing_indices = kwargs.get("indices", None)
        if existing_indices is None:
            indices = list(range(num_episodes))
            
            # Shuffle the index list for train/val split
            if preshuffle:
                np.random.shuffle(indices)

            # Train/Val split
            split_idx = int(np.floor(num_episodes*val_split))
            train_idx = indices[:num_episodes-split_idx]
            val_idx = indices[num_episodes-split_idx:]
            path = os.path.join(cfg["training"]["out_dir"],'train_val_indices.pickle')
            
            # Save index split to pickle file
            if os.path.exists(path):
                print("Replacing existing train/val index file")
                with open(path,'wb') as f:
                    pickle.dump((train_idx, val_idx), f)
            else:
                print("Creating new train/val index file")
                with open(path,'xb') as f:
                    pickle.dump((train_idx, val_idx), f)
        else:
            train_idx, val_idx = existing_indices

        # Apply augmentations
        if augment:
            train_augmentation = DataAugmentation(cfg)
            if cfg["data"]["augmentation"].get("val_augmentation",False):
                val_augmentation = train_augmentation
            else:
                val_augmentation = None
        else:
            train_augmentation = None
            val_augmentation = None

        # Get padding config
        pad_train = cfg_data.get("pad_train", True)
        pad_val = cfg_data.get("pad_val", True)

        # Create datasets
        train_dataset = ManiSkillrgbSeqDataset(dataset_file, train_idx, 
                                               max_seq_len, max_skill_len,
                                               pad_train, train_augmentation,
                                               action_scaling)
        val_dataset = ManiSkillrgbSeqDataset(dataset_file, val_idx, 
                                             max_seq_len, max_skill_len,
                                             pad_val, val_augmentation,
                                             action_scaling)

        if kwargs.get("return_datasets", False):
            return train_dataset, val_dataset

        # Create loaders
        train_loader =  DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], 
                                   num_workers=cfg["training"]["n_workers"], 
                                   pin_memory=True, drop_last=False, shuffle=True)
        val_loader =  DataLoader(val_dataset, batch_size=cfg["training"]["batch_size_val"], 
                                 num_workers=cfg["training"]["n_workers_val"], 
                                 pin_memory=True, drop_last=False, shuffle=True)
        
        return train_loader, val_loader


class DataAugmentation:
    def __init__(self, cfg) -> None:
        self.max_seq_len = cfg["data"]["max_seq_len"]
        self.max_skill_len = cfg["model"]["max_skill_len"]
        self.masking_rate = cfg["data"]["augmentation"].get("masking_rate", 0)
        self.subsequence_rate = cfg["data"]["augmentation"].get("subsequence_rate", 0)

    def __call__(self, data):
        seq_pad_mask = data["seq_pad_mask"]
        num_unpad_seq = torch.sum(torch.logical_not(seq_pad_mask).to(torch.int16))

        val = torch.rand(1)
        if self.subsequence_rate > val and self.subsequence_rate > 0:
            # Uniformly sample how much of the sequence to use for the batch
            # from 1 to entire (unpadded) seq
            num_seq = torch.randint(1, num_unpad_seq+1, (1,1)).squeeze()

            # Pick a random index to start at in the possible window
            # TODO Start at least max_skill_len away from the end of each demo in the batch?
            # As is, it is biased towards smaller sequences
            buffer = self.max_seq_len - num_unpad_seq
            seq_idx_start = torch.randint(0, buffer+1, (1,1)).squeeze()
            seq_idx_end = seq_idx_start + num_seq
            
            # Reset "new" sequences to the beginning (for positional encodings)
            for k,v in data.items():
                if k not in ("skill_pad_mask", "skill_mask", "seq_pad_mask"):
                    new_seq = torch.zeros_like(v)
                    new_seq[:num_seq,...] = v[seq_idx_start:seq_idx_end,...]
                    data[k] = new_seq

            # Recalculate appropriate masking 
            # (also start from the begining of the seq)
            num_unpad_skills = torch.ceil(torch.clone(num_seq / self.max_skill_len)).to(torch.int16)
            data["skill_pad_mask"][num_unpad_skills:] = True
            data["seq_pad_mask"][num_seq:] = True

        if self.masking_rate > 0:
            raise NotImplementedError # TODO

        return data


if __name__ == "__main__":
    path = "/home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.rgbd.pd_joint_delta_pos.h5"

    dataset = ManiSkillrgbSeqDataset(path, None, pad=False)
    print("Length of Dataset: ",len(dataset))

    # obs, action = dataset[13]    
    # # Plot image observations
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # img_idx = -22
    # imgs = obs["rgb"]
    # ax1.imshow(np.transpose(imgs[img_idx,0,:,:,:],(1,2,0)))
    # ax2.imshow(np.transpose(imgs[img_idx,1,:,:,:],(1,2,0)))
    # plt.show()


    ### Commands for trajectory replay ###
    # python -m mani_skill2.trajectory.replay_trajectory   --traj-path /home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.h5   --save-traj --target-control-mode pd_joint_delta_pos --obs-mode rgbd --num-procs 10
    # python -m policy/dataset/replay_trajectory.py --traj-path data/demos/v0/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_joint_delta_pos --num-procs 2 --cam-res 480