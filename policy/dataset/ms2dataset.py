# Import required packages
import os
import os.path as osp
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pylab as plt
import time
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.common import flatten_state_dict
import h5py
import dill as pickle
import sklearn.preprocessing as skp


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
                 max_seq_len: int = 200, max_skill_len: int = 10, 
                 pad: bool=True, augmentation=None,
                 action_scaling=None, state_scaling=None,
                 full_seq: bool = True) -> None:
        self.dataset_file = dataset_file
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.owned_indices = indices
        self.max_seq_len = max_seq_len
        self.max_skill_len = max_skill_len
        self.max_num_skills = int(max_seq_len/max_skill_len)
        self.pad = pad
        self.augmentation = augmentation
        self.full_seq = full_seq

        if action_scaling is None:
            self.action_scaling = lambda x: x
        else:
            self.action_scaling = action_scaling
        
        if state_scaling is None:
            self.state_scaling = lambda x: x
        else:
            self.state_scaling = state_scaling

    def __len__(self):
        return len(self.owned_indices)

    def __getitem__(self, idx):
        if not self.full_seq:
            eps = self.episodes[self.owned_indices[idx]]
            i0 = 0
        else:
            mp = self.owned_indices[idx]
            eps = self.episodes[mp[0]]
            i0 = mp[1]

        trajectory = self.data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)

        # convert the original raw observation with our batch-aware function
        obs = convert_observation(trajectory["obs"], robot_state_only=True)
        
        # we use :-1 to ignore the last obs as terminal observations are included
        # and they don't have actions
        actions = self.action_scaling(torch.from_numpy(trajectory["actions"])[i0:,:].float()) # (seq, act_dim)
        assert torch.all(torch.logical_not(torch.isnan(actions))), "NAN found in actions"
        state = self.state_scaling(torch.from_numpy(obs["state"][:-1])[i0:,:].float()) # (seq, state_dim) 

        rgbd = obs["rgbd"][:-1]
        rgb = rescale_rgbd(rgbd, discard_depth=True, separate_cams=True)
        rgb = torch.from_numpy(rgb).float().permute((0, 4, 3, 1, 2))[i0:,...] # (seq, num_cams, channels, img_h, img_w)

        # Add padding to sequences to match lengths and generate padding masks
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
        else: # If not padding, this is being passed directly to model. 
            seq_pad_mask = torch.zeros(rgb.shape[0]).to(torch.bool)
        
        # Infer skill padding mask from input sequence mask
        skill_pad_mask = get_skill_pad_from_seq_pad(seq_pad_mask, self.max_skill_len)
            
        # Assume no masking TODO
        # seq_mask = torch.zeros(seq_pad_mask.shape[-1], seq_pad_mask.shape[-1]).to(torch.bool)
        # skill_mask = torch.zeros(skill_pad_mask.shape[-1], skill_pad_mask.shape[-1]).to(torch.bool)

        data = dict(rgb=rgb, state=state, 
                    seq_pad_mask=seq_pad_mask, skill_pad_mask=skill_pad_mask,
                    actions=actions)
        
        if self.augmentation is not None:
            data = self.augmentation(data)

        if not self.pad: # Add extra dimension for batch size as model expects this.
            for k,v in data.items():
                data[k] = v.unsqueeze(0)

        return data


def get_skill_pad_from_seq_pad(seq_pad, max_skill_len):
    """Functions for generating a skill padding mask from a given sequence padding mask,
    based on the max skill length from the config. Always adds the padding at the end"""
    max_num_skills = torch.ceil(torch.tensor(seq_pad.shape[0]/max_skill_len)).to(torch.int)
    num_unpad_seq = torch.sum(torch.logical_not(seq_pad).to(torch.int16))
    num_unpad_skills = torch.ceil(num_unpad_seq / max_skill_len) # get skills with relevant outputs TODO handle non divisible skill lengths
    skill_pad_mask = torch.zeros(max_num_skills) # True is unattended
    skill_pad_mask[num_unpad_skills.to(torch.int16):] = 1
    skill_pad_mask = skill_pad_mask.to(torch.bool)
    return skill_pad_mask
    

def get_next_seq_timestep(cfg, data):
    """Function to step to the next timestep for sequence data input of size (bs, seq, ...)
    Also eliminates batched seq that no longer have any non-padded inputs to avoid nans"""
    
    max_skill_len = cfg["model"]["max_skill_len"]
    new_data = dict()

    for k,v in data.items():
        if "skill" not in k:
            data[k] = v[:,1:,...]
        new_data[k] = []
    
    seq_pad_mask = data["seq_pad_mask"]
    bs, seq = seq_pad_mask.shape
    if seq==0:
        return None
    for b in range(bs):
        spmb = seq_pad_mask[b,:]
        if not torch.all(spmb):
            for k,v in data.items():
                if "skill" not in k:
                    new_data[k].append(v[b,...])
                else:
                    skpmb = get_skill_pad_from_seq_pad(spmb, max_skill_len)
                    new_data[k].append(skpmb)
    
    for k,v in new_data.items():
        if len(v) == 0:
            return None
        new_data[k] = torch.stack(v, dim=0)
    
    return new_data


def get_MS_loaders(cfg,  **kwargs) -> None:
        cfg_data = cfg["data"]
        dataset_file: str = cfg_data["dataset"]
        val_split: float = cfg_data.get("val_split", 0.1)
        preshuffle: bool = cfg_data.get("preshuffle", True)
        augment: bool = cfg_data.get("augment", False) # Augmentation
        count: int = cfg_data.get("max_count", 0) # Dataset count limitations
        max_skill_len: int = cfg["model"]["max_skill_len"] # Max skill length
        full_seq: bool = cfg_data.get("full_seq") # Whether to use a mapping for the episodes to start at each timestep

        # Scale actions/states, or compute normalization for the dataset
        action_scaling = cfg_data.get("action_scaling",1)
        state_scaling = cfg_data.get("state_scaling",1)
        gripper_scaling = cfg_data.get("gripper_scaling", True)
        max_seq_len = cfg_data.get("max_seq_len", 0)
        recompute_scaling = True # Whether recomputing action/state scaling is needed
        recompute_fullseq = False # Whether recomputing full sequence data map is needed

        assert osp.exists(dataset_file)
        data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        json_data = load_json(json_path)
        episodes = json_data["episodes"]

        # Try loading existing data config info pickle file
        path = os.path.join(cfg["training"]["out_dir"],'data_info.pickle')
        try:
            with open(path,'rb') as f:
                    data_info = pickle.load(f)
            print("Found existing data info file")
        except FileNotFoundError:
            data_info = dict()

        # Try loading existing train/val split indices
        override_indices = kwargs.get("indices", None)
        if override_indices is not None:
            print("Using override indices")
            train_mapping, val_mapping = override_indices
            if all([i in data_info.keys() for i in ("action_scaling", "state_scaling")]):
                print("Loading action and state scaling from file")
                act_scaling = data_info["action_scaling"]
                stt_scaling = data_info["state_scaling"]
                recompute_scaling = False
        elif all([i in data_info.keys() for i in ("train_indices", "val_indices")]):
            print(f"Loading indices from file: {path}")
            train_mapping = data_info["train_indices"]
            val_mapping = data_info["val_indices"]
            if all([i in data_info.keys() for i in ("action_scaling", "state_scaling")]):
                print("Loading action and state scaling from file")
                act_scaling = data_info["action_scaling"]
                stt_scaling = data_info["state_scaling"]
                recompute_scaling = False
        else:
            print("Updating train & val indices")
            num_episodes = len(episodes)
            indices = list(range(num_episodes))
            
            # Shuffle the index list for train/val split
            if preshuffle:
                np.random.shuffle(indices)
            
            # Limit number of loaded episodes if needed
            if count > 0 and count < num_episodes: 
                num_episodes = count
                indices = indices[:count]

            # Train/Val split
            split_idx = int(np.floor(num_episodes*val_split))
            train_idx = indices[:num_episodes-split_idx]
            val_idx = indices[num_episodes-split_idx:]
            
            # Save index split
            data_info["train_ep_indices"] = train_idx
            data_info["val_ep_indices"] = val_idx
            if full_seq:
                recompute_fullseq = True
            else:
                data_info["train_indices"] = train_idx
                data_info["val_indices"] = val_idx
                train_mapping = train_idx
                val_mapping = val_idx

        # Create scaling functions if needed
        if recompute_scaling:
            print("Recomputing scaling functions...")
            train_acts = []
            train_states = []
            train_lengths = []
            for i in tqdm(range(len(train_idx)), "Collecting all training data info:"):
                idx = train_idx[i]
                eps = episodes[idx]
                trajectory = data[f"traj_{eps['episode_id']}"]
                trajectory = load_h5_data(trajectory)
                obs = convert_observation(trajectory["obs"], robot_state_only=True)
                actions = torch.from_numpy(trajectory["actions"]).float()
                states = torch.from_numpy(obs["state"][:-1]).float()
                train_acts.append(actions)
                train_states.append(states)
                seq_size = actions.shape[0]
                train_lengths.append(seq_size)
                if seq_size > max_seq_len:
                    print(f"Updating max sequence length: {seq_size}")
                    max_seq_len = seq_size
            
            train_acts = torch.vstack(train_acts)
            train_states = torch.vstack(train_states)
            
            if not gripper_scaling:
                print("Computing seperate gripper scaling")
                sep_idx = -1
                joint_acts = train_acts[:,:sep_idx]
                act_scaling = ScalingFunction(action_scaling, joint_acts, sep_idx)
            else:
                act_scaling = ScalingFunction(action_scaling, train_acts)
            
            stt_scaling = ScalingFunction(state_scaling, train_states, None)

            data_info["action_scaling"] = act_scaling
            data_info["state_scaling"] = stt_scaling

            # Compute full sequence mapping if needed
            if recompute_fullseq:
                print("Computing full sequence mapping...")
                val_lengths = []
                for i in tqdm(range(len(val_idx)), "Collecting all val data info:"):
                    idx = val_idx[i]
                    eps = episodes[idx]
                    trajectory = data[f"traj_{eps['episode_id']}"]
                    trajectory = load_h5_data(trajectory)
                    obs = convert_observation(trajectory["obs"], robot_state_only=True)
                    actions = torch.from_numpy(trajectory["actions"]).float()
                    seq_size = actions.shape[0]
                    val_lengths.append(seq_size)

                train_mapping = []        
                for i in range(len(train_idx)):
                    for j in range(train_lengths[i]):
                        train_mapping.append((train_idx[i], j))

                val_mapping = []        
                for i in range(len(val_idx)):
                    for j in range(val_lengths[i]):
                        val_mapping.append((val_idx[i], j))
                
                data_info["train_indices"] = train_mapping
                data_info["val_indices"] = val_mapping

        # Save updated data info file
        if recompute_scaling:
            print("Saving data info file")
            with open(path,'wb') as f:
                pickle.dump(data_info, f)

        # Obtain augmentations
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
        train_dataset = ManiSkillrgbSeqDataset(dataset_file, train_mapping, 
                                               max_seq_len, max_skill_len,
                                               pad_train, train_augmentation,
                                               act_scaling, stt_scaling,
                                               full_seq)
        val_dataset = ManiSkillrgbSeqDataset(dataset_file, val_mapping, 
                                             max_seq_len, max_skill_len,
                                             pad_val, val_augmentation,
                                             act_scaling, stt_scaling,
                                             full_seq)

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


class ScalingFunction:
    def __init__(self, scaling, data, sep_idx=None) -> None:
        if isinstance(scaling, (int, float, list, tuple)):
            print("Computing linear scaling")
            self.scaling_for = lambda x: x * torch.tensor(scaling)
            self.scaling_inv = lambda x: x / torch.tensor(scaling)
        elif scaling=="norm": 
            print("Computing norm scaling")
            act_mu = torch.mean(data, 0)
            act_std = torch.std(data, 0)
            self.scaling_for = lambda x: (x - act_mu) / torch.sqrt(act_std)
            self.scaling_inv = lambda x: x * torch.sqrt(act_std) + act_mu
        elif scaling=="robust_scaler":
            print("Computing robust scaler")
            scaler = skp.RobustScaler().fit(data)
            self.scaling_for = lambda x: torch.from_numpy(scaler.transform(x.numpy()))
            self.scaling_inv = lambda x: torch.from_numpy(scaler.inverse_transform(x.numpy()))
        elif scaling in ("normal","uniform"):
            print(f"Computing {scaling} quantile transform")
            scaler = skp.QuantileTransformer(output_distribution=scaling)
            scaler.fit(data)
            self.scaling_for = lambda x: torch.from_numpy(scaler.transform(x.numpy()))
            self.scaling_inv = lambda x: torch.from_numpy(scaler.inverse_transform(x.numpy()))
        elif scaling=="power":
            print(f"Computing power transform")
            scaler = skp.PowerTransformer()
            scaler.fit(data)
            self.scaling_for = lambda x: torch.from_numpy(scaler.transform(x.numpy()))
            self.scaling_inv = lambda x: torch.from_numpy(scaler.inverse_transform(x.numpy()))
        else:
            raise ValueError(f"Unsupported action scaling given: {scaling}")

        if sep_idx is not None:
            self.scaling_fcn_forward = lambda x: self.seperate_scaling(self.scaling_for, 
                                                                       x, sep_idx)
            self.scaling_fcn_inverse = lambda x: self.seperate_scaling(self.scaling_inv, 
                                                                       x, sep_idx)
        else:
            self.scaling_fcn_forward = lambda x: self.scaling_for(x)
            self.scaling_fcn_inverse = lambda x: self.scaling_inv(x)
    
    def __call__(self, x, mode="forward"):
        if mode=="forward":
            return self.scaling_fcn_forward(x)
        elif mode=="inverse":
            return self.scaling_fcn_inverse(x)
    
    def seperate_scaling(self, fcn, x, idx):
        return torch.hstack((fcn(x[:,:idx]), x[:,idx:]))


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