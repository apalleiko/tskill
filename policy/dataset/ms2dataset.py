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
from policy.dataset.data_utils import ScalingFunction, DataAugmentation, load_h5_data
from policy.dataset.masking_utils import get_dec_ar_masks, get_enc_causal_masks, get_plan_ar_masks, get_skill_pad_from_seq_pad

def tensor_to_numpy(x):
    # moves all tensors to numpy. This is just for SB3 as SB3 does not optimize for observations stored on the GPU.
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return x


def convert_observation(observation, pos_only=True):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images
    # we provide a simple tool to flatten dictionaries with state data
    if pos_only:
        state = observation["agent"]["qpos"]
    else:
        state = np.hstack(
        [
            observation["agent"]["qpos"],
            observation["agent"]["qvel"],
        ]
    )
            
    obs = dict(state=state)

    # image data is not scaled here and is kept as uint16 to save space
    if "image" in observation.keys():
        image_obs = observation["image"]
        cams = []
        # combine the RGB and depth images
        for c in image_obs.keys():
            cams.append(image_obs[c]["rgb"])
            cams.append(image_obs[c]["depth"])

        rgbd = np.concatenate(cams, axis=-1)
        obs["rgbd"] = rgbd
        
    return obs


def rescale_rgbd(rgbd, scale_rgb_only=False, discard_depth=False,
                 separate_cams=False):
    # rescales rgbd data and changes them to floats
    rgbs = []
    depths = []
    for i in range(int(rgbd.shape[-1]/4)):
        rgbs.append(rgbd[..., 4*i:4*i+3] / 255.0)
        depths.append(rgbd[..., 4*i+3:4*i+4])
    if not scale_rgb_only:
        depths = [d / (2**10) for d in depths]
    
    if discard_depth:
        if separate_cams:
            rgbd = np.stack(rgbs, axis=-1)
        else:
            rgbd = np.concatenate(rgbs, axis=-1)
    else:
        if separate_cams:
            rgbd = np.stack([np.concatenate([rgbs[i], depths[i]], axis=-1) for i in range(len(rgbs))], axis=-1)
        else:
            rgbd = np.concatenate([np.concatenate([rgbs[i], depths[i]], axis=-1) for i in range(len(rgbs))], axis=-1)

    return rgbd

    
class ManiSkillrgbSeqDataset:
    """Class that organizes maniskill demo dataset into distinct rgb sequences
    for each episode"""
    def __init__(self, method: str, dataset_file: str, indices: list,
                 max_seq_len: int = 200, max_skill_len: int = 10, 
                 pad: bool=True, augmentation=None,
                 action_scaling=None, state_scaling=None,
                 full_seq: bool = True, autoregressive_decode = False,
                 encoder_is_causal = False,
                 **kwargs) -> None:
        self.method = method
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
        self.generate_dec_ar_masks = autoregressive_decode
        self.generate_enc_causal_masks = encoder_is_causal
        self.add_batch_dim = kwargs.get("add_batch_dim",False)

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

        data = dict()

        trajectory = self.data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)

        # convert the original raw observation with our batch-aware function
        obs = convert_observation(trajectory["obs"], pos_only=False)
        
        # we use :-1 to ignore the last obs as terminal observations are included
        # and they don't have actions
        actions = self.action_scaling(torch.from_numpy(trajectory["actions"])[i0:,:].float()) # (seq, act_dim)
        state = self.state_scaling(torch.from_numpy(obs["state"][:-1])[i0:,:].float()) # (seq, state_dim)

        if "resnet18" in trajectory["obs"].keys():
            use_precalc = True
            img_feat = torch.from_numpy(trajectory["obs"]["resnet18"]["img_feat"][i0:,...]) # (seq, num_cams, h*w, c)
            img_pe =  torch.from_numpy(trajectory["obs"]["resnet18"]["img_pe"][i0:,...]) # (seq, num_cams, h*w, hidden)
            num_cam = img_feat.shape[1]
            num_feats = img_feat.shape[2]
        else:
            use_precalc = False
            rgbd = obs["rgbd"][:-1]
            rgb = rescale_rgbd(rgbd, discard_depth=True, separate_cams=True)
            rgb = torch.from_numpy(rgb).float().permute((0, 4, 3, 1, 2))[i0:,...] # (seq, num_cams, channels, img_h, img_w)
            num_cam = rgb.shape[1]
            num_feats = 16 # HARDCODED

        if self.method == "plan":
            if use_precalc:
                data["goal_feat"] = img_feat[-1:,...]
                data["goal_pe"] = img_pe[-1:,...]
            else:
                data["goal"] = rgb[-1:,...]

        # Add padding to sequences to match lengths and generate padding masks
        if self.pad:
            num_unpad_seq = actions.shape[0]
            pad = self.max_seq_len - num_unpad_seq
            seq_pad_mask = torch.cat((torch.zeros(actions.shape[0]), torch.ones(pad)), axis=0).to(torch.bool)

            state_pad = torch.zeros([pad] + list(state.shape[1:]))
            state = torch.cat((state, state_pad), axis=0).to(torch.float32)
            
            act_pad = torch.zeros([pad] + list(actions.shape[1:]))
            actions = torch.cat((actions, act_pad), axis=0).to(torch.float32)
            
            if use_precalc:
                img_feat_pad = torch.zeros([pad] + list(img_feat.shape[1:]))
                img_feat = torch.cat((img_feat, img_feat_pad), axis=0).to(torch.float32)
                img_pe_pad = torch.zeros([pad] + list(img_pe.shape[1:]))
                img_pe = torch.cat((img_pe, img_pe_pad), axis=0).to(torch.float32)
            else:
                rgb_pad = torch.zeros([pad] + list(rgb.shape[1:]))
                rgb = torch.cat((rgb, rgb_pad), axis=0).to(torch.float32)
                
        else: # If not padding, this is being passed directly to model.
            seq_pad_mask = torch.zeros(actions.shape[0]).to(torch.bool)
        
        # Infer skill padding mask from input sequence mask
        skill_pad_mask = get_skill_pad_from_seq_pad(seq_pad_mask, self.max_skill_len)

        data.update(dict(state=state, 
                    seq_pad_mask=seq_pad_mask, skill_pad_mask=skill_pad_mask,
                    actions=actions))
        
        # Add precalculated features to data if applicable
        if use_precalc:
            data["img_feat"] = img_feat
            data["img_pe"] = img_pe
        else:
            data["rgb"] = rgb

        # Some augmentation assumes masking
        if self.augmentation is not None:
            data = self.augmentation(data)

        # Generate autoregressive masks for vae decoder if applicable
        if self.generate_dec_ar_masks:
            dec_src_mask, dec_mem_mask, dec_tgt_mask = get_dec_ar_masks(num_feats*num_cam, self.max_skill_len)
            data["dec_src_mask"] = dec_src_mask
            data["dec_mem_mask"] = dec_mem_mask
            data["dec_tgt_mask"] = dec_tgt_mask

        # Create causal masks for vae encoder if applicable
        if self.generate_enc_causal_masks:
            enc_causal_mask, enc_mem_mask, enc_tgt_mask = get_enc_causal_masks(self.max_seq_len, self.max_num_skills, self.max_skill_len)
            # Merge with existing masks if applicable
            if "enc_src_mask" in data.keys():
                data["enc_src_mask"] = data["enc_src_mask"] | enc_causal_mask
            else:
                data["enc_src_mask"] = enc_causal_mask
            data["enc_mem_mask"] = enc_mem_mask
            data["enc_tgt_mask"] = enc_tgt_mask

        # Create autoregressive masks for skill planner
        if self.method == "plan":
            plan_src_mask, plan_mem_mask, plan_tgt_mask = get_plan_ar_masks(num_feats*num_cam, self.max_num_skills)
            data["plan_tgt_mask"] = plan_tgt_mask
            data["plan_src_mask"] = plan_src_mask
            data["plan_mem_mask"] = plan_mem_mask

        # Add extra dimension for batch size if not using dataloader as model expects this.
        if self.add_batch_dim:
            for k,v in data.items():
                data[k] = v.unsqueeze(0)

        return data


def get_MS_loaders(cfg,  **kwargs) -> None:
        method = cfg["method"]
        if method == "plan":
            cfg_model_vae = cfg["vae_cfg"]["model"]
            cfg_vae = cfg["vae_cfg"]
        else:
            cfg_model_vae = cfg["model"]
            cfg_vae = cfg
        cfg_data = cfg["data"]
        dataset_file: str = cfg_data["dataset"]
        val_split: float = cfg_data.get("val_split", 0.1)
        preshuffle: bool = cfg_data.get("preshuffle", True)
        augment: bool = cfg_data.get("augment", False) # Augmentation
        pad: bool = cfg_data.get("pad", True)
        count: int = cfg_data.get("max_count", 0) # Dataset count limitations
        max_skill_len: int = cfg_model_vae["max_skill_len"] # Max skill length
        autoregressive_decode: bool = cfg_model_vae["autoregressive_decode"] # Whether decoding autoregressively
        encoder_is_causal: bool = cfg_model_vae.get("encoder_is_causal",True) # Whether encoder has causal masks applied
        full_seq: bool = cfg_data.get("full_seq") # Whether to use a mapping for the episodes to start at each timestep
        max_seq_len = cfg_data.get("max_seq_len", 0)

        # Scale actions/states, or compute normalization for the dataset
        action_scaling = cfg_vae["data"].get("action_scaling",1)
        state_scaling = cfg_vae["data"].get("state_scaling",1)
        gripper_scaling = cfg_vae["data"].get("gripper_scaling", True)
        recompute_scaling = True # Whether recomputing action/state scaling is needed
        recompute_fullseq = False # Whether recomputing full sequence data map is needed
        save_override = kwargs.get("save_override", False)
        fullseq_override = kwargs.get("fullseq_override", False)
        recalc_override = kwargs.get("recalc_override", False)
        # recalc_override = True

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
        if not recalc_override and override_indices is not None:
            print("Using override indices")
            train_mapping, val_mapping = override_indices
            if all([i in data_info.keys() for i in ("action_scaling", "state_scaling")]):
                print("Loading action and state scaling from file")
                act_scaling = data_info["action_scaling"]
                stt_scaling = data_info["state_scaling"]
                # recompute_scaling = False
        elif not recalc_override and all([i in data_info.keys() for i in ("train_indices", "val_indices")]):
            print(f"Loading indices from file: {path}")
            if fullseq_override:
                print("Overriding full seq config!")
                train_mapping = data_info["train_ep_indices"]
                val_mapping = data_info["val_ep_indices"]
            else:
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
                obs = convert_observation(trajectory["obs"], pos_only=False)
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
            
            # Determine whether to scale gripper actions
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
                    obs = convert_observation(trajectory["obs"], pos_only=False)
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
        if recompute_scaling and not save_override:
            print("Saving data info file")
            with open(path,'wb') as f:
                pickle.dump(data_info, f)

        # Obtain augmentations
        if augment:
            train_augmentation = DataAugmentation(cfg, cfg_model_vae)
        else:
            train_augmentation = None

        # Get extra configs
        return_dataset = kwargs.get("return_datasets", False)
        override_batch_dim = kwargs.get("add_batch_dim",False)
        add_batch_dim = not pad or return_dataset or override_batch_dim
        if add_batch_dim:
            print("Adding batch dimension to returned data!")

        # Create datasets
        train_dataset = ManiSkillrgbSeqDataset(method, dataset_file, train_mapping, 
                                               max_seq_len, max_skill_len,
                                               pad, train_augmentation,
                                               act_scaling, stt_scaling,
                                               full_seq, autoregressive_decode, encoder_is_causal,
                                               add_batch_dim=add_batch_dim)
        val_dataset = ManiSkillrgbSeqDataset(method, dataset_file, val_mapping, 
                                             max_seq_len, max_skill_len,
                                             pad, None,
                                             act_scaling, stt_scaling,
                                             full_seq, autoregressive_decode, encoder_is_causal,
                                             add_batch_dim=add_batch_dim)

        if return_dataset:
            return train_dataset, val_dataset

        # Create loaders
        shuffle = kwargs.get("shuffle", True)
        print(f"Shuffling: {shuffle}")
        train_loader =  DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], 
                                   num_workers=cfg["training"]["n_workers"],
                                   pin_memory=True, drop_last=True, shuffle=shuffle)
        val_loader =  DataLoader(val_dataset, batch_size=cfg["training"]["batch_size_val"], 
                                 num_workers=cfg["training"]["n_workers_val"], 
                                 pin_memory=True, drop_last=True, shuffle=shuffle)
        
        return train_loader, val_loader


### Commands for trajectory replay ###
# python -m mani_skill2.trajectory.replay_trajectory   --traj-path /home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.h5   --save-traj --target-control-mode pd_joint_delta_pos --obs-mode rgbd --num-procs 10
# python -m policy/dataset/replay_trajectory.py --traj-path data/demos/v0/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_joint_delta_pos --num-procs 2 --cam-res 480