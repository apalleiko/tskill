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
from torchvision.transforms import v2


def tensor_to_numpy(x):
    # moves all tensors to numpy. This is just for SB3 as SB3 does not optimize for observations stored on the GPU.
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return x


def convert_observation(observation, robot_state_only, pos_only=True):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images
    # we provide a simple tool to flatten dictionaries with state data
    if robot_state_only:
        if pos_only:
            state = observation["agent"]["qpos"]
        else:
            state = np.hstack(
            [
                observation["agent"]["qpos"],
                observation["agent"]["qvel"],
            ]
        )
    else:
        state = np.hstack(
            [
                flatten_state_dict(observation["agent"]),
                flatten_state_dict(observation["extra"]),
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
    
    
class ManiSkillrgbSeqDataset(ManiSkillDataset):
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
        obs = convert_observation(trajectory["obs"], robot_state_only=True, pos_only=False)
        
        # we use :-1 to ignore the last obs as terminal observations are included
        # and they don't have actions
        actions = self.action_scaling(torch.from_numpy(trajectory["actions"])[i0:,:].float()) # (seq, act_dim)
        assert torch.all(torch.logical_not(torch.isnan(actions))), "NAN found in actions"
        state = self.state_scaling(torch.from_numpy(obs["state"][:-1])[i0:,:].float()) # (seq, state_dim) 

        if "resnet18" in trajectory["obs"].keys():
            use_precalc = True
            img_feat = torch.from_numpy(trajectory["obs"]["resnet18"]["img_feat"][i0:,...]) # (seq, num_cams, h*w, c)
            img_pe =  torch.from_numpy(trajectory["obs"]["resnet18"]["img_pe"][i0:,...]) # (seq, num_cams, h*w, hidden)
            num_cam = img_feat.shape[1]
        else:
            use_precalc = False
            rgbd = obs["rgbd"][:-1]
            rgb = rescale_rgbd(rgbd, discard_depth=True, separate_cams=True)
            rgb = torch.from_numpy(rgb).float().permute((0, 4, 3, 1, 2))[i0:,...] # (seq, num_cams, channels, img_h, img_w)
            num_cam = rgb.shape[1]

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
            dec_src_mask, dec_mem_mask, dec_tgt_mask = get_dec_ar_masks(16*num_cam, self.max_skill_len)
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
            plan_tgt_mask = get_plan_ar_masks(self.max_num_skills)
            data["plan_tgt_mask"] = plan_tgt_mask

        # Add extra dimension for batch size as model expects this.
        if self.add_batch_dim:
            for k,v in data.items():
                data[k] = v.unsqueeze(0)

        return data


def get_skill_pad_from_seq_pad(seq_pad, max_skill_len):
    """Functions for generating a skill padding mask from a given sequence padding mask,
    based on the max skill length from the config. Always adds the padding at the end.
        - seq_pad: tensor (seq)
        - max_skill_len: int
    """
    max_num_skills = torch.ceil(torch.tensor(seq_pad.shape[0]/max_skill_len)).to(torch.int)
    num_unpad_seq = torch.sum(torch.logical_not(seq_pad).to(torch.int16))
    num_unpad_skills = torch.ceil(num_unpad_seq / max_skill_len) # get skills with relevant outputs TODO handle non divisible skill lengths
    skill_pad_mask = torch.zeros(max_num_skills) # True is unattended
    skill_pad_mask[num_unpad_skills.to(torch.int16):] = 1
    skill_pad_mask = skill_pad_mask.to(torch.bool)
    return skill_pad_mask
    

def get_dec_ar_masks(num_img_feats, max_skill_len):
    """ 
    The masks for this information will be diagonal such that at each timestep, 
    the model can only attend to the conditional info at that timestep. This is
    for the specific order that is used in the SkillVAE model
    """
    dec_src_len = max_skill_len * (2 + num_img_feats) # (MSL*(q + img_feats + z))
    # tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(max_skill_len)
    tgt_mask = ~(torch.eye(max_skill_len).to(torch.bool)) # Only allow self attention for single step prediction
    mem_mask = torch.ones(max_skill_len, dec_src_len).to(torch.bool) # Start with everything masked
    dec_mask = torch.ones(dec_src_len, dec_src_len).to(torch.bool) # Start with everything masked
    
    sa_mask = ~torch.diag(torch.ones(max_skill_len)).to(torch.bool) # Self attention mask
    dec_mask[:max_skill_len,:max_skill_len] =  sa_mask # Unmask qpos self attention
    dec_mask[-max_skill_len:,-max_skill_len:] = sa_mask # Unmask z self attention
    for s in range(max_skill_len):
        im_start = max_skill_len + s*num_img_feats
        im_end = im_start + num_img_feats
        zs = -max_skill_len + s
        # Decoder mask
        dec_mask[im_start:im_end, im_start:im_end] = False # Unmask img features self attention block
        dec_mask[s,im_start:im_end] = False # Unmask qpos attention to img features
        dec_mask[zs,im_start:im_end] = False # Unmask z attention to img features
        dec_mask[im_start:im_end,s] = False # Unmask img features attention to qpos
        dec_mask[im_start:im_end,zs] = False # Unmask img features attention to z
        dec_mask[s,zs] = False # Unmask qpos attention to z
        dec_mask[zs,s] = False # Unmask z attention to qpos
        # Memory mask, leave column 0 all True to mask attention to start token 
        mem_mask[s,im_start:im_end] = False # Unmask action attention to img features
        mem_mask[s,zs] = False # Unmask action attention to z
        mem_mask[s,s] = False # Unmask action attention to qpos

    return dec_mask, mem_mask, tgt_mask


def get_plan_ar_masks(max_num_skills):
    """
    The masks for this information will be causal for the skills
    """
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(max_num_skills)
    return tgt_mask


def get_enc_causal_masks(max_seq_len, max_num_skills, max_skill_len):
    """
    Gets a causal mask for the encoder. Is only the size of max seq len, so has to be repeated in the encoder itself.
    """
    enc_causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(max_seq_len)
    enc_tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(max_num_skills)
    enc_mem_mask = torch.ones(max_num_skills, max_seq_len).to(torch.bool)
    for s in range(max_num_skills):
        sk_end = s*max_skill_len + max_skill_len
        enc_mem_mask[s,:sk_end] = False # Unmask skill attention to prior sequence items
    return enc_causal_mask, enc_mem_mask, enc_tgt_mask


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
    def __init__(self, cfg, cfg_model) -> None:
        self.method = cfg["method"]
        self.max_seq_len = cfg["data"]["max_seq_len"]
        self.max_skill_len = cfg_model["max_skill_len"]
        self.cond_dec = cfg_model.get("conditional_decode", True)
        cfg_aug = cfg["data"]["augmentation"]
        self.subsequence_rate = cfg_aug.get("subsequence_rate", 0)
        self.seq_masking_rate = cfg_aug.get("seq_masking_rate", 0)
        self.type_masking_rate = cfg_aug.get("type_masking_rate", 0)
        self.img_aug = cfg_aug.get("image_aug", 0)
        self.input_noise = cfg_aug.get("input_noise", False)

    def __call__(self, data):
        
        if self.subsequence_rate > 0:
            data = self.subsequence(data)

        if self.seq_masking_rate > 0:
            data = self.seq_masking(data)

        if self.type_masking_rate > 0:
            data = self.type_masking(data)

        return data
    
    def subsequence(self, data):
        seq_pad_mask = data["seq_pad_mask"]
        num_unpad_seq = torch.sum((~seq_pad_mask).to(torch.int16))

        val = torch.rand(1)
        if self.subsequence_rate > val:
            # Uniformly sample how much of the sequence to use for the batch
            # from 1 to entire (unpadded) seq
            num_seq = torch.randint(1, num_unpad_seq+1, (1,1)).squeeze()
            
            # Reset "new" sequences to the beginning (for positional encodings)
            for k,v in data.items():
                if "mask" not in k and "goal" not in k:
                    new_seq = torch.zeros_like(v)
                    new_seq[:num_seq,...] = v[:num_seq,...]
                    data[k] = new_seq

            # Reset new "goal" state
            if self.method == "plan":
                if "img_feat" in data.keys():
                    data["goal_feat"] = data["img_feat"][num_seq:num_seq+1,...]
                    data["goal_pe"] = data["img_pe"][num_seq:num_seq+1,...]
                else:
                    data["goal"] = data["rgb"][num_seq:num_seq+1,...]

            # Recalculate appropriate masking 
            # (also start from the begining of the seq)
            num_unpad_skills = torch.ceil(torch.clone(num_seq / self.max_skill_len)).to(torch.int16)
            data["skill_pad_mask"][num_unpad_skills:] = True
            data["seq_pad_mask"][num_seq:] = True

        return data

    def image_aug(self, data):
        """Image augmentation function for input images"""
        n = data["rgb"].shape[0]
        jit = v2.ColorJitter(.5, .5)
        # gs = v2.Grayscale(3)
        gb = v2.GaussianBlur(3)
        re = v2.RandomErasing(0.5, (0.02,0.1))
        ra = v2.RandomChoice([jit, gb, re], [self.img_aug, self.img_aug, self.img_aug])

        for m in range(n):
            data["rgb"][m,...] = ra(data["rgb"][m,...]).clamp(0,1)

        return data
    
    def seq_masking(self, data):
        """Encoder input sequence masking function. Masks a specfic timestep for all inputs.
        Doesn't mask inputs to decoder"""
        if "rgb" in data.keys():
            n_cam = data["rgb"].shape[1]
        elif "img_feat" in data.keys():
            n_cam = data["img_feat"].shape[1]

        # Randomly apply mask to each input timestep
        enc_mask = torch.rand(self.max_seq_len) < self.seq_masking_rate
        enc_mask = enc_mask.unsqueeze(0).repeat(self.max_seq_len, 1)
        enc_mask = enc_mask.fill_diagonal_(False) # Allow self attention

        data["enc_mask"] = enc_mask

        # Check if mask + padding yields a fully masked input sequence
        # If so, deactivate the input mask
        if torch.all(data["seq_pad_mask"] | data["enc_src_mask"][0,:]):
            data["enc_src_mask"] = torch.zeros_like(data["enc_src_mask"])

        return data

    def type_masking(self, data):
        """Encoder/decoder type masking function. 
        Always leaves at least 1 unmasked input type."""

        raise NotImplementedError

        if "rgb" in data.keys():
            n_cam = data["rgb"].shape[1]
        elif "img_feat" in data.keys():
            n_cam = data["img_feat"].shape[1]
        n_seq = (2 + n_cam) # HARDCODED
        # Randomly apply mask to each input type (encoder img,act,qpos & decoder imgs)
        enc_type_mask = torch.rand(n_seq) < self.type_masking_rate
        # Unmask an input if all input masks are True
        if torch.all(enc_type_mask):
            i = torch.randint(0,n_seq,(1,1)).squeeze()
            enc_type_mask[i] = False

        enc_inp_len = self.max_seq_len * (n_seq)
        enc_mask = torch.zeros(enc_inp_len).to(torch.bool)
        for s in range(enc_type_mask.shape[0]-1):
            if enc_type_mask[s]:
                enc_mask[s*self.max_seq_len:(s+1)*self.max_seq_len] = True
        enc_mask = enc_mask.unsqueeze(0).repeat(enc_inp_len, 1)
        enc_mask = enc_mask.fill_diagonal_(False)

        # Merge with existing masks if applicable
        if "enc_mask" in data.keys():
            data["enc_mask"] = data["enc_mask"] | enc_mask
        else:
            data["enc_mask"] = enc_mask

        # Check if input mask + padding yields a fully masked input sequence
        # If so, deactivate the input mask
        if torch.sum(torch.logical_not(data["seq_pad_mask"].unsqueeze(0).repeat(enc_inp_len, n_seq) | data["enc_mask"]).to(torch.int)) < 10:
            data["enc_mask"] = torch.zeros_like(data["enc_mask"])
        
        return data


### Commands for trajectory replay ###
# python -m mani_skill2.trajectory.replay_trajectory   --traj-path /home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.h5   --save-traj --target-control-mode pd_joint_delta_pos --obs-mode rgbd --num-procs 10
# python -m policy/dataset/replay_trajectory.py --traj-path data/demos/v0/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_joint_delta_pos --num-procs 2 --cam-res 480