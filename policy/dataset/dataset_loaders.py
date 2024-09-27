import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import dill as pickle
import os
import os.path as osp
import h5py
import numpy as np
import copy

from mani_skill2.utils.io_utils import load_json
from policy.dataset.data_utils import ScalingFunction, DataAugmentation, load_h5_data
from policy.dataset.ms2dataset import ManiSkillrgbSeqDataset, convert_observation as convert_obs_ms
from policy.dataset.LIBEROdataset import LiberoDataset, convert_observation as convert_obs_libero
from policy.dataset.multitask_dataset import MultitaskDataset


def dataset_loader(cfg, **kwargs) -> None:
    dataset = cfg["data"]["dataset"]

    if isinstance(dataset, (list,tuple)):
        dataset_list = dataset
        return multitask_dataset_loader(dataset_list, cfg, **kwargs)
    elif os.path.isdir(dataset):
        dataset_list = os.listdir(dataset)
        dataset_list = [os.path.join(dataset, f) for f in dataset_list if ".h5" in f or ".hdf5" in f]
        return multitask_dataset_loader(dataset_list, cfg, **kwargs)
    elif os.path.isfile(dataset):
        return singletask_dataset_loader(cfg, **kwargs)


def singletask_dataset_loader(cfg, **kwargs) -> None:
        method = cfg["method"]
        if method == "plan":
            cfg_model_vae = cfg["vae_cfg"]["model"]
            cfg_vae = cfg["vae_cfg"]
        else:
            cfg_model_vae = cfg["model"]
            cfg_vae = cfg
        cfg_data = cfg["data"]
        mode = cfg_data.get("mode","maniskill")
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
        batch_size = cfg["training"]["batch_size"]
        batch_size_val = cfg["training"]["batch_size_val"]

        # Scale actions/states, or compute normalization for the dataset
        action_scaling = cfg_vae["data"].get("action_scaling",1)
        state_scaling = cfg_vae["data"].get("state_scaling",1)
        gripper_scaling = cfg_vae["data"].get("gripper_scaling", True)
        recompute_scaling = True # Whether recomputing action/state scaling is needed
        recompute_fullseq = False # Whether recomputing full sequence data map is needed

        # kwargs
        save_override = kwargs.get("save_override", False)
        fullseq_override = kwargs.get("fullseq_override", False)
        recalc_override = kwargs.get("recalc_override", False)
        multitask = kwargs.get("multitask",False)

        assert osp.exists(dataset_file)
        data = h5py.File(dataset_file, "r")
        if mode == "maniskill":
            json_path = dataset_file.replace(".h5", ".json")
            json_data = load_json(json_path)
            episodes = json_data["episodes"]
        elif mode == "libero":
            episodes = data["data"]
        else:
            raise ValueError("Unknown data mode passed")

        # Try loading existing data config info pickle file
        data_info = kwargs.get("data_info",None)
        info_path = os.path.join(cfg["training"]["out_dir"],'data_info.pickle')
        if data_info is None:
            try:
                with open(info_path,'rb') as f:
                        data_info = pickle.load(f)
                print("Found existing data info file")
            except FileNotFoundError:
                data_info = dict()

        ### Get train/val indices or sequence mapping
        override_indices = kwargs.get("indices", None)
        if override_indices is not None:
            print("Using override indices")
            train_mapping, val_mapping = override_indices
            if not recalc_override and all([i in data_info.keys() for i in ("action_scaling", "state_scaling")]):
                print("Loading action and state scaling from file")
                act_scaling = data_info["action_scaling"]
                stt_scaling = data_info["state_scaling"]
                recompute_scaling = False
        elif not recalc_override and all([i in data_info.keys() for i in ("train_indices", "val_indices")]):
            print(f"Loading indices from file: {info_path}")
            if fullseq_override:
                print("Overriding full seq config!")
                train_mapping = data_info["train_ep_indices"]
                val_mapping = data_info["val_ep_indices"]
            else:
                train_mapping = data_info["train_indices"]
                val_mapping = data_info["val_indices"]

            if not multitask:
                print("Loading action and state scaling from file")
                act_scaling = data_info["action_scaling"]
                stt_scaling = data_info["state_scaling"]
            recompute_scaling = False
        else:
            print("Updating new train & val indices")
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

        # Collect train/val actions/states if needed
        if recompute_scaling or recompute_fullseq:
            train_acts = []
            train_states = []
            train_lengths = []
            t = tqdm(range(len(train_idx)), "Collecting all training data info:")
            for i in t:
                idx = train_idx[i]
                if mode == "maniskill":
                    eps = episodes[idx]
                    trajectory = data[f"traj_{eps['episode_id']}"]
                    trajectory = load_h5_data(trajectory)
                    obs = convert_obs_ms(trajectory["obs"], pos_only=False)
                elif mode == "libero":
                    trajectory = episodes[f"demo_{idx}"]
                    trajectory = load_h5_data(trajectory)
                    obs = convert_obs_libero(trajectory["obs"])

                actions = torch.from_numpy(trajectory["actions"]).float()
                states = torch.from_numpy(obs["state"][:-1]).float()
                train_acts.append(actions)
                train_states.append(states)
                seq_size = actions.shape[0]
                train_lengths.append(seq_size)
                if seq_size > max_seq_len:
                    t.set_postfix_str(f"New max sequence length: {seq_size}")
                    max_seq_len = seq_size
            
            train_acts = torch.vstack(train_acts)
            train_states = torch.vstack(train_states)

            if recompute_fullseq:
                val_lengths = []
                for i in tqdm(range(len(val_idx)), "Collecting all val data info:"):
                    idx = val_idx[i]

                    if mode == "maniskill":
                        eps = episodes[idx]
                        trajectory = data[f"traj_{eps['episode_id']}"]
                        trajectory = load_h5_data(trajectory)
                        obs = convert_obs_ms(trajectory["obs"], pos_only=False)
                    elif mode == "libero":
                        trajectory = episodes[f"demo_{idx}"]
                        trajectory = load_h5_data(trajectory)
                        obs = convert_obs_libero(trajectory["obs"])

                    actions = torch.from_numpy(trajectory["actions"]).float()
                    seq_size = actions.shape[0]
                    val_lengths.append(seq_size)

        ### Create scaling functions if needed
        if recompute_scaling and not multitask:
            print("Recomputing scaling functions...")
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
        elif multitask:
            act_scaling = stt_scaling = None

        # Compute full sequence mapping if needed
        if recompute_fullseq:
            print("Computing full sequence mapping...")
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

        ### Save updated data info file
        if recompute_scaling and not save_override and not multitask:
            print("Saving data info file")
            with open(info_path,'wb') as f:
                pickle.dump(data_info, f)

        ### Obtain augmentations
        if augment:
            train_augmentation = DataAugmentation(cfg, cfg_model_vae)
        else:
            train_augmentation = None

        ### Get extra configs
        return_dataset = kwargs.get("return_datasets", False)
        override_batch_dim = kwargs.get("add_batch_dim",False)
        add_batch_dim = not pad or return_dataset or override_batch_dim
        if add_batch_dim:
            print("Adding batch dimension to returned data!")
        if batch_size == 1:
            pad2msl_train = True
        else:
            pad2msl_train = False
        if batch_size_val == 1:
            pad2msl_val = True
        else:
            pad2msl_val = False

        ### Create datasets
        if mode == "maniskill":
            ds = ManiSkillrgbSeqDataset
        elif mode == "libero":
            ds = LiberoDataset

        train_dataset = ds(method, dataset_file, train_mapping, 
                            max_seq_len, max_skill_len,
                            pad, train_augmentation,
                            act_scaling, stt_scaling,
                            full_seq, autoregressive_decode, encoder_is_causal,
                            add_batch_dim=add_batch_dim,
                            pad2msl=pad2msl_train)
        val_dataset = ds(method, dataset_file, val_mapping, 
                            max_seq_len, max_skill_len,
                            pad, None,
                            act_scaling, stt_scaling,
                            full_seq, autoregressive_decode, encoder_is_causal,
                            add_batch_dim=add_batch_dim,
                            pad2msl=pad2msl_val)

        if multitask:
            return train_dataset, val_dataset, data_info

        if return_dataset:
            return train_dataset, val_dataset        

        ### Create loaders
        shuffle = kwargs.get("shuffle", True)
        print(f"Shuffling: {shuffle}")
        train_loader =  DataLoader(train_dataset, batch_size=batch_size, 
                                   num_workers=cfg["training"]["n_workers"],
                                   pin_memory=True, drop_last=True, shuffle=shuffle,
                                   persistent_workers=True)
        val_loader =  DataLoader(val_dataset, batch_size=batch_size_val, 
                                 num_workers=cfg["training"]["n_workers_val"], 
                                 pin_memory=True, drop_last=True, shuffle=shuffle,
                                 persistent_workers=True)
        
        return train_loader, val_loader


def multitask_dataset_loader(dataset_list, cfg, **kwargs):
    method = cfg["method"]
    if method == "plan":
        cfg_vae = cfg["vae_cfg"]
    else:
        cfg_vae = cfg
    cfg_data = cfg["data"]
    mode = cfg_data.get("mode","maniskill")
    batch_size = cfg["training"]["batch_size"]
    batch_size_val = cfg["training"]["batch_size_val"]
    dataset_file: str = cfg_data["dataset"]
    max_seq_len = cfg_data.get("max_seq_len", 0)

    train_sequence_datasets = []
    val_sequence_datasets = []

    # Scale actions/states, or compute normalization for the dataset
    action_scaling = cfg_vae["data"].get("action_scaling",1)
    state_scaling = cfg_vae["data"].get("state_scaling",1)
    gripper_scaling = cfg_vae["data"].get("gripper_scaling", True)
    recompute_scaling = True # Whether recomputing action/state scaling is needed

    # kwargs
    save_override = kwargs.get("save_override", False)
    recalc_override = kwargs.get("recalc_override", False)

    # Try loading existing data config info pickle file
    path = os.path.join(cfg["training"]["out_dir"],'data_info.pickle')
    try:
        with open(path,'rb') as f:
                data_info = pickle.load(f)
        print("Found existing multitask data info file")
    except FileNotFoundError:
        data_info = dict()

    ### Get train/val indices or sequence mapping
    if not recalc_override and len(data_info.keys()) > 0:
        print(f">>> Loading multitask dataset from file: {path}")
        for i,data_info_i in data_info["datasets"].items():
            print(f"Loading dataset: {i}")
            cfg_i = copy.deepcopy(cfg)
            cfg_i["data"]["dataset"] = i
            train_i, val_i, _ = dataset_loader(cfg_i, multitask=True, data_info=data_info_i, **kwargs)
            train_sequence_datasets.append(train_i)
            val_sequence_datasets.append(val_i)

        print(">>> Loading action and state scaling from file")
        act_scaling = data_info["action_scaling"]
        stt_scaling = data_info["state_scaling"]
        recompute_scaling = False
    else:
        print("Updating new multitask train & val datasets")
        data_info["datasets"] = dict()
        for i in dataset_list:
            assert osp.exists(i)
            print(f"Loading dataset: {i}")
            cfg_i = copy.deepcopy(cfg)
            cfg_i["data"]["dataset"] = i
            train_i, val_i, data_info_i = dataset_loader(cfg_i, multitask=True, **kwargs)
            data_info["datasets"][i] = data_info_i
            train_sequence_datasets.append(train_i)
            val_sequence_datasets.append(val_i)

    ### Create scaling functions across all datasets if needed
    if recompute_scaling:
        print("Recomputing scaling functions...")
        train_acts = []
        train_states = []
        train_lengths = []
        for d_file,d_info in data_info["datasets"].items():
            train_idx = d_info["train_ep_indices"]
            data = h5py.File(d_file, "r")

            if mode == "maniskill":
                json_path = d_file.replace(".h5", ".json")
                json_data = load_json(json_path)
                episodes = json_data["episodes"]
            elif mode == "libero":
                episodes = data["data"]
            else:
                raise ValueError("Unknown data mode passed")

            t = tqdm(range(len(train_idx)), "Collecting all training data info:")
            for i in t:
                idx = train_idx[i]

                if mode == "maniskill":
                    eps = episodes[idx]
                    trajectory = data[f"traj_{eps['episode_id']}"]
                    trajectory = load_h5_data(trajectory)
                    obs = convert_obs_ms(trajectory["obs"], pos_only=False)
                elif mode == "libero":
                    trajectory = episodes[f"demo_{idx}"]
                    trajectory = load_h5_data(trajectory)
                    obs = convert_obs_libero(trajectory["obs"])

                actions = torch.from_numpy(trajectory["actions"]).float()
                states = torch.from_numpy(obs["state"][:-1]).float()
                train_acts.append(actions)
                train_states.append(states)
                seq_size = actions.shape[0]
                train_lengths.append(seq_size)
                if seq_size > max_seq_len:
                    t.set_postfix_str(f"New max sequence length: {seq_size}")
                    max_seq_len = seq_size

            data.close()
        
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

    ### Apply scaling function to all loaded datasets
    for ds in train_sequence_datasets:
        ds.action_scaling = act_scaling
        ds.state_scaling = stt_scaling
        ds.max_seq_len = max_seq_len
    
    for ds in val_sequence_datasets:
        ds.action_scaling = act_scaling
        ds.state_scaling = stt_scaling
        ds.max_seq_len = max_seq_len

    ### Save updated data info file
    if recompute_scaling and not save_override:
        print("Saving data info file")
        with open(path,'wb') as f:
            pickle.dump(data_info, f)

    ### Create datasets
    train_dataset = MultitaskDataset(train_sequence_datasets)
    val_dataset = MultitaskDataset(val_sequence_datasets)

    ### Get extra configs
    return_dataset = kwargs.get("return_datasets", False)
    if return_dataset:
        return train_dataset, val_dataset

    ### Create loaders
    shuffle = kwargs.get("shuffle", True)
    print(f"Shuffling: {shuffle}")
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, 
                                num_workers=cfg["training"]["n_workers"],
                                pin_memory=True, drop_last=True, shuffle=shuffle,
                                persistent_workers=True)
    val_loader =  DataLoader(val_dataset, batch_size=batch_size_val,
                                num_workers=cfg["training"]["n_workers_val"], 
                                pin_memory=True, drop_last=True, shuffle=shuffle,
                                persistent_workers=True)
    
    return train_loader, val_loader