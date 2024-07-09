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


# Below are commented out mani skill functions that I can't load
# If there are functions below they are replicated mani skill functions
from policy.dataset.helpers import load_json, pixelToCoordinate
# from mani_skill2.utils.io_utils import load_json
# from mani_skill2.utils.common import flatten_state_dict


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
    Given u,v,d pixel coordinates and returns x,y,z reference frame coordinates for that point
    followed by the corresponding RGB values.

    Should only be used right before input into ConvONet
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
            x, y, z = pixelToCoordinate([u, v, depth_map[u, v]])
            R = img[u, v, 0]
            G = img[u, v, 1]
            B = img[u, v, 2]

            input_vec.append([x, y, z, R, G, B])
    
    return input_vec


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


class ManiSkillConvONetDataSet(ManiSkillDataset):
    """Class that preps data for ConvONet processing"""
    def __init__(self, dataset_file: str, indices: list = None, 
                 max_seq_len: int = 0, pad: bool = False, 
                 augmentation= None) -> None:
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
        self.augmentation = augmentation

    def __len__(self):
        return len(self.owned_indices)

    def __getitem__(self, idx):
        eps = self.episodes[self.owned_indices[idx]]
        trajectory = self.data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)

        # convert the original raw observation with our batch-aware function
        obs = convert_observation(trajectory["obs"])
        
        # we use :-1 to ignore the last obs as terminal observations are included
        rgbd = obs["rgbd"][:-1]
        rgbd = rescale_rgbd(rgbd, separate_cams=True)
        rgbd = rgbd.transpose((0, 4, 3, 1, 2)) # (seq, num_cams, channels, img_h, img_w)
        rgbd = rgbd.tolist()
        for seq, seq_data in tqdm(enumerate(rgbd)):
            for cam, cam_data in enumerate(seq_data):
                rgbd[seq][cam] = generate_input_vector(cam_data)
        
        rgbd = torch.tensor(rgbd).float()

        # Add padding to sequences to match lengths and generate padding masks
        if self.pad:
            num_unpad_seq = rgbd.shape[0]
            pad = self.max_seq_len - num_unpad_seq
            seq_pad_mask = torch.cat((torch.zeros(rgbd.shape[0]), torch.ones(pad)), axis=0).to(torch.bool)

            rgbd_pad = torch.zeros([pad] + list(rgbd.shape[1:]))
            
            rgbd = torch.cat((rgbd, rgbd_pad), axis=0).to(torch.float32)

        else: # If not padding, this is being passed directly to model. 
            seq_pad_mask = torch.zeros(rgbd.shape[0]).to(torch.bool)
            
        # Assume no masking for now
        seq_mask = torch.zeros(seq_pad_mask.shape[-1], seq_pad_mask.shape[-1]).to(torch.bool)
        
        data = dict(rgbd=rgbd, seq_pad_mask=seq_pad_mask, seq_mask=seq_mask)
        
        if self.augmentation is not None:
            data = self.augmentation(data)

        if not self.pad: # Add extra dimension for "batch", so model runs properly when testing.
            for k,v in data.items():
                data[k] = v.unsqueeze(0)

        return data
    
# Need to edit/rewrite this for Conv Occ Net - idea: create scene by scene reconstructions using only the RGBD inputs
def get_MS_loaders(cfg,  **kwargs) -> None:
        cfg_data = cfg["data"]
        dataset_file: str = cfg_data["dataset"]
        val_split: float = cfg_data.get("val_split", 0.1)
        preshuffle: bool = cfg_data.get("preshuffle", True)
        augment: bool = cfg["data"].get("augment", False) # Augmentation
        count = cfg_data.get("max_count", None) # Dataset count limitations

        assert os.path.exists(dataset_file)
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
            print("Max sequence length found to be: ", max_seq_len)

        print("Collecting all observations...")
        for idx in tqdm(range(num_episodes)):
            eps = episodes[idx]
            trajectory = data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            obs = convert_observation(trajectory["obs"])

        path = os.path.join(cfg["training"]["out_dir"],'scaling_functions.pickle')            

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
            
            # # Save index split to pickle file
            # if os.path.exists(path):
            #     print("Replacing existing train/val index file")
            #     with open(path,'wb') as f:
            #         pickle.dump((train_idx, val_idx), f)
            # else:
            #     print("Creating new train/val index file")
            #     with open(path,'xb') as f:
            #         pickle.dump((train_idx, val_idx), f)
        elif existing_indices=="file":
            assert os.path.exists(path), "out directory does not contain index file"
            print(f"Loading indices from file: {path}")
            with open(path,'rb') as f:
                    train_idx, val_idx = pickle.load(f)
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
        train_dataset = ManiSkillConvONetDataSet(dataset_file, train_idx, 
                                               max_seq_len,
                                               pad_train, train_augmentation)
        val_dataset = ManiSkillConvONetDataSet(dataset_file, val_idx, 
                                             max_seq_len,
                                             pad_val, val_augmentation)

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


def get_scaling_functions(data, scaling, sep_idx):
    if isinstance(scaling, (int, float, list, tuple)):
        print("Computing linear scaling")
        def action_scaling(x):
            return x * torch.tensor(scaling)
        def action_scaling_inv(x):
            return x / torch.tensor(scaling)
    elif scaling=="norm": 
        print("Computing action norm")
        act_mu = torch.mean(data, 0)
        act_std = torch.std(data, 0)
        def action_scaling(x):
            return (x - act_mu) / torch.sqrt(act_std)
        def action_scaling_inv(x):
            return x * torch.sqrt(act_std) + act_mu
    elif scaling=="robust_scaler":
        print("Computing robust scaler")
        scaler = skp.RobustScaler().fit(data)
        def action_scaling(x):
            return torch.from_numpy(scaler.transform(x.numpy()))
        def action_scaling_inv(x):
            return torch.from_numpy(scaler.inverse_transform(x.numpy()))
    elif scaling in ("normal","uniform"):
        print(f"Computing {scaling} quantile transform")
        scaler = skp.QuantileTransformer(output_distribution=scaling)
        scaler.fit(data)
        def action_scaling(x):
            return torch.from_numpy(scaler.transform(x.numpy()))
        def action_scaling_inv(x):
            return torch.from_numpy(scaler.inverse_transform(x.numpy()))
    elif scaling=="power":
        print(f"Computing power transform")
        scaler = skp.PowerTransformer()
        scaler.fit(data)
        def action_scaling(x):
            return torch.from_numpy(scaler.transform(x.numpy()))
        def action_scaling_inv(x):
            return torch.from_numpy(scaler.inverse_transform(x.numpy()))
    else:
        raise ValueError(f"Unsupported action scaling given: {scaling}")

    if sep_idx is not None:
        def seperate_scaling(function, x, idx):
            return torch.hstack((function(x[:,:idx]), x[:,idx:]))
        scaling_fcn_forward = lambda x: seperate_scaling(action_scaling, x, sep_idx)
        scaling_fcn_inverse = lambda x: seperate_scaling(action_scaling_inv, x, sep_idx)
    else:
        scaling_fcn_forward = action_scaling
        scaling_fcn_inverse = action_scaling_inv

    return scaling_fcn_forward, scaling_fcn_inverse


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


import matplotlib.animation as animation
# import open3d as o3d
# import trimesh

if __name__ == "__main__":
    path = "tskill/data/demos/PegInsertionSide-v0/trajectory.rgbd.pd_joint_delta_pos.h5"
    dataset = ManiSkillConvONetDataSet(path)

    # print(dataset[0]["rgbd"][0][0][0])


    animate = False
    pointcloud = False
    trimesh_demo = False
    data_conversion = False
    save_data = False

    if save_data:
        data = dataset[0]["rgbd"][0][0]
        data = tensor_to_numpy(data)
        cam1 = data[0]
        cam2 = data[1]
        
        # file name, arrayname, arraydata
        np.savez("test_data", cam1=cam1, cam2=cam2)
    

    if data_conversion:
        episode = dataset[0]["rgbd"]
        batch = episode[0]
        scene = batch[100]
        cam = scene[0]
        img = tensor_to_numpy(torch.permute(cam[0:3, :, :], (1, 2, 0)))
        depth_map = tensor_to_numpy(cam[3, :, :])
        h, w = depth_map.shape
        input_vec = []

        for u in range(h):
            for v in range(w):
                x, y, z = pixelToCoordinate([u, v, depth_map[u, v]])
                R = img[u, v, 0]
                G = img[u, v, 1]
                B = img[u, v, 2]

                input_vec.append([x, y, z, R, G, B])
        
        input_vec = np.array(input_vec, dtype=np.float32)
        print(input_vec.shape)


    if trimesh_demo:
        episode = dataset[0]["rgbd"]
        batch = episode[0]
        scene = batch[100]
        cam = scene[1]
        img = tensor_to_numpy(torch.permute(cam[0:3, :, :], (1, 2, 0)))
        depth_map = tensor_to_numpy(cam[3, :, :])
        print(min(depth_map[-1, :]))
        
        depth_map_reform = []

        row, col = depth_map.shape
        for r in range(row):
            for c in range(col):
                vertex = [r, c, depth_map[r][c]]
                depth_map_reform.append(vertex)

        # depth_map_reform = np.array(depth_map_reform)
        # print(depth_map_reform)

        # mesh  = trimesh.Trimesh(depth_map_reform)
        # print(mesh.vertices)
        # print(mesh.vertex_normals)
        

    if pointcloud:
        episode = dataset[0]["rgbd"]
        batch = episode[0]
        scene = batch[100]
        cam = scene[1]
        img = tensor_to_numpy(torch.permute(cam[0:3, :, :], (1, 2, 0)))
        depth_map = tensor_to_numpy(cam[3, :, :])
        
        # Intrinsic camera parameters (example values)
        fx = 100.0  # Focal length in x
        fy = 100.0  # Focal length in y
        cx = 64   # Principal point x
        cy = 64   # Principal point y

        # Generate 3D points
        height, width = depth_map.shape
        i, j = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_map
        x = (j - cx) * z / fx
        y = (i - cy) * z / fy
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # Generate colors
        colors = img.reshape(-1, 3) / 255.0

        # Create Open3D point cloud object
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # # pcd.colors = o3d.utility.Vector3dVector(colors)

        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])
    

    if animate:
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Generate a series of images
        images = []

        episode = dataset[0]['rgbd']
        batch = episode[0]
        for scene in batch:
            cam = scene[0]
            img = tensor_to_numpy(torch.permute(cam[0:3, :, :], (1, 2, 0)))
            images.append(img)

        # Initialize the image
        im = ax.imshow(images[0])

        # Function to update the image
        def update(frame):
            im.set_array(images[frame])
            return im,

        # Create the animation
        ani = animation.FuncAnimation(
            fig,        # The figure object
            update,     # The update function
            frames=len(images),  # Number of frames
            interval=100,  # Interval in milliseconds
            blit=True    # Use blitting to optimize drawing
        )

        # Display the animation
        plt.show()


    ### Commands for trajectory replay ###
    # python -m mani_skill2.trajectory.replay_trajectory   --traj-path /home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.h5   --save-traj --target-control-mode pd_joint_delta_pos --obs-mode rgbd --num-procs 10
    # python -m policy/dataset/replay_trajectory.py --traj-path data/demos/v0/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_joint_delta_pos --num-procs 2 --cam-res 480