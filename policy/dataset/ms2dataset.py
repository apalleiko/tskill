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


def convert_observation(observation):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images

    # image data is not scaled here and is kept as uint16 to save space
    image_obs = observation["image"]
    rgb = image_obs["base_camera"]["rgb"]
    depth = image_obs["base_camera"]["depth"]
    rgb2 = image_obs["hand_camera"]["rgb"]
    depth2 = image_obs["hand_camera"]["depth"]

    # we provide a simple tool to flatten dictionaries with state data

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
        rgbd = np.concatenate([rgb1, rgb2], axis=-1)
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
    def __init__(self, dataset_file: str) -> None:
        self.dataset_file = dataset_file
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
    
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
    def __init__(self, dataset_file: str) -> None:
        self.dataset_file = dataset_file
        super().__init__(dataset_file)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        eps = self.episodes[idx]
        trajectory = self.data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)

        # convert the original raw observation with our batch-aware function
        obs = convert_observation(trajectory["obs"])
        
        # we use :-1 to ignore the last obs as terminal observations are included
        # and they don't have actions
        action = th.from_numpy(trajectory["actions"]).float()
        rgbd = obs["rgbd"][:-1]

        rgbd = rescale_rgbd(rgbd, discard_depth=True)
        
        # permute data so that channels are the first dimension as PyTorch expects this
        # (bs, num_cams, channels, img_w, img_h)
        rgbd = th.from_numpy(rgbd).float().permute((0, 3, 1, 2))
        state = th.from_numpy(obs["state"][:-1]).float()
        return dict(rgbd=rgbd, state=state), action
    

if __name__ == "__main__":
    path = "/home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.rgbd.pd_joint_delta_pos.h5"
    assert osp.exists(path)
    # dataset = ManiSkillrgbdDataset(path)
    dataset = ManiSkillrgbSeqDataset(path)
    print("Length of Dataset: ",len(dataset))
    
    # Have to batch with only 1 since episodes are not the same length
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True, drop_last=True, shuffle=True)
    obs, action = dataset[13]
    print("Observation: ", [(key,item.shape) for key,item in obs.items()])
    
    # Plot image observations
    fig, (ax1, ax2) = plt.subplots(1, 2)
    img_idx = -22
    imgs = [item for key,item in obs.items()][0]
    ax1.imshow(np.transpose(imgs[img_idx,:3,:,:],(1,2,0)))
    ax2.imshow(np.transpose(imgs[img_idx,3:,:,:],(1,2,0)))
    plt.show()
    # print("Action: ", action.shape)
    
    # Sequence load time benchmark
    dl = iter(dataloader)
    t0 = time.time()
    for i in range(10):
        next(dl)
    tf = time.time()
    print("Load Time: ",(tf-t0)/10)

    ### Commands for trajectory replay ###
    # python -m mani_skill2.trajectory.replay_trajectory   --traj-path /home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.h5   --save-traj --target-control-mode pd_joint_delta_pos --obs-mode rgbd --num-procs 10
    # python -m policy/dataset/replay_trajectory.py --traj-path data/demos/v0/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_joint_delta_pos --num-procs 2 --cam-res 480