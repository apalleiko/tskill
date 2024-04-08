# Import required packages
import argparse
import os.path as osp
import numpy as np
import h5py
import torch as th
from tqdm.notebook import tqdm

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
from torch.utils.data import Dataset, DataLoader

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
    from mani_skill2.utils.common import flatten_state_dict
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

def rescale_rgbd(rgbd, scale_rgb_only=False):
    # rescales rgbd data and changes them to floats
    rgb1 = rgbd[..., 0:3] / 255.0
    rgb2 = rgbd[..., 4:7] / 255.0
    depth1 = rgbd[..., 3:4]
    depth2 = rgbd[..., 7:8]
    if not scale_rgb_only:
        depth1 = rgbd[..., 3:4] / (2**10)
        depth2 = rgbd[..., 7:8] / (2**10)
    return np.concatenate([rgb1, depth1, rgb2, depth2], axis=-1)

# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

class ManiSkill2Dataset(Dataset):
    def __init__(self, dataset_file: str, load_count=-1) -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        import h5py
        from mani_skill2.utils.io_utils import load_json
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.observations = []
        self.actions = []
        self.total_frames = 0
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            self.observations.append(trajectory["obs"][:-1])
            self.actions.append(trajectory["actions"])
        self.observations = np.vstack(self.observations)
        self.actions = np.vstack(self.actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = th.from_numpy(self.actions[idx]).float()
        obs = th.from_numpy(self.observations[idx]).float()
        return obs, action
    
if __name__ == "__main__":
    env_id = "LiftCube-v0"
    traj_name = "trajectory.state.pd_ee_delta_pose.h5"
    path = f"data/demos/v0/rigid_body/{env_id}/{traj_name}"
    # path = "data/demos/v0/rigid_body/LiftCube-v0/trajectory.state.pd_ee_delta_pose.h5"
    assert osp.exists(path)
    dataset = ManiSkill2Dataset(path)
    print(len(dataset.episodes))
    dataloader = DataLoader(dataset, batch_size=256, num_workers=0, pin_memory=True, drop_last=True, shuffle=True)
    obs, action = dataset[0]
    print("Observation: ", obs.shape)
    print("Action: ", action.shape)