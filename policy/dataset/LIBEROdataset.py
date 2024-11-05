import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from policy.dataset.data_utils import load_h5_data, pad2size


def tensor_to_numpy(x):
    # moves all tensors to numpy. This is just for SB3 as SB3 does not optimize for observations stored on the GPU.
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return x


def convert_observation(observation):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images
    # we provide a simple tool to flatten dictionaries with state data

    state = observation["robot_states"]
    obs = dict(state=state)

    # image data is not scaled here and is kept as uint16 to save space
    cams = []
    # combine the RGB and depth images
    cams.append(observation["obs"]["agentview_rgb"][:,::-1,:])
    cams.append(observation["obs"]["eye_in_hand_rgb"])

    rgb = np.concatenate(cams, axis=-1)
    obs["rgb"] = rgb
        
    return obs


def convert_realtime_observation(observation):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images
    # we provide a simple tool to flatten dictionaries with state data
    
    state = np.hstack(
            [observation["robot0_gripper_qpos"],
             observation["robot0_eef_pos"],
             observation["robot0_eef_quat"]])
    obs = dict(state=state)

    # image data is not scaled here and is kept as uint16 to save space
    cams = []
    # combine the RGB and depth images
    cams.append(observation["agentview_image"][:,::-1,:])
    cams.append(observation["robot0_eye_in_hand_image"])

    rgb = np.concatenate(cams, axis=-1)
    obs["rgb"] = rgb
    
    return obs


def rescale_rgbd(rgbd, separate_cams=False):
    # rescales rgbd data and changes them to floats
    rgbs = []
    for i in range(int(rgbd.shape[-1]/3)):
        rgbs.append(rgbd[..., 3*i:3*i+3] / 255.0)
    
    if separate_cams:
        rgbd = np.stack(rgbs, axis=-1)
    else:
        rgbd = np.concatenate(rgbs, axis=-1)
 
    return rgbd
    
    
class LiberoDataset(Dataset):
    """Class that organizes a single libero demo dataset into distinct rgb sequences
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
        self.episodes = self.data["data"]
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
        self.pad2msl = kwargs.get("pad2msl",False)
        if self.pad2msl:
            print("Padding only to a multiple of max_skill_len. Ensure batch size is 1!")

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
            eps = self.owned_indices[idx]
            i0 = 0
        else:
            mp = self.owned_indices[idx]
            eps = mp[0]
            i0 = mp[1]

        data = dict()

        trajectory = self.episodes[f"demo_{eps}"]
        trajectory = load_h5_data(trajectory)

        # convert the original raw observation with our batch-aware function
        obs = convert_observation(trajectory)
        
        actions = self.action_scaling(torch.from_numpy(trajectory["actions"])[i0:,:].float()) # (seq, act_dim)
        state = self.state_scaling(torch.from_numpy(obs["state"])[i0:,:].float()) # (seq, state_dim)

        if "resnet18" in trajectory["obs"].keys():
            use_precalc = True
            img_feat = torch.from_numpy(trajectory["obs"]["resnet18"]["img_feat"][i0:,...]) # (seq, num_cams, h*w, c)
            img_pe =  torch.from_numpy(trajectory["obs"]["resnet18"]["img_pe_512"][i0:,...]) # (seq, num_cams, h*w, hidden)
            # img_pe2 =  torch.from_numpy(trajectory["obs"]["resnet18"]["img_pe"][i0:,...]) # (seq, num_cams, h*w, hidden) #BUG
        else:
            use_precalc = False
            rgb = rescale_rgbd(obs["rgb"], separate_cams=True)
            rgb = torch.from_numpy(rgb).float().permute((0, 4, 3, 1, 2))[i0:,...] # (seq, num_cams, channels, img_h, img_w)

        if self.method == "plan":
            if use_precalc:
                data["goal_feat"] = img_feat[-1:,...]
                data["goal_pe"] = img_pe[-1:,...]
                # data["goal_pe_plan"] = img_pe2[-1:,...]
            else:
                data["goal"] = rgb[-1:,...]

        # Add padding to sequences to match lengths and generate padding masks
        if use_precalc:
            inputs = dict(state=state,actions=actions,img_feat=img_feat,img_pe=img_pe,) # img_pe_plan=img_pe2) #BUG
        else:
            inputs = dict(state=state,actions=actions,rgb=rgb)
            
        num_unpad_seq = actions.shape[0]
        if self.pad:
            if self.pad2msl:
                sz = num_unpad_seq - (num_unpad_seq % self.max_skill_len) + self.max_skill_len
            else:
                sz = self.max_seq_len
            
            inputs = pad2size(inputs, sz, self.max_skill_len)

        data.update(inputs)

        # Some augmentation assumes masking
        if self.augmentation is not None:
            data = self.augmentation(data)

        # Add extra dimension for batch size as model expects this.
        if self.add_batch_dim:
            for k,v in data.items():
                data[k] = v.unsqueeze(0)

        return data
    
    def from_obs(self, obs):
        # Obtain observation data in the proper form
        o = convert_realtime_observation(obs)
        # State
        state = self.state_scaling(torch.from_numpy(o["state"]).unsqueeze(0)).float().unsqueeze(0) # (1 (bs), 1 (seq), state_dim)
        # Image
        rgb = o["rgb"]
        rgb = rescale_rgbd(rgb, separate_cams=True)
        rgb = torch.from_numpy(rgb).float().permute((3, 2, 0, 1)).unsqueeze(0).unsqueeze(0) # (1 (bs), 1 (seq), num_cams, channels, img_h, img_w)

        data = dict()
        data["rgb"] = rgb
        data["state"] = state

        return data