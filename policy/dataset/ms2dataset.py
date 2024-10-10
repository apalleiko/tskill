import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mani_skill2.utils.io_utils import load_json
from policy.dataset.data_utils import load_h5_data
from policy.dataset.masking_utils import get_skill_pad_from_seq_pad


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

    
class ManiSkillrgbSeqDataset(Dataset):
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
        else:
            use_precalc = False
            rgbd = obs["rgbd"][:-1]
            rgb = rescale_rgbd(rgbd, discard_depth=True, separate_cams=True)
            rgb = torch.from_numpy(rgb).float().permute((0, 4, 3, 1, 2))[i0:,...] # (seq, num_cams, channels, img_h, img_w)

        if self.method == "plan":
            if use_precalc:
                data["goal_feat"] = img_feat[-1:,...]
                data["goal_pe"] = img_pe[-1:,...]
            else:
                data["goal"] = rgb[-1:,...]

        # Add padding to sequences to match lengths and generate padding masks
        if self.pad:
            num_unpad_seq = actions.shape[0]
            if self.pad2msl:
                pad = self.max_skill_len - (num_unpad_seq % self.max_skill_len)
            else:
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

        # Add extra dimension for batch size if not using dataloader as model expects this.
        if self.add_batch_dim:
            for k,v in data.items():
                data[k] = v.unsqueeze(0)

        return data
    

    def from_obs(self, obs):
        # Obtain observation data in the proper form
        o = convert_observation(obs, pos_only=False)
        # State
        state = self.state_scaling(torch.from_numpy(o["state"]).unsqueeze(0)).float().unsqueeze(0) # (1 (bs), 1 (seq), state_dim)
        # Image
        rgbd = o["rgbd"]
        rgb = rescale_rgbd(rgbd, discard_depth=True, separate_cams=True)
        rgb = torch.from_numpy(rgb).float().permute((3, 2, 0, 1)).unsqueeze(0).unsqueeze(0) # (1 (bs), 1 (seq), num_cams, channels, img_h, img_w)

        data = dict()
        data["rgb"] = rgb
        data["state"] = state

        return data


### Commands for trajectory replay ###
# python -m mani_skill2.trajectory.replay_trajectory   --traj-path /home/mrl/Documents/Projects/tskill/data/demos/v0/rigid_body/PegInsertionSide-v0/trajectory.h5   --save-traj --target-control-mode pd_joint_delta_pos --obs-mode rgbd --num-procs 10
# python -m policy/dataset/replay_trajectory.py --traj-path data/demos/v0/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_joint_delta_pos --num-procs 2 --cam-res 480