import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from policy.dataset.data_utils import load_h5_data
from policy.dataset.masking_utils import get_dec_ar_masks, get_enc_causal_masks, get_plan_ar_masks, get_skill_pad_from_seq_pad


def tensor_to_numpy(x):
    # moves all tensors to numpy. This is just for SB3 as SB3 does not optimize for observations stored on the GPU.
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return x


def convert_observation(observation):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images
    # we provide a simple tool to flatten dictionaries with state data
    state = np.hstack(
            [observation["joint_states"],
             observation["gripper_states"]])
    obs = dict(state=state)

    # image data is not scaled here and is kept as uint16 to save space
    cams = []
    # combine the RGB and depth images
    cams.append(observation["agentview_rgb"][:,::-1,:])
    cams.append(observation["eye_in_hand_rgb"])

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
        obs = convert_observation(trajectory["obs"])
        
        actions = self.action_scaling(torch.from_numpy(trajectory["actions"])[i0:,:].float()) # (seq, act_dim)
        state = self.state_scaling(torch.from_numpy(obs["state"])[i0:,:].float()) # (seq, state_dim)

        if "resnet18" in trajectory["obs"].keys():
            use_precalc = True
            img_feat = torch.from_numpy(trajectory["obs"]["resnet18"]["img_feat"][i0:,...]) # (seq, num_cams, h*w, c)
            img_pe =  torch.from_numpy(trajectory["obs"]["resnet18"]["img_pe"][i0:,...]) # (seq, num_cams, h*w, hidden)
            num_cam = img_feat.shape[1]
            num_feats = img_feat.shape[2]
        else:
            use_precalc = False
            rgb = rescale_rgbd(obs["rgb"], separate_cams=True)
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

        # Add extra dimension for batch size as model expects this.
        if self.add_batch_dim:
            for k,v in data.items():
                data[k] = v.unsqueeze(0)

        return data
