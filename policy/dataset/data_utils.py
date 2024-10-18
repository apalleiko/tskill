import sys
import torch
import sklearn.preprocessing as skp
from torchvision.transforms import v2
import h5py
from policy.dataset.masking_utils import get_skill_pad_from_seq_pad

# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def pad2size(items,sz,max_skill_len):
    """Function for padding a dictionary of input items of the same length to a specific length 
    along the first dimension. Also creates a sequence padding mask for the items."""
    num_unpad_seq = items["actions"].shape[0]
    pad = sz - num_unpad_seq
    seq_pad_mask = torch.cat((torch.zeros(num_unpad_seq), torch.ones(pad)), axis=0).to(torch.bool)
    skill_pad_mask = get_skill_pad_from_seq_pad(seq_pad_mask, max_skill_len)
    new_items = dict(seq_pad_mask=seq_pad_mask, skill_pad_mask=skill_pad_mask)
    for k,v in items.items():
        if "goal" in k:
            new_items[k] = v
        else:
            v_pad = torch.zeros([pad] + list(v.shape[1:]))
            v_new = torch.cat((v, v_pad), axis=0).to(torch.float32)
            new_items[k] = v_new
    return new_items


def efficient_collate_fn(batch):
    max_skill_len = 10 #HARDCODED
    collate_fn = torch.utils.data.default_collate
    sizes = [b["actions"].shape[0] for b in batch]

    # Get largest size
    max_seq_len = max(sizes)
    max_seq_len = int(np.ceil(max_seq_len / 10) * 10)
    for i in range(len(batch)):
        batch[i] = pad2size(batch[i], max_seq_len, max_skill_len)
    
    return collate_fn(batch)


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
            self.scaling_fcn_forward = lambda x: self.separate_scaling(self.scaling_for, 
                                                                       x, sep_idx)
            self.scaling_fcn_inverse = lambda x: self.separate_scaling(self.scaling_inv, 
                                                                       x, sep_idx)
        else:
            self.scaling_fcn_forward = lambda x: self.scaling_for(x)
            self.scaling_fcn_inverse = lambda x: self.scaling_inv(x)
    
    def __call__(self, x, mode="forward"):
        if mode=="forward":
            return self.scaling_fcn_forward(x)
        elif mode=="inverse":
            return self.scaling_fcn_inverse(x)
    
    def separate_scaling(self, fcn, x, idx):
        return torch.hstack((fcn(x[:,:idx]), x[:,idx:]))


class DataAugmentation:
    def __init__(self, cfg, cfg_model) -> None:
        self.cfg = cfg
        self.method = cfg["method"]
        self.max_seq_len = cfg["data"]["max_seq_len"]
        self.max_skill_len = cfg_model["max_skill_len"]
        self.cond_dec = cfg_model.get("conditional_decode", True)
        cfg_aug = cfg["data"]["augmentation"]
        self.subsequence_rate = cfg_aug.get("subsequence_rate", 0)
        self.seq_masking_rate = cfg_aug.get("seq_masking_rate", 0)
        self.type_masking_rate = cfg_aug.get("type_masking_rate", 0)
        self.img_aug = cfg_aug.get("image_aug", 0)
        self.input_noise = cfg_aug.get("input_noise", 0)

    def __call__(self, data):
        
        if self.subsequence_rate > 0:
            data = self.subsequence(data)

        if self.seq_masking_rate > 0:
            data = self.seq_masking(data)

        if self.type_masking_rate > 0:
            data = self.type_masking(data)

        if self.input_noise > 0:
            data = self.additive_input_noise(data)

        return data
    
    def additive_input_noise(self, data):
        feat_std = 0.05
        pos_std = 0.01
        # vel_std = 0.005
        
        val = torch.rand(1)
        if self.input_noise > val:
            state_dim = data["state"].shape[-1] // 2
            state_noise = torch.randn(data["state"].shape)
            state_noise[:,:state_dim] = state_noise[:,:state_dim]*pos_std
            # state_noise[:,state_dim:] = state_noise[:,state_dim:]*vel_std
            feat_noise = feat_std*torch.abs(torch.randn(data["img_feat"].shape))
            
            data["state"] = data["state"] + state_noise
            data["img_feat"] = data["img_feat"] + feat_noise

        return data

    def subsequence(self, data):
        """Takes a subsequence of the original data. Assumes no masking in data yet.
        For planning model, resets new goal as last image in the sequence."""
        num_unpad_seq = data["actions"].shape[0]

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
                    data["goal_feat"] = data["img_feat"][num_seq-1:num_seq,...]
                    data["goal_pe"] = data["img_pe"][num_seq-1:num_seq,...]
                else:
                    data["goal"] = data["rgb"][num_seq-1:num_seq,...]

            # Recalculate appropriate masking 
            # (also start from the begining of the seq)
            # num_unpad_skills = torch.ceil(torch.clone(num_seq / self.max_skill_len)).to(torch.int16)
            # data["skill_pad_mask"][num_unpad_skills:] = True
            # data["seq_pad_mask"][num_seq:] = True

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
    