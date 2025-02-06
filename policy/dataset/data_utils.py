import sys
import torch
import sklearn.preprocessing as skp
from torchvision.transforms import v2
import h5py
from policy.dataset.masking_utils import get_skill_pad_from_seq_pad
import numpy as np
from transformers import AutoModel, AutoTokenizer, logging
from hydra.utils import to_absolute_path

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
    max_skill_len = 8 #HARDCODED
    collate_fn = torch.utils.data.default_collate
    sizes = [b["actions"].shape[0] for b in batch]

    # Get largest size
    max_seq_len = max(sizes)
    max_seq_len = int(np.ceil(max_seq_len / max_skill_len) * max_skill_len)
    for i in range(len(batch)):
        batch[i] = pad2size(batch[i], max_seq_len, max_skill_len)
    
    return collate_fn(batch)


class ScalingFunction:
    def __init__(self, scaling, fit_data, sep_idx=None) -> None:
        if isinstance(scaling, (int, float, list, tuple)):
            print("Computing linear scaling")
            self.scaling_for = lambda x: x * torch.tensor(scaling)
            self.scaling_inv = lambda x: x / torch.tensor(scaling)
        elif scaling=="norm":
            print("Computing norm scaling")
            act_mu = torch.mean(fit_data, 0)
            act_std = torch.std(fit_data, 0)
            self.scaling_for = lambda x: (x - act_mu) / torch.sqrt(act_std)
            self.scaling_inv = lambda x: x * torch.sqrt(act_std) + act_mu
        elif scaling=="robust_scaler":
            print("Computing robust scaler")
            scaler = skp.RobustScaler().fit(fit_data)
            self.scaling_for = lambda x: torch.from_numpy(scaler.transform(x.numpy()))
            self.scaling_inv = lambda x: torch.from_numpy(scaler.inverse_transform(x.numpy()))
        elif scaling in ("normal","uniform"):
            print(f"Computing {scaling} quantile transform")
            scaler = skp.QuantileTransformer(output_distribution=scaling)
            scaler.fit(fit_data)
            self.scaling_for = lambda x: torch.from_numpy(scaler.transform(x.numpy()))
            self.scaling_inv = lambda x: torch.from_numpy(scaler.inverse_transform(x.numpy()))
        elif scaling=="power":
            print(f"Computing power transform")
            scaler = skp.PowerTransformer()
            scaler.fit(fit_data)
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
        self.img_aug = cfg_aug.get("image_aug", 0)
        self.input_noise = cfg_aug.get("input_noise", 0)

    def __call__(self, data):
        
        if self.subsequence_rate > 0:
            data = self.subsequence(data)

        if self.seq_masking_rate > 0:
            data = self.seq_masking(data)

        if self.img_aug > 0:
            data = self.image_aug(data)

        if self.input_noise > 0:
            data = self.additive_input_noise(data)

        return data
    
    def additive_input_noise(self, data):
        # feat_std = 0.00
        pos_std = 0.002
        act_std = 0.02
        
        val = torch.rand(1)
        if self.input_noise > val:
            state_noise = pos_std * torch.randn(data["state"].shape)
            # feat_noise = feat_std*torch.abs(torch.randn(data["img_feat"].shape))
            act_noise = act_std*torch.randn(data["actions"].shape)

            data["state"] = data["state"] + state_noise
            # data["img_feat"] = data["img_feat"] + feat_noise
            data["actions"] = data["actions"] + act_noise

        return data

    def subsequence(self, data):
        """Takes a subsequence of the original data. Assumes no masking in data yet.
        For planning model, resets new goal as last image in the sequence."""
        num_unpad_seq = data["actions"].shape[0]

        val = torch.rand(1)
        if self.subsequence_rate > val:
            # Uniformly sample how much of the sequence to use for the batch
            # from 1 to entire (unpadded) seq
            num_seq = torch.randint(1, num_unpad_seq-5, (1,1)).squeeze()
            
            # Start at random point in the sequence
            for k,v in data.items():
                if "mask" not in k and "goal" not in k:
                    data[k] = v[num_seq:,...]

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
    

def get_task_embs(task_embedding_format, descriptions):
    logging.set_verbosity_error()
    if task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    elif task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True)
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
    return task_embs
    