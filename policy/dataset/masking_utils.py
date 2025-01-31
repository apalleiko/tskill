import torch

def get_skill_pad_from_seq_pad(seq_pad, max_skill_len):
    """Functions for generating a skill padding mask from a given sequence padding mask,
    based on the max skill length from the config. Always adds the padding at the end.
        - seq_pad: tensor (seq)
        - max_skill_len: int
    """
    max_num_skills = torch.ceil(torch.tensor(seq_pad.shape[0]/max_skill_len)).to(torch.int)
    num_unpad_seq = torch.sum(torch.logical_not(seq_pad).to(torch.int16))
    num_unpad_skills = torch.ceil(num_unpad_seq / max_skill_len) # get skills with relevant outputs
    skill_pad_mask = torch.zeros(max_num_skills) # True is unattended
    skill_pad_mask[num_unpad_skills.to(torch.int16):] = 1
    skill_pad_mask = skill_pad_mask.to(torch.bool)
    return skill_pad_mask
    

def get_dec_ar_masks(max_num_skills, max_skill_len, num_obs, device='cpu'):
    """ 
    Decoder autoregressive masks
    """
    seq = max_num_skills*max_skill_len
    # tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq, device=device)
    # tgt_mask = ~(torch.eye(max_skill_len).to(torch.bool)) # Only allow self attention for single step prediction
    tgt_mask = torch.ones(seq, seq, device=device).to(torch.bool) # Only allow self attention for single step prediction
    mem_mask = torch.ones(seq, max_num_skills, device=device).to(torch.bool) # Start with everything masked
    for s in range(max_num_skills):
        if num_obs > s+1:
            v = 0
        else:
            v = s+1-num_obs
        act_start = s*max_skill_len
        act_end = (s+1)*max_skill_len 
        mem_mask[act_start:act_end,v:s+1] = False # Unmask action attention to num_obs skills
        tgt_mask[act_start:act_end,v*max_skill_len:s*max_skill_len] = False # Unmask action attention to previous num_obs sets of actions
        tgt_mask[act_start:act_end,act_start:act_end] = torch.nn.Transformer.generate_square_subsequent_mask(max_skill_len, device=device).to(torch.bool)

    return mem_mask, tgt_mask

def get_plan_ar_masks(num_img_feats, max_num_skills, goal_mode, num_obs, device='cpu'):
    """
    Planning autoregressive masks.
    """
    if goal_mode == "image":
        goal_tokens = num_img_feats
    else:
        goal_tokens = 1
    plan_src_len = max_num_skills * (1 + num_img_feats) + goal_tokens # (MNS*(q + img_feats) + goal)
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(max_num_skills, device=device)
    mem_mask = torch.ones(max_num_skills, plan_src_len, device=device).to(torch.bool) # Start with everything masked
    src_mask = torch.ones(plan_src_len, plan_src_len, device=device).to(torch.bool) # Start with everything masked
    src_mask[-goal_tokens:,-goal_tokens:] = False # Unmask goal self attention block
    for s in range(max_num_skills):
        if num_obs > s+1:
            v = 0
        else:
            v = s+1-num_obs
        im_begin = max_num_skills
        im_start = max_num_skills + v*num_img_feats
        im_end = max_num_skills + (s+1)*num_img_feats
        q_begin = 0
        q_start = v
        q_end = s+1
        # Src mask.
        src_mask[s,q_start:q_end] = False # Unmask qpos self attention
        src_mask[im_start:im_end, im_start:im_end] = False # Unmask img features attention block(s)
        src_mask[s,im_start:im_end] = False # Unmask qpos attention to img features
        src_mask[im_start:im_end,q_start:q_end] = False # Unmask img features attention to qpos
        src_mask[im_start:im_end,-goal_tokens:] = False # Unmask img features attention to goal
        src_mask[s,-goal_tokens:] = False # Unmask qpos attention to goal
        # Memory mask
        mem_mask[s,im_start:im_end] = False # Unmask skill attention to img features
        mem_mask[s,-goal_tokens:] = False # Unmask skill attention to goal
        mem_mask[s,q_start:q_end] = False # Unmask skill attention to qpos

    return src_mask, mem_mask, tgt_mask


def get_enc_causal_masks(max_seq_len, max_num_skills, max_skill_len, device='cpu'):
    """
    Gets a causal mask for the encoder. Is only the size of max seq len, so has to be repeated in the encoder itself.
    """
    enc_src_mask = torch.nn.Transformer.generate_square_subsequent_mask(max_seq_len, device=device)
    enc_tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(max_num_skills, device=device)
    enc_mem_mask = torch.ones(max_num_skills, max_seq_len, device=device).to(torch.bool)
    for s in range(max_num_skills):
        sk_start = s*max_skill_len
        sk_end = s*max_skill_len + max_skill_len
        enc_mem_mask[s,:sk_end] = False # Unmask skill attention to prior sequence items
    return enc_src_mask, enc_mem_mask, enc_tgt_mask