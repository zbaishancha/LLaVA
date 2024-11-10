import torch

dinov2_ckpt = "/mnt/csi-data-aly/shared/public/openpilot_deepdive/haozhou/image_tokenizer_exp/runs/clip_dinov2_head/epoch_5.pth"
mask2former_ckpt = "/mnt/csi-data-aly/user/haozhou/Projects/Mask2Former/output/model_final.pth"


def merge_checkpoints(ckpt_path1, ckpt_path2, output_path):

    ckpt1 = torch.load(ckpt_path1, map_location="cpu")
    ckpt2 = torch.load(ckpt_path2, map_location="cpu")['model']
    

    state_dict_dinov2 = {
        k.replace('module.', 'dinov2_head.'): v
        for k, v in ckpt1.items() if 'module.decoder.' in k
    }
    
    state_dict_mask2former = {
        ('mask2former_head.' + k): v
        for k, v in ckpt2.items() if 'neck.' in k or 'sem_seg_head.' in k
    }

    merged_state_dict = state_dict_dinov2.copy()
    merged_state_dict.update(state_dict_mask2former)
    
    torch.save(merged_state_dict, output_path)
    print(f"Merged checkpoint saved at {output_path}")


output_path = "pretrained/pytorch_model_v2.bin"

merge_checkpoints(dinov2_ckpt, mask2former_ckpt, output_path)