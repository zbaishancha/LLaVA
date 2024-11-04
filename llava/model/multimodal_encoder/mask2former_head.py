# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from .modules import MaskFormerHead

from PIL import Image
import math

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_res4 = nn.Conv2d(1024, 768, kernel_size=1, stride=1, padding=0)
        self.downsample_res5 = nn.Sequential(
            nn.Conv2d(768, 1536, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(1536),
            nn.GELU()
        )
        self.upsample_res3 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
            nn.BatchNorm2d(384),
            nn.GELU()
        )
        self.upsample_res2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
            nn.GELU()
        )

    def forward(self, x):
        """
            x: b, n, c
        """
        from einops import rearrange
        _, n, _ = x.shape
        h = w = int(math.sqrt(n))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        res4 = self.to_res4(x)
        res5 = self.downsample_res5(res4)
        res3 = self.upsample_res3(res4)
        res2 = self.upsample_res2(res3)

        return {
            "res2": res2,  # b, 192, 96, 96
            "res3": res3,  # b, 384, 48, 48
            "res4": res4,  # b, 768, 24, 24
            "res5": res5   # b, 1536, 12, 12
        }



CKPT = "/mnt/csi-data-aly/user/haozhou/Projects/Mask2Former/output/model_final.pth"

class DistillMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    def __init__(self, ckpt=CKPT):
        super().__init__()
        self.neck = MultiScaleFeatureExtractor()
        self.sem_seg_head = MaskFormerHead()
        
        self.neck.requires_grad_(False)
        self.sem_seg_head.requires_grad_(False)
        
        state_dict = torch.load(ckpt, map_location='cpu')['model']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        assert len(missing) == 0
        
        self.class_embeds = nn.Embedding(20, 256)
        self.class_embeds.requires_grad_(True)
        self.has_class = True
        self.num_k = 16
    
    @property
    def dtype(self):
        return self.class_embeds.weight.dtype

    @property
    def device(self):
        return self.class_embeds.weight.device

    def forward(self, clip_features):
        clip_features = clip_features.to(device=self.device, dtype=self.dtype)
        features = self.neck(clip_features)
        ori_dtype = features['res2'].dtype
        self.sem_seg_head.to(dtype=torch.float32)
        outputs_float = self.sem_seg_head(features)
        outputs = {k: v.to(dtype=ori_dtype) for k, v in outputs_float.items()}
        class_queries_logits = outputs['pred_logits']
        mask_queries = outputs["transformer_decoder_last_hidden_state"]
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        predicted_classes = torch.argmax(masks_classes, dim=-1) 
        confidence_scores = torch.max(masks_classes, dim=-1).values
        _, topk_indices = torch.topk(confidence_scores, k=self.num_k, dim=-1)
        topk_mask_queries = mask_queries.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, mask_queries.size(-1)))
        topk_labels = predicted_classes.gather(1, topk_indices)
        if self.has_class:
            topk_labels_embeds = self.class_embeds(topk_labels)
            topk_mask_queries += topk_labels_embeds
        
        return topk_mask_queries

if __name__ == "__main__":
    model = DistillMaskFormer().cuda().to(dtype=torch.float16)
    x = torch.randn(2, 576, 1024)
    out = model(x)
    print(out.shape)
    print(out.dtype)