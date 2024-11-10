# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from .modules import MaskFormerHead

from PIL import Image
import math


def get_norm(norm, num_features):
    return nn.BatchNorm2d(num_features)
class FeaturePyramid(nn.Module):
    def __init__(self, in_channels=1024, out_channels_list=[192, 384, 768, 1536]):
        super(FeaturePyramid, self).__init__()
        self.stages = nn.ModuleList()

        # Scale factors to match the output dimensions for res2, res3, res4, and res5
        scale_factors = [4.0, 2.0, 1.0, 0.5]
        strides = [4, 2, 1, 0.5]  # strides matching each output resolution

        for idx, scale in enumerate(scale_factors):
            out_dim = in_channels  # Start from input channels (1024)
            out_channels = out_channels_list[idx]

            if scale == 4.0:  # Upsample for res2
                layers = [
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                    get_norm("batch", in_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2),
                ]
                out_dim = in_channels // 4
            elif scale == 2.0:  # Upsample for res3
                layers = [nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)]
                out_dim = in_channels // 2
            elif scale == 1.0:  # Identity for res4
                layers = []  # Identity layer (no transformation for res4)
            elif scale == 0.5:  # Downsample for res5
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            # Add 1x1 conv + norm + 3x3 conv with the required output channels
            layers.extend(
                [
                    nn.Conv2d(out_dim, out_channels, kernel_size=1, bias=False),
                    get_norm("batch", out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    get_norm("batch", out_channels),
                ]
            )
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x):
        from einops import rearrange
        _, n, _ = x.shape
        h = w = int(math.sqrt(n))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        res2 = self.stages[0](x)  # b, 192, 96, 96
        res3 = self.stages[1](x)  # b, 384, 48, 48
        res4 = self.stages[2](x)  # b, 768, 24, 24
        res5 = self.stages[3](x)  # b, 1536, 12, 12

        return {"res2": res2, "res3": res3, "res4": res4, "res5": res5}

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
        self.neck = FeaturePyramid()
        self.sem_seg_head = MaskFormerHead()
        
        self.neck.requires_grad_(False)
        self.sem_seg_head.requires_grad_(False)
        
        # state_dict = torch.load(ckpt, map_location='cpu')['model']
        # missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # assert len(missing) == 0
        self.num_k = 16
    
    @property
    def dtype(self):
        return next(self.neck.parameters()).dtype

    @property
    def device(self):
        return next(self.neck.parameters()).device

    @torch.no_grad()
    def forward(self, clip_features):
        self.eval()
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
        
        return topk_mask_queries, topk_labels

if __name__ == "__main__":
    model = DistillMaskFormer().cuda().to(dtype=torch.float16)
    x = torch.randn(2, 576, 1024)
    out = model(x)
    print(out.shape)
    print(out.dtype)