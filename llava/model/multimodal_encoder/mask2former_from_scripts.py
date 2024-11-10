# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput

from .modules import MaskFormerHead
from .backbone.swin import SwinTransformer


CKPT = "/mnt/csi-data-aly/user/haozhou/Projects/Mask2Former/output_swin_baseline/model_final.pth"

class MaskFormerScripts(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    def __init__(self, ckpt=CKPT):
        super().__init__()
        self.backbone = SwinTransformer()
        self.sem_seg_head = MaskFormerHead()
        state_dict = torch.load(ckpt, map_location='cpu')['model']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        assert len(missing) == 0
    
    @property
    def dtype(self):
        return next(self.backbone.parameters()).dtype

    @property
    def device(self):
        return next(self.backbone.parameters()).device
    
    def forward(self, images):
        features = self.backbone(images.to(device=self.device, dtype=self.dtype))
        ori_dtype = features['res2'].dtype
        self.sem_seg_head.to(dtype=torch.float32)
        outputs_float = self.sem_seg_head(features)
        outputs = {k: v.to(dtype=ori_dtype) for k, v in outputs_float.items()}
        outputs_transformers = Mask2FormerForUniversalSegmentationOutput()
        outputs_transformers.class_queries_logits = outputs['pred_logits']
        outputs_transformers.transformer_decoder_last_hidden_state = outputs["transformer_decoder_last_hidden_state"]
        return outputs_transformers

if __name__ == "__main__":
    model = MaskFormerScripts().cuda().to(dtype=torch.float16)
    x = torch.randn(2, 3, 384, 384).cuda().to(dtype=torch.float16)
    out = model(x)
    print(out)