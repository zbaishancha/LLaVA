import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from .dinov2_head import DistillDINOv2
from .mask2former_head import DistillMaskFormer


class CustomModelConfig(PretrainedConfig):
    model_type = "custom_model_with_two_heads"
    
    def __init__(self, dinov2_dim=512, mask2former_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.dinov2_dim = dinov2_dim
        self.mask2former_dim = mask2former_dim


class EfficientHead(PreTrainedModel):
    config_class = CustomModelConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.dinov2_head = DistillDINOv2()
        self.mask2former_head = DistillMaskFormer()
        self.post_init()

    @torch.no_grad()
    def forward(self, x):
        self.eval()
        decoded_features = self.dinov2_head(x)
        topk_mask_queries, topk_labels = self.mask2former_head(x)
        return decoded_features, topk_mask_queries, topk_labels


if __name__ == "__main__":
    model = EfficientHead.from_pretrained("/mnt/csi-data-aly/user/haozhou/Projects/LLaVA/pretrained")
    # 示例输入
    x = torch.randn(2, 576, 1024)
    output = model(x)
    print(output)