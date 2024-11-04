import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .modules import Transformer 

transformer_kwargs = {
    "dim": 512,
    "depth": 4,
    "heads": 8,
    "dim_head": 128,
    "mlp_dim": 512 * 4, 
    "dropout": 0.1
}

CKPT = "/mnt/csi-data-aly/shared/public/openpilot_deepdive/haozhou/image_tokenizer_exp/runs/clip_dinov2_head/epoch_5.pth"

class DistillDINOv2(nn.Module):
    def __init__(self, ckpt=CKPT):
        super(DistillDINOv2, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            Transformer(**transformer_kwargs),
            nn.Linear(512, 1024)
        )
        state_dict = torch.load(ckpt, map_location='cpu')
        state_dict_real = {
            k.replace('module.', ''): v
            for k, v in state_dict.items()
        }
        missing, unexpected = self.load_state_dict(state_dict_real, strict=False)
        assert len(missing) == 0
        self.decoder.requires_grad_(False)
    
    @property
    def dtype(self):
        return next(self.decoder.parameters()).dtype

    @property
    def device(self):
        return next(self.decoder.parameters()).device
    
    @torch.no_grad()
    def forward(self, clip_features):
        self.decoder.eval()
        decoded_features = self.decoder(clip_features.to(device=self.device, dtype=self.dtype))
        return decoded_features


if __name__ == "__main__":
    model = DistillDINOv2().cuda().to(dtype=torch.float16)
    x = torch.randn(2, 576, 1024)
    out = model(x)
    print(out.shape)
    print(out.dtype)