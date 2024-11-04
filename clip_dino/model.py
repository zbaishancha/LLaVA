import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, Dinov2Model, AutoImageProcessor
import math
import sys
sys.path.append('.')

from model.clip_dino.modules import Transformer 

transformer_kwargs = {
    "dim": 512,
    "depth": 4,
    "heads": 8,
    "dim_head": 128,
    "mlp_dim": 512 * 4, 
    "dropout": 0.1
}

class DistillDINOv2(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", dino_model_name="facebook/dino-v2", ckpt=None):
        super(DistillDINOv2, self).__init__()

        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.select_layer = -2


        self.dino_image_size = 518
        self._interp_size = self.num_patches = 576
        self.dino_patch_size = 14
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name, 
                                                                 crop_size=dict(height=self.dino_image_size, width=self.dino_image_size), 
                                                                 size=dict(shortest_edge=self.dino_image_size))
        self.dino_model = Dinov2Model.from_pretrained(dino_model_name)
        self.dino_model.eval()
        for param in self.dino_model.parameters():
            param.requires_grad = False
        
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            Transformer(**transformer_kwargs),
            nn.Linear(512, 1024)
        ) # 
        self.rec_loss_type = 'cosine'
        
    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size ** 0.5)
            h = w = int(num_tokens ** 0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features  
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:]
        return image_features

    def calculate_rec_loss(self, rec, target):
        if self.rec_loss_type == 'cosine':
            target = target / target.norm(dim=-1, keepdim=True)
            rec = rec / rec.norm(dim=-1, keepdim=True)
            rec_loss = (1 - (target * rec).sum(-1)).mean()
        else:
            raise NotImplementedError

        return rec_loss
    
    def forward(self, clip_image, dinov2_image):
        with torch.no_grad():
            self.clip_model.eval()
            self.dino_model.eval()
            image_forward_out = self.clip_model(clip_image, output_hidden_states=True)
            clip_features = self.feature_select(image_forward_out)
            target_features = self.dino_model(dinov2_image).last_hidden_state[:, 1:]
            target_features = self.interpolate(target_features)
        
        decoded_features = self.decoder(clip_features)
        loss = self.calculate_rec_loss(decoded_features, target_features)
        return loss, decoded_features