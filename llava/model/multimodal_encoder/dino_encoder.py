import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import math
import json
from transformers import Dinov2Model, AutoImageProcessor, Dinov2Config

from .base_encoder import BaseVisionTower


def extract_res_interp(model_name):
    valid_model_prefixes = [
        "facebook/dinov2-small",
        "facebook/dinov2-base",
        "/mnt/csi-data-aly/shared/public/haozhou/checkpoints/dinov2-large",
        "facebook/dinov2-giant-imagenet1k-1-layer",
        "facebook/dinov2-giant",
    ]

    for prefix in valid_model_prefixes:
        if model_name.startswith(prefix):
            base_model_name = prefix
            break
    else:
        raise ValueError(f"Unknown vision tower: {model_name}")

    res = None
    interp = None

    parts = model_name[len(base_model_name):].split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp


class SelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here.
    """

    def __init__(self, config):
        super().__init__()
        
        self.n_embd = 1024
        self.n_head = 16
        self.head_dim = self.n_embd // self.num_heads
        self.dropout = 0.1
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        
        # regularization
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.dropout = 0.1
        
        # support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        C = int(C)
        T = int(T)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        if self.flash:
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        else:
            k = k.view(B, T, self.n_head, C // self.n_head).permute((0, 2, 3, 1)) # (B, nh, hs, T)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # Full attention set is_causal as False
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                 attn_mask=None, 
                                                                 dropout_p=self.dropout if self.training else 0., 
                                                                 is_causal=False)
        else:
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k) * (1.0 / math.sqrt(q.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CrossModalAttention(nn.Module):
    def __init__(self, config):
        super(CrossModalAttention, self).__init__()

        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = 0.1
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(self.embed_dim)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def forward(self, image_features, text_embedding):
        B, T, C = image_features.size() # batch size, sequence length, embedding dimensionality (n_embd)
        C = int(C)
        T = int(T)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(image_features)
        k = self.k_proj(text_embedding)
        v = self.v_proj(text_embedding)
        if self.flash:
            k = k.view(B, k.size(1), self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        else:
            k = k.view(B, T, self.num_heads, C // self.num_heads).permute((0, 2, 3, 1)) # (B, nh, hs, T)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, v.size(1), self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # Full attention set is_causal as False
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                 attn_mask=None, 
                                                                 dropout_p=self.dropout if self.training else 0., 
                                                                 is_causal=False)
        else:
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k) * (1.0 / math.sqrt(q.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.out_proj(y))
        
        return y

class DinoVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=True):
        super(DinoVisionTower, self).__init__(vision_tower, args, delay_load)

        """try to extract an image resolution and interpolation res from the model name string

        valid model names:
            facebook/dinov2-small
            facebook/dinov2-base
            facebook/dinov2-large
            facebook/dinov2-giant
            facebook/dinov2-giant-imagenet1k-1-layer

        res pattern: <model_name>-res<res>-interp<interp>

        eg: facebook/dinov2-small-res518-interp224
        """

        # extract image resolution from model name
        # if self.vision_tower_name.startswith("facebook/dinov2-"):
        #     parts = self.vision_tower_name.split("-res")
        #     if len(parts) == 2:
        #         self._image_size = int(parts[1])
        #         logger.warning(f"Extracted image size {self._image_size} from passed model name '{self.vision_tower_name}' (using {parts[0]})")
        #         self.vision_tower_name = parts[0]
        #     else:
        #         self._image_size = None  # default is 518px
        base_model_name, res, interp = extract_res_interp(self.vision_tower_name)
        self._vision_tower_name = vision_tower
        self.vision_tower_name = base_model_name
        self._image_size = res
        self._interp_size = 576
        self._patch_size = 14  # default patch size

        if not self.delay_load:
            self.load_model()
        else:
            self.cfg_only = Dinov2Config.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None, model_path=None, feature_fusion_strategy='one-cross'):

        self.vision_tower = Dinov2Model.from_pretrained(self.vision_tower_name)
        """ValueError: Dinov2Model does not support `device_map='auto'`. To implement support, the model class needs to implement the `_no_split_modules` attribute."""
        self.vision_tower._no_split_modules = ["Dinov2SwiGLUFFN"]

        _image_size = self.vision_tower.config.image_size
        if self._image_size is None:
            self._image_size = _image_size
        else:
            logger.warning(f"Overriding DinoVisionTower image size of {_image_size} with {self._image_size}")

        # increase shortest edge to prevent edge case crops
        default_shortest_ratio = 8/7  # 224/256
        # shortest_edge = int(default_shortest_ratio * self._image_size)
        shortest_edge = self._image_size

        processor = AutoImageProcessor.from_pretrained(self.vision_tower_name, crop_size=dict(height=self._image_size, width=self._image_size), size=dict(shortest_edge=shortest_edge))

        self.image_processor = processor

        # Assign the output channels of the projection convolution as the hidden size
        self._hidden_size = self.vision_tower.embeddings.patch_embeddings.projection.out_channels
        # Assign the first value of the stride of the projection convolution as the patch size
        self._patch_size = self.vision_tower.embeddings.patch_embeddings.projection.stride[0]

        #print(self._hidden_size, self._patch_size)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.text_projection = nn.Linear(4096, 1024)
        self.query_projection = nn.Linear(256, 1024)
        self.query_projection.requires_grad_(True)
        self.text_projection.requires_grad_(True)
        
        self.feature_fusion_strategy = feature_fusion_strategy
        
        if self.feature_fusion_strategy == 'cat':
            self.prompt_module = nn.Identity()

        elif self.feature_fusion_strategy == 'series-connection-cross':
            self.prompt_module = nn.ModuleList([CrossModalAttention(self.vision_tower_name),
                                               CrossModalAttention(self.vision_tower_name),
                                               CrossModalAttention(self.vision_tower_name)])
            self.prompt_module.requires_grad_(True)
        elif self.feature_fusion_strategy == 'parallel-connection-cross':
            self.prompt_module_text = CrossModalAttention(self.vision_tower_name)
            self.prompt_module_prompt = CrossModalAttention(self.vision_tower_name)
            self.prompt_module_object = CrossModalAttention(self.vision_tower_name)
            self.prompt_module_text.requires_grad_(True)
            self.prompt_module_prompt.requires_grad_(True)
            self.prompt_module_object.requires_grad_(True)
        elif self.feature_fusion_strategy == 'one-cross':
            self.prompt_module = CrossModalAttention(self.vision_tower_name)
            self.prompt_module.requires_grad_(True)
        
        
        self.is_loaded = True
        if model_path is not None and os.path.exists(os.path.join(model_path, "model-00003-of-00004.safetensors")):
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(model_path, "model-00003-of-00004.safetensors"))
            state_dict_real = {
                k.replace('model.prompt_tower.', ''): v
                for k, v in state_dict.items()
            }
            missing, unexpected = self.load_state_dict(state_dict_real, strict=False)
            assert len(missing) ==0

    @property
    def image_size(self):
        return self._image_size

    def feature_select(self, outputs):
        sequence_output = outputs["last_hidden_state"]  # batch_size, sequence_length, hidden_size

        if self.select_feature == 'cls_patch':
            image_features = sequence_output
        elif self.select_feature == 'patch':
            image_features = sequence_output[:, 1:]
        elif self.select_feature == 'cls':
            image_features = sequence_output[:, 0]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

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

    def _forward(self, images):
        # logger.warning(f"images shape: {images.shape}")
        self.vision_tower.eval()
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower.forward(images.to(device=self.device, dtype=self.dtype))
            # logger.warning(f"image_forward_outs shape: {image_forward_outs['last_hidden_state'].shape}")
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # logger.warning(f"image_features shape: {image_features.shape}")
            interp_features = self.interpolate(image_features)
            # logger.warning(f"interp_features shape: {interp_features.shape}")
            return interp_features
    
    def forward(self, images, input_embeds, image_features, object_queries):
        prompt_image_features = self._forward(images) # B, N, D
        text_embedding = self.text_projection(input_embeds) # B, N, D
        queries_embedding = self.query_projection(object_queries) # B, N, D
        
        if self.feature_fusion_strategy == 'one-cross':
            prompt_features = torch.cat([image_features, prompt_image_features, text_embedding, queries_embedding], dim=1)
            image_features = image_features + self.prompt_module(image_features, prompt_features)
        
        elif self.feature_fusion_strategy == 'cat':
            prompt_features = torch.cat([image_features, prompt_image_features, text_embedding, queries_embedding], dim=1)
            image_features = self.prompt_module(prompt_features)
        
        elif self.feature_fusion_strategy == 'series-connection-cross':
            feature_list = [prompt_image_features, queries_embedding, text_embedding]
            for fusion_module, feature in zip(self.prompt_module, feature_list):
                image_features = image_features + fusion_module(image_features, torch.cat([image_features, feature], dim=1))
        
        elif self.feature_fusion_strategy == 'parallel-connection-cross':
            image_features = image_features + \
                                self.prompt_module_object(image_features, torch.cat([image_features, queries_embedding], dim=1)) + \
                                    self.prompt_module_prompt(image_features, torch.cat([image_features, prompt_image_features], dim=1)) + \
                                        self.prompt_module_text(image_features, torch.cat([image_features, text_embedding], dim=1))
        
        return image_features


    @property
    def num_patches_per_side(self):
        return int(self.num_patches ** 0.5)

    @property
    def num_patches(self):
        if self._interp_size is None:
            return (self._image_size // self._patch_size) ** 2
        else:
            return self._interp_size
