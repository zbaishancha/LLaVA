import torch
import torch.nn as nn
import os
import json
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoTokenizer, CLIPTextModel
from safetensors.torch import load_file

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)

class CrossModalAttention(nn.Module):
    def __init__(self, vision_dim, text_dim, top_k_ratio, temperature=1.0):
        super(CrossModalAttention, self).__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.top_k_ratio = top_k_ratio
        self.text_projection = nn.Linear(text_dim, vision_dim)
        self.temperature = temperature
    
    def forward(self, image_features, text_embedding):

        projected_text = self.text_projection(text_embedding)
        projected_text = projected_text.unsqueeze(1)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        projected_text /= projected_text.norm(dim=-1, keepdim=True)
        
        # * projected_text.size(-1) ** -0.5
        attention_scores = (image_features @ projected_text.transpose(-2, -1)) 
        attention_scores = attention_scores.squeeze(-1)
        attention_scores = attention_scores / self.temperature
        attention_scores = F.softmax(attention_scores, dim=-1)

        top_k = int(image_features.size(1) * self.top_k_ratio)
         
        top_k_values, top_k_indices = torch.topk(attention_scores, top_k, dim=-1)
        
        mask = torch.zeros_like(attention_scores, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top_k_indices, src=torch.ones_like(top_k_indices, dtype=torch.bool))

        selected_image_features = image_features[mask].view(-1, top_k, self.vision_dim)

        return selected_image_features, mask


class CLIPTextTower(nn.Module):
    def __init__(self, text_tower, args, **kwargs) -> None:
        super().__init__()
        self.is_loaded = False
        self.text_tower = text_tower
        config_path = os.path.join(text_tower, "config.json")
        assert os.path.exists(config_path)

        with open(config_path, 'r') as file:
            self.config = json.load(file)
        
        self.vision_dim = self.config["vision_config_dict"]["hidden_size"]
        self.text_dim = self.config["text_config_dict"]["hidden_size"]
    
    def load_model(self, model_args=None, device_map=None, model_path=None, top_k_ratio=None, temperature=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.text_tower))
            return
        if model_args is not None:
            top_k_ratio = getattr(model_args, "top_k_ratio", 0.75)
            temperature = getattr(model_args, "temperature", 0.05)

        self.text_tower = CLIPTextModel.from_pretrained(self.text_tower, device_map=device_map)
        self.text_tower.requires_grad_(False)
        self.feature_select_module = CrossModalAttention(self.vision_dim, self.text_dim, top_k_ratio, temperature)
        self.feature_select_module.requires_grad_(True)
        self.is_loaded = True
        if model_path is not None:
            state_dict = load_file(os.path.join(model_path, "model-00003-of-00003.safetensors"))
            state_dict_real = {
                k.replace('model.text_tower.', ''): v
                for k, v in state_dict.items()
            }
            missing, unexpected = self.load_state_dict(state_dict_real, strict=False)
            assert len(missing) ==0 

    def forward(self, input_ids, image_features):
        outputs = self.text_tower(input_ids)
        pooled_output = outputs.pooler_output
        selected_image_features, _ = self.feature_select_module(image_features, pooled_output)

        return selected_image_features