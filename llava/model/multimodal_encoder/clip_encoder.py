import torch
import torch.nn as nn
import os
import json
import math
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoTokenizer, CLIPModel, CLIPTextModel
from safetensors.torch import load_file

from llava.model.multimodal_encoder.dinov2_head import DistillDINOv2
from llava.model.multimodal_encoder.mask2former_head import DistillMaskFormer
from llava.model.multimodal_encoder.modules import CrossModalAttention

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

        self.dinov2_head = DistillDINOv2()
        self.dinov2_head.requires_grad_(False)
        self.mask2former_head = DistillMaskFormer()
        self.mask2former_head.requires_grad_(False)      
          
        self.text_projection = nn.Linear(4096, 1024)
        self.query_projection = nn.Linear(256, 1024)
        self.query_projection.requires_grad_(True)
        self.text_projection.requires_grad_(True)
        self.prompt_module = CrossModalAttention()
        self.prompt_module.requires_grad_(True)
        
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

    def forward(self, images, inputs_embeds):
        with torch.no_grad():
            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
        prompt_image_features = self.dinov2_head(image_features)
        object_queries = self.mask2former_head(image_features)
        text_embedding = self.text_projection(inputs_embeds) # B, N, D
        queries_embedding = self.query_projection(object_queries) # B, N, D
        prompt_features = torch.cat([image_features, prompt_image_features, text_embedding, queries_embedding], dim=1)
        image_features = image_features + self.prompt_module(image_features, prompt_features)
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

class CLIPTextTower(nn.Module):
    def __init__(self, text_tower, args, **kwargs) -> None:
        super().__init__()
        self.is_loaded = False
        self.text_tower = text_tower
        config_path = os.path.join(text_tower, "config.json")
        assert os.path.exists(config_path)

        with open(config_path, 'r') as file:
            self.config = json.load(file)
        
        self.vision_embed_dim = self.config["vision_config_dict"]["hidden_size"]
        self.text_embed_dim = self.config["text_config_dict"]["hidden_size"]
        self.projection_dim = self.config['projection_dim']
        
    def load_model(self, model_args=None, device_map=None, model_path=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.text_tower))
            return

        # self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        # self.visual_projection.requires_grad_(True)
        self.text_projection = nn.Linear(4096, self.vision_embed_dim)
        self.text_projection.requires_grad_(True)

        # if os.path.exists(os.path.join(self.text_tower, "pytorch_model.bin")):
        #     state_dict = torch.load(os.path.join(self.text_tower, "pytorch_model.bin"))
        #     state_dict_real = {"visual_projection.weight": state_dict['visual_projection.weight'],
        #                        "text_projection.weight": state_dict['text_projection.weight']}
        #     missing, unexpected = self.load_state_dict(state_dict_real, strict=False)
        #     assert len(missing) == 0

        # self.text_model = CLIPTextModel.from_pretrained(self.text_tower, device_map=device_map)
        # self.text_model.requires_grad_(False)
        
        self.question_aware_module = CrossModalAttention(self.config)
        self.question_aware_module.requires_grad_(True)

        if model_path is not None:
            state_dict = load_file(os.path.join(model_path, "model-00003-of-00003.safetensors"))
            state_dict_real = {
                k.replace('model.text_tower.', ''): v
                for k, v in state_dict.items()
            }
            missing, unexpected = self.load_state_dict(state_dict_real, strict=False)
            assert len(missing) ==0 
        self.is_loaded = True

    def forward(self, input_embeds, image_features):
        # outputs = self.text_model(input_ids)
        # outputs = input_embeds
        ori_img_features = image_features.clone()
        text_embedding = input_embeds
        text_embedding = self.text_projection(text_embedding)
        # image_features = self.visual_projection(image_features)
        image_features = ori_img_features + self.question_aware_module(image_features, text_embedding)
        return image_features