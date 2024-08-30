import torch
import torch.nn as nn
import os
import json
from safetensors.torch import load_file
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .clip_qavit import InstructCLIPVisionModel

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

class QACLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        
        self.vision_tower_name = vision_tower
        self.select_layer = -2
        
        self.clip_type = 'qavit'

        self.instruction_dim = 4096

        self.integration_point = 'late'
        
        if not delay_load:
            self.load_model(args=args)
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model(args=args)

    def load_model(self, device_map=None, args=None, model_base=None):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        if self.clip_type == 'qavit':
            # # Supporting different fusions
            config = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self.vision_tower = InstructCLIPVisionModel(config=config, instruction_dim=self.instruction_dim,
                                                        integration_point=self.integration_point)
            weights = torch.load(self.vision_tower_name + '/pytorch_model.bin')
            pretrained_dict = dict()
            for k,v in weights.items():
                if 'vision_model.embeddings.position_ids' in k:
                    continue
                if 'vision_model' in k:
                    pretrained_dict[k] = v
            missing, unexpected = self.vision_tower.load_state_dict(pretrained_dict, strict=False)
            assert len(unexpected) == 0     # asserts that loading weights was as expected
            self.vision_tower.init_qavit_comps()
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        if args is not None and getattr(args, "mm_instruct_pretrain", None) is not None:
            instruct_weights = torch.load(args.mm_instruct_pretrain, map_location='cpu')
            instruct_weights_real = dict()
            for k,v in instruct_weights.items():
                if 'instruct' in k:
                    instruct_weights_real[k.replace('model.vision_tower.vision_tower.', '')] = v
            self.vision_tower.load_state_dict(instruct_weights_real, strict=False)
        for name, param in self.vision_tower.named_parameters():
            if 'instruct' not in name:  # qa-vit components are named with instruct and are trainables
                param.requires_grad = False
        
        if args is not None or model_base is not None:
            base_path = args.model_name_or_path if args is not None else model_base
            if base_path is None:
                raise ValueError("Both args and model_base cannot be None.")

            index_path = os.path.join(base_path, "model.safetensors.index.json")
            with open(index_path, 'r') as file:
                model_state_dict_index_map = json.load(file)

            weight_key = "model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.instruction_gate"
            weight_map_path = model_state_dict_index_map["weight_map"][weight_key]

            vision_model_state_dict = load_file(os.path.join(base_path, weight_map_path))

            vision_model_state_dict_real = {
                k.replace('model.vision_tower.vision_tower.', ''): v
                for k, v in vision_model_state_dict.items()
            }

            missing, unexpected = self.vision_tower.load_state_dict(vision_model_state_dict_real, strict=False)
            assert len(missing) == 0

            self.vision_tower.requires_grad_(False)  # pretrain with single conversation
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:] # get patch features
        return image_features

    def forward(self, pixel_values, **kwargs):
        images = pixel_values
        if type(images) is list:
            raise ValueError(f'pixel_values is expected to be a torch tensor')
        else:
            if self.clip_type == 'qavit':
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                       instruct_states=kwargs['instruct_states'].to(device=self.device,
                                                                                                    dtype=self.dtype),
                                                       instruct_masks=kwargs['instruct_masks'].to(device=self.device,
                                                                                                  dtype=self.dtype) \
                                                                                                if kwargs['instruct_masks'] is not None else None,
                                                       output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                       output_hidden_states=True)
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

    @property
    def hidden_size(self):
        return self.config.hidden_size
    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2