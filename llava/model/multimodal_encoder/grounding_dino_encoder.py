import torch
import torch.nn as nn
import os
import json
import math
import torch.nn.functional as F
from safetensors.torch import load_file

from transformers import AutoProcessor, GroundingDinoForObjectDetection, GroundingDinoConfig, GroundingDinoImageProcessor, AutoImageProcessor, Mask2FormerForUniversalSegmentation

# from PIL import Image
# import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# text = "a cat."

# processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
# model = AutoModel.from_pretrained("IDEA-Research/grounding-dino-tiny")

# inputs = processor(images=image, text=text, return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# list(last_hidden_states.shape)


class GroundingDinoVisionTower(nn.Module):
    def __init__(self, vision_tower, args=None, delay_load=True):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.args = args
        self.delay_load = delay_load

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = GroundingDinoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None, model_path=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        # load Mask2Former fine-tuned on Cityscapes semantic segmentation
        self.vision_tower = Mask2FormerForUniversalSegmentation.from_pretrained(self.vision_tower_name)
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.class_embeds = nn.Embedding(20, 256)
        self.class_embeds.requires_grad_(True)
        self.num_k = 16
        self.is_loaded = True
        if model_path is not None and os.path.exists(os.path.join(model_path, "model-00004-of-00004.safetensors")):
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(model_path, "model-00004-of-00004.safetensors"))
            state_dict_real = {
                k.replace('model.object_tower.', ''): v
                for k, v in state_dict.items() if "class_embeds" in k
            }
            missing, unexpected = self.load_state_dict(state_dict_real, strict=False)

    def forward(self, multi_images: torch.Tensor):
        self.vision_tower.eval()
        with torch.no_grad():
            outputs = self.vision_tower(multi_images.to(device=self.device, dtype=self.dtype))
        
        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        class_queries_logits = outputs.class_queries_logits
        mask_queries = outputs.transformer_decoder_last_hidden_state
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        predicted_classes = torch.argmax(masks_classes, dim=-1) 
        confidence_scores = torch.max(masks_classes, dim=-1).values
        _, topk_indices = torch.topk(confidence_scores, k=self.num_k, dim=-1)
        topk_mask_queries = mask_queries.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, mask_queries.size(-1)))
        topk_labels = predicted_classes.gather(1, topk_indices)
        topk_labels_embeds = self.class_embeds(topk_labels)
        topk_mask_queries += topk_labels_embeds
        
        return topk_mask_queries

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

