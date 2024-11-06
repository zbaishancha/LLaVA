import torch
import torch.nn as nn
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
from safetensors.torch import load_file

from llava.model.multimodal_encoder.dino_encoder import DinoVisionTower

path = "/mnt/csi-data-aly/user/haozhou/Projects/LLaVA/playground/data/LingoQA/action/images/train/0a9f751521f9e4c38c317b9f20f9f533/0.jpg"
vision_tower = "/mnt/csi-data-aly/shared/public/haozhou/checkpoints/dinov2-large"
model_path = "checkpoints/llava-v1.5-7b-task-pros-of-pro-only-affinity-feature-v1"


model = DinoVisionTower(vision_tower, args=None)
model.load_model(model_path=model_path)

image = Image.open(path).convert('RGB')
image_tensor = model.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0)
with torch.no_grad():
    feature_map = model.affinity_forward(image_tensor)
print(feature_map.shape)
torch.save(feature_map, "./feature_map.pt")