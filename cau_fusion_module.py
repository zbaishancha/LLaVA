import torch
import torch.nn as nn
from llava.model.multimodal_encoder.dino_encoder import SelfAttention, CrossModalAttention

from torchinfo import summary
from ptflops import get_model_complexity_info

class FusionModel(nn.Module):
    def __init__(self, feature_fusion_strategy, vision_tower_name):
        super(FusionModel, self).__init__()
        self.feature_fusion_strategy = feature_fusion_strategy
        self.vision_tower_name = vision_tower_name

        if self.feature_fusion_strategy == 'series-connection-cross':
            self.prompt_module = nn.ModuleList([CrossModalAttention(self.vision_tower_name) for _ in range(3)])
        elif self.feature_fusion_strategy == 'parallel-connection-cross':
            self.prompt_module_text = CrossModalAttention(self.vision_tower_name)
            self.prompt_module_prompt = CrossModalAttention(self.vision_tower_name)
            self.prompt_module_object = CrossModalAttention(self.vision_tower_name)
        elif self.feature_fusion_strategy == 'one-cross':
            self.prompt_module = CrossModalAttention(self.vision_tower_name)
        elif self.feature_fusion_strategy == 'self-cross':
            self.global_attn = SelfAttention(self.vision_tower_name)
            self.prompt_module = CrossModalAttention(self.vision_tower_name)
            
    
    def forward(self, image_features, prompt_image_features, text_embedding, queries_embedding):
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
        elif self.feature_fusion_strategy == 'self-cross':
            prompt_features = torch.cat([prompt_image_features, text_embedding, queries_embedding], dim=1)
            image_features = image_features + self.global_attn(image_features)
            image_features = image_features + self.prompt_module(image_features, prompt_features)
        
        return image_features

# Assuming CrossModalAttention is defined, initialize the model and input
model = FusionModel(feature_fusion_strategy='series-connection-cross', vision_tower_name='example_tower')
image_features = torch.randn(1, 64, 256)  # Example input shape (Batch, Tokens, Features)
prompt_image_features = torch.randn(1, 64, 256)
text_embedding = torch.randn(1, 64, 256)
queries_embedding = torch.randn(1, 64, 256)

# Parameter count and FLOPs calculation
def get_params_and_flops(model, image_features, prompt_image_features, text_embedding, queries_embedding):
    # Print parameter summary
    print("Model Parameter Summary:")
    summary(model, input_data=(image_features, prompt_image_features, text_embedding, queries_embedding))

    # Calculate FLOPs
    print("FLOPs Calculation:")
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model, 
            (image_features.shape, prompt_image_features.shape, text_embedding.shape, queries_embedding.shape), 
            as_strings=True,
            print_per_layer_stat=True
        )
    print(f"FLOPs: {macs}, Params: {params}")

# Run calculations
get_params_and_flops(model, image_features, prompt_image_features, text_embedding, queries_embedding)