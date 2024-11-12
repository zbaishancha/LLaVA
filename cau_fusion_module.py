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

        self.text_projection = nn.Linear(4096, 1024)
        self.query_projection = nn.Linear(256, 1024)
        
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
        text_embedding = self.text_projection(text_embedding) # B, N, D
        queries_embedding = self.query_projection(queries_embedding) # B, N, D
        
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

# Wrapper class for multi-input compatibility with ptflops
class FusionModelWrapper(nn.Module):
    def __init__(self, model, image_features, prompt_image_features, text_embedding, queries_embedding):
        super(FusionModelWrapper, self).__init__()
        self.model = model
        self.image_features = image_features
        self.prompt_image_features = prompt_image_features
        self.text_embedding = text_embedding
        self.queries_embedding = queries_embedding

    def forward(self, x):
        return self.model(self.image_features, self.prompt_image_features, self.text_embedding, self.queries_embedding)

# Initialize the model and inputs
model = FusionModel(feature_fusion_strategy='one-cross', vision_tower_name='example_tower').cuda()
image_features = torch.randn(1, 576, 1024).cuda()  # Example input shape (Batch, Tokens, Features)
prompt_image_features = torch.randn(1, 576, 1024).cuda()
text_embedding = torch.randn(1, 20, 4096).cuda()
queries_embedding = torch.randn(1, 16, 256).cuda()

# Wrap the model for FLOPs calculation
wrapped_model = FusionModelWrapper(model, image_features, prompt_image_features, text_embedding, queries_embedding)

# Parameter count and FLOPs calculation
def get_params_and_flops(model):
    # Print parameter summary
    print("Model Parameter Summary:")
    summary(model, input_size=(1,))

    # Calculate FLOPs
    print("FLOPs Calculation:")
    macs, params = get_model_complexity_info(
        model, 
        (1,),  # Dummy input to comply with ptflops
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )
    
    # Convert to GFLOPs and million parameters
    flops_in_gflops = float(macs.replace(" GMac", ""))  # Assuming output in GMac
    params_in_million = float(params.replace(" M", ""))  # Assuming output in M
    print(f"{model.model.feature_fusion_strategy}_GFLOPs: {flops_in_gflops} GFLOPs, Params: {params_in_million} M")

# Run calculations
get_params_and_flops(wrapped_model)