import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
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

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    
    
class MaskFormerHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_features = ['res2', 'res3', 'res4', 'res5']

        self.ignore_value = 255
        self.common_stride = 4
        self.loss_weight = 1.0

        self.pixel_decoder = MSDeformAttnPixelDecoder()
        self.predictor = MultiScaleMaskedTransformerDecoder()
        self.transformer_in_feature = 'multi_scale_pixel_decoder'

        self.num_classes = 19


    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions

if __name__ == "__main__":
    transformer_kwargs = {
        "dim": 512,
        "depth": 4,
        "heads": 8,
        "dim_head": 128,
        "mlp_dim": 512 * 4, 
        "dropout": 0.1
    }
    model = Transformer(**transformer_kwargs)
    num_params = sum([p.numel() for p in model.parameters()]) / 1e6
    print(num_params)
    
    