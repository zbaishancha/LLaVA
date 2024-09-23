import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, CLIPTextTower
from .dino_encoder import DinoVisionTower
from .siglip_encoder import SiglipVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if "clip" in vision_tower and is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    if "dinov2" in vision_tower.lower():
        return DinoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    if "siglip" in vision_tower.lower():
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_prompt_tower(prompt_tower_cfg, **kwargs):
    prompt_tower = getattr(prompt_tower_cfg, 'mm_prompt_tower', getattr(prompt_tower_cfg, 'prompt_tower', None))
    is_absolute_path_exists = os.path.exists(prompt_tower)

    if is_absolute_path_exists:
        return DinoVisionTower(prompt_tower, args=prompt_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {prompt_tower_cfg}')
