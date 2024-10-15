import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, CLIPTextTower
from .dino_encoder import DinoVisionTower
from .siglip_encoder import SiglipVisionTower
from .grounding_dino_encoder import GroundingDinoVisionTower
from .sam import SAMVisionTower

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
        if 'dinov2' in prompt_tower:
            return DinoVisionTower(prompt_tower, args=prompt_tower_cfg, **kwargs)
        elif 'sam' in prompt_tower:
            return SAMVisionTower(prompt_tower)

    raise ValueError(f'Unknown vision tower: {prompt_tower_cfg}')


def build_object_tower(object_tower_cfg, **kwargs):
    object_tower = getattr(object_tower_cfg, 'mm_object_tower', getattr(object_tower_cfg, 'object_tower', None))
    is_absolute_path_exists = os.path.exists(object_tower)
    
    if is_absolute_path_exists:
        return GroundingDinoVisionTower(object_tower, args=object_tower_cfg, **kwargs)
    
    raise ValueError(f'Unknown vision tower: {object_tower_cfg}')
