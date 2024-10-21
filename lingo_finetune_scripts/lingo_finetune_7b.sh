#!/bin/bash
set -e -x  # stop on 1st error, debug output of args used

export CLEARML_API_ACCESS_KEY="E6D6L0KI5ZI79TKD1AW5"
export CLEARML_API_SECRET_KEY="wlGIykhRIQIJ7Em8duOkkBSrZhR67WGsbSBFp1WvkwfG5eepsT"

torchrun --nproc_per_node=8 \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /mnt/csi-data-aly/shared/public/haozhou/checkpoints/LLaVA/llava-v1.5-7b \
    --version v1 \
    --data_path ./playground/data/LingoQA/train_multi.json \
    --vision_tower /mnt/csi-data-aly/shared/public/haozhou/checkpoints/clip-vit-large-patch14-336 \
    --prompt_tower /mnt/csi-data-aly/shared/public/haozhou/checkpoints/dinov2-large \
    --object_tower /mnt/csi-data-aly/shared/public/haozhou/checkpoints/mask2former-swin-large-cityscapes-semantic \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task-pros-of-pro-mask2former-self-cross \
    --exp_name finetune_llava_v1.5_7b_lingoqa_prompts_of_prompt_mask2former_self_cross \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --crop False \
    --feature_fusion_strategy one-cross

# multi nodes
# --nnodes=${WORLD_SIZE} \
# --node_rank=${RANK} \
# --master_addr=${MASTER_ADDR} \
# --master_port=${MASTER_PORT} \
