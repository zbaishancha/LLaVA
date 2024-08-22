#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /mnt/csi-data-aly/user/haozhou/Projects/LLaVA/checkpoints/llava-v1.5-7b-finetune-dinov2-large \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-finetune-dinov2-large.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-finetune-dinov2-large.jsonl
