#!/bin/bash
# Finetune VideoLLaVA on MONITRS v2 dataset
#
# Prerequisites:
#   - Run setup_videollava.sh first
#   - QA data generated (train_total.json, test_total.json)
#   - Images downloaded in Data/images/
#
# Usage:
#   conda activate videollava
#   bash finetune_videollava.sh

set -e

cd Video-LLaVA

MODEL_NAME="LanguageBind/Video-LLaVA-7B"
DATA_PATH="../../train_total.json"
IMAGE_DIR="../.."
OUTPUT_DIR="../../checkpoints/videollava-monitrs-v2"

echo "=========================================="
echo "Finetuning VideoLLaVA on MONITRS v2"
echo "  Model: $MODEL_NAME"
echo "  Data:  $DATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

deepspeed --num_gpus=1 videollava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --model_name_or_path $MODEL_NAME \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_DIR \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

echo ""
echo "Finetuning complete! Checkpoint: $OUTPUT_DIR"
