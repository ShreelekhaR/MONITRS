#!/bin/bash
# Finetune Qwen2.5-VL-7B on MONITRS v2 with LoRA
# Runs on single A100 40GB
#
# Usage:
#   source ~/qwen-env/bin/activate
#   cd ~/MONITRS  # data files are in repo root
#   bash Train/finetune_qwen.sh

set -e

# Data paths (in repo root)
TRAIN_DATA="$(pwd)/train_qwen.json"
TEST_DATA="$(pwd)/test_qwen.json"
OUTPUT_DIR="$(pwd)/checkpoints/qwen2.5-vl-monitrs"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: $TRAIN_DATA not found. Run first:"
    echo "  python Train/convert_qa_to_qwen.py"
    exit 1
fi

echo "=========================================="
echo "Finetuning Qwen2.5-VL-7B on MONITRS v2"
echo "  Train: $TRAIN_DATA"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

# LoRA training on single A100 40GB
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --train_type lora \
    --dataset "$TRAIN_DATA" \
    --val_dataset "$TEST_DATA" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_checkpointing true \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 500 \
    --save_total_limit 3 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --output_dir "$OUTPUT_DIR" \
    --dataloader_num_workers 4 \
    --bf16 true \
    --attn_impl flash_attn \
    --max_length 4096 \
    --freeze_vit true

echo ""
echo "Training complete! Checkpoint: $OUTPUT_DIR"
