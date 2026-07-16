#!/bin/bash
# Setup Qwen2.5-VL for finetuning on MONITRS v2
# Works with Python 3.10+, better than VideoLLaVA
#
# Usage:
#   cd Train
#   bash setup_qwen.sh

set -e

echo "=========================================="
echo "Setting up Qwen2.5-VL for MONITRS finetuning"
echo "=========================================="

# Create venv
if [ ! -d ~/qwen-env ]; then
    echo "Creating Python venv..."
    python3 -m venv ~/qwen-env
fi

source ~/qwen-env/bin/activate
pip install --upgrade pip

echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate deepspeed peft
pip install "ms-swift[llm]" -U
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation

echo ""
echo "=========================================="
echo "Qwen2.5-VL setup complete!"
echo ""
echo "To finetune on MONITRS:"
echo "  source ~/qwen-env/bin/activate"
echo "  python convert_qa_to_qwen.py"
echo "  bash finetune_qwen.sh"
echo "=========================================="
