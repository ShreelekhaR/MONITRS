#!/bin/bash
# Setup VideoLLaVA for finetuning on MONITRS v2
#
# Prerequisites:
#   - NVIDIA GPU (A100 80GB recommended)
#   - CUDA 11.8+
#   - conda installed
#
# Usage:
#   cd Train
#   bash setup_videollava.sh

set -e

echo "=========================================="
echo "Setting up VideoLLaVA for MONITRS finetuning"
echo "=========================================="

# Clone VideoLLaVA
if [ ! -d "Video-LLaVA" ]; then
    echo "Cloning VideoLLaVA..."
    git clone https://github.com/PKU-YuanGroup/Video-LLaVA.git
else
    echo "VideoLLaVA already cloned"
fi

cd Video-LLaVA

# Create environment (venv with Python 3.10 - required by VideoLLaVA)
echo "Creating Python 3.10 venv..."
if ! command -v python3.10 &> /dev/null; then
    echo "Installing Python 3.10..."
    sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv
fi
python3.10 -m venv ~/videollava-env
source ~/videollava-env/bin/activate
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install packaging wheel
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install deepspeed

# Link MONITRS data
echo "Linking MONITRS data..."
ln -sf ../../Data/images ./monitrs_images
ln -sf ../../MONITRS_QA ./monitrs_qa

echo ""
echo "=========================================="
echo "VideoLLaVA setup complete!"
echo ""
echo "To finetune on MONITRS:"
echo "  conda activate videollava"
echo "  bash ../finetune_videollava.sh"
echo "=========================================="
