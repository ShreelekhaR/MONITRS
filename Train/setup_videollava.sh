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

# Create conda/micromamba environment
echo "Creating environment..."
if command -v micromamba &> /dev/null; then
    micromamba create -n videollava python=3.10 -y
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate videollava
else
    conda create -n videollava python=3.10 -y
    eval "$(conda shell.bash hook)"
    conda activate videollava
fi

# Install dependencies
echo "Installing dependencies..."
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
