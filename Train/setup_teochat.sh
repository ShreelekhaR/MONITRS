#!/bin/bash
# Setup TEOChat for finetuning on MONITRS v2
#
# Prerequisites:
#   - NVIDIA GPU (A100 80GB recommended)
#   - CUDA 11.8+
#   - conda installed
#
# Usage:
#   cd Train
#   bash setup_teochat.sh

set -e

echo "=========================================="
echo "Setting up TEOChat for MONITRS finetuning"
echo "=========================================="

# Clone TEOChat
if [ ! -d "TEOChat" ]; then
    echo "Cloning TEOChat..."
    git clone https://github.com/ermongroup/TEOChat.git
else
    echo "TEOChat already cloned"
fi

cd TEOChat

# Create conda environment
echo "Creating conda environment..."
conda create -n teochat python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate teochat

# Install dependencies
echo "Installing dependencies..."
pip install -e .
pip install flash-attn --no-build-isolation
pip install deepspeed

# Link MONITRS data
echo "Linking MONITRS data..."
ln -sf ../../Data/images ./monitrs_images
ln -sf ../../MONITRS_QA ./monitrs_qa

echo ""
echo "=========================================="
echo "TEOChat setup complete!"
echo ""
echo "To finetune on MONITRS:"
echo "  conda activate teochat"
echo "  bash ../finetune_teochat.sh"
echo "=========================================="
