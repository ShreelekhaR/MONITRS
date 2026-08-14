#!/bin/bash
# Setup GeoChat + TEOChat for benchmarking
# Uses separate venv (~/rs-env) - different torch/transformers pinning than qwen-env

set -e

echo "=========================================="
echo "Setting up GeoChat + TEOChat"
echo "=========================================="

# Debian bookworm doesn't have python3.10-venv → use micromamba in $HOME
# Override pkgs cache since /opt/micromamba/pkgs is unwritable on GCP Workbench
export CONDA_PKGS_DIRS="$HOME/.mamba/pkgs"
export MAMBA_ROOT_PREFIX="$HOME/.mamba"
mkdir -p "$CONDA_PKGS_DIRS"

if [ ! -d ~/rs-env ]; then
    echo "Creating micromamba env at ~/rs-env..."
    if ! command -v micromamba &> /dev/null; then
        echo "micromamba not found. Install it first or run this on GCP Workbench."
        exit 1
    fi
    micromamba create -p ~/rs-env -c conda-forge python=3.10 -y
fi

# Activate the prefix env
eval "$(micromamba shell hook --shell bash)"
micromamba activate ~/rs-env
pip install --upgrade pip

# Clone GeoChat
if [ ! -d ~/GeoChat ]; then
    git clone https://github.com/mbzuai-oryx/GeoChat.git ~/GeoChat
fi

# Clone TEOChat
if [ ! -d ~/TEOChat ]; then
    git clone https://github.com/ermongroup/TEOChat.git ~/TEOChat
fi

# Install GeoChat (installs torch etc.)
echo "Installing GeoChat deps..."
cd ~/GeoChat
pip install -e . --no-deps || true
# Install required deps manually (avoid bnb path-scan on GCP)
pip install "transformers==4.31.0" "tokenizers>=0.13.3" sentencepiece shortuuid \
    "accelerate==0.21.0" peft "einops==0.6.1" "einops-exts==0.0.4" \
    "timm==0.6.13" "scikit-learn==1.2.2" gradio_client shortuuid httpx \
    "protobuf<4" torchvision

# TEOChat (VideoLLaVA-based). Install without conflicting gradio pin.
echo "Installing TEOChat deps..."
cd ~/TEOChat
pip install -e . --no-deps || true
# TEOChat needs videollava — likely already at the same path via -e
pip install "decord" "opencv-python==4.7.0.72" "av==11.0.0" imageio || true

# Bitsandbytes causes GCP permission errors. Uninstall — we don't quantize for eval.
pip uninstall -y bitsandbytes || true
# Older accelerate imports bnb unconditionally. Bump to a version that doesn't.
pip install "accelerate>=0.25.0" --upgrade

# Fix numpy/sklearn binary mismatch from newer wheels
pip install --force-reinstall --no-deps "numpy==1.26.4" "scipy==1.11.4"

# Metrics
pip install nltk rouge-score
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"

echo ""
echo "=========================================="
echo "Setup done."
echo "Test:"
echo "  source ~/rs-env/bin/activate"
echo "  cd ~/MONITRS"
echo "  python Train/benchmark.py --models geochat --out benchmark_geochat.json"
echo "  python Train/benchmark.py --models teochat --out benchmark_teochat.json"
echo "=========================================="
