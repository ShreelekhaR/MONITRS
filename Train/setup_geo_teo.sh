#!/bin/bash
# Setup GeoChat + TEOChat for benchmarking
# Uses separate venv (~/rs-env) - different torch/transformers pinning than qwen-env

set -e

echo "=========================================="
echo "Setting up GeoChat + TEOChat"
echo "=========================================="

# Debian bookworm doesn't have python3.10-venv → use micromamba in $HOME
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
pip install -e . || true
pip install -e ".[train]" || true

# TEOChat is VideoLLaVA-based; install its extras too
echo "Installing TEOChat deps..."
cd ~/TEOChat
pip install -e . || true

# Metrics for benchmark
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
