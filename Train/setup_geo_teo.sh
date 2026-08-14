#!/bin/bash
# Setup GeoChat + TEOChat for benchmarking
# Uses separate venv (~/rs-env) - different torch/transformers pinning than qwen-env

set -e

echo "=========================================="
echo "Setting up GeoChat + TEOChat"
echo "=========================================="

# Both use LLaVA/VideoLLaVA-based code that pins torch==2.0.1 → needs Python 3.10
if ! command -v python3.10 &> /dev/null; then
    echo "Installing Python 3.10..."
    sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
fi

# Create venv
if [ ! -d ~/rs-env ]; then
    python3.10 -m venv ~/rs-env
fi
source ~/rs-env/bin/activate
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
