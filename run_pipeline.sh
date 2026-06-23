#!/bin/bash
# MONITRS v2 — Pipeline runner
#
# Three stages:
#   1. articles  — Scrape/search for articles about each FEMA event
#   2. process   — Extract locations, determine centers, generate captions (+ optional imagery)
#   3. images    — (Optional) Download Sentinel-2 imagery separately
#
# Usage:
#   ./run_pipeline.sh articles          # Step 1: get articles
#   ./run_pipeline.sh process           # Step 2: language pipeline (text only)
#   ./run_pipeline.sh process --images  # Step 2: language pipeline + download images
#   ./run_pipeline.sh images            # Step 3: download images separately
#   ./run_pipeline.sh merge             # Merge split result files
#   ./run_pipeline.sh summary           # Show progress
#
# Required env vars:
#   GCP_PROJECT_ID  — Vertex AI project
#   FIRMS_MAP_KEY   — NASA FIRMS key (free: https://firms.modaps.eosdis.nasa.gov/api/map_key/)
#   EE_PROJECT_ID   — Earth Engine project (for image download, optional)

set -e

STAGE=${1:-""}
EXTRA_ARGS="${@:2}"
DATA_DIR="Data"
SPLIT_A="$DATA_DIR/events_processed_0_5000.json"
SPLIT_B="$DATA_DIR/events_processed_5000_end.json"
MERGED="$DATA_DIR/events_processed.json"

# ============================================================
# Stage 1: Get articles
# ============================================================
run_articles() {
    echo ""
    echo "=========================================="
    echo "STAGE 1: Get articles"
    echo "=========================================="
    echo "Searching for articles for each FEMA event..."
    echo "Output: $DATA_DIR/articles.csv"
    echo ""

    python MONITRS/get_articles.py
}

# ============================================================
# Stage 2: Language pipeline (captions + geolocation)
# ============================================================
run_process() {
    echo ""
    echo "=========================================="
    echo "STAGE 2: Language pipeline"
    echo "=========================================="

    if [ -z "$GCP_PROJECT_ID" ]; then
        echo "ERROR: GCP_PROJECT_ID not set"; exit 1
    fi
    echo "GCP_PROJECT_ID=$GCP_PROJECT_ID"
    echo "FIRMS_MAP_KEY=${FIRMS_MAP_KEY:-(not set)}"

    mkdir -p "$DATA_DIR"

    # Check if --images flag is passed
    IMAGE_FLAG="--no-images"
    if echo "$EXTRA_ARGS" | grep -q "\-\-images"; then
        IMAGE_FLAG=""
        echo "Mode: text + images"
        echo "EE_PROJECT_ID=${EE_PROJECT_ID:-$GCP_PROJECT_ID}"
    else
        echo "Mode: text only (use './run_pipeline.sh images' for imagery later)"
    fi

    echo ""
    echo "Running in 2 parallel processes..."
    echo "  Output A: $SPLIT_A"
    echo "  Output B: $SPLIT_B"
    echo ""

    python MONITRS/run_language_pipeline.py $IMAGE_FLAG --start 0 --end 5000 &
    PID_A=$!
    python MONITRS/run_language_pipeline.py $IMAGE_FLAG --start 5000 &
    PID_B=$!

    echo "PIDs: $PID_A (0-5000), $PID_B (5000+)"
    echo "Ctrl+C to stop — progress is saved automatically"
    echo ""

    wait $PID_A || true
    wait $PID_B || true

    echo ""
    echo "Merging results..."
    merge_results
}

# ============================================================
# Stage 3: Image download (optional, separate)
# ============================================================
run_images() {
    echo ""
    echo "=========================================="
    echo "STAGE 3: Image download (Earth Engine)"
    echo "=========================================="

    if [ -z "$EE_PROJECT_ID" ]; then
        export EE_PROJECT_ID="${GCP_PROJECT_ID:-}"
    fi
    if [ -z "$EE_PROJECT_ID" ]; then
        echo "ERROR: EE_PROJECT_ID not set"; exit 1
    fi
    echo "EE_PROJECT_ID=$EE_PROJECT_ID"

    if [ ! -f "$MERGED" ]; then
        echo "ERROR: No merged results at $MERGED"
        echo "Run './run_pipeline.sh process' first, then './run_pipeline.sh merge'"
        exit 1
    fi

    TOTAL=$(python -c "import json; d=json.load(open('$MERGED')); print(sum(1 for v in d.values() if 'error' not in v))")
    echo "Downloading images for $TOTAL events..."
    echo ""

    python MONITRS/download_images.py --start 0 --end 5000 &
    PID_A=$!
    python MONITRS/download_images.py --start 5000 &
    PID_B=$!

    echo "PIDs: $PID_A (0-5000), $PID_B (5000+)"
    wait $PID_A || true
    wait $PID_B || true

    echo "Image download complete."
}

# ============================================================
# Merge split result files
# ============================================================
merge_results() {
    python -c "
import json, os

merged = {}
for f in ['$SPLIT_A', '$SPLIT_B']:
    if os.path.exists(f):
        data = json.load(open(f))
        merged.update(data)
        print(f'  {f}: {len(data)} events')

if os.path.exists('$MERGED'):
    existing = json.load(open('$MERGED'))
    for k, v in existing.items():
        if k not in merged:
            merged[k] = v

json.dump(merged, open('$MERGED', 'w'), indent=2)
n_ok = sum(1 for v in merged.values() if 'error' not in v)
n_err = sum(1 for v in merged.values() if 'error' in v)

strats = {}
for v in merged.values():
    s = v.get('strategy', 'error')
    strats[s] = strats.get(s, 0) + 1

print(f'Merged: {len(merged)} events ({n_ok} good, {n_err} errors)')
print(f'Strategies: {strats}')
"
}

# ============================================================
# Summary
# ============================================================
summary() {
    echo ""
    echo "=========================================="
    echo "PIPELINE STATUS"
    echo "=========================================="
    python -c "
import json, os

print('Articles:')
if os.path.exists('$DATA_DIR/articles.csv'):
    n = sum(1 for _ in open('$DATA_DIR/articles.csv'))
    events = set()
    for line in open('$DATA_DIR/articles.csv'):
        events.add(line.split(',')[0])
    print(f'  {n} article links across {len(events)} events')
else:
    print('  Not found')

print()
print('Text pipeline:')
for f in ['$SPLIT_A', '$SPLIT_B', '$MERGED']:
    if os.path.exists(f):
        d = json.load(open(f))
        n_ok = sum(1 for v in d.values() if 'error' not in v)
        print(f'  {os.path.basename(f)}: {len(d)} events ({n_ok} good)')

print()
print('Images:')
if os.path.exists('Data/download_progress.json'):
    p = json.load(open('Data/download_progress.json'))
    n_img = sum(1 for v in p.values() if 'error' not in v and 'skipped' not in v)
    print(f'  {n_img} events downloaded')
img_dir = '$DATA_DIR/images'
if os.path.isdir(img_dir):
    folders = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    print(f'  {len(folders)} image folders')
"
}

# ============================================================
# Main
# ============================================================
case $STAGE in
    articles)   run_articles ;;
    process)    run_process ;;
    images)     run_images ;;
    merge)      merge_results ;;
    summary)    summary ;;
    *)
        echo "MONITRS v2 Pipeline"
        echo ""
        echo "Usage: ./run_pipeline.sh <stage>"
        echo ""
        echo "Stages:"
        echo "  articles          Step 1: Scrape articles for FEMA events"
        echo "  process           Step 2: Language pipeline (text only)"
        echo "  process --images  Step 2: Language pipeline + image download"
        echo "  images            Step 3: Download imagery separately"
        echo "  merge             Merge split result files"
        echo "  summary           Show pipeline status"
        exit 1
        ;;
esac

summary
