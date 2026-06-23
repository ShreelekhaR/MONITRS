#!/bin/bash
# MONITRS v2 — Full pipeline runner
#
# Usage:
#   # Text only (laptop or workbench):
#   ./run_pipeline.sh text
#
#   # Images only (workbench with EE auth):
#   ./run_pipeline.sh images
#
#   # Everything:
#   ./run_pipeline.sh all
#
# Required env vars:
#   GCP_PROJECT_ID  — Vertex AI project
#   FIRMS_MAP_KEY   — NASA FIRMS key (free)
#   EE_PROJECT_ID   — Earth Engine project (for image download)

set -e

STAGE=${1:-all}
DATA_DIR="Data"
SPLIT_A="$DATA_DIR/events_processed_0_5000.json"
SPLIT_B="$DATA_DIR/events_processed_5000_end.json"
MERGED="$DATA_DIR/events_processed.json"

# --- Checks ---
check_env() {
    if [ -z "$GCP_PROJECT_ID" ]; then
        echo "ERROR: GCP_PROJECT_ID not set"
        exit 1
    fi
    echo "GCP_PROJECT_ID=$GCP_PROJECT_ID"
    echo "FIRMS_MAP_KEY=${FIRMS_MAP_KEY:-(not set, FIRMS disabled)}"
    echo "EE_PROJECT_ID=${EE_PROJECT_ID:-$GCP_PROJECT_ID}"
}

# --- Stage 1: Text pipeline ---
run_text() {
    echo ""
    echo "=========================================="
    echo "STAGE 1: Text pipeline (language + geolocation)"
    echo "=========================================="
    check_env

    mkdir -p "$DATA_DIR"

    if [ -f "$MERGED" ]; then
        EXISTING=$(python -c "import json; print(len(json.load(open('$MERGED'))))")
        echo "Existing merged file has $EXISTING events"
        read -p "Re-run from scratch? (y/N): " RERUN
        if [ "$RERUN" = "y" ]; then
            rm -f "$SPLIT_A" "$SPLIT_B" "$MERGED"
        fi
    fi

    echo ""
    echo "Running text pipeline in 2 parallel processes..."
    echo "  Output A: $SPLIT_A"
    echo "  Output B: $SPLIT_B"
    echo ""

    python MONITRS/run_language_pipeline.py --no-images --start 0 --end 5000 &
    PID_A=$!
    python MONITRS/run_language_pipeline.py --no-images --start 5000 &
    PID_B=$!

    echo "PIDs: $PID_A (0-5000), $PID_B (5000+)"
    echo "Ctrl+C to stop — progress is saved automatically"

    wait $PID_A
    wait $PID_B

    echo ""
    echo "Merging results..."
    merge_results
}

# --- Merge ---
merge_results() {
    python -c "
import json, os

merged = {}
for f in ['$SPLIT_A', '$SPLIT_B']:
    if os.path.exists(f):
        data = json.load(open(f))
        merged.update(data)
        print(f'  {f}: {len(data)} events')

# Also merge any existing results
if os.path.exists('$MERGED'):
    existing = json.load(open('$MERGED'))
    for k, v in existing.items():
        if k not in merged:
            merged[k] = v
    print(f'  {\"$MERGED\"} (existing): {len(existing)} events')

json.dump(merged, open('$MERGED', 'w'), indent=2)
n_ok = sum(1 for v in merged.values() if 'error' not in v)
n_err = sum(1 for v in merged.values() if 'error' in v)

strats = {}
for v in merged.values():
    s = v.get('strategy', 'error')
    strats[s] = strats.get(s, 0) + 1

print()
print(f'Merged: {len(merged)} events ({n_ok} good, {n_err} errors)')
print(f'Strategies: {strats}')
"
}

# --- Stage 2: Image download ---
run_images() {
    echo ""
    echo "=========================================="
    echo "STAGE 2: Image download (Earth Engine)"
    echo "=========================================="

    if [ -z "$EE_PROJECT_ID" ]; then
        export EE_PROJECT_ID="$GCP_PROJECT_ID"
    fi
    echo "EE_PROJECT_ID=$EE_PROJECT_ID"

    if [ ! -f "$MERGED" ]; then
        echo "ERROR: No merged results at $MERGED"
        echo "Run './run_pipeline.sh text' first"
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
    echo "Ctrl+C to stop — progress is saved automatically"

    wait $PID_A
    wait $PID_B

    echo ""
    echo "Image download complete."
}

# --- Stage 3: QA generation ---
run_qa() {
    echo ""
    echo "=========================================="
    echo "STAGE 3: QA generation"
    echo "=========================================="
    check_env

    echo "Generating templated MCQ..."
    python MONITRS_QA/templated_mcq.py

    echo "Generating LLM MCQ..."
    python MONITRS_QA/generated_mcq.py

    echo "Generating open-ended QA..."
    python MONITRS_QA/generated_q_a.py

    echo "Merging train/test..."
    python MONITRS_QA/merge_train_test.py

    echo "QA generation complete."
}

# --- Summary ---
summary() {
    echo ""
    echo "=========================================="
    echo "PIPELINE SUMMARY"
    echo "=========================================="
    python -c "
import json, os

if os.path.exists('$MERGED'):
    r = json.load(open('$MERGED'))
    n_ok = sum(1 for v in r.values() if 'error' not in v)
    print(f'Events processed: {n_ok}/{len(r)}')

if os.path.exists('Data/download_progress.json'):
    p = json.load(open('Data/download_progress.json'))
    n_img = sum(1 for v in p.values() if 'error' not in v and 'skipped' not in v)
    print(f'Events with images: {n_img}')

img_dir = 'Data/images'
if os.path.isdir(img_dir):
    n_folders = len([d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])
    print(f'Image folders: {n_folders}')
"
}

# --- Main ---
case $STAGE in
    text)
        run_text
        summary
        ;;
    merge)
        merge_results
        ;;
    images)
        run_images
        summary
        ;;
    qa)
        run_qa
        ;;
    all)
        run_text
        run_images
        run_qa
        summary
        ;;
    summary)
        summary
        ;;
    *)
        echo "Usage: ./run_pipeline.sh {text|merge|images|qa|all|summary}"
        exit 1
        ;;
esac
