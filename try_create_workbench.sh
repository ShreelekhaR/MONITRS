#!/bin/bash
# Try creating an A100 80GB workbench in each US zone until one succeeds.
# Stops at first success. Aborts a zone quickly on ZONE_RESOURCE_POOL_EXHAUSTED.
#
# Usage:
#   bash try_create_workbench.sh
#   bash try_create_workbench.sh --name my-instance --disk 200

NAME="monitrs-train"
DISK_SIZE=500
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --disk) DISK_SIZE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PROJECT="$(gcloud config get-value project 2>/dev/null)"
if [[ -z "$PROJECT" ]]; then
    echo "Error: no project set. Run: gcloud config set project YOUR_PROJECT"
    exit 1
fi

# US zones known to offer A100 80GB — try in order of likely availability
ZONES=(
    us-central1-c us-central1-a us-central1-b us-central1-f
    us-east4-c us-east4-a us-east4-b
    us-east5-a us-east5-b us-east5-c
    us-west1-b us-west1-a us-west1-c
    us-west4-b us-west4-a us-west4-c
    us-east1-c us-east1-b us-east1-d
    us-south1-a us-south1-b us-south1-c
)

echo "Project: $PROJECT"
echo "Instance name: $NAME"
echo "Disk size: ${DISK_SIZE}GB"
echo ""
echo "Trying to create A100 80GB workbench in each zone..."
echo "==========================================="

for zone in "${ZONES[@]}"; do
    echo ""
    echo "Trying zone: $zone"

    if $DRY_RUN; then
        echo "  [DRY RUN] Would try: $zone"
        continue
    fi

    output=$(gcloud workbench instances create "$NAME" \
        --project="$PROJECT" \
        --location="$zone" \
        --machine-type="a2-ultragpu-1g" \
        --accelerator-type="NVIDIA_A100_80GB" \
        --accelerator-core-count=1 \
        --install-gpu-driver \
        --boot-disk-size=150 \
        --boot-disk-type="PD_SSD" \
        --data-disk-size="$DISK_SIZE" \
        --data-disk-type="PD_SSD" \
        2>&1)
    exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo ""
        echo "==========================================="
        echo "SUCCESS: Created $NAME in $zone"
        echo "==========================================="
        echo ""
        echo "Access at:"
        echo "  https://console.cloud.google.com/vertex-ai/workbench/instances?project=$PROJECT"
        exit 0
    fi

    if echo "$output" | grep -q "ZONE_RESOURCE_POOL_EXHAUSTED\|resource pool exhausted\|does not have enough resources"; then
        echo "  [OUT OF CAPACITY]"
    elif echo "$output" | grep -q "QUOTA_EXCEEDED\|Quota"; then
        echo "  [QUOTA EXCEEDED — request quota increase]"
        echo "  https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT"
        exit 1
    elif echo "$output" | grep -q "already exists"; then
        echo "  [ERROR: instance name '$NAME' already exists — use --name to pick another]"
        exit 1
    else
        echo "  [ERROR]"
        echo "$output" | head -5 | sed 's/^/    /'
    fi
done

echo ""
echo "==========================================="
echo "No A100 80GB capacity found in any US zone."
echo "Try again in 10-30 min, or request different accelerator."
echo "==========================================="
exit 1
