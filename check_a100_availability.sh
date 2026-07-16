#!/bin/bash
# Sweep GCP US zones for A100 80GB availability + quota.
#
# Usage:
#   bash check_a100_availability.sh
#   bash check_a100_availability.sh --project my-project
#
# Requires: gcloud CLI authenticated

set -u

PROJECT="${1:-$(gcloud config get-value project 2>/dev/null)}"
if [[ "$1" == "--project" ]]; then
    PROJECT="$2"
fi

echo "Checking A100 80GB availability for project: $PROJECT"
echo "==========================================="
echo ""

# US zones to check (all US regions, A100 usually in these)
ZONES=(
    us-central1-a us-central1-b us-central1-c us-central1-f
    us-east1-b us-east1-c us-east1-d
    us-east4-a us-east4-b us-east4-c
    us-east5-a us-east5-b us-east5-c
    us-west1-a us-west1-b us-west1-c
    us-west2-a us-west2-b us-west2-c
    us-west3-a us-west3-b us-west3-c
    us-west4-a us-west4-b us-west4-c
    us-south1-a us-south1-b us-south1-c
)

# A100 80GB accelerator name
A100_80GB="nvidia-a100-80gb"

echo "Zones offering NVIDIA A100 80GB:"
echo "--------------------------------"
AVAILABLE_ZONES=()
for zone in "${ZONES[@]}"; do
    result=$(gcloud compute accelerator-types list \
        --filter="zone:$zone AND name=$A100_80GB" \
        --format="value(name)" 2>/dev/null)
    if [[ -n "$result" ]]; then
        echo "  ✓ $zone"
        AVAILABLE_ZONES+=("$zone")
    fi
done

if [[ ${#AVAILABLE_ZONES[@]} -eq 0 ]]; then
    echo "  No zones with A100 80GB found."
    exit 1
fi

echo ""
echo "Quota check (regions with available zones):"
echo "-------------------------------------------"

# Get unique regions from available zones
REGIONS=()
for zone in "${AVAILABLE_ZONES[@]}"; do
    region="${zone%-*}"
    if [[ ! " ${REGIONS[@]} " =~ " $region " ]]; then
        REGIONS+=("$region")
    fi
done

for region in "${REGIONS[@]}"; do
    quota=$(gcloud compute regions describe "$region" \
        --project="$PROJECT" \
        --format="value(quotas[].limit,quotas[].usage,quotas[].metric)" 2>/dev/null | \
        grep -A2 "NVIDIA_A100_80GB_GPUS" | head -3 || echo "")

    limit=$(gcloud compute regions describe "$region" \
        --project="$PROJECT" \
        --format="json(quotas)" 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
for q in data.get('quotas', []):
    if 'A100_80GB' in q.get('metric', ''):
        limit = int(q.get('limit', 0))
        usage = int(q.get('usage', 0))
        available = limit - usage
        print(f'{q[\"metric\"]}: {available}/{limit} available ({usage} in use)')
        break
else:
    print('N/A')
" 2>/dev/null || echo "N/A")

    echo "  $region: $limit"
done

echo ""
echo "Try creating a workbench instance in the zones with ✓ above."
echo "Cloud Console: https://console.cloud.google.com/vertex-ai/workbench"
echo ""
echo "Or via gcloud (adjust ZONE):"
echo "  gcloud workbench instances create monitrs-train \\"
echo "    --location=us-central1-c \\"
echo "    --machine-type=a2-ultragpu-1g \\"
echo "    --accelerator-type=NVIDIA_A100_80GB \\"
echo "    --accelerator-core-count=1"
