"""
Merge split pipeline results and print summary.

Usage:
    python merge_results.py
"""

import json
import os
import glob


def main():
    merged = {}

    # Find all split result files
    split_files = sorted(glob.glob('Data/events_processed_*.json'))
    if not split_files:
        print("No split files found in Data/events_processed_*.json")
        return

    for f in split_files:
        try:
            data = json.load(open(f))
            merged.update(data)
            print(f"  {f}: {len(data)} events")
        except json.JSONDecodeError:
            print(f"  {f}: CORRUPTED, skipping")

    # Also merge existing merged file if present
    merged_file = 'Data/events_processed.json'
    if os.path.exists(merged_file):
        try:
            existing = json.load(open(merged_file))
            for k, v in existing.items():
                if k not in merged:
                    merged[k] = v
        except json.JSONDecodeError:
            print(f"  {merged_file}: corrupted, overwriting")

    # Save merged
    json.dump(merged, open(merged_file, 'w'), indent=2)

    # Summary
    n_ok = sum(1 for v in merged.values() if 'error' not in v)
    n_err = sum(1 for v in merged.values() if 'error' in v)

    strats = {}
    types = {}
    for v in merged.values():
        s = v.get('strategy', 'error')
        strats[s] = strats.get(s, 0) + 1
        t = v.get('type', 'unknown')
        if t not in types:
            types[t] = {'total': 0, 'ok': 0}
        types[t]['total'] += 1
        if 'error' not in v:
            types[t]['ok'] += 1

    print(f"\nMerged: {len(merged)} events ({n_ok} good, {n_err} errors)")
    print(f"Strategies: {strats}")
    print(f"\nBy event type:")
    for t, c in sorted(types.items(), key=lambda x: -x[1]['total']):
        print(f"  {t:<25} {c['ok']}/{c['total']}")

    # Image stats
    img_dir = 'Data/images'
    if os.path.isdir(img_dir):
        folders = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
        n_imgs = sum(1 for root, _, files in os.walk(img_dir)
                     for f in files if f.endswith('.png') or f.endswith('.jpg'))
        total_size = sum(os.path.getsize(os.path.join(root, f))
                         for root, _, files in os.walk(img_dir)
                         for f in files if f.endswith('.png') or f.endswith('.jpg'))
        print(f"\nImages:")
        print(f"  {len(folders)} event folders")
        print(f"  {n_imgs} total images")
        print(f"  {total_size / 1e9:.1f} GB")


if __name__ == '__main__':
    main()
