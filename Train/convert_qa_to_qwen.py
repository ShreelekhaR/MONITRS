"""
Convert MONITRS v2 QA data to Qwen2.5-VL training format.

Input:  train_total.json / test_total.json (VideoLLaVA format)
Output: train_qwen.json / test_qwen.json (Qwen chat format)

Qwen format:
{
  "messages": [
    {"role": "user", "content": "<image><image><image>...question"},
    {"role": "assistant", "content": "answer"}
  ],
  "images": ["path1.png", "path2.png", ...]
}
"""

import json
import os
import re
import sys
import random
import argparse


def convert(input_path, output_path, max_samples=None, seed=42):
    with open(input_path) as f:
        data = json.load(f)

    if max_samples and len(data) > max_samples:
        random.seed(seed)
        # Stratified sample by task type to preserve balance
        by_task = {}
        for item in data:
            t = item.get('task', 'unknown')
            by_task.setdefault(t, []).append(item)

        sampled = []
        per_task = max_samples // len(by_task)
        for t, items in by_task.items():
            random.shuffle(items)
            sampled.extend(items[:per_task])
        random.shuffle(sampled)
        data = sampled
        print(f"  Subsampled to {len(data)} across {len(by_task)} tasks")

    converted = []
    skipped = 0

    for item in data:
        video_paths = item.get('video', [])
        convos = item.get('conversations', [])

        if not video_paths or len(convos) < 2:
            skipped += 1
            continue

        # Keep full image sequence
        valid_paths = [p for p in video_paths if os.path.exists(p)]
        if len(valid_paths) == 0:
            skipped += 1
            continue

        question = convos[0]['value']
        answer = convos[1]['value']

        # Clean the <video> marker — replace with per-image <image> tokens
        question = question.replace('<video>', '').strip()
        # Prepend image tokens
        image_tokens = '<image>' * len(valid_paths)
        user_content = f"{image_tokens}\n{question}"

        converted.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer}
            ],
            "images": valid_paths
        })

    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)

    print(f"{input_path} -> {output_path}")
    print(f"  Converted: {len(converted)}")
    print(f"  Skipped (missing images/data): {skipped}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-max', type=int, default=20000,
                        help='Max training samples (stratified by task). Default 20000.')
    parser.add_argument('--test-max', type=int, default=2000,
                        help='Max test samples. Default 2000.')
    args = parser.parse_args()

    convert('train_total.json', 'train_qwen.json', max_samples=args.train_max)
    convert('test_total.json', 'test_qwen.json', max_samples=args.test_max)
