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


def convert(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)

    converted = []
    skipped = 0

    for item in data:
        video_paths = item.get('video', [])
        convos = item.get('conversations', [])

        if not video_paths or len(convos) < 2:
            skipped += 1
            continue

        # Verify all images exist, cap at 6 images per sample for training speed
        valid_paths = [p for p in video_paths if os.path.exists(p)][:6]
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
    convert('train_total.json', 'train_qwen.json')
    convert('test_total.json', 'test_qwen.json')
