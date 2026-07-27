"""
Test the finetuned Qwen2.5-VL model on MONITRS test samples.

Usage:
    source ~/qwen-env/bin/activate
    python Train/test_qwen.py --n 10                    # 10 random test samples
    python Train/test_qwen.py --event 88                # specific event
    python Train/test_qwen.py --n 20 --html             # save HTML report
"""

import json
import os
import re
import argparse
import random
import base64
from glob import glob

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def find_latest_checkpoint(ckpt_root='checkpoints/qwen2.5-vl-monitrs'):
    # Find the most recent v* run
    runs = sorted(glob(f'{ckpt_root}/v*'))
    if not runs:
        raise FileNotFoundError(f"No runs found in {ckpt_root}")
    latest_run = runs[-1]
    # Find highest checkpoint
    ckpts = sorted(glob(f'{latest_run}/checkpoint-*'),
                   key=lambda p: int(p.split('-')[-1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {latest_run}")
    return ckpts[-1]


def load_model(checkpoint_dir):
    print(f"Loading base model: {BASE_MODEL}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    print(f"Loading LoRA adapter: {checkpoint_dir}")
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    return model, processor


def build_message(question, image_paths):
    content = []
    for p in image_paths:
        content.append({"type": "image", "image": p})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


@torch.no_grad()
def generate(model, processor, question, image_paths, max_new_tokens=128):
    messages = build_message(question, image_paths)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(trimmed, skip_special_tokens=True)
    return response.strip()


def load_test_samples():
    with open('test_qwen.json') as f:
        return json.load(f)


def clean_question(user_content):
    # Strip <image> tokens from question text
    return re.sub(r'<image>', '', user_content).strip()


def render_html(results, output='test_results.html'):
    html = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
            '<title>Qwen2.5-VL MONITRS Test</title>',
            '<style>',
            'body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }',
            '.item { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px 0; }',
            '.imgs { display: flex; gap: 6px; overflow-x: auto; margin: 8px 0; }',
            '.imgs img { width: 140px; height: 140px; object-fit: cover; border-radius: 4px; }',
            '.q { font-weight: bold; color: #333; }',
            '.gt { color: #093; background: #e8f5e8; padding: 6px 10px; border-radius: 4px; margin: 4px 0; }',
            '.pred { color: #06c; background: #e8f0f5; padding: 6px 10px; border-radius: 4px; margin: 4px 0; }',
            '</style></head><body><h1>Qwen2.5-VL MONITRS Test Results</h1>']

    for r in results:
        html.append('<div class="item">')
        html.append('<div class="imgs">')
        for p in r['images'][:6]:
            try:
                with open(p, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                ext = 'png' if p.endswith('.png') else 'jpeg'
                html.append(f'<img src="data:image/{ext};base64,{b64}">')
            except Exception:
                html.append('<div style="width:140px;height:140px;background:#eee"></div>')
        html.append('</div>')
        html.append(f'<div class="q">Q: {r["question"]}</div>')
        html.append(f'<div class="gt">Ground truth: {r["ground_truth"]}</div>')
        html.append(f'<div class="pred">Model: {r["prediction"]}</div>')
        html.append('</div>')

    html.append('</body></html>')
    with open(output, 'w') as f:
        f.write('\n'.join(html))
    print(f"HTML: {output} ({os.path.getsize(output)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='Path to LoRA checkpoint (default: latest)')
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--event', type=int, default=None)
    parser.add_argument('--html', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ckpt = args.checkpoint or find_latest_checkpoint()
    print(f"Using checkpoint: {ckpt}\n")

    model, processor = load_model(ckpt)
    samples = load_test_samples()

    # Filter to specific event if requested
    if args.event is not None:
        # test_qwen.json doesn't have event_id — need to load train_total.json to map
        # For now, just show samples matching event in image paths
        eid_str = f'/{args.event}/'
        samples = [s for s in samples if any(eid_str in p for p in s['images'])]

    random.seed(args.seed)
    samples = random.sample(samples, min(args.n, len(samples)))

    results = []
    for i, s in enumerate(samples, 1):
        question = clean_question(s['messages'][0]['content'])
        gt = s['messages'][1]['content']
        images = s['images'][:6]  # cap images for speed

        print(f"\n{'='*70}")
        print(f"Sample {i}/{len(samples)}")
        print(f"Images: {len(images)}")
        print(f"Q: {question[:200]}")
        print(f"GT: {gt[:200]}")

        try:
            pred = generate(model, processor, question, images)
        except Exception as e:
            pred = f"[ERROR: {e}]"

        print(f"Pred: {pred[:200]}")

        results.append({
            'question': question,
            'ground_truth': gt,
            'prediction': pred,
            'images': images,
        })

    if args.html:
        render_html(results)


if __name__ == '__main__':
    main()
