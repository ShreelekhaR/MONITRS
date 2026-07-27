"""
Evaluate base Qwen2.5-VL vs finetuned on MCQ questions.
Same samples, both models, deterministic seed.

Usage:
    source ~/qwen-env/bin/activate
    python Train/eval_mcq.py --n 100                # 100 MCQ samples
    python Train/eval_mcq.py --n 200 --html         # + HTML report
    python Train/eval_mcq.py --skip-base            # only test finetuned
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
    runs = sorted(glob(f'{ckpt_root}/v*'))
    if not runs:
        raise FileNotFoundError(f"No runs found in {ckpt_root}")
    ckpts = sorted(glob(f'{runs[-1]}/checkpoint-*'),
                   key=lambda p: int(p.split('-')[-1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {runs[-1]}")
    return ckpts[-1]


def load_base_model():
    print(f"Loading base model: {BASE_MODEL}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    return model


def load_finetuned_model(base_model, ckpt):
    print(f"Loading LoRA: {ckpt}")
    model = PeftModel.from_pretrained(base_model, ckpt)
    model.eval()
    return model


@torch.no_grad()
def generate(model, processor, question, image_paths, max_new_tokens=32):
    content = [{"type": "image", "image": p} for p in image_paths]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to("cuda:0")

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = output_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def extract_letter(text):
    """Pull first letter A-D or a-d from response."""
    m = re.search(r'\b([A-Da-d])\b', text)
    return m.group(1).lower() if m else '?'


def load_mcq_samples(n, seed=42):
    """Load samples that are MCQ (contain 'a.', 'b.', 'c.', 'd.' or 'A)', 'B)' options)."""
    # Load from the original train_total.json to filter by task
    with open('test_qwen.json') as f:
        all_samples = json.load(f)

    mcq_samples = []
    for s in all_samples:
        q = s['messages'][0]['content']
        gt = s['messages'][1]['content'].strip()
        # MCQ: gt is a single letter, question has A/B/C/D or a/b/c/d options
        if len(gt) <= 3 and re.match(r'^[a-dA-D]', gt):
            if re.search(r'\b[a-dA-D][.)]\s', q):
                mcq_samples.append(s)

    print(f"Found {len(mcq_samples)} MCQ samples in test set")
    random.seed(seed)
    return random.sample(mcq_samples, min(n, len(mcq_samples)))


def clean_question(text):
    return re.sub(r'<image>', '', text).strip()


def render_html(results, base_acc, ft_acc, output='eval_results.html'):
    html = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
            '<title>MCQ Eval: Base vs Finetuned</title>',
            '<style>',
            'body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }',
            '.summary { background: #f0f0f0; padding: 16px; border-radius: 8px; margin: 20px 0; }',
            '.item { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 12px 0; }',
            '.imgs { display: flex; gap: 4px; overflow-x: auto; margin: 8px 0; }',
            '.imgs img { width: 100px; height: 100px; object-fit: cover; border-radius: 4px; }',
            '.q { font-size: 13px; margin: 8px 0; }',
            '.row { display: flex; gap: 12px; margin: 4px 0; }',
            '.tag { font-weight: bold; min-width: 60px; }',
            '.correct { color: #093; }',
            '.wrong { color: #c00; }',
            '</style></head><body>',
            f'<h1>MCQ Eval: {len(results)} samples</h1>',
            '<div class="summary">',
            f'<div>Base Qwen2.5-VL: <b>{base_acc:.1%}</b> ({sum(r["base_correct"] for r in results)}/{len(results)})</div>',
            f'<div>Finetuned:       <b>{ft_acc:.1%}</b> ({sum(r["ft_correct"] for r in results)}/{len(results)})</div>',
            f'<div>Improvement: <b>{(ft_acc - base_acc)*100:+.1f} points</b></div>',
            '</div>']

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
                pass
        html.append('</div>')
        html.append(f'<div class="q">{r["question"]}</div>')
        html.append(f'<div class="row"><span class="tag">GT:</span><span>{r["gt"]}</span></div>')
        base_cls = 'correct' if r['base_correct'] else 'wrong'
        ft_cls = 'correct' if r['ft_correct'] else 'wrong'
        html.append(f'<div class="row"><span class="tag">Base:</span><span class="{base_cls}">{r["base_pred"]} → {r["base_letter"]}</span></div>')
        html.append(f'<div class="row"><span class="tag">FT:</span><span class="{ft_cls}">{r["ft_pred"]} → {r["ft_letter"]}</span></div>')
        html.append('</div>')

    html.append('</body></html>')
    with open(output, 'w') as f:
        f.write('\n'.join(html))
    print(f"HTML: {output} ({os.path.getsize(output)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--html', action='store_true')
    parser.add_argument('--skip-base', action='store_true', help='Only test finetuned')
    args = parser.parse_args()

    ckpt = args.checkpoint or find_latest_checkpoint()
    print(f"Checkpoint: {ckpt}\n")

    samples = load_mcq_samples(args.n, seed=args.seed)
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    # Run BOTH models on same samples
    # Load base first
    base_model = load_base_model()

    base_preds = []
    if not args.skip_base:
        print(f"\n== Base model on {len(samples)} MCQ samples ==")
        for i, s in enumerate(samples, 1):
            q = clean_question(s['messages'][0]['content'])
            imgs = s['images'][:6]
            try:
                pred = generate(base_model, processor, q, imgs)
            except Exception as e:
                pred = f"[ERR: {e}]"
            base_preds.append(pred)
            print(f"  [{i}/{len(samples)}] base: {pred[:60]}")
    else:
        base_preds = [''] * len(samples)

    # Now load finetuned (LoRA on top of base)
    ft_model = load_finetuned_model(base_model, ckpt)

    ft_preds = []
    print(f"\n== Finetuned model on {len(samples)} MCQ samples ==")
    for i, s in enumerate(samples, 1):
        q = clean_question(s['messages'][0]['content'])
        imgs = s['images'][:6]
        try:
            pred = generate(ft_model, processor, q, imgs)
        except Exception as e:
            pred = f"[ERR: {e}]"
        ft_preds.append(pred)
        print(f"  [{i}/{len(samples)}] ft:   {pred[:60]}")

    # Grade
    results = []
    for s, bp, fp in zip(samples, base_preds, ft_preds):
        gt = s['messages'][1]['content'].strip()
        gt_letter = extract_letter(gt)
        base_letter = extract_letter(bp)
        ft_letter = extract_letter(fp)
        results.append({
            'question': clean_question(s['messages'][0]['content']),
            'images': s['images'],
            'gt': gt,
            'base_pred': bp,
            'ft_pred': fp,
            'base_letter': base_letter,
            'ft_letter': ft_letter,
            'base_correct': base_letter == gt_letter,
            'ft_correct': ft_letter == gt_letter,
        })

    base_acc = sum(r['base_correct'] for r in results) / len(results)
    ft_acc = sum(r['ft_correct'] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"Base:     {base_acc:.1%} ({sum(r['base_correct'] for r in results)}/{len(results)})")
    print(f"Finetuned: {ft_acc:.1%} ({sum(r['ft_correct'] for r in results)}/{len(results)})")
    print(f"Delta:    {(ft_acc - base_acc)*100:+.1f} points")

    if args.html:
        render_html(results, base_acc, ft_acc)


if __name__ == '__main__':
    main()
