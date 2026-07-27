"""
Benchmark suite: run multiple models on MONITRS test set, produce comparison tables.

Test set splits (from task field in test_total.json):
  - event_type              -> Table 2: Event Classification
  - temporal_grounding      -> Table 2: Temporal Grounding
  - location_identification -> Table 2: Location Grounding
  - multiple_choice         -> Table 3: Generated MCQ
  - custom                  -> Table 3: Open-Ended (BLEU/METEOR/ROUGE)

Models supported (via --models flag, comma-separated):
  - qwen-base       Baseline Qwen2.5-VL-7B-Instruct
  - qwen-ft         Finetuned (LoRA on top of base)
  - gemini          Gemini 2.0/2.5-flash via Vertex AI
  - teochat         TEOChat (requires separate install)

Usage:
    source ~/qwen-env/bin/activate
    pip install nltk rouge-score
    python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"

    # Small sanity run
    python Train/benchmark.py --models qwen-base,qwen-ft --n-per-task 20

    # Full benchmark
    python Train/benchmark.py --models qwen-base,qwen-ft,gemini --n-per-task 200 --out benchmark_results.json
"""

import json
import os
import re
import argparse
import random
from glob import glob
from collections import defaultdict


TASKS_MCQ = ['event_type', 'temporal_grounding', 'location_identification', 'multiple_choice']
TASK_OPEN = 'custom'


# ─── Data loading ────────────────────────────────────────────────────────────

def load_test_data(path='test_total.json'):
    """Load full test set (has 'task' field for splitting)."""
    with open(path) as f:
        return json.load(f)


def sample_by_task(data, n_per_task, seed=42):
    """Return {task: [samples]} with up to n_per_task per task."""
    by_task = defaultdict(list)
    for item in data:
        by_task[item.get('task', 'unknown')].append(item)

    print(f"Task distribution in test set:")
    for t, items in sorted(by_task.items()):
        print(f"  {t}: {len(items)}")

    random.seed(seed)
    sampled = {}
    for t, items in by_task.items():
        random.shuffle(items)
        sampled[t] = items[:n_per_task]
    return sampled


def extract_question_and_images(sample):
    convos = sample['conversations']
    question = convos[0]['value']
    answer = convos[1]['value']
    # Strip <video>, <image> tokens
    question = re.sub(r'<video>|<image>', '', question).strip()
    question = re.sub(r'^This is a sequence of .*?:\s*', '', question).strip()
    image_paths = [p for p in sample.get('video', []) if os.path.exists(p)][:6]
    return question, answer, image_paths


def extract_letter(text):
    m = re.search(r'\b([A-Da-d])\b', text)
    return m.group(1).lower() if m else '?'


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_mcq_accuracy(preds, gts):
    correct = sum(1 for p, g in zip(preds, gts)
                  if extract_letter(p) == extract_letter(g))
    return correct / len(preds) if preds else 0.0


def compute_open_metrics(preds, gts):
    """BLEU-1..4, METEOR, ROUGE-L."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    from rouge_score import rouge_scorer

    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bleus = [[] for _ in range(4)]
    meteors = []
    rouges = []

    for p, g in zip(preds, gts):
        ref_tokens = word_tokenize(g.lower())
        hyp_tokens = word_tokenize(p.lower())
        if not hyp_tokens or not ref_tokens:
            for i in range(4):
                bleus[i].append(0.0)
            meteors.append(0.0)
            rouges.append(0.0)
            continue

        for n in range(1, 5):
            weights = tuple([1/n]*n + [0]*(4-n))
            bleus[n-1].append(sentence_bleu([ref_tokens], hyp_tokens,
                                            weights=weights, smoothing_function=smooth))
        try:
            meteors.append(meteor_score([ref_tokens], hyp_tokens))
        except Exception:
            meteors.append(0.0)
        rouges.append(rouge.score(g, p)['rougeL'].fmeasure)

    def avg(xs): return sum(xs) / len(xs) if xs else 0.0
    return {
        'BLEU-1': avg(bleus[0]),
        'BLEU-2': avg(bleus[1]),
        'BLEU-3': avg(bleus[2]),
        'BLEU-4': avg(bleus[3]),
        'METEOR': avg(meteors),
        'ROUGE-L': avg(rouges),
    }


# ─── Model backends ──────────────────────────────────────────────────────────

class QwenBackend:
    """Base or LoRA-finetuned Qwen2.5-VL."""
    def __init__(self, lora_ckpt=None, name='qwen'):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.torch = torch
        BASE = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"[{name}] Loading base model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, device_map="cuda:0")
        if lora_ckpt:
            from peft import PeftModel
            print(f"[{name}] Loading LoRA: {lora_ckpt}")
            self.model = PeftModel.from_pretrained(self.model, lora_ckpt)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(BASE)
        self.name = name

    def generate(self, question, image_paths, max_new_tokens=128):
        from PIL import Image
        content = [{"type": "image", "image": p} for p in image_paths]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to("cuda:0")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()

    def close(self):
        del self.model
        del self.processor
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()


class GeminiBackend:
    """Gemini via Vertex AI (uses GCP_PROJECT_ID env)."""
    def __init__(self, model_id='gemini-2.5-flash', name='gemini'):
        from google import genai
        from google.genai.types import HttpOptions
        project = os.environ.get('GCP_PROJECT_ID', 'ai-sandbox-dev-f139')
        self.client = genai.Client(vertexai=True, project=project, location='us-central1',
                                    http_options=HttpOptions(api_version='v1'))
        self.model_id = model_id
        self.name = name
        print(f"[{name}] Using Vertex AI project={project}, model={model_id}")

    def generate(self, question, image_paths, max_new_tokens=128):
        from google.genai import types
        from PIL import Image
        import io

        parts = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type='image/png'))
        parts.append(types.Part.from_text(text=question))

        try:
            resp = self.client.models.generate_content(
                model=self.model_id,
                contents=parts,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_new_tokens,
                    temperature=0.0,
                ),
            )
            return (resp.text or '').strip()
        except Exception as e:
            return f'[ERR: {e}]'

    def close(self):
        pass


class TEOChatBackend:
    """TEOChat placeholder — requires separate install."""
    def __init__(self, name='teochat'):
        raise NotImplementedError("TEOChat backend not yet implemented. "
                                  "Clone https://github.com/ermongroup/TEOChat and adapt.")


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_model(backend, samples_by_task, max_new_tokens_mcq=16, max_new_tokens_open=128):
    """Run one model on all task splits, return {task: [(pred, gt), ...]}."""
    results = {}
    for task, samples in samples_by_task.items():
        print(f"\n[{backend.name}] Task: {task} ({len(samples)} samples)")
        pairs = []
        max_tokens = max_new_tokens_open if task == TASK_OPEN else max_new_tokens_mcq
        for i, s in enumerate(samples, 1):
            q, gt, imgs = extract_question_and_images(s)
            if not imgs:
                continue
            try:
                pred = backend.generate(q, imgs, max_new_tokens=max_tokens)
            except Exception as e:
                pred = f'[ERR: {e}]'
            pairs.append((pred, gt))
            if i % 10 == 0:
                print(f"  {i}/{len(samples)}")
        results[task] = pairs
    return results


def score_results(results):
    """Compute per-task metrics."""
    scored = {}
    for task, pairs in results.items():
        preds = [p for p, _ in pairs]
        gts = [g for _, g in pairs]
        if task == TASK_OPEN:
            scored[task] = compute_open_metrics(preds, gts)
        else:
            scored[task] = {'accuracy': compute_mcq_accuracy(preds, gts)}
    return scored


def print_tables(all_scores):
    """Print Table 2 (MCQ classification/grounding) and Table 3 (Generated VQA)."""
    print(f"\n{'='*80}")
    print("Table 2: Multiple Choice Event Classification & Grounding")
    print(f"{'='*80}")
    header = f"{'Method':<20} {'Event Class':>12} {'Temporal':>12} {'Location':>12}"
    print(header)
    print('-' * len(header))
    for model_name, scores in all_scores.items():
        ec = scores.get('event_type', {}).get('accuracy', 0) * 100
        tg = scores.get('temporal_grounding', {}).get('accuracy', 0) * 100
        lg = scores.get('location_identification', {}).get('accuracy', 0) * 100
        print(f"{model_name:<20} {ec:>11.2f}% {tg:>11.2f}% {lg:>11.2f}%")

    print(f"\n{'='*80}")
    print("Table 3: Generated VQA")
    print(f"{'='*80}")
    header = f"{'Method':<20} {'MCQ Acc':>10} {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-3':>8} {'BLEU-4':>8} {'METEOR':>8} {'ROUGE-L':>8}"
    print(header)
    print('-' * len(header))
    for model_name, scores in all_scores.items():
        mcq = scores.get('multiple_choice', {}).get('accuracy', 0) * 100
        op = scores.get(TASK_OPEN, {})
        print(f"{model_name:<20} {mcq:>9.2f}% "
              f"{op.get('BLEU-1', 0):>8.4f} {op.get('BLEU-2', 0):>8.4f} "
              f"{op.get('BLEU-3', 0):>8.4f} {op.get('BLEU-4', 0):>8.4f} "
              f"{op.get('METEOR', 0):>8.4f} {op.get('ROUGE-L', 0):>8.4f}")


# ─── Main ────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_root='checkpoints/qwen2.5-vl-monitrs'):
    runs = sorted(glob(f'{ckpt_root}/v*'))
    if not runs:
        return None
    ckpts = sorted(glob(f'{runs[-1]}/checkpoint-*'),
                   key=lambda p: int(p.split('-')[-1]))
    return ckpts[-1] if ckpts else None


def build_backend(name, args):
    if name == 'qwen-base':
        return QwenBackend(lora_ckpt=None, name='Qwen2.5-VL-base')
    elif name == 'qwen-ft':
        ckpt = args.checkpoint or find_latest_checkpoint()
        if not ckpt:
            raise RuntimeError("No LoRA checkpoint found")
        return QwenBackend(lora_ckpt=ckpt, name='Ours (Qwen2.5-VL-ft)')
    elif name == 'gemini':
        return GeminiBackend(model_id=args.gemini_model, name='Gemini 2.5-flash')
    elif name == 'teochat':
        return TEOChatBackend()
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='qwen-base,qwen-ft',
                        help='Comma-separated: qwen-base,qwen-ft,gemini,teochat')
    parser.add_argument('--n-per-task', type=int, default=100)
    parser.add_argument('--test-file', default='test_total.json')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--gemini-model', default='gemini-2.5-flash')
    parser.add_argument('--out', default='benchmark_results.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data = load_test_data(args.test_file)
    samples_by_task = sample_by_task(data, args.n_per_task, seed=args.seed)

    model_names = [m.strip() for m in args.models.split(',')]
    all_scores = {}
    all_raw = {}

    for name in model_names:
        print(f"\n{'#'*80}\n# Running {name}\n{'#'*80}")
        backend = build_backend(name, args)
        raw = run_model(backend, samples_by_task)
        scored = score_results(raw)
        all_scores[backend.name] = scored
        all_raw[backend.name] = {t: [{'pred': p, 'gt': g} for p, g in pairs]
                                 for t, pairs in raw.items()}
        backend.close()

    print_tables(all_scores)

    with open(args.out, 'w') as f:
        json.dump({'scores': all_scores, 'raw': all_raw, 'n_per_task': args.n_per_task}, f, indent=2)
    print(f"\nResults saved: {args.out}")


if __name__ == '__main__':
    main()
