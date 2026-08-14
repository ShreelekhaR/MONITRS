"""Quick sanity check: run Gemini on one sample per task."""
import sys
sys.path.insert(0, 'Train')
from benchmark import GeminiBackend, load_test_data, sample_by_task, extract_question_and_images

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'gemini-2.5-flash'
b = GeminiBackend(model_id=MODEL)

data = load_test_data()
samples = sample_by_task(data, 1)

for task, xs in samples.items():
    q, gt, imgs = extract_question_and_images(xs[0])
    if task != 'custom':
        q = q + "\nAnswer with a single letter (a, b, c, or d)."
    print(f"\n=== {task} ===")
    print(f"Q: {q[:200]}")
    print(f"GT: {gt[:100]}")
    print(f"Images: {len(imgs)}")
    pred = b.generate(q, imgs, max_new_tokens=64)
    print(f"Pred: {pred[:200]}")
