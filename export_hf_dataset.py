"""
Export MONITRS v2 as a HuggingFace Dataset with viewable samples.
Creates a parquet file with embedded image paths, captions, QA pairs,
and event metadata that HF dataset viewer can render.

Usage:
    pip install datasets geopandas pyarrow Pillow
    python export_hf_dataset.py
"""

import json
import os
import re
from PIL import Image
from datasets import Dataset, Features, Value, Sequence, Image as HFImage
import pandas as pd


RESULTS_FILE = 'Data/events_processed.json'
IMAGES_DIR = 'Data/images'
QA_TRAIN = 'train_total.json'
QA_TEST = 'test_total.json'
OUTPUT_DIR = 'Data/hf_dataset'


def get_image_files(event_id):
    paths = []
    for suffix in ['', '_firms', '_llm', '_fema']:
        img_dir = os.path.join(IMAGES_DIR, f"{event_id}{suffix}")
        if os.path.isdir(img_dir):
            for f in sorted(os.listdir(img_dir)):
                if f.endswith('.png') or f.endswith('.jpg'):
                    paths.append(os.path.join(img_dir, f))
            break
    return paths


def parse_captions(caption_text):
    captions = {}
    if not caption_text:
        return captions
    for line in caption_text.strip().split('\n'):
        match = re.match(r'(\d{4}-\d{2}-\d{2}):\s*(.*)', line.strip())
        if match:
            captions[match.group(1)] = match.group(2)
    return captions


def main():
    print("Loading events...")
    with open(RESULTS_FILE) as f:
        events = json.load(f)

    # Load QA data indexed by folder_id
    qa_by_event = {}
    for qa_file in [QA_TRAIN, QA_TEST]:
        if os.path.exists(qa_file):
            split = 'train' if 'train' in qa_file else 'test'
            qa_data = json.load(open(qa_file))
            for item in qa_data:
                eid = str(item.get('folder_id', ''))
                if eid not in qa_by_event:
                    qa_by_event[eid] = []
                item['split'] = split
                qa_by_event[eid].append(item)
            print(f"  {qa_file}: {len(qa_data)} questions")

    # Build per-image rows (one row per image for HF viewer)
    print("Building dataset rows...")
    rows = []
    for eid, edata in sorted(events.items(), key=lambda x: int(x[0])):
        if 'error' in edata:
            continue

        center = edata['center']
        halfwidth = edata.get('halfwidth', 0.05)
        images = get_image_files(eid)
        captions = parse_captions(edata.get('captions', ''))
        event_qa = qa_by_event.get(eid, [])

        if not images:
            continue

        for img_path in images:
            # Extract date from filename
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', img_path)
            img_date = date_match.group(1) if date_match else ''

            # Find nearest caption
            caption = captions.get(img_date, '')
            if not caption and captions:
                from datetime import datetime
                try:
                    img_dt = datetime.strptime(img_date, '%Y-%m-%d')
                    earlier = {d: c for d, c in captions.items()
                               if datetime.strptime(d, '%Y-%m-%d') <= img_dt}
                    if earlier:
                        caption = earlier[max(earlier.keys())]
                except ValueError:
                    pass

            # Determine phase
            if '_pre_' in img_path:
                phase = 'pre'
            elif '_post_' in img_path:
                phase = 'post'
            elif '_during_' in img_path:
                phase = 'during'
            else:
                phase = 'during'

            rows.append({
                'image': img_path,
                'event_id': int(eid),
                'event_name': edata.get('event', ''),
                'event_type': edata.get('type', ''),
                'state': edata.get('state', ''),
                'county': edata.get('county', ''),
                'date': img_date,
                'phase': phase,
                'caption': caption,
                'start_date': edata.get('start_date', ''),
                'end_date': edata.get('end_date', ''),
                'center_lat': center[0],
                'center_lon': center[1],
                'bbox_west': center[1] - halfwidth,
                'bbox_east': center[1] + halfwidth,
                'bbox_south': center[0] - halfwidth,
                'bbox_north': center[0] + halfwidth,
                'strategy': edata.get('strategy', ''),
                'n_qa_questions': len(event_qa),
            })

    print(f"Total rows (per-image): {len(rows)}")

    # Create HF Dataset
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ds = Dataset.from_dict({
        'image': [r['image'] for r in rows],
        'event_id': [r['event_id'] for r in rows],
        'event_name': [r['event_name'] for r in rows],
        'event_type': [r['event_type'] for r in rows],
        'state': [r['state'] for r in rows],
        'county': [r['county'] for r in rows],
        'date': [r['date'] for r in rows],
        'phase': [r['phase'] for r in rows],
        'caption': [r['caption'] for r in rows],
        'start_date': [r['start_date'] for r in rows],
        'end_date': [r['end_date'] for r in rows],
        'center_lat': [r['center_lat'] for r in rows],
        'center_lon': [r['center_lon'] for r in rows],
        'bbox_west': [r['bbox_west'] for r in rows],
        'bbox_east': [r['bbox_east'] for r in rows],
        'bbox_south': [r['bbox_south'] for r in rows],
        'bbox_north': [r['bbox_north'] for r in rows],
        'strategy': [r['strategy'] for r in rows],
        'n_qa_questions': [r['n_qa_questions'] for r in rows],
    })

    # Cast image column
    ds = ds.cast_column('image', HFImage())

    # Save
    ds.save_to_disk(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}/")

    # Also save as parquet for direct upload
    parquet_path = os.path.join(OUTPUT_DIR, 'monitrs_v2.parquet')
    ds.to_parquet(parquet_path)
    print(f"Parquet: {parquet_path} ({os.path.getsize(parquet_path) / 1e9:.2f} GB)")

    # Also export QA as separate parquet
    if qa_by_event:
        qa_rows = []
        for eid, qas in qa_by_event.items():
            for qa in qas:
                convos = qa.get('conversations', [])
                question = convos[0]['value'] if len(convos) > 0 else ''
                answer = convos[1]['value'] if len(convos) > 1 else ''
                qa_rows.append({
                    'event_id': int(eid),
                    'split': qa.get('split', 'train'),
                    'task': qa.get('task', ''),
                    'question': question,
                    'answer': answer,
                })
        qa_df = pd.DataFrame(qa_rows)
        qa_path = os.path.join(OUTPUT_DIR, 'monitrs_v2_qa.parquet')
        qa_df.to_parquet(qa_path)
        print(f"QA parquet: {qa_path} ({os.path.getsize(qa_path) / 1e6:.1f} MB)")

    # Summary
    print(f"\n{'='*50}")
    print(f"MONITRS v2 HuggingFace Dataset")
    print(f"{'='*50}")
    print(f"Events:      {len(set(r['event_id'] for r in rows))}")
    print(f"Images:      {len(rows)}")
    print(f"QA pairs:    {sum(r['n_qa_questions'] for r in rows if r['phase'] == 'during') // max(1, len([r for r in rows if r['phase'] == 'during']))}")
    print(f"Event types: {len(set(r['event_type'] for r in rows))}")


if __name__ == '__main__':
    main()
