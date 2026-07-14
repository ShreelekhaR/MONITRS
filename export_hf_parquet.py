"""
Export MONITRS v2 as parquet files for HuggingFace.

Creates:
  - monitrs_v2_events.parquet: event metadata + captions + bbox
  - monitrs_v2_qa_train.parquet: training QA pairs
  - monitrs_v2_qa_test.parquet: test QA pairs

Usage:
    pip install pandas pyarrow
    python export_hf_parquet.py
"""

import json
import os
import re
import pandas as pd


def parse_captions(caption_text):
    captions = []
    if not caption_text:
        return captions
    for line in caption_text.strip().split('\n'):
        match = re.match(r'(\d{4}-\d{2}-\d{2}):\s*(.*)', line.strip())
        if match:
            captions.append({'date': match.group(1), 'caption': match.group(2)})
    return captions


def export_events():
    print("Exporting events...")
    r = json.load(open('Data/events_processed.json'))

    rows = []
    for eid, v in sorted(r.items(), key=lambda x: int(x[0])):
        if 'error' in v:
            continue
        center = v['center']
        hw = v.get('halfwidth', 0.05)
        captions = parse_captions(v.get('captions', ''))

        # Count images
        img_dir = f"Data/images/{eid}"
        n_images = 0
        image_dates = []
        if os.path.isdir(img_dir):
            for f in os.listdir(img_dir):
                if f.endswith('.png') or f.endswith('.jpg'):
                    n_images += 1
                    m = re.search(r'(\d{4}-\d{2}-\d{2})', f)
                    if m:
                        image_dates.append(m.group(1))

        rows.append({
            'event_id': int(eid),
            'event_name': v.get('event', ''),
            'event_type': v.get('type', ''),
            'state': v.get('state', ''),
            'county': v.get('county', ''),
            'start_date': v.get('start_date', ''),
            'end_date': v.get('end_date', ''),
            'center_lat': center[0],
            'center_lon': center[1],
            'bbox_south': center[0] - hw,
            'bbox_north': center[0] + hw,
            'bbox_west': center[1] - hw,
            'bbox_east': center[1] + hw,
            'strategy': v.get('strategy', ''),
            'n_images': n_images,
            'image_dates': json.dumps(sorted(set(image_dates))),
            'n_captions': len(captions),
            'captions': json.dumps(captions),
            'n_location_events': len(v.get('location_events', [])),
            'n_articles': v.get('num_articles_scraped', 0),
        })

    df = pd.DataFrame(rows)
    out = 'Data/monitrs_v2_events.parquet'
    df.to_parquet(out, index=False)
    print(f"  {out}: {len(df)} events, {os.path.getsize(out)/1e6:.1f} MB")
    return df


def export_qa(split='train'):
    print(f"Exporting {split} QA...")
    data = json.load(open(f'{split}_total.json'))

    rows = []
    for item in data:
        convos = item.get('conversations', [])
        question = convos[0]['value'] if len(convos) > 0 else ''
        answer = convos[1]['value'] if len(convos) > 1 else ''

        # Clean question
        question = question.replace('<video>', '').strip()
        question = re.sub(r'^This is a sequence of .*?:\n', '', question).strip()

        rows.append({
            'qa_id': item.get('id', 0),
            'event_id': item.get('folder_id', 0),
            'task': item.get('task', ''),
            'question': question,
            'answer': answer,
            'split': split,
        })

    df = pd.DataFrame(rows)
    out = f'Data/monitrs_v2_qa_{split}.parquet'
    df.to_parquet(out, index=False)
    print(f"  {out}: {len(df)} questions, {os.path.getsize(out)/1e6:.1f} MB")
    return df


def main():
    events_df = export_events()
    train_df = export_qa('train')
    test_df = export_qa('test')

    print(f"\n{'='*50}")
    print(f"MONITRS v2 Parquet Export")
    print(f"{'='*50}")
    print(f"Events: {len(events_df)}")
    print(f"QA Train: {len(train_df)}")
    print(f"QA Test: {len(test_df)}")
    print(f"\nBy event type:")
    print(events_df['event_type'].value_counts().to_string())
    print(f"\nBy QA task:")
    all_qa = pd.concat([train_df, test_df])
    print(all_qa['task'].value_counts().to_string())


if __name__ == '__main__':
    main()
