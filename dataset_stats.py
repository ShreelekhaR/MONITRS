"""
Compute dataset statistics for MONITRS v2.

Usage:
    python dataset_stats.py
"""

import json
import os
import re
from datetime import datetime


def main():
    r = json.load(open('Data/events_processed.json'))

    imgs_per_event = []
    days_per_event = []
    imgs_by_type = {}

    img_dir = 'Data/images'

    for eid, v in r.items():
        if 'error' in v:
            continue

        # Count images
        event_dir = os.path.join(img_dir, str(eid))
        if os.path.isdir(event_dir):
            imgs = [f for f in os.listdir(event_dir) if f.endswith('.png') or f.endswith('.jpg')]
            n_imgs = len(imgs)

            # Get date span from image filenames
            dates = []
            for f in imgs:
                m = re.search(r'(\d{4}-\d{2}-\d{2})', f)
                if m:
                    dates.append(m.group(1))
            if dates:
                dates = sorted(set(dates))
                span = (datetime.strptime(dates[-1], '%Y-%m-%d') - datetime.strptime(dates[0], '%Y-%m-%d')).days
            else:
                span = 0
        else:
            n_imgs = 0
            span = 0

        if n_imgs > 0:
            imgs_per_event.append(n_imgs)
            days_per_event.append(span)

            etype = v.get('type', 'unknown')
            if etype not in imgs_by_type:
                imgs_by_type[etype] = {'imgs': [], 'days': []}
            imgs_by_type[etype]['imgs'].append(n_imgs)
            imgs_by_type[etype]['days'].append(span)

    print("MONITRS v2 Dataset Statistics")
    print("=" * 60)
    print(f"Total events with images: {len(imgs_per_event)}")
    print(f"Total images: {sum(imgs_per_event)}")
    print(f"Avg images per event: {sum(imgs_per_event)/len(imgs_per_event):.2f}")
    print(f"Avg days spanned per event: {sum(days_per_event)/len(days_per_event):.2f}")
    print(f"Median images per event: {sorted(imgs_per_event)[len(imgs_per_event)//2]}")
    print(f"Median days spanned: {sorted(days_per_event)[len(days_per_event)//2]}")
    print(f"Min/Max images: {min(imgs_per_event)} / {max(imgs_per_event)}")
    print(f"Min/Max days: {min(days_per_event)} / {max(days_per_event)}")

    print(f"\nBy event type:")
    print(f"{'Type':<25} {'Events':>7} {'Avg Imgs':>9} {'Avg Days':>9}")
    print("-" * 55)
    for t, data in sorted(imgs_by_type.items(), key=lambda x: -len(x[1]['imgs'])):
        n = len(data['imgs'])
        avg_i = sum(data['imgs']) / n
        avg_d = sum(data['days']) / n
        print(f"{t:<25} {n:>7} {avg_i:>9.2f} {avg_d:>9.2f}")


if __name__ == '__main__':
    main()
