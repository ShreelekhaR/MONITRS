"""
Export MONITRS v2 dataset as GeoParquet for HuggingFace.
Includes bounding boxes, event metadata, image paths, and QA data.

Usage:
    pip install geopandas pyarrow
    python export_geoparquet.py
"""

import json
import os
import re
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point


RESULTS_FILE = 'Data/events_processed.json'
IMAGES_DIR = 'Data/images'
QA_TRAIN = 'train_total.json'
QA_TEST = 'test_total.json'
OUTPUT = 'Data/monitrs_v2.geoparquet'


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

    # Load QA data and index by folder_id
    qa_by_event = {}
    for qa_file in [QA_TRAIN, QA_TEST]:
        if os.path.exists(qa_file):
            qa_data = json.load(open(qa_file))
            for item in qa_data:
                eid = str(item.get('folder_id', ''))
                if eid not in qa_by_event:
                    qa_by_event[eid] = []
                qa_by_event[eid].append(item)
            print(f"  {qa_file}: {len(qa_data)} questions")

    rows = []
    for eid, edata in events.items():
        if 'error' in edata:
            continue

        center = edata['center']
        halfwidth = edata.get('halfwidth', 0.05)

        # Bounding box
        bbox = box(
            center[1] - halfwidth,  # min lon
            center[0] - halfwidth,  # min lat
            center[1] + halfwidth,  # max lon
            center[0] + halfwidth,  # max lat
        )

        # Image files
        images = get_image_files(eid)
        image_dates = []
        for img in images:
            m = re.search(r'(\d{4}-\d{2}-\d{2})', img)
            if m:
                image_dates.append(m.group(1))

        # Captions
        captions = parse_captions(edata.get('captions', ''))

        # Location events
        loc_events = edata.get('location_events', [])
        visual_events = [le for le in loc_events if le.get('type') == 'visual']
        contextual_events = [le for le in loc_events if le.get('type') != 'visual']

        # QA count
        n_qa = len(qa_by_event.get(eid, []))

        rows.append({
            'event_id': int(eid),
            'event_name': edata.get('event', ''),
            'event_type': edata.get('type', ''),
            'state': edata.get('state', ''),
            'county': edata.get('county', ''),
            'start_date': edata.get('start_date', ''),
            'end_date': edata.get('end_date', ''),
            'center_lat': center[0],
            'center_lon': center[1],
            'fema_lat': edata.get('fema_center', [0, 0])[0],
            'fema_lon': edata.get('fema_center', [0, 0])[1],
            'halfwidth': halfwidth,
            'strategy': edata.get('strategy', ''),
            'firms_hotspots': edata.get('firms_hotspots_count', 0) if edata.get('firms') else 0,
            'n_images': len(images),
            'n_image_dates': len(set(image_dates)),
            'image_dates': json.dumps(sorted(set(image_dates))),
            'image_paths': json.dumps(images),
            'n_visual_events': len(visual_events),
            'n_contextual_events': len(contextual_events),
            'location_events': json.dumps(loc_events),
            'captions': json.dumps(captions),
            'n_qa_questions': n_qa,
            'n_articles_scraped': edata.get('num_articles_scraped', 0),
            'geometry': bbox,
        })

    print(f"\nBuilding GeoDataFrame: {len(rows)} events")
    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')

    # Sort by event_id
    gdf = gdf.sort_values('event_id').reset_index(drop=True)

    # Save
    gdf.to_parquet(OUTPUT)
    print(f"Saved: {OUTPUT}")
    print(f"  Size: {os.path.getsize(OUTPUT) / 1e6:.1f} MB")

    # Summary
    print(f"\nDataset Summary:")
    print(f"  Events: {len(gdf)}")
    print(f"  Images: {gdf['n_images'].sum()}")
    print(f"  QA questions: {gdf['n_qa_questions'].sum()}")
    print(f"  Event types: {gdf['event_type'].nunique()}")
    print(f"\n  By type:")
    for t, group in gdf.groupby('event_type'):
        print(f"    {t:<25} {len(group):>5} events, {group['n_images'].sum():>6} images")


if __name__ == '__main__':
    main()
