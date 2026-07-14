"""
Upload MONITRS v2 dataset to GCS.

Usage:
    python upload_to_gcs.py
    python upload_to_gcs.py --bucket lekhas-new-bucket --prefix monitrs
"""

import os
import argparse
import json
from google.cloud import storage
from tqdm import tqdm


def upload_dir(bucket, local_dir, gcs_prefix, extensions=None):
    """Upload a directory to GCS, skipping existing files."""
    count = 0
    skipped = 0
    for root, dirs, files in os.walk(local_dir):
        for fname in sorted(files):
            if extensions and not any(fname.endswith(e) for e in extensions):
                continue
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            gcs_path = f"{gcs_prefix}/{rel_path}"

            blob = bucket.blob(gcs_path)
            if blob.exists():
                skipped += 1
                continue
            blob.upload_from_filename(local_path)
            count += 1

            if (count + skipped) % 100 == 0:
                print(f"  {count} uploaded, {skipped} skipped", flush=True)

    return count, skipped


def upload_file(bucket, local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  {gcs_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='lekhas-new-bucket')
    parser.add_argument('--prefix', default='monitrs')
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    prefix = args.prefix

    print(f"Uploading to gs://{args.bucket}/{prefix}/\n")

    # 1. Events processed (metadata + captions)
    print("1. Events metadata:")
    for f in ['Data/events_processed.json', 'Data/events_processed_0_5000.json',
              'Data/events_processed_5000_end.json']:
        if os.path.exists(f):
            upload_file(bucket, f, f"{prefix}/{f}")

    # 2. QA data
    print("\n2. QA data:")
    qa_files = [
        'new_train_multiple_choice.json', 'new_test_multiple_choice.json',
        'train_generated_q_a.json', 'test_generated_q_a.json',
        'train_generated_multiple_choice_q_a.json', 'test_generated_multiple_choice_q_a.json',
        'train_total.json', 'test_total.json',
    ]
    for f in qa_files:
        if os.path.exists(f):
            upload_file(bucket, f, f"{prefix}/qa/{f}")

    # 3. Geocode cache
    if os.path.exists('Data/geocode_cache.json'):
        print("\n3. Geocode cache:")
        upload_file(bucket, 'Data/geocode_cache.json', f"{prefix}/Data/geocode_cache.json")

    # 4. GeoParquet
    if os.path.exists('Data/monitrs_v2.geoparquet'):
        print("\n4. GeoParquet:")
        upload_file(bucket, 'Data/monitrs_v2.geoparquet', f"{prefix}/Data/monitrs_v2.geoparquet")

    # 5. Images (biggest upload)
    print("\n5. Images:")
    if os.path.isdir('Data/images'):
        n_folders = len(os.listdir('Data/images'))
        print(f"  {n_folders} event folders to upload...")
        count, skipped = upload_dir(bucket, 'Data/images', f"{prefix}/Data/images",
                                     extensions=['.png', '.jpg'])
        print(f"  Done: {count} uploaded, {skipped} already existed")

    # 6. FEMA data + articles
    print("\n6. Source data:")
    for f in ['Data/FEMA_filtered.csv', 'Data/articles.csv']:
        if os.path.exists(f):
            upload_file(bucket, f, f"{prefix}/{f}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Upload complete to gs://{args.bucket}/{prefix}/")
    print(f"  Events: Data/events_processed.json")
    print(f"  QA: qa/*.json")
    print(f"  Images: Data/images/")


if __name__ == '__main__':
    main()
