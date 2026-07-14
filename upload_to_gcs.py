"""
Upload MONITRS v2 dataset to GCS.
Zips images per event for fast transfer, uploads metadata + QA as-is.

Usage:
    python upload_to_gcs.py
    python upload_to_gcs.py --bucket lekhas-new-bucket --prefix monitrs
"""

import os
import argparse
import json
import zipfile
import tempfile
from google.cloud import storage


def upload_file(bucket, local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    if blob.exists():
        print(f"  [skip] {gcs_path}")
        return
    blob.upload_from_filename(local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  [ok] {gcs_path} ({size_mb:.1f} MB)")


def zip_and_upload_images(bucket, prefix, chunk_size=1000):
    """Zip images in chunks and upload to GCS."""
    img_dir = 'Data/images'
    if not os.path.isdir(img_dir):
        print("  No images directory found")
        return

    event_dirs = sorted([d for d in os.listdir(img_dir)
                         if os.path.isdir(os.path.join(img_dir, d))])

    print(f"  {len(event_dirs)} event folders to zip")

    for chunk_start in range(0, len(event_dirs), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(event_dirs))
        chunk_dirs = event_dirs[chunk_start:chunk_end]
        zip_name = f"images_{chunk_start:05d}_{chunk_end:05d}.zip"
        gcs_path = f"{prefix}/images/{zip_name}"

        blob = bucket.blob(gcs_path)
        if blob.exists():
            print(f"  [skip] {zip_name} (already uploaded)")
            continue

        print(f"  Zipping events {chunk_start}-{chunk_end}...", end=' ', flush=True)

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name

        n_files = 0
        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_STORED) as zf:
            for event_dir in chunk_dirs:
                full_dir = os.path.join(img_dir, event_dir)
                for fname in sorted(os.listdir(full_dir)):
                    if fname.endswith('.png') or fname.endswith('.jpg'):
                        file_path = os.path.join(full_dir, fname)
                        arc_name = f"{event_dir}/{fname}"
                        zf.write(file_path, arc_name)
                        n_files += 1

        size_mb = os.path.getsize(tmp_path) / 1e6
        print(f"{n_files} files, {size_mb:.0f} MB", flush=True)

        print(f"  Uploading {zip_name}...", flush=True)
        blob.upload_from_filename(tmp_path)
        os.remove(tmp_path)
        print(f"  [ok] {zip_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='lekhas-new-bucket')
    parser.add_argument('--prefix', default='monitrs')
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    prefix = args.prefix

    print(f"Uploading to gs://{args.bucket}/{prefix}/\n")

    # 1. Events metadata
    print("1. Events metadata:")
    for f in ['Data/events_processed.json']:
        if os.path.exists(f):
            upload_file(bucket, f, f"{prefix}/{f}")

    # 2. QA data
    print("\n2. QA data:")
    for f in ['train_total.json', 'test_total.json',
              'new_train_multiple_choice.json', 'new_test_multiple_choice.json',
              'train_generated_q_a.json', 'test_generated_q_a.json',
              'train_generated_multiple_choice_q_a.json', 'test_generated_multiple_choice_q_a.json']:
        if os.path.exists(f):
            upload_file(bucket, f, f"{prefix}/qa/{f}")

    # 3. Source data
    print("\n3. Source data:")
    for f in ['Data/FEMA_filtered.csv', 'Data/articles.csv',
              'Data/geocode_cache.json', 'Data/monitrs_v2.geoparquet']:
        if os.path.exists(f):
            upload_file(bucket, f, f"{prefix}/{f}")

    # 4. Images (zipped)
    print("\n4. Images (zipped):")
    zip_and_upload_images(bucket, prefix)

    print(f"\n{'='*50}")
    print(f"Done! gs://{args.bucket}/{prefix}/")


if __name__ == '__main__':
    main()
