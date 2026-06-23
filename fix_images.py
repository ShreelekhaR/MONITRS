"""
Fix corrupt PNG files — re-save JPEGs that were named .png as actual PNGs.
Removes truly broken files.

Usage:
    python fix_images.py
"""

from PIL import Image
import os

ODIR = 'Data/images'

fixed = 0
broken = 0
ok = 0

for root, dirs, files in os.walk(ODIR):
    for f in files:
        if not f.endswith('.png'):
            continue
        path = os.path.join(root, f)
        try:
            img = Image.open(path)
            if img.format != 'PNG':
                img.save(path, 'PNG')
                fixed += 1
            else:
                ok += 1
        except Exception:
            print(f'  broken: {path}')
            os.remove(path)
            broken += 1

print(f'Done: {fixed} converted, {broken} removed, {ok} already valid')
