"""
Test OSM location matching for a few events.

Usage:
    python test_osm_match.py
"""

import sys
sys.path.insert(0, 'MONITRS_QA')
from load_v2 import load_all_v1_format

events = load_all_v1_format()
with_locs = {k: v for k, v in events.items() if v['locations']}
print(f"{len(with_locs)} events with OSM-matched locations")

for k, v in list(with_locs.items())[:10]:
    locs = v['locations']
    loc_names = list(locs.keys())
    print(f"  Event {k}: {loc_names}")
