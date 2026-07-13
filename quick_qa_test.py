"""
Quick end-to-end test: generate QA for 10 events and visualize them.
All in one pass — no separate steps.

Usage:
    export GCP_PROJECT_ID=ai-sandbox-dev-f139
    python quick_qa_test.py
"""

import sys
import json
import os

sys.path.insert(0, 'MONITRS_QA')
from load_v2 import load_all_v1_format, get_image_paths
from generated_q_a import query_q_a, create_training_example as create_qa
from generated_mcq import query_multiple_choice_q_a, create_training_example as create_mcq

RESULTS_FILE = 'Data/events_processed.json'

# Load events
all_events = load_all_v1_format()
event_ids = sorted(all_events.keys(), key=int)

# Pick 10 events that have images
test_ids = []
for eid in event_ids:
    if get_image_paths(eid):
        test_ids.append(eid)
    if len(test_ids) >= 10:
        break

print(f"Testing {len(test_ids)} events with images\n")

all_qa = []
for eid in test_ids:
    edata = all_events[eid]
    captions = edata.get('captions', '')
    paths = get_image_paths(eid)

    print(f"Event {eid}: {edata.get('event_type', '?')} — {edata['events'][0]['event'][:50] if edata['events'] else '?'}")

    # Generate open-ended QA from captions
    qa_text = query_q_a(captions)
    if qa_text:
        print(f"  Open QA:")
        for line in qa_text.split('\n'):
            if line.strip().startswith('**'):
                print(f"    {line.strip()}")

    # Generate MCQ from captions
    mcq_text = query_multiple_choice_q_a(captions)
    if mcq_text:
        print(f"  MCQ:")
        for line in mcq_text.split('\n'):
            if line.strip().startswith('**') or line.strip().startswith('A)') or line.strip().startswith('B)') or line.strip().startswith('C)') or line.strip().startswith('D)'):
                print(f"    {line.strip()}")

    print()

# Now visualize these specific events
print("Generating visualizations...")
os.system(f"python visualize_qa.py --event {' '.join(test_ids)}")
