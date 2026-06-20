"""
Test using Gemini with Google Search grounding to fetch disaster articles
instead of scraping URLs directly.

Usage:
    export GCP_PROJECT_ID=your-project-id
    python test_gemini_search.py
"""

import os
import json
import pandas as pd
from google import genai
from google.genai.types import Tool, GoogleSearch

PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-project-id')
LOCATION = os.environ.get('GCP_LOCATION', 'us-central1')

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
MODEL = "gemini-2.5-flash"

google_search_tool = Tool(google_search=GoogleSearch())


def search_disaster(event_name, event_type, state, county, start_date, end_date):
    prompt = f"""
    Search for news articles about this natural disaster and provide a detailed summary:

    Event: {event_name}
    Type: {event_type}
    Location: {county}, {state}
    Dates: {start_date} to {end_date}

    Find and summarize articles covering:
    1. What happened — specific locations affected, extent of damage
    2. Timeline — key dates and progression of the event
    3. Impact — acres burned, structures destroyed, people displaced, roads closed
    4. Visual changes — what would be visible from satellite imagery

    Provide specific facts, numbers, and place names from the articles you find.
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={'tools': [google_search_tool]},
    )
    return response.text


def main():
    df = pd.read_csv('Data/FEMA_filtered.csv', header=0)

    # Pick 10 diverse events
    test_indices = [0, 1, 5, 6, 9, 50, 130, 280, 1661, 2656]

    results = {}
    for idx in test_indices:
        row = df[df['index'] == idx].iloc[0]
        print(f"\n{'='*60}")
        print(f"EVENT {idx}: {row['declarationTitle']}")
        print(f"  {row['incidentType']} | {row['state']} | {row['designatedArea']}")
        print(f"  {row['incidentBeginDate']} to {row['incidentEndDate']}")

        try:
            content = search_disaster(
                row['declarationTitle'], row['incidentType'],
                row['state'], row['designatedArea'],
                row['incidentBeginDate'], row['incidentEndDate'],
            )
            print(f"\n  Gemini response ({len(content)} chars):")
            print(f"  {content[:500]}...")
            results[str(idx)] = {
                'event': row['declarationTitle'],
                'type': row['incidentType'],
                'content': content,
                'chars': len(content),
            }
        except Exception as e:
            print(f"  [!] Failed: {e}")
            results[str(idx)] = {'error': str(e)}

    with open('Data/gemini_search_test.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    for idx, r in results.items():
        if 'error' in r:
            print(f"  Event {idx}: ERROR — {r['error'][:60]}")
        else:
            print(f"  Event {idx} ({r['event'][:30]}): {r['chars']} chars")


if __name__ == '__main__':
    main()
