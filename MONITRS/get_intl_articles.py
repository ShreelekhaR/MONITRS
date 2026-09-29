"""
MONITRS v2 — Article search using DuckDuckGo

Randomly selects 10 events from EM-DAT and searches for news articles
about each event from local news sources, falling back to English queries if needed.

Usage:
    python MONITRS/get_intl_articles.py

Output:
    Data/articles.csv
"""

import pandas as pd
from tqdm import tqdm
from ddgs import DDGS
from time import sleep
import os
import argparse

from google import genai

gemini_client = genai.Client()
model_name = "gemini-3.5-flash-lite"


def get_articles(search_query, ddgs_region, max_results=5):
    try:
        results = DDGS().news(
            search_query,
            region=ddgs_region,
            max_results=max_results
        )

        return [r['url'] for r in results if r.get('url')]

    except Exception as e:
        print(f"  Search failed: {e}")
        sleep(2)
        return []
    
    
def translate_event(
    gemini_client,
    model_name,
    disaster_subtype,
    event_name,
    country,
    month
):
    prompt = f"""
Translate the following disaster information into the primary language
used for news reporting in {country}.

Disaster subtype: {disaster_subtype}
Event name: {event_name}
Country: {country}
Month: {month}

Also determine the appropriate DuckDuckGo search region code for news
searches in this country. Return the country-language region code used
by DuckDuckGo, such as:
- Greece: gr-el
- Brazil: br-pt
- Guatemala: gt-es
- France: fr-fr
- Germany: de-de
- Spain: es-es
- China: cn-zh
- Japan: jp-jp

Return ONLY valid JSON:

{{
    "disaster_subtype": "...",
    "event_name": "...",
    "country": "...",
    "month": "...",
    "ddgs_region": "..."
}}

Use natural terminology used by local news organizations.
Preserve proper names when appropriate.
Do not provide any explanation.
"""

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        import json

        text = response.text.strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()

        return json.loads(text)

    except Exception as e:
        print(f"Error calling Gemini API: {e}")

        return {
            "disaster_subtype": disaster_subtype,
            "event_name": event_name,
            "country": country,
            "ddgs_region": "us-en"
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output', type=str, default='Data/articles.csv')
    args = parser.parse_args()

    df = pd.read_csv('MONITRS/emdat_events.csv')
    print(df.columns.tolist())

    print(f"Total EM-DAT events: {len(df)}")

    # Only use events with a complete start date
    df = df.dropna(subset=['Event Name', 'Country', 'Disaster Type', 'Start Year', 'Start Month', 'Start Day'])

    print(f"Events with valid start dates: {len(df)}")

    # Randomly select 10 events
    sample = df.sample(n=10, random_state=args.seed)

    print("\nSelected events:")
    for _, row in sample.iterrows():
        print(f"  {row['DisNo.']}: {row['Event Name']} ({row['Country']}) ({row['Disaster Type']})")

    eng_queries = []
    local_queries = []

    for _, row in sample.iterrows():
        dis_no = row['DisNo.']

        # Translate this event
        translated = translate_event(gemini_client, model_name, row['Disaster Subtype'], row['Event Name'], row['Country'], int(row['Start Month']))

        english_month = pd.to_datetime(
            f'{int(row["Start Year"])}-{int(row["Start Month"])}-01'
        ).strftime("%B")

        english_query = (
            f'{row["Event Name"]} '
            f'{row["Country"]} '
            f'{english_month} '
            f'{int(row["Start Year"])}'
        )

        local_query = (
            f'{translated["event_name"]} '
            f'{translated["country"]} '
            f'{translated["month"]} '
            f'{int(row["Start Year"])}'
        )

        ddgs_region = translated["ddgs_region"]
        eng_queries.append((dis_no, "english", ddgs_region, english_query))
        local_queries.append((dis_no, "local", ddgs_region, local_query))

    print(f"\nQueries to run: {len(eng_queries)}")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write header if file doesn't exist
    file_exists = os.path.exists(args.output)

    with open(args.output, 'a') as f:
        if not file_exists:
            f.write("DisNo.,url\n")

        for i, (dis_no, lang, ddgs_region, search_query) in enumerate(tqdm(local_queries)):
            print(f"\n[{dis_no}]")
            print(f"Region: {ddgs_region}")
            print(f"Query: {search_query}")

            local_query = search_query
            eng_query = eng_queries[i][3]

            # Try local search first
            links = get_articles(
                local_query,
                ddgs_region
            )
            lang = "local"

            # If local search found nothing, try English in US region
            if not links:
                print("  No local results. Trying English fallback...")

                links = get_articles(
                    eng_query,
                    "us-en"
                )
                lang = "eng"

            for link in links:
                f.write(f'"{dis_no}","{lang}","{link}"\n')

            f.flush()

            # Rate limit — DDG can throttle
            if (i + 1) % 5 == 0:
                sleep(2)

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()