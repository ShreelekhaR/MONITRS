"""
MONITRS v2 — Article search using DuckDuckGo

Randomly selects 10 events from EM-DAT and searches for news articles
about each event. Tries local news, then local web search (which reaches older
articles than DDG News), then the same two searches in English if needed.

Usage:
    python MONITRS/get_intl_articles_new.py

Output:
    Data/ddg_articles.csv
"""

import pandas as pd
from tqdm import tqdm
from ddgs import DDGS
from time import sleep
import os
import csv
import re
import argparse

from google import genai

gemini_client = genai.Client()
model_name = "gemini-3.5-flash-lite"


def get_articles(search_query, ddgs_region, start_date, end_date, max_results=5):
    try:
        # DDG news has no custom date range (timelimit is only d/w/m/y), so
        # fetch extra results and keep the ones published during the event
        results = DDGS().news(
            search_query,
            region=ddgs_region,
            max_results=30
        )

        links = []
        for r in results:
            published = pd.to_datetime(r.get('date'), utc=True, errors='coerce')
            if r.get('url') and pd.notna(published) and start_date <= published <= end_date:
                links.append(r['url'])

        return links[:max_results]

    except Exception as e:
        print(f"  Search failed: {e}")
        sleep(2)
        return []


def get_web_articles(search_query, ddgs_region, start_date, end_date, max_results=5):
    # DDG News skews heavily towards recent articles, so older events often
    # have nothing in range. Web search reaches older pages, but results have
    # no date and DDG returns nothing for a custom timelimit range, so the
    # query's month and year are what keep them on the right event.
    # start_date/end_date are unused; they keep the signature the same as get_articles.
    try:
        results = DDGS().text(
            search_query,
            region=ddgs_region,
            max_results=max_results
        )
        return [r['href'] for r in results if r.get('href')]

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
    month,
    location
):
    # Many EM-DAT events (floods, earthquakes, droughts...) have no name
    if pd.isna(event_name):
        event_name = None
    if pd.isna(location):
        location = None

    prompt = f"""
Translate the following disaster information into the primary language
used for news reporting in {country}.

Disaster subtype: {disaster_subtype}
Event name: {event_name or "(none - unnamed event)"}
Country: {country}
Month: {month}
Affected locations: {location or "(unknown)"}

Make the disaster subtype, event name and country sound natural, the way
a news headline would write them. For example:
- "United Kingdom of Great Britain and Northern Ireland" -> "UK"
- "Iran (Islamic Republic of)" -> "Iran"
- "Flood (General)" -> "floods"
- "Ground movement" -> "earthquake"
- "Tropical cyclone 'Fred'" -> "Hurricane Fred" or "Storm Fred"
- "Typhoon 'Yinxing' (Marce)" -> "Typhoon Yinxing"

From the affected locations, pick the single most specific, well-known
place a news headline would mention (a city, province or state, e.g.
"Kerala" or "Durban"). Return it as "location" (local spelling) and
"english_location" (English spelling). Leave both empty ("") if unknown.
Fix spelling mistakes and remove quotes and parentheses from names.
If the event is unnamed, leave event_name empty ("").

Also give the English versions, cleaned up the same way, as
"english_event_name", "english_disaster_subtype" and "english_country".

Also write a short, natural search query (2-6 words, no dates) for this
event in the local language (local_query) and in English (english_query),
e.g. "Typhoon Kong-Rey Taiwan" or "Kerala floods".

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
    "location": "...",
    "english_event_name": "...",
    "english_disaster_subtype": "...",
    "english_country": "...",
    "english_location": "...",
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

        #text = response.text.strip()
        text = next(
          part.text
          for part in response.candidates[0].content.parts
          if part.text
        ).strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()

        return json.loads(text)

    except Exception as e:
        print(f"Error calling Gemini API: {e}")

        # First place in EM-DAT's free-text list, e.g. "Kerala state; Karnataka" -> "Kerala state"
        first_location = re.split(r"[;,(]", location)[0].strip() if location else ""

        return {
            "disaster_subtype": disaster_subtype,
            "event_name": event_name or "",
            "country": country,
            "month": "",  # main() falls back to the English month name
            "location": first_location,
            "english_location": first_location,
            "english_event_name": event_name or "",
            "english_disaster_subtype": disaster_subtype,
            "english_country": country,
            "ddgs_region": "us-en"
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output', type=str, default='Data/ddg_articles.csv')
    args = parser.parse_args()

    df = pd.read_csv('MONITRS/emdat_events.csv')
    print(df.columns.tolist())

    print(f"Total EM-DAT events: {len(df)}")

    # Only use events with a complete start date
    df = df.dropna(subset=['Country', 'Disaster Type', 'Start Year', 'Start Month'])

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
        translated = translate_event(gemini_client, model_name, row['Disaster Subtype'], row['Event Name'], row['Country'], int(row['Start Month']), row['Location'])

        english_month = pd.to_datetime(
            f'{int(row["Start Year"])}-{int(row["Start Month"])}-01'
        ).strftime("%B")

        # Unnamed events are searched by disaster subtype instead of name
        event_name = row["Event Name"] if pd.notna(row["Event Name"]) else row["Disaster Subtype"]
        local_name = translated.get("event_name") or translated.get("disaster_subtype") or event_name

        # Keep articles published from the start month until 30 days after the event ends
        start_date = pd.Timestamp(year=int(row["Start Year"]), month=int(row["Start Month"]), day=1, tz="UTC")
        if pd.notna(row["End Year"]) and pd.notna(row["End Month"]):
            end_date = pd.Timestamp(year=int(row["End Year"]), month=int(row["End Month"]), day=1, tz="UTC")
            end_date = end_date + pd.offsets.MonthEnd(0)
        else:
            end_date = start_date + pd.offsets.MonthEnd(0)
        end_date = end_date + pd.Timedelta(days=30)

        english_name = (
            translated.get("english_event_name")
            or translated.get("english_disaster_subtype")
            or event_name
        )
        english_query = (
            f'{english_name} '
            f'{translated.get("english_location") or ""} '
            f'{translated.get("english_country") or row["Country"]} '
            f'{english_month} '
            f'{int(row["Start Year"])}'
        )

        local_query = (
            f'{local_name} '
            f'{translated.get("location") or ""} '
            f'{translated.get("country") or row["Country"]} '
            f'{translated.get("month") or english_month} '
            f'{int(row["Start Year"])}'
        )

        # Collapse the double space left when there is no location
        english_query = " ".join(english_query.split())
        local_query = " ".join(local_query.split())

        ddgs_region = translated.get("ddgs_region", "us-en")
        eng_queries.append((dis_no, "english", ddgs_region, english_query, start_date, end_date))
        local_queries.append((dis_no, "local", ddgs_region, local_query, start_date, end_date))

    print(f"\nQueries to run: {len(eng_queries)}")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write header if file doesn't exist
    file_exists = os.path.exists(args.output)

    with open(args.output, 'a', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["DisNo.", "lang", "url"])

        for i, (dis_no, lang, ddgs_region, search_query, start_date, end_date) in enumerate(tqdm(local_queries)):
            print(f"\n[{dis_no}]")
            print(f"Region: {ddgs_region}")
            print(f"Query: {search_query}")
            print(f"Dates: {start_date.date()} to {end_date.date()}")

            local_query = search_query
            eng_query = eng_queries[i][3]

            # Try local news first, then local web search, then the same in English (US region)
            searches = [
                ("local", get_articles, local_query, ddgs_region),
                ("local_web", get_web_articles, local_query, ddgs_region),
                ("eng", get_articles, eng_query, "us-en"),
                ("eng_web", get_web_articles, eng_query, "us-en"),
            ]

            links = []
            for lang, search_fn, query, region in searches:
                print(f"  Trying {lang} search...")
                links = search_fn(query, region, start_date, end_date)
                if links:
                    break

            for link in links:
                writer.writerow([dis_no, lang, link])

            f.flush()

            # Rate limit — DDG can throttle
            if (i + 1) % 5 == 0:
                sleep(2)

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()