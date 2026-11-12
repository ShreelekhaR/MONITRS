"""
MONITRS v2 — Article search for Gemini-generated queries using DuckDuckGo

Runs the queries from generate_queries.py through the same DuckDuckGo search
as get_intl_articles_new.py. For each query, tries in order and stops at the
first search that finds links:

    1. local news search    (local query, local region, filtered to the date range)
    2. English news search  (English query, US region, filtered to the date range)
    3. local web search     (local query, local region)
    4. English web search   (English query, US region)

The date range runs from the start of the query's month to 30 days after the
end of that month, the same window get_intl_articles_new.py uses for events
with no end date. Web results have no date, so only the month and year in the
query keep them on the right event.

Usage:
    python MONITRS/search_generated_queries.py
    python MONITRS/search_generated_queries.py --input Data/generated_queries.csv --output Data/generated_articles.csv

Output:
    Data/generated_articles.csv — appended to; query_ids already in it (and
    any below its highest query_id) are skipped
"""

import pandas as pd
from tqdm import tqdm
from ddgs import DDGS
from time import sleep
import os
import csv
import argparse


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='Data/generated_queries.csv')
    parser.add_argument('--output', type=str, default='Data/generated_articles.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Queries: {len(df)}")

    # Skip queries already searched. A query that found no links writes no
    # rows, so also skip everything up to the highest query_id in the output:
    # queries run in query_id order, so those were all searched already.
    if os.path.exists(args.output):
        done = set(pd.read_csv(args.output)["query_id"])
        if done:
            last = max(done)
            df = df[~df["query_id"].isin(done) & (df["query_id"] > last)]
            print(f"Already searched up to query_id {last}, {len(df)} new queries to search")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write header if file doesn't exist
    file_exists = os.path.exists(args.output)

    with open(args.output, 'a', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["query_id", "country", "lang", "url"])

        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            query_id = row["query_id"]
            country = row["english_country"]
            local_query = row["local_query"]
            eng_query = row["english_query"]
            ddgs_region = row["ddgs_region"] if pd.notna(row["ddgs_region"]) else "us-en"

            # e.g. "July 2021" -> 2021-07-01 to 2021-08-30
            start_date = pd.to_datetime(f'{row["english_month"]} {int(row["year"])}', format="%B %Y", errors="coerce")
            if pd.isna(start_date):
                print(f"\n[{query_id}] Skipping, can't parse month: {row['english_month']} {row['year']}")
                continue
            start_date = start_date.tz_localize("UTC")
            end_date = start_date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=30)

            print(f"\n[{query_id}]")
            print(f"Region: {ddgs_region}")
            print(f"Local query: {local_query}")
            print(f"English query: {eng_query}")
            print(f"Dates: {start_date.date()} to {end_date.date()}")

            # News first (local, then English), then web search (local, then English)
            searches = [
                ("local", get_articles, local_query, ddgs_region),
                ("eng", get_articles, eng_query, "us-en"),
                ("local_web", get_web_articles, local_query, ddgs_region),
                ("eng_web", get_web_articles, eng_query, "us-en"),
            ]

            links = []
            tried = set()
            for lang, search_fn, query, region in searches:
                # In English-speaking countries the local and English searches
                # can be identical, so don't run the same search twice
                if (search_fn, query, region) in tried:
                    continue
                tried.add((search_fn, query, region))

                print(f"  Trying {lang} search...")
                links = search_fn(query, region, start_date, end_date)
                if links:
                    break

            for link in links:
                writer.writerow([query_id, country, lang, link])

            f.flush()

            # Rate limit — DDG can throttle
            if (i + 1) % 5 == 0:
                sleep(2)

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
