"""
MONITRS v2 — Article search using DuckDuckGo
Searches for news articles about each FEMA disaster event.

Usage:
    python MONITRS/get_articles.py

    # Process specific range:
    python MONITRS/get_articles.py --start 0 --end 1000

Output: Data/articles.csv (appends, resumable)
"""

import pandas as pd
from tqdm import tqdm
from ddgs import DDGS
from time import sleep
import os
import argparse


def get_articles(search_query, max_results=5):
    try:
        results = DDGS().text(search_query, max_results=max_results)
        return [r['href'] for r in results if r.get('href')]
    except Exception as e:
        print(f"  Search failed: {e}")
        sleep(2)
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--output', type=str, default='Data/articles.csv')
    args = parser.parse_args()

    df = pd.read_csv('Data/FEMA_filtered.csv', header=0)
    print(f"Total FEMA events: {len(df)}")

    # Check what's already been scraped
    already_done = set()
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            for line in f:
                try:
                    idx = int(line.split(',')[0])
                    already_done.add(idx)
                except ValueError:
                    continue
        print(f"Already scraped: {len(already_done)} events")

    queries = []
    for _, row in df.iterrows():
        idx = int(row['index'])
        if args.end and (idx < args.start or idx >= args.end):
            continue
        if idx < args.start:
            continue
        if idx in already_done:
            continue
        query = (f"{row['declarationTitle']} {row['incidentType']} "
                 f"{row['designatedArea']} {row['state']} {row['incidentBeginDate']}")
        queries.append((idx, query))

    print(f"Queries to run: {len(queries)}")

    f = open(args.output, 'a+')

    for i, (idx, search_query) in enumerate(tqdm(queries)):
        links = get_articles(search_query)

        for link in links:
            f.write(f"{idx},{link},\n")

        f.flush()

        # Rate limit — DDG can throttle
        if (i + 1) % 50 == 0:
            sleep(2)

    f.close()
    print(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
