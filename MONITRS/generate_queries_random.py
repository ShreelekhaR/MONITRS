"""
MONITRS v2 — Generate disaster search queries with Gemini

Picks a different random country from countries.csv for each query and asks
Gemini for new search queries, each about a natural disaster in a region of
its assigned country, written in the local language and in
English. The events are hypothetical: the point is to search for any event
that could plausibly have news coverage, even ones neither we nor Gemini know
about. Queries follow the same style as get_intl_articles_new.py:

    {disaster} {location} {country} {month} {year}

Large runs are split into prompts of at most --batch-size queries (default
50), and each batch is appended to the output file as soon as it finishes, so
a crash or Ctrl+C partway through a 10,000-query run keeps the finished
batches.

Usage:
    python MONITRS/generate_queries_random.py
    python MONITRS/generate_queries_random.py --n 10000 --output Data/random_queries.csv

Output:
    Data/generated_queries.csv — appended to if it exists, with query_id
    continuing from the highest one already in the file
"""

import pandas as pd
import os
import json
import time
import random
import argparse

from google import genai

gemini_client = genai.Client()
model_name = "gemini-3.5-flash-lite"

MAX_BATCH_SIZE = 50     # most queries asked for in one prompt
MAX_RETRIES = 3         # attempts per batch before skipping it

COUNTRIES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "countries.csv")

FIELDS = [
    "country",
    "english_country",
    "location",
    "english_location",
    "disaster_type",
    "month",
    "english_month",
    "year",
    "ddgs_region",
    "local_query",
    "english_query",
]


def pick_random_countries(n, path=COUNTRIES_CSV):
    # keep_default_na=False so Namibia's code "NA" isn't read as missing
    names = pd.read_csv(path, keep_default_na=False)["name"].tolist()
    # Go through the countries in shuffled rounds, so with more queries than
    # countries every country is used equally often (give or take one)
    countries = []
    while len(countries) < n:
        countries += random.sample(names, len(names))
    return countries[:n]


def build_prompt(countries, start_year=2018, end_year=2025):
    n = len(countries)
    country_list = "\n".join(f"{i}. {c}" for i, c in enumerate(countries, 1))
    return f"""
Generate {n} new news search queries about natural disasters, one for each
country in this list, in the same order:

{country_list}

For each query, randomly select a region, province, state or city within its
country, a type of natural disaster that could plausibly happen there, and a
month and year between {start_year} and {end_year}. Disaster types to choose from: flood, storm (hurricane, typhoon, cyclone, tornado, severe
storm), extreme temperature (heatwave, cold wave), earthquake, wildfire,
drought, volcanic eruption.

The events are hypothetical: they do not need to be disasters you know
happened. The queries will be used to search for any event that could have
had news coverage, including ones you are not aware of.

Write each query twice: once in the primary language used for news reporting
in that country (local_query) and once in English (english_query). Both must
follow this exact pattern, with no quotes, parentheses or punctuation:

    <disaster> <location> <country> <month> <year>

Write the disaster the way a news headline would (e.g. "floods",
"earthquake", "heatwave", "typhoon"). Do not invent event names such as
storm names, since a made-up name will not match real articles. Use short,
natural country names the way local news would write them (e.g. "UK" not
"United Kingdom of Great Britain and Northern Ireland", "Iran" not
"Iran (Islamic Republic of)").

Examples:
- local_query: "Hochwasser Ahrweiler Deutschland Juli 2021"
  english_query: "floods Ahrweiler Germany July 2021"
- local_query: "颱風 台東 台灣 10月 2024"
  english_query: "typhoon Taitung Taiwan October 2024"
- local_query: "floods Kerala India August 2018"
  english_query: "floods Kerala India August 2018"

Also give the DuckDuckGo search region code for news searches in that
country, such as: Greece: gr-el, Brazil: br-pt, France: fr-fr,
Germany: de-de, Spain: es-es, China: cn-zh, Japan: jp-jp, India: in-en.

Return ONLY a valid JSON list of {n} objects, each with these keys:

{{
    "country": "<country in the local language>",
    "english_country": "...",
    "location": "<location in the local language>",
    "english_location": "...",
    "disaster_type": "<flood, storm, extreme temperature, earthquake, wildfire, drought or volcanic activity>",
    "month": "<month in the local language>",
    "english_month": "...",
    "year": <number from {start_year} to {end_year}>,
    "ddgs_region": "...",
    "local_query": "...",
    "english_query": "..."
}}

Do not provide any explanation.
"""


def generate_queries(countries, start_year, end_year):
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=build_prompt(countries, start_year, end_year),
        config={
            "response_mime_type": "application/json",
            # Higher temperature for more varied countries and disasters
            "temperature": 1.0,
        }
    )

    text = next(
        part.text
        for part in response.candidates[0].content.parts
        if part.text
    ).strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    queries = json.loads(text)
    if not isinstance(queries, list):
        raise ValueError(f"Expected a JSON list, got {type(queries).__name__}")
    return queries


def clean_queries(queries):
    """Queries as a DataFrame with whitespace cleaned and empty or duplicate queries dropped."""
    df = pd.DataFrame(queries).reindex(columns=FIELDS)

    # Same whitespace cleanup as get_intl_articles_new.py
    for col in ["local_query", "english_query"]:
        df[col] = df[col].fillna("").astype(str).map(lambda q: " ".join(q.split()))

    df = df[(df["local_query"] != "") & (df["english_query"] != "")]
    return df.drop_duplicates(subset=["english_query"])


def generate_batch(countries, start_year, end_year):
    """Queries for one batch of countries, retrying failed calls; None if every attempt fails."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return generate_queries(countries, start_year, end_year)
        except Exception as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50, help='Total number of queries to generate')
    parser.add_argument('--batch-size', type=int, default=MAX_BATCH_SIZE,
                        help=f'Queries per prompt (at most {MAX_BATCH_SIZE})')
    parser.add_argument('--start-year', type=int, default=2018)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--output', type=str, default='Data/generated_queries.csv')
    parser.add_argument('--countries', type=str, default=COUNTRIES_CSV)
    args = parser.parse_args()

    batch_size = max(1, min(args.batch_size, MAX_BATCH_SIZE))
    countries = pick_random_countries(args.n, args.countries)
    batches = [countries[i:i + batch_size] for i in range(0, len(countries), batch_size)]
    print(f"Generating {args.n} queries in {len(batches)} prompt(s) of up to {batch_size}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Append to an existing file: continue numbering from its highest
    # query_id and drop queries it already has
    seen = set()
    next_id = 1
    write_header = True
    if os.path.exists(args.output):
        existing = pd.read_csv(args.output)
        if len(existing):
            seen = set(existing["english_query"])
            next_id = int(existing["query_id"].max()) + 1
            write_header = False
    first_id = next_id

    added = []
    failed = 0
    try:
        for b, batch in enumerate(batches, 1):
            print(f"\n[{b}/{len(batches)}] {', '.join(batch)}")
            queries = generate_batch(batch, args.start_year, args.end_year)
            if queries is None:
                print("  Skipping this batch")
                failed += len(batch)
                continue

            df = clean_queries(queries)
            before = len(df)
            df = df[~df["english_query"].isin(seen)]
            if len(df) < before:
                print(f"  Dropped {before - len(df)} duplicate queries")
            if len(df) < len(batch):
                print(f"  Warning: asked for {len(batch)} queries, got {len(df)} usable ones")

            df.insert(0, "query_id", range(next_id, next_id + len(df)))
            df.to_csv(args.output, mode="a", index=False, header=write_header)
            write_header = False
            next_id += len(df)
            seen.update(df["english_query"])
            added.append(df)
            print(f"  Appended {len(df)} queries")
    except KeyboardInterrupt:
        print("\n\nInterrupted. Finished batches are saved.")

    if not added:
        print(f"\nNo queries added to {args.output}")
        return

    df = pd.concat(added)
    print(f"\nCountries: {df['english_country'].nunique()}")
    print(df["disaster_type"].value_counts().to_string())
    print()
    print(df[["local_query", "english_query"]].head(10).to_string(index=False))
    if failed:
        print(f"\n{failed} queries skipped because their batch failed")
    print(f"\nDone. {len(df)} queries appended to {args.output} "
          f"(query_id {first_id} to {next_id - 1})")


if __name__ == "__main__":
    main()
