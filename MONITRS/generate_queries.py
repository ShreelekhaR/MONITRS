"""
MONITRS v2 — Generate disaster search queries with Gemini

Asks Gemini for new search queries, each about a natural disaster in a
randomly chosen country or region, written in the local language and in
English. The events are hypothetical: the point is to search for any event
that could plausibly have news coverage, even ones neither we nor Gemini know
about. Queries follow the same style as get_intl_articles_new.py:

    {disaster} {location} {country} {month} {year}

Usage:
    python MONITRS/generate_queries.py
    python MONITRS/generate_queries.py --n 50 --output Data/generated_queries.csv

Output:
    Data/generated_queries.csv
"""

import pandas as pd
import os
import json
import argparse

from google import genai

gemini_client = genai.Client()
model_name = "gemini-3.5-flash-lite"

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


def build_prompt(n, start_year=2018, end_year=2025):
    return f"""
Generate {n} new news search queries about natural disasters.

For each query, randomly select a country (and a region, province, state or
city within it), a type of natural disaster, and a month and year between
{start_year} and {end_year}. Spread the queries across all continents and many
different countries; do not repeat a country more than twice. Disaster types
to choose from: flood, storm (hurricane, typhoon, cyclone, tornado, severe
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


def generate_queries(n, start_year, end_year):
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=build_prompt(n, start_year, end_year),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--start-year', type=int, default=2018)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--output', type=str, default='Data/generated_queries.csv')
    args = parser.parse_args()

    queries = generate_queries(args.n, args.start_year, args.end_year)

    df = pd.DataFrame(queries).reindex(columns=FIELDS)

    # Same whitespace cleanup as get_intl_articles_new.py
    for col in ["local_query", "english_query"]:
        df[col] = df[col].fillna("").astype(str).map(lambda q: " ".join(q.split()))

    df = df[(df["local_query"] != "") & (df["english_query"] != "")]
    df = df.drop_duplicates(subset=["english_query"])

    if len(df) < args.n:
        print(f"Warning: asked for {args.n} queries, got {len(df)} usable ones")

    print(f"Countries: {df['english_country'].nunique()}")
    print(df["disaster_type"].value_counts().to_string())
    print()
    print(df[["local_query", "english_query"]].head(10).to_string(index=False))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.insert(0, "query_id", range(1, len(df) + 1))
    df.to_csv(args.output, index=False)

    print(f"\nDone. {len(df)} queries written to {args.output}")


if __name__ == "__main__":
    main()
