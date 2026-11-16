# Scaling up the random-query pipeline

Every command below runs from the repo root (`MONITRS/`) with the `.venv`
environment (`source .venv/bin/activate` or prefix with `uv run`).

The geo outputs need the `geo` extra (geopandas, shapely):

```bash
uv sync --extra geo        # add --extra for any other extras you use; sync removes unlisted ones
```

```bash
export GEMINI_API_KEY=your-api-key           # steps 1, 3, 4
export EE_PROJECT_ID=your-earth-engine-project   # step 3 (image download)
```

---

## 1. Pipeline at a glance

```
countries.csv
   │  1. generate_queries_random.py
   ▼
Data/random_queries.csv
   │  2. search_generated_queries.py
   ▼
Data/random_articles.csv
   │  3. scrape_articles.py
   ▼
Data/random_events_processed.json      (resume checkpoint)
Data/random_events_processed.geojson   (locations)
Data/random_images/{query_id}_{k}/*.tif, *.png
   │  4. generate_article_captions.py
   ▼
Data/random_images/{query_id}_{k}/captions.npz
Data/random_dataset.geoparquet         (dataset index)
```

Keep the `random_` prefix on every file and folder. Query IDs restart at 1 in
each query file, so if two query sets share `Data/images/`, the image folders
(`{query_id}_{k}`) of one set overwrite the other's.

---

## 2. Paths to files to read

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `MONITRS/generate_queries_random.py` | `MONITRS/countries.csv` | `Data/random_queries.csv` |
| 2 | `MONITRS/search_generated_queries.py` | `Data/random_queries.csv` | `Data/random_articles.csv` |
| 3 | `MONITRS/scrape_articles.py` | `Data/random_articles.csv` | `Data/random_events_processed.json`, `Data/random_events_processed.geojson`, `Data/random_images/` |
| 4 | `MONITRS/generate_article_captions.py` | `Data/random_events_processed.geojson`, `Data/random_images/`, `Data/article_texts.json` (cache) | `Data/random_images/*/captions.npz`, `Data/random_dataset.geoparquet`, `Data/article_texts.json` |

### File contents

- **`countries.csv`**: `country, latitude, longitude, name` (244 rows). Only
  `name` is used.
- **`random_queries.csv`**: one row per query: `query_id, country,
  english_country, location, english_location, disaster_type, month,
  english_month, year, ddgs_region, local_query, english_query`.
- **`random_articles.csv`**: one row per article link: `query_id, country,
  lang, url`. `lang` is the search that found it: `local`, `eng` (news) or
  `local_web`, `eng_web` (web, not date-filtered).
- **`random_events_processed.json`**: keyed by `query_id`. Each event has
  `country`, `locations` (center `lat`/`lon`, `halfwidth`, `start_date`,
  `end_date`, `disaster_type`, `image_dir_id`, `image_dates`),
  `location_events` and `articles`. Failed events hold `{"error": ...}`.
  Only used to resume step 3.
- **`random_events_processed.geojson`**: one Feature per location (see
  section 4).
- **`random_images/{query_id}_{k}/`**: one folder per location `k` of an
  event, with per date `{id}_{YYYY-MM-DD}_{type}.tif` (all Sentinel-2 bands)
  and `.png` (RGB preview), plus `captions.npz` after step 4.
- **`random_dataset.geoparquet`**: one row per captioned location (see
  section 4).

---

## 3. Which scripts to run

### Step 1: Generate queries

```bash
python MONITRS/generate_queries_random.py --n 10000 --output Data/random_queries.csv
```

- Each query gets a random country.
- The run is split into prompts of at most 50 queries (`--batch-size`, which
  can't go above 50). 10,000 queries = 200 Gemini calls.
- Each batch is appended to the CSV as soon as it finishes. A failed batch is
  retried 3 times, then skipped. After Ctrl+C or a crash, the finished
  batches are kept; rerunning appends new queries and keeps numbering from
  the highest `query_id`.
- Duplicate `english_query` values are dropped, so the final count can be a
  little under `--n`.

### Step 2: Search for articles

```bash
python MONITRS/search_generated_queries.py \
    --input Data/random_queries.csv --output Data/random_articles.csv
```

- Up to 4 searches per query (local news → English
  news → local web → English web), stopping at the first that finds links.
- Resumable: rerun the same command and it skips every `query_id` up to the
  highest one already in the output.
- This is the slowest step at scale and DuckDuckGo rate-limits (searches
  start timing out). If that happens, stop, wait, and rerun.

### Step 3: Scrape articles, find locations, download images

```bash
python MONITRS/scrape_articles.py \
    --articles Data/random_articles.csv \
    --output Data/random_events_processed.json \
    --images-dir Data/random_images \
    --batch-size 10000
```

- `--batch-size` is the number of events per run (default 5). Set it to the
  number of queries to do them all in one run. Progress is saved after every
  event, so rerunning the same command continues where it stopped.
- Events that failed with an error are retried on the next run, except ones
  whose articles had nothing to extract (add `--retry-empty` for those).
- `--rgb-only` skips the all-band GeoTIFFs (much smaller, but you lose the
  multispectral data). `--no-images` skips Earth Engine entirely.
- Earth Engine sometimes returns `503 memory capacity exceeded` or `429`
  under load. Each GeoTIFF is retried 4 times (waiting 15/30/45s); if it
  still fails, the PNG is kept without its `.tif`. After the run, download
  the missing ones by rerunning the same command with `--fill-missing-tifs`.
- The GeoJSON is rewritten after every event, next to `--output` with a
  `.geojson` extension (change it with `--geojson`).
- There's no `--end`, so don't run two copies on the same output file.

### Step 4: Generate captions (aligned prompt)

```bash
python MONITRS/generate_article_captions.py \
    --input Data/random_events_processed.geojson \
    --output Data/random_dataset.geoparquet \
    --images-dir Data/random_images
```

- One Gemini call per location: one caption for each Sentinel-2 image date.
- Each location's captions are saved to its `captions.npz` as soon as they're
  done. The GeoParquet index is rebuilt from all `captions.npz` files at the
  end of the run, and after Ctrl+C.
- Resumable: locations that already have a `captions.npz` are skipped
  (`--overwrite` redoes them). Check the prompt first with
  `--dry-run --limit 1`. Rebuild only the index with `--index-only`.
- `--input` also accepts an older events JSON (made before the GeoJSON
  output), converted on the fly.

---

## 4. How to have it stored

### Disk space

The current `Data/images/` is 6.2 GB for 89 location folders, about
**70 MB per location** with GeoTIFFs. Many queries produce no location, but
plan for hundreds of GB for 10,000 queries. Put `Data/random_images/` on a
large disk or bucket, or use `--rgb-only`.

### Layout

| Data | File | Format |
|---|---|---|
| Queries | `Data/random_queries.csv` | CSV |
| Article links | `Data/random_articles.csv` | CSV |
| Locations | `Data/random_events_processed.geojson` | **GeoJSON** |
| Imagery | `Data/random_images/{id}/*.tif` (+ `.png` preview) | **GeoTIFF** |
| Captions | `Data/random_images/{id}/captions.npz` | **`.npz`** |
| Dataset index | `Data/random_dataset.geoparquet` | **GeoParquet** |

All coordinates are WGS84 longitude/latitude (EPSG:4326). The readers and
writers are in `MONITRS/geo_outputs.py`.

### GeoJSON for locations

A `FeatureCollection` with one `Feature` per location. The geometry is the
image footprint, a `Polygon` of center ± `halfwidth` degrees (the region the
images are downloaded for). Properties:

`image_dir_id, query_id, country, disaster_type, start_date, end_date, lat,
lon, halfwidth, radius_km, places, image_dates, article_urls`

```python
import geopandas as gpd
locations = gpd.read_file("Data/random_events_processed.geojson")
```

Opens directly in QGIS and web maps. `places`, `image_dates` and
`article_urls` are JSON arrays; `gpd.read_file` may load them as strings, so
use `json.load` if you need them as lists.

### GeoTIFF for imagery

`scrape_articles.py` writes one georeferenced all-band `.tif` per image date.
The `.png` files are RGB previews; the GeoTIFFs are the source of truth.
`--rgb-only` skips them.

### `.npz` for captions

One `captions.npz` per location, next to its images. All arrays are strings;
`dates[i]` goes with `captions[i]`:

| Array | Contents |
|---|---|
| `dates` | `YYYY-MM-DD` image dates that got a caption |
| `captions` | one caption per date |
| `article_urls` | articles the captions were written from |
| `article_published` | their publish dates (`""` if unknown) |

```python
import numpy as np
cap = np.load("Data/random_images/12_0/captions.npz")
cap["dates"][0], cap["captions"][0]
```

A date can be missing if the LLM skipped it, so match captions to images by
date, not by position.

### GeoParquet for the full dataset

One row per captioned location: the footprint `geometry`, the location
properties above, plus `tif_paths`, `png_paths`, `captions_path`,
`caption_dates`, `captions` and `article_urls`. One file to filter by region,
date or disaster type without opening any images:

```python
import geopandas as gpd
ds = gpd.read_parquet("Data/random_dataset.geoparquet")
floods = ds[ds.disaster_type == "flood"]
```
