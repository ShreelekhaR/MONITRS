"""
Builds a caption viewer for the GeoParquet dataset written by
MONITRS/generate_article_captions.py: pick a location, slide through its images
over time and read the caption for each image date underneath.

Everything comes from the dataset index (--dataset): locations, image paths
(png_paths / tif_paths), captions (caption_dates / captions) and the articles
the captions were written from (article_urls).

Band checklist: images with a full-band GeoTIFF next to the PNG can also be
viewed band by band, as false-color composites, NDVI / NDWI, or the SCL cloud
mask. Check several to see them side by side. Browsers can't read GeoTIFFs, so
previews are rendered once into {dataset folder}/band_previews/ (only new or
changed GeoTIFFs are re-rendered on later runs).

The page links images by relative path, so keep it where it's written.

Usage:
    python make_dataset_viewer.py                       # Data/pipeline_test
    python make_dataset_viewer.py --dataset Data/random_dataset.geoparquet \\
        --output Data/random_viewer.html
    python make_dataset_viewer.py --no-bands            # skip rendering band previews
    open Data/pipeline_test/caption_viewer.html
"""

import os
import re
import json
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

DATASET_FILE = 'Data/pipeline_test/dataset.geoparquet'

# Band order in the GeoTIFFs (S2_BANDS in MONITRS/scrape_articles.py)
TIF_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'SCL']

# Reflectance x 10000 mapped to white. Fixed per band rather than per image, so
# the same band is comparable across dates (a per-image stretch would make a
# burn scar look the same brightness as the forest it replaced).
BAND_MAX = {'B1': 3000, 'B2': 3000, 'B3': 3000, 'B4': 3000, 'B5': 3500, 'B6': 5000,
            'B7': 5000, 'B8': 5000, 'B8A': 5000, 'B9': 3000, 'B11': 4000, 'B12': 3000}

# (key, label, description, group). 'rgb' is the existing PNG; the rest are
# rendered from the GeoTIFF.
VIEWS = [
    ('rgb', 'RGB', 'True color (B4/B3/B2), the PNG the model sees', 'Composites'),
    ('cir', 'Color infrared', 'B8/B4/B3: vegetation bright red; water, burn scars, bare ground dark', 'Composites'),
    ('swir', 'SWIR', 'B12/B8A/B4: burn scars red-brown, active fire bright orange, water black', 'Composites'),
    ('ndvi', 'NDVI', '(B8−B4)/(B8+B4) from −0.2 (brown) to 0.8 (green): vegetation and its loss', 'Indices & mask'),
    ('ndwi', 'NDWI', '(B3−B8)/(B3+B8) from −0.5 to 0.5 (blue): open water and flooding', 'Indices & mask'),
    ('scl', 'SCL', 'Scene classification: cloud, shadow, water, snow, vegetation, soil', 'Indices & mask'),
    ('B1', 'B1', 'Coastal aerosol · 443 nm · 60 m', 'Bands'),
    ('B2', 'B2', 'Blue · 490 nm · 10 m', 'Bands'),
    ('B3', 'B3', 'Green · 560 nm · 10 m', 'Bands'),
    ('B4', 'B4', 'Red · 665 nm · 10 m', 'Bands'),
    ('B5', 'B5', 'Red edge 1 · 705 nm · 20 m', 'Bands'),
    ('B6', 'B6', 'Red edge 2 · 740 nm · 20 m', 'Bands'),
    ('B7', 'B7', 'Red edge 3 · 783 nm · 20 m', 'Bands'),
    ('B8', 'B8', 'NIR · 842 nm · 10 m', 'Bands'),
    ('B8A', 'B8A', 'Narrow NIR · 865 nm · 20 m', 'Bands'),
    ('B9', 'B9', 'Water vapor · 945 nm · 60 m', 'Bands'),
    ('B11', 'B11', 'SWIR 1 · 1610 nm · 20 m', 'Bands'),
    ('B12', 'B12', 'SWIR 2 · 2190 nm · 20 m', 'Bands'),
]
RENDERED = [v[0] for v in VIEWS if v[0] != 'rgb']

# Sentinel-2 SCL classes: (name, color)
SCL_CLASSES = [
    ('No data', '#000000'), ('Saturated / defective', '#ff0000'), ('Dark / shadowed', '#2f2f2f'),
    ('Cloud shadow', '#643200'), ('Vegetation', '#00a000'), ('Bare soil', '#ffe65a'),
    ('Water', '#0000ff'), ('Unclassified', '#808080'), ('Cloud (medium)', '#c0c0c0'),
    ('Cloud (high)', '#ffffff'), ('Thin cirrus', '#64c8ff'), ('Snow / ice', '#ff96ff'),
]

# Color stops (index value, RGB) for the NDVI and NDWI renderings
NDVI_STOPS = [(-0.2, (120, 70, 30)), (0.2, (235, 225, 170)), (0.5, (90, 170, 60)), (0.8, (0, 90, 30))]
NDWI_STOPS = [(-0.5, (245, 240, 225)), (0.0, (160, 190, 210)), (0.5, (0, 50, 160))]


# --- Band previews ---

def preview_ext(key):
    # SCL is categorical, so lossless; everything else is a JPEG to keep it small
    return 'png' if key == 'scl' else 'jpg'


def preview_dir(tif_path, previews_dir):
    """.../images/2_0/2_0_2021-01-29_earthquake.tif -> {previews_dir}/2_0/2_0_2021-01-29_earthquake"""
    folder = os.path.basename(os.path.dirname(tif_path))
    name = os.path.splitext(os.path.basename(tif_path))[0]
    return os.path.join(previews_dir, folder, name)


def _ramp(x, stops):
    """Map x through color stops [(value, (r, g, b)), ...] to a uint8 RGB image."""
    vals = [s[0] for s in stops]
    return np.stack([np.interp(x, vals, [s[1][c] for s in stops]) for c in range(3)],
                    axis=-1).astype(np.uint8)


def render_views(tif_path):
    """Every view in RENDERED for one GeoTIFF, as {key: uint8 array} (no-data pixels black)."""
    import tifffile
    # Earth Engine writes GDAL_NODATA as "0.0", which tifffile logs a harmless error about
    logging.getLogger('tifffile').setLevel(logging.CRITICAL)

    a = tifffile.imread(tif_path)
    band = {b: a[..., i].astype(np.float32) for i, b in enumerate(TIF_BANDS)}
    nodata = a[..., :12].max(axis=-1) == 0

    def gray(b):
        return np.clip(band[b] / BAND_MAX[b] * 255, 0, 255).astype(np.uint8)

    def index(p, q):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num((band[p] - band[q]) / (band[p] + band[q]))

    out = {b: gray(b) for b in BAND_MAX}
    out['cir'] = np.stack([gray('B8'), gray('B4'), gray('B3')], axis=-1)
    out['swir'] = np.stack([gray('B12'), gray('B8A'), gray('B4')], axis=-1)
    out['ndvi'] = _ramp(index('B8', 'B4'), NDVI_STOPS)
    out['ndwi'] = _ramp(index('B3', 'B8'), NDWI_STOPS)
    palette = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for _, c in SCL_CLASSES], np.uint8)
    out['scl'] = palette[np.clip(a[..., TIF_BANDS.index('SCL')], 0, len(palette) - 1)]
    for key, img in out.items():
        if key != 'scl':
            img[nodata] = 0
    return out


def render_previews(tif_path, previews_dir):
    """Render every view in RENDERED from one GeoTIFF into its preview folder."""
    from PIL import Image
    out = render_views(tif_path)
    d = preview_dir(tif_path, previews_dir)
    os.makedirs(d, exist_ok=True)
    for key in RENDERED:
        path = os.path.join(d, f'{key}.{preview_ext(key)}')
        Image.fromarray(out[key]).save(path + '.part', format='PNG' if key == 'scl' else 'JPEG', quality=85)
        os.replace(path + '.part', path)


def previews_current(tif_path, previews_dir):
    last = os.path.join(preview_dir(tif_path, previews_dir), f'{RENDERED[-1]}.{preview_ext(RENDERED[-1])}')
    return os.path.isfile(last) and os.path.getmtime(last) >= os.path.getmtime(tif_path)


def render_all(tifs, previews_dir, workers):
    todo = [t for t in tifs if not previews_current(t, previews_dir)]
    if not todo:
        return
    print(f"Rendering band previews for {len(todo)} GeoTIFFs into {previews_dir}/ ...")
    from tqdm import tqdm
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(render_previews, t, previews_dir): t for t in todo}
        for fut in tqdm(futs, total=len(futs)):
            try:
                fut.result()
            except Exception as e:
                failed += 1
                print(f"  Failed {futs[fut]}: {e}")
    if failed:
        print(f"  {failed} GeoTIFFs failed; their dates show RGB only")


# --- Data ---

def as_list(x):
    return [] if x is None else list(x)


def collect(df):
    """One entry per location: metadata, articles and one frame per PNG."""
    locations = []
    df = df.sort_values('image_dir_id', key=lambda s: s.map(
        lambda i: [int(p) if p.isdigit() else p for p in re.split(r'_', i)]))
    for row in df.itertuples():
        captions = dict(zip(as_list(row.caption_dates), as_list(row.captions)))
        frames = []
        for png in as_list(row.png_paths):
            m = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(png))
            if not m:
                continue
            tif = os.path.splitext(png)[0] + '.tif'
            frames.append({'date': m.group(1), 'png': png,
                           'tif': tif if os.path.isfile(tif) else None,
                           'caption': captions.get(m.group(1), '')})
        if not frames:
            continue
        frames.sort(key=lambda f: f['date'])
        places = as_list(row.places)
        locations.append({
            'id': row.image_dir_id,
            'query_id': str(row.query_id),
            'country': row.country,
            'place': places[0] if places else row.country,
            'type': row.disaster_type,
            'start': row.start_date,
            'end': row.end_date,
            'lat': float(row.lat),
            'lon': float(row.lon),
            'articles': as_list(row.article_urls),
            'frames': frames,
        })
    return locations


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MONITRS Caption Viewer</title>
<style>
  :root {
    --bg: #f6f5f2; --panel: #ffffff; --text: #1d1d1f; --muted: #6b6b70;
    --border: #e2e0db; --accent: #2f6fdb; --pre: #6b7a8f; --during: #d2492a; --post: #b7801a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #141416; --panel: #1e1e21; --text: #ececee; --muted: #9a9aa2;
      --border: #303035; --accent: #6b9cff; --pre: #8d9bb0; --during: #ff7a5c; --post: #e0ae4c;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  main { max-width: 1320px; margin: 0 auto; padding: 20px 16px 48px; }
  h1 { font-size: 18px; margin: 0 0 14px; font-weight: 600; }
  h1 span { color: var(--muted); font-weight: 400; font-size: 14px; margin-left: 8px; }
  /* Image on the left (stays in view while the side panel scrolls), picker and caption on the right */
  .layout { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 16px; align-items: start; }
  .viewer { position: sticky; top: 16px; }
  /* Keep the square image short enough that the controls under it stay on screen */
  .viewer .frame { width: min(100%, calc(100vh - 190px)); margin: 0 auto; }
  .side { display: flex; flex-direction: column; gap: 16px; }
  .picker { display: flex; flex-direction: column; gap: 8px; }
  .picker label { font-size: 13px; color: var(--muted); }
  .picker .nav { display: flex; gap: 8px; }
  .picker .nav button { flex: 1; }
  select { width: 100%; padding: 8px 10px; font: inherit; color: var(--text);
           background: var(--panel); border: 1px solid var(--border); border-radius: 8px; }
  @media (max-width: 820px) {
    .layout { grid-template-columns: 1fr; }
    .viewer { position: static; }
  }
  button { font: inherit; padding: 7px 12px; border-radius: 8px; cursor: pointer;
           border: 1px solid var(--border); background: var(--panel); color: var(--text); }
  button:hover { border-color: var(--accent); }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .meta { color: var(--muted); font-size: 13px; margin-top: 4px; }
  .meta strong { color: var(--text); font-size: 15px; display: block; }
  .frame { position: relative; aspect-ratio: 1; background: #000; border-radius: 8px; overflow: hidden; }
  .frame img { width: 100%; height: 100%; object-fit: contain; display: block; }
  .badge { position: absolute; top: 10px; left: 10px; padding: 3px 9px; border-radius: 99px;
           font-size: 12px; font-weight: 600; color: #fff; }
  .date { position: absolute; top: 10px; right: 10px; padding: 3px 9px; border-radius: 99px;
          font-size: 13px; background: rgba(0,0,0,.6); color: #fff; font-variant-numeric: tabular-nums; }
  .controls { display: flex; gap: 8px; align-items: center; margin: 14px 0 6px; }
  input[type=range] { flex: 1; accent-color: var(--accent); }
  .count { color: var(--muted); font-size: 13px; min-width: 52px; text-align: right;
           font-variant-numeric: tabular-nums; }
  .timeline { position: relative; height: 10px; margin: 0 0 4px; }
  .tick { position: absolute; top: 0; width: 4px; height: 8px; border-radius: 2px; transform: translateX(-2px);
          cursor: pointer; }
  .tick.nocap { opacity: .35; }
  .tick.cur { outline: 2px solid var(--text); outline-offset: 1px; }
  .legend { color: var(--muted); font-size: 12px; margin: 0 0 14px; }
  .legend i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin: 0 4px 0 10px;
              vertical-align: -1px; }
  .legend i:first-child { margin-left: 0; }
  .cap { border-left: 3px solid var(--accent); padding-left: 10px; }
  .cap-label { font-size: 12px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
               color: var(--muted); }
  .cap p { margin: 2px 0 0; font-size: 16px; }
  .cap .empty { color: var(--muted); font-style: italic; }
  details { margin-top: 12px; color: var(--muted); font-size: 13px; }
  details ul, details ol { margin: 6px 0 0; padding-left: 18px; }
  details li { margin-bottom: 4px; }
  details li.cur { color: var(--text); }
  details .d { font-variant-numeric: tabular-nums; margin-right: 6px; }
  details a { color: var(--accent); word-break: break-all; }
  .hint { color: var(--muted); font-size: 12px; margin-top: 10px; }
  /* One panel per checked band, tiled inside the square frame */
  .panels { position: absolute; inset: 0; display: grid; gap: 2px; }
  .panel { position: relative; min-width: 0; min-height: 0; background: #000; }
  .panel .lbl { position: absolute; bottom: 6px; left: 6px; padding: 1px 7px; border-radius: 99px;
                font-size: 11px; font-weight: 600; background: rgba(0,0,0,.6); color: #fff; }
  .panel .none { position: absolute; inset: 0; display: grid; place-items: center; padding: 8px;
                 text-align: center; color: #8a8a90; font-size: 12px; }
  .bands-head { display: flex; justify-content: space-between; align-items: center; gap: 8px;
                flex-wrap: wrap; margin-bottom: 8px; }
  .bands-head .presets { display: flex; gap: 6px; flex-wrap: wrap; }
  .bands-head button { padding: 3px 9px; font-size: 12px; }
  .band-group { margin-top: 8px; }
  .band-group > .g { font-size: 12px; color: var(--muted); margin-bottom: 2px; }
  .band-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 2px 12px; }
  .band-list label { display: flex; gap: 6px; align-items: baseline; font-size: 13px; cursor: pointer; }
  .band-list label.off { color: var(--muted); cursor: default; }
  .band-list input { accent-color: var(--accent); }
  .band-list .d { color: var(--muted); font-size: 12px; }
  #sclLegend { display: flex; flex-wrap: wrap; gap: 2px 12px; font-size: 12px; color: var(--muted);
               margin-top: 10px; }
  #sclLegend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px;
                 vertical-align: -1px; border: 1px solid var(--border); }
  .empty-state { color: var(--muted); }
</style>
</head>
<body>
<main>
  <h1>MONITRS Caption Viewer<span id="summary"></span></h1>
  <div class="layout" id="layout">
  <div class="card viewer">
    <div class="frame">
      <div class="panels" id="panels"></div>
      <span class="badge" id="badge"></span>
      <span class="date" id="date"></span>
    </div>
    <div class="controls">
      <button id="play">Play</button>
      <input type="range" id="slider" min="0" value="0" aria-label="Image date">
      <span class="count" id="count"></span>
    </div>
    <div class="timeline" id="timeline"></div>
    <div class="legend"><i style="background:var(--pre)"></i>pre <i style="background:var(--during)"></i>during
      <i style="background:var(--post)"></i>post-event image dates &middot; faded = no caption</div>
    <div class="hint">&larr; / &rarr; step through dates &middot; &uarr; / &darr; change location &middot; space plays</div>
  </div>
  <div class="side">
    <div class="card picker">
      <label for="loc">Event / location</label>
      <select id="loc"></select>
      <div class="nav">
        <button id="prevLoc" title="Previous location">&uarr; Prev</button>
        <button id="nextLoc" title="Next location">Next &darr;</button>
      </div>
      <div class="meta" id="meta"></div>
    </div>
    <div class="card">
      <div class="cap"><div class="cap-label">Caption</div><p id="caption"></p></div>
      <details id="allBox"><summary id="allSummary"></summary><ol id="allList"></ol></details>
      <details id="articlesBox"><summary id="articlesSummary"></summary><ul id="articles"></ul></details>
    </div>
    <div class="card">
      <div class="bands-head">
        <span class="cap-label">Bands</span>
        <span class="presets">
          <button data-preset="rgb">RGB only</button>
          <button data-preset="disaster">Fire &amp; flood</button>
          <button data-preset="bands">All 12 bands</button>
          <button data-preset="none">Clear</button>
        </span>
      </div>
      <div id="bandList"></div>
      <div id="sclLegend"></div>
      <div class="meta" id="bandNote"></div>
    </div>
  </div>
  </div>
</main>
<script>
const DATA = __DATA__;
const VIEWS = __VIEWS__;
const SCL = __SCL__;
const KEY = 'datasetViewer:' + location.pathname;
const $ = id => document.getElementById(id);
let loc = null, timer = null;

function store(k, v) { try { localStorage.setItem(KEY + k, JSON.stringify(v)); } catch (e) {} }
function recall(k) { try { return JSON.parse(localStorage.getItem(KEY + k)); } catch (e) { return null; } }

// --- Band checklist ---
const PRESETS = {
  rgb: ['rgb'],
  disaster: ['rgb', 'swir', 'cir', 'ndwi'],
  bands: VIEWS.filter(v => v.group === 'Bands').map(v => v.key),
  none: [],
};
let checked = ['rgb'];
const savedBands = recall('bands');
if (Array.isArray(savedBands)) checked = savedBands.filter(k => VIEWS.some(v => v.key === k));

function buildChecklist() {
  let lastGroup = null, list = null;
  VIEWS.forEach(v => {
    if (v.group !== lastGroup) {
      const g = document.createElement('div');
      g.className = 'band-group';
      g.innerHTML = '<div class="g"></div><div class="band-list"></div>';
      g.querySelector('.g').textContent = v.group;
      $('bandList').appendChild(g);
      list = g.querySelector('.band-list');
      lastGroup = v.group;
    }
    const label = document.createElement('label');
    label.title = v.desc;
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.value = v.key;
    box.onchange = () => {
      // Keep the checklist order, not the click order
      const on = new Set([...$('bandList').querySelectorAll('input:checked')].map(b => b.value));
      setChecked(VIEWS.map(x => x.key).filter(k => on.has(k)));
    };
    const name = document.createElement('b');
    name.textContent = v.label;
    const d = document.createElement('span');
    d.className = 'd';
    d.textContent = v.key === 'rgb' || v.group === 'Bands' ? v.desc : '';
    label.append(box, name, d);
    list.appendChild(label);
  });
  SCL.forEach(([name, color]) => {
    const s = document.createElement('span');
    s.innerHTML = '<i></i>';
    s.querySelector('i').style.background = color;
    s.append(name);
    $('sclLegend').appendChild(s);
  });
  document.querySelectorAll('[data-preset]').forEach(b => b.onclick = () => setChecked(PRESETS[b.dataset.preset]));
}

function setChecked(keys) {
  checked = keys;
  store('bands', checked);
  $('bandList').querySelectorAll('input').forEach(b => b.checked = checked.includes(b.value));
  $('sclLegend').style.display = checked.includes('scl') ? '' : 'none';
  renderPanels();
  if (loc) show(+$('slider').value);
}

function viewSrc(f, key) {
  if (key === 'rgb') return f.src;
  const v = VIEWS.find(x => x.key === key);
  return f.bands ? `${f.bands}${key}.${v.ext}` : null;
}

// One panel per checked view, tiled as close to square as possible
function renderPanels() {
  const keys = checked.length ? checked : ['rgb'];
  const cols = Math.ceil(Math.sqrt(keys.length)), rows = Math.ceil(keys.length / cols);
  $('panels').style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  $('panels').style.gridTemplateRows = `repeat(${rows}, 1fr)`;
  $('panels').innerHTML = '';
  keys.forEach(key => {
    const v = VIEWS.find(x => x.key === key);
    const p = document.createElement('div');
    p.className = 'panel';
    p.dataset.key = key;
    p.title = v.desc;
    p.innerHTML = '<img alt=""><span class="none">No band data for this date</span><span class="lbl"></span>';
    p.querySelector('.lbl').textContent = v.label;
    $('panels').appendChild(p);
  });
}

function phase(date, l) {
  if (date < l.start) return ['Pre-event', 'var(--pre)'];
  if (date <= l.end) return ['During', 'var(--during)'];
  return ['Post-event', 'var(--post)'];
}

// Group the dropdown by event (query_id)
let group = null, lastQ = null;
DATA.forEach((l, i) => {
  if (l.query_id !== lastQ) {
    group = document.createElement('optgroup');
    group.label = `Event ${l.query_id} — ${l.country}`;
    $('loc').appendChild(group);
    lastQ = l.query_id;
  }
  const o = document.createElement('option');
  o.value = i;
  o.textContent = `${l.id} · ${l.type} · ${l.place}`;
  group.appendChild(o);
});

function selectLoc(i) {
  stop();
  loc = DATA[i];
  $('loc').value = i;
  const nCap = loc.frames.filter(f => f.caption).length;
  $('meta').innerHTML = '';
  const title = document.createElement('strong');
  title.textContent = `${loc.place} — ${loc.type}`;
  $('meta').append(title, `${loc.start} to ${loc.end} · (${loc.lat.toFixed(4)}, ${loc.lon.toFixed(4)}) · ` +
    `${loc.frames.length} images, ${nCap} captioned · folder ${loc.id}`);
  $('slider').max = loc.frames.length - 1;
  // Timeline: image dates colored by phase, spaced by actual date
  const ts = loc.frames.map(f => Date.parse(f.date));
  const t0 = Math.min(...ts), t1 = Math.max(...ts);
  const pos = d => (t1 > t0 ? (Date.parse(d) - t0) / (t1 - t0) * 100 : 50) + '%';
  $('timeline').innerHTML = '';
  loc.frames.forEach((f, k) => {
    const tick = document.createElement('span');
    tick.className = 'tick' + (f.caption ? '' : ' nocap');
    tick.style.left = pos(f.date);
    tick.style.background = phase(f.date, loc)[1];
    tick.title = f.date + (f.caption ? '' : ' (no caption)');
    tick.onclick = () => { stop(); show(k); };
    $('timeline').appendChild(tick);
  });
  // Every caption for this location
  $('allSummary').textContent = `All captions (${nCap})`;
  $('allList').innerHTML = '';
  loc.frames.forEach((f, k) => {
    const li = document.createElement('li');
    const d = document.createElement('span');
    d.className = 'd';
    d.textContent = f.date;
    li.append(d, f.caption || '(no caption)');
    li.style.cursor = 'pointer';
    li.onclick = () => { stop(); show(k); };
    $('allList').appendChild(li);
  });
  // Articles the captions were written from
  $('articlesSummary').textContent = `Articles used (${loc.articles.length})`;
  $('articles').innerHTML = '';
  loc.articles.forEach(url => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = url;
    li.appendChild(a);
    $('articles').appendChild(li);
  });
  $('articlesBox').style.display = loc.articles.length ? '' : 'none';
  // Which band views this location can show
  const nb = loc.frames.filter(f => f.bands).length;
  $('bandNote').textContent = nb === loc.frames.length ? ''
    : nb ? `Band data for ${nb} of ${loc.frames.length} dates in this location.`
    : 'No band data for this location (no GeoTIFFs, or the page was built with --no-bands).';
  $('bandList').querySelectorAll('label').forEach(l => {
    l.classList.toggle('off', !nb && l.querySelector('input').value !== 'rgb');
  });
  // Start on the image just before the event start
  const first = loc.frames.findIndex(f => f.date >= loc.start);
  show(first > 0 ? first - 1 : 0);
  store('loc', loc.id);
}

function show(k) {
  const f = loc.frames[k];
  $('slider').value = k;
  $('panels').querySelectorAll('.panel').forEach(p => {
    const src = viewSrc(f, p.dataset.key), img = p.querySelector('img');
    img.style.display = src ? '' : 'none';
    p.querySelector('.none').style.display = src ? 'none' : '';
    if (src) { img.src = src; img.alt = `${p.dataset.key} · ${loc.place} on ${f.date}`; }
  });
  $('date').textContent = f.date;
  const [label, color] = phase(f.date, loc);
  $('badge').textContent = label;
  $('badge').style.background = color;
  $('count').textContent = `${k + 1} / ${loc.frames.length}`;
  $('caption').textContent = f.caption || 'No caption for this date.';
  $('caption').className = f.caption ? '' : 'empty';
  [...$('timeline').children].forEach((t, j) => t.classList.toggle('cur', j === k));
  [...$('allList').children].forEach((li, j) => li.classList.toggle('cur', j === k));
  // Preload neighbors so sliding is smooth
  [k - 1, k + 1].forEach(j => {
    if (!loc.frames[j]) return;
    (checked.length ? checked : ['rgb']).forEach(key => {
      const src = viewSrc(loc.frames[j], key);
      if (src) new Image().src = src;
    });
  });
}

function step(d) {
  const k = +$('slider').value + d;
  if (k >= 0 && k < loc.frames.length) show(k);
}
function stop() { clearInterval(timer); timer = null; $('play').textContent = 'Play'; }
function togglePlay() {
  if (timer) return stop();
  if (+$('slider').value === loc.frames.length - 1) show(0);
  $('play').textContent = 'Pause';
  timer = setInterval(() => {
    if (+$('slider').value >= loc.frames.length - 1) return stop();
    step(1);
  }, 1200);
}
function changeLoc(d) {
  const i = +$('loc').value + d;
  if (i >= 0 && i < DATA.length) selectLoc(i);
}

$('loc').onchange = e => selectLoc(+e.target.value);
$('slider').oninput = e => { stop(); show(+e.target.value); };
$('play').onclick = togglePlay;
$('prevLoc').onclick = () => changeLoc(-1);
$('nextLoc').onclick = () => changeLoc(1);
document.addEventListener('keydown', e => {
  if (!loc || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') { stop(); step(-1); }
  else if (e.key === 'ArrowRight') { stop(); step(1); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); changeLoc(-1); }
  else if (e.key === 'ArrowDown') { e.preventDefault(); changeLoc(1); }
  else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
});

const nImages = DATA.reduce((n, l) => n + l.frames.length, 0);
$('summary').textContent = `${DATA.length} locations · ${nImages} images`;
buildChecklist();
setChecked(checked);
if (DATA.length) {
  const start = DATA.findIndex(l => l.id === recall('loc'));
  selectLoc(start >= 0 ? start : 0);
} else {
  $('layout').innerHTML = '<p class="empty-state">No captioned locations in this dataset.</p>';
}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=DATASET_FILE,
                        help='GeoParquet index from MONITRS/generate_article_captions.py')
    parser.add_argument('--output', type=str, default=None,
                        help='HTML file (default: caption_viewer.html next to --dataset)')
    parser.add_argument('--previews-dir', type=str, default=None,
                        help='Band preview folder (default: band_previews/ next to --dataset)')
    parser.add_argument('--no-bands', action='store_true',
                        help="Don't render band previews (only previews rendered before are shown)")
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 4,
                        help='Processes for rendering band previews')
    args = parser.parse_args()

    data_dir = os.path.dirname(args.dataset) or '.'
    output = args.output or os.path.join(data_dir, 'caption_viewer.html')
    previews_dir = args.previews_dir or os.path.join(data_dir, 'band_previews')

    locations = collect(pd.read_parquet(args.dataset))
    frames = [fr for l in locations for fr in l['frames']]
    if not args.no_bands:
        render_all([fr['tif'] for fr in frames if fr['tif']], previews_dir, args.workers)

    # Image paths relative to the page, so it works wherever it's written
    page_dir = os.path.dirname(os.path.abspath(output))
    rel = lambda p: os.path.relpath(os.path.abspath(p), page_dir).replace(os.sep, '/')
    for fr in frames:
        tif = fr.pop('tif')
        fr['src'] = rel(fr.pop('png'))
        fr['bands'] = (rel(preview_dir(tif, previews_dir)) + '/') \
            if tif and previews_current(tif, previews_dir) else None

    views = [{'key': k, 'label': lbl, 'desc': d, 'group': g, 'ext': preview_ext(k)}
             for k, lbl, d, g in VIEWS]
    data = json.dumps(locations, ensure_ascii=False).replace('</', '<\\/')
    html = (TEMPLATE.replace('__DATA__', data)
            .replace('__VIEWS__', json.dumps(views, ensure_ascii=False))
            .replace('__SCL__', json.dumps(SCL_CLASSES)))
    os.makedirs(page_dir, exist_ok=True)
    with open(output, 'w') as f:
        f.write(html)

    n_caps = sum(1 for fr in frames if fr['caption'])
    n_bands = sum(1 for fr in frames if fr['bands'])
    print(f"Wrote {output}: {len(locations)} locations, {len(frames)} images "
          f"({n_caps} captioned, {n_bands} with band data)")


if __name__ == '__main__':
    main()
