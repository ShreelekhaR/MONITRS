"""
Builds a stats viewer for a pipeline data folder: bar charts of how many
events and articles there are per country, per disaster type, and per article
language, plus a world map shaded by count per country. Click a bar or a
country to filter the other charts to it (click again to clear).

Reads from the data folder (--data):
    queries.csv   one row per event: query_id, english_country, disaster_type,
                  ddgs_region (written by MONITRS/generate_queries.py)
    articles.csv  one row per article: query_id, lang, url
                  (written by MONITRS/search_generated_queries.py)

The `lang` column in articles.csv is the search that found the article, not
the language itself: `local` / `local_web` searched the local-language query in
the event's ddgs_region, `eng` / `eng_web` searched the English query in us-en.
The article language is derived from that: local searches take the language of
the region (pr-es -> Spanish, cm-fr -> French), English searches are English.

Country borders are Natural Earth 1:50m (as in Data/visualize_article_countries.py),
downloaded once and cached at --world. Country names are matched against all of
Natural Earth's name fields; unmatched names are printed and left off the map.
The map loads Leaflet from a CDN, so viewing it needs an internet connection.

Usage:
    python make_stats_viewer.py                                  # Data/pipeline_test
    python make_stats_viewer.py --data Data/some_run --output Data/some_run/stats_viewer.html
    open Data/pipeline_test/stats_viewer.html
"""

import os
import json
import argparse
import urllib.request

import pandas as pd

DATA_DIR = 'Data/pipeline_test'
WORLD_CACHE = 'Data/ne_50m_admin_0_countries.geojson'
# 50m rather than 110m, which leaves out small countries such as Singapore
WORLD_URL = ('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
             'master/geojson/ne_50m_admin_0_countries.geojson')
NAME_FIELDS = ['ADMIN', 'NAME', 'NAME_LONG', 'NAME_EN', 'FORMAL_EN', 'GEOUNIT']

# Language part of a DuckDuckGo region code ("pr-es" -> "es").
LANGUAGES = {
    'ar': 'Arabic', 'bg': 'Bulgarian', 'ca': 'Catalan', 'cs': 'Czech', 'da': 'Danish',
    'de': 'German', 'el': 'Greek', 'en': 'English', 'es': 'Spanish', 'et': 'Estonian',
    'fi': 'Finnish', 'fr': 'French', 'he': 'Hebrew', 'hr': 'Croatian', 'hu': 'Hungarian',
    'id': 'Indonesian', 'it': 'Italian', 'jp': 'Japanese', 'ja': 'Japanese', 'kr': 'Korean',
    'ko': 'Korean', 'lt': 'Lithuanian', 'lv': 'Latvian', 'ms': 'Malay', 'nl': 'Dutch',
    'no': 'Norwegian', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian',
    'sk': 'Slovak', 'sl': 'Slovenian', 'sv': 'Swedish', 'th': 'Thai', 'tl': 'Tagalog',
    'tr': 'Turkish', 'tzh': 'Chinese (Traditional)', 'uk': 'Ukrainian', 'vi': 'Vietnamese',
    'zh': 'Chinese', 'hi': 'Hindi', 'bn': 'Bengali', 'ur': 'Urdu', 'fa': 'Persian',
    'sw': 'Swahili',
}

SEARCH_LABELS = {
    'local': 'Local news', 'local_web': 'Local web',
    'eng': 'English news', 'eng_web': 'English web',
}


def region_language(region):
    if not isinstance(region, str) or '-' not in region:
        return 'Unknown'
    code = region.split('-', 1)[1].lower()
    if code == 'wt':  # "wt-wt" = no region
        return 'Unknown'
    return LANGUAGES.get(code, code)


def article_language(search, region):
    if isinstance(search, str) and search.startswith('eng'):
        return 'English'
    return region_language(region)


def load_world(cache):
    """Natural Earth countries, trimmed for embedding: code + name, coordinates
    rounded to 0.01 degrees (~1 km, plenty at world zoom)."""
    if not os.path.exists(cache):
        print(f'downloading country borders -> {cache}')
        with urllib.request.urlopen(WORLD_URL) as r:
            data = r.read()
        with open(cache, 'wb') as f:
            f.write(data)
    world = json.load(open(cache))

    def ring(pts):
        out = []
        for x, y in pts:
            pt = [round(x, 2), round(y, 2)]
            if not out or out[-1] != pt:
                out.append(pt)
        return out

    features, name_to_code = [], {}
    for f in world['features']:
        props, geom = f['properties'], f['geometry']
        code = props['ADM0_A3']
        for field in NAME_FIELDS:
            if props.get(field):
                name_to_code.setdefault(str(props[field]).lower(), code)
        polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
        polys = [[r for r in (ring(rr) for rr in poly) if len(r) >= 4] for poly in polys]
        polys = [p for p in polys if p]
        features.append({'type': 'Feature', 'properties': {'code': code, 'name': props['ADMIN']},
                         'geometry': {'type': 'MultiPolygon', 'coordinates': polys}})
    return {'type': 'FeatureCollection', 'features': features}, name_to_code


def load(data_dir):
    queries = pd.read_csv(os.path.join(data_dir, 'queries.csv'), dtype={'query_id': str})
    articles_path = os.path.join(data_dir, 'articles.csv')
    articles = (pd.read_csv(articles_path, dtype={'query_id': str})
                if os.path.exists(articles_path)
                else pd.DataFrame(columns=['query_id', 'lang', 'url']))

    country_col = 'english_country' if 'english_country' in queries else 'country'
    events = []
    by_id = {}
    for _, q in queries.iterrows():
        e = {
            'id': q['query_id'],
            'country': str(q[country_col]),
            'type': str(q['disaster_type']).strip().lower(),
            'region': q.get('ddgs_region'),
            'place': str(q.get('english_location', q.get('location', ''))),
            'year': str(q.get('year', '')),
        }
        events.append(e)
        by_id[e['id']] = e

    rows = []
    unmatched = 0
    for _, a in articles.iterrows():
        e = by_id.get(a['query_id'])
        if e is None:
            unmatched += 1
            continue
        rows.append({
            'event': e['id'],
            'language': article_language(a['lang'], e['region']),
            'search': SEARCH_LABELS.get(a['lang'], str(a['lang'])),
        })
    if unmatched:
        print(f'warning: {unmatched} articles have a query_id not in queries.csv, skipped')

    for e in events:
        e.pop('region')
    return events, rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=DATA_DIR, help='folder holding queries.csv and articles.csv')
    p.add_argument('--output', help='HTML to write (default: {data}/stats_viewer.html)')
    p.add_argument('--world', default=WORLD_CACHE, help='cached Natural Earth countries GeoJSON')
    args = p.parse_args()

    events, articles = load(args.data)
    world, name_to_code = load_world(args.world)
    unmatched = set()
    for e in events:
        e['code'] = name_to_code.get(e['country'].lower())
        if not e['code']:
            unmatched.add(e['country'])
    for name in sorted(unmatched):
        print(f"warning: no map match for country '{name}', left off the map")
    output = args.output or os.path.join(args.data, 'stats_viewer.html')
    payload = json.dumps({'source': os.path.normpath(args.data), 'events': events, 'articles': articles,
                          'world': world}, separators=(',', ':'))
    with open(output, 'w') as f:
        f.write(TEMPLATE.replace('/*DATA*/null', payload.replace('</', '<\\/')))
    print(f'{len(events)} events, {len(articles)} articles -> {output}')


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Stats</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root {
  color-scheme: light;
  --surface-0: #f5f5f3;
  --surface-1: #fcfcfb;
  --border: #e4e3df;
  --grid: #ecebe7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #7a7974;
  --series-1: #2a78d6;
  --series-dim: #c9dcf3;
  --focus: #2a78d6;
  --seq-1: #86b6ef; --seq-2: #5598e7; --seq-3: #2a78d6; --seq-4: #1c5cab; --seq-5: #0d366b;
  --no-data: #e9e8e4; --map-line: #fcfcfb; --map-sel: #0b0b0b; --map-bg: #f5f5f3;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #121211;
    --surface-1: #1a1a19;
    --border: #2e2e2c;
    --grid: #262624;
    --text-primary: #f0efec;
    --text-secondary: #c3c2b7;
    --text-muted: #8f8e86;
    --series-1: #3987e5;
    --series-dim: #24364d;
    --focus: #3987e5;
  --seq-1: #184f95; --seq-2: #256abf; --seq-3: #3987e5; --seq-4: #86b6ef; --seq-5: #cde2fb;
  --no-data: #2e2e2c; --map-line: #1a1a19; --map-sel: #f0efec; --map-bg: #121211;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211;
  --surface-1: #1a1a19;
  --border: #2e2e2c;
  --grid: #262624;
  --text-primary: #f0efec;
  --text-secondary: #c3c2b7;
  --text-muted: #8f8e86;
  --series-1: #3987e5;
  --series-dim: #24364d;
  --focus: #3987e5;
  --seq-1: #184f95; --seq-2: #256abf; --seq-3: #3987e5; --seq-4: #86b6ef; --seq-5: #cde2fb;
  --no-data: #2e2e2c; --map-line: #1a1a19; --map-sel: #f0efec; --map-bg: #121211;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1200px; margin: 0 auto; padding: 24px 16px 48px; }
header h1 { font-size: 20px; margin: 0 0 2px; }
header p { margin: 0; color: var(--text-secondary); }
.toolbar {
  display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  margin: 20px 0 16px;
}
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button {
  border: 0; background: var(--surface-1); color: var(--text-secondary);
  padding: 6px 12px; font: inherit; cursor: pointer;
}
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--text-primary); color: var(--surface-1); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
  display: inline-flex; gap: 6px; align-items: center;
  border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary);
  border-radius: 999px; padding: 3px 10px; font: inherit; font-size: 13px; cursor: pointer;
}
.chip span { color: var(--text-muted); }
.spacer { flex: 1; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .k { color: var(--text-secondary); font-size: 12px; }
.tile .v { font-size: 24px; font-weight: 600; font-variant-numeric: tabular-nums; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px 10px; min-width: 0; }
.card h2 { font-size: 15px; margin: 0; }
.card .sub { color: var(--text-muted); font-size: 12px; margin: 2px 0 10px; }
.bars { display: grid; grid-template-columns: minmax(70px, max-content) 1fr; column-gap: 10px; row-gap: 2px; }
.row { display: contents; cursor: pointer; }
.row .lab {
  color: var(--text-secondary); font-size: 13px; padding: 4px 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 180px;
}
.row .track { position: relative; display: flex; align-items: center; min-height: 28px; border-left: 1px solid var(--text-muted); }
.row .bar { height: 18px; background: var(--series-1); border-radius: 0 4px 4px 0; min-width: 2px; transition: width .2s; }
.row .val { margin-left: 6px; font-size: 12px; color: var(--text-primary); font-variant-numeric: tabular-nums; }
.row:hover .lab, .row.sel .lab { color: var(--text-primary); }
.row.sel .lab { font-weight: 600; }
.bars.has-sel .row:not(.sel) .bar { background: var(--series-dim); }
.row:focus-visible .lab { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: 3px; }
.more { margin-top: 6px; background: none; border: 0; color: var(--text-secondary); font: inherit; font-size: 12px; cursor: pointer; padding: 4px 0; text-decoration: underline; }
.empty { color: var(--text-muted); padding: 12px 0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--grid); }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
th { color: var(--text-secondary); font-weight: 500; }
.map-card { margin-bottom: 16px; }
.map-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 16px; align-items: flex-start; }
#map { height: 440px; border-radius: 8px; background: var(--map-bg); margin-top: 4px; }
#map:focus-visible { outline: 2px solid var(--focus); }
.leaflet-container { font: inherit; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12px; color: var(--text-secondary); align-items: center; }
.legend i { display: inline-block; width: 14px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.unmapped { color: var(--text-muted); font-size: 12px; margin-top: 6px; }
@media (max-width: 600px) { #map { height: 300px; } }
#tip {
  position: fixed; pointer-events: none; z-index: 2000; display: none;
  background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.15);
  max-width: 260px;
}
#tip b { display: block; font-size: 13px; margin-bottom: 2px; }
#tip .m { color: var(--text-secondary); }
</style>
</head>
<body>
<main>
  <header>
    <h1>Dataset stats</h1>
    <p id="source"></p>
  </header>

  <div class="toolbar">
    <div class="seg" role="group" aria-label="Count">
      <button data-measure="articles" aria-pressed="true">Count articles</button>
      <button data-measure="events" aria-pressed="false">Count events</button>
    </div>
    <div class="chips" id="chips"></div>
    <div class="spacer"></div>
    <div class="seg" role="group" aria-label="View">
      <button data-view="chart" aria-pressed="true">Chart</button>
      <button data-view="table" aria-pressed="false">Table</button>
    </div>
  </div>

  <div class="tiles" id="tiles"></div>
  <section class="card map-card">
    <div class="map-head">
      <div><h2 id="map-title">Articles per country</h2><div class="sub" id="map-sub"></div></div>
      <div class="legend" id="legend"></div>
    </div>
    <div id="map" role="img" aria-label="World map shaded by count per country"></div>
    <div class="unmapped" id="unmapped"></div>
  </section>
  <div class="grid" id="grid"></div>
</main>
<div id="tip" role="tooltip"></div>

<script>
const DATA = /*DATA*/null;
const DIMS = [
  {key: 'country',  title: 'Country',       noun: 'countries'},
  {key: 'type',     title: 'Disaster type', noun: 'disaster types'},
  {key: 'language', title: 'Article language', noun: 'languages'},
];
const TOP_N = 15;
const state = {measure: 'articles', view: 'chart', filter: {}, expanded: {}};

const eventsById = Object.fromEntries(DATA.events.map(e => [e.id, e]));
// One record per article, carrying its event's fields.
const articleRecs = DATA.articles.map(a => ({...eventsById[a.event], language: a.language, search: a.search, event: a.event}));
// Events: language = every language among its articles (an event can have several).
const eventLangs = {};
for (const a of DATA.articles) (eventLangs[a.event] ||= new Set()).add(a.language);
const eventRecs = DATA.events.map(e => ({...e, event: e.id, languages: [...(eventLangs[e.id] || ['No articles'])]}));

document.getElementById('source').textContent =
  `${DATA.source} · ${DATA.events.length.toLocaleString()} events · ${DATA.articles.length.toLocaleString()} articles`;

const valuesOf = (r, dim) => dim === 'language' && r.languages ? r.languages : [r[dim]];
function passes(r, skipDim) {
  for (const [dim, v] of Object.entries(state.filter)) {
    if (dim === skipDim) continue;
    if (dim === 'language' && !r.languages) { if (r.language !== v) return false; }
    else if (!valuesOf(r, dim).includes(v)) return false;
  }
  return true;
}
// Article-language filter applied to events: keep events with >=1 article in that language.
function recs() { return state.measure === 'events' ? eventRecs : articleRecs; }

function counts(dim) {
  const m = new Map();
  for (const r of recs()) {
    if (!passes(r, dim)) continue;
    for (const v of valuesOf(r, dim)) m.set(v, (m.get(v) || 0) + 1);
  }
  return [...m.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(b[0]));
}

// Breakdown for the tooltip: the other count, and search source for languages.
function detail(dim, value) {
  const evs = eventRecs.filter(r => passes(r, dim) && valuesOf(r, dim).includes(value));
  const arts = articleRecs.filter(r => passes(r, dim) && r[dim] === value);
  const lines = [`${evs.length.toLocaleString()} event${evs.length === 1 ? '' : 's'} · ${arts.length.toLocaleString()} article${arts.length === 1 ? '' : 's'}`];
  if (dim === 'language' && arts.length) {
    const s = {};
    for (const a of arts) s[a.search] = (s[a.search] || 0) + 1;
    lines.push('Found by: ' + Object.entries(s).map(([k, n]) => `${k} ${n.toLocaleString()}`).join(', '));
  }
  if (dim !== 'country' && evs.length && evs.length <= 4) {
    lines.push(evs.map(e => `${e.place}, ${e.country} ${e.year}`).join('<br>'));
  }
  return lines;
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const tip = document.getElementById('tip');
function showTip(ev, dim, value, n) {
  const d = detail(dim, value);
  tip.innerHTML = `<b>${esc(value)}</b>${n.toLocaleString()} ${state.measure}<div class="m">${d.join('<br>')}</div>`;
  tip.style.display = 'block';
  moveTip(ev);
}
function moveTip(ev) {
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
const hideTip = () => { tip.style.display = 'none'; };

function toggleFilter(dim, value) {
  if (state.filter[dim] === value) delete state.filter[dim];
  else state.filter[dim] = value;
  render();
}

function renderTiles() {
  const evs = eventRecs.filter(r => passes(r));
  const arts = articleRecs.filter(r => passes(r));
  const distinct = (rs, k) => new Set(rs.flatMap(r => valuesOf(r, k))).size;
  const tiles = [
    ['Events', evs.length], ['Articles', arts.length],
    ['Countries', distinct(evs, 'country')], ['Disaster types', distinct(evs, 'type')],
    ['Languages', new Set(arts.map(a => a.language)).size],
  ];
  document.getElementById('tiles').innerHTML = tiles.map(([k, v]) =>
    `<div class="tile"><div class="k">${k}</div><div class="v">${v.toLocaleString()}</div></div>`).join('');
}

function renderChips() {
  const el = document.getElementById('chips');
  const entries = Object.entries(state.filter);
  el.innerHTML = entries.map(([dim, v]) =>
    `<button class="chip" data-dim="${dim}" title="Remove filter"><span>${DIMS.find(d => d.key === dim).title}:</span>${esc(v)} ✕</button>`).join('') +
    (entries.length > 1 ? '<button class="chip" data-clear>Clear all</button>' : '');
  el.querySelectorAll('[data-dim]').forEach(b => b.onclick = () => { delete state.filter[b.dataset.dim]; render(); });
  el.querySelector('[data-clear]')?.addEventListener('click', () => { state.filter = {}; render(); });
}

function renderCard(dim) {
  const c = counts(dim.key);
  const card = document.createElement('section');
  card.className = 'card';
  const subs = [`${c.length} ${dim.noun}`];
  if (dim.key === 'language' && state.measure === 'events') subs.push('an event counts once per language among its articles');
  card.innerHTML = `<h2>${dim.title}</h2><div class="sub">${subs.join(' · ')}</div>`;
  if (!c.length) { card.insertAdjacentHTML('beforeend', '<div class="empty">Nothing matches the filters.</div>'); return card; }

  if (state.view === 'table') {
    const total = c.reduce((s, [, n]) => s + n, 0);
    card.insertAdjacentHTML('beforeend', `<table><thead><tr><th>${dim.title}</th><th class="n">${state.measure}</th><th class="n">share</th></tr></thead><tbody>` +
      c.map(([v, n]) => `<tr><td>${esc(v)}</td><td class="n">${n.toLocaleString()}</td><td class="n">${(100 * n / total).toFixed(1)}%</td></tr>`).join('') +
      '</tbody></table>');
    return card;
  }

  const shown = state.expanded[dim.key] ? c : c.slice(0, TOP_N);
  const max = c[0][1];
  const bars = document.createElement('div');
  bars.className = 'bars' + (state.filter[dim.key] !== undefined ? ' has-sel' : '');
  for (const [v, n] of shown) {
    const row = document.createElement('div');
    row.className = 'row' + (state.filter[dim.key] === v ? ' sel' : '');
    row.tabIndex = 0;
    row.setAttribute('role', 'button');
    row.setAttribute('aria-label', `${v}: ${n.toLocaleString()} ${state.measure}. Click to filter.`);
    row.innerHTML = `<div class="lab" title="${esc(v)}">${esc(v)}</div>` +
      `<div class="track"><div class="bar" style="width:calc((100% - 64px) * ${n / max})"></div><span class="val">${n.toLocaleString()}</span></div>`;
    row.onclick = () => toggleFilter(dim.key, v);
    row.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleFilter(dim.key, v); } };
    row.onmouseenter = e => showTip(e, dim.key, v, n);
    row.onmousemove = moveTip;
    row.onmouseleave = hideTip;
    bars.appendChild(row);
  }
  card.appendChild(bars);
  if (c.length > TOP_N) {
    const more = document.createElement('button');
    more.className = 'more';
    more.textContent = state.expanded[dim.key] ? 'Show top ' + TOP_N : `Show all ${c.length}`;
    more.onclick = () => { state.expanded[dim.key] = !state.expanded[dim.key]; render(); };
    card.appendChild(more);
  }
  return card;
}

// ---- Map -------------------------------------------------------------
const codeToCountry = {};
for (const e of DATA.events) if (e.code) codeToCountry[e.code] = e.country;
let map = null, layer = null, dots = null;
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Up to 5 bins over 1..max. Counts per country are heavily skewed (a few
// countries with thousands, many with a handful), so bin edges are spaced
// logarithmically and snapped to round numbers: 1–9, 10–49, 50–199, ...
// With small maxima this falls back to one bin per value.
const NICE = [1, 2, 3, 5];
for (let p = 10; p <= 1e7; p *= 10) NICE.push(p, 2 * p, 5 * p);
function bins(max) {
  if (max <= 5) return Array.from({length: max}, (_, i) => [i + 1, i + 1]);
  const starts = [1];
  for (let i = 1; i < 5; i++) {
    const target = Math.pow(max, i / 5);
    const snap = NICE.reduce((a, b) => Math.abs(Math.log(b / target)) < Math.abs(Math.log(a / target)) ? b : a);
    if (snap > starts[starts.length - 1] && snap <= max) starts.push(snap);
  }
  return starts.map((lo, i) => [lo, i + 1 < starts.length ? starts[i + 1] - 1 : Infinity]);
}
const fmt = n => n.toLocaleString();
const binLabel = ([lo, hi]) => hi === Infinity ? `${fmt(lo)}+` : lo === hi ? fmt(lo) : `${fmt(lo)}–${fmt(hi)}`;
// Spread the bins across the 5 ramp steps, always ending on the darkest.
function binColor(i, n) { return css(`--seq-${n === 1 ? 5 : 1 + Math.round(4 * i / (n - 1))}`); }

function initMap() {
  if (!window.L) {
    document.getElementById('map').innerHTML = '<div class="empty" style="padding:16px">Map library failed to load (needs an internet connection).</div>';
    return;
  }
  map = L.map('map', {worldCopyJump: true, minZoom: 1, maxZoom: 7, zoomSnap: 0.5, attributionControl: false})
    .setView([20, 10], 1.5);
  layer = L.geoJSON(DATA.world, {
    onEachFeature: (f, l) => {
      l.on('mouseover', e => {
        const country = codeToCountry[f.properties.code];
        const n = mapCounts.get(f.properties.code) || 0;
        if (country) showTip(e.originalEvent, 'country', country, n);
        else { tip.innerHTML = `<b>${esc(f.properties.name)}</b><div class="m">Not in the dataset</div>`; tip.style.display = 'block'; moveTip(e.originalEvent); }
        if (n) l.setStyle({weight: 2, color: css('--map-sel')});
      });
      l.on('mousemove', e => moveTip(e.originalEvent));
      l.on('mouseout', () => { hideTip(); layer.resetStyle(l); });
      l.on('click', () => { const c = codeToCountry[f.properties.code]; if (c) toggleFilter('country', c); });
    },
  }).addTo(map);
  // Countries too small to see at world zoom (Puerto Rico, Singapore) also get a dot.
  dots = L.layerGroup().addTo(map);
  layer.eachLayer(l => {
    const b = l.getBounds(), code = l.feature.properties.code;
    if (!codeToCountry[code]) return;
    if (b.getNorth() - b.getSouth() < 4 && b.getEast() - b.getWest() < 4) {
      const dot = L.circleMarker(b.getCenter(), {radius: 5, weight: 1.5});
      dot.code = code;
      for (const ev of ['mouseover', 'mousemove', 'click']) dot.on(ev, e => l.fire(ev, e));
      dot.on('mouseout', () => { hideTip(); });
      dots.addLayer(dot);
    }
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateMap);
}

let mapCounts = new Map();
function updateMap() {
  const byName = new Map(counts('country'));
  mapCounts = new Map();
  for (const [code, country] of Object.entries(codeToCountry)) {
    if (byName.has(country)) mapCounts.set(code, byName.get(country));
  }
  const unmapped = [...byName.keys()].filter(c => !DATA.events.some(e => e.country === c && e.code));
  const max = Math.max(0, ...mapCounts.values());
  const bs = max ? bins(max) : [];
  const colorFor = n => {
    const i = bs.findIndex(([lo, hi]) => n >= lo && n <= hi);
    return i < 0 ? css('--no-data') : binColor(i, bs.length);
  };
  const noun = state.measure === 'events' ? 'Events' : 'Articles';
  document.getElementById('map-title').textContent = `${noun} per country`;
  document.getElementById('map-sub').textContent =
    `${mapCounts.size} countr${mapCounts.size === 1 ? 'y' : 'ies'} · click a country to filter`;
  document.getElementById('legend').innerHTML = bs.map(([lo, hi], i) =>
    `<span><i style="background:${binColor(i, bs.length)}"></i>${binLabel([lo, hi])}</span>`).join('') +
    `<span><i style="background:${css('--no-data')}"></i>0 / none</span>`;
  document.getElementById('unmapped').textContent = unmapped.length ? `Not on map (no border match): ${unmapped.join(', ')}` : '';
  if (!layer) return;
  const sel = state.filter.country;
  const line = css('--map-line'), selLine = css('--map-sel');
  // resetStyle (on mouseout) restores from options.style, so keep it current.
  layer.options.style = f => {
    const n = mapCounts.get(f.properties.code) || 0;
    const isSel = sel && codeToCountry[f.properties.code] === sel;
    return {fillColor: colorFor(n), fillOpacity: sel && !isSel && n ? 0.45 : 1,
            color: isSel ? selLine : line, weight: isSel ? 2 : 0.6};
  };
  layer.setStyle(layer.options.style);
  dots.eachLayer(d => {
    const n = mapCounts.get(d.code) || 0;
    d.setStyle({fillColor: colorFor(n), fillOpacity: n ? 1 : 0, opacity: n ? 1 : 0, color: selLine});
  });
}

function render() {
  hideTip();
  updateMap();
  document.querySelectorAll('[data-measure]').forEach(b => b.setAttribute('aria-pressed', b.dataset.measure === state.measure));
  document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', b.dataset.view === state.view));
  renderChips();
  renderTiles();
  const grid = document.getElementById('grid');
  grid.replaceChildren(...DIMS.map(renderCard));
}

document.querySelectorAll('[data-measure]').forEach(b => b.onclick = () => { state.measure = b.dataset.measure; render(); });
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => { state.view = b.dataset.view; render(); });
initMap();
render();
</script>
</body>
</html>
'''

if __name__ == '__main__':
    main()
