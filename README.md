# MONITRS v2
Official Repo for MONITRS — Multi-temporal Observation of Natural disasters through Intelligent Text-image Retrieval from Sentinel-2

## 1. Initial Setup

### 1.1 Clone the repo
```bash
git clone https://github.com/ShreelekhaR/MONITRS.git
cd MONITRS
git checkout v2
```

### 1.2 Install dependencies
```bash
pip install -r requirements.txt
```

### 1.3 Set environment variables
```bash
export GCP_PROJECT_ID=your-vertex-ai-project
export EE_PROJECT_ID=your-earth-engine-project    # optional, defaults to GCP_PROJECT_ID
export FIRMS_MAP_KEY=your-nasa-firms-key           # get free at https://firms.modaps.eosdis.nasa.gov/api/map_key/
```

### 1.4 Authenticate
```bash
gcloud auth application-default login
earthengine authenticate
```

## 2. MONITRS v2 Creation

The pipeline has two stages that can run on different machines:

### Stage 1: Language Pipeline (text processing — no GPU needed)

Scrapes articles, extracts disaster locations and events, determines image centers
(FIRMS for fires, LLM for all other events), and generates captions.

```bash
# Single terminal:
python MONITRS/run_language_pipeline.py --no-images

# Or split across 2 terminals for speed:
# Terminal 1:
python MONITRS/run_language_pipeline.py --no-images --start 0 --end 5000
# Terminal 2:
python MONITRS/run_language_pipeline.py --no-images --start 5000

# Merge results:
python -c "
import json
a = json.load(open('Data/events_processed_0_5000.json'))
b = json.load(open('Data/events_processed_5000_end.json'))
a.update(b)
json.dump(a, open('Data/events_processed.json','w'), indent=2)
print(f'Merged: {len(a)} events')
"
```

**Geolocation strategy (validated on 30 events):**
- Fire events → NASA FIRMS hotspot center (ground truth from thermal detections)
- All other events → LLM coordinate estimation from article text
- Fallback → FEMA county centroid

**Features:**
- DuckDuckGo fallback search when article scraping fails
- Visual + contextual event extraction for QA
- Factual captions aligned to actual image dates
- Resume on crash (Ctrl+C saves progress)
- Separate output files per terminal (no write conflicts)

### Stage 2: Image Download (requires Earth Engine)

Downloads Sentinel-2 imagery for each processed event.

```bash
python MONITRS/download_images.py

# Or in batches:
python MONITRS/download_images.py --start 0 --end 5000
```

**Features:**
- Pre-event (14 days before), during-event, and post-event (14 days after) images
- Cloud filtering (<30% cloud cover metadata)
- Quality filtering (rejects black/white/nodata images)
- Same-day tile mosaicing (no chopped edges)
- PNG output at 512x512

## 3. MONITRS-QA Creation

### 3.1 Create templated multiple choice questions
```bash
python MONITRS_QA/templated_mcq.py
```

### 3.2 Create generated multiple choice questions
```bash
python MONITRS_QA/generated_mcq.py
```

### 3.3 Create generated open-ended questions
```bash
python MONITRS_QA/generated_q_a.py
```

### 3.4 Merge train and test sets
```bash
python MONITRS_QA/merge_train_test.py
```

## 4. Evaluation

```bash
# MCQ accuracy + McNemar's test
python Evaluate/eval.py

# LLM-as-Judge evaluation
python Evaluate/LLM_eval.py --qa_json qa.json --answers_path answers.json
```

## 5. Testing & Verification

```bash
# Test pipeline on sample events
python test_pipeline.py

# Verify caption-image alignment (uses Gemini Pro vision)
python verify_captions.py

# Visualize geocoded points on images
python visualize_points.py
```

## 6. Citation
If you use MONITRS or MONITRS-QA in your research, please cite the following paper:

