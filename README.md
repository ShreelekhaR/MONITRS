# MONITRS
Official Repo for MONITRS 

# 1. Initial Setup
## 1.1 Clone the repo
```bash
git clone https://github.com/ShreelekhaR/MONITRS.git
cd MONITRS
```
## 1.2 Create a conda environment with the required dependencies    
```bash
conda create -n monitrs python=3.9 
conda activate monitrs
pip install -r requirements.txt # coming soon
```

# 2. MONITRS Creation

## 2.1 Retrieve articles from the web on the disasters
```bash
python MONITRS/get_articles.py
```
## 2.2 Get API keys for Gemini and Geocode

Create gemini api key and set it in the environment variable. https://ai.google.dev/gemini-api/docs/api-key
Create geocode api key and set it in the environment variable.
https://geocode.maps.co/
Both keys to be set in the get_article_aggregate_locations.py file.


## 2.3 Run script to obtain article content and locations data
```bash
python MONITRS/get_article_aggregate_locations.py
```
## 2.4 Filter cloudy/corrupted images
```bash
python MONITRS/filter_invalid_images.py
```

## 2.5 Consolidate the captions for the images
```bash
python MONITRS/consolidate_captions.py
```
# 3. MONITRS-QA Creation

## 3.1 Create the templated multiple choice questions
```bash
python MONITRS_QA/templated_mcq.py
```

## 3.2 Create the generated multiple choice questions
```bash
python MONITRS_QA/generated_mcq.py
```
## 3.3 Create the generated open ended questions
```bash
python MONITRS_QA/generated_q_a.py
```
## 3.4 Merge the (available) training and test sets

Merges all train and test sets into a single train and test set for MONITRS-QA. 

Please note that to create subsets of the dataset, for example MONITRS-QA-tiny, there are lines that can be uncommented in the merge_train_test.py file.
```bash
python MONITRS_QA/merge_train_test.py
```