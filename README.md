# DisasterSet
Official Repo for DisasterSet 

# 1. Initial Setup
## 1.1 Clone the repo
```bash
git clone https://github.com/ShreelekhaR/DisasterSet.git
cd DisasterSet
```
## 1.2 Create a conda environment with the required dependencies    
```bash
conda create -n disaster python=3.9 
conda activate disaster
pip install -r requirements.txt # coming soon
```

# 2. DisasterSet Creation

## 2.1 Retrieve articles from the web on the disasters
```bash
python DisasterSet/get_articles.py
```
## 2.2 Get API keys for Gemini and Geocode

Create gemini api key and set it in the environment variable. https://ai.google.dev/gemini-api/docs/api-key
Create geocode api key and set it in the environment variable.
https://geocode.maps.co/
Both keys to be set in the get_article_aggregate_locations.py file.


## 2.3 Run script to obtain article content and locations data
```bash
python DisasterSet/get_article_aggregate_locations.py
```
## 2.4 Filter cloudy/corrupted images
```bash
python DisasterSet/filter_invalid_images.py
```

## 2.5 Consolidate the captions for the images
```bash
python DisasterSet/consolidate_captions.py
```
# 3. DisasterSet-QA Creation

## 3.1 Create the templated multiple choice questions
```bash
python DS_QA/templated_mcq.py
```

## 3.2 Create the templated open ended questions
```bash
python DS_QA/templated_q_a.py
```

## 3.3 Create the generated multiple choice questions
```bash
python DS_QA/generated_mcq.py
```
## 3.4 Create the generated open ended questions
```bash
python DS_QA/generated_q_a.py
```
## 3.5 Merge the (available) training and test sets

Merges all train and test sets into a single train and test set for DisasterSet-QA. 

Please note that to create subsets of the dataset, for example DisasterSet-QA-tiny, there are lines that can be uncommented in the merge_train_test.py file.
```bash
python DS_QA/merge_train_test.py
```