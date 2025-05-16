# read multiple jsons, and re-do the id 

import json
import os
from tqdm import tqdm

tiny = False
# for tiny uncomment the following line
# tiny = True

train_jsons = [ "train_templated_q_a.json", "train_multiple_choice.json", "train_generated_q_a.json", "train_generated_multiple_choice_q_a.json"]
test_jsons = ["test_templated_q_a.json", "test_multiple_choice.json", "test_generated_q_a.json", "test_generated_multiple_choice_q_a.json"]

# merge available jsons for train and test respectively

def merge_jsons(jsons):
    """
    Merge multiple JSON files into one.
    
    Args:
        jsons (list): List of JSON file paths to merge.
    
    Returns:
        list: Merged data from all JSON files.
    """
    merged_data = []
    
    for json_file in jsons:
        with open(json_file, "r") as f:
            data = json.load(f)
            merged_data.extend(data)
    
    
    # re-do the id
    for i,item in enumerate(merged_data):
        item["id"] = i
    return merged_data

# merge train and test jsons
train_data = merge_jsons(train_jsons)
test_data = merge_jsons(test_jsons)

if not tiny:
    # save the combined jsons
    with open("train_total.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open("test_total.json", "w") as f:
        json.dump(test_data, f, indent=2)
else:
    # choosing every 10th item
    train_data = train_data[::10]
    test_data = test_data[::10]
    # save the combined jsons
    with open("tiny_train_total.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open("tiny_test_total.json", "w") as f:
        json.dump(test_data, f, indent=2)