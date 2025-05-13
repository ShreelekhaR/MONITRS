# given a set of captions, generate questions and answers

import json
from typing import List, Dict, Tuple
import ast
from datetime import datetime
import os
import re
import google.generativeai as genai
from time import sleep
from tqdm import tqdm


genai.configure(api_key="Your_api_key_here")
model = genai.GenerativeModel("gemini-1.5-flash")


def geo_to_pixel(locations, center, radius = 5):
    """Convert geographical coordinates to pixel coordinates."""
    height = 512
    width = 512
    center_lat, center_lon = center

    pixel_locations = {}

    for loc in locations:
        lat, lon = locations[loc]

        # convert lat, lon to pixel coordinates
        x_offset = int((lon - center_lon) * (width / 360.0) + width / 2)
        y_offset = int((lat - center_lat) * (height / 180.0) + height / 2)

        pixel_locations[loc] = (x_offset, y_offset)
    return pixel_locations



def parse_line(line: str) -> Dict:
    """Parse a single line of the data file."""
    # Split line into main components
    parts = []
    
    parts = line.split(',')
    # print(parts)    
    # Extract components
    id_num = parts[0]
    # print(id_num)
    url = parts[1]
    # print(url)
    # part 2 is the base in format (coord1, coord2)
    base_coords = (float(parts[2].strip('(')), float(parts[3].strip(')')))
    # print(base_coords)
    # next part is the locations in format {location1: (coord1, coord2), location2: (coord1, coord2)}
    locations = {}
    location_line = line[line.find('{'):line.find('}')+1]
    locations = parse_locations(location_line)
    # print(locations)
    
    event_line = line[line.find('['):line.find(']')+1]
    # print(event_line)
    events = parse_events(event_line)
    # print(events)
    
    return {
        "id": id_num,
        "url": url,
        "base_coordinates": base_coords,
        "locations": locations,
        "events": events
    }


def parse_locations(locations_str: str) -> Dict[str, Tuple[float, float]]:
        """Parse the locations dictionary string."""
        locations = {}
        if locations_str and locations_str != '{}':
            # Clean up the string
            # print(locations_str)
            locations_str = locations_str.strip('{}')
            # print(locations_str)
            location_pairs = locations_str.split('),')
           
            # print(location_pairs)
    
            
            for pair in location_pairs:
                if ':' in pair:
                    try:
                        name, coords = pair.split(':')
                        coords = coords[2:]
                        coord1, coord2 = coords.split(',')
                        coords = (float(coord1), float(coord2.strip(')')))
                        name = name.strip().strip('"\'')
                        name = name.strip().strip("'")
                        locations[name] = coords
                    except:
                        continue
                   

            # print(locations)
          
                   
        return locations

def parse_events(events_str: str) -> List[Dict]:
        """Parse the events list string."""
        events = []
        if events_str and events_str != '[]':
            
            events_str = events_str.strip('[]\'')
            # Split by year pattern (\d{4}-) instead of hardcoded year
            event_pairs = re.split(r'(\d{4}-)', events_str)
            
            current_date = ""
            for part in event_pairs:
                if re.match(r'\d{4}-', part):  # This is a year part
                    current_date = part
                elif part.strip():  # This is the rest of the date and event
                    if ':' in part:
                        date_part, event = part.split(':', 1)
                        full_date = current_date + date_part.strip()
                        if 'No events' not in event and 'No specific event' not in event:
                            events.append({
                                "date": full_date,
                                "event": event.strip()
                            })

        return events

def generate_image_paths(id_num: str) -> str:
        """Generate image paths for all images in the ID folder"""\
        # check if directory exists
        if not os.path.exists(f"all_events/{id_num}"):
            return ""
        image_paths = []
        for image in os.listdir(f"all_events/{id_num}"):
            image_paths.append(f"all_events/{id_num}/{image}")
       
        return image_paths

def query_q_a(events) -> List[str]:

    # prompt
    prompt = f"""Given a set of statements in an order I'd like you to make 3 questions about the events described.\
        Make the questions diverse, covering different aspects of the events that could be aided answerable using satellite imagery of the event \
        Statements: \n{events}\
        Format your response exactly like this:\
        **Question 1:** [Your first question here]\
        **Answer 1:** [Your first answer as a complete sentence]\
        **Question 2:** [Your second question here\
        **Answer 2:** [Your second answer as a complete sentence]\
        **Question 3:** [Your third question here]\
        **Answer 3:** [Your third answer as a complete sentence]\
        \
        Here are some examples of statements: 2021-12-11:  No events described in the article are visible from this date. 2021-12-15: Very strong winds in Kansas, Texas, and Oklahoma caused numerous wildfires to spread rapidly. Blowing dust severely reduced visibility, causing streetlights to turn on at midday in some areas. 2021-12-16: A large wildfire in Russell and Ellis Counties, Kansas burned approximately 365,850 acres, destroying at least 10 homes.  High winds, gusting up to 100 mph, fueled the fire and other blazes across western Kansas, Oklahoma, and Texas. 2021-12-21: No events described in the article are visible from this date. 2021-12-26: No events described in the article are visible from this date. 2021-12-31: No events described in the article are visible from this date. 2022-01-05: No events described in the article are visible from this date. 2022-01-10: No events described in the article are visible from this date. 2022-01-15: No events described in the article are visible from this date. \
        Here are some examples of questions: \

        **Question 1:** What were the conditions that led to the rapid spread of wildfires in Kansas, Texas, and Oklahoma?\
        **Answer 1:** The conditions that led to the rapid spread of wildfires in Kansas, Texas, and Oklahoma were very strong winds, low humidity, and high temperatures.\
        **Question 2:** What was the impact of the wildfires in Russell and Ellis Counties, Kansas?\
        **Answer 2:** The impact of the wildfires in Russell and Ellis Counties, Kansas was the burning of approximately 365,850 acres and the destruction
        of at least 10 homes.\
        **Question 3:** when did the wildfires in Kansas, Texas, and Oklahoma occur?\
        **Answer 3:** The wildfires in Kansas, Texas, and Oklahoma occurred on December 15, 2021\
        """
    try:
        summary = model.generate_content(prompt)
        # make the query 
    except Exception as e:
        print(e)
        for i in range(3):
            sleep(100)
        try:
            summary = model.generate_content(prompt)
        except Exception as e:
            return ''
    if summary.text == 'no':
        return ''
    return summary.text
     


def create_training_example(task_type: "custom", question_id_base: int,
                              event_data: Dict,
                              image_paths: str = None, no_dates = bool) -> Dict:
        """Create a training example in the specified format."""
        id_num = event_data['id']
        # print(id_num)
        events = event_data['events']
        # print(events)
        locations = event_data['locations']
        # print(locations)
        dates = sorted(list(set(e['date'] for e in events)))
        # print(dates)

        # Generate image paths if not provided
        if image_paths is None:
            image_paths = generate_image_paths(id_num)
            if image_paths == "":
                return None
            
        base_example = {
            "folder_id": int(id_num),
            "video": image_paths,
            "dataset": "FemaEvents",
            "lat_lon": [event_data['base_coordinates']] * len(dates),
            "timestamp": [datetime.strptime(d, "%Y-%m-%d").isoformat() for d in dates],
            "sensor": ["satellite"] * len(dates)
        }

        # replaces keys with indexes for events
        if no_dates == True:
            no_dates_events = {}
            for i, event in enumerate(events):
                no_dates_events[i] = event['event']
            events = no_dates_events

        
        # includes pixel coordinates for locations
        pixel_locations = geo_to_pixel(locations, event_data['base_coordinates'])

        # augment locations in the events with pixel coordinates locations (pixel_coordinates)
        for order, description in events.items():
            # print("description before: ", description)
            for loc_name in locations:
                # find all instances of loc_name in description and replace with loc_name + pixel_coordinates
                description = re.sub(rf'\b{loc_name}\b', f"{loc_name} ({pixel_locations[loc_name][0]}, {pixel_locations[loc_name][1]})", description)
            # print("description after: ", description)
            events[order] = description

        # print(events)
        # exit(0)

        # Generate questions and answers using gemini
        # Generate questions and answers using gemini
        qa = query_q_a(events)

        if not qa or qa == 'no' or qa == '':
            return None

        # parse the questions and answers
        q_a = []

        # Extract Question 1 and Answer 1
        q1 = qa[qa.find('**Question 1:**')+len('**Question 1:**'):qa.find('**Answer 1:**')].strip()
        a1_start = qa.find('**Answer 1:**')+len('**Answer 1:**')
        a1_end = qa.find('**Question 2:**')
        a1 = qa[a1_start:a1_end].strip()
        q_a.append({"question": q1, "answer": a1})

        # Extract Question 2 and Answer 2
        q2 = qa[qa.find('**Question 2:**')+len('**Question 2:**'):qa.find('**Answer 2:**')].strip()
        a2_start = qa.find('**Answer 2:**')+len('**Answer 2:**')
        a2_end = qa.find('**Question 3:**')
        a2 = qa[a2_start:a2_end].strip()
        q_a.append({"question": q2, "answer": a2})

        # Extract Question 3 and Answer 3
        q3 = qa[qa.find('**Question 3:**')+len('**Question 3:**'):qa.find('**Answer 3:**')].strip()
        a3_start = qa.find('**Answer 3:**')+len('**Answer 3:**')
        a3 = qa[a3_start:].strip()
        q_a.append({"question": q3, "answer": a3})

        # add video tokens to the questions

        # find the first \n and add <video> do not replace the first \n, simply add <video> after it
        for i in range(len(q_a)):
            q_a[i]['question'] = "This is a sequence of satellite images:\n<video>\n"+ q_a[i]['question']

    
# **Answer 1:** Large-scale evacuations began near Helena, Jordan, and Roundup in Montana on August 30, 2020, due to high winds, low 80s temperatures, and a surge in wildfire activity.
        # Add the questions and answers to the example in the specified format
        examples = []

        for i,qa in enumerate(q_a):
            examples.append( {
                **base_example,
                "task": task_type,
                "conversations": [
                     {
                          "from": "human",
                          "value": qa["question"]
                        },
                        {
                          "from": "gpt",
                          "value": qa["answer"]
                        }
                ],
                "id": question_id_base

            }
            )
            question_id_base += 1
        return examples, question_id_base

if __name__ == "__main__":
    # Load the data file
    file = open('reorganized_total_data.csv', 'r')
    lines = file.readlines()
    file.close()
    
    print("number of lines in file: ", len(lines))

    train_lines = lines[:int(len(lines)*0.8)]
    test_lines = lines[int(len(lines)*0.8):]

    images_path = 'all_events'

    image_paths = {}
    for id_num in os.listdir(images_path):
        image_paths[id_num] = generate_image_paths(id_num)
    
    lines = train_lines

    dataset = []
    question_id_base = 0
    start_val = 0
    with open('train_generated_q_a.json', 'a+') as f:
        if start_val == 0:
            f.write('[')
        for line in tqdm(lines):
            event_data = parse_line(line)
            task_type = "custom"
            try:
                examples, question_id_base = create_training_example(task_type, question_id_base, event_data, image_paths[event_data['id']], True)
                if examples:
                    for example in examples:
                        # print(example)
                        dataset.append(example)
                        # write to json file
                        f.write(json.dumps(example, indent=2))
                        f.write(',')
                        start_val += 1
                        
            
            except Exception as e:
                print(e)
                continue
            f.flush()
        f.write(']')


    
    lines = test_lines

    dataset = []
    question_id_base = 0
    start_val = 0
    with open('test_generated_q_a.json', 'a+') as f:
        if start_val == 0:
            f.write('[')
        for line in tqdm(lines):
            event_data = parse_line(line)
            task_type = "custom"
            try:
                examples, question_id_base = create_training_example(task_type, question_id_base, event_data, image_paths[event_data['id']], True)
                if examples:
                    for example in examples:
                        # print(example)
                        dataset.append(example)
                        # write to json file
                        f.write(json.dumps(example, indent=2))
                        f.write(',')
                        start_val += 1
                        
            
            except Exception as e:
                print(e)
                continue
            f.flush()
        f.write(']')
