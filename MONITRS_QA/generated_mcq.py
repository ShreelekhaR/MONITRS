# given a set of captions, generate multiple choice questions and answers

import json
from typing import List, Dict, Tuple
import ast
from datetime import datetime
import os
import re
import random
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

def query_multiple_choice_q_a(events) -> List[str]:
    """Generate multiple choice questions and answers using Gemini."""
    # prompt
    prompt = f"""Given a set of statements in an order I'd like you to make 3 multiple choice questions about the events described.
        Make the questions diverse, covering different aspects of the events that could be answerable using satellite imagery of the event.
        Each question should have 4 options (A, B, C, and D) with only one correct answer.
        
        Statements: \n{events}
        
        Format your response exactly like this:
        **Question 1:** [Your first question here]
        A) [First option]
        B) [Second option]
        C) [Third option]
        D) [Fourth option]
        **Correct Answer 1:** [Correct option letter]
        
        **Question 2:** [Your second question here]
        A) [First option]
        B) [Second option]
        C) [Third option]
        D) [Fourth option]
        **Correct Answer 2:** [Correct option letter]
        
        **Question 3:** [Your third question here]
        A) [First option]
        B) [Second option]
        C) [Third option]
        D) [Fourth option]
        **Correct Answer 3:** [Correct option letter]
        
        Here are some examples of statements: 2021-12-11: No events described in the article are visible from this date. 2021-12-15: Very strong winds in Kansas, Texas, and Oklahoma caused numerous wildfires to spread rapidly. Blowing dust severely reduced visibility, causing streetlights to turn on at midday in some areas. 2021-12-16: A large wildfire in Russell and Ellis Counties, Kansas burned approximately 365,850 acres, destroying at least 10 homes. High winds, gusting up to 100 mph, fueled the fire and other blazes across western Kansas, Oklahoma, and Texas. 2021-12-21: No events described in the article are visible from this date.
        
        Here are some examples of questions:
        
        **Question 1:** What natural disaster is visible in the satellite images from mid-December 2021?
        A) Hurricane
        B) Tornado
        C) Wildfire
        D) Flooding
        **Correct Answer 1:** C
        
        **Question 2:** Approximately how many acres were burned in Russell and Ellis Counties, Kansas?
        A) 36,585 acres
        B) 365,850 acres
        C) 3,658 acres
        D) 3,658,500 acres
        **Correct Answer 2:** B
        
        **Question 3:** What weather condition contributed significantly to the spread of wildfires in December 2021?
        A) Heavy rainfall
        B) Strong winds
        C) Freezing temperatures
        D) High humidity
        **Correct Answer 3:** B
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


def parse_multiple_choice_qa(qa_text):
    """Parse the multiple choice questions and answers from the model's response."""
    q_a = []
    
    # Find all question and answer blocks
    for i in range(1, 4):  # For questions 1-3
        question_marker = f"**Question {i}:**"
        next_question_marker = f"**Question {i+1}:**" if i < 3 else None
        answer_marker = f"**Correct Answer {i}:**"
        
        # Find the question
        q_start = qa_text.find(question_marker) + len(question_marker)
        if next_question_marker:
            ans_end = qa_text.find(next_question_marker)
        else:
            ans_end = len(qa_text)
            
        q_block_end = qa_text.find(answer_marker, q_start)
        if q_block_end == -1 or q_block_end > ans_end:
            # Malformed response, skip this question
            continue
            
        # Extract question and options
        q_and_options = qa_text[q_start:q_block_end].strip()
        
        # Split the question from options
        lines = q_and_options.split('\n')
        question = lines[0].strip()
        
        # Extract options (A-D)
        options = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith(('A)', 'B)', 'C)', 'D)')):
                option_letter = line[0]
                option_text = line[2:].strip()
                options.append({"letter": option_letter, "text": option_text})
        
        # Extract correct answer
        ans_start = qa_text.find(answer_marker, q_block_end) + len(answer_marker)
        if next_question_marker and qa_text.find(next_question_marker, ans_start) != -1:
            ans_end = qa_text.find(next_question_marker, ans_start)
        else:
            ans_end = len(qa_text)
            
        correct_answer = qa_text[ans_start:ans_end].strip()
        
        # Create the QA object
        if options and correct_answer:
            q_a.append({
                "question": question,
                "options": options,
                "correct_answer": correct_answer
            })
    
    return q_a


def create_training_example(task_type: str, question_id_base: int,
                            event_data: Dict,
                            image_paths: str = None, no_dates: bool = False) -> Dict:
    """Create a training example in the specified format."""
    id_num = event_data['id']
    events = event_data['events']
    locations = event_data['locations']
    dates = sorted(list(set(e['date'] for e in events)))

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
    if no_dates:
        no_dates_events = {}
        for i, event in enumerate(events):
            no_dates_events[i] = event['event']
        events = no_dates_events

    # includes pixel coordinates for locations
    pixel_locations = geo_to_pixel(locations, event_data['base_coordinates'])

    # augment locations in the events with pixel coordinates 
    for order, description in events.items():
        for loc_name in locations:
            description = re.sub(rf'\b{loc_name}\b', f"{loc_name} ({pixel_locations[loc_name][0]}, {pixel_locations[loc_name][1]})", description)
        events[order] = description

    # Generate multiple choice questions and answers using gemini
    qa_text = query_multiple_choice_q_a(events)

    if not qa_text or qa_text == 'no' or qa_text == '':
        return None

    # Parse the multiple choice questions and answers
    q_a = parse_multiple_choice_qa(qa_text)

    if not q_a:
        return None

    # Add video tokens to the questions
    for i in range(len(q_a)):
        q_a[i]['question'] = "This is a sequence of satellite images:\n<video>\n" + q_a[i]['question']

    # Format answers for the API
    examples = []
    for i, qa in enumerate(q_a):
        # Format the multiple choice options
        options_text = "\n".join([f"{option['letter']}) {option['text']}" for option in qa["options"]])
        full_question = f"{qa['question']}\n\n{options_text}"
        
        # Format the answer response
        correct_letter = qa['correct_answer']
        correct_text = next((opt['text'] for opt in qa['options'] if opt['letter'] == correct_letter), "")
        full_answer = f"The correct answer is {correct_letter}) {correct_text}"
        
        examples.append({
            **base_example,
            "task": task_type,
            "conversations": [
                {
                    "from": "human",
                    "value": full_question
                },
                {
                    "from": "gpt",
                    "value": full_answer
                }
            ],
            "id": question_id_base,
            "metadata": {
                "correct_answer": correct_letter,
                "all_options": qa['options']
            }
        })
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
    with open('train_generated_multiple_choice_q_a.json', 'a+') as f:
        if start_val == 0:
            f.write('[')
        for line in tqdm(lines):
            event_data = parse_line(line)
            task_type = "multiple_choice"
            try:
                examples, question_id_base = create_training_example(task_type, question_id_base, event_data, image_paths.get(event_data['id']), True)
                if examples:
                    for example in examples:
                        dataset.append(example)
                        # write to json file
                        f.write(json.dumps(example, indent=2))
                        f.write(',')
                        start_val += 1
            
            except Exception as e:
                print(e)
                continue
            f.flush()
        # Remove the trailing comma and close the array
        f.seek(f.tell() - 1, 0)
        f.write(']')

    # Process test lines
    dataset = []
    question_id_base = 0
    start_val = 0
    with open('test_generated_multiple_choice_q_a.json', 'a+') as f:
        if start_val == 0:
            f.write('[')
        for line in tqdm(lines):
            event_data = parse_line(line)
            task_type = "multiple_choice"
            try:
                examples, question_id_base = create_training_example(task_type, question_id_base, event_data, image_paths.get(event_data['id']), True)
                if examples:
                    for example in examples:
                        dataset.append(example)
                        # write to json file
                        f.write(json.dumps(example, indent=2))
                        f.write(',')
                        start_val += 1
            
            except Exception as e:
                print(e)
                continue
            f.flush()
        # Remove the trailing comma and close the array
        f.seek(f.tell() - 1, 0)
        f.write(']')