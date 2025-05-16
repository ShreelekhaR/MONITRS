import json
from typing import List, Dict, Tuple
import ast
from datetime import datetime
import os
import re

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


class EventDataProcessor:
    def __init__(self):
        # Initialize a global ID counter for the class
        self.global_id_counter = 0
        
        self.task_templates = {
            "change_detection": {
                "prompt": "This is a sequence of satellite images:\n<video>\nGiven these satellite images for {dates}, describe any changes or events that occurred.",
                "response_template": "{event}"
            },
            "temporal_grounding": {
                "templates": [
                    "This is a sequence of satellite images:\n<video>\nWhat events occurred on {date}?",
                    "This is a sequence of satellite images:\n<video>\nDescribe the sequence of events between {start_date} and {end_date}.",
                    "This is a sequence of satellite images:\n<video>\nWhat happened on {date}?"
                ]
            },
            "captioning": {
                "templates": [
                    "This is a sequence of satellite images:\n<video>\nDescribe in detail everything you can see in these satellite images.",
                    "This is a sequence of satellite images:\n<video>\nWhat changes or events are visible in these satellite images? Describe all visible details.",
                    "This is a sequence of satellite images:\n<video>\nProvide a comprehensive description of what you observe in these satellite images."
                ]
            },
            "location_grounding": {
                "templates": [
                    "This is a sequence of satellite images at {base_coordinates}:\n<video>\nWhat are the coordinates of {location}?",
                ]
            }
        }

    def parse_line(self, line: str) -> Dict:
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
        locations = self._parse_locations(location_line)
        # print(locations)
        
        event_line = line[line.find('['):line.find(']')+1]
        # print(event_line)
        events = self._parse_events(event_line)
        # print(events)
      
        return {
            "id": id_num,
            "url": url,
            "base_coordinates": base_coords,
            "locations": locations,
            "events": events
        }

    def _parse_locations(self, locations_str: str) -> Dict[str, Tuple[float, float]]:
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
                    name, coords = pair.split(':')
                    coords = coords[2:]
                    coord1, coord2 = coords.split(',')
                    coords = (float(coord1), float(coord2.strip(')')))
                    name = name.strip().strip('"\'')
                    name = name.strip().strip("'")
                   
                    locations[name] = coords
                   
        return locations

    def _parse_events(self, events_str: str) -> List[Dict]:
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

    def generate_image_paths(self,id_num: str) -> str:
            """Generate image paths for all images in the ID folder"""\
            # check if directory exists
            if not os.path.exists(f"all_events/{id_num}"):
                return ""
            image_paths = []
            for image in os.listdir(f"all_events/{id_num}"):
                image_paths.append(f"/scratch/datasets/ssr234/all_events/{id_num}/{image}")

        
            return image_paths

    def has_evacuation_events(self, events: List[Dict]) -> bool:
        """Check if any events mention evacuations."""
        evacuation_keywords = ['evacuat', 'shelter']
        for event in events:
            event_text = event['event'].lower()
            if any(keyword in event_text for keyword in evacuation_keywords):
                return True
        return False

    def create_training_example(self, 
                              task_type: str,
                              event_data: Dict,
                              image_paths: List[str] = None,
                              no_dates: bool = False) -> Dict:
        """Create a training example in the specified format."""
        id_num = event_data['id']
        # print(id_num)
        events = event_data['events']
        # print(events)
        locations = event_data['locations']
        # print(locations)
        dates = sorted(list(set(e['date'] for e in events)))

        # # Generate image paths if not provided
        # if image_paths is None:
        #     image_paths = generate_image_paths(id_num)
        #     if image_paths == "":
        #         return None
            
        # includes pixel coordinates for locations
        pixel_locations = geo_to_pixel(locations, event_data['base_coordinates'])

        # augment location names with pixel coordinates so before: {'location1': (lat, lon), 'location2': (lat, lon)} --> {'location1 (pixel_coords)': (lat, lon), 'location2 (pixel_coords)': (lat, lon)}
        augmented_locations = {}
        for loc_name, coords in locations.items():
            pixel_coords = pixel_locations[loc_name]
            if pixel_coords:
                augmented_locations[f"{loc_name} {pixel_coords}"] = f"{coords} {pixel_coords}"
            else:
                augmented_locations[loc_name] = coords
        # print("------------------------augmented_locations------------------------")
        # print(augmented_locations)
        # exit()
        locations = augmented_locations


            
        base_example = {
            "folder_id": int(id_num),
            "video": image_paths,
            "dataset": "FemaEvents",
            "lat_lon": [event_data['base_coordinates']] * len(dates),
            "timestamp": dates,  # Keep dates in YYYY-MM-DD format without datetime conversion
            "sensor": ["satellite"] * len(dates)
        }
        
        # replaces keys with indexes for events
        if no_dates == True:
            no_dates_events = {}
            for i, event in enumerate(events):
                no_dates_events[i] = event['event']
            events = no_dates_events

        if task_type == "change_detection":
            # print("change_detection")
            example = {
                **base_example,
                "task": "change_detection",
                "conversations": [
                    {   
                        "from": "human",
                        "value": self.task_templates["change_detection"]["prompt"].format(
                            dates=f"{dates[0]} to {dates[-1]}"
                        )
                    },
                    {
                        "from": "gpt",
                        "value": " ".join(e['event'] for e in events)
                    }
                ],
                "id": self.global_id_counter  # Use the global counter
            }
            self.global_id_counter += 1  # Increment counter
            return example
        
        elif task_type == "temporal_grounding":
            examples = []
            # Add base templates
            templates = self.task_templates["temporal_grounding"]["templates"]

            for template in templates:
                if "{date}" in template:
                    for date in dates:
                        date_events = [e for e in events if e['date'] == date]
                        if date_events:
                            examples.append({
                                **base_example,
                                "task": "temporal_grounding",
                                "conversations": [
                                    {
                                        "from": "human",
                                        "value": template.format(date=date)
                                    },
                                    {
                                        "from": "gpt",
                                        "value": " ".join(e['event'] for e in date_events)
                                    }
                                ],
                                "id": self.global_id_counter  # Use the global counter
                            })
                            self.global_id_counter += 1  # Increment counter
            return examples

        elif task_type == "location_grounding":

            examples = []
            for i, template in enumerate(self.task_templates["location_grounding"]["templates"]):
                if "{base_coordinates}" in template:
                    template = template.replace("{base_coordinates}", str(event_data['base_coordinates']))
                if "{location}" in template:
                    # we want max 5 locations
                    for loc_name, coords in list(locations.items())[:5]:
                    # for loc_name, coords in locations.items():
                        examples.append({
                            **base_example,
                            "task": "location_grounding",
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": template.format(location=loc_name)
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The coordinates are {coords}"
                                }
                            ],
                            "id": self.global_id_counter  # Use the global counter
                        })
                        self.global_id_counter += 1  # Increment counter
            return examples

        elif task_type == "captioning":
            examples = []
            for i, template in enumerate(self.task_templates["captioning"]["templates"]):
                examples.append({
                    **base_example,
                    "task": "dense_captioning",
                    "conversations": [
                        {
                            "from": "human",
                            "value": template
                        },
                        {
                            "from": "gpt",
                            "value": " ".join(e['event'] for e in events)
                        }
                    ],
                    "id": self.global_id_counter  # Use the global counter
                })
                self.global_id_counter += 1  # Increment counter
            return examples
        
        elif task_type == "temporal_location_grounding":
            examples = []
            for template in self.task_templates["temporal_location_grounding"]["templates"]:
                for loc_name, coords in locations.items():
                    if "{start_date}" in template and "{end_date}" in template:
                        # Handle date range templates
                        examples.append({
                            **base_example,
                            "task": "temporal_location_grounding",
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": template.format(
                                        location=loc_name, 
                                        start_date=dates[0], 
                                        end_date=dates[-1]
                                    )
                                },
                                {
                                    "from": "gpt",
                                    "value": " ".join(e['event'] for e in events)
                                }
                            ],
                            "id": self.global_id_counter  # Use the global counter
                        })
                        self.global_id_counter += 1  # Increment counter
                    elif "{date}" in template:
                        # Handle single date templates
                        for date in dates:
                            date_events = [e for e in events if e['date'] == date]
                            if date_events:
                                examples.append({
                                    **base_example,
                                    "task": "temporal_location_grounding",
                                    "conversations": [
                                        {   "from": "human",
                                            "value": template.format(
                                                location=loc_name,
                                                date=date
                                            )
                                        },
                                        {
                                            "from": "gpt",
                                            "value": " ".join(e['event'] for e in date_events)
                                        }
                                    ],
                                    "id": self.global_id_counter  # Use the global counter
                                })
                                self.global_id_counter += 1  # Increment counter
            return examples
        return None

    def process_file(self, 
                    input_lines: List[str], 
                    image_paths: Dict[str, List[str]]) -> List[Dict]:
        """Process the entire file and create training examples."""
    
        dataset = []
        # Reset global ID counter when processing a new file
        self.global_id_counter = 0
        
        for line in input_lines:
            if not line.strip():
                continue
                
            try:
                event_data = self.parse_line(line)
 
               
                paths = image_paths.get(event_data['id'])
                if not paths:
                    print(f"No image paths found for ID: {event_data['id']}")
                    continue
               
                
                # Determine which tasks to run based on available data
                tasks = ["change_detection", "temporal_grounding", "captioning"]
                if event_data['locations']:  # Only add location_grounding if we have locations
                    tasks.append("location_grounding")
                    # tasks.append("temporal_location_grounding")

                # print(tasks)    
                    
                for task_type in tasks:
                    result = self.create_training_example(task_type, event_data, paths, False)
                    if isinstance(result, list):
                        dataset.extend(result)
                    elif result:
                        dataset.append(result)
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
           
        return dataset

# Example usage
if __name__ == "__main__":
    processor = EventDataProcessor()
    

    file = open('reorganized_total_data.csv', 'r')
    lines = file.readlines()
    
    print("number of lines in file: ", len(lines))
    # train / test split
    # keep first 80% for training and last 20% for testing
    train_lines = lines[:int(len(lines)*0.8)]
    test_lines = lines[int(len(lines)*0.8):]

    lines = train_lines
 
    # use only ids that are in file
    ids = []
    for line in lines:
        id_num = line.split(',')[0]
        ids.append(id_num)

    images_path = 'all_events'

    image_paths = {}
    for id_num in ids:
        image_paths[id_num] = processor.generate_image_paths(id_num)
    
    print("number of image paths: ", len(image_paths))
  
    
    dataset = processor.process_file(lines, image_paths)

    with open('train_templated_q_a.json', 'w+') as f:
        json.dump(dataset, f, indent=2)

    lines = test_lines

    # use only ids that are in file
    ids = []
    for line in lines:
        id_num = line.split(',')[0]
        ids.append(id_num)

    images_path = 'all_events'

    image_paths = {}
    for id_num in ids:
        image_paths[id_num] = processor.generate_image_paths(id_num)
    
    print("number of image paths: ", len(image_paths))
  
    
    dataset = processor.process_file(lines, image_paths)

    with open('test_templated_q_a.json', 'w+') as f:
        json.dump(dataset, f, indent=2)
    