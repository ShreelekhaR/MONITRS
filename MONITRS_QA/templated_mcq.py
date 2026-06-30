import json
from typing import List, Dict, Tuple, Optional, Any
import ast
import random
import os
import re
from datetime import datetime
from copy import deepcopy
import pandas as pd

# Reuse the existing geo_to_pixel function
def geo_to_pixel(locations, center, radius=5):
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


class MultipleChoiceGenerator:
    def __init__(self):
        # Initialize a global ID counter for the class
        self.global_id_counter = 0
        
        # Define event types for classification questions
        self.event_types = ["Fire", "Flood", "Hurricane", "Tornado", "Earthquake", "Volcano", "Drought", "Landslide", "Storm"]
        
        # Define multiple choice templates
        self.mc_templates = {
            "temporal_grounding": {
                "templates": [
                    "Here are the dates for the images: {dates}. On which date did the {event_type} begin?",
                    "Looking at these satellite images from {dates}, when did the {event_type} start?",
                    "Based on this sequence of satellite images from {dates}, which date shows the first evidence of the {event_type}?"
                ]
            },
            "event_type": {
                "templates": [
                    "What type of event is shown in these satellite images?",
                    "What natural disaster is occurring in this location?",
                    "Based on these satellite images, what event happened at this location?"
                ]
            },
            "location_identification": {
                "templates": [
                    "Where is {location}? In pixel coordinates.",
                ]
            },
            "event_sequence": {
                "templates": [
                    "What is the correct sequence of events shown in these satellite images?",
                    "Which option correctly describes the progression of the {event_type}?",
                    "What sequence of events is shown in these images from {dates}?"
                ]
            }
        }

    def parse_line(self, line: str) -> Dict:
        """Parse a single line of the data file (CSV2 format)."""
        # Split line into main components
        parts = line.split(',')
        
        # Extract components
        id_num = parts[0]
        url = parts[1]
        
        # Parse base coordinates
        base_coords = (float(parts[2].strip('(')), float(parts[3].strip(')')))
        
        # Parse locations
        locations = {}
        location_line = line[line.find('{'):line.find('}')+1]
        locations = self._parse_locations(location_line)
        
        # Parse events
        event_line = line[line.find('['):line.find(']')+1]
        events = self._parse_events(event_line)
        
        return {
            "id": id_num,
            "url": url,
            "base_coordinates": base_coords,
            "locations": locations,
            "events": events
        }
    
    def parse_csv1_line(self, line: str) -> Dict:
        """Parse a single line from CSV1 format."""
        parts = line.strip().split(',')
        if len(parts) < 9:
            return None
            
        try:
            return {
                "id": parts[0],
                "event_type": parts[1],
                "start_date": parts[2],
                "end_date": parts[3],
                "state": parts[4],
                "county": parts[5],
                "name": parts[6],
                "latitude": float(parts[7]),
                "longitude": float(parts[8])
            }
        except Exception as e:
            print(f"Error parsing CSV1 line: {e}")
            return None

    def _parse_locations(self, locations_str: str) -> Dict[str, Tuple[float, float]]:
        """Parse the locations dictionary string."""
        locations = {}
        if locations_str and locations_str != '{}':
            locations_str = locations_str.strip('{}')
            location_pairs = locations_str.split('),')
            for pair in location_pairs:
                if ':' in pair:
                    try:
                        name, coords = pair.split(':')
                    except:
                        print(f"Error parsing location pair: {pair}")
                        continue

                    coords = coords[2:]
                    coord1, coord2 = coords.split(',')
                    coords = (float(coord1), float(coord2.strip(')')))
                    name = name.strip().strip('"\'')
                    name = name.strip().strip("'")

                    # clean up name remove any extra spaces and single and double quotes
                    name = re.sub(r"['\"]", "", name)
                    name = re.sub(r"\s+", " ", name).strip()
                    # remove any extra spaces
                    name = re.sub(r"\s+", " ", name).strip()
                    #remove slashes 
                    name = name.replace("/", " ")
                    # remove backslashes
                    name = name.replace("\\", " ")
                    
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
                        if 'No events' not in event and 'No specific event' not in event and 'No known significant events' not in event:
                            events.append({
                                "date": full_date,
                                "event": event.strip()
                            })
        return events

    def generate_image_paths(self, id_num: str) -> List[str]:
        """Generate image paths for all images in the ID folder"""
        # Check if directory exists
        if not os.path.exists(f"all_events/{id_num}"):
            return []
        
        image_paths = []
        for image in os.listdir(f"all_events/{id_num}"):
            image_paths.append(f"/scratch/datasets/ssr234/all_events/{id_num}/{image}")
        
        return image_paths

    def _detect_event_type(self, events: List[Dict]) -> str:
        """Detect the type of event from the event descriptions."""
        event_text = " ".join([e["event"].lower() for e in events])
        
        event_keywords = {
            "Fire": ["fire", "burn", "flame", "wildfire", "forest fire", "brush fire"],
            "Flood": ["flood", "inundation", "water level", "high water", "overflow"],
            "Hurricane": ["hurricane", "cyclone", "typhoon", "storm surge"],
            "Tornado": ["tornado", "twister", "funnel cloud"],
            "Earthquake": ["earthquake", "quake", "tremor", "seismic"],
            "Volcano": ["volcano", "eruption", "lava", "ash cloud"],
            "Drought": ["drought", "dry", "water shortage"],
            "Landslide": ["landslide", "mudslide", "rockslide", "debris flow"],
            "Evacuation": ["evacuation", "evacuate", "evacuated", "evacuees"]
        }
        
        for event_type, keywords in event_keywords.items():
            if any(keyword in event_text for keyword in keywords):
                return event_type
                
        return "Unknown"

    def _extract_significant_event_date(self, events: List[Dict]) -> Optional[str]:
        """Extract the date when a significant event occurred."""
        if not events:
            return None
            
        # Start with the first date that has a non-trivial event
        for event in events:
            if len(event["event"]) > 20 and "No known significant events" not in event["event"]:
                return event["date"]
                
        # If no significant event found, return the first date
        return events[0]["date"] if events else None

    def _find_event_beginning_date(self, events: List[Dict]) -> Optional[str]:
        """Find the date when an event began."""
        if not events:
            return None
            
        # Look for keywords indicating beginning of an event
        beginning_keywords = ["began", "started", "initiated", "first", "beginning", "outbreak", "outbreak of"]
        
        for event in events:
            event_text = event["event"].lower()
            if any(keyword in event_text for keyword in beginning_keywords):
                return event["date"]
        
        # If no clear beginning found, return the first date with a significant event
        return self._extract_significant_event_date(events)

    def _get_event_description(self, events: List[Dict]) -> str:
        """Extract a short description of the event."""
        if not events:
            return "event"
            
        # Concatenate all event descriptions
        all_text = " ".join([e["event"] for e in events])
        
        # Find a noun phrase that describes the event
        event_phrases = [
            "fire", "wildfire", "forest fire", "hurricane", "storm", "flooding", 
            "flood", "earthquake", "tsunami", "landslide", "evacuation", 
            "tornado", "disaster", "emergency", "drought"
        ]
        
        for phrase in event_phrases:
            if phrase in all_text.lower():
                return phrase
                
        # If no specific phrase found, extract the first 2-3 words after "The"
        match = re.search(r"The ([A-Z][a-z]+ [A-Z]?[a-z]+ ?[A-Z]?[a-z]+)", all_text)
        if match:
            return match.group(1)
            
        # Default to a generic description
        return "event"

    def _generate_options(self, correct_answer: str, option_pool: List[str], num_options: int = 4) -> Tuple[List[str], str]:
        """Generate multiple choice options with the correct answer included."""
        # Ensure we have enough options in the pool
        if len(option_pool) < num_options - 1:
            # Add some generic options if needed
            option_pool.extend(["None of the above", "Cannot be determined", "No event occurred"])
        
        # Select random options from the pool, excluding the correct answer
        other_options = [opt for opt in option_pool if opt != correct_answer]
        selected_options = random.sample(other_options, min(num_options - 1, len(other_options)))
        
        # Add the correct answer
        all_options = selected_options + [correct_answer]
        
        # Shuffle options
        random.shuffle(all_options)
        
        # Create option labels (a, b, c, d)
        option_labels = [chr(97 + i) for i in range(len(all_options))]
        
        # Find the correct answer label
        correct_label = option_labels[all_options.index(correct_answer)]
        
        # Format options as "a. option1", "b. option2", etc.
        formatted_options = [f"{label}. {option}" for label, option in zip(option_labels, all_options)]
        
        return formatted_options, correct_label

    def create_temporal_grounding_question(self, event_data: Dict) -> Dict:
        """Create a temporal grounding multiple choice question."""
        events = event_data['events']
        if not events or len(events) < 2:
            return None
            
        # Get all dates from events
        dates = sorted(list(set(e['date'] for e in events)))
        if len(dates) < 3:
            return None
            
        # Get event type and description
        # event_type = self._detect_event_type(events)
        event_type = event_data['event_type']
        event_description = self._get_event_description(events)
        
        # Find the beginning date of the event
        beginning_date = self._find_event_beginning_date(events)
        if not beginning_date:
            beginning_date = dates[0]
            
        # Select template
        template = random.choice(self.mc_templates["temporal_grounding"]["templates"])
        
        # Format dates for display
        formatted_dates = ", ".join(dates)
        
        # Generate question
        question = template.format(
            dates=formatted_dates,
            event_type=event_type,
            event_description=event_description
        )
        
        # Generate options
        options, correct_label = self._generate_options(
            beginning_date, 
            dates + ["No event occurred on these dates"]
        )
        
        return {
            "type": "temporal_grounding",
            "question": question,
            "options": options,
            "correct_answer": f"{correct_label}",
            "explanation": f"The {event_type.lower()} began on {beginning_date} as indicated by the satellite imagery."
        }

    def create_event_type_question(self, event_data: Dict) -> Dict:
        """Create an event type identification multiple choice question."""
        events = event_data['events']
        if not events:
            return None
            
        # Detect event type
        # event_type = self._detect_event_type(events)
        event_type = event_data['event_type']
        if "disaster" in event_type.lower():
            return None
            
        # Select template
        template = random.choice(self.mc_templates["event_type"]["templates"])
        
        # Generate question
        question = template
        
        # Generate options - use all event types as the option pool
        options, correct_label = self._generate_options(event_type, self.event_types)
        
        return {
            "type": "event_type",
            "question": question,
            "options": options,
            "correct_answer": f"{correct_label}",
            "explanation": f"The satellite images show a {event_type.lower()} event."
        }

    def create_location_identification_question(self, event_data: Dict) -> Dict:
        """Create a location identification multiple choice question."""
        locations = event_data['locations']
        events = event_data['events']
        
        if not locations or len(locations) < 2 or not events:
            return None
            
        # Detect event type and description
        event_type = event_data['event_type']
        
        # Choose a random location as the correct answer
        location_names = list(locations.keys())
        
        chosen_location = random.choice(location_names)

        # offer locations in format name: (lat, lon) 
        chosen_location_coords = locations[chosen_location]


        question_location = chosen_location + " (" + str(chosen_location_coords[0]) + ", " + str(chosen_location_coords[1]) + ")"

        # correct pixel coordinates
        correct_pixel_coords = geo_to_pixel({chosen_location: chosen_location_coords}, event_data['base_coordinates'])
        correct_pixel_coords = correct_pixel_coords[chosen_location]

        # Select template
        template = random.choice(self.mc_templates["location_identification"]["templates"])
        
        # Generate question
        question = template.format(
            location=question_location
        )
        
        # generate random pixel coordinates for other optionn
        random_pixel_coords = []
        for i in range(3):
            coords_x = random.randint(0, 511)
            coords_y = random.randint(0, 511)
            random_pixel_coords.append((coords_x, coords_y))
        
        # Generate options
        options, correct_label = self._generate_options(correct_pixel_coords, random_pixel_coords + [correct_pixel_coords])
        
        return {
            "type": "location_identification",
            "question": question,
            "options": options,
            "correct_answer": f"{correct_label}",
            "explanation": f"The {chosen_location} is located at {correct_pixel_coords} as shown in the satellite imagery."
        }

    def create_damage_assessment_question(self, event_data: Dict) -> Dict:
        """Create a damage assessment multiple choice question."""
        locations = event_data['locations']
        events = event_data['events']
        
        if not locations or len(locations) < 2 or not events:
            return None
            
        # Detect event type and description
        # event_type = self._detect_event_type(events)
        event_type = event_data['event_type']
        event_description = self._get_event_description(events)
        
        # Choose a random location as the most damaged area
        location_names = list(locations.keys())
        most_damaged = random.choice(location_names)
        
        # Select template
        template = random.choice(self.mc_templates["damage_assessment"]["templates"])
        
        # Generate question
        question = template.format(
            event_type=event_type,
            event_description=event_description
        )
        
        # Generate options
        options, correct_label = self._generate_options(most_damaged, location_names)
        
        return {
            "type": "damage_assessment",
            "question": question,
            "options": options,
            "correct_answer": f"{correct_label}",
            "explanation": f"The satellite images show that {most_damaged} experienced the greatest damage from the {event_type.lower()}."
        }

    def create_event_sequence_question(self, event_data: Dict) -> Dict:
        """Create an event sequence multiple choice question."""
        events = event_data['events']
        
        if not events or len(events) < 3:
            return None
            
        # Sort events by date
        sorted_events = sorted(events, key=lambda e: e['date'])
        
        # Create a correct sequence description
        correct_sequence = "; ".join([f"{e['date']}: {e['event'][:30]}..." for e in sorted_events[:3]])
        
        # Create incorrect sequences by shuffling dates
        sequences = [correct_sequence]
        
        # Create 3 incorrect sequences
        for _ in range(3):
            shuffled_events = sorted_events.copy()
            random.shuffle(shuffled_events)
            if shuffled_events == sorted_events:
                # Ensure it's different from the correct sequence
                shuffled_events[0], shuffled_events[1] = shuffled_events[1], shuffled_events[0]
            
            incorrect_sequence = "; ".join([f"{e['date']}: {e['event'][:30]}..." for e in shuffled_events[:3]])
            sequences.append(incorrect_sequence)
        
        # Detect event type
        # event_type = self._detect_event_type(events)
        event_type = event_data['event_type']
        
        # Select template
        template = random.choice(self.mc_templates["event_sequence"]["templates"])
        
        # Get all dates
        dates = sorted(list(set(e['date'] for e in events)))
        formatted_dates = ", ".join(dates)
        
        # Generate question
        question = template.format(
            event_type=event_type,
            dates=formatted_dates
        )
        
        # Generate options
        options, correct_label = self._generate_options(correct_sequence, sequences)
        
        return {
            "type": "event_sequence",
            "question": question,
            "options": options,
            "correct_answer": f"{correct_label}",
            "explanation": f"The correct sequence of events is: {correct_sequence}"
        }

    def create_multiple_choice_example(self, event_data: Dict, image_paths: List[str] = None) -> List[Dict]:
        """Create multiple choice questions for the event data."""
        if not event_data['events']:
            return []
            
        # Generate image paths if not provided
        if image_paths is None or not image_paths:
            image_paths = self.generate_image_paths(event_data['id'])
            if not image_paths:
                return []
        
        # Get dates
        dates = sorted(list(set(e['date'] for e in event_data['events'])))
        
        # Base example structure
        base_example = {
            "folder_id": int(event_data['id']),
            "video": image_paths,
            "dataset": "FemaEvents",
            "lat_lon": [event_data['base_coordinates']] * len(dates),
            "timestamp": dates,
            "sensor": ["satellite"] * len(dates)
        }
        
        examples = []
        
        # Create different types of questions
        question_creators = [
            self.create_temporal_grounding_question,
            self.create_event_type_question,
            self.create_location_identification_question,
            # self.create_damage_assessment_question,
            # self.create_event_sequence_question
        ]
        
        for creator in question_creators:
            question_data = creator(event_data)
            if question_data:
                # Add to examples
                example = deepcopy(base_example)
                example.update({
                    # we want the type of question to be the same as the type of question from template
                    "task": question_data['type'],
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"This is a sequence of sentinel-2 satellite images, centered at {event_data['base_coordinates']}:\n<video>\n{question_data['question']}\n{chr(10).join(question_data['options'])}"
                        },
                        {
                            "from": "gpt",
                            "value": f"{question_data['correct_answer']} \n{question_data['explanation']}"
                        }
                    ],
                    "id": self.global_id_counter
                })
                examples.append(example)
                self.global_id_counter += 1

        return examples

    def process_file(self, 
                    input_lines: List[str], 
                    image_paths: Dict[str, List[str]],event_types:Dict[str, str] ) -> List[Dict]:
        """Process the entire file and create multiple choice examples."""
        dataset = []
        # Reset global ID counter
        self.global_id_counter = 0
        
        for line in input_lines:
            if not line.strip():
                continue
            
            event_data = self.parse_line(line)
            # add event type to event_data
            event_data['event_type'] = event_types.get(event_data['id'], "natural disaster")
            
            paths = image_paths.get(event_data['id'])
            if not paths:
                print(f"No image paths found for ID: {event_data['id']}")
                continue
            
            # Create multiple choice examples
            examples = self.create_multiple_choice_example(event_data, paths)
            dataset.extend(examples)
            # try:
            #     event_data = self.parse_line(line)
            #     # add event type to event_data
            #     event_data['event_type'] = event_types.get(event_data['id'], "natural disaster")
                
            #     paths = image_paths.get(event_data['id'])
            #     if not paths:
            #         print(f"No image paths found for ID: {event_data['id']}")
            #         continue
                
            #     # Create multiple choice examples
            #     examples = self.create_multiple_choice_example(event_data, paths)
            #     dataset.extend(examples)
                
            # except Exception as e:
            #     print(f"Error processing line: {e}")
            #     continue
                
        return dataset

    def merge_csv1_data(self, dataset: List[Dict], csv1_lines: List[str]) -> List[Dict]:
        """Merge additional data from CSV1 format into the dataset."""
        # Create a mapping of IDs to CSV1 data
        csv1_data = {}
        for line in csv1_lines:
            if not line.strip():
                continue
                
            data = self.parse_csv1_line(line)
            if data:
                csv1_data[data["id"]] = data
        
        # Merge data into the dataset
        for example in dataset:
            folder_id = str(example["folder_id"])
            if folder_id in csv1_data:
                # Add additional information from CSV1
                example["event_type"] = csv1_data[folder_id]["event_type"]
                example["event_name"] = csv1_data[folder_id]["name"]
                example["state"] = csv1_data[folder_id]["state"]
                example["county"] = csv1_data[folder_id]["county"]
                
        return dataset


# Example usage
if __name__ == "__main__":
    from load_v2 import load_all_v1_format, get_image_paths

    generator = MultipleChoiceGenerator()

    # Load v2 data
    all_events = load_all_v1_format()
    event_ids = sorted(all_events.keys(), key=int)
    print(f"Loaded {len(event_ids)} events")

    # 80/20 train/test split
    split = int(len(event_ids) * 0.8)
    train_ids = event_ids[:split]
    test_ids = event_ids[split:]

    for split_name, ids in [('train', train_ids), ('test', test_ids)]:
        image_paths = {}
        event_types = {}
        lines_for_process = []

        for eid in ids:
            edata = all_events[eid]
            paths = get_image_paths(eid)
            if not paths:
                continue
            image_paths[eid] = paths
            event_types[eid] = edata['event_type']
            # Build a fake CSV line that parse_line can handle, OR
            # directly create examples using the v1-format dict
            lines_for_process.append(edata)

        print(f"{split_name}: {len(lines_for_process)} events with images")

        # Process using v1-format dicts directly
        dataset = []
        generator.global_id_counter = 0
        for edata in lines_for_process:
            edata['event_type'] = event_types.get(edata['id'], 'Unknown')
            paths = image_paths.get(edata['id'])
            if not paths:
                continue
            examples = generator.create_multiple_choice_example(edata, paths)
            dataset.extend(examples)

        with open(f'new_{split_name}_multiple_choice.json', 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} questions to new_{split_name}_multiple_choice.json")