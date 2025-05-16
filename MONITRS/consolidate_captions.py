import os
import re
import csv
import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

def parse_data(csv_content: str) -> List[Dict]:
    """
    Parse the data from CSV content.
    
    Args:
        csv_content: String containing the CSV data
        
    Returns:
        List of dictionaries representing parsed rows
    """
    rows = []
    # Split by newline while preserving multiline fields
    lines = csv_content.split('\n')
    
    # Join lines that are part of the same record
    row_pattern = re.compile(r'^(\d+),')
    current_row = ""
    
    for line in lines:
        if row_pattern.match(line) and current_row:
            rows.append(current_row)
            current_row = line
        elif line:
            current_row += line
    
    if current_row:  # Add the last row
        rows.append(current_row)
    
    # Parse each row
    parsed_rows = []
    for row in rows:
        # Extract fields using regex to handle complex structure
        match = re.match(r'^(\d+),(.*?),\((.*?)\),(.*?),\[(.*?)\]"?$', row)
        # filter " from  locations
        match = re.sub(r'"', '', match.group(4))
        if match:
            parsed_rows.append({
                'id': match.group(1),
                'url': match.group(2),
                'coordinates': match.group(3),
                'locations': match.group(4),
                'statements': match.group(5)
            })
    
    return parsed_rows

def extract_dated_statements(statements_text: str, filter_no_events: bool = True) -> List[Dict]:
    """
    Extract dated statements from the text, optionally filtering out non-informative ones.
    
    Args:
        statements_text: Text containing dated statements
        filter_no_events: Whether to filter out "no events" statements
        
    Returns:
        List of dictionaries with date and statement
    """
    statements = []
    # Use regex to extract date-statement pairs
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}):\s*(.*?)(?=\s*\d{4}-\d{2}-\d{2}:|$)')
    
    no_event_indicators = [
        "No events", "No significant updates", "No specific event", 
        "Information on", "not provided", "No event from the provided article"
    ]
    
    for match in date_pattern.finditer(statements_text):
        date = match.group(1)
        statement = match.group(2).strip()
        
        # Determine if statement is non-informative
        is_no_event = any(phrase in statement for phrase in no_event_indicators)
        
        # Add statement if it's informative or if we're not filtering
        if not filter_no_events or not is_no_event:
            statements.append({
                'date': date,
                'statement': statement,
                'is_no_event': is_no_event
            })
    
    return statements

def get_image_dates_by_row(image_folder: str) -> Dict[str, List[str]]:
    """
    Retrieve image dates from the filenames in the given folder structure,
    organized by row index (from your folder structure).
    
    Args:
        image_folder: Path to the base folder containing all_events
        
    Returns:
        Dictionary mapping row indices to lists of dates (YYYY-MM-DD)
    """
    image_dates_by_row = defaultdict(set)
    all_events_folder = os.path.join(image_folder, "all_events")
    
    # Check if all_events folder exists
    if not os.path.exists(all_events_folder):
        print(f"Warning: {all_events_folder} not found.")
        return {}
    
    # Iterate through row folders
    for row_index in os.listdir(all_events_folder):
        row_folder = os.path.join(all_events_folder, row_index)
        
        if os.path.isdir(row_folder):
            # Extract dates from filenames in this row folder
            date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
            
            for filename in os.listdir(row_folder):
                match = date_pattern.search(filename)
                if match:
                    date_str = match.group(1)
                    image_dates_by_row[row_index].add(date_str)
                    # print(f"Found image date for row {row_index}: {date_str}")
    
    # Convert sets to sorted lists
    return {row: sorted(list(dates)) for row, dates in image_dates_by_row.items()}

def consolidate_statements(statements: List[Dict], image_dates: List[str]) -> List[Dict]:
    """
    Consolidate statements based on image dates.
    
    Args:
        statements: List of dictionaries with date and statement
        image_dates: List of dates for which images are available
        
    Returns:
        List of dictionaries with image date and consolidated statements
    """
    # If no image dates, return empty list
    if not image_dates:
        return []
        
    # Sort statements by date
    sorted_statements = sorted(statements, key=lambda x: x['date'])
    
    # Sort image dates
    sorted_image_dates = sorted(image_dates)
    
    # Create date ranges from image dates
    ranges = []
    
    # First range: beginning of time to first image date
    first_date = sorted_image_dates[0]
    ranges.append({
        'start': '0000-00-00',
        'end': first_date,
        'image_date': first_date,
        'statements': []
    })
    
    # Middle ranges: between consecutive image dates
    for i in range(1, len(sorted_image_dates)):
        ranges.append({
            'start': sorted_image_dates[i-1],
            'end': sorted_image_dates[i],
            'image_date': sorted_image_dates[i],
            'statements': []
        })
    
    # Assign statements to appropriate date ranges
    for statement in sorted_statements:
        statement_date = statement['date']
        
        # Skip statements that are after the last image date
        if statement_date > sorted_image_dates[-1]:
            continue
            
        for range_info in ranges:
            if range_info['start'] < statement_date <= range_info['end']:
                range_info['statements'].append(statement['statement'])
                break
    
    # Create consolidated output
    consolidated = []
    for range_info in ranges:
        # If there are meaningful statements for this range, consolidate them
        if range_info['statements']:
            consolidated.append({
                'image_date': range_info['image_date'],
                'consolidated_statements': ' '.join(range_info['statements'])
            })
        else:
            # If no meaningful statements, report that there were no events
            consolidated.append({
                'image_date': range_info['image_date'],
                'consolidated_statements': "No known significant events reported for this timeframe."
            })
    
    return consolidated

def reorganize_data(csv_content: str, image_folder: str) -> List[Dict]:
    """
    Reorganize wildfire data by consolidating statements based on available image dates.
    
    Args:
        csv_content: String containing the CSV data
        image_folder: Path to the folder containing dated images
        
    Returns:
        List of dictionaries with reorganized data
    """
    # Parse the CSV data
    parsed_data = parse_data(csv_content)
    
    # Get image dates by row from the folder
    image_dates_by_row = get_image_dates_by_row(image_folder)
    
    # If no image dates found, provide a message
    if not image_dates_by_row:
        print("No image dates found in the folder structure. Using test dates for demonstration.")
        # Example test dates for each row
        image_dates_by_row = {
            "0": ["2022-07-18", "2022-07-28"],
            "1": ["2022-05-10", "2022-05-20", "2022-05-25"],
            "2": ["2022-07-18", "2022-07-28"]
        }
    
    result = []
    for row in parsed_data:
        row_id = row['id']
        
        # Extract all statements (including non-informative ones)
        all_statements = extract_dated_statements(row['statements'], filter_no_events=False)
        
        # Check if ALL statements are non-informative (No event...)
        all_no_event = all(statement.get('is_no_event', False) for statement in all_statements)
        
        # If all statements are "No event...", skip this row entirely
        if all_no_event:
            print(f"Skipping row {row_id} as all statements are 'No event...'")
            continue
        
        # Filter to get only the informative statements for consolidation
        informative_statements = [
            statement for statement in all_statements 
            if not statement.get('is_no_event', False)
        ]
        
        # Get image dates for this row if available
        image_dates = image_dates_by_row.get(row_id, [])
        
        if not image_dates:
            # print(f"Warning: No image dates found for row {row_id}. Skipping consolidation.")
            continue

        # if more than 8 image dates, skip
        if len(image_dates) > 8:
            # print(f"Warning: More than 8 image dates found for row {row_id}. Skipping consolidation.")
            continue
        
        # Skip if no informative statements after filtering
        if not informative_statements:
            print(f"Skipping row {row_id} as no informative statements remain after filtering")
            continue
        
        # We pass both the informative statements and all statements to the consolidation function
        # This allows us to check if there are dates with only "no event" statements
        consolidated = consolidate_statements(informative_statements, image_dates)
        # print(f"Consolidated data for row {row_id}: {consolidated}")

        new_statements = '['
        # reformat to match date: consolidated statements
        for entry in consolidated:
            date = entry['image_date']
            statements = entry['consolidated_statements']
            new_statements+= f"{date}: {statements} "
        
        new_statements += "]"

        # print(f"New statements for row {row_id}: {new_statements}")
        
        # Create new row with consolidated statements
        result.append({
            'id': row_id,
            'url': row['url'],
            'coordinates': row['coordinates'],
            'locations': row['locations'],
            'consolidated_data': new_statements
        })
    
    return result

def save_reorganized_data(reorganized_data: List[Dict], output_file: str):
    """
    Save the reorganized data to a file.
    
    Args:
        reorganized_data: List of dictionaries with reorganized data
        output_file: Path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as csvfile:
        for row in reorganized_data:
            line_to_write = ''
            line_to_write += row['id'] + ','
            line_to_write += row['url'] + ','
            line_to_write += '(' + row['coordinates'] + '),'
            line_to_write += row['locations'] + ','
            line_to_write += row['consolidated_data'] + '\n'
            csvfile.write(line_to_write)

def print_example_results(reorganized_data: List[Dict]):
    """
    Print examples of the reorganized data for verification.
    
    Args:
        reorganized_data: The reorganized data to print examples from
    """
    print("\n=== EXAMPLE RESULTS ===")
    
    for row in reorganized_data:
        print(f"\nRow ID: {row['id']}")
        print(f"URL: {row['url']}")
        
        if not row.get('consolidated_data'):
            print("  No consolidated data available for this row.")
            continue
            
        print(f"Consolidated Data: {row['consolidated_data']}")

def main():
    """
    Main function to process the data.
    """
    # File paths
    csv_file = 'parsed_image_text.csv'  # Replace with your actual file
    image_folder = '.'  # Base folder containing all_events
    output_file = 'reorganized_total_data.csv'
    
    # For testing purposes, using the provided data
    sample_data = """0,"https://wildfiretoday.com/2022/07/20/firefighters-work-to-control-two-fires-in-north-texas-chalk-mountain-and-1148/,(32.7615226, -97.7980825),{"" 'FM51'"": (32.7615226, -97.7980825), "" 'Palo Pinto County'"": (32.7215726, -98.2814881), "" 'Rock Church Highway'"": (32.345025, -97.948555), "" 'Texas'"": (31.2638905, -98.5456116)},[2022-07-13: No events described in the article are visible from this date. 2022-07-18: The 1148 Fire near Possum Kingdom Lake started on Monday (July 17th), and by this date, 50 homes had been evacuated and at least two homes were visibly gutted. 2022-07-18: The Chalk Mountain Fire began on Monday (July 17th), and by this date a mandatory evacuation order for certain areas had been issued and later rescinded. 2022-07-23: No events described in the article are visible from this date. 2022-07-23: No events described in the article are visible from this date. 2022-07-28: No events described in the article are visible from this date. 2022-07-28: No events described in the article are visible from this date. 2022-08-02: No events described in the article are visible from this date. 2022-08-02: No events described in the article are visible from this date. 2022-08-07: No events described in the article are visible from this date. 2022-08-07: No events described in the article are visible from this date. 2022-08-12: No events described in the article are visible from this date. 2022-08-12: No events described in the article are visible from this date. 2022-08-17: No events described in the article are visible from this date. 2022-08-17: No events described in the article are visible from this date. 2022-08-22: No events described in the article are visible from this date. 2022-08-22: No events described in the article are visible from this date.  ]"
1,"https://www.1011now.com/2022/04/27/fire-crews-report-road-702-wildfire-74-contained/,(40.5300832055, -100.394202624),{'FEMA location 1': (40.5300832055, -100.394202624)},[2022-04-20: The Road 702 wildfire was 74% contained, with firefighters making progress despite high winds causing spot fires.  National Guard helicopters assisted in containing these spot fires.  2022-04-20:  Spot fires outside the main perimeter of the Road 702 wildfire were quickly contained due to good coordination between crews and local landowners.  2022-04-25: Firefighters continued working on containing the Road 702 wildfire, focusing on uncontained sections in Branch I (south of US-6, east of Bartley) and Branch III (south of Wilsonville and south of US-6, west of Cambridge).  2022-04-25:  Heavy equipment was used to remove hazardous trees and create fire lines around unburned vegetation in Branches I and III of the Road 702 wildfire.  2022-04-30: Elevated fire weather conditions, including warm temperatures and low humidity, were expected to continue, although winds were predicted to be lighter.  2022-04-30:  A slight chance of thunderstorms with lightning and gusty winds was predicted for the Road 702 wildfire area after 6 p.m.  2022-05-05:  Firefighters continued mopping up and patrolling the contained portions of the Road 702 wildfire to ensure no hot spots remained.  The portion of the wildfire in Kansas was contained.  2022-05-05:  Work continued on containing the uncontained fire edges in Branches I and III of the Road 702 wildfire.  2022-05-10:  No significant updates on the Road 702 wildfire are provided in the given text to associate with this date.  2022-05-10: No significant updates on the Road 702 wildfire are provided in the given text to associate with this date.  2022-05-15: No significant updates on the Road 702 wildfire are provided in the given text to associate with this date.  2022-05-15: No significant updates on the Road 702 wildfire are provided in the given text to associate with this date.  2022-05-20: No significant updates on the Road 702 wildfire are provided in the given text to associate with this date.  2022-05-20: No significant updates on the Road 702 wildfire are provided in the given text to associate with this date.  2022-05-25: No significant updates on the Road 702 wildfire are provided in the given text to associate with this date.  2022-05-25: No significant updates on the Road 702 wildfire are provided in the given text to associate with this date. ]"
2,"https://www.ktre.com/2022/07/21/chalk-mountain-fire-has-burned-more-than-6700-acres-is-only-10-percent-contained/,(31.2638905, -98.5456116),{' ""Texas""': (31.2638905, -98.5456116)},[2022-07-13:  The Chalk Mountain Fire began burning near Glen Rose, Texas,  eventually affecting over 6,700 acres.  2022-07-18: Governor Abbott reported on the fire's impact, including 16 homes destroyed, 5 damaged, 60 evacuated, and 40 threatened.  The Somervell County Expo Center opened to assist affected residents.  2022-07-23:  Road closures remained in effect on FM 205 and several county roads due to fire activity.  Firefighters focused on constructing fire lines on the west and east flanks.  2022-07-28:  The fire's progression continued, with spotting up to 200 yards in timbered areas and faster movement in lighter fuels.  A fire line was completed from the southern tip to FM 205.  2022-08-02:  Information on the Chalk Mountain Fire's status and containment after July 28th is not provided in the article.  2022-08-07: Information on the Chalk Mountain Fire's status and containment after July 28th is not provided in the article.  2022-08-12: Information on the Chalk Mountain Fire's status and containment after July 28th is not provided in the article.  2022-08-17: Information on the Chalk Mountain Fire's status and containment after July 28th is not provided in the article.  2022-08-22: Information on the Chalk Mountain Fire's status and containment after July 28th is not provided in the article. ]"
3,"https://www.knopnews2.com/2022/04/22/firefighter-injured-while-fighting-fire-near-cambridge/,(53.25542565000001, -9.745303893505355),{' ""Furnace County""': (53.25542565000001, -9.745303893505355), ' ""Road 702 Wildfire""': (37.6667969, -120.9580104), ' ""Southwest Elementary School""': (34.7329769, -77.5256269), ' ""Holbrook""': (34.9037105, -110.1593261)},[2022-04-18:  The Road 702 Wildfire began raging from Cambridge south to the Kansas border, prompting emergency response. 2022-04-18:  High winds fueled the wildfire in southwest Nebraska. 2022-04-20:  No specific event from the article is tied to this date. 2022-04-20: No specific event from the article is tied to this date. 2022-04-23: No specific event from the article is tied to this date. 2022-04-23: No specific event from the article is tied to this date. 2022-04-25: No specific event from the article is tied to this date. 2022-04-25: No specific event from the article is tied to this date. 2022-04-28: No specific event from the article is tied to this date. 2022-04-28: No specific event from the article is tied to this date. 2022-04-30: No specific event from the article is tied to this date. 2022-04-30: No specific event from the article is tied to this date. 2022-05-03: No specific event from the article is tied to this date. 2022-05-03: No specific event from the article is tied to this date. 2022-05-05: No specific event from the article is tied to this date. 2022-05-05: No specific event from the article is tied to this date. 2022-05-08: No specific event from the article is tied to this date. 2022-05-08: No specific event from the article is tied to this date. 2022-05-10: No specific event from the article is tied to this date. 2022-05-10: No specific event from the article is tied to this date. 2022-05-13: No specific event from the article is tied to this date. 2022-05-13: No specific event from the article is tied to this date. 2022-05-15: No specific event from the article is tied to this date. 2022-05-15: No specific event from the article is tied to this date. 2022-05-18: No specific event from the article is tied to this date. 2022-05-18: No specific event from the article is tied to this date. 2022-05-20: No specific event from the article is tied to this date. 2022-05-20: No specific event from the article is tied to this date. 2022-05-23: No specific event from the article is tied to this date. 2022-05-23: No specific event from the article is tied to this date. 2022-05-25: No specific event from the article is tied to this date. 2022-05-25: No specific event from the article is tied to this date.  ]"
4,"https://example.com/no-events-article/,(40.0, -100.0),{'Location': (40.0, -100.0)},[2022-05-01: No event from the provided article is associated with this date. 2022-05-02: No event from the provided article is associated with this date. 2022-05-03: No event from the provided article is associated with this date. 2022-05-04: No event from the provided article is associated with this date.]"
"""
    
    try:
        # Try to read CSV content from file
        with open(csv_file, 'r', encoding='utf-8') as f:
            csv_content = f.read()
            print(f"Successfully read data from {csv_file}")
    except FileNotFoundError:
        # Use sample data for testing
        print(f"File {csv_file} not found. Using sample data for demonstration.")
        csv_content = sample_data
    
    # Simulate folder structure for testing
    # In a real scenario, this would scan your actual folders
    print(f"Checking for image dates in folder: {image_folder}")
    image_dates_by_row = get_image_dates_by_row(image_folder)
    
    # If no folder structure found, use example data
    if not image_dates_by_row:
        print("Using example image dates for demonstration:")
        image_dates_by_row = {
            "0": ["2022-07-18", "2022-07-28"],
            "1": ["2022-04-20", "2022-04-25", "2022-04-30", "2022-05-10"],
            "2": ["2022-07-13", "2022-07-18", "2022-07-28"],
            "3": ["2022-04-18", "2022-04-30"],
            "4": ["2022-05-01", "2022-05-04"]  # Example for a row with all "No event" statements
        }
        
        for row_id, dates in image_dates_by_row.items():
            print(f"  Row {row_id}: {', '.join(dates)}")
    
    # Process the data
    reorganized_data = reorganize_data(csv_content, image_folder)
    
    # Save to file
    save_reorganized_data(reorganized_data, output_file)
    print(f"\nReorganized data saved to {output_file}")
    

if __name__ == "__main__":
    main()