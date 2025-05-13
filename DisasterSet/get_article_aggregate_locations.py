# script to get search for something on the internet and return the first result

import requests
from bs4 import BeautifulSoup
import pandas as pd
import transformers
from tqdm import tqdm
import google.generativeai as genai
import os
import datetime
import calendar
from googleapi import google

from os.path import join, isfile, isdir
from os import mkdir
import numpy as np
import urllib.request
from multiprocessing.dummy import Pool
import csv
from dateutil.relativedelta import relativedelta
# import wget
import ee

ee.Initialize(project='ee-ssr234-new')
import pandas as pd
from tqdm import tqdm
from PIL import Image
import time
from time import sleep

# import multiprocessing as mp
# create black_list
black_list = ['google','wikipedia', 'youtube', 'twitter', 'facebook', 'instagram', 'linkedin', 'pinterest', 'reddit', 'quora', 'tiktok', 'tumblr']
white_list = ['.com', '.org']

genai.configure(api_key="Your_API_Key_Here")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-flash")

geocode_API_key = 'Your_API_Key_Here'  # Replace with your actual API key

def mask_s2_clouds(image):
  """Masks clouds in a Sentinel-2 image using the QA band.

  Args:
      image (ee.Image): A Sentinel-2 image.

  Returns:
      ee.Image: A cloud-masked Sentinel-2 image.
  """
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = (
      qa.bitwiseAnd(cloud_bit_mask)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
  )

  return image.updateMask(mask).divide(10000)


def summarize_text(text,startdate, enddate):
    
    try:
    
        summary = model.generate_content(f"Please, given these articles, could you pull out a list of proper noun locations in the article. responses should be string list format\n {text} \n if the article does not contain any information about the event, please return 'no'")
    except Exception as e:
        print(e)
        sleep(10)
        try:
            summary = model.generate_content(f"Please, given these articles, could you pull out a list of proper noun locations in the article. responses should be string list format\n {text} \n if the article does not contain any information about the event, please return 'no'")
        except Exception as e:
            return ''
    if summary.text == 'no':
        return ''

    return summary.text

def get_statements(text,dates):

    statements = model.generate_content(f"Please, given these articles and a list of dates, can you please assign a statement to each date describing the event in the article that could be visible from. The statement should be a sentence or two that describes the event that happened on or by that date. Format should be a list with no special characters or special formatting.\n {text} \n {dates}")
    if statements.text == 'no':
        return ''
    
    return statements.text

def get_images(center, starttime, endtime, incident_type, index):
    halfwidth=0.05
    odir='all_events'
    buffer_days = 5

    min_lon = center[1] - halfwidth
    max_lon = center[1] + halfwidth
    min_lat = center[0] - halfwidth
    max_lat = center[0] + halfwidth
 

    if not isdir(odir):
        mkdir(odir)

    outdir = join(odir, str(index))
    if not isdir(outdir):
        mkdir(outdir)

    # bounds is center +- halfwidth
    region = ee.Geometry.Rectangle([[min_lon, min_lat], [max_lon, max_lat]])
    
    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    col_cloud = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

    img = col.filterBounds(region)
    cloudmask = col_cloud.filterBounds(region)
   
    # starttime and endtime are strings
    start_date_str = starttime
    start_date_buffer_str = (datetime.datetime.strptime(starttime, '%Y-%m-%d') - relativedelta(days=buffer_days)).strftime('%Y-%m-%d')
    
    end_date_str = endtime
    # if endtime has 00:00:00, remove it
    if len(endtime) > 10:
        endtime = endtime[:10]
    end_date_buffer_str = (datetime.datetime.strptime(endtime, '%Y-%m-%d') + relativedelta(days=buffer_days)).strftime('%Y-%m-%d')
   
    img = img.filterDate(start_date_buffer_str, end_date_buffer_str)


    try:
        num_images = img.size().getInfo()
        print("num_images", num_images)
        
    except Exception as e:
        print("Error getting number of images:", e)
        return
    
   
    # if there are no images in the collection, return
    if num_images == 0:
        print("No images found for event in SR Harmonized", index)
        col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        img = col.filterBounds(region)
        img = img.filterDate(start_date_buffer_str, end_date_buffer_str)
        try:
            num_images = img.size().getInfo()
            print("num_images found harmonized", num_images)
        except Exception as e:
            print("Error getting number of images:", e)
            return
        if num_images == 0:
            print("No images found for event in Harmonized", index)
            col = ee.ImageCollection('COPERNICUS/S2')
            img = col.filterBounds(region)
            img = img.filterDate(start_date_buffer_str, end_date_buffer_str)
            try:
                num_images = img.size().getInfo()
                print("num_images found S2", num_images)
            except Exception as e:
                print("Error getting number of images:", e)
                return
            if num_images == 0:
                print("No images found for event in S2, all tried", index)
                return []

    img = img.toList(num_images)
    
    dates_list = []
    # iterate through the images and download them
    for i in tqdm(range(num_images)):
        image = img.get(i)
        image = ee.Image(image)

        img_date = image.date().format('YYYY-MM-dd').getInfo()
        dates_list.append(img_date)
        # create output file name
        output_file = join(outdir, f'{index}_{img_date}.jpg')
        if isfile(output_file):
            continue

        # Download the image
        try:
            url = image.getThumbURL({'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma':1, 'dimensions': '512x512', 'region': region})
            urllib.request.urlretrieve(url, output_file)
            # if image is more than 30% black, redo the download
            img_array = np.array(Image.open(output_file))
            if np.mean(img_array) < 30:
                # print("Image is too dark, redownloading...")
                urllib.request.urlretrieve(url, output_file)
            # if images is all white delete the image
            if np.mean(img_array) > 200:
                # print("Image is too white, deleting...")
                os.remove(output_file)
                continue
        except ee.ee_exception.EEException as e:
            print("error with image", e)
            continue
        
        # download the cloud mask image
        # find where QA60 band is and create a black and white image as cloud mask
        cloud_image = mask_s2_clouds(image)
        cloud_output_file = join(outdir, f'{index}_cloud_{img_date}.jpg')
        
        try:
            # get cloud mask image for same date
           
            url = cloud_image.getThumbURL({'min': 0, 'max': 100, 'gamma':1, 'dimensions': '512x512', 'region': region})
            urllib.request.urlretrieve(url, cloud_output_file)
            img_array = np.array(Image.open(cloud_output_file))
            # keep only the probability band
            img_array = img_array[:,:,0]
            # make any non black pixel white
            img_array[img_array != 0] = 255
            # save the image
            Image.fromarray(img_array).save(cloud_output_file)
        except ee.ee_exception.EEException as e:
            print("error with cloud image", e)
            continue
           
        
    print(f"Downloaded images for event {index} ({incident_type})")
    return dates_list


def get_bounding_box(list_of_locs):
    lats = []
    lons = []
    locations = {}
    for loc in list_of_locs:
        try:
            link = f'https://geocode.maps.co/search?q={loc}&api_key={geocode_API_key}'
            response = requests.get(link)
            if response.json():
                # print(response.json())
                lat = float(response.json()[0]['lat'])
                lon = float(response.json()[0]['lon'])
                lats.append(lat)
                lons.append(lon)
                locations[loc] = (lat, lon)
        except Exception as e:
            continue
    try:
        # get the bounding box
        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)
   
    
        return [[min_lon, min_lat], [max_lon, max_lat]], locations
    except Exception as e:
        return None, None

def get_article_content(url):
    print(f"Getting article content from {url}")
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract the title and content of the article
        title = soup.title.string if soup.title else 'No title found'
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return title, content
    else:
        return None, None

def get_image_center(list_of_locs):
    # find center for square of 0.1 degrees maximizing the number of locations within the square
    lats = []
    lons = []
    locations = {}
    for loc in list_of_locs:
        try:
            link = f'https://geocode.maps.co/search?q={loc}&api_key=6765c55c11f9b212367893hxkb616bc'
            response = requests.get(link)
            if response.json():
                # print(response.json())
                lat = float(response.json()[0]['lat'])
                lon = float(response.json()[0]['lon'])
                lats.append(lat)
                lons.append(lon)
                locations[loc] = (lat, lon)
        except Exception as e:
            continue
    try:
        
         # Find optimal square center
        if not lats or not lons:
            return None, None
            
        # Create grid of potential centers
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        
        best_count = 0
        best_center = (None, None)
        
        # Check each location as potential center
        for lat in lats:
            for lon in lons:
                count = sum(1 for loc_lat, loc_lon in zip(lats, lons)
                          if abs(loc_lat - lat) <= 0.05 and abs(loc_lon - lon) <= 0.05)
                
                if count > best_count:
                    best_count = count
                    best_center = (lat, lon)
                    
        return best_center, locations
        
    except Exception as e:
        return None, None


def main():
    # read the csv file
    csv = open("articles.csv", "r").readlines()
    f = open('parsed_image_text.csv', 'a+')  # Open the CSV file in append mode
    df = pd.read_csv('FEMA_filtered_processed.csv', header=0)
   

    # create a dict of all the events in articles, with the index as the key and the links as the values list
    events = {}
    for line in csv:
        index = int(line[:line.find(",")])
        if index not in events:
            events[index] = []
        events[index].append(line.split(",")[1])

    for event_index, links in events.items():
        # get the dates from df using the index
        try:
            start_date = df.loc[df['index'] == event_index, 'incidentBeginDate'].values[0]
        except Exception as e:
            print(f"Error getting start date for index {event_index}: {e}")
            continue
        # get date with day of week like "2021-01-01, Friday"
        str_start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d, %A')
        end_date = df.loc[df['index'] == event_index, 'incidentEndDate'].values[0]
        # if date has 00:00:00, remove it
        if len(end_date) > 10:
            end_date = end_date[:10]
        str_end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d, %A')

        # get the article content for all links
        content = ''
        for link in links:
            title, article_content = get_article_content(link)
            if article_content:
                content += article_content + '\n'
        
        if not content:
            continue

        print("summarize_text")
        try:
            list_of_locs = summarize_text(content, str_start_date, str_end_date)
        except requests.exceptions.RequestException as e:
            print(f"Error summarizing content: {e}")
             # if error due to too many requests, wait for 10 seconds and try again
            print("waiting for 10 seconds...")
            time.sleep(10)
            try:
                print("Trying again...")
                list_of_locs = summarize_text(content, str_start_date, str_end_date)
            except requests.exceptions.RequestException as e:
                print(f"Error summarizing content: {e} skipping...")
                continue

        if list_of_locs == '':
            continue
        
        print("get_bounding_box")

        list_of_locs = list_of_locs[list_of_locs.find("[")+1:list_of_locs.find("]")]
        list_of_locs = list_of_locs.split(',')
        # remove duplicates
        list_of_locs = list(set(list_of_locs))

        if len(list_of_locs) == 0:
            print("No locations found, skipping...")
            continue
        center, locations = get_image_center(list_of_locs)

        if len(list_of_locs) == 0 or center is None:
            print("No center found, using fema location")
            fema_lat = df.loc[df['index'] == event_index, 'lat'].values[0]
            fema_lon = df.loc[df['index'] == event_index, 'lon'].values[0]
            center = (fema_lat, fema_lon)
            locations = {f"FEMA location {event_index}": (fema_lat, fema_lon)}
            print("FEMA location", center)

        print("get_images")

        # get images from google earth engine for the bounding box from start_date to end_date
        try:
            dates = get_images(center, start_date, end_date, df.loc[df['index'] == event_index, 'incidentType'].values[0], event_index)
        except Exception as e:
            print(f"Error getting images for index {event_index}: {e}")
            continue
        if not dates:
            continue

        print("get_statements")
        # get text from article corresponding to the image dates
        try:
            statements = get_statements(content, dates)
        except requests.exceptions.RequestException as e:
            # wait for 10 seconds and try again
            print(f"Error getting statements: {e}")
            time.sleep(20)
            try:
                statements = get_statements(content, dates)
            except requests.exceptions.RequestException as e:
                print(f"Error getting statements: {e}")
                continue
        if not statements:
            continue
        # remove newlines from statements
        statements = statements.replace('\n', ' ')
        f.write(f"{event_index},{links},{center},{locations},[{statements}]\n")
        f.flush()
    f.close()
    print("Done")

    

if __name__ == "__main__":
    main()  # Call the main function to execute the script