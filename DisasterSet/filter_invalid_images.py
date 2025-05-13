# filter images which are not valid i.e. mostly white or black
# remove empty folders and files


import os
import PIL
from PIL import Image
import numpy as np
import cv2

def is_valid_image(img_path):
    try:
        img = Image.open(img_path)
        # get img as numpy array
        img = np.array(img)
        # print(img_path,img.shape)
        # if 75% of the image is white or black, then it is invalid
        if np.mean(img) < 25 or np.mean(img) > 240:
            return False
        # if the colors are only black or white, then it is invalid
        if np.unique(img).shape[0] < 3:
            return False
        # if number of pixels with value 0 is 5% of the total pixels, then it is invalid
        if np.count_nonzero(img == 0) > (0.05 * (512*512)):
            return False
        return True
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return False

def filter_images(dir_path):
    for root, _, files in os.walk(dir_path):
        if len(files) == 1:
            print(f"Removing empty folder: {root}")
            # remove file
            os.remove(os.path.join(root, files[0]))
            os.rmdir(root)
            continue
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                # if file is cloud mask, skip
                if 'cloud' in file:
                    continue
                img_path = os.path.join(root, file)
                # if path contains 'before' or 'after', then it should be deleted
                if 'before' in img_path or 'after' in img_path or not is_valid_image(img_path):
                    print(f"Removing invalid image: {img_path}")
                    os.remove(img_path)
        if not os.listdir(root):
            print(f"Removing empty folder: {root}")
            os.rmdir(root)


if __name__ == '__main__':
    # removes invalid images
    filter_images('all_events')
    
   