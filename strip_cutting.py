# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 08:20:13 2025

@author: roya
"""

import os
from PIL import Image
import numpy as np

# Function to classify the strip based on the mean pixel intensity
def classify_strip(image):
    gray_image = image.convert("L")  # Convert to grayscale
    mean_intensity = np.mean(np.array(gray_image))  # Calculate mean intensity
    threshold = 190  # Classification threshold
    return "w" if mean_intensity > threshold else "g"

# Directory containing images
image_dir = r'S:\04_data\2025\IfZ\Baitsticks\IFZ Pictures Baitstick 17_06'
crop_output_dir = os.path.join(image_dir, 'Cropped')
strip_output_base = os.path.join(image_dir, 'Strips')

# Ensure output directories exist
os.makedirs(crop_output_dir, exist_ok=True)
os.makedirs(strip_output_base, exist_ok=True)

# Total number of vertical strips
total_strips = 17

# Process all JPG images in the directory
for image_b_name in os.listdir(crop_output_dir):
    if image_b_name.lower().endswith('.jpg'):
        try:
            image_path = os.path.join(crop_output_dir, image_b_name)
            img = Image.open(image_path)

            # Crop the image
            # img = img.crop((920, 1300, img.width - 800, img.height))
            # cropname = f"{image_b_name}-cropped.jpg"
            # crop_path = os.path.join(crop_output_dir, cropname)
            # img.save(crop_path)

            # Prepare strip output folder per image
            image_strip_dir = os.path.join(strip_output_base, image_b_name.split('.')[0])
            os.makedirs(image_strip_dir, exist_ok=True)

            strip_width = img.height // total_strips

            for i in range(1,total_strips,2):  # Loop over vertical strips
                strip = img.crop((0,(i) * strip_width, img.width, (i+1) * strip_width))
                classification = classify_strip(strip)
                strip_filename = f"{image_b_name}-s{i + 1}-{classification}.jpg"
                strip_path = os.path.join(image_strip_dir, strip_filename)
                strip.save(strip_path)

            print(f"Processed: {image_b_name}")
        except Exception as e:
            print(f"Error processing {image_b_name}: {e}")

print("All images have been processed and strips classified.")
