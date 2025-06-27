import cv2
import numpy as np
import os
import pandas as pd

# Define categories and corresponding colors
categories = {
    "not_eaten": (0, 255, 0),      # Green
    "25% eaten": (255, 0, 0),     # Blue
    "50% eaten": (255, 100, 100), # Light Red
    "75% eaten": (255, 255, 0),   # Cyan
    "90% eaten": (0, 0, 255),   # Red
    "fully_eaten": (250, 150, 0)   # Orange
}

# Folder paths
folder_path = r'S:\04_data\2025\IfZ\Baitsticks\IFZ Pictures Baitstick 17_06'
strips_path = os.path.join(folder_path, 'Strips')
output_folder = os.path.join(folder_path, "circled_detected")
excel_output = os.path.join(folder_path, "bait_lamina_results_n.xlsx")

# Ensure output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize a DataFrame to store results
data = []

# Function to check if a circle is within bounds
def is_within_bounds(x, y, min_x, max_x, min_y, max_y, x_buffer, y_buffer):
    return (min_x - x_buffer <= x <= max_x + x_buffer) and (min_y - y_buffer <= y <= max_y + y_buffer)

# Process each folder inside the "Strips" directory
for strip_folder in os.listdir(strips_path):
    strip_folder_path = os.path.join(strips_path, strip_folder)

    if not os.path.isdir(strip_folder_path):
        continue  # Skip files, only process directories

    print(f"Processing folder: {strip_folder}")

    # Ensure output subfolder exists
    output_subfolder = os.path.join(output_folder, strip_folder)
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # Process each strip in the folder
    for file_name in os.listdir(strip_folder_path):
        if not file_name.endswith(".jpg"):
            continue

        image_path = os.path.join(strip_folder_path, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        width = np.shape(image)[1]
        height = np.shape(image)[0]
        print('width: ', np.shape(image)[1])
        print('height: ', np.shape(image)[0])
        # Determine classification (white or gray)
        tone = "White" if file_name.endswith('-w.jpg') else "Gray"

        # Extract original file name (remove '-g' or '-w')
        original_file_name = file_name.rsplit('-', 1)[0]
        
        mean_intensity = np.mean(gray)  # Calculate mean intensity
        print(tone, mean_intensity)
                
        # Set parameters based on tone
        if tone == "White":
            param1_value, param2_value = 35, 30
            binary_threshold, md = mean_intensity-40, 65
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        else:
            param1_value, param2_value = 35, 30
            binary_threshold, md = mean_intensity-30, 65
            blurred = cv2.GaussianBlur(gray, (9, 9), 2.5)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=2,
            minDist=md,
            param1=param1_value,
            param2=param2_value,
            minRadius=15,
            maxRadius=26
        )

        # Initialize category counts
        hole_counts = {category: 0 for category in categories}

        if circles is not None:
            circles = np.round(circles[0, :]).astype('int')
                
            stddev_strip = np.std(circles[:, 1])
            print (stddev_strip)
            if stddev_strip > 11:
                min_x = np.median(circles[:, 1]) - 35
                max_x = np.median(circles[:, 1]) + 35
            else:
                min_x = np.median(circles[:, 1]) - 25
                max_x = np.median(circles[:, 1]) + 20
    
            min_y = max(180, np.min(circles[:, 0])) 
            max_y = min(1900, np.max(circles[:, 0])) 
            filtered_circles = [circle for circle in circles 
                                if is_within_bounds(circle[1], circle[0],
                                                    min_x, max_x, min_y, max_y, 0, 0)]

            circles1 = np.array(filtered_circles)

            _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)

            for (x, y, r) in circles1:
                r_cropped = max(r-1 , 3)

                mask = np.zeros_like(binary, dtype=np.uint8)
                cv2.circle(mask, (x, y), r_cropped, 255, -1)

                circle_pixels = binary[mask == 255]
                whitened_pixels = np.sum(circle_pixels == 255)
                total_pixels = len(circle_pixels)

                whitened_percentage = (whitened_pixels / total_pixels) * 100 if total_pixels > 0 else 0

                # Determine category
                if whitened_percentage <= 5:
                    category = "not_eaten"
                elif 5 < whitened_percentage <= 25:
                    category = "25% eaten"
                elif 25 < whitened_percentage <= 45:
                    category = "50% eaten"
                elif 45 < whitened_percentage <= 70:
                    category = "75% eaten"
                elif 70 < whitened_percentage <= 90:
                    category = "90% eaten"
                else:
                    category = "fully_eaten"

                hole_counts[category] += 1
                color = categories[category]
                cv2.circle(image, (x, y), r, color, 2)

                category_labels = {
                    "not_eaten": "No feeding\n(0-5%)",
                    "25% eaten": "Low feeding\n(5-25%)",
                    "50% eaten": "Moderate feeding\n(25-50%)",
                    "75% eaten": "High feeding\n(50-75%)",
                    "90% eaten": "Very high feeding\n(75-95%)",
                    "fully_eaten": "Full feeding\n(95-100%)"
                }
                
                # Draw circle and label
                label = category_labels.get(category, category)
                for i, line in enumerate(label.split('\n')):
                    cv2.putText(image, line, (x - r, y - r - 10 - i*15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


            # Save annotated image
            output_image_path = os.path.join(output_subfolder, f"annotated_{file_name}")
            cv2.imwrite(output_image_path, image)

        total_detected_holes = sum(hole_counts.values())

        EXPECTED_HOLES = 16
        total_detected_holes = sum(hole_counts.values())
        
        # 2) adjust “not-eaten” when the detector MISSES or OVER-COUNTS
        if total_detected_holes < EXPECTED_HOLES:
            # detector missed some holes → assume the missing ones are “not eaten”
            missing = EXPECTED_HOLES - total_detected_holes
            adjusted_not_eaten = hole_counts["not_eaten"] + missing
        
        elif total_detected_holes > EXPECTED_HOLES:
            # detector found too many holes → treat extras as false-positives
            extras  = total_detected_holes - EXPECTED_HOLES
            adjusted_not_eaten = min(0, hole_counts["not_eaten"] - extras)   # never negative
        
        else:  # perfect 16
            adjusted_not_eaten = hole_counts["not_eaten"]
        

        adjusted_total_holes = EXPECTED_HOLES

        # Store results
        row = {
            "Strip Name": file_name,
            "Original File Name": original_file_name,
            "Tone": tone,
            "Detected Not Eaten": hole_counts["not_eaten"],
            "Adjusted Not Eaten": adjusted_not_eaten,
            "25% Eaten": hole_counts["25% eaten"],
            "50% Eaten": hole_counts["50% eaten"],
            "75% Eaten": hole_counts["75% eaten"],
            "90% Eaten": hole_counts["90% eaten"],
            "Fully Eaten": hole_counts["fully_eaten"],
            "Total Detected Holes": total_detected_holes,
            "Adjusted Total Holes": max(16, total_detected_holes)
        }
        data.append(row)

        print(f"Processed: {file_name} - Saved to {output_image_path}")

# Convert data to Pandas DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel(excel_output, index=False)

print(f"\nProcessing complete. Results saved to: {excel_output}")
