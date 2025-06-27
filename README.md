# baitstick-analysis
Automated mesofauna feeding activity detection using image processing
This repository contains Python scripts to automate the classification of mesofauna feeding activity using bait lamina strips.

## Overview

1. `strip_cutting.py`: Splits full baitstick images into individual strip images and classifies them as `White` or `Gray`.
2. `holeclassification_Strips.py`: Detects feeding holes in the strips, classifies them by consumption level, and saves results to Excel and images.

## Folder Structure

Pictures Baitstick/
├── Cropped/                    # Input: cropped full baitstick images  
├── Strips/                     # Auto-created: split strips  
├── circled_detected/          # Output: images with feeding hole annotations  
└── bait_lamina_results.xlsx # Output: Excel summary of activity
## 🧪 Requirements

This project requires Python 3 and the following Python packages:
```
numpy
opencv-python
pandas
pillow
```
You can install them all at once by running:

```bash
pip install -r requirements.txt
```
## Setup Instructions
1. Download and install Python 3: https://www.python.org/downloads/

2. Download or clone this repository.

3. Open a terminal (Command Prompt on Windows).

4. Navigate to the project folder, for example:

```bash
cd path/to/baitstick-analysis
```
5. Install required packages:

```bash
pip install -r requirements.txt
```
6. Place your baitstick images inside the Pictures Baitstick/Cropped/ folder.

7. Run the scripts in order:

```bash
python strip_cutting.py
python holeclassification_Strips.py
```
8. Check the outputs in the Strips/, circled_detected/ folders and the Excel file bait_lamina_results.xlsx.


