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
