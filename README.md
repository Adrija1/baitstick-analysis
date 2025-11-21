# baitstick-analysis
Automated mesofauna feeding activity detection using image processing
This repository contains a two-stage Python workflow for analysing bait lamina strips and computing feeding activity using continuous activity fractions.  
It was prepared to address reviewer requirements regarding reproducibility, clarity, and test data availability.

---

## Repository Structure
```
baitstick-analysis/
├── DATA/ # Sample images for testing 
│ ├── example1.jpg
│ ├── example2.jpg
│ └── ...
│
├── REQUIREMENTS/
│ ├── requirements.txt # pip dependencies 
│ └── requirements.yml # conda environment file
│
├── strip_cutting.py # Stage 1: crop + split baitstick into strips
├── Image_analysis_Activity_Calculation.py # Stage 2: hole detection + activity
└── README.md # Documentation
```
---

##  Requirements

This project requires Python 3.8–3.12.  
Install dependencies using either pip or conda.

### Using pip
pip install -r REQUIREMENTS/requirements.txt

### Using conda
Linux/macOS:
conda env create -f REQUIREMENTS/requirements.yml
conda activate baitstick-analysis

Windows:
conda env create -f REQUIREMENTS\requirements.yml
conda activate baitstick-analysis

**Usage**
**Stage 1 — Split & Crop Strips**
```
python strip_cutting.py \
    --in examples/Images \
    --crop-out examples/Cropped \
    --strip-out examples/Strips
```
```
This script:

crops full bait lamina photographs

splits them into individual strips

classifies each strip as "w" (white) or "g" (gray)

Outputs are saved into Cropped and Strips folders.
```

**Stage 2 — Detect Holes & Compute Activity**
```
python Image_analysis_Activity_Calculation.py \
    --in examples/Strips \
    --out examples/Annotated \
    --results examples/results.xlsx \
    --per-hole-csv examples/per_hole.csv
This script:
  i. detects the 16 feeding holes per strip
  ii. computes continuous feeding fraction for each hole
  iii. computes total activity per strip
  iv. outputs annotated images with detection rings
  v. exports per-strip and per-hole tables
  vi. Outputs are saved into Annotated, results.xlsx, and per_hole.csv.
```
**Features**
```
Robust hole detection tuned for bait lamina dimensions
Continuous feeding fraction (0–1 scale)
Masks false detections on strip edges
Right-edge protection to avoid false circles

Supports JPG/PNG/TIFF images
Reproducible and reviewer-friendly structure
```
**Contact**
For questions or collaboration, please contact the repository author (adrijaroy1994@gmail.com).
