# baitstick-analysis
Automated mesofauna feeding activity detection using image processing
This repository contains a two-stage Python workflow for analysing bait lamina strips and computing feeding activity using continuous activity fractions.  
It was prepared to address reviewer requirements regarding reproducibility, clarity, and test data availability.

---

## Repository Structure
```
baitstick-analysis/
├── DATA/ # Sample images for testing 
│ ├── Images
│   ├── 20250826_155709.jpg
│   ├── 20250826_161616.jpg
│ ├── Cropped
│   ├── 20250826_155709-cropped.jpg
│   ├── 20250826_161616-cropped.jpg
│ ├── Strips
│     ├── 20250826_155709/
│     │     ├── 20250826_155709-s1-g.jpg
│     │     ├── 20250826_155709-s2-g.jpg
│     │     └── ...
│     │
│     ├── 20250826_161616/
│           ├── 20250826_161616-s1-g.jpg
│           ├── 20250826_161616-s2-w.jpg
│           └── ...
│
│ ├── Annotated
│     ├── 20250826_155709/
│      │     ├── annotated_20250826_155709-s1-g.jpg
│      │     ├── annotated_20250826_155709-s2-g.jpg
│      │     └── ...
│      │
│      ├── 20250826_161616/
│            ├── annotated_20250826_161616-s1-w.jpg
│            ├── annotated_20250826_161616-s2-g.jpg
│            └── ...
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
```
conda env create -f REQUIREMENTS/requirements.yml  
conda activate baitstick-analysis  
```
Windows:
```
conda env create -f REQUIREMENTS\requirements.yml
conda activate baitstick-analysis
```
**Usage**
**Stage 1 — Split & Crop Strips**  
The user needs to print the Template provided (Template_Baitsticks.docx) in an A4 sheet, and arrange the baitsticks sequentially on the A4 sheet of template, placed on a Light Table.
The camera is to be placed to capture the red-line bounded area from 21 cm above surface.
```
python strip_cutting.py \
    --in examples/Images \
    --crop-out examples/Cropped \
    --strip-out examples/Strips
```

This script:  
    i. crops full bait lamina photographs  
    ii. splits them into individual strips  
    iii. classifies each strip as "w" (white) or "g" (gray)  
    iv. Outputs are saved into Cropped and Strips folders.  

**Stage 2 — Detect Holes & Compute Activity**
```
python Image_analysis_Activity_Calculation.py \
    --in examples/Strips \
    --out examples/Annotated \
    --results examples/results.xlsx \
    --per-hole-csv examples/per_hole.csv
```

This script:  
- detects 16 feeding holes per strip  
- computes **continuous feeding fraction** per hole  
- calculates total activity per strip  
- draws detection rings on output images  
- exports **per-strip** and **per-hole** tables  
- saves results in **Annotated/**, `results.xlsx`, and `per_hole.csv`

---

## Features

- Accurate circle detection tuned for bait-lamina geometry  
- Continuous feeding fraction (0–1 scale)  
- Right-edge masking removes false detections  
- Supports JPG / PNG / TIFF  
- Fully reproducible structure for peer review  

---

## Contact  
For issues or collaboration, email: **adrijaroy1994@gmail.com**

---
