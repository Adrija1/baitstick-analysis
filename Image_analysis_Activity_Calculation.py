"""
Detect and quantify feeding holes on bait strips (continuous scale).

Typical usage:

    python Image_analysis_Activity_Calculation.py \
        --in examples/Strips \
        --out examples/Annotated \
        --results examples/results.xlsx \
        --per-hole-csv examples/per_hole.csv
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

# Accepted image extensions (case-insensitive)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Expected holes per strip; used for QA / interpretation
EXPECTED_HOLES = 16


def is_within_bounds(
    x: int,
    y: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    x_buffer: int = 0,
    y_buffer: int = 0,
) -> bool:
    """Check whether (x, y) lies within a (possibly buffered) rectangle."""
    return (min_x - x_buffer <= x <= max_x + x_buffer) and (
        min_y - y_buffer <= y <= max_y + y_buffer
    )


def find_strip_folders(strips_path: Path) -> List[Path]:
    """Return all subfolders that contain strip images."""
    return sorted([p for p in strips_path.iterdir() if p.is_dir()])


def find_images_in_folder(folder: Path) -> List[Path]:
    """Return all valid image files in a folder."""
    files: List[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return files


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV or Excel, based on extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in {".xlsx", ".xls"}:
        try:
            df.to_excel(path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Writing Excel files requires 'openpyxl' or 'xlsxwriter' to be installed."
            ) from exc
    else:
        csv_path = path.with_suffix(".csv")
        print(
            f"Unknown extension '{ext}' for results file. "
            f"Saving as CSV to {csv_path} instead."
        )
        df.to_csv(csv_path, index=False)


def process_strip_image(
    image_path: Path,
    output_subfolder: Path,
) -> Tuple[dict, List[dict]]:
    """
    Process one strip image on a continuous scale.

    Returns:
        per_strip_row: dict with per-strip metrics
        per_hole_rows: list of dicts with per-hole details
    """
    per_hole_rows: List[dict] = []

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Warning: unreadable image {image_path}")
        return {
            "Strip Name": image_path.name,
            "Original File Name": image_path.stem,
            "Tone": "",
            "QA": "fail",
            "Reason": "unreadable_image",
            "DetectedHoles": 0,
            "MeanHolePct_Detected": 0.0,
            "SumHoleFrac": 0.0,
            "MeanStripFraction_16": 0.0,
        }, per_hole_rows

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Tone from filename suffix (-w / -g), case-insensitive
    fname_lower = image_path.name.lower()
    suffix = image_path.suffix.lower()
    if fname_lower.endswith("-w" + suffix):
        tone = "White"
    elif fname_lower.endswith("-g" + suffix):
        tone = "Gray"
    else:
        tone = "Gray"  # fallback

    original_file_name = image_path.name.rsplit("-", 1)[0]

    # Mean intensity over the whole strip (as in your working script)
    mean_intensity = float(np.mean(gray))
    print(f"  {image_path.name}: {tone} mean intensity = {mean_intensity:.2f}")

    # ===== detection parameters: copied from your working discrete script =====
    if tone == "White":
        param1_value, param2_value = 35, 30
        binary_threshold, md = mean_intensity - 40, 60
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    else:
        param1_value, param2_value = 35, 25
        binary_threshold, md = mean_intensity - 30, 60
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles (holes)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=md,
        param1=param1_value,
        param2=param2_value,
        minRadius=15,
        maxRadius=25,
    )

    # Binary mask for eaten pixels
    _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    hole_pcts: List[float] = []   # 0–100
    hole_fracs: List[float] = []  # 0–1

    qa = "ok"
    reason: str = ""

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"    Raw circles detected: {len(circles)}")

        # Filtering band – keep exactly your working behaviour
        stddev_strip = float(np.std(circles[:, 1]))
        if stddev_strip > 11:
            min_x = np.median(circles[:, 1]) - 35
            max_x = np.median(circles[:, 1]) + 35
        else:
            min_x = np.median(circles[:, 1]) - 25
            max_x = np.median(circles[:, 1]) + 20

        min_y = max(50, np.min(circles[:, 0]))
        max_y = min(1900, np.max(circles[:, 0]))

        filtered_circles = [
            c
            for c in circles
            if is_within_bounds(c[1], c[0], min_x, max_x, min_y, max_y, 0, 0)
        ]
        circles1 = np.array(filtered_circles)
        print(f"    Circles after band filter: {len(circles1)}")

        if len(circles1) < 14 or len(circles1) > 20:
            qa = "fail"
            reason = "circles_offgrid_or_wrong_count"

        for hid, (x, y, r) in enumerate(circles1, start=1):
            # --- Right-edge exclusion to prevent false detections ---
            RIGHT_EDGE_MARGIN = 120   # adjust if needed
            if x > (width - RIGHT_EDGE_MARGIN):
                continue
            LEFT_EDGE_MARGIN = 30
            if x < LEFT_EDGE_MARGIN:
                continue
            r_cropped = max(r, 7)
            mask = np.zeros_like(binary, dtype=np.uint8)
            cv2.circle(mask, (x, y), r_cropped, 255, -1)

            circle_pixels = binary[mask == 255]
            total_pixels = int(circle_pixels.size)
            if total_pixels == 0:
                eaten_pct = 0.0
                eaten_frac = 0.0
            else:
                whitened_pixels = int(np.sum(circle_pixels == 255))
                eaten_pct = 100.0 * (whitened_pixels / total_pixels)
                eaten_frac = float(np.clip(eaten_pct / 100.0, 0.0, 1.0))

            hole_pcts.append(eaten_pct)
            hole_fracs.append(eaten_frac)

            per_hole_rows.append(
                {
                    "Strip Name": image_path.name,
                    "Original File Name": original_file_name,
                    "HoleID": hid,
                    "EatenPct": round(eaten_pct, 2),
                    "EatenFrac": round(eaten_frac, 4),
                    "x": int(x),
                    "y": int(y),
                    "r": int(r),
                    "Tone": tone,
                    "QA": qa,
                }
            )

            # Draw circle for QC
            cv2.circle(image, (x, y), r+1, (0, 255, 0), 2)
    else:
        qa = "fail"
        reason = "no_circles_found"
        circles1 = []
        print("    No circles detected.")

    detected = len(circles1)
    sum_frac = float(np.sum(hole_fracs)) if hole_fracs else 0.0
    mean_pct_detected = float(np.mean(hole_pcts)) if hole_pcts else 0.0
    mean_strip_fraction_16 = sum_frac / float(EXPECTED_HOLES) if EXPECTED_HOLES > 0 else 0.0

    # Save annotated image
    output_subfolder.mkdir(parents=True, exist_ok=True)
    output_image_path = output_subfolder / f"annotated_{image_path.name}"
    cv2.imwrite(str(output_image_path), image)

    per_strip_row = {
        "Strip Name": image_path.name,
        "Original File Name": original_file_name,
        "Tone": tone,
        "QA": qa,
        "Reason": reason,
        "DetectedHoles": detected,
        "MeanHolePct_Detected": round(mean_pct_detected, 2),
        "SumHoleFrac": round(sum_frac, 4),
        "MeanStripFraction_16": round(mean_strip_fraction_16, 4),
    }

    return per_strip_row, per_hole_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect feeding holes on bait-stick strips and export continuous activity metrics."
    )
    parser.add_argument(
        "--in",
        "--input",
        dest="strips_path",
        required=True,
        help="Root directory containing strip subfolders.",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="output_root",
        required=True,
        help="Directory where annotated images will be written.",
    )
    parser.add_argument(
        "--results",
        dest="results_path",
        required=True,
        help="Path to per-strip results file (.csv or .xlsx).",
    )
    parser.add_argument(
        "--per-hole-csv",
        dest="per_hole_csv",
        default=None,
        help="Optional CSV/Excel file path for per-hole details (extension decides format).",
    )

    args = parser.parse_args()

    strips_path = Path(args.strips_path).resolve()
    output_root = Path(args.output_root).resolve()
    results_path = Path(args.results_path).resolve()
    per_hole_csv: Optional[Path] = (
        Path(args.per_hole_csv).resolve() if args.per_hole_csv else None
    )

    if not strips_path.exists() or not strips_path.is_dir():
        raise SystemExit(f"Strips directory does not exist or is not a directory: {strips_path}")

    strip_folders = find_strip_folders(strips_path)
    if not strip_folders:
        print(f"No strip subfolders found in {strips_path}")
        return

    print(f"Found {len(strip_folders)} strip folder(s) in {strips_path}")

    per_strip_rows: List[dict] = []
    all_per_hole_rows: List[dict] = []

    for idx, strip_folder in enumerate(strip_folders, start=1):
        print(f"[{idx}/{len(strip_folders)}] Processing folder: {strip_folder.name}")
        output_subfolder = output_root / strip_folder.name

        images = find_images_in_folder(strip_folder)
        if not images:
            print(f"  No images with extensions {sorted(VALID_EXTS)} in {strip_folder}")
            continue

        for jdx, image_path in enumerate(images, start=1):
            print(f"   - ({jdx}/{len(images)}) {image_path.name}")
            try:
                per_strip_row, per_hole_rows = process_strip_image(
                    image_path=image_path,
                    output_subfolder=output_subfolder,
                )
                per_strip_rows.append(per_strip_row)
                all_per_hole_rows.extend(per_hole_rows)
            except Exception as exc:
                print(f"     Error processing {image_path.name}: {exc}")

    if per_strip_rows:
        df_strip = pd.DataFrame(per_strip_rows)
        save_dataframe(df_strip, results_path)
        print(f"\nPer-strip results saved to: {results_path}")
    else:
        print("\nNo per-strip results to save.")

    if per_hole_csv and all_per_hole_rows:
        df_per_hole = pd.DataFrame(all_per_hole_rows)
        save_dataframe(df_per_hole, per_hole_csv)
        print(f"Per-hole details saved to: {per_hole_csv}")

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
