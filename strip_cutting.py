# -*- coding: utf-8 -*-
"""
Split baitstick photos into cropped images and horizontal strips.

Based on original script (Jun 19, 2025, roya), now with:
- cropping from top, bottom, left, right
- command-line arguments
- multi-format input
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

# Accepted image extensions (case-insensitive)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def classify_strip(image: Image.Image, threshold: float = 190.0) -> str:
    """
    Classify strip as white ('w') or gray ('g') based on mean intensity.
    """
    gray_image = image.convert("L")
    mean_intensity = float(np.mean(np.array(gray_image)))
    return "w" if mean_intensity > threshold else "g"


def find_images(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def crop_image_4sides(
    img: Image.Image,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
) -> Image.Image:
    """
    Crop from all four sides.

    crop_left/top/right/bottom are pixels to trim from each side.
    """
    width, height = img.size

    left = crop_left
    top = crop_top
    right = width - crop_right
    bottom = height - crop_bottom

    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid crop: ({left}, {top}, {right}, {bottom}) "
            f"for image size ({width}, {height})"
        )

    return img.crop((left, top, right, bottom))


def process_image(
    image_path: Path,
    crop_output_dir: Path,
    strip_output_base: Path,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
    total_strips: int,
) -> int:
    """
    Process a single image:
    - crop from 4 sides
    - save cropped version
    - cut into horizontal strips (using every second strip starting from 1)
    - classify strips and save as JPG

    Returns:
        number of strips written
    """
    img = Image.open(image_path)

    # 4-side crop
    img_cropped = crop_image_4sides(
        img,
        crop_left=crop_left,
        crop_top=crop_top,
        crop_right=crop_right,
        crop_bottom=crop_bottom,
    )

    # Save cropped image
    cropname = f"{image_path.stem}-cropped.jpg"
    crop_path = crop_output_dir / cropname
    img_cropped.save(crop_path)

    # Prepare strip folder per image
    image_strip_dir = strip_output_base / image_path.stem
    image_strip_dir.mkdir(parents=True, exist_ok=True)

    # Horizontal strip height
    strip_height = img_cropped.height // total_strips
    if strip_height <= 0:
        raise ValueError(
            f"Computed strip_height <= 0 for {image_path.name} "
            f"with total_strips={total_strips}"
        )

    strips_written = 0
    t = 0

    # Your original logic: use indices 1,3,5,... (every second strip starting from 1)
    for i in range(1, total_strips, 2):
        y1 = i * strip_height
        y2 = (i + 1) * strip_height

        strip = img_cropped.crop((0, y1, img_cropped.width, y2))
        classification = classify_strip(strip)

        # Always save strips as JPG for compatibility with the hole-classification script
        strip_filename = f"{image_path.stem}-s{t + 1}-{classification}.jpg"
        strip_path = image_strip_dir / strip_filename
        strip.save(strip_path)
        t += 1
        strips_written += 1

    return strips_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop baitstick photos from all sides and cut into horizontal strips."
    )
    parser.add_argument(
        "--in",
        "--input",
        dest="input_dir",
        required=True,
        help="Directory containing original baitstick photos.",
    )
    parser.add_argument(
        "--crop-out",
        dest="crop_output_dir",
        required=True,
        help="Directory to save cropped images.",
    )
    parser.add_argument(
        "--strip-out",
        dest="strip_output_dir",
        required=True,
        help="Directory to save strip subfolders.",
    )
    parser.add_argument(
        "--crop-left",
        type=int,
        default=255,
        help="Pixels to crop from the left (default: 255).",
    )
    parser.add_argument(
        "--crop-top",
        type=int,
        default=220,
        help="Pixels to crop from the top (default: 220).",
    )
    parser.add_argument(
        "--crop-right",
        type=int,
        default=1300,
        help="Pixels to crop from the right (default: 1300).",
    )
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=1510,
        help="Pixels to crop from the bottom (default: 1510).",
    )
    parser.add_argument(
        "--total-strips",
        type=int,
        default=17,
        help="Total number of horizontal strips (default: 17).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    crop_output_dir = Path(args.crop_output_dir).resolve()
    strip_output_dir = Path(args.strip_output_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    crop_output_dir.mkdir(parents=True, exist_ok=True)
    strip_output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(input_dir)
    if not images:
        print(f"No images with extensions {sorted(VALID_EXTS)} found in {input_dir}")
        return

    print(f"Found {len(images)} image(s) in {input_dir}")
    total_strips_written = 0

    for idx, image_path in enumerate(images, start=1):
        try:
            strips_written = process_image(
                image_path=image_path,
                crop_output_dir=crop_output_dir,
                strip_output_base=strip_output_dir,
                crop_left=args.crop_left,
                crop_top=args.crop_top,
                crop_right=args.crop_right,
                crop_bottom=args.crop_bottom,
                total_strips=args.total_strips,
            )
            total_strips_written += strips_written
            print(
                f"[{idx}/{len(images)}] Processed {image_path.name}: "
                f"{strips_written} strips"
            )
        except Exception as exc:
            print(f"[{idx}/{len(images)}] Error processing {image_path.name}: {exc}")

    print(
        f"Done. Wrote {total_strips_written} strips from {len(images)} input image(s).\n"
        f"Cropped images in: {crop_output_dir}\n"
        f"Strip folders in:  {strip_output_dir}"
    )


if __name__ == "__main__":
    main()
