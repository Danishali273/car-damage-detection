# Car Damage Detection

A two-stage computer vision pipeline for detecting car parts and assessing specific types of damage on each part. Built with Python, OpenCV, and Ultralytics YOLO.

## Overview

This project uses two YOLO models to perform sequential detection:
1. **Stage 1 (Part Detection):** Identifies specific car parts in total 21 classes (e.g., Hood, Bumper, Windshield, Wheels, etc...) using `best_car_part.pt`.
2. **Stage 2 (Damage Detection):** Analyzes the cropped region of each detected part to find specific damages in total 7 classes (e.g., Dent, Scratch, Smash, Glass Break, etc...) using `best_damage_type.pt`.

The system intelligently maps specific damages to relevant parts (e.g., a "flat tire" can only occur on a wheel, and a "glass break" can only occur on windows). 

## Features

- **Two-Stage Inference:** Reduces false positives by analyzing damage only within the context of specific car parts.
- **Per-Class Thresholds:** Confidence thresholds can be tuned independently for every single part and damage type inside the script.
- **Logical Damage Mapping:** Restricts impossible detections (like a "dented" windshield or "glass break" on a tire).
- **Rich Visualization:** Generates output images with bounding boxes, instance segmentation polygons (if available), confidence bars, and a clear "Pass/Fail" indicator for thresholds.
- **Summary Report:** Displays a combined dashboard image showing the detected parts, the overlaid damage, and a tabular textual summary report.

## Prerequisites

- Python 3.x
- `ultralytics`
- `opencv-python`
- `numpy`

You can install the required dependencies via pip:
```bash
pip install ultralytics opencv-python numpy
```

## Usage

Run the script from the command line by providing an input image:

```bash
python test_images.py path/to/image.jpg
```

### Advanced CLI Options

```bash
python test_images.py car8.jpeg --output out.jpg --parts-conf 0.10 --damage-conf 0.10
```

- `image` (positional): Path to the input image.
- `--output`: Specifies the base name for the output images. The script will append `_parts` and `_damage` to the outputs automatically.
- `--parts-conf`: Global minimum confidence floor for the parts model before per-class filtering is applied.
- `--damage-conf`: Global minimum confidence floor for the damage model before per-class filtering is applied.

*Note: True filtering is done by the `PART_THRESHOLDS` and `DAMAGE_THRESHOLDS` dictionaries inside `test_images.py`. It is recommended to keep the CLI global confidences relatively low so the script's internal per-class thresholds can act as the primary filter.*

## Folder Structure

- `test_images.py`: The main inference pipeline script for images.
- `test_video.py`: The main inference pipeline script for videos.
- `models/best_car_part.pt`: YOLO model weights for car part detection.
- `models/best_damage_type.pt`: YOLO model weights for damage classification.

## Configuration / Tuning

If you are getting too many false positives or missing detections, edit the `PART_THRESHOLDS` and `DAMAGE_THRESHOLDS` dictionaries directly at the top of `test_images.py` and `test_video.py`.

```python
DAMAGE_THRESHOLDS = {
    'dent':         0.50,
    'glass_break':  0.80,
    'scratch':      0.50,
    # ...
}
```
