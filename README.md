# Feature Extraction from Ladybug Camera

A methodology for processing images from a **Mobile Mapping System (MMS)** equipped with a **Teledyne FLIR Ladybug 5+** multi-camera system. Two independent pipelines are provided:

1. **Feature Extraction** — detect urban features (traffic signs, crosswalks, safety cones, etc.) and compute their **3D ground coordinates in the Greek EGSA87 coordinate system (EPSG:2100)** via photogrammetric forward intersection.
2. **Image Blurring** — apply Gaussian blur to detected objects for privacy masking / anonymisation.

Both pipelines are powered by the [Detectron2](https://github.com/facebookresearch/detectron2) computer vision framework and support all model types (OD, IS, P, SS).

Output 1 | Output 2
---|---
![output1](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/8d0d418e-ae61-4b1e-bdcf-d14cfb379736) | ![output2](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/c644c7d6-a202-49ce-b058-ef8507033075)

Centroid extraction |
:---: |
![centroid](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/4237f4b4-6ae3-4bc9-8961-5c7f0627c357) |

---

## Overview

Two input modes are supported depending on which image product is available from the Ladybug:

| | Mode A — Panoramic | Mode B — Raw |
|---|---|---|
| **Input** | 360° equirectangular panoramas | Individual raw camera images (Cam0–Cam5) |
| **Rotation** | None | 90° CW for detection |
| **Coord. transform** | None (already panoramic) | DistortedSpline (`.cal`) → panoramic |

Both modes feed into the same forward intersection step to produce EGSA87 coordinates, and both support the blur pipeline as an alternative output.

---

## Pipeline 1 — Feature Extraction

### Mode A — Panoramic images

```
Panoramic .jpg
      │
      ▼
 run_detection()               ← Detectron.py
      │
 ┌────┴─────┐
P / SS      OD
 │           │
masks    bbox centres
 │           │
Centroid()  write CSV
      │
image_coords.csv               ← output/coords/
      │
      ▼
run_intersection()             ← forward_intersection.py
(+ EOP CSV from GPS/IMU)
      │
      ▼
image_coords_EGSA87.csv        ← output/egsa87/
```

### Mode B — Raw images

```
Raw .jpg (Cam0–5)
      │
      ▼  rotate 90° CW
 run_detection()               ← Detectron.py
      │
 ┌────┴─────┐
P / SS      OD
 │           │
masks    bbox centres
(unrotate)  (unrotate)
      │
raw_coords.csv                 ← output/coords/
      │
      ▼
RawLadybugTransformer          ← raw_to_panorama.py
(DistortedSpline from .cal)
      │
image_coords.csv               ← output/coords/
      │
      ▼
run_intersection()             ← forward_intersection.py
(+ EOP CSV from GPS/IMU)
      │
      ▼
image_coords_EGSA87.csv        ← output/egsa87/
```

---

## Pipeline 2 — Image Blurring

Blurs detected regions in-place using Gaussian blur. Works for **all model types**:

| Model type | Blurred region |
|---|---|
| OD | Bounding box of each detection |
| IS | Instance segmentation mask |
| P | Panoptic segment mask |
| SS | All non-background pixels |

```
Images (Mode A or B)
      │
      ▼  Mode B: rotate 90° CW for detection
 run_detection(..., output_mode='blur')
      │
      ▼
Gaussian blur applied to detected regions
      │
      ▼
Blurred image saved in detection space
(Mode B: portrait orientation, no back-rotation)
      │
      ▼
output/blurred/
```

---

## Ladybug 5+

![ladybug5plus](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/83de8cfa-3f98-4303-8c37-20e5e7db9a97)

The [Ladybug 5+](https://www.flir.com/products/ladybug5plus/?vertical=machine+vision&segment=iis) is a Teledyne FLIR multi-camera system (5 side cameras + 1 top camera, 30 MP total) designed for mobile mapping. Camera calibration is provided as a `.cal` file containing per-lens focal lengths, principal points, and 2D B-spline warp tables for lens distortion correction.

---

## Repository Structure

```
code/
  scripts/                       ← scripts the user runs
    pipeline_mode_a.py           ← feature extraction, Mode A (panoramic)
    pipeline_mode_b.py           ← feature extraction, Mode B (raw)
    pipeline_blur.py             ← image blurring, Mode A or B

  lib/                           ← modules (imported by scripts)
    Detectron.py                 ← Detector class + run_detection()
    Centroid.py                  ← centroid extraction from segmentation masks
    raw_to_panorama.py           ← raw pixel → panoramic coordinate transformer
                                    + image stitching (raw → panorama)
    forward_intersection.py      ← run_intersection(): N-ray triangulation,
                                    GET EOP format (lat/lon → EGSA87 via pyproj)

data/
  Ladybug5_plus/
    ladybug20344317.cal          ← Ladybug camera calibration file

output/                          ← all outputs land here (gitignored)
  masks/                         ← binary mask images (P / SS, feature extraction)
  coords/                        ← image_coords.csv, raw_coords.csv (Mode B)
  egsa87/                        ← <name>_EGSA87.csv (feature extraction)
  blurred/                       ← blurred images (blur pipeline)
  panorama/                      ← panorama_from_raw.jpg (stitching)

projects/                        ← pre-trained model weights (see projects.md)
  Cityscapes/panoptic/
  Crosswalk/output/
  Traffic_Sign/output/
  MaskFormer/panoptic/
```

---

## Detection Models

| Model | Type | Classes |
|---|---|---|
| `COCO` | OD / IS / P | person, car, traffic light, stop sign, … |
| `Cityscapes` | SS / P | road, pole, traffic light, traffic sign, … |
| `Crosswalk` | OD | Crosswalk |
| `Traffic_Sign` | OD | 24 sign classes (Stop, Speed limits, Give way, …) |
| `Safety_Cones` | OD | Safety Cone |

Model types: **OD** = object detection, **IS** = instance segmentation, **P** = panoptic segmentation, **SS** = semantic segmentation.

Pre-trained weights and training notebooks are linked in [projects.md](projects.md).

---

## Required Inputs — Feature Extraction

| Input | Description |
|---|---|
| Image folder | Panoramic `.jpg` (Mode A) or raw Ladybug `.jpg` (Mode B) |
| `.cal` file | Ladybug calibration file (Mode B only) |
| EOP CSV | Tab-separated: `gps_seconds[s]`, `panorama_file_name`, `latitude[deg]`, `longitude[deg]`, `altitude_ellipsoidal[m]`, `roll[deg]`, `pitch[deg]`, `heading[deg]` |

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/GElpida/feature-extraction-from-ladybug-camera.git
cd feature-extraction-from-ladybug-camera
```

**2. Create the base conda environment** (Python 3.8 + PyTorch + NumPy)
```bash
conda create --name detectron2_env --file detectron2_env.txt
conda activate detectron2_env
```

**3. Install Detectron2** into the activated environment

- **Windows** — follow the guide:
  https://haroonshakeel.medium.com/detectron2-setup-on-windows-10-and-linux-407e5382df1
- **Linux / macOS**:
  ```bash
  pip install 'git+https://github.com/facebookresearch/detectron2.git'
  ```

**4. Install additional pipeline dependencies**
```bash
pip install -r requirements.txt
```
Adds `opencv-python`, `matplotlib`, and `pyproj` on top of the Detectron2 environment.

**5. Download model weights**

See [projects.md](projects.md) for download links. Place them under `projects/`.

---

## Usage

### Feature Extraction (Pipeline 1)

Edit the configuration block at the top of the relevant script and run:

```bash
# Mode A — panoramic images
python code/scripts/pipeline_mode_a.py

# Mode B — raw Ladybug images
python code/scripts/pipeline_mode_b.py
```

Configuration variables:

| Variable | Description |
|---|---|
| `IMAGE_FOLDER` | Absolute path to input images |
| `CAL_FILE` | Path to `.cal` calibration file (Mode B only) |
| `MODELS` | List of Detectron2 models to run |
| `EOP_CSV` | Path to EOP CSV (leave empty to skip triangulation) |

Outputs:
- `output/coords/image_coords.csv` — panoramic pixel coordinates
- `output/masks/` — binary mask images (P / SS models)
- `output/egsa87/<name>_EGSA87.csv` — 3D ground coordinates (if EOP provided)

Output CSV format:

| point_name | X_egsa87 | Y_egsa87 | Z_egsa87 |
|---|---|---|---|

> **Note:** Forward intersection requires ≥ 2 panorama captures where the same object is visible.

---

### Image Blurring (Pipeline 2)

Edit the configuration block at the top and run:

```bash
python code/scripts/pipeline_blur.py
```

Configuration variables:

| Variable | Description |
|---|---|
| `IMAGE_FOLDER` | Absolute path to input images |
| `MODE` | `'A'` (panoramic) or `'B'` (raw Ladybug) |
| `CAL_FILE` | Path to `.cal` calibration file (Mode B only) |
| `MODELS` | List of Detectron2 models to run |

Output: `output/blurred/` — one blurred `.jpg` per input image.
In Mode B the blurred images are saved in detection (portrait) space.

---

### Panorama stitching from raw images (optional)

```bash
python code/lib/raw_to_panorama.py
```

Stitches all raw camera images in a folder into a single equirectangular panorama using the Ladybug `.cal` calibration file. No dependencies beyond NumPy and OpenCV.
Output: `output/panorama/panorama_from_raw.jpg`
