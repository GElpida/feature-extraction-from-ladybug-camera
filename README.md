# Feature Extraction from Ladybug Camera

A methodology for extracting urban features (traffic signs, crosswalks, safety cones, road markings, etc.) from a **Mobile Mapping System (MMS)** equipped with a **Teledyne FLIR Ladybug 5+** multi-camera system, and computing their **3D ground coordinates in the Greek EGSA87 coordinate system (EPSG:2100)**.

Detection is powered by the [Detectron2](https://github.com/facebookresearch/detectron2) computer vision framework.

Output 1 | Output 2
---|---
![output1](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/8d0d418e-ae61-4b1e-bdcf-d14cfb379736) | ![output2](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/c644c7d6-a202-49ce-b058-ef8507033075)

Centroid extraction |
:---: |
![centroid](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/4237f4b4-6ae3-4bc9-8961-5c7f0627c357) |

---

## Overview

The pipeline takes images from the Ladybug 5+, runs deep-learning-based detection, extracts the image coordinates of detected objects, and finally triangulates their 3D position using photogrammetric forward intersection with known camera Exterior Orientation Parameters (EOP).

Two input modes are supported depending on which image product is available:

| | Mode A — Panoramic | Mode B — Raw |
|---|---|---|
| **Input** | 360° equirectangular panoramas | Individual raw camera images (Cam0–Cam5) |
| **Coord. transform** | None (already panoramic) | DistortedSpline (`.cal`) → panoramic |
| **Output** | `image_coords.csv` → EGSA87 CSV | same |

---

## Pipeline

### Mode A — Panoramic images

```
Panoramic .jpg  ──►  use_Detectron.py
                          │
                 ┌────────┴─────────┐
              P / SS              OD
            (segmentation)  (detection)
                 │                 │
           masks saved       bbox centres
                 │            written to
           Centroid.py       image_coords.csv
                 │
         image_coords.csv
                 │
                 ▼
       forward_intersection.py
       (+ EOP CSV from GPS/IMU)
                 │
                 ▼
         output_EGSA87.csv
```

### Mode B — Raw images

```
Raw .jpg (Cam0–5)  ──►  use_Detectron.py  (rotate 90° CW)
                               │
                      ┌────────┴─────────┐
                   P / SS              OD
                      │                 │
                 masks saved       bbox centres
                 (rotate CCW)     (unrotated)
                      │                 │
                Centroid.py       raw_coords.csv
                      │
                raw_coords.csv
                      │
                      ▼
              raw_to_panorama.py
          (DistortedSpline from .cal)
                      │
               image_coords.csv
                      │
                      ▼
          forward_intersection.py
          (+ EOP CSV from GPS/IMU)
                      │
                      ▼
            output_EGSA87.csv
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
    pipeline_mode_a.py           ← end-to-end Mode A (configure & run)
    pipeline_mode_b.py           ← end-to-end Mode B (configure & run)

  lib/                           ← modules (imported by scripts)
    Detectron.py                 ← Detector class + run_detection()
    Centroid.py                  ← centroid extraction from segmentation masks
    raw_to_panorama.py           ← raw pixel → panoramic coordinate transformer
                                    + image stitching (raw → panorama)
    forward_intersection.py      ← run_intersection(): N-ray triangulation,
                                    GET EOP format (lat/lon → EGSA87 via pyproj)

data/
  Ladybug5_plus/
    ladybug20344317.cal     ← Ladybug camera calibration file

output/                     ← all outputs land here (gitignored)
  masks/                    ← Step 1: binary mask images (P / SS detection)
  coords/                   ← Step 2: image_coords.csv, raw_coords.csv (Mode B)
  egsa87/                   ← Step 3: <name>_EGSA87.csv
  panorama/                 ← panorama_from_raw.jpg (stitching only)

projects/                   ← pre-trained model weights (see projects.md)
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

## Required Inputs

| Input | Description |
|---|---|
| Image folder | Panoramic `.jpg` (Mode A) or raw Ladybug `.jpg` (Mode B) |
| `.cal` file | Ladybug calibration file (Mode B only) |
| EOP CSV | Exterior Orientation Parameters: `panorama_file_name`, `x[m]` or `latitude[deg]`, `y[m]` or `longitude[deg]`, `altitude_ellipsoidal[m]`, `roll[deg]`, `pitch[deg]`, `heading[deg]` |
| Points CSV | Image observations: `point_name`, `image_name`, `x[px]`, `y[px]` (output of step 1–2) |

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
This adds `opencv-python` and `matplotlib` — the only packages the pipeline needs that are not already provided by the Detectron2 environment.

**5. Download model weights**

See [projects.md](projects.md) for download links. Place them under `projects/`.

---

## Usage

### Step 1 — Detection + image coordinates

Edit the configuration block at the top of the relevant pipeline script, then run it:

```bash
# Mode A — panoramic images
python code/scripts/pipeline_mode_a.py

# Mode B — raw Ladybug images
python code/scripts/pipeline_mode_b.py
```

Set in the script:
- `IMAGE_FOLDER` — absolute path to input images
- `CAL_FILE` — `.cal` calibration file (Mode B only)
- `MODELS` — which Detectron2 model(s) to run
- `EOP_CSV` — path to GET EOP file (leave empty to skip triangulation)

Outputs `output/coords/image_coords.csv`. Segmentation masks go to `output/masks/`.

### Step 2 — Forward Intersection

```bash
python forward_intersection.py
```

Prompts (defaults shown, press Enter to accept):
- Path to `image_coords.csv` → default `output/coords/image_coords.csv`
- Path to EOP CSV

EOP format (tab-separated):
```
gps_seconds[s]  panorama_file_name  x[m] (or latitude[deg])  y[m] (or longitude[deg])
altitude_ellipsoidal[m]  roll[deg]  pitch[deg]  heading[deg]
```

Requires **≥ 2 panorama captures** where the same object is visible.

Outputs `output/egsa87/<input_name>_EGSA87.csv` with columns:

| point_name | X_egsa87 | Y_egsa87 | Z_egsa87 |
|---|---|---|---|

### Panorama stitching from raw images (optional)

```bash
python raw_to_panorama.py
```

Stitches all raw camera images in a folder into a single equirectangular panorama using the Ladybug calibration file (no external dependencies beyond numpy and opencv).
