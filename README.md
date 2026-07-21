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
| **Rotation** | None | 90° CW before detection, unrotated after |
| **Coord. transform** | None (already panoramic) | DistortedSpline (`.cal`) → panoramic |

Both modes produce the same `image_coords.csv` format and feed into the same 3D coordinate computation step.

---

## Pipeline 1 — Feature Extraction

### Data flow

```
Image folder (.jpg)
      │
      ▼  [Mode B only: rotate 90° CW]
run_detection()                   ← Detectron.py
      │
 ┌────┴───────────────┐
OD (bounding boxes)   P / SS (masks)
 │                     │
 │                    Centroid.py
 │                     │
 └──────────┬──────────┘
            │
            ▼  [Mode B only]
   raw_coords.csv                 ← output/coords/
            │
            ▼  [Mode B only]
   RawLadybugTransformer          ← raw_to_panorama.py
   (DistortedSpline from .cal)
            │
            ▼
   image_coords.csv               ← output/coords/
            │
      ┌─────┴──────────────────────────────┐
      │                                    │
      ▼                                    ▼
associate()                      run_intersection()
← point_association.py           ← forward_intersection.py
(recommended for OD)             (for P / SS masks)
      │                                    │
      ▼                                    ▼
*_associated_EGSA87.csv          *_EGSA87.csv
                  └──────────────────────┘
                           │
                     output/egsa87/
```

### 3D coordinate methods

Two methods are available, selected via `INTERSECTION_MODE` in the notebook:

| `INTERSECTION_MODE` | Method | Best for |
|---|---|---|
| `'association'` | Multi-view graph grouping + WLS forward intersection | OD (object detection) |
| `'intersection'` | N-ray WLS per named point (K×K pixel neighbourhood, Gaussian weights) | P / SS (masks) |

**Association** (`point_association.py`): groups detections of the same physical object across captures using a graph where edges require matching class, ray proximity (≤ 0.15 m), positive depth, and object distance ≤ 20 m from the camera. Connected components with ≥ 2 observations are triangulated. Output includes one row per confirmed real-world object.

**Intersection** (`forward_intersection.py`): groups observations by `point_name`, applies a K×K pixel neighbourhood around each centroid, selects the best ray per image using Gaussian weights, and solves via weighted least squares. Requires ≥ 2 observations per point.

---

## Pipeline 2 — Image Blurring

Detects objects and blurs their regions in-place using Gaussian blur `(31×31, σ=0)`:

| Model type | Blurred region |
|---|---|
| OD | Bounding box |
| IS | Instance mask |
| P | Panoptic segment mask |
| SS | All non-background pixels |

Run via `code/scripts/pipeline_blur.py`. Output: `output/blurred/` — one blurred `.jpg` per input image. In Mode B the blurred images are saved in detection (portrait) orientation.

---

## Ladybug 5+

![ladybug5plus](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/83de8cfa-3f98-4303-8c37-20e5e7db9a97)

The [Ladybug 5+](https://www.flir.com/products/ladybug5plus/?vertical=machine+vision&segment=iis) is a Teledyne FLIR multi-camera system (5 side cameras + 1 top camera, 30 MP total) designed for mobile mapping. Camera calibration is provided as a `.cal` file containing per-lens focal lengths, principal points, and 2D B-spline warp tables for lens distortion correction.

---

## Repository Structure

```
code/
  scripts/
    pipeline.ipynb          ← main notebook — run this
    pipeline_blur.py        ← image blurring (standalone script)

  lib/
    Detectron.py            ← Detector class + run_detection()
    Centroid.py             ← centroid extraction from segmentation masks
    raw_to_panorama.py      ← raw pixel → panoramic coord + image stitching
    forward_intersection.py ← N-ray WLS triangulation (per named point)
    point_association.py    ← graph-based grouping + WLS triangulation

data/
  Ladybug5_plus/
    ladybug20344317.cal     ← Ladybug 5+ camera calibration

output/                     ← all outputs (gitignored)
  masks/                    ← binary mask images (P / SS models)
  coords/                   ← image_coords.csv, raw_coords.csv
  egsa87/                   ← *_EGSA87.csv, *_associated_EGSA87.csv
  blurred/                  ← blurred images
  panorama/                 ← panorama_from_raw.jpg

projects/                   ← pre-trained model weights (see projects.md)
  Crosswalk/output/
  Traffic_Sign/output/
  Cityscapes/panoptic/
  MaskFormer/panoptic/

help/
  visualize_coords.py       ← debug: draw detections on raw/panoramic images
```

---

## Detection Models

| Model | Type | Classes |
|---|---|---|
| `COCO` | OD / IS / P | person, car, traffic light, stop sign, … |
| `Cityscapes` | SS / P | road, pole, traffic light, traffic sign, … |
| `Crosswalk` | OD | Crosswalk |
| `Traffic_Sign` | OD | 24 sign classes (Stop, Speed limits, Give way, Attention, …) |
| `Safety_Cones` | OD | Safety Cone |

Model types: **OD** = object detection, **IS** = instance segmentation, **P** = panoptic segmentation, **SS** = semantic segmentation.

Pre-trained weights and training notebooks are linked in [projects.md](projects.md).

---

## Required Inputs

| Input | Description |
|---|---|
| Image folder | Panoramic `.jpg` (Mode A) or raw Ladybug `.jpg` per camera (Mode B) |
| `.cal` file | Ladybug calibration file — required for Mode B |
| EOP CSV | Tab-separated exterior orientation parameters (see format below) |

**EOP CSV columns** (tab-separated):

| Column | Description |
|---|---|
| `panorama_file_name` | Must match `image_name` in `image_coords.csv` exactly |
| `latitude[deg]` | WGS84 latitude |
| `longitude[deg]` | WGS84 longitude |
| `altitude_ellipsoidal[m]` | Ellipsoidal height |
| `roll[deg]` | Platform roll |
| `pitch[deg]` | Platform pitch |
| `heading[deg]` | Platform heading |

> In Mode B, `panorama_file_name` must match the raw filename with the `_Cam<N>` suffix stripped (e.g. `pano_0001_0042_Cam2.jpg` → `pano_0001_0042`).

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
Adds `opencv-python`, `matplotlib`, `scipy`, `networkx`, and `pyproj` on top of the Detectron2 environment.

**5. Download model weights**

See [projects.md](projects.md) for download links. Place them under `projects/`.

---

## Usage

### Feature Extraction

Open `code/scripts/pipeline.ipynb` in VS Code or JupyterLab and run the cells in order.

**Cell 1 — Configuration** (only cell that needs editing):

| Variable | Description |
|---|---|
| `MODE` | `'A'` for panoramic images, `'B'` for raw Ladybug images |
| `IMAGE_FOLDER` | Absolute path to folder with input `.jpg` images |
| `CAL_FILE` | Path to `.cal` calibration file (Mode B only) |
| `MODELS` | List of `{'model': ..., 'model_type': ...}` dicts |
| `EOP_CSV` | Path to EOP CSV; leave empty to skip 3D computation |
| `INTERSECTION_MODE` | `'association'` (OD) or `'intersection'` (P / SS) |

**Cell 2 — Imports**: sets up `sys.path` and imports all modules.

**Cell 3 — Step 1: Detection**: runs all models, writes `output/coords/image_coords.csv`.

**Cell 4 — (Optional) coords override**: set `_coords_override` to an existing `image_coords.csv` path to run Step 2 without re-running detection. Must point to `image_coords.csv`, not `raw_coords.csv`.

**Cell 5 — Step 2: 3D Coordinates**: reads `image_coords.csv` + EOP CSV, writes to `output/egsa87/`.

**Output CSV — `association` mode** (`*_associated_EGSA87.csv`):

| object_id | cls | X_egsa87 | Y_egsa87 | Z_egsa87 | n_obs | residual_m | image_ids | detection_ids |
|---|---|---|---|---|---|---|---|---|

**Output CSV — `intersection` mode** (`*_EGSA87.csv`):

| point_name | X_egsa87 | Y_egsa87 | Z_egsa87 |
|---|---|---|---|

> Both modes require ≥ 2 captures where the same object is visible.

---

### Image Blurring

Edit the configuration block at the top of `code/scripts/pipeline_blur.py` and run:

```bash
python code/scripts/pipeline_blur.py
```

Output: `output/blurred/` — one blurred `.jpg` per input image.

---

### Panorama stitching from raw images (optional)

```bash
python code/lib/raw_to_panorama.py
```

Stitches all raw Cam0–Cam5 images into a single 8000×4000 equirectangular panorama using the Ladybug `.cal` calibration file. Output: `output/panorama/panorama_from_raw.jpg`.
