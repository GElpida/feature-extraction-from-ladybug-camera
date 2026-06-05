"""
Blur pipeline — Gaussian-blur detected objects and save the blurred images.

Alternative use of Detectron2: instead of extracting coordinates, detected
regions are blurred in-place (anonymisation, privacy masking, etc.).

Works for all model types:
  OD  — blurs each bounding box
  IS  — blurs each instance segmentation mask
  P   — blurs all panoptic segments
  SS  — blurs all non-background pixels

Mode A: panoramic images, no rotation.
Mode B: raw Ladybug images, rotated 90° CW for detection; blurred image is
        saved in detection (portrait) space — no back-rotation applied.

Output: output/blurred/
"""

import os, sys

# ============================================================
# CONFIGURATION  — edit this block
# ============================================================

IMAGE_FOLDER = r''   # absolute path to folder with input images

MODE     = 'A'       # 'A' = panoramic  |  'B' = raw Ladybug

CAL_FILE = r''       # .cal calibration file  (required when MODE = 'B')

MODELS = [
    {'model': 'Traffic_Sign', 'model_type': 'OD'},
    # {'model': 'Crosswalk',    'model_type': 'OD'},
    # {'model': 'Safety_Cones', 'model_type': 'OD'},
    # {'model': 'COCO',         'model_type': 'P'},
    # {'model': 'Cityscapes',   'model_type': 'P'},
]

# ============================================================
# RUN
# ============================================================

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

from Detectron import run_detection

assert IMAGE_FOLDER, "Set IMAGE_FOLDER in the configuration block."
if MODE == 'B':
    assert CAL_FILE, "Set CAL_FILE in the configuration block (required for Mode B)."

blur_dir = run_detection(IMAGE_FOLDER, MODELS, mode=MODE,
                         cal_file=CAL_FILE or None,
                         output_mode='blur')

print(f"\nBlur pipeline complete.  Output -> {blur_dir}")
