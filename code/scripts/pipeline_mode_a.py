"""
End-to-end Mode A pipeline — panoramic images.

Configure the block below, then run:
    python pipeline_mode_a.py

Steps:
  1+2  Detection + image_coords.csv  (output/coords/)
  3    Forward intersection -> EGSA87 CSV  (output/egsa87/)  [optional]
"""

import os, sys

# ============================================================
# CONFIGURATION  — edit this block
# ============================================================

IMAGE_FOLDER = r''   # absolute path to folder with panoramic .jpg images

MODELS = [
    {'model': 'Traffic_Sign', 'model_type': 'OD'},
    # {'model': 'Crosswalk',    'model_type': 'OD'},
    # {'model': 'Safety_Cones', 'model_type': 'OD'},
    # {'model': 'COCO',         'model_type': 'P'},
    # {'model': 'Cityscapes',   'model_type': 'P'},
]

EOP_CSV = r''        # path to GET EOP CSV; leave empty to skip step 3

# ============================================================
# RUN
# ============================================================

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

from Detectron            import run_detection
from forward_intersection import run_intersection

assert IMAGE_FOLDER, "Set IMAGE_FOLDER in the configuration block."

coords_csv = run_detection(IMAGE_FOLDER, MODELS, mode='A')

if EOP_CSV:
    run_intersection(coords_csv, EOP_CSV)

print("\nMode A pipeline complete.")
