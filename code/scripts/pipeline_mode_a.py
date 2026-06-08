"""
End-to-end Mode A pipeline — panoramic images.

Configure the block below, then run:
    python pipeline_mode_a.py

Steps:
  1+2  Detection + image_coords.csv  (output/coords/)
  3    Forward intersection -> EGSA87 CSV  (output/egsa87/)  [optional]
  4    Graph association   -> deduplicated EGSA87 CSV        [optional, requires EOP]
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

# How to compute 3-D coordinates from the detections (requires EOP_CSV):
#   'association' — graph-based multi-view grouping + RANSAC (recommended for OD)
#   'intersection' — classic N-ray WLS per named point  (for P / SS masks)
INTERSECTION_MODE = 'association'

# Raw class names from the model(s) above — used only when INTERSECTION_MODE='association'.
# Must match what the detector writes into point_name.
# Update this list whenever you change MODELS.
CLASS_NAMES = [
    # Traffic_Sign (default model above)
    "", "Attention", "Bend_to_left", "Bend_to_right", "Crosswalk",
    "Fork_road", "Give_way", "Narrow_road", "No_entry", "No_left_turn",
    "No_right_turn", "No_u_turn", "Roundabout_mandatory",
    "Speed_limit_100KM", "Speed_limit_110KM", "Speed_limit_120KM",
    "Speed_limit_20KM", "Speed_limit_30KM", "Speed_limit_40KM",
    "Speed_limit_50KM", "Speed_limit_60KM", "Speed_limit_70KM",
    "Speed_limit_80KM", "Speed_limit_90KM", "Stop",
    # Crosswalk model  (uncomment if used)
    # "", "Crosswalk",
    # Safety_Cones model  (uncomment if used)
    # "", "Safety_Cone",
    # COCO OD / IS  (uncomment if used)
    # "stop sign", "traffic light", "person", "car", ...
]

# ============================================================
# RUN
# ============================================================

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

from Detectron            import run_detection
from forward_intersection import run_intersection
from graph_association    import associate

assert IMAGE_FOLDER, "Set IMAGE_FOLDER in the configuration block."

coords_csv = run_detection(IMAGE_FOLDER, MODELS, mode='A')

if EOP_CSV:
    if INTERSECTION_MODE == 'association':
        associate(coords_csv, EOP_CSV, CLASS_NAMES)
    else:
        run_intersection(coords_csv, EOP_CSV)

print("\nMode A pipeline complete.")
