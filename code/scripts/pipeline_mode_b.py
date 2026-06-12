"""
End-to-end Mode B pipeline — raw Ladybug images.

Configure the block below, then run:
    python pipeline_mode_b.py

Steps:
  1+2+3  Detection + raw_coords.csv + image_coords.csv  (output/coords/)
  4      Forward intersection -> EGSA87 CSV  (output/egsa87/)  [optional]
  5      Graph association   -> deduplicated EGSA87 CSV        [optional, requires EOP]

EOP note:
  panorama_file_name in EOP must exactly match image_name in image_coords.csv.
  image_name is derived by stripping '_Cam<N>...' from the raw filename.
"""

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================
# CONFIGURATION  — edit this block
# ============================================================

IMAGE_FOLDER = r''   # absolute path to folder with raw Ladybug .jpg images

CAL_FILE = r''       # absolute path to .cal calibration file

MODELS = [
    {'model': 'COCO', 'model_type': 'OD'},
    # {'model': 'Crosswalk',    'model_type': 'OD'},
    # {'model': 'Safety_Cones', 'model_type': 'OD'},
    # {'model': 'COCO',         'model_type': 'P'},
    # {'model': 'Cityscapes',   'model_type': 'P'},
]

EOP_CSV = r''        # path to GET EOP CSV; leave empty to skip step 4

# How to compute 3-D coordinates from the detections (requires EOP_CSV):
#   'association' — graph-based multi-view grouping + RANSAC (recommended for OD)
#   'intersection' — classic N-ray WLS per named point  (for P / SS masks)
INTERSECTION_MODE = 'association'


# ============================================================
# RUN
# ============================================================

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

from Detectron            import run_detection
from forward_intersection import run_intersection
from graph_association    import associate

assert IMAGE_FOLDER, "Set IMAGE_FOLDER in the configuration block."
assert CAL_FILE,     "Set CAL_FILE in the configuration block."

coords_csv = run_detection(IMAGE_FOLDER, MODELS, mode='B', cal_file=CAL_FILE)

if EOP_CSV:
    if INTERSECTION_MODE == 'association':
        # per_camera mode uses raw_coords.csv (before equirectangular reprojection)
        # so that the intra-capture merge operates on original per-camera rays.
        raw_coords_csv = os.path.join(os.path.dirname(coords_csv), 'raw_coords.csv')
        associate(raw_coords_csv, EOP_CSV, mode='per_camera', cal_file=CAL_FILE)
    else:
        run_intersection(coords_csv, EOP_CSV)

print("\nMode B pipeline complete.")
