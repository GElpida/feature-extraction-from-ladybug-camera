"""
Centroid extraction from binary segmentation masks.

compute_centroids(masks_folder, output_csv, mode, point_name)
  mode='A'  Panoramic pipeline: no rotation, output columns are x[px]/y[px]
            (already in panoramic space).
  mode='B'  Raw pipeline: rotate mask 90° CCW before centroid (undoes the
            detection-time CW rotation), output raw image coordinates plus
            cam_id / img_w / img_h for downstream panoramic transform.

Output CSV for Mode A : point_name, image_name, x[px], y[px]
Output CSV for Mode B : point_name, image_name, cam_id, img_w, img_h,
                        raw_x[px], raw_y[px]

Mask filename convention expected:
  <raw_or_pano_image_name>_<model>_<cat_id>_<instance_id>.jpg
  The raw image name must contain  _Cam<N>  to allow cam_id parsing in Mode B.
"""

import glob
import re
import csv
import cv2
import numpy as np


def compute_centroids(masks_folder, output_csv, mode='A', point_name=''):
    """
    Compute centroids of all masks in masks_folder and write to output_csv.

    Parameters
    ----------
    masks_folder : str   Path to folder containing *_mask.jpg files.
    output_csv   : str   Destination CSV path.
    mode         : 'A' or 'B'
    point_name   : str   Label prefix for all points (shared across images).
    """
    mask_paths = sorted(glob.glob(masks_folder + '/*_*.jpg'))

    if not mask_paths:
        print(f"No mask files found in {masks_folder}")
        return

    rows = []

    for path in mask_paths:
        # ---- parse filename -------------------------------------------------
        filename = path.replace('\\', '/').split('/')[-1]
        # strip .jpg extension
        base = filename[:-4] if filename.endswith('.jpg') else filename

        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # ---- Mode B: unrotate mask 90° CCW ---------------------------------
        if mode == 'B':
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_h, img_w = mask.shape

        # ---- centroid -------------------------------------------------------
        M = cv2.moments(mask)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # ---- extract image name and (for Mode B) cam_id --------------------
        # Convention: base = <image_name>_<model>_<cat>_<instance>
        # We split from the right to get image_name (which may contain '_')
        # Use a fixed split: last 3 underscored segments are model/cat/instance
        parts = base.rsplit('_', 3)
        image_name_full = parts[0] if len(parts) == 4 else base

        # Strip file extension that may be embedded in the name
        image_name_full = re.sub(r'\.(jpg|png)$', '', image_name_full,
                                  flags=re.IGNORECASE)

        if mode == 'A':
            # image_name is already the panoramic image name
            rows.append({
                'point_name': point_name,
                'image_name': image_name_full,
                'x[px]':      cx,
                'y[px]':      cy,
            })

        else:  # mode == 'B'
            m = re.search(r'Cam(\d+)', image_name_full)
            cam_id = int(m.group(1)) if m else -1
            rows.append({
                'point_name': point_name,
                'image_name': image_name_full,
                'cam_id':     cam_id,
                'img_w':      img_w,
                'img_h':      img_h,
                'raw_x[px]':  cx,
                'raw_y[px]':  cy,
            })

    if not rows:
        print("No valid centroids computed.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Centroids saved ({len(rows)} rows) -> {output_csv}")


# ============================================================
# STANDALONE USAGE
# ============================================================

if __name__ == '__main__':
    masks_folder = input("Path to masks folder : ").strip()
    output_csv   = input("Output CSV path      : ").strip()
    mode         = input("Mode (A / B)         : ").strip().upper() or 'A'
    point_name   = input("Point name label     : ").strip()

    compute_centroids(masks_folder, output_csv, mode=mode, point_name=point_name)
