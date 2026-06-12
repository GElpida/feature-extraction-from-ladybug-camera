"""
Raw Ladybug image -> panoramic coordinate transformer.

Two capabilities:
  1. Coordinate transform  : raw pixel (u,v) + cam_id  ->  panoramic (u,v)
     Pipeline: raw pixel -> DistortedSpline undistort -> pinhole ray -> equirectangular

  2. Image stitching (main): folder of raw images -> undistort -> stitch -> panorama.jpg
     Uses RectifiedSpline with cv2.remap for fast per-image undistortion, then
     vectorised ray projection into the panorama canvas.

Calibration format: FLIR Ladybug .cal file (v2.0).
  Per-camera:  focalLength, Center (principal point), CamToLadybugEulerZYX
  Warp tables: BeginWarp / EndWarp blocks with 2-D cubic B-spline coefficients.
    RectifiedSpline warpUId warpVId  ->  rectified pixel -> raw pixel  (for remap)
    DistortedSpline warpUId warpVId  ->  raw pixel -> rectified pixel  (for points)

No external dependencies beyond numpy and opencv.
"""

import os
import re
import glob
import csv
import numpy as np
import cv2
from math import sin, cos


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def _get_T_ZYX(Rx, Ry, Rz, Tx, Ty, Tz):
    r11 = cos(Ry)*cos(Rz)
    r12 = sin(Rx)*sin(Ry)*cos(Rz) - cos(Rx)*sin(Rz)
    r13 = cos(Rx)*sin(Ry)*cos(Rz) + sin(Rx)*sin(Rz)
    r21 = cos(Ry)*sin(Rz)
    r22 = sin(Rx)*sin(Ry)*sin(Rz) + cos(Rx)*cos(Rz)
    r23 = cos(Rx)*sin(Ry)*sin(Rz) - sin(Rx)*cos(Rz)
    r31 = -sin(Ry)
    r32 = sin(Rx)*cos(Ry)
    r33 = cos(Rx)*cos(Ry)
    return np.array([
        [r11, r12, r13, Tx],
        [r21, r22, r23, Ty],
        [r31, r32, r33, Tz],
        [0,   0,   0,   1 ]
    ])


# ============================================================
# MAIN CLASS
# ============================================================

class RawLadybugTransformer:
    """
    Transforms raw (distorted) Ladybug pixel coordinates to panoramic
    equirectangular coordinates, and undistorts full images for stitching.
    """

    def __init__(self, cal_path):
        self.iop = {}            # cam_id -> {fx_norm, fy_norm, cx_norm, cy_norm}
        self.eop = {}            # cam_id -> {Rx, Ry, Rz, Tx, Ty, Tz}
        self.rect_splines = {}   # cam_id -> (warp_u_id, warp_v_id)  rectified->raw
        self.dist_splines = {}   # cam_id -> (warp_u_id, warp_v_id)  raw->rectified
        self.warp_tables  = {}   # warp_id -> {knots_x, knots_y, coefs, n_u, n_v}
        self._remap_cache = {}   # (cam_id, img_w, img_h) -> (map_u, map_v)
        self._parse_cal(cal_path)

    # --------------------------------------------------------
    # CALIBRATION PARSER
    # --------------------------------------------------------

    def _parse_cal(self, cal_path):
        with open(cal_path, 'r') as f:
            lines = f.readlines()

        # Pass 1: camera blocks
        cam = None
        in_cam = False
        for line in lines:
            s = line.strip()
            if s.startswith('BeginCamera'):
                in_cam = True; cam = None; continue
            if s.startswith('EndCamera'):
                in_cam = False; cam = None; continue
            if not in_cam:
                continue
            if s.startswith('Id '):
                cam = int(s.split()[1])
                self.iop[cam] = {}; self.eop[cam] = {}
            elif s.startswith('focalLength') and cam is not None:
                p = s.split()
                self.iop[cam]['fx_norm'] = float(p[1])
                self.iop[cam]['fy_norm'] = float(p[2])
            elif s.startswith('Center') and cam is not None:
                p = s.split()
                self.iop[cam]['cx_norm'] = float(p[4])
                self.iop[cam]['cy_norm'] = float(p[5])
            elif s.startswith('CamToLadybugEulerZYX') and cam is not None:
                p = s.split()
                self.eop[cam] = {k: float(v) for k, v in zip(
                    ['Rx','Ry','Rz','Tx','Ty','Tz'], p[1:7])}
            elif s.startswith('RectifiedSpline') and cam is not None:
                p = s.split()
                self.rect_splines[cam] = (int(p[1]), int(p[2]))
            elif s.startswith('DistortedSpline') and cam is not None:
                p = s.split()
                self.dist_splines[cam] = (int(p[1]), int(p[2]))

        # Pass 2: warp table blocks
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith('BeginWarp'):
                warp_id = None
                knots_x = knots_y = None
                n_u = n_v = None
                coef_vals = []
                reading_coefs = False
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('EndWarp'):
                    s = lines[i].strip()
                    if s.startswith('Id '):
                        warp_id = int(s.split()[1])
                    elif s.startswith('KnotsX'):
                        knots_x = np.array([float(v) for v in s.split()[1:]])
                    elif s.startswith('KnotsY'):
                        knots_y = np.array([float(v) for v in s.split()[1:]])
                    elif s.startswith('NumberCoefs'):
                        p = s.split(); n_u, n_v = int(p[1]), int(p[2])
                    elif s == 'Coefs':
                        reading_coefs = True
                    elif reading_coefs and s:
                        try:
                            coef_vals.extend(float(v) for v in s.split())
                        except ValueError:
                            pass
                    i += 1
                if warp_id is not None and n_u is not None \
                        and len(coef_vals) >= n_u * n_v:
                    self.warp_tables[warp_id] = {
                        'knots_x': knots_x,
                        'knots_y': knots_y,
                        'coefs':   np.array(coef_vals[:n_u * n_v]).reshape(n_u, n_v),
                        'n_u': n_u, 'n_v': n_v,
                    }
            i += 1

        print(f"Calibration loaded: {len(self.iop)} cameras, "
              f"{len(self.warp_tables)} warp tables.")

    # --------------------------------------------------------
    # B-SPLINE BASIS FUNCTIONS  (pure numpy, no scipy)
    # Cox-De Boor recurrence — vectorised over t_values.
    # --------------------------------------------------------

    @staticmethod
    def _basis(knots, t_values, degree=3):
        """
        Evaluate all cubic B-spline basis functions at each value in t_values.
        Returns array of shape (n_coefs, len(t_values))
        where n_coefs = len(knots) - degree - 1.

        Implements Cox-De Boor with the standard convention 0/0 = 0.
        Handles clamped boundary (t == right endpoint) explicitly.
        """
        knots = np.asarray(knots, dtype=float)
        t = np.clip(np.atleast_1d(np.asarray(t_values, dtype=float)),
                    knots[degree], knots[-degree - 1])
        n_t = len(t)

        # --- degree-0 indicator functions N_{i,0}(t) ---
        # N[i] = 1  iff  knots[i] <= t < knots[i+1]
        n_spans = len(knots) - 1          # total number of spans
        N = np.zeros((n_spans, n_t))
        for k in range(n_spans - 1):     # all spans except last dummy one
            N[k] = (t >= knots[k]) & (t < knots[k + 1])

        # Right boundary: t == knots[-degree-1] falls into the last active span
        # (the last span where knots[k] < knots[k+1])
        last_active = len(knots) - degree - 2  # start search here
        while last_active > 0 and knots[last_active] == knots[last_active + 1]:
            last_active -= 1
        at_end = (t >= knots[-degree - 1])
        if np.any(at_end):
            N[:, at_end] = 0.0
            N[last_active, at_end] = 1.0

        # --- recurrence: degree 1, 2, 3 ---
        for d in range(1, degree + 1):
            n_new = len(knots) - d - 1
            N_new = np.zeros((n_new, n_t))
            for k in range(n_new):
                d1 = knots[k + d]     - knots[k]
                d2 = knots[k + d + 1] - knots[k + 1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    left  = np.where(d1 > 0,
                                     (t - knots[k])         / d1 * N[k],     0.0)
                    right = np.where(d2 > 0,
                                     (knots[k+d+1] - t)     / d2 * N[k + 1], 0.0)
                N_new[k] = left + right
            N = N_new

        return N   # shape (n_coefs, n_t)

    # --------------------------------------------------------
    # WARP TABLE EVALUATION
    # --------------------------------------------------------

    def _eval_warp(self, warp_id, u_norm, v_norm):
        """
        Evaluate one 2-D warp table at paired normalised coordinates.
        u_norm, v_norm: equal-length 1-D arrays or scalars, values in [0,1].
        Returns output values (same shape as input).
        """
        wt = self.warp_tables[warp_id]
        scalar = np.isscalar(u_norm)
        u = np.atleast_1d(np.asarray(u_norm, float))
        v = np.atleast_1d(np.asarray(v_norm, float))

        Bu = self._basis(wt['knots_x'], u)   # (n_u, len)
        Bv = self._basis(wt['knots_y'], v)   # (n_v, len)

        # result[k] = Bu[:,k] @ C @ Bv[:,k]
        temp    = wt['coefs'] @ Bv          # (n_u, len)
        results = (Bu * temp).sum(axis=0)   # (len,)

        return float(results[0]) if scalar else results

    def _eval_warp_grid(self, warp_id, u_grid, v_grid):
        """
        Evaluate a 2-D warp table on a full rectangular grid.
        u_grid: 1-D normalised column positions, length img_w
        v_grid: 1-D normalised row    positions, length img_h
        Returns array of shape (img_h, img_w).
        """
        wt = self.warp_tables[warp_id]
        Bu = self._basis(wt['knots_x'], u_grid)   # (n_u, img_w)
        Bv = self._basis(wt['knots_y'], v_grid)   # (n_v, img_h)

        # result[y,x] = Bu[:,x] @ C @ Bv[:,y]
        temp   = wt['coefs'] @ Bv    # (n_u, img_h)
        result = temp.T @ Bu         # (img_h, img_w)
        return result

    # --------------------------------------------------------
    # COORDINATE TRANSFORMS
    # --------------------------------------------------------

    def raw_pixel_to_rectified(self, u_pix, v_pix, cam_id, img_w, img_h):
        """Map a raw pixel to rectified pixel coordinates via DistortedSpline."""
        wu, wv = self.dist_splines[cam_id]
        u_rect = self._eval_warp(wu, u_pix / img_w, v_pix / img_h) * img_w
        v_rect = self._eval_warp(wv, u_pix / img_w, v_pix / img_h) * img_h
        return float(u_rect), float(v_rect)

    def rectified_pixel_to_panorama(self, u_rect, v_rect, cam_id,
                                     img_w, img_h, W_pano=8000, H_pano=4000):
        """Map a rectified pixel to equirectangular panoramic coordinates."""
        iop = self.iop[cam_id]
        eop = self.eop[cam_id]

        fx = iop['fx_norm'] * img_w
        fy = iop['fy_norm'] * img_h
        cx = iop['cx_norm'] * img_w
        cy = iop['cy_norm'] * img_h

        ray_cam = np.array([(u_rect - cx) / fx,
                             (v_rect - cy) / fy,
                             1.0])
        R   = _get_T_ZYX(eop['Rx'], eop['Ry'], eop['Rz'],
                          eop['Tx'], eop['Ty'], eop['Tz'])[:3, :3]
        x, y, z = R @ ray_cam

        r = np.sqrt(x*x + y*y + z*z)
        if r < 1e-12:
            return None, None

        theta  = np.arctan2(-y, x)
        phi    = np.arccos(np.clip(z / r, -1.0, 1.0))
        u_pano = int(((theta + np.pi) / (2.0 * np.pi)) * W_pano) % W_pano
        v_pano = int(np.clip((phi / np.pi) * H_pano, 0, H_pano - 1))
        return u_pano, v_pano

    def raw_pixel_to_panorama(self, u_pix, v_pix, cam_id,
                               img_w, img_h, W_pano=8000, H_pano=4000):
        """Full pipeline: raw pixel -> rectified -> panoramic."""
        u_rect, v_rect = self.raw_pixel_to_rectified(
            u_pix, v_pix, cam_id, img_w, img_h)
        return self.rectified_pixel_to_panorama(
            u_rect, v_rect, cam_id, img_w, img_h, W_pano, H_pano)

    # --------------------------------------------------------
    # IMAGE UNDISTORTION  (remap via RectifiedSpline)
    # --------------------------------------------------------

    def build_undistort_maps(self, cam_id, img_w, img_h):
        """
        Build cv2.remap source maps for undistorting a raw image.
        Uses RectifiedSpline: for each output rectified pixel (x,y) the maps
        give the source raw pixel coordinates (standard inverse-mapping).
        Results are cached.
        """
        key = (cam_id, img_w, img_h)
        if key in self._remap_cache:
            return self._remap_cache[key]

        wu, wv = self.rect_splines[cam_id]
        u_grid = np.linspace(0.0, 1.0, img_w)
        v_grid = np.linspace(0.0, 1.0, img_h)

        print(f"  Building remap maps for Cam {cam_id} ({img_w}x{img_h}) ...")
        map_u = (self._eval_warp_grid(wu, u_grid, v_grid) * img_w).astype(np.float32)
        map_v = (self._eval_warp_grid(wv, u_grid, v_grid) * img_h).astype(np.float32)

        self._remap_cache[key] = (map_u, map_v)
        return map_u, map_v

    def undistort_image(self, img, cam_id):
        """Undistort a raw image using RectifiedSpline remap maps."""
        h, w = img.shape[:2]
        map_u, map_v = self.build_undistort_maps(cam_id, w, h)
        return cv2.remap(img, map_u, map_v, cv2.INTER_LINEAR)

    # --------------------------------------------------------
    # BATCH CSV TRANSFORM  (Mode B pipeline step)
    # --------------------------------------------------------

    def transform_csv(self, input_csv, output_csv, W_pano=8000, H_pano=4000):
        """
        Transform a CSV of raw image coordinates to panoramic coordinates.

        Input columns : point_name, image_name, cam_id, img_w, img_h,
                        raw_x[px], raw_y[px]
        Output columns: point_name, image_name, x[px], y[px]

        'image_name' in the output is derived by stripping '_Cam<N>' from the
        raw image name so it matches 'panorama_file_name' in the EOP CSV.
        """
        with open(input_csv, newline='') as f:
            rows = list(csv.DictReader(f))

        out_rows = []
        for row in rows:
            cam_id = int(row['cam_id'])
            img_w  = int(row['img_w'])
            img_h  = int(row['img_h'])
            u_raw  = float(row['raw_x[px]'])
            v_raw  = float(row['raw_y[px]'])

            u_pano, v_pano = self.raw_pixel_to_panorama(
                u_raw, v_raw, cam_id, img_w, img_h, W_pano, H_pano)

            if u_pano is None:
                continue

            pano_name = re.sub(r'_Cam\d+.*$', '', row['image_name'])
            out_rows.append({
                'point_name': row['point_name'],
                'image_name': pano_name,
                'cls':        row.get('cls', ''),
                'x[px]':      u_pano,
                'y[px]':      v_pano,
                'angular_w':  row.get('angular_w', 0.0),
            })

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['point_name', 'image_name', 'cls',
                               'x[px]', 'y[px]', 'angular_w'])
            writer.writeheader()
            writer.writerows(out_rows)

        print(f"Transformed {len(out_rows)} observations -> {output_csv}")


# ============================================================
# MAIN  –  stitch raw images into a panorama
#
# Mirrors rectified_to_panorama.py exactly:
#   1. Filter black pixels from the original raw image first.
#   2. Use DistortedSpline (grid evaluation) to get the rectified coordinates
#      of each valid raw pixel.
#   3. Apply pinhole model + CamToLadybugEulerZYX to build rays.
#   4. Project rays to equirectangular canvas.
#
# No intermediate remap / undistortion step — remap border artefacts
# never enter the pipeline.
# ============================================================

def main():
    base = os.path.dirname(os.path.abspath(__file__))

    _root      = os.path.join(base, '..', '..')   # lib/ -> code/ -> project root
    cal_path   = os.path.join(_root, 'data', 'Ladybug5_plus',
                               'ladybug20344317.cal')
    raw_folder = os.path.join(_root, 'images', 'raw')
    out_dir    = os.path.join(_root, 'output', 'panorama')
    out_path   = os.path.join(out_dir, 'panorama_from_raw.jpg')
    W_pano     = 8000
    H_pano     = 4000

    print(f"Cal file  : {cal_path}")
    print(f"Raw folder: {raw_folder}")
    print(f"Output    : {out_path}")
    print(f"Panorama  : {W_pano} x {H_pano}")
    print()

    transformer = RawLadybugTransformer(cal_path)

    panorama = np.zeros((H_pano, W_pano, 3), dtype=np.float64)
    weight   = np.zeros((H_pano, W_pano),    dtype=np.float64)

    image_paths = sorted(
        glob.glob(os.path.join(raw_folder, '*.jpg')) +
        glob.glob(os.path.join(raw_folder, '*.png'))
    )

    if not image_paths:
        print(f"No images found in {raw_folder}")
        return

    for path in image_paths:
        name   = os.path.basename(path)
        m      = re.search(r'Cam(\d+)', name)
        if not m:
            print(f"Skipping {name} (no Cam ID in filename)")
            continue
        cam_id = int(m.group(1))
        if cam_id not in transformer.dist_splines:
            print(f"Skipping {name} (cam {cam_id} not in calibration)")
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"Could not read {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        print(f"Cam {cam_id}  {name}  ({w}x{h})")

        # --- Step 1: filter black pixels from the raw image ---
        v_arr, u_arr = np.mgrid[0:h, 0:w]
        pixels = img.reshape(-1, 3).astype(np.float64)
        valid  = ~((pixels[:, 0] == 0) &
                   (pixels[:, 1] == 0) &
                   (pixels[:, 2] == 0))
        pix_v  = pixels[valid]
        v_idx  = v_arr.flatten()[valid]   # integer row   indices of valid pixels
        u_idx  = u_arr.flatten()[valid]   # integer column indices of valid pixels

        # --- Step 2: DistortedSpline on full grid, gather valid pixels ---
        # _eval_warp_grid is fast (separable tensor product over w+h values).
        wu, wv = transformer.dist_splines[cam_id]
        u_grid = np.linspace(0.0, 1.0, w)
        v_grid = np.linspace(0.0, 1.0, h)
        print(f"  Computing DistortedSpline grid ...")
        u_rect_norm = transformer._eval_warp_grid(wu, u_grid, v_grid)  # (h, w)
        v_rect_norm = transformer._eval_warp_grid(wv, u_grid, v_grid)  # (h, w)

        u_rect = u_rect_norm[v_idx, u_idx] * w   # rectified x in pixels
        v_rect = v_rect_norm[v_idx, u_idx] * h   # rectified y in pixels

        # --- Step 3: pinhole model + rotation → Ladybug frame ---
        iop = transformer.iop[cam_id]
        eop = transformer.eop[cam_id]
        fx  = iop['fx_norm'] * w;   fy = iop['fy_norm'] * h
        cx  = iop['cx_norm'] * w;   cy = iop['cy_norm'] * h
        R   = _get_T_ZYX(eop['Rx'], eop['Ry'], eop['Rz'],
                          eop['Tx'], eop['Ty'], eop['Tz'])[:3, :3]

        n_valid = int(valid.sum())
        rays    = np.stack([(u_rect - cx) / fx,
                             (v_rect - cy) / fy,
                             np.ones(n_valid)], axis=1)
        rays_lb = (R @ rays.T).T

        norms   = np.linalg.norm(rays_lb, axis=1, keepdims=True).clip(min=1e-12)
        rays_lb /= norms

        # --- Step 4: project to equirectangular canvas ---
        theta  = np.arctan2(-rays_lb[:, 1], rays_lb[:, 0])
        phi    = np.arccos(np.clip(rays_lb[:, 2], -1.0, 1.0))
        u_pano = (((theta + np.pi) / (2.0 * np.pi)) * W_pano
                  ).astype(int) % W_pano
        v_pano = np.clip((phi / np.pi * H_pano).astype(int), 0, H_pano - 1)

        np.add.at(panorama, (v_pano, u_pano), pix_v)
        np.add.at(weight,   (v_pano, u_pano), 1.0)
        print(f"  -> {n_valid} pixels projected.")

    # normalise and save
    mask = weight > 0
    panorama[mask] /= weight[mask, np.newaxis]
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
