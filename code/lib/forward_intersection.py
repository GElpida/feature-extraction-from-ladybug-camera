"""
Forward intersection using GET EOP format.

Can be used as a library:
    from forward_intersection import run_intersection
    output_csv = run_intersection(point_path, eop_path)

Or run standalone:
    python forward_intersection.py

EOP CSV columns (tab-separated):
    gps_seconds[s]  panorama_file_name  latitude[deg]  longitude[deg]
    altitude_ellipsoidal[m]  roll[deg]  pitch[deg]  heading[deg]

EOP name matching:
    Exact  panorama_file_name == image_name.  Skips if no match.

Writes:
    output/egsa87/<stem>_EGSA87.csv   columns: point_name, X_egsa87, Y_egsa87, Z_egsa87
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

# ============================================================
# MODULE-LEVEL CONSTANTS
# ============================================================

_BASE      = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_to_egsa87 = Transformer.from_crs("EPSG:4326", "EPSG:2100", always_xy=True)

# ============================================================
# INTERNAL HELPERS
# ============================================================

def _rotation_egsa(roll_deg, pitch_deg, heading_deg):
    roll    = np.radians(roll_deg)
    pitch   = np.radians(pitch_deg)
    heading = np.radians(heading_deg)
    Rz = np.array([[np.cos(heading), -np.sin(heading), 0],
                   [np.sin(heading),  np.cos(heading), 0],
                   [0,                0,               1]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [ 0,             1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0,             0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    T_ned_to_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    return T_ned_to_enu @ Rz @ Ry @ Rx


def _closest_point_n_rays(origins, directions, weights=None):
    N = len(origins)
    if weights is None:
        weights = np.ones(N)
    A = np.zeros((3, 3));  b = np.zeros(3)
    for P, d, w in zip(origins, directions, weights):
        d = d / np.linalg.norm(d)
        M = np.eye(3) - np.outer(d, d)
        A += w * M;  b += w * (M @ P)
    return np.linalg.solve(A, b)


def _ray_dist(P, origin, direction):
    d = direction / np.linalg.norm(direction)
    v = P - origin
    return np.linalg.norm(v - np.dot(v, d) * d)

# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_intersection(point_path, eop_path,
                     W=8000, H=4000, K=3,
                     visualize=True):
    """
    Run N-ray weighted least-squares forward intersection.

    Parameters
    ----------
    point_path : str   Path to image_coords.csv
                       (columns: point_name, image_name, x[px], y[px])
    eop_path   : str   Path to GET EOP CSV (tab-separated, lat/lon format)
    W, H       : int   Panorama dimensions in pixels  (default 8000 x 4000)
    K          : int   Neighbourhood kernel size in pixels  (default 3)
    visualize  : bool  Show 3-D ray plot per point  (default True)

    Returns
    -------
    str   Path to the output EGSA87 CSV, or None if no points computed.
    """

    # --- load ---
    with open(point_path, newline='') as f:
        points = list(csv.DictReader(f, delimiter=','))
    with open(eop_path, newline='') as f:
        EOP = list(csv.DictReader(f, delimiter='\t'))

    p_names = list(dict.fromkeys(po['point_name'] for po in points))
    point_list    = []
    Ladybug_center = []
    half  = K // 2
    sigma = max(half, 0.5)

    for name in p_names:

        image_candidates = []

        for po in points:
            if po['point_name'] != name:
                continue

            u0 = float(po['x[px]']);  v0 = float(po['y[px]'])
            if u0 < 0 or u0 >= W or v0 < 0 or v0 >= H:
                continue

            # ----------------------------
            # FIND EOP MATCH
            # ----------------------------
            eop_match = None
            for param in EOP:
                if param['panorama_file_name'] == po['image_name']:
                    eop_match = param
                    break

            if eop_match is None:
                continue

            Xgr_c, Ygr_c = _to_egsa87.transform(
                float(eop_match['longitude[deg]']),
                float(eop_match['latitude[deg]'])
            )
            h     = float(eop_match['altitude_ellipsoidal[m]'])
            roll  = float(eop_match['roll[deg]'])
            pitch = float(eop_match['pitch[deg]'])
            head  = float(eop_match['heading[deg]'])

            if [Xgr_c, Ygr_c, h, eop_match['panorama_file_name']] not in Ladybug_center:
                Ladybug_center.append([Xgr_c, Ygr_c, h, eop_match['panorama_file_name']])

            R_egsa = _rotation_egsa(roll, pitch, head)
            origin = np.array([Xgr_c, Ygr_c, h])

            # ----------------------------
            # K×K NEIGHBOURHOOD
            # ----------------------------
            candidates = []
            for du in range(-half, half + 1):
                for dv in range(-half, half + 1):
                    u = u0 + du;  v = v0 + dv
                    if u < 0 or u >= W or v < 0 or v >= H:
                        continue
                    theta = (u / W - 0.5) * 2 * np.pi
                    elev  = (0.5 - v / H) * np.pi
                    r_cam = np.array([np.cos(elev)*np.cos(theta),
                                      np.cos(elev)*np.sin(theta),
                                      np.sin(elev)])
                    direction = R_egsa @ r_cam
                    if np.linalg.norm(direction) < 1e-12:
                        continue
                    candidates.append((origin.copy(), direction, np.sqrt(du**2 + dv**2)))

            if candidates:
                image_candidates.append(candidates)

        if len(image_candidates) < 2:
            print(f"  SKIP '{name}': only {len(image_candidates)} observation(s) "
                  f"(need >= 2 for triangulation)")
            continue

        # STEP 1 — initial solve with nominal pixels
        nominal_o = [min(c, key=lambda x: x[2])[0] for c in image_candidates]
        nominal_d = [min(c, key=lambda x: x[2])[1] for c in image_candidates]
        P0 = _closest_point_n_rays(np.array(nominal_o), np.array(nominal_d))

        # STEP 2 — best ray per image, Gaussian weights
        sel_o = [];  sel_d = [];  sel_w = []
        for cands in image_candidates:
            best     = min(cands, key=lambda c: _ray_dist(P0, c[0], c[1]))
            o, d, pd = best
            sel_o.append(o);  sel_d.append(d)
            sel_w.append(np.exp(-(pd**2) / (2 * sigma**2)))

        # STEP 3 — final weighted solve
        X, Y, Z = _closest_point_n_rays(
            np.array(sel_o), np.array(sel_d), np.array(sel_w))

        point_list.append([X, Y, Z, name])

        print(f'\nPOINT: {name}')
        print(f'  Observations : {len(image_candidates)}')
        print(f'  EGSA87       : X={X:.3f}  Y={Y:.3f}  Z={Z:.3f}')

        if visualize:
            fig = plt.figure()
            ax  = fig.add_subplot(projection='3d')
            ax.scatter(X, Y, Z, c='magenta', s=50)
            ax.text(X, Y, Z, name)
            for i, (P1, d) in enumerate(zip(sel_o, sel_d)):
                P2 = P1 + d * 20
                ax.scatter(*P1, c='green', s=20);  ax.text(*P1, f'C{i}')
                ax.scatter(*P2, c='blue',  s=20);  ax.text(*P2, f'R{i}')
                ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], color='black')
            ax.set_xlabel('X');  ax.set_ylabel('Y');  ax.set_zlabel('Z')
            ax.set_title(f'N-Ray LSQ – {name}')
            plt.show()

    if not point_list:
        print('\nNo points computed.')
        print('Check that panorama_file_name in EOP matches image_name in image_coords.csv.')
        return None

    # --- write output ---
    egsa87_dir = os.path.join(_BASE, 'output', 'egsa87')
    os.makedirs(egsa87_dir, exist_ok=True)
    stem       = os.path.splitext(os.path.basename(point_path))[0]
    output_csv = os.path.join(egsa87_dir, stem + '_EGSA87.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['point_name', 'X_egsa87', 'Y_egsa87', 'Z_egsa87'])
        writer.writeheader()
        for pt in point_list:
            writer.writerow({'point_name': pt[3],
                             'X_egsa87':   pt[0],
                             'Y_egsa87':   pt[1],
                             'Z_egsa87':   pt[2]})
    print(f'\n{len(point_list)} point(s) saved -> {output_csv}')
    return output_csv

# ============================================================
# STANDALONE USAGE
# ============================================================

if __name__ == '__main__':
    _default_coords = os.path.join(_BASE, 'output', 'coords', 'image_coords.csv')
    _default_eop    = os.path.join(_BASE, 'data',   'GET',    'import_locations.csv')

    point_path = input(f'Image coords CSV [{_default_coords}]: ').strip() or _default_coords
    eop_path   = input(f'EOP CSV          [{_default_eop}]: ').strip()    or _default_eop

    run_intersection(point_path, eop_path)
