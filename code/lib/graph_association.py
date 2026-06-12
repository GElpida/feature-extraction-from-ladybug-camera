"""
Graph-based multi-view association for offline batch detection pipelines.

Two operating modes
-------------------
'panorama'   (Mode A): reads stitched equirectangular image_coords.csv.
             Each panorama = one capture.  Rays built from equirectangular
             pixel coordinates.  Capture key = panorama filename.

'per_camera' (Mode B): reads raw per-camera raw_coords.csv from the
             Ladybug rig.  Within each capture the 6 camera images are
             first merged via depth-free reprojection (intra-capture merge,
             B1) so that a single physical object seen by multiple cameras
             becomes one observation node.  That node's ray is the mean of
             the per-camera EGSA87 rays (B2).  The resulting per-capture
             observations feed the common association core identically to
             Mode A.

Common core (both modes)
------------------------
1. Filter A — discard nodes whose angular size implies depth > max_obj_dist.
2. Candidate pairs via cKDTree on EGSA87 (X, Y) — O(n log n).
3. Association graph: 5-criterion edge filter + exp(-ray_dist) weights.
4. Connected components — each node in exactly one group, zero duplicates.
5. RANSAC + scipy.optimize.least_squares per group.
   · Capture-unique dedup: keep at most one inlier per capture (nearest).
   · Filter B: drop inliers with camera > max_obj_dist from fitted P.
6. Write one CSV row per confirmed physical object.

Input files
-----------
Mode A  point_path : image_coords.csv — columns:
            point_name, image_name, cls, x[px], y[px], angular_w

Mode B  point_path : raw_coords.csv — columns:
            point_name, image_name, cls, cam_id, img_w, img_h,
            raw_x[px], raw_y[px], angular_w

eop_path (both modes) : GET EOP CSV (tab-separated) — columns:
    gps_seconds[s]  panorama_file_name  latitude[deg]  longitude[deg]
    altitude_ellipsoidal[m]  roll[deg]  pitch[deg]  heading[deg]

Output
------
output/egsa87/<stem>_associated_EGSA87.csv
Columns: object_id, cls, X_egsa87, Y_egsa87, Z_egsa87,
         n_obs, residual_m, image_ids, detection_ids
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from pyproj import Transformer

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

from forward_intersection import (
    _rotation_egsa,
    _closest_point_n_rays,
    _ray_dist,
)

# ============================================================
# MODULE-LEVEL CONSTANTS
# ============================================================

_BASE      = os.path.normpath(os.path.join(_LIB_DIR, '..', '..'))
_to_egsa87 = Transformer.from_crs("EPSG:4326", "EPSG:2100", always_xy=True)

#: Canonical class-name mapping applied before graph building.
DEFAULT_CLASS_ALIASES: dict[str, str] = {
    "Stop":      "stop_sign",   # Traffic_Sign model
    "stop sign": "stop_sign",   # COCO model (class index 11)
}

#: Real-world size ranges (metres) per canonical class.
#: Used by the depth prior (criterion 5 of _check_edge) and Filter A.
OBJECT_SIZES: dict[str, dict] = {
    "traffic light": {"min_w": 0.15, "max_w": 0.36},
    "traffic sign":  {"min_w": 0.45, "max_w": 1.20},
    "stop_sign":     {"min_w": 0.45, "max_w": 1.20},
}

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class DetectionNode:
    """One per-capture observation — one node in the association graph."""

    detection_id : str                         # unique node identifier
    image_id     : str                         # capture key (panorama_file_name)
    cls          : str                         # canonical class name
    u            : float                       # x pixel (panoramic or raw)
    v            : float                       # y pixel
    ray_origin   : np.ndarray                  # [X, Y, Z] EGSA87 camera centre
    ray_dir      : np.ndarray                  # unit direction in EGSA87
    station_pos  : np.ndarray                  # = ray_origin (for KDTree)
    angular_w    : float = 0.0                 # bbox angular width in radians
    appearance   : Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class _RawDet:
    """
    Internal: one raw per-camera detection before the intra-capture merge.
    Used only by Mode B / per_camera.
    """
    det_id      : str           # point_name from raw_coords.csv
    image_name  : str           # e.g. 'pano_0001_0042_Cam2.jpg'
    capture_key : str           # e.g. 'pano_0001_0042'
    cls         : str
    cam_id      : int
    img_w       : int
    img_h       : int
    raw_x       : float
    raw_y       : float
    angular_w   : float
    ray_rig     : np.ndarray    # unit ray in Ladybug (rig) frame
    ray_egsa    : np.ndarray    # unit ray in EGSA87 frame
    origin      : np.ndarray    # vehicle position in EGSA87
    u_rect      : float         # rectified x (cached for distance check in B1)
    v_rect      : float         # rectified y


# ============================================================
# INTERNAL HELPERS — shared
# ============================================================

def _parse_class(
    point_name  : str,
    class_names : list[str],
    aliases     : dict[str, str],
) -> str:
    """Extract canonical class label from a point_name string (fallback)."""
    for cls in sorted(class_names, key=len, reverse=True):
        if re.search(rf'_{re.escape(cls)}_\d+$', point_name):
            return aliases.get(cls, cls)
    return 'unknown'


# ============================================================
# NODE LOADING — Mode A ('panorama')
# ============================================================

def _load_nodes_panorama(
    point_path   : str,
    eop_path     : str,
    aliases      : dict[str, str],
    W            : int = 8000,
    H            : int = 4000,
    object_sizes : Optional[dict] = None,
    max_obj_dist : float = 20.0,
) -> list[DetectionNode]:
    """
    Parse image_coords.csv (equirectangular) and the EOP file, build a ray
    for every detection, and return DetectionNode objects.

    **Filter A**: if angular_w > 0 and cls is in object_sizes, the minimum
    plausible depth is ``min_w / angular_w``.  Nodes where that exceeds
    *max_obj_dist* are discarded before graph construction.
    """
    with open(point_path, newline='') as f:
        points = list(csv.DictReader(f, delimiter=','))
    with open(eop_path, newline='') as f:
        eop_rows = list(csv.DictReader(f, delimiter='\t'))

    eop_index = {row['panorama_file_name']: row for row in eop_rows}

    nodes: list[DetectionNode] = []
    filtered_a = 0

    for po in points:
        image_id = po['image_name']
        eop = eop_index.get(image_id)
        if eop is None:
            continue

        Xgr, Ygr = _to_egsa87.transform(
            float(eop['longitude[deg]']),
            float(eop['latitude[deg]']),
        )
        h     = float(eop['altitude_ellipsoidal[m]'])
        origin = np.array([Xgr, Ygr, h])
        R_egsa = _rotation_egsa(
            float(eop['roll[deg]']),
            float(eop['pitch[deg]']),
            float(eop['heading[deg]']),
        )

        u = float(po['x[px]'])
        v = float(po['y[px]'])
        if not (0 <= u < W and 0 <= v < H):
            continue

        theta = (u / W - 0.5) * 2.0 * np.pi
        elev  = (0.5 - v / H) * np.pi
        r_cam = np.array([
            np.cos(elev) * np.cos(theta),
            np.cos(elev) * np.sin(theta),
            np.sin(elev),
        ])
        ray_dir = R_egsa @ r_cam
        norm = np.linalg.norm(ray_dir)
        if norm < 1e-12:
            continue
        ray_dir /= norm

        raw_cls = po.get('cls') or _parse_class(po['point_name'], [], aliases)
        cls     = aliases.get(raw_cls, raw_cls)
        angular_w = float(po.get('angular_w') or 0.0)

        # Filter A
        if object_sizes and angular_w > 0:
            sizes = object_sizes.get(cls)
            if sizes and sizes['min_w'] / angular_w > max_obj_dist:
                filtered_a += 1
                continue

        nodes.append(DetectionNode(
            detection_id = po['point_name'],
            image_id     = image_id,
            cls          = cls,
            u            = u,
            v            = v,
            ray_origin   = origin,
            ray_dir      = ray_dir,
            station_pos  = origin.copy(),
            angular_w    = angular_w,
        ))

    if filtered_a:
        print(f"Filter A removed {filtered_a} node(s) (object too small/far).")
    return nodes


# ============================================================
# NODE LOADING — Mode B ('per_camera')
# ============================================================

def _reproject_to_rectified_j(
    det_i    : _RawDet,
    cam_j    : int,
    img_w_j  : int,
    img_h_j  : int,
    transformer,
    _cam_rot,
) -> Optional[tuple[float, float]]:
    """
    Project the direction of *det_i* into the rectified image plane of
    camera *cam_j* via a depth-free rotation (valid because the Ladybug
    baseline is only a few centimetres).

    Returns (u_rect_j, v_rect_j) or None if the ray points behind cam_j.
    Reuses transformer.eop / transformer.iop — geometry not rewritten.
    """
    eop_j = transformer.eop.get(cam_j)
    iop_j = transformer.iop.get(cam_j)
    if eop_j is None or iop_j is None:
        return None

    # Ladybug frame → camera j frame  (inverse of CamToLadybug rotation)
    R_j       = _cam_rot(eop_j['Rx'], eop_j['Ry'], eop_j['Rz'],
                         eop_j['Tx'], eop_j['Ty'], eop_j['Tz'])[:3, :3]
    ray_cam_j = R_j.T @ det_i.ray_rig

    if ray_cam_j[2] <= 0:
        return None   # behind camera j

    fx_j = iop_j['fx_norm'] * img_w_j
    fy_j = iop_j['fy_norm'] * img_h_j
    cx_j = iop_j['cx_norm'] * img_w_j
    cy_j = iop_j['cy_norm'] * img_h_j

    u_j = ray_cam_j[0] / ray_cam_j[2] * fx_j + cx_j
    v_j = ray_cam_j[1] / ray_cam_j[2] * fy_j + cy_j
    return u_j, v_j


def _intra_capture_merge(
    raw_dets    : list[_RawDet],
    transformer,
    tol_px      : float,
    _cam_rot,
) -> list[DetectionNode]:
    """
    B1+B2: For each (capture_key, cls) group build an intra-capture graph
    via pixel reprojection, extract connected components, and return one
    DetectionNode per component (representative EGSA87 ray = mean of
    per-camera unit vectors, renormalised).

    Edge rule: reproject det_i into camera j's rectified plane; add edge
    if predicted pixel is within image bounds AND within *tol_px* of det_j's
    actual rectified position.  If prediction falls outside bounds → no
    conclusion (not a cannot-link).  Same image_name → hard cannot-link.
    """
    # Group by (capture_key, cls)
    groups: dict = defaultdict(list)
    for det in raw_dets:
        groups[(det.capture_key, det.cls)].append(det)

    result_nodes: list[DetectionNode] = []

    for (cap_key, cls), dets in groups.items():
        # Build intra-capture graph
        G_intra = nx.Graph()
        for k in range(len(dets)):
            G_intra.add_node(k)

        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                di, dj = dets[i], dets[j]
                if di.image_name == dj.image_name:  # hard cannot-link
                    continue

                # Reproject di into camera j
                pred = _reproject_to_rectified_j(
                    di, dj.cam_id, dj.img_w, dj.img_h, transformer, _cam_rot)

                if pred is None:
                    continue   # behind cam j — no conclusion

                u_pred, v_pred = pred
                if not (0 <= u_pred < dj.img_w and 0 <= v_pred < dj.img_h):
                    continue   # outside FOV — no conclusion

                # Compare to dj's rectified position
                dist = np.hypot(u_pred - dj.u_rect, v_pred - dj.v_rect)
                if dist < tol_px:
                    G_intra.add_edge(i, j)

        # One DetectionNode per connected component
        for comp_idx, comp in enumerate(nx.connected_components(G_intra)):
            comp_dets = [dets[k] for k in comp]

            # B2: representative ray = mean of unit EGSA87 vectors, renormalised
            rays = np.array([d.ray_egsa for d in comp_dets])
            mean_ray = rays.mean(axis=0)
            norm = np.linalg.norm(mean_ray)
            if norm < 1e-12:
                continue
            mean_ray /= norm

            # Mean angular_w over detections with valid values
            ang_ws = [d.angular_w for d in comp_dets if d.angular_w > 0]
            mean_angular_w = float(np.mean(ang_ws)) if ang_ws else 0.0

            origin = comp_dets[0].origin  # same vehicle position for whole capture
            det_id = f"{cap_key}_{cls}_{comp_idx:04d}"

            result_nodes.append(DetectionNode(
                detection_id = det_id,
                image_id     = cap_key,
                cls          = cls,
                u            = comp_dets[0].raw_x,
                v            = comp_dets[0].raw_y,
                ray_origin   = origin,
                ray_dir      = mean_ray,
                station_pos  = origin.copy(),
                angular_w    = mean_angular_w,
            ))

    return result_nodes


def _load_nodes_per_camera(
    point_path   : str,
    eop_path     : str,
    aliases      : dict[str, str],
    object_sizes : Optional[dict],
    max_obj_dist : float,
    transformer,
    tol_px       : float,
) -> list[DetectionNode]:
    """
    Parse raw_coords.csv and the EOP file for Mode B.

    Steps:
    B0 — extract capture_key from each raw image_name via
         ``re.sub(r'_Cam\\d+.*$', '', image_name)``.
    B1 — intra-capture merge: same-class detections from different cameras
         within the same capture are linked by reprojection (see
         :func:`_intra_capture_merge`).
    B2 — representative EGSA87 ray per merged group (mean + renormalise).

    **Filter A** is applied to individual raw detections before B1.

    Parameters
    ----------
    point_path   : path to raw_coords.csv from run_detection(mode='B')
    eop_path     : path to GET EOP CSV (tab-separated)
    aliases      : canonical class-name mapping
    object_sizes : real-world size ranges per class (for Filter A)
    max_obj_dist : maximum plausible object distance from camera (m)
    transformer  : RawLadybugTransformer instance (from raw_to_panorama)
    tol_px       : reprojection tolerance in pixels for intra-capture edge;
                   use a slightly looser value for nearby objects where the
                   small (centimetre-level) Ladybug baseline produces a
                   non-negligible residual parallax
    """
    # Lazy import — keep Mode A dependency-free from raw_to_panorama
    from raw_to_panorama import _get_T_ZYX as _cam_rot

    with open(point_path, newline='') as f:
        rows = list(csv.DictReader(f))
    with open(eop_path, newline='') as f:
        eop_rows = list(csv.DictReader(f, delimiter='\t'))

    eop_index = {row['panorama_file_name']: row for row in eop_rows}

    raw_dets: list[_RawDet] = []
    filtered_a = 0

    for row in rows:
        image_name  = row['image_name']
        capture_key = re.sub(r'_Cam\d+.*$', '', image_name)

        eop = eop_index.get(capture_key)
        if eop is None:
            continue

        cam_id = int(row['cam_id'])
        if cam_id not in transformer.iop or cam_id not in transformer.eop:
            continue

        img_w = int(row['img_w'])
        img_h = int(row['img_h'])
        raw_x = float(row['raw_x[px]'])
        raw_y = float(row['raw_y[px]'])
        angular_w = float(row.get('angular_w') or 0.0)

        # Vehicle position + orientation in EGSA87
        Xgr, Ygr = _to_egsa87.transform(
            float(eop['longitude[deg]']),
            float(eop['latitude[deg]']),
        )
        h      = float(eop['altitude_ellipsoidal[m]'])
        origin = np.array([Xgr, Ygr, h])
        R_egsa = _rotation_egsa(
            float(eop['roll[deg]']),
            float(eop['pitch[deg]']),
            float(eop['heading[deg]']),
        )

        # Raw pixel → rectified (DistortedSpline; reuses transformer)
        try:
            u_rect, v_rect = transformer.raw_pixel_to_rectified(
                raw_x, raw_y, cam_id, img_w, img_h)
        except Exception:
            continue

        # Rectified → ray in camera frame (pinhole)
        iop = transformer.iop[cam_id]
        fx  = iop['fx_norm'] * img_w
        fy  = iop['fy_norm'] * img_h
        cx  = iop['cx_norm'] * img_w
        cy  = iop['cy_norm'] * img_h
        ray_cam = np.array([(u_rect - cx) / fx, (v_rect - cy) / fy, 1.0])

        # Camera → Ladybug (rig) frame  (CamToLadybugEulerZYX)
        eop_cam = transformer.eop[cam_id]
        R_cam   = _cam_rot(
            eop_cam['Rx'], eop_cam['Ry'], eop_cam['Rz'],
            eop_cam['Tx'], eop_cam['Ty'], eop_cam['Tz'],
        )[:3, :3]
        ray_rig = R_cam @ ray_cam
        r = np.linalg.norm(ray_rig)
        if r < 1e-12:
            continue
        ray_rig = ray_rig / r

        # Ladybug → EGSA87
        ray_egsa = R_egsa @ ray_rig
        r = np.linalg.norm(ray_egsa)
        if r < 1e-12:
            continue
        ray_egsa = ray_egsa / r

        raw_cls = row.get('cls') or _parse_class(row['point_name'], [], aliases)
        cls     = aliases.get(raw_cls, raw_cls)

        # Filter A — angular size plausibility
        if object_sizes and angular_w > 0:
            sizes = object_sizes.get(cls)
            if sizes and sizes['min_w'] / angular_w > max_obj_dist:
                filtered_a += 1
                continue

        raw_dets.append(_RawDet(
            det_id      = row['point_name'],
            image_name  = image_name,
            capture_key = capture_key,
            cls         = cls,
            cam_id      = cam_id,
            img_w       = img_w,
            img_h       = img_h,
            raw_x       = raw_x,
            raw_y       = raw_y,
            angular_w   = angular_w,
            ray_rig     = ray_rig,
            ray_egsa    = ray_egsa,
            origin      = origin,
            u_rect      = float(u_rect),
            v_rect      = float(v_rect),
        ))

    if filtered_a:
        print(f"Filter A removed {filtered_a} raw detection(s) (object too small/far).")

    print(f"Raw per-camera detections after Filter A: {len(raw_dets)}")

    # B1 + B2: intra-capture merge → per-capture DetectionNodes
    nodes = _intra_capture_merge(raw_dets, transformer, tol_px, _cam_rot)
    print(f"Per-capture observation nodes after intra-capture merge: {len(nodes)}")
    return nodes


# ============================================================
# INTERNAL HELPERS — common core
# ============================================================

def _candidate_pairs(
    nodes    : list[DetectionNode],
    window_m : float,
) -> list[tuple[int, int]]:
    """
    Return (i, j) index pairs of nodes from **different captures** whose
    camera centres are within *window_m* metres.  O(n log n) via cKDTree.
    """
    positions = np.array([n.station_pos[:2] for n in nodes])
    tree  = cKDTree(positions)
    pairs: list[tuple[int, int]] = []

    for i in range(len(nodes)):
        for j in tree.query_ball_point(positions[i], window_m):
            if j <= i:
                continue
            if nodes[i].image_id != nodes[j].image_id:
                pairs.append((i, j))

    return pairs


def _check_edge(
    a                 : DetectionNode,
    b                 : DetectionNode,
    ray_dist_tol      : float,
    max_obj_dist      : float,
    appearance_weight : float = 0.0,
    object_sizes      : Optional[dict] = None,
    depth_tolerance   : float = 2.0,
) -> Optional[float]:
    """
    Evaluate whether two nodes should be linked by a graph edge.

    Five criteria must all pass:
    1. Same canonical class.
    2. Average ray-distance of candidate intersection < *ray_dist_tol*.
    3. Positive depth from both cameras.
    4. Intersection within *max_obj_dist* of both camera centres.
    5. Depth prior from bbox angular width:
       D ∈ [min_w / angular_w / tol,  max_w / angular_w * tol].

    Returns exp(-ray_dist) edge weight, or None on failure.
    """
    if a.cls != b.cls:
        return None

    P = _closest_point_n_rays(
        np.array([a.ray_origin, b.ray_origin]),
        np.array([a.ray_dir,    b.ray_dir]),
    )

    dist_a = _ray_dist(P, a.ray_origin, a.ray_dir)
    dist_b = _ray_dist(P, b.ray_origin, b.ray_dir)
    ray_dist_val = (dist_a + dist_b) * 0.5
    if ray_dist_val > ray_dist_tol:
        return None

    if np.dot(P - a.ray_origin, a.ray_dir) <= 0.0:
        return None
    if np.dot(P - b.ray_origin, b.ray_dir) <= 0.0:
        return None

    if np.linalg.norm(P - a.station_pos) > max_obj_dist:
        return None
    if np.linalg.norm(P - b.station_pos) > max_obj_dist:
        return None

    if object_sizes:
        for node in (a, b):
            if node.angular_w <= 0:
                continue
            sizes = object_sizes.get(node.cls)
            if sizes is None:
                continue
            D_min = sizes['min_w'] / node.angular_w / depth_tolerance
            D_max = sizes['max_w'] / node.angular_w * depth_tolerance
            D     = np.linalg.norm(P - node.ray_origin)
            if not (D_min < D < D_max):
                return None

    geom_w = float(np.exp(-ray_dist_val))
    if (appearance_weight > 0.0
            and a.appearance is not None
            and b.appearance is not None):
        cos_sim = float(
            np.dot(a.appearance, b.appearance)
            / (np.linalg.norm(a.appearance) * np.linalg.norm(b.appearance) + 1e-12)
        )
        app_sim = (cos_sim + 1.0) * 0.5
        return (1.0 - appearance_weight) * geom_w + appearance_weight * app_sim

    return geom_w


def _build_graph(
    nodes             : list[DetectionNode],
    pairs             : list[tuple[int, int]],
    ray_dist_tol      : float,
    max_obj_dist      : float,
    appearance_weight : float,
    object_sizes      : Optional[dict] = None,
) -> nx.Graph:
    """Build the inter-capture association graph."""
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.detection_id, node_obj=node)

    for i, j in pairs:
        w = _check_edge(nodes[i], nodes[j],
                        ray_dist_tol, max_obj_dist, appearance_weight,
                        object_sizes)
        if w is not None:
            G.add_edge(nodes[i].detection_id, nodes[j].detection_id, weight=w)

    return G


def _ransac_group(
    group_nodes  : list[DetectionNode],
    ransac_tol   : float,
    min_inliers  : int,
    max_obj_dist : float,
    n_iter       : int = 100,
) -> list[dict]:
    """
    RANSAC + weighted least-squares on a single connected component.

    Algorithm
    ---------
    1. Sample 2 rays, compute candidate intersection.
    2. Count inliers (ray_dist < ransac_tol).
    3. Keep hypothesis with most inliers.
    4. **Capture-unique dedup**: keep at most one inlier per capture key
       (``image_id``); when duplicates exist, retain the one closest to the
       hypothesis point.  This deterministically resolves the rare case where
       two nodes from the same capture end up in the same component.
    5. **Filter B**: discard inliers with camera distance > max_obj_dist from
       the hypothesis.  If fewer than min_inliers survive, reject the group.
    6. Final refit via scipy.optimize.least_squares on surviving inliers.

    Returns at most one result dict (non-recursive).
    """
    N = len(group_nodes)
    if N < min_inliers:
        return []

    origins    = np.array([n.ray_origin for n in group_nodes])
    directions = np.array([n.ray_dir    for n in group_nodes])

    rng = np.random.default_rng(0)
    best_mask  = np.zeros(N, dtype=bool)
    best_P_hyp = np.zeros(3)

    for _ in range(n_iter):
        idx   = rng.choice(N, size=2, replace=False)
        P_try = _closest_point_n_rays(origins[idx], directions[idx])
        res   = np.array([_ray_dist(P_try, origins[k], directions[k])
                          for k in range(N)])
        mask  = res < ransac_tol
        if mask.sum() > best_mask.sum():
            best_mask  = mask
            best_P_hyp = P_try

    if best_mask.sum() < min_inliers:
        return []

    inlier_idx = np.where(best_mask)[0]

    # Step 4 — capture-unique dedup: one inlier per image_id
    capture_best: dict[str, tuple[int, float]] = {}   # image_id → (node_idx, dist)
    for k in inlier_idx:
        ckey = group_nodes[k].image_id
        d    = _ray_dist(best_P_hyp, origins[k], directions[k])
        if ckey not in capture_best or d < capture_best[ckey][1]:
            capture_best[ckey] = (k, d)
    inlier_idx = np.array([v[0] for v in capture_best.values()])

    # Step 5 — Filter B: drop cameras too far from hypothesis
    near = np.array([
        np.linalg.norm(origins[k] - best_P_hyp) < max_obj_dist
        for k in inlier_idx
    ])
    inlier_idx = inlier_idx[near]

    if len(inlier_idx) < min_inliers:
        return []

    # Step 6 — final LSQ refit
    inlier_o = origins[inlier_idx]
    inlier_d = directions[inlier_idx]
    P0       = _closest_point_n_rays(inlier_o, inlier_d)

    def _res(P: np.ndarray) -> np.ndarray:
        return np.array([_ray_dist(P, inlier_o[k], inlier_d[k])
                         for k in range(len(inlier_idx))])

    if len(inlier_idx) >= 3:
        fit     = least_squares(_res, P0, method='trf')
        P_final = fit.x
    else:
        P_final = P0

    res_vals     = _res(P_final)
    inlier_nodes = [group_nodes[k] for k in inlier_idx]

    return [{
        'X':             float(P_final[0]),
        'Y':             float(P_final[1]),
        'Z':             float(P_final[2]),
        'residual_m':    float(np.mean(res_vals)),
        'n_obs':         len(inlier_idx),
        'cls':           group_nodes[0].cls,
        'image_ids':     [n.image_id     for n in inlier_nodes],
        'detection_ids': [n.detection_id for n in inlier_nodes],
    }]


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def associate(
    point_path        : str,
    eop_path          : str,
    mode              : str   = 'panorama',
    W                 : int   = 8000,
    H                 : int   = 4000,
    window_m          : float = 10.0,
    ray_dist_tol      : float = 0.15,
    max_obj_dist      : float = 20.0,
    min_obs           : int   = 2,
    ransac_tol        : float = 0.15,
    appearance_weight : float = 0.2,
    class_aliases     : Optional[dict[str, str]] = None,
    object_sizes      : Optional[dict] = None,
    tol_px            : float = 10.0,
    cal_file          : Optional[str] = None,
    output_path       : Optional[str] = None,
) -> Optional[str]:
    """
    Full offline multi-view association pipeline.

    Steps
    -----
    1. Load detections + EOP → DetectionNode list (mode-specific; Filter A
       removes nodes whose angular size implies depth > max_obj_dist).
       Mode B additionally performs intra-capture merge before this step.
    2. Candidate pairs via cKDTree — O(n log n).
    3. Association graph with 5-criterion edge filter.
    4. Connected components — zero-overlap groups guaranteed.
    5. RANSAC + scipy.optimize.least_squares per group.
       Capture-unique dedup and Filter B applied inside RANSAC.
    6. Write one CSV row per confirmed physical object.

    Parameters
    ----------
    point_path        : path to image_coords.csv (mode='panorama') or
                        raw_coords.csv (mode='per_camera')
    eop_path          : path to GET EOP CSV (tab-separated)
    mode              : 'panorama' (Mode A) or 'per_camera' (Mode B)
    W, H              : panorama dimensions — used only for mode='panorama'
    window_m          : cKDTree search radius for candidate pairs (m)
    ray_dist_tol      : max average ray-distance for a graph edge (m)
    max_obj_dist      : max object distance from camera (m); used by
                        Filter A, Filter B, and edge criterion 4
    min_obs           : minimum inlier observations per confirmed object
    ransac_tol        : RANSAC inlier threshold (m)
    appearance_weight : blend factor for appearance similarity; 0 = geometry
    class_aliases     : override DEFAULT_CLASS_ALIASES
    object_sizes      : override OBJECT_SIZES
    tol_px            : reprojection tolerance (px) for intra-capture edge
                        in mode='per_camera'.  Increase slightly for nearby
                        objects where the centimetre-level Ladybug baseline
                        produces a small but non-zero residual parallax.
    cal_file          : path to .cal calibration file — required for
                        mode='per_camera'
    output_path       : explicit output CSV path; auto-generated if None

    Returns
    -------
    Absolute path to output CSV, or None if no objects were confirmed.
    """
    if class_aliases is None:
        class_aliases = DEFAULT_CLASS_ALIASES
    if object_sizes is None:
        object_sizes = OBJECT_SIZES

    # ── 1. Load nodes ────────────────────────────────────────────────────────
    if mode == 'per_camera':
        if cal_file is None:
            raise ValueError("cal_file is required for mode='per_camera'")
        from raw_to_panorama import RawLadybugTransformer
        transformer = RawLadybugTransformer(cal_file)
        nodes = _load_nodes_per_camera(
            point_path, eop_path, class_aliases,
            object_sizes, max_obj_dist, transformer, tol_px,
        )
    else:
        nodes = _load_nodes_panorama(
            point_path, eop_path, class_aliases,
            W, H, object_sizes, max_obj_dist,
        )

    print(f"Loaded {len(nodes)} observation nodes.")
    if len(nodes) < 2:
        print("Not enough observations for association.")
        return None

    # ── 2. Candidate pairs ───────────────────────────────────────────────────
    pairs = _candidate_pairs(nodes, window_m)
    print(f"Candidate pairs (window={window_m} m): {len(pairs)}")

    # ── 3. Association graph ─────────────────────────────────────────────────
    G = _build_graph(nodes, pairs, ray_dist_tol, max_obj_dist,
                     appearance_weight, object_sizes)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── 4. Connected components ──────────────────────────────────────────────
    groups = [list(c) for c in nx.connected_components(G) if len(c) >= min_obs]
    print(f"Groups (connected components, min_obs={min_obs}): {len(groups)}")

    # ── 5. RANSAC per group ──────────────────────────────────────────────────
    node_by_id  = {n.detection_id: n for n in nodes}
    all_results : list[dict] = []

    for group in groups:
        group_nodes = [node_by_id[did] for did in group]
        all_results.extend(
            _ransac_group(group_nodes, ransac_tol, min_obs, max_obj_dist))

    for i, res in enumerate(all_results):
        res['object_id'] = f"obj_{i:04d}"

    print(f"Confirmed objects after RANSAC: {len(all_results)}")

    if not all_results:
        print("No objects confirmed. "
              "Try relaxing ray_dist_tol, window_m, or reducing min_obs.")
        return None

    # ── 6. Write output ──────────────────────────────────────────────────────
    if output_path is None:
        out_dir = os.path.join(_BASE, 'output', 'egsa87')
        os.makedirs(out_dir, exist_ok=True)
        stem        = os.path.splitext(os.path.basename(point_path))[0]
        output_path = os.path.join(out_dir, stem + '_associated_EGSA87.csv')

    fieldnames = [
        'object_id', 'cls',
        'X_egsa87', 'Y_egsa87', 'Z_egsa87',
        'n_obs', 'residual_m',
        'image_ids', 'detection_ids',
    ]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                'object_id':     r['object_id'],
                'cls':           r['cls'],
                'X_egsa87':      f"{r['X']:.3f}",
                'Y_egsa87':      f"{r['Y']:.3f}",
                'Z_egsa87':      f"{r['Z']:.3f}",
                'n_obs':         r['n_obs'],
                'residual_m':    f"{r['residual_m']:.4f}",
                'image_ids':     '|'.join(r['image_ids']),
                'detection_ids': '|'.join(r['detection_ids']),
            })

    print(f"\n{len(all_results)} object(s) saved → {output_path}")
    return output_path


# ============================================================
# STANDALONE / EXAMPLE
# ============================================================

if __name__ == '__main__':
    """
    Mode A ('panorama')
    -------------------
    image_coords.csv + EOP
          │
          ▼
    _load_nodes_panorama()   build DetectionNode list from equirectangular
    Filter A                 discard nodes with D_min_plausible > max_obj_dist
          │
          ▼  ──────────────── common core ────────────────────────────
          ▼
    _candidate_pairs()       O(n log n) cKDTree on EGSA87 (X, Y)
          │
          ▼
    _build_graph()           5-criterion edge filter:
                               same class · ray dist · positive depth
                               max object dist · depth prior (angular size)
          │
          ▼
    connected_components()   zero-overlap groups; each node in exactly one
          │
          ▼
    _ransac_group()          RANSAC + least_squares; capture-unique dedup;
    Filter B                 drop inliers with camera > max_obj_dist from P
          │
          ▼
    output CSV               one row per confirmed object (EGSA87 + metadata)

    Mode B ('per_camera')
    ---------------------
    raw_coords.csv + EOP + .cal file
          │
          ▼
    _load_nodes_per_camera()
      B0  extract capture_key  re.sub(r'_Cam\\d+.*$', '', image_name)
      Filter A                 discard individual raw detections too small
      B1  intra-capture merge  reproject each detection into other cameras;
                               link pairs within tol_px; connected components
                               → one observation per physical object per capture
      B2  representative ray   mean of EGSA87 unit vectors, renormalised
          │
          ▼  ──────────────── common core (same as Mode A) ────────────
    """

    _default_coords = os.path.join(_BASE, 'output', 'coords', 'raw_coords.csv')
    _default_eop    = os.path.join(_BASE, 'data', '')

    _point_path = (input(f'Coords CSV [{_default_coords}]: ').strip()
                   or _default_coords)
    _eop_path   = (input(f'EOP CSV    [{_default_eop}]: ').strip()
                   or _default_eop)
    _mode       = (input('Mode [panorama / per_camera]: ').strip()
                   or 'panorama')

    _default_cal = os.path.join(_BASE, 'data', 'Ladybug5_plus', 'ladybug20344317.cal')
    if _mode == 'per_camera':
        _cal = (input(f'Cal file [{_default_cal}]: ').strip() 
                or _default_cal)

    associate(
        point_path = _point_path,
        eop_path   = _eop_path,
        mode       = _mode,
        cal_file   = _cal,
    )
