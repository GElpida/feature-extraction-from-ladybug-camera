"""
Point association: group multi-view detections of the same physical object
and compute EGSA87 coordinates via forward intersection.

Input:
  image_coords.csv  — panoramic pixel coordinates (from Detectron / centroid)
  EOP CSV           — GET EOP (tab-separated)

Output:
  output/egsa87/<stem>_associated_EGSA87.csv

Mode A: image_coords.csv already has panoramic coordinates — pass directly.
Mode B: convert raw_coords.csv to panoramic first with raw_to_panorama.py,
        then pass the resulting image_coords.csv here.
"""

import os
import csv
import numpy as np
import networkx as nx
from dataclasses import dataclass
from scipy.spatial import cKDTree
from typing import Optional
from pyproj import Transformer

from forward_intersection import _rotation_egsa, _closest_point_n_rays, _ray_dist

_BASE      = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_to_egsa87 = Transformer.from_crs("EPSG:4326", "EPSG:2100", always_xy=True)

DEFAULT_CLASS_ALIASES = {
    "Stop":      "stop_sign",
    "stop sign": "stop_sign",
}


# ============================================================
# DATA
# ============================================================

@dataclass
class DetectionNode:
    detection_id : str
    image_id     : str          # panorama_file_name
    cls          : str
    ray_origin   : np.ndarray   # EGSA87 camera centre [X, Y, Z]
    ray_dir      : np.ndarray   # unit direction in EGSA87
    station_pos  : np.ndarray   # = ray_origin (for KDTree)


# ============================================================
# NODE LOADING
# ============================================================

def _load_nodes(
    point_path : str,
    eop_path   : str,
    aliases    : dict,
    W          : int = 8000,
    H          : int = 4000,
) -> list:
    with open(point_path, newline='') as f:
        points = list(csv.DictReader(f))
    with open(eop_path, newline='') as f:
        eop_rows = list(csv.DictReader(f, delimiter='\t'))

    eop_index = {row['panorama_file_name']: row for row in eop_rows}
    nodes = []

    for po in points:
        image_id = po['image_name']
        eop = eop_index.get(image_id)
        if eop is None:
            continue

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

        raw_cls = po.get('cls', '')
        cls = aliases.get(raw_cls, raw_cls)

        nodes.append(DetectionNode(
            detection_id = po['point_name'],
            image_id     = image_id,
            cls          = cls,
            ray_origin   = origin,
            ray_dir      = ray_dir,
            station_pos  = origin.copy(),
        ))

    return nodes


# ============================================================
# CANDIDATE PAIRS
# ============================================================

def _candidate_pairs(nodes: list, window_m: float) -> list:
    positions = np.array([n.station_pos[:2] for n in nodes])
    tree  = cKDTree(positions)
    pairs = []
    for i in range(len(nodes)):
        for j in tree.query_ball_point(positions[i], window_m):
            if j <= i:
                continue
            if nodes[i].image_id != nodes[j].image_id:
                pairs.append((i, j))
    return pairs


# ============================================================
# EDGE FILTER
# ============================================================

def _check_edge(
    a            : DetectionNode,
    b            : DetectionNode,
    ray_dist_tol : float,
    max_obj_dist : float,
) -> Optional[float]:
    """
    Four criteria must all pass:
    1. Same class.
    2. Average ray-distance to candidate intersection <= ray_dist_tol.
    3. Positive depth from both cameras.
    4. Intersection within max_obj_dist of both camera centres.

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
    if (dist_a + dist_b) * 0.5 > ray_dist_tol:
        return None

    if np.dot(P - a.ray_origin, a.ray_dir) <= 0.0:
        return None
    if np.dot(P - b.ray_origin, b.ray_dir) <= 0.0:
        return None

    if np.linalg.norm(P - a.station_pos) > max_obj_dist:
        return None
    if np.linalg.norm(P - b.station_pos) > max_obj_dist:
        return None

    return float(np.exp(-(dist_a + dist_b) * 0.5))


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def associate(
    point_path   : str,
    eop_path     : str,
    W            : int   = 8000,
    H            : int   = 4000,
    window_m     : float = 10.0,
    ray_dist_tol : float = 0.15,
    max_obj_dist : float = 20.0,
    min_obs      : int   = 2,
    class_aliases: Optional[dict] = None,
    output_path  : Optional[str]  = None,
) -> Optional[str]:
    """
    Full offline multi-view association pipeline.

    Steps
    -----
    1. Load detections + EOP -> DetectionNode list with EGSA87 rays.
    2. Candidate pairs via cKDTree — O(n log n).
    3. Association graph with 4-criterion edge filter.
    4. Connected components — zero-overlap groups guaranteed.
    5. Forward intersection (_closest_point_n_rays WLS) per group.
    6. Write one CSV row per confirmed physical object.

    Parameters
    ----------
    point_path   : path to image_coords.csv (panoramic pixel coordinates)
    eop_path     : path to GET EOP CSV (tab-separated)
    W, H         : panorama dimensions in pixels
    window_m     : cKDTree search radius for candidate pairs (m)
    ray_dist_tol : max average ray-distance to accept a graph edge (m)
    max_obj_dist : max object distance from camera (m)
    min_obs      : minimum observations per confirmed object
    class_aliases: canonical class-name mapping (overrides DEFAULT_CLASS_ALIASES)
    output_path  : explicit output CSV path; auto-generated if None

    Returns
    -------
    Absolute path to output CSV, or None if no objects were confirmed.
    """
    if class_aliases is None:
        class_aliases = DEFAULT_CLASS_ALIASES

    # 1. Load nodes
    nodes = _load_nodes(point_path, eop_path, class_aliases, W, H)
    print(f"Loaded {len(nodes)} observation nodes.")
    if len(nodes) < 2:
        print("Not enough observations for association.")
        return None

    # 2. Candidate pairs
    pairs = _candidate_pairs(nodes, window_m)
    print(f"Candidate pairs (window={window_m} m): {len(pairs)}")

    # 3. Association graph
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.detection_id, node_obj=node)
    for i, j in pairs:
        w = _check_edge(nodes[i], nodes[j], ray_dist_tol, max_obj_dist)
        if w is not None:
            G.add_edge(nodes[i].detection_id, nodes[j].detection_id, weight=w)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 4. Connected components
    node_by_id = {n.detection_id: n for n in nodes}
    groups = [list(c) for c in nx.connected_components(G) if len(c) >= min_obs]
    print(f"Groups (>={min_obs} obs): {len(groups)}")

    # 5. Forward intersection per group
    results = []
    for i, group in enumerate(groups):
        group_nodes = [node_by_id[did] for did in group]
        origins    = np.array([n.ray_origin for n in group_nodes])
        directions = np.array([n.ray_dir    for n in group_nodes])
        P = _closest_point_n_rays(origins, directions)
        residuals = [_ray_dist(P, origins[k], directions[k])
                     for k in range(len(group_nodes))]
        results.append({
            'object_id':     f"obj_{i:04d}",
            'cls':           group_nodes[0].cls,
            'X_egsa87':      f"{P[0]:.3f}",
            'Y_egsa87':      f"{P[1]:.3f}",
            'Z_egsa87':      f"{P[2]:.3f}",
            'n_obs':         len(group_nodes),
            'residual_m':    f"{float(np.mean(residuals)):.4f}",
            'image_ids':     '|'.join(n.image_id     for n in group_nodes),
            'detection_ids': '|'.join(n.detection_id for n in group_nodes),
        })

    print(f"Confirmed objects: {len(results)}")
    if not results:
        print("No objects confirmed. "
              "Try relaxing ray_dist_tol, window_m, or reducing min_obs.")
        return None

    # 6. Write output
    if output_path is None:
        out_dir = os.path.join(_BASE, 'output', 'egsa87')
        os.makedirs(out_dir, exist_ok=True)
        stem        = os.path.splitext(os.path.basename(point_path))[0]
        output_path = os.path.join(out_dir, stem + '_associated_EGSA87.csv')

    fieldnames = ['object_id', 'cls', 'X_egsa87', 'Y_egsa87', 'Z_egsa87',
                  'n_obs', 'residual_m', 'image_ids', 'detection_ids']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Output -> {output_path}")
    return output_path
