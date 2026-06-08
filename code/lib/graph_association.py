"""
Graph-based multi-view association for offline batch detection pipelines.

Groups detections of the same physical object that appear in multiple
panoramic captures into one confirmed 3-D position in EGSA87.  The method
uses a geometry-driven association graph and N-ray RANSAC + weighted
least-squares to eliminate duplicates and improve accuracy.

Designed for:
    - Offline / batch operation: all detections and EOP are known in advance.
    - Mode A input: equirectangular panoramas produced by run_detection().

Can be used as a library
------------------------
    from graph_association import associate

    output_csv = associate(
        point_path  = 'output/coords/image_coords.csv',
        eop_path    = '', # tab-separated csv file with EOP
        class_names = ['Stop', 'Give_way', 'Speed_limit_50KM', ...],
    )

Or run standalone (prompts for paths):
    python graph_association.py

Input files
-----------
point_path
    image_coords.csv produced by run_detection() — columns:
        point_name, image_name, x[px], y[px]

eop_path
    GET EOP CSV (tab-separated) — columns:
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
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from pyproj import Transformer

# ── re-use helpers from forward_intersection — do NOT rewrite ────────────────
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

#: Canonical class-name mapping.
#: Detections from different models that refer to the same physical object
#: class are unified under one canonical name before graph building, so that
#: cross-model edges can be formed.
#:
#: Extend this dict (or pass a custom one to ``associate()``) as needed.
DEFAULT_CLASS_ALIASES: dict[str, str] = {
    "Stop":      "stop_sign",   # Traffic_Sign model
    "stop sign": "stop_sign",   # COCO model (class index 11)
}

# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class DetectionNode:
    """One detection from image_coords.csv — one node in the association graph."""

    detection_id : str                         # point_name (unique per CSV row)
    image_id     : str                         # panorama_file_name
    cls          : str                         # canonical class name
    u            : float                       # x[px] in panoramic image
    v            : float                       # y[px] in panoramic image
    ray_origin   : np.ndarray                  # [X, Y, Z] EGSA87 — camera centre
    ray_dir      : np.ndarray                  # unit direction vector in EGSA87
    station_pos  : np.ndarray                  # = ray_origin (kept separate for KDTree)
    appearance   : Optional[np.ndarray] = field(default=None, repr=False)
    # appearance: optional descriptor (e.g. HOG, CNN embedding) for the
    # detected region.  When provided and appearance_weight > 0, it
    # contributes to the edge weight.  Pass None to use geometry only.


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _parse_class(
    point_name  : str,
    class_names : list[str],
    aliases     : dict[str, str],
) -> str:
    """
    Extract the canonical class label from a *point_name* string.

    *point_name* has the form ``{image_name}_{cls_name}_{i}`` where both
    ``image_name`` and ``cls_name`` may contain underscores (e.g.
    ``Speed_limit_50KM``).  We search for ``_{cls_name}_{integer}$`` at the
    **end** of the string, trying class names from longest to shortest so that
    ``Speed_limit_50KM`` is matched before a hypothetical ``KM`` class.

    Parameters
    ----------
    point_name  : full point_name string from image_coords.csv
    class_names : known raw class names from the Detectron model(s)
    aliases     : mapping raw → canonical name (see DEFAULT_CLASS_ALIASES)

    Returns
    -------
    Canonical class name, or ``'unknown'`` if no class matches.
    """
    for cls in sorted(class_names, key=len, reverse=True):
        if re.search(rf'_{re.escape(cls)}_\d+$', point_name):
            return aliases.get(cls, cls)
    return 'unknown'


def _load_nodes(
    point_path  : str,
    eop_path    : str,
    class_names : list[str],
    aliases     : dict[str, str],
    W           : int = 8000,
    H           : int = 4000,
) -> list[DetectionNode]:
    """
    Parse *image_coords.csv* and the EOP file, build a ray for every
    detection, and return a list of :class:`DetectionNode` objects.

    Detections without an EOP match are silently skipped — consistent with
    the behaviour of :func:`forward_intersection.run_intersection`.

    Parameters
    ----------
    point_path  : path to image_coords.csv from run_detection()
    eop_path    : path to GET EOP CSV (tab-separated)
    class_names : raw class names for point_name parsing
    aliases     : canonical class-name mapping
    W, H        : panorama dimensions in pixels
    """
    with open(point_path, newline='') as f:
        points = list(csv.DictReader(f, delimiter=','))
    with open(eop_path, newline='') as f:
        eop_rows = list(csv.DictReader(f, delimiter='\t'))

    # Index EOP by panorama_file_name for O(1) lookups
    eop_index: dict[str, dict] = {row['panorama_file_name']: row for row in eop_rows}

    nodes: list[DetectionNode] = []

    for po in points:
        image_id = po['image_name']
        eop = eop_index.get(image_id)
        if eop is None:
            continue

        # Camera centre in EGSA87
        Xgr, Ygr = _to_egsa87.transform(
            float(eop['longitude[deg]']),
            float(eop['latitude[deg]']),
        )
        h     = float(eop['altitude_ellipsoidal[m]'])
        roll  = float(eop['roll[deg]'])
        pitch = float(eop['pitch[deg]'])
        head  = float(eop['heading[deg]'])

        origin = np.array([Xgr, Ygr, h])
        R_egsa = _rotation_egsa(roll, pitch, head)   # reused — not rewritten

        # Ray direction from equirectangular pixel coordinates
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
        ray_dir = ray_dir / norm

        cls = _parse_class(po['point_name'], class_names, aliases)

        nodes.append(DetectionNode(
            detection_id = po['point_name'],
            image_id     = image_id,
            cls          = cls,
            u            = u,
            v            = v,
            ray_origin   = origin,
            ray_dir      = ray_dir,
            station_pos  = origin.copy(),
        ))

    return nodes


def _candidate_pairs(
    nodes    : list[DetectionNode],
    window_m : float,
) -> list[tuple[int, int]]:
    """
    Return index pairs *(i, j)* of detections from **different images** whose
    camera centres are within *window_m* metres of each other.

    Uses ``scipy.spatial.cKDTree`` on EGSA87 (X, Y) — O(n log n) instead of
    the naive O(n²) all-pairs check.

    Parameters
    ----------
    nodes    : list of DetectionNode
    window_m : spatial search radius in metres (EGSA87 X/Y plane)
    """
    positions = np.array([n.station_pos[:2] for n in nodes])   # (N, 2)
    tree  = cKDTree(positions)
    pairs : list[tuple[int, int]] = []

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
) -> Optional[float]:
    """
    Evaluate whether two detections should be linked by a graph edge.

    All five criteria must pass:

    1. Same canonical class.
    2. Average ray-distance of the candidate intersection below *ray_dist_tol*.
       Uses :func:`_closest_point_n_rays` and :func:`_ray_dist` — both reused
       from ``forward_intersection``, not rewritten here.
    3. Positive depth — candidate intersection is in front of both cameras
       (dot product of ``P - origin`` with ``ray_dir`` must be positive).
    4. Candidate intersection within *max_obj_dist* of both camera centres.
    5. Spatial window — pre-filtered upstream by :func:`_candidate_pairs`.

    Edge weight
    -----------
    ``weight = exp(-ray_dist)``  (geometry only, *appearance_weight* == 0)

    When *appearance_weight* > 0 and both nodes carry appearance descriptors,
    cosine similarity is blended in:
    ``weight = (1 - aw) * geom + aw * app_sim``

    Parameters
    ----------
    a, b              : pair of candidate detection nodes
    ray_dist_tol      : max average ray-distance (m)
    max_obj_dist      : max distance of intersection from each camera centre (m)
    appearance_weight : blend factor for appearance similarity in [0, 1]

    Returns
    -------
    Edge weight (float > 0) if all criteria pass, else ``None``.
    """
    # Criterion 1 — same canonical class
    if a.cls != b.cls:
        return None

    # Candidate intersection via 2-ray weighted LSQ (reused)
    P = _closest_point_n_rays(
        np.array([a.ray_origin, b.ray_origin]),
        np.array([a.ray_dir,    b.ray_dir]),
    )

    # Criterion 2 — ray distance
    dist_a = _ray_dist(P, a.ray_origin, a.ray_dir)
    dist_b = _ray_dist(P, b.ray_origin, b.ray_dir)
    ray_dist_val = (dist_a + dist_b) * 0.5
    if ray_dist_val > ray_dist_tol:
        return None

    # Criterion 3 — positive depth from both cameras
    if np.dot(P - a.ray_origin, a.ray_dir) <= 0.0:
        return None
    if np.dot(P - b.ray_origin, b.ray_dir) <= 0.0:
        return None

    # Criterion 4 — object not unreasonably far from the trajectory
    if np.linalg.norm(P - a.station_pos) > max_obj_dist:
        return None
    if np.linalg.norm(P - b.station_pos) > max_obj_dist:
        return None

    # Edge weight
    geom_w = float(np.exp(-ray_dist_val))
    if (appearance_weight > 0.0
            and a.appearance is not None
            and b.appearance is not None):
        cos_sim = float(
            np.dot(a.appearance, b.appearance)
            / (np.linalg.norm(a.appearance) * np.linalg.norm(b.appearance) + 1e-12)
        )
        app_sim = (cos_sim + 1.0) * 0.5   # normalise to [0, 1]
        return (1.0 - appearance_weight) * geom_w + appearance_weight * app_sim

    return geom_w


def _build_graph(
    nodes             : list[DetectionNode],
    pairs             : list[tuple[int, int]],
    ray_dist_tol      : float,
    max_obj_dist      : float,
    appearance_weight : float,
) -> nx.Graph:
    """
    Build the association graph.

    Every node stores the :class:`DetectionNode` object as the ``'node_obj'``
    attribute.  Every edge stores ``'weight'`` (higher = stronger match).

    Parameters
    ----------
    nodes             : full detection list
    pairs             : candidate (i, j) index pairs from _candidate_pairs()
    ray_dist_tol      : edge criterion — max average ray-distance (m)
    max_obj_dist      : edge criterion — max object distance from camera (m)
    appearance_weight : blend factor for appearance in edge weight
    """
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.detection_id, node_obj=node)

    for i, j in pairs:
        w = _check_edge(nodes[i], nodes[j],
                        ray_dist_tol, max_obj_dist, appearance_weight)
        if w is not None:
            G.add_edge(nodes[i].detection_id, nodes[j].detection_id, weight=w)

    return G


def _cluster(
    G        : nx.Graph,
    strategy : str,
    min_obs  : int,
) -> list[list[str]]:
    """
    Partition the graph into candidate object groups.

    Parameters
    ----------
    G        : association graph from :func:`_build_graph`
    strategy : ``'connected_components'`` or ``'cliques'``
    min_obs  : discard groups with fewer than this many detections

    Chaining risk and why cliques mitigate it
    -----------------------------------------
    With ``connected_components``, a sequence of similar objects A–B–C–D along
    the trajectory can be merged into one component: each adjacent pair might
    pass the edge criteria (A–B, B–C, C–D all valid), even though A and D are
    geometrically incompatible.  The entire chain gets one intersection point,
    which will be wrong.

    With ``cliques``, **every** pair inside a group must be mutually connected.
    A detection from object D cannot join the A–B–C group unless it also forms
    valid edges with A and B.  This breaks chaining at the cost of potentially
    splitting a single true object into several small overlapping cliques —
    handled downstream by RANSAC, which accepts the largest consistent inlier
    set and recursively processes the rest.
    """
    if strategy == 'connected_components':
        return [list(c) for c in nx.connected_components(G) if len(c) >= min_obs]

    if strategy == 'cliques':
        return [c for c in nx.find_cliques(G) if len(c) >= min_obs]

    raise ValueError(
        f"Unknown strategy {strategy!r}. "
        "Use 'connected_components' or 'cliques'."
    )


def _ransac_group(
    group_nodes : list[DetectionNode],
    ransac_tol  : float,
    min_inliers : int,
    n_iter      : int = 100,
) -> list[dict]:
    """
    RANSAC + weighted least-squares on a single detection group.

    Algorithm
    ---------
    1. Sample 2 rays, compute candidate intersection via
       :func:`_closest_point_n_rays`.
    2. Count inliers (rays with :func:`_ray_dist` < *ransac_tol*).
    3. Keep the hypothesis with the most inliers.
    4. Refit with **scipy.optimize.least_squares** on all inliers.
    5. Outlier detections (residual >= *ransac_tol*) are recursively
       processed — they may form a second nearby physical object that was
       incorrectly merged into this group by the clustering step.

    Parameters
    ----------
    group_nodes : detections belonging to this candidate group
    ransac_tol  : inlier threshold in metres
    min_inliers : minimum accepted inlier count (rejects noisy singletons)
    n_iter      : number of RANSAC iterations

    Returns
    -------
    List of result dicts — 0 if rejected, 1 normally, 2+ if the group was
    split.  Each dict contains:
    ``X, Y, Z, residual_m, n_obs, cls, image_ids, detection_ids``.
    """
    N = len(group_nodes)
    if N < min_inliers:
        return []

    origins    = np.array([n.ray_origin for n in group_nodes])   # (N, 3)
    directions = np.array([n.ray_dir    for n in group_nodes])   # (N, 3)

    rng = np.random.default_rng(0)
    best_mask = np.zeros(N, dtype=bool)

    for _ in range(n_iter):
        idx   = rng.choice(N, size=2, replace=False)
        P_try = _closest_point_n_rays(origins[idx], directions[idx])
        res   = np.array([_ray_dist(P_try, origins[k], directions[k])
                          for k in range(N)])
        mask  = res < ransac_tol
        if mask.sum() > best_mask.sum():
            best_mask = mask

    if best_mask.sum() < min_inliers:
        return []

    # Refit on inlier subset with scipy.optimize.least_squares
    inlier_idx = np.where(best_mask)[0]
    inlier_o   = origins[inlier_idx]
    inlier_d   = directions[inlier_idx]
    P0         = _closest_point_n_rays(inlier_o, inlier_d)

    def _res(P: np.ndarray) -> np.ndarray:
        return np.array([_ray_dist(P, inlier_o[k], inlier_d[k])
                         for k in range(len(inlier_idx))])

    # least_squares('lm') requires n_residuals >= n_variables (3).
    # With exactly 2 inliers the analytical WLS solution is already optimal.
    if len(inlier_idx) >= 3:
        fit     = least_squares(_res, P0, method='trf')
        P_final = fit.x
    else:
        P_final = P0
    res_vals = _res(P_final)

    inlier_nodes = [group_nodes[k] for k in inlier_idx]
    result = {
        'X':             float(P_final[0]),
        'Y':             float(P_final[1]),
        'Z':             float(P_final[2]),
        'residual_m':    float(np.mean(res_vals)),
        'n_obs':         int(best_mask.sum()),
        'cls':           group_nodes[0].cls,
        'image_ids':     [n.image_id      for n in inlier_nodes],
        'detection_ids': [n.detection_id  for n in inlier_nodes],
    }

    # Recursively handle outliers — may belong to a second nearby object
    outlier_nodes = [group_nodes[k] for k in range(N) if not best_mask[k]]
    sub_results   = _ransac_group(outlier_nodes, ransac_tol, min_inliers, n_iter)

    return [result] + sub_results


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def associate(
    point_path        : str,
    eop_path          : str,
    class_names       : list[str],
    W                 : int   = 8000,
    H                 : int   = 4000,
    window_m          : float = 20.0,
    ray_dist_tol      : float = 2.0,
    max_obj_dist      : float = 100.0,
    min_obs           : int   = 2,
    ransac_tol        : float = 1.0,
    cluster_strategy  : str   = 'cliques',
    appearance_weight : float = 0.0,
    class_aliases     : Optional[dict[str, str]] = None,
    output_path       : Optional[str] = None,
) -> Optional[str]:
    """
    Full offline multi-view association pipeline.

    Steps
    -----
    1. Load detections + EOP → :class:`DetectionNode` list with EGSA87 rays.
    2. Find candidate pairs via cKDTree — O(n log n).
    3. Build association graph (5-criterion edge filter + weights).
    4. Cluster with *cluster_strategy*.
    5. RANSAC + ``scipy.optimize.least_squares`` per cluster; split groups
       whose outlier detections form a second valid object.
    6. Write one row per confirmed physical object to a CSV.

    Parameters
    ----------
    point_path        : path to image_coords.csv from run_detection()
    eop_path          : path to GET EOP CSV (tab-separated)
    class_names       : raw class names from the Detectron model(s) used —
                        needed to parse the class label out of *point_name*
    W, H              : panorama dimensions in pixels
    window_m          : spatial search radius for candidate pairs (m);
                        only stations within this radius are compared
    ray_dist_tol      : max average ray-distance to accept a graph edge (m)
    max_obj_dist      : max distance of intersection from camera centre (m);
                        rejects hypothetical objects behind the horizon
    min_obs           : minimum observations required per confirmed object
    ransac_tol        : RANSAC inlier threshold in metres
    cluster_strategy  : ``'cliques'`` (recommended, chaining-resistant) or
                        ``'connected_components'``
    appearance_weight : blend factor for appearance similarity in edge weight;
                        0.0 = geometry only (default)
    class_aliases     : override :data:`DEFAULT_CLASS_ALIASES`; ``None`` uses
                        the module default
    output_path       : explicit output CSV path; auto-generated if ``None``

    Returns
    -------
    Absolute path to the output CSV, or ``None`` if no objects confirmed.
    """
    if class_aliases is None:
        class_aliases = DEFAULT_CLASS_ALIASES

    # ── 1. Load nodes ────────────────────────────────────────────────────────
    nodes = _load_nodes(point_path, eop_path, class_names, class_aliases, W, H)
    print(f"Loaded {len(nodes)} detection nodes.")
    if len(nodes) < 2:
        print("Not enough detections for association.")
        return None

    # ── 2. Candidate pairs ───────────────────────────────────────────────────
    pairs = _candidate_pairs(nodes, window_m)
    print(f"Candidate pairs (window={window_m} m): {len(pairs)}")

    # ── 3. Association graph ─────────────────────────────────────────────────
    G = _build_graph(nodes, pairs, ray_dist_tol, max_obj_dist, appearance_weight)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── 4. Cluster ───────────────────────────────────────────────────────────
    clusters = _cluster(G, cluster_strategy, min_obs)
    print(f"Clusters ({cluster_strategy}): {len(clusters)}")

    # ── 5. RANSAC per cluster ────────────────────────────────────────────────
    node_by_id  = {n.detection_id: n for n in nodes}
    all_results : list[dict] = []
    obj_counter = 0

    for cluster in clusters:
        group_nodes = [node_by_id[did] for did in cluster]
        for res in _ransac_group(group_nodes, ransac_tol, min_obs):
            res['object_id'] = f"obj_{obj_counter:04d}"
            all_results.append(res)
            obj_counter += 1

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
    Example workflow
    ----------------
    detections (image_coords.csv) + EOP
          │
          ▼
    _load_nodes()            build DetectionNode list — rays in EGSA87
    (reuses _rotation_egsa from forward_intersection)
          │
          ▼
    _candidate_pairs()       O(n log n) via cKDTree — only nearby stations
          │
          ▼
    _build_graph()           5-criterion edge filter:
                               same class · ray dist · positive depth
                               max object dist · spatial window
          │
          ▼
    _cluster()               'cliques' strategy — chaining-resistant
          │
          ▼
    _ransac_group()          RANSAC + scipy.optimize.least_squares
                             recursive split of outlier sub-groups
          │
          ▼
    output CSV               one row per physical object (EGSA87 + metadata)
    """

    # Traffic_Sign class list (matches Detectron.py Traffic_Sign model)
    _SIGN_CLASSES = [
        "", "Attention", "Bend_to_left", "Bend_to_right", "Crosswalk",
        "Fork_road", "Give_way", "Narrow_road", "No_entry", "No_left_turn",
        "No_right_turn", "No_u_turn", "Roundabout_mandatory",
        "Speed_limit_100KM", "Speed_limit_110KM", "Speed_limit_120KM",
        "Speed_limit_20KM", "Speed_limit_30KM", "Speed_limit_40KM",
        "Speed_limit_50KM", "Speed_limit_60KM", "Speed_limit_70KM",
        "Speed_limit_80KM", "Speed_limit_90KM", "Stop",
    ]

    _default_coords = os.path.join(_BASE, 'output', 'coords', 'image_coords.csv')
    _default_eop    = os.path.join(_BASE, '')

    _point_path = (input(f'Image coords CSV [{_default_coords}]: ').strip()
                   or _default_coords)
    _eop_path   = (input(f'EOP CSV          [{_default_eop}]: ').strip()
                   or _default_eop)

    associate(
        point_path       = _point_path,
        eop_path         = _eop_path,
        class_names      = _SIGN_CLASSES,
        window_m         = 20.0,
        ray_dist_tol     = 1.5,
        ransac_tol       = 1.0,
        cluster_strategy = 'cliques',
    )