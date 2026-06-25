from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def clean_mesh(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove invalid or degenerate mesh data before sampling/evaluation."""
    if hasattr(m, "remove_infinite_values"):
        m.remove_infinite_values()
    if hasattr(m, "remove_unreferenced_vertices"):
        m.remove_unreferenced_vertices()
    if hasattr(m, "remove_duplicate_faces"):
        m.remove_duplicate_faces()
    if hasattr(m, "area_faces") and hasattr(m, "update_faces"):
        mask = m.area_faces > 1e-16
        if mask.shape[0] == len(m.faces) and np.any(~mask):
            m.update_faces(mask)
            if hasattr(m, "remove_unreferenced_vertices"):
                m.remove_unreferenced_vertices()
    return m


def to_trimesh(mesh_or_scene) -> trimesh.Trimesh:
    """Convert a loaded trimesh object or scene into a single Trimesh."""
    if isinstance(mesh_or_scene, trimesh.Scene):
        geoms = list(mesh_or_scene.geometry.values())
        if len(geoms) == 0:
            raise ValueError("Loaded an empty trimesh.Scene")
        return trimesh.util.concatenate(geoms)
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    raise TypeError(f"Unsupported type: {type(mesh_or_scene)}")


def load_mesh(path) -> trimesh.Trimesh:
    """Load a mesh file and normalize it to a cleaned Trimesh object."""
    m = trimesh.load(path, force="mesh")
    return clean_mesh(to_trimesh(m))


def sample_surface_points(mesh: trimesh.Trimesh, n_points: int, seed: int | None = None) -> np.ndarray:
    """Sample points uniformly on the mesh surface, optionally deterministically."""
    if seed is None:
        pts, _ = trimesh.sample.sample_surface(mesh, int(n_points))
        return np.asarray(pts, dtype=np.float64)

    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        pts, _ = trimesh.sample.sample_surface(mesh, int(n_points))
    finally:
        np.random.set_state(state)
    return np.asarray(pts, dtype=np.float64)


def nn_distances(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    """Compute nearest-neighbor distances from one point set to another."""
    tree = cKDTree(p_to)
    d, _ = tree.query(p_from, k=1)
    return np.asarray(d, dtype=np.float64)


def symmetric_chamfer(d_pred_to_gt: np.ndarray, d_gt_to_pred: np.ndarray) -> float:
    """Return the symmetric Chamfer distance from the two directed distance arrays."""
    d_pred_to_gt = np.asarray(d_pred_to_gt, dtype=np.float64)
    d_gt_to_pred = np.asarray(d_gt_to_pred, dtype=np.float64)
    return float(0.5 * (np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)))


def _finite_distances(d: np.ndarray) -> np.ndarray:
    """Return only finite distances, avoiding NaN/Inf pollution in metrics."""
    d = np.asarray(d, dtype=np.float64)
    return d[np.isfinite(d)]


def directed_hausdorff(d_from_to: np.ndarray) -> float:
    """Return the maximum finite directed nearest-neighbor distance."""
    d = _finite_distances(d_from_to)
    if d.size == 0:
        return float("nan")
    return float(np.max(d))


def directed_hausdorff_p95(d_from_to: np.ndarray) -> float:
    """Return a robust directed Hausdorff proxy based on the 95th percentile."""
    d = _finite_distances(d_from_to)
    if d.size == 0:
        return float("nan")
    return float(np.quantile(d, 0.95))


def symmetric_hausdorff(d_pred_to_gt: np.ndarray, d_gt_to_pred: np.ndarray) -> float:
    """Return the larger of the two directed Hausdorff distances."""
    return float(np.maximum(
        directed_hausdorff(d_pred_to_gt),
        directed_hausdorff(d_gt_to_pred),
    ))


def symmetric_hausdorff_p95(d_pred_to_gt: np.ndarray, d_gt_to_pred: np.ndarray) -> float:
    """Return the larger of the two directed 95th-percentile Hausdorff proxies."""
    return float(np.maximum(
        directed_hausdorff_p95(d_pred_to_gt),
        directed_hausdorff_p95(d_gt_to_pred),
    ))
