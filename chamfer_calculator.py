#!/usr/bin/env python3
"""
compute_chamfer_unisurf.py

Compute symmetric Chamfer distance (L2^2) between two meshes,
following UNISURF and "Multiview Neural Surface Reconstruction
by Disentangling Geometry and Appearance".

- Uniform surface sampling via trimesh.sample_surface
- Symmetric, squared L2 Chamfer:
    CD = mean_p(min_q ||p-q||^2) + mean_q(min_p ||p-q||^2)
"""

import argparse
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_and_sample(mesh_path: str,
                    n_points: int = 300_000,
                    seed: int | None = 0) -> np.ndarray:
    """
    Load mesh and sample n_points uniformly on its surface.

    Returns:
        points: (n_points, 3) float32
    """
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh = trimesh.load(mesh_path, process=True)
    # Gestisci il caso Scene -> concatena le geometrie
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"Scene has no geometry: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if seed is not None:
        np.random.seed(seed)

    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def chamfer_L2_squared(p_pred: np.ndarray,
                       p_gt: np.ndarray) -> dict:
    """
    Symmetric Chamfer distance with squared L2 norm.

    Args:
        p_pred: (N, 3) predicted surface samples
        p_gt  : (M, 3) ground truth surface samples

    Returns:
        dict with:
          cd_l2_sq          : mean squared Chamfer (symm)
          pred_to_gt_mean   : mean ||p-q||^2, p in pred, q in gt
          gt_to_pred_mean   : mean ||q-p||^2, q in gt, p in pred
    """
    if p_pred.ndim != 2 or p_pred.shape[1] != 3:
        raise ValueError(f"p_pred must be (N,3), got {p_pred.shape}")
    if p_gt.ndim != 2 or p_gt.shape[1] != 3:
        raise ValueError(f"p_gt must be (M,3), got {p_gt.shape}")

    # pred -> gt
    tree_gt = cKDTree(p_gt)
    d_pred_to_gt, _ = tree_gt.query(p_pred, k=1)
    d_pred_to_gt_sq = d_pred_to_gt.astype(np.float64) ** 2
    pred_to_gt_mean = float(d_pred_to_gt_sq.mean())

    # gt -> pred
    tree_pred = cKDTree(p_pred)
    d_gt_to_pred, _ = tree_pred.query(p_gt, k=1)
    d_gt_to_pred_sq = d_gt_to_pred.astype(np.float64) ** 2
    gt_to_pred_mean = float(d_gt_to_pred_sq.mean())

    cd_l2_sq = pred_to_gt_mean + gt_to_pred_mean

    return {
        "cd_l2_sq": cd_l2_sq,
        "pred_to_gt_mean": pred_to_gt_mean,
        "gt_to_pred_mean": gt_to_pred_mean,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Symmetric Chamfer L2^2 between two meshes (UNISURF-style)."
    )
    parser.add_argument("--pred_mesh", type=str, required=True,
                        help="Predicted mesh path (e.g. NeuS output .ply)")
    parser.add_argument("--gt_mesh", type=str, required=True,
                        help="Ground-truth mesh path")
    parser.add_argument("--n_points", type=int, default=300_000,
                        help="Number of surface samples per mesh (default: 300k)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampling (None to disable)")
    parser.add_argument("--output", type=str, default="",
                        help="Optional JSON file to save metrics")

    args = parser.parse_args()

    print("=== UNISURF-style Chamfer (L2^2) ===")
    print(f"Pred mesh: {args.pred_mesh}")
    print(f"GT mesh  : {args.gt_mesh}")
    print(f"n_points : {args.n_points}")
    print(f"seed     : {args.seed}")

    # Sample
    p_pred = load_and_sample(args.pred_mesh, args.n_points, args.seed)
    p_gt   = load_and_sample(args.gt_mesh, args.n_points, args.seed)

    print(f"Sampled {p_pred.shape[0]} pts (pred), {p_gt.shape[0]} pts (gt)")

    # Chamfer
    metrics = chamfer_L2_squared(p_pred, p_gt)

    print("\n--- Results (squared L2) ---")
    print(f"cd_l2_sq        : {metrics['cd_l2_sq']:.6e}")
    print(f"pred_to_gt_mean : {metrics['pred_to_gt_mean']:.6e}")
    print(f"gt_to_pred_mean : {metrics['gt_to_pred_mean']:.6e}")

    if args.output:
        out_path = args.output
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nSaved metrics to:", out_path)


if __name__ == "__main__":
    main()
