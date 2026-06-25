#!/usr/bin/env python3
"""
colormap.py

For each checkpoint, generate 3D colored point clouds in the normalized mesh
frame. The absolute scale is the mesh scale used by the evaluation inputs.

The script writes two point clouds:
        * pred_to_gt.ply
        * gt_to_pred.ply

Each cloud contains:
    - RGB colors for quick visual inspection;
    - a scalar field named `dist_raw`, storing raw nearest-neighbor distances
      in mesh units for CloudCompare legends and quantitative inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plyfile import PlyData, PlyElement

from metrics_utils import load_mesh, nn_distances, sample_surface_points


# ============================================================
# Simple colormap (RGB only for visualization)
# ============================================================
def _clamp01(x: np.ndarray) -> np.ndarray:
    """Clamp numeric values to the [0, 1] interval used by the RGB colormap."""
    return np.clip(x, 0.0, 1.0)


def scalar_to_rgb(s: np.ndarray) -> np.ndarray:
    """Map normalized scalar values to a blue-cyan-green-yellow-red ramp."""
    s = _clamp01(s)
    r = _clamp01(1.5 * s - 0.5)
    g = _clamp01(1.5 - 3.0 * np.abs(s - 0.5))
    b = _clamp01(1.0 - 1.5 * s)
    rgb = np.stack([r, g, b], axis=1)
    return (255.0 * rgb).astype(np.uint8)


# ============================================================
# Save point cloud with scalar field 'dist_raw'
# ============================================================
def save_pointcloud_with_dist(
    out_path: Path,
    points: np.ndarray,
    dist_raw: np.ndarray,
    color_hi: float = 0.10,
) -> None:
    """
    Save PLY point cloud with:
      - xyz
      - uchar rgb (from dist_raw normalized by a fixed global scale, for visualization)
      - scalar field:
          * dist_raw (mesh units, RAW values)
    """
    pts = np.asarray(points, dtype=np.float32)
    d   = np.asarray(dist_raw, dtype=np.float64)

    # Fixed global color scale (meters):
    # 0.00 m -> blue
    # 0.02 m -> cyan
    # 0.04 m -> green
    # 0.06 m -> yellow
    # 0.08 m -> orange
    # 0.10 m -> red
    hi = float(color_hi)
    s = d / hi
    colors = scalar_to_rgb(s)

    vertex = np.empty(len(pts), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("dist_raw", "f4"),
    ])

    vertex["x"]         = pts[:, 0]
    vertex["y"]         = pts[:, 1]
    vertex["z"]         = pts[:, 2]
    vertex["red"]       = colors[:, 0]
    vertex["green"]     = colors[:, 1]
    vertex["blue"]      = colors[:, 2]
    vertex["dist_raw"] = d.astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(out_path))

    pct_over = 100.0 * float(np.mean(d >= hi)) if len(d) > 0 else 0.0
    print(f"  [colormap] color_hi={hi:.6f} m, points dist_raw>=color_hi: {pct_over:.2f}%")


# ============================================================
# CLI
# ============================================================
def parse_args():
    """Parse command-line arguments for checkpoint-wise colormap generation."""
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path",   type=Path, required=True)
    p.add_argument("--mesh_dir",  type=Path, required=True)
    p.add_argument("--checks",    nargs="+", required=True)
    p.add_argument("--n_points",  type=int,  default=50_000)
    p.add_argument("--seed",      type=int,  default=0,
                   help="Set -1 to disable seeding.")
    p.add_argument("--color_hi",  type=float, default=0.10)
    p.add_argument("--out_dir",   type=Path, required=True)
    return p.parse_args()


def main():
    """Load meshes, sample surfaces, compute distances, and save colored PLYs."""
    args = parse_args()

    if not args.gt_path.exists():
        raise FileNotFoundError(f"gt_path not found: {args.gt_path}")
    if not args.mesh_dir.exists():
        raise FileNotFoundError(f"mesh_dir not found: {args.mesh_dir}")

    pred_paths = [args.mesh_dir / ck for ck in args.checks]
    for pth in pred_paths:
        if not pth.exists():
            raise FileNotFoundError(f"Missing predicted mesh: {pth}")

    gt_mesh = load_mesh(args.gt_path)

    print("Assuming predicted meshes are already in the same frame as GT.")

    for ck_idx, (ck, pred_path) in enumerate(zip(args.checks, pred_paths)):
        name = Path(ck).stem
        print(f"\n[CHECKPOINT] {name}")

        pred_mesh = load_mesh(pred_path)

        out_base = args.out_dir / name
        out_base.mkdir(parents=True, exist_ok=True)

        if args.seed == -1:
            gt_seed = None
            pred_seed = None
        else:
            # Keep deterministic and explicit sampling per checkpoint.
            gt_seed = int(args.seed) + (2 * ck_idx)
            pred_seed = gt_seed + 1

        gt_pts   = sample_surface_points(gt_mesh,   args.n_points, seed=gt_seed)
        pred_pts = sample_surface_points(pred_mesh, args.n_points, seed=pred_seed)

        d_pred_to_gt = nn_distances(pred_pts, gt_pts)
        d_gt_to_pred = nn_distances(gt_pts, pred_pts)

        save_pointcloud_with_dist(
            out_base / "pred_to_gt.ply",
            pred_pts,
            d_pred_to_gt,
            color_hi=args.color_hi,
        )
        save_pointcloud_with_dist(
            out_base / "gt_to_pred.ply",
            gt_pts,
            d_gt_to_pred,
            color_hi=args.color_hi,
        )

        print(f"  [saved] pred_to_gt.ply, gt_to_pred.ply -> {out_base}")


if __name__ == "__main__":
    main()
