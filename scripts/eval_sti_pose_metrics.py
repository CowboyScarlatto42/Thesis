#!/usr/bin/env python3
"""Minimal single-frame STI-Pose evaluator with dual CAD/NeuS evaluation."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.append("external/sti_pose")
from SilhouettePE import Process


def parse_args():
    parser = argparse.ArgumentParser(description="Single-frame STI-Pose evaluator (CAD + NeuS).")
    parser.add_argument("--mesh_cad", type=str, required=True, help="Path to CAD mesh.")
    parser.add_argument("--mesh_neus", type=str, required=True, help="Path to NeuS mesh.")
    parser.add_argument("--cameras_npz", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--pred_pose", type=str, required=True)
    parser.add_argument("--pred_pose_internal", type=str, default=None, help="Raw pose_internal for rendering.")
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--n_points", type=int, default=10000)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def load_gt_pose_and_intrinsics(cameras_npz: str, idx: int):
    """Load GT pose T_CW (object at origin) and intrinsics K."""
    data = np.load(cameras_npz)
    key = f"world_mat_{idx}"
    if key not in data:
        sys.exit(f"Key '{key}' not found in cameras file.")
    P = data[key][:3, :4].astype(np.float64)
    K, R, T_homog, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    K = K / K[2, 2]
    C = T_homog[:3, 0] / T_homog[3, 0]
    t = -R @ C
    T_gt = np.eye(4, dtype=np.float64)
    T_gt[:3, :3] = R
    T_gt[:3, 3] = t.flatten()
    return T_gt, K


def load_mask(mask_path: str, width: int, height: int) -> np.ndarray:
    """Load binary mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        sys.exit(f"Failed to load mask: {mask_path}")
    if mask.shape[0] != height or mask.shape[1] != width:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def sample_points_and_diameter(mesh_path: str, n_points: int):
    """Sample points from mesh surface and compute diameter."""
    mesh = trimesh.load(mesh_path, force="mesh")
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points = np.array(points)
    diameter = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return points, diameter


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    return (T[:3, :3] @ points.T).T + T[:3, 3]


def compute_add(points: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    """Compute ADD metric."""
    pts_pred = transform_points(points, T_pred)
    pts_gt = transform_points(points, T_gt)
    return float(np.linalg.norm(pts_pred - pts_gt, axis=1).mean())


def compute_adds(points: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    """Compute ADD-S metric using cKDTree."""
    pts_pred = transform_points(points, T_pred)
    pts_gt = transform_points(points, T_gt)
    tree = cKDTree(pts_gt)
    distances, _ = tree.query(pts_pred, k=1)
    return float(distances.mean())


def compute_iou(rendered: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between rendered silhouette and GT mask."""
    rendered_bin = rendered > 0
    gt_bin = gt_mask > 0
    intersection = np.logical_and(rendered_bin, gt_bin).sum()
    union = np.logical_or(rendered_bin, gt_bin).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def create_overlay(rendered: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Create overlay: GREEN=rendered, RED=GT."""
    overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 1] = (rendered > 0).astype(np.uint8) * 255
    overlay[:, :, 2] = (gt_mask > 0).astype(np.uint8) * 255
    return overlay


def load_pose_internal(path: str):
    """Load pose_internal from text or JSON file."""
    with open(path, "r") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return np.loadtxt(path).tolist()


def main():
    args = parse_args()

    # Load poses
    T_gt, K = load_gt_pose_and_intrinsics(args.cameras_npz, args.idx)
    T_pred = np.loadtxt(args.pred_pose).astype(np.float64)
    gt_mask = load_mask(args.mask, args.width, args.height)

    # Sample points from both meshes
    pts_cad, diam_cad = sample_points_and_diameter(args.mesh_cad, args.n_points)
    pts_neus, diam_neus = sample_points_and_diameter(args.mesh_neus, args.n_points)

    # CAD-based metrics (includes reconstruction error from NeuS)
    add_cad = compute_add(pts_cad, T_pred, T_gt)
    adds_cad = compute_adds(pts_cad, T_pred, T_gt)
    thr_cad = 0.1 * diam_cad
    pass_add_cad = add_cad < thr_cad
    pass_adds_cad = adds_cad < thr_cad

    # NeuS-based metrics (pose error only, no reconstruction error)
    add_neus = compute_add(pts_neus, T_pred, T_gt)
    adds_neus = compute_adds(pts_neus, T_pred, T_gt)
    thr_neus = 0.1 * diam_neus
    pass_add_neus = add_neus < thr_neus
    pass_adds_neus = adds_neus < thr_neus

    # Render silhouette using NeuS mesh
    p = Process((args.width, args.height), K, 1)
    p.set_model(args.mesh_neus)
    if args.pred_pose_internal:
        pose_internal = load_pose_internal(args.pred_pose_internal)
        rendered = p.render_silhouette(pose_internal)
    else:
        print("[WARNING] No --pred_pose_internal provided; using T_pred for rendering (may be inconsistent).")
        rendered = p.render_silhouette(T_pred)

    # IoU (NeuS mesh only)
    iou_pred = compute_iou(rendered, gt_mask)

    # Print metrics
    print("# CAD-based (includes reconstruction error)")
    print(f"ADD_CAD={add_cad:.6f}, ADD-S_CAD={adds_cad:.6f}, "
          f"thr_CAD={thr_cad:.6f}, pass_add_CAD={pass_add_cad}, pass_adds_CAD={pass_adds_cad}")
    print("# NeuS-based (pose error only)")
    print(f"ADD_NeuS={add_neus:.6f}, ADD-S_NeuS={adds_neus:.6f}, "
          f"thr_NeuS={thr_neus:.6f}, pass_add_NeuS={pass_add_neus}, pass_adds_NeuS={pass_adds_neus}")
    print(f"IoU_pred={iou_pred:.4f}")

    # Create overlay
    overlay = create_overlay(rendered, gt_mask)

    # Display images
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gt_mask, cmap="gray")
    axes[0].set_title("GT Mask")
    axes[0].axis("off")
    axes[1].imshow(rendered, cmap="gray")
    axes[1].set_title("Rendered Silhouette (NeuS)")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (R=GT, G=Pred)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

    # Save outputs if --out provided
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "overlay.png"), overlay[:, :, ::-1])
        with open(out_dir / "metrics.txt", "w") as f:
            f.write("# CAD-based (includes reconstruction error)\n")
            f.write(f"ADD_CAD={add_cad:.6f}\n")
            f.write(f"ADD-S_CAD={adds_cad:.6f}\n")
            f.write(f"diameter_CAD={diam_cad:.6f}\n")
            f.write(f"threshold_CAD={thr_cad:.6f}\n")
            f.write(f"pass_add_CAD={pass_add_cad}\n")
            f.write(f"pass_adds_CAD={pass_adds_cad}\n")
            f.write("\n# NeuS-based (pose error only)\n")
            f.write(f"ADD_NeuS={add_neus:.6f}\n")
            f.write(f"ADD-S_NeuS={adds_neus:.6f}\n")
            f.write(f"diameter_NeuS={diam_neus:.6f}\n")
            f.write(f"threshold_NeuS={thr_neus:.6f}\n")
            f.write(f"pass_add_NeuS={pass_add_neus}\n")
            f.write(f"pass_adds_NeuS={pass_adds_neus}\n")
            f.write(f"\nIoU_pred={iou_pred:.4f}\n")


if __name__ == "__main__":
    main()
