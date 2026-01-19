#!/usr/bin/env python3
"""Minimal single-frame STI-Pose evaluator with dual CAD/NeuS evaluation."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
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
    parser.add_argument("--mask_cad", type=str, default=None, help="CAD mask for GT sanity check.")
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
    """Create 3-class overlay: RED=GT only, GREEN=Pred only, YELLOW=Overlap."""
    pred_bin = rendered > 0
    gt_bin = gt_mask > 0
    overlap = np.logical_and(pred_bin, gt_bin)
    gt_only = np.logical_and(gt_bin, ~pred_bin)
    pred_only = np.logical_and(pred_bin, ~gt_bin)
    overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    overlay[gt_only] = [255, 0, 0]      # RED
    overlay[pred_only] = [0, 255, 0]    # GREEN
    overlay[overlap] = [255, 255, 0]    # YELLOW
    return overlay


def load_pose_internal(path: str) -> np.ndarray:
    """Load pose_internal as a 4x4 float64 numpy array."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"pose_internal not found: {p}")

    # Try numeric text first (np.savetxt style)
    try:
        arr = np.loadtxt(str(p), dtype=np.float64)
        arr = np.asarray(arr, dtype=np.float64)
    except Exception:
        # Fallback to JSON list
        with open(p, "r") as f:
            arr = json.load(f)
        arr = np.asarray(arr, dtype=np.float64)

    # Accept 3x4 or 4x4; promote 3x4 -> 4x4
    if arr.shape == (3, 4):
        arr4 = np.eye(4, dtype=np.float64)
        arr4[:3, :4] = arr
        arr = arr4

    if arr.shape != (4, 4):
        raise ValueError(f"pose_internal must be 4x4 (or 3x4). Got shape {arr.shape} from {p}")

    return arr



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

    # Render predicted silhouette using NeuS mesh
    p_neus = Process((args.width, args.height), K, 1)
    p_neus.set_model(args.mesh_neus)

    if args.pred_pose_internal:
        pose_internal = load_pose_internal(args.pred_pose_internal)
        rendered_pred = p_neus.render_silhouette(pose_internal)
    else:
        print("[WARNING] No --pred_pose_internal provided; using T_pred for rendering (may be inconsistent).")
        rendered_pred = p_neus.render_silhouette(T_pred)

    iou_pred_neus = compute_iou(rendered_pred, gt_mask)

    # Sanity check: render GT pose with CAD mesh vs CAD mask (optional)
    iou_gt_cad = None
    rendered_gt_cad = None
    mask_cad = None
    if args.mask_cad:
        mask_cad = load_mask(args.mask_cad, args.width, args.height)
        # Compute CAD mesh center for STI-Pose recentering correction
        cad_mesh = trimesh.load(args.mesh_cad, force="mesh")
        cad_center = cad_mesh.bounding_box.centroid
        # Build T_gt_centered: account for STI-Pose internal recentering
        T_gt_centered = T_gt.copy()
        T_gt_centered[:3, 3] = T_gt[:3, 3] + T_gt[:3, :3] @ cad_center
        p_cad = Process((args.width, args.height), K, 1)
        p_cad.set_model(args.mesh_cad)
        rendered_gt_cad = p_cad.render_silhouette(T_gt_centered)
        iou_gt_cad = compute_iou(rendered_gt_cad, mask_cad)
    else:
        print("[INFO] No --mask_cad provided; skipping GT sanity check.")

    # Print metrics
    print("# CAD-based (includes reconstruction error)")
    print(f"ADD_CAD={add_cad:.6f}, ADD-S_CAD={adds_cad:.6f}, "
          f"thr_CAD={thr_cad:.6f}, pass_add_CAD={pass_add_cad}, pass_adds_CAD={pass_adds_cad}")
    print("# NeuS-based (pose error only)")
    print(f"ADD_NeuS={add_neus:.6f}, ADD-S_NeuS={adds_neus:.6f}, "
          f"thr_NeuS={thr_neus:.6f}, pass_add_NeuS={pass_add_neus}, pass_adds_NeuS={pass_adds_neus}")
    iou_gt_str = f"{iou_gt_cad:.4f}" if iou_gt_cad is not None else "SKIPPED"
    print(f"IoU_pred_neus={iou_pred_neus:.4f}, IoU_gt_cad={iou_gt_str}")

    # Create overlay (predicted vs GT mask)
    overlay_pred = create_overlay(rendered_pred, gt_mask)

    # Display images (predicted vs GT mask)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Hubble Space Telescope (HST)", fontsize=14)
    axes[0].imshow(gt_mask, cmap="gray")
    axes[0].set_title("GT Mask")
    axes[0].axis("off")
    axes[1].imshow(rendered_pred, cmap="gray")
    axes[1].set_title("Rendered Silhouette (NeuS)")
    axes[1].axis("off")
    axes[2].imshow(overlay_pred)
    axes[2].set_title("Overlay (Pred vs GT)")
    axes[2].axis("off")
    legend_patches = [
        mpatches.Patch(color="red", label="GT only"),
        mpatches.Patch(color="green", label="Pred only"),
        mpatches.Patch(color="yellow", label="Overlap"),
    ]
    axes[2].legend(handles=legend_patches, loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.show()

    # Display sanity check figure if computed
    if iou_gt_cad is not None:
        overlay_sanity = create_overlay(rendered_gt_cad, mask_cad)
        fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
        fig2.suptitle("Sanity Check: CAD GT Pose vs CAD Mask", fontsize=14)
        axes2[0].imshow(mask_cad, cmap="gray")
        axes2[0].set_title("CAD Mask")
        axes2[0].axis("off")
        axes2[1].imshow(rendered_gt_cad, cmap="gray")
        axes2[1].set_title("Rendered GT (CAD)")
        axes2[1].axis("off")
        axes2[2].imshow(overlay_sanity)
        axes2[2].set_title("Overlay (Sanity)")
        axes2[2].axis("off")
        axes2[2].legend(handles=legend_patches, loc="lower left", framealpha=0.9)
        plt.tight_layout()
        plt.show()

    # Save outputs if --out provided
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "overlay_pred.png"), overlay_pred[:, :, ::-1])
        if iou_gt_cad is not None:
            overlay_sanity = create_overlay(rendered_gt_cad, mask_cad)
            cv2.imwrite(str(out_dir / "overlay_sanity.png"), overlay_sanity[:, :, ::-1])
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
            f.write(f"\nIoU_pred_neus={iou_pred_neus:.4f}\n")
            if iou_gt_cad is not None:
                f.write(f"IoU_gt_cad={iou_gt_cad:.4f}\n")
            else:
                f.write("IoU_gt_cad=SKIPPED\n")


if __name__ == "__main__":
    main()
