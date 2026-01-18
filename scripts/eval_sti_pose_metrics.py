#!/usr/bin/env python3
"""
eval_sti_pose_metrics.py

Minimal evaluator for STI-Pose predictions using official paper metrics:
- ADD (Average Distance of Model Points)
- ADD-S (ADD-Symmetric, using nearest-neighbor matching)
- Success rate at 0.1*diameter threshold
- AUC (Area Under Curve) up to 0.10 meters

GT pose convention:
    T_CO_gt = T_CW (object assumed at world origin)
    T_CW is extracted from cameras_spe3r.npz via cv2.decomposeProjectionMatrix.

Usage:
    # Single frame
    python scripts/eval_sti_pose_metrics.py \
        --mesh /path/to/mesh.ply \
        --cameras_npz /path/to/cameras_spe3r.npz \
        --mask_dir /path/to/masks \
        --pred_pose_file /path/to/pose_000.txt \
        --idx 0

    # Multi-frame
    python scripts/eval_sti_pose_metrics.py \
        --mesh /path/to/mesh.ply \
        --cameras_npz /path/to/cameras_spe3r.npz \
        --mask_dir /path/to/masks \
        --pred_pose_dir /path/to/predictions \
        --idx_list /path/to/indices.txt \
        --save --out_dir /path/to/output
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Optional: scipy for fast ADD-S
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not available. ADD-S will use brute-force (slower).")

# Mesh loading: trimesh or open3d
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

if not HAS_TRIMESH and not HAS_OPEN3D:
    print("[ERROR] Either trimesh or open3d is required.")
    sys.exit(1)

# STI-Pose renderer for IoU
HAS_STI_POSE = False
try:
    sys.path.append("external/sti_pose")
    from SilhouettePE import Process as STIPoseProcess
    HAS_STI_POSE = True
except ImportError:
    pass  # Will warn only if IoU is requested


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate STI-Pose predictions (ADD, ADD-S, AUC)."
    )
    # Required
    parser.add_argument("--mesh", type=str, required=True, help="Path to .ply mesh.")
    parser.add_argument("--cameras_npz", type=str, required=True, help="Path to cameras_spe3r.npz.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory with GT masks {idx:03d}.png.")

    # Frame selection (mutually exclusive groups)
    parser.add_argument("--idx", type=int, default=None, help="Single frame index.")
    parser.add_argument("--pred_pose_file", type=str, default=None, help="Single predicted pose file.")
    parser.add_argument("--idx_list", type=str, default=None, help="Text file with frame indices.")
    parser.add_argument("--pred_pose_dir", type=str, default=None, help="Directory with pose_{idx:03d}.txt files.")

    # Optional parameters
    parser.add_argument("--width", type=int, default=256, help="Image width (default: 256).")
    parser.add_argument("--height", type=int, default=256, help="Image height (default: 256).")
    parser.add_argument("--n_points", type=int, default=10000, help="Points to sample from mesh (default: 10000).")
    parser.add_argument("--mesh_unit_scale", type=float, default=1.0, help="Scale factor for mesh units (default: 1.0).")
    parser.add_argument("--auc_max_threshold", type=float, default=0.10, help="Max threshold for AUC in meters (default: 0.10).")
    parser.add_argument("--n_thresholds", type=int, default=100, help="Number of thresholds for AUC (default: 100).")

    # IoU checks
    parser.add_argument("--check_gt_iou", action="store_true", help="Compute IoU of GT pose vs GT mask (sanity check).")
    parser.add_argument("--check_pred_iou", action="store_true", help="Compute IoU of predicted pose vs GT mask.")
    parser.add_argument("--iou_warn_threshold", type=float, default=0.9, help="Warn if GT IoU < this (default: 0.9).")

    # Output
    parser.add_argument("--save", action="store_true", help="Save results to disk.")
    parser.add_argument("--out_dir", type=str, default="eval_output", help="Output directory (default: eval_output).")

    return parser.parse_args()


# =============================================================================
# Mesh Loading
# =============================================================================

def load_mesh_points(mesh_path: str, n_points: int = 10000, scale: float = 1.0) -> np.ndarray:
    """Load mesh and sample points from surface."""
    mesh_file = Path(mesh_path)
    if not mesh_file.exists():
        print(f"[ERROR] Mesh not found: {mesh_file}")
        sys.exit(1)

    if HAS_TRIMESH:
        mesh = trimesh.load(str(mesh_file), process=False)
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        points = np.array(points)
    else:
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))
        if mesh.is_empty():
            print(f"[ERROR] Failed to load mesh: {mesh_file}")
            sys.exit(1)
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        points = np.asarray(pcd.points)

    points = points * scale
    print(f"[INFO] Loaded {len(points)} points from mesh (scale={scale})")
    return points


def compute_diameter(points: np.ndarray) -> float:
    """Compute object diameter as bounding box diagonal."""
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    return float(np.linalg.norm(max_pt - min_pt))


# =============================================================================
# Pose Loading
# =============================================================================

def load_gt_pose(cameras_npz: str, idx: int) -> tuple:
    """
    Load GT pose T_CO (= T_CW, object at world origin) and intrinsics K.

    Returns:
        T_CO: 4x4 pose matrix (object in camera frame)
        K: 3x3 intrinsic matrix (normalized)
    """
    cameras_path = Path(cameras_npz)
    if not cameras_path.exists():
        print(f"[ERROR] Cameras file not found: {cameras_path}")
        sys.exit(1)

    data = np.load(str(cameras_path))
    key = f"world_mat_{idx}"
    if key not in data:
        print(f"[ERROR] Key '{key}' not found in cameras file.")
        sys.exit(1)

    world_mat = data[key]
    P = world_mat[:3, :4].astype(np.float64)

    # Decompose projection matrix
    K, R, T_homog, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

    # Normalize K
    K = K / K[2, 2]

    # Camera center in world coords
    C = T_homog[:3, 0] / T_homog[3, 0]

    # Translation: t = -R @ C
    t = -R @ C

    # Build T_CW = T_CO (object at world origin)
    T_CO = np.eye(4, dtype=np.float64)
    T_CO[:3, :3] = R
    T_CO[:3, 3] = t.flatten()

    return T_CO, K


def load_pred_pose(pose_path: str) -> np.ndarray:
    """Load predicted 4x4 pose from text file."""
    path = Path(pose_path)
    if not path.exists():
        print(f"[ERROR] Pose file not found: {path}")
        sys.exit(1)

    T = np.loadtxt(str(path))
    if T.shape != (4, 4):
        print(f"[ERROR] Invalid pose shape {T.shape}, expected (4, 4)")
        sys.exit(1)

    return T.astype(np.float64)


# =============================================================================
# Metrics
# =============================================================================

def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t


def compute_add(points: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    """Compute ADD (Average Distance of Model Points)."""
    pts_pred = transform_points(points, T_pred)
    pts_gt = transform_points(points, T_gt)
    return float(np.linalg.norm(pts_pred - pts_gt, axis=1).mean())


def compute_adds(points: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    """Compute ADD-S (symmetric, nearest-neighbor matching)."""
    pts_pred = transform_points(points, T_pred)
    pts_gt = transform_points(points, T_gt)

    if HAS_SCIPY:
        tree = cKDTree(pts_gt)
        distances, _ = tree.query(pts_pred, k=1)
    else:
        distances = np.array([np.linalg.norm(pts_gt - pt, axis=1).min() for pt in pts_pred])

    return float(distances.mean())


def compute_auc(errors: List[float], max_threshold: float, n_thresholds: int) -> float:
    """Compute AUC (area under accuracy-threshold curve)."""
    thresholds = np.linspace(0, max_threshold, n_thresholds + 1)
    errors = np.array(errors)
    accuracies = [(errors < th).mean() for th in thresholds]
    return float(np.trapz(accuracies, thresholds) / max_threshold)


def compute_pass_rate(errors: List[float], threshold: float) -> float:
    """Compute fraction of errors below threshold."""
    return float((np.array(errors) < threshold).mean())


# =============================================================================
# IoU (STI-Pose Renderer)
# =============================================================================

def load_gt_mask(mask_dir: str, idx: int, width: int, height: int) -> np.ndarray:
    """Load GT mask as binary (0/1)."""
    mask_path = Path(mask_dir) / f"{idx:03d}.png"
    if not mask_path.exists():
        print(f"[ERROR] Mask not found: {mask_path}")
        sys.exit(1)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[ERROR] Failed to load mask: {mask_path}")
        sys.exit(1)

    if mask.shape[0] != height or mask.shape[1] != width:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    return (mask > 127).astype(np.uint8)


def compute_iou(rendered: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between rendered silhouette and GT mask."""
    rendered_bin = (rendered > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(rendered_bin, gt_bin).sum()
    union = np.logical_or(rendered_bin, gt_bin).sum()

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_frame(
    idx: int,
    points: np.ndarray,
    diameter: float,
    cameras_npz: str,
    mask_dir: str,
    pred_pose_path: str,
    mesh_path: str,
    width: int,
    height: int,
    sti_process: Optional["STIPoseProcess"],
    check_gt_iou: bool,
    check_pred_iou: bool,
    iou_warn_threshold: float,
) -> Dict:
    """Evaluate a single frame."""
    # Load GT pose and intrinsics
    T_gt, K = load_gt_pose(cameras_npz, idx)

    # Load predicted pose
    T_pred = load_pred_pose(pred_pose_path)

    # Compute metrics
    add = compute_add(points, T_pred, T_gt)
    adds = compute_adds(points, T_pred, T_gt)

    threshold_01d = 0.1 * diameter
    add_pass = add < threshold_01d
    adds_pass = adds < threshold_01d

    metrics = {
        "idx": idx,
        "add": add,
        "adds": adds,
        "add_pass_01d": add_pass,
        "adds_pass_01d": adds_pass,
    }

    # IoU checks
    if (check_gt_iou or check_pred_iou) and sti_process is not None:
        gt_mask = load_gt_mask(mask_dir, idx, width, height)

        if check_gt_iou:
            rendered_gt = sti_process.render_silhouette(T_gt)
            iou_gt = compute_iou(rendered_gt, gt_mask)
            metrics["iou_gt"] = iou_gt

            if iou_gt < iou_warn_threshold:
                print(f"\n{'!'*60}")
                print(f"[WARNING] Frame {idx:03d}: GT IoU = {iou_gt:.4f} < {iou_warn_threshold}")
                print(f"          Possible POSE CONVENTION MISMATCH!")
                print(f"          Check: object at world origin? Correct mesh coords?")
                print(f"{'!'*60}\n")

        if check_pred_iou:
            rendered_pred = sti_process.render_silhouette(T_pred)
            iou_pred = compute_iou(rendered_pred, gt_mask)
            metrics["iou_pred"] = iou_pred

    return metrics


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("STI-Pose Evaluation")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Determine frame indices
    # -------------------------------------------------------------------------
    if args.idx is not None and args.pred_pose_file is not None:
        indices = [args.idx]
        pose_mode = "single"
    elif args.idx_list is not None and args.pred_pose_dir is not None:
        idx_file = Path(args.idx_list)
        if not idx_file.exists():
            print(f"[ERROR] Index list not found: {idx_file}")
            sys.exit(1)
        with open(idx_file, "r") as f:
            indices = [int(line.strip()) for line in f if line.strip()]
        pose_mode = "multi"
    else:
        print("[ERROR] Provide (--idx + --pred_pose_file) OR (--idx_list + --pred_pose_dir)")
        sys.exit(1)

    print(f"[INFO] Evaluating {len(indices)} frame(s)")

    # -------------------------------------------------------------------------
    # Load mesh and compute diameter
    # -------------------------------------------------------------------------
    points = load_mesh_points(args.mesh, args.n_points, args.mesh_unit_scale)
    diameter = compute_diameter(points)
    threshold_01d = 0.1 * diameter

    print(f"[INFO] Diameter: {diameter:.6f} m")
    print(f"[INFO] 0.1*d threshold: {threshold_01d:.6f} m")

    # -------------------------------------------------------------------------
    # Setup STI-Pose renderer if IoU checks requested
    # -------------------------------------------------------------------------
    sti_process = None
    if args.check_gt_iou or args.check_pred_iou:
        if not HAS_STI_POSE:
            print("[WARNING] STI-Pose not available. IoU checks disabled.")
            print("         Ensure external/sti_pose is accessible.")
            args.check_gt_iou = False
            args.check_pred_iou = False
        else:
            # Get K from first frame to initialize renderer
            _, K = load_gt_pose(args.cameras_npz, indices[0])
            sti_process = STIPoseProcess((args.width, args.height), K, 1)
            sti_process.set_model(args.mesh)
            print("[INFO] STI-Pose renderer initialized for IoU checks")

    # -------------------------------------------------------------------------
    # Evaluate frames
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Evaluating frames")
    print("-" * 40)

    all_metrics = []
    add_errors = []
    adds_errors = []
    iou_gt_values = []
    iou_pred_values = []

    for idx in indices:
        # Determine pose path
        if pose_mode == "single":
            pred_pose_path = args.pred_pose_file
        else:
            pred_pose_path = str(Path(args.pred_pose_dir) / f"pose_{idx:03d}.txt")

        if not Path(pred_pose_path).exists():
            print(f"[WARNING] Pose not found: {pred_pose_path}, skipping frame {idx}")
            continue

        metrics = evaluate_frame(
            idx=idx,
            points=points,
            diameter=diameter,
            cameras_npz=args.cameras_npz,
            mask_dir=args.mask_dir,
            pred_pose_path=pred_pose_path,
            mesh_path=args.mesh,
            width=args.width,
            height=args.height,
            sti_process=sti_process,
            check_gt_iou=args.check_gt_iou,
            check_pred_iou=args.check_pred_iou,
            iou_warn_threshold=args.iou_warn_threshold,
        )

        all_metrics.append(metrics)
        add_errors.append(metrics["add"])
        adds_errors.append(metrics["adds"])

        if "iou_gt" in metrics:
            iou_gt_values.append(metrics["iou_gt"])
        if "iou_pred" in metrics:
            iou_pred_values.append(metrics["iou_pred"])

        # Print per-frame
        iou_str = ""
        if "iou_gt" in metrics:
            iou_str += f", IoU_gt={metrics['iou_gt']:.4f}"
        if "iou_pred" in metrics:
            iou_str += f", IoU_pred={metrics['iou_pred']:.4f}"

        print(f"  Frame {idx:03d}: ADD={metrics['add']:.6f}, ADD-S={metrics['adds']:.6f}, "
              f"pass@0.1d=({metrics['add_pass_01d']}, {metrics['adds_pass_01d']}){iou_str}")

    if len(all_metrics) == 0:
        print("[ERROR] No frames evaluated.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    add_pass_rate = compute_pass_rate(add_errors, threshold_01d)
    adds_pass_rate = compute_pass_rate(adds_errors, threshold_01d)
    auc_add = compute_auc(add_errors, args.auc_max_threshold, args.n_thresholds)
    auc_adds = compute_auc(adds_errors, args.auc_max_threshold, args.n_thresholds)

    print(f"Frames evaluated:    {len(all_metrics)}")
    print(f"Diameter:            {diameter:.6f} m")
    print(f"0.1*d threshold:     {threshold_01d:.6f} m")
    print(f"\nADD success@0.1d:    {add_pass_rate * 100:.2f}%")
    print(f"ADD-S success@0.1d:  {adds_pass_rate * 100:.2f}%")
    print(f"AUC-ADD@10cm:        {auc_add * 100:.2f}%")
    print(f"AUC-ADD-S@10cm:      {auc_adds * 100:.2f}%")

    if iou_gt_values:
        print(f"\nMean IoU_gt:         {np.mean(iou_gt_values):.4f}")
    if iou_pred_values:
        print(f"Mean IoU_pred:       {np.mean(iou_pred_values):.4f}")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    if args.save:
        print("\n" + "-" * 40)
        print("Saving outputs")
        print("-" * 40)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        csv_path = out_dir / "per_frame_metrics.csv"
        fieldnames = ["idx", "add", "adds", "add_pass_01d", "adds_pass_01d"]
        if args.check_gt_iou:
            fieldnames.append("iou_gt")
        if args.check_pred_iou:
            fieldnames.append("iou_pred")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                row = {k: m.get(k, "") for k in fieldnames}
                writer.writerow(row)
        print(f"[INFO] Saved: {csv_path}")

        # Summary JSON
        summary = {
            "n_frames": len(all_metrics),
            "diameter_m": diameter,
            "threshold_01d_m": threshold_01d,
            "auc_max_threshold_m": args.auc_max_threshold,
            "add_success_01d_pct": add_pass_rate * 100,
            "adds_success_01d_pct": adds_pass_rate * 100,
            "auc_add_pct": auc_add * 100,
            "auc_adds_pct": auc_adds * 100,
        }
        if iou_gt_values:
            summary["mean_iou_gt"] = float(np.mean(iou_gt_values))
        if iou_pred_values:
            summary["mean_iou_pred"] = float(np.mean(iou_pred_values))

        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Saved: {summary_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
