#!/usr/bin/env python3
"""
runner_sti_pose_vanilla.py

Vanilla STI-Pose baseline

Runs STI-Pose exactly as implemented in the official repository,
adapted to the SPE3R + NeuS setup, without modifying the STI-Pose submodule.

Usage:
    python scripts/runner_sti_pose_vanilla.py \
        --mesh /path/to/mesh.ply \
        --cameras_npz /path/to/cameras_spe3r.npz \
        --mask /path/to/mask.png \
        --idx 0 \
        --width 256 \
        --height 256 \
        --iters 100 \
        --particles 20 \
        --th 0.02 \
        --save_output \
        --out /path/to/output
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

# Import STI-Pose from external submodule
sys.path.append("external/sti_pose")
from SilhouettePE import Process


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Vanilla STI-Pose baseline for SPE3R + NeuS setup."
    )
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to NeuS .ply mesh file.",
    )
    parser.add_argument(
        "--cameras_npz",
        type=str,
        required=True,
        help="Path to cameras_spe3r.npz file.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to GT silhouette mask (PNG).",
    )
    parser.add_argument(
        "--idx",
        type=int,
        required=True,
        help="Frame index (integer) selecting world_mat_{idx}.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Image width (default: 256).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Image height (default: 256).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="PSO iterations (default: 100).",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=20,
        help="PSO particle count (default: 20).",
    )
    parser.add_argument(
        "--th",
        type=float,
        default=0.02,
        help="STI-Pose termination threshold (default: 0.02).",
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        help="If set, save outputs to disk.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output",
        help="Output directory (used only if --save_output is set).",
    )
    return parser.parse_args()


def load_camera(cameras_npz: str, idx: int) -> tuple:
    """
    Load camera intrinsics and extrinsics from SPE3R cameras_npz file.

    Args:
        cameras_npz: Path to cameras_spe3r.npz file.
        idx: Frame index.

    Returns:
        (K, R, t): K is 3x3 normalized intrinsic matrix,
                   R is 3x3 rotation matrix (world->camera),
                   t is 3x1 translation vector (world->camera).
    """
    cameras_path = Path(cameras_npz)
    if not cameras_path.exists():
        print(f"[ERROR] Cameras file not found: {cameras_path}")
        sys.exit(1)

    cameras_data = np.load(str(cameras_path))
    world_mat_key = f"world_mat_{idx}"

    if world_mat_key not in cameras_data:
        available_keys = [k for k in cameras_data.keys() if k.startswith("world_mat_")]
        print(f"[ERROR] Key '{world_mat_key}' not found in cameras file.")
        print(f"        Available world_mat keys: {available_keys[:10]}...")
        sys.exit(1)

    world_mat = cameras_data[world_mat_key]

    # Extract projection matrix P = world_mat[:3, :4]
    P = world_mat[:3, :4].astype(np.float64)

    # Decompose using cv2.decomposeProjectionMatrix
    K, R, T_homog, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

    # Normalize intrinsics so that K[2,2] = 1
    K = K / K[2, 2]

    # Compute camera center C in world coordinates (from homogeneous T)
    C = T_homog[:3, 0] / T_homog[3, 0]

    # Compute translation t = -R @ C so that X_cam = R @ X_world + t
    t = -R @ C

    return K, R, t


def load_mask(mask_path: str, width: int, height: int) -> np.ndarray:
    """
    Load and preprocess GT silhouette mask.

    Args:
        mask_path: Path to mask PNG file.
        width: Target width.
        height: Target height.

    Returns:
        Binary mask as uint8 {0, 255}.
    """
    mask_file = Path(mask_path)
    if not mask_file.exists():
        print(f"[ERROR] Mask file not found: {mask_file}")
        sys.exit(1)

    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[ERROR] Failed to load mask: {mask_file}")
        sys.exit(1)

    # Resize if necessary
    if mask.shape[0] != height or mask.shape[1] != width:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # Convert to binary uint8 {0, 255}
    mask = ((mask > 127).astype(np.uint8)) * 255

    return mask


def compute_depth_range(mesh_path: str) -> tuple:
    """
    Compute depth search range from mesh bounding box diagonal.

    Args:
        mesh_path: Path to .ply mesh file.

    Returns:
        (z_near, z_far, diagonal): Depth range and bounding box diagonal.
    """
    mesh_file = Path(mesh_path)
    if not mesh_file.exists():
        print(f"[ERROR] Mesh file not found: {mesh_file}")
        sys.exit(1)

    mesh = o3d.io.read_triangle_mesh(str(mesh_file))
    if mesh.is_empty():
        print(f"[ERROR] Failed to load mesh or mesh is empty: {mesh_file}")
        sys.exit(1)

    # Compute axis-aligned bounding box diagonal
    aabb = mesh.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    diagonal = np.linalg.norm(extent)

    # Set depth search range
    z_near = 1.5 * diagonal
    z_far = 10.0 * diagonal

    return z_near, z_far, diagonal


def main():
    args = parse_args()

    # =========================================================================
    # Step 1: Load camera intrinsics and extrinsics
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Load camera intrinsics and extrinsics")
    print("=" * 60)

    K, R, t = load_camera(args.cameras_npz, args.idx)
    print(f"[INFO] Intrinsics K (normalized):\n{K}")
    print(f"\n[INFO] Rotation R (world->camera):\n{R}")
    print(f"\n[INFO] Translation t (world->camera): {t.flatten()}")

    # =========================================================================
    # Step 2: Load GT mask
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Load GT mask")
    print("=" * 60)

    mask = load_mask(args.mask, args.width, args.height)
    print(f"[INFO] Loaded mask: {args.mask}")
    print(f"       Shape: {mask.shape}, Non-zero pixels: {np.sum(mask > 0)}")

    # =========================================================================
    # Step 3: Compute depth search range
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Compute depth search range")
    print("=" * 60)

    z_near, z_far, diagonal = compute_depth_range(args.mesh)
    print(f"[INFO] Mesh bounding box diagonal: {diagonal:.6f}")
    print(f"[INFO] z_near: {z_near:.6f}")
    print(f"[INFO] z_far: {z_far:.6f}")

    # =========================================================================
    # Step 4: Setup STI-Pose
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Setup STI-Pose")
    print("=" * 60)

    img_size = (args.width, args.height)
    p = Process(img_size, K, 1)
    p.set_model(args.mesh)
    p.set_ref(mask)
    print(f"[INFO] STI-Pose initialized with image size: {img_size}")
    print(f"[INFO] Model set: {args.mesh}")

    # =========================================================================
    # Step 5: Run pose estimation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Run pose estimation (vanilla STI-Pose)")
    print("=" * 60)

    print(f"[INFO] Running PSO with iters={args.iters}, particles={args.particles}, th={args.th}")

    pose_pred, pose_internal = p.pose_es(
        z_near,
        z_far,
        args.iters,
        args.particles,
        args.th
    )

    # pose_pred is T_C_O (object in camera frame)
    T_C_O = pose_pred

    # Build T_C_W (camera <- world) from R, t
    T_C_W = np.eye(4)
    T_C_W[:3, :3] = R
    T_C_W[:3, 3] = t.reshape(3,)

    # Invert to get T_W_C (world <- camera)
    T_W_C = np.linalg.inv(T_C_W)

    # Compute object in world frame: T_W_O = T_W_C @ T_C_O
    T_W_O = T_W_C @ T_C_O

    print("\n[INFO] T_C_O (object in camera frame):")
    print(T_C_O)
    print("\n[INFO] T_W_O (object in world frame):")
    print(T_W_O)

    # =========================================================================
    # Step 6: Final render
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 6: Final render")
    print("=" * 60)

    rendered = p.render_silhouette(pose_internal)
    print(f"[INFO] Rendered silhouette shape: {rendered.shape}")

    # =========================================================================
    # Step 7: Output handling (conditional)
    # =========================================================================
    if args.save_output:
        print("\n" + "=" * 60)
        print("Step 7: Save outputs")
        print("=" * 60)

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save rendered silhouette
        rendered_path = out_dir / "rendered.png"
        cv2.imwrite(str(rendered_path), rendered)
        print(f"[INFO] Saved rendered silhouette: {rendered_path}")

        # Save T_C_O (object in camera frame)
        pose_T_C_O_path = out_dir / "pose_T_C_O.txt"
        np.savetxt(str(pose_T_C_O_path), T_C_O, fmt="%.10f")
        print(f"[INFO] Saved T_C_O (object in camera): {pose_T_C_O_path}")

        # Save T_W_O (object in world frame)
        pose_T_W_O_path = out_dir / "pose_T_W_O.txt"
        np.savetxt(str(pose_T_W_O_path), T_W_O, fmt="%.10f")
        print(f"[INFO] Saved T_W_O (object in world): {pose_T_W_O_path}")

        # Save parameters
        params_path = out_dir / "params.txt"
        with open(params_path, "w") as f:
            f.write(f"z_near: {z_near}\n")
            f.write(f"z_far: {z_far}\n")
            f.write(f"iters: {args.iters}\n")
            f.write(f"particles: {args.particles}\n")
            f.write(f"threshold: {args.th}\n")
        print(f"[INFO] Saved parameters: {params_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
