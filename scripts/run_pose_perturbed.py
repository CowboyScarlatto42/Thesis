#!/usr/bin/env python3
"""
run_pose_perturbed.py

Single-frame pose perturbation test for a 3D object pose pipeline (NO optimization).

Purpose:
    Starting from the ground-truth camera pose (from SPE3R world_mat_{idx}),
    apply a user-specified delta pose (dz, rx, ry) to the camera pose,
    render the silhouette of a NeuS mesh, and compute IoU vs the GT mask.
    This validates that pose composition works and that IoU is sensitive to pose changes.

Usage:
    python scripts/run_pose_perturbed.py \
        --mesh /path/to/mesh.ply \
        --cameras_npz /path/to/cameras_spe3r.npz \
        --mask_dir /path/to/masks \
        --idx 0 \
        --dz 0.1 \
        --rx_deg 5.0 \
        --ry_deg 0.0 \
        --show

"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Single-frame pose perturbation test for NeuS mesh with SPE3R cameras."
    )
    
    # Data inputs
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to a NeuS .ply mesh file.",
    )
    parser.add_argument(
        "--cameras_npz",
        type=str,
        required=True,
        help="Path to cameras_spe3r.npz file.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Directory containing GT masks named {idx:03d}.png.",
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
        "--use_egl",
        action="store_true",
        help="If set, use EGL platform for headless rendering (set PYOPENGL_PLATFORM=egl).",
    )
    
    # Perturbation parameters (camera-frame)
    parser.add_argument(
        "--dz",
        type=float,
        default=0.0,
        help="Translation along camera +Z axis (forward) in OpenCV camera coordinates.",
    )
    parser.add_argument(
        "--rx_deg",
        type=float,
        default=0.0,
        help="Rotation around camera X axis in degrees.",
    )
    parser.add_argument(
        "--ry_deg",
        type=float,
        default=0.0,
        help="Rotation around camera Y axis in degrees.",
    )
    parser.add_argument(
        "--rot_order",
        type=str,
        choices=["rx_then_ry", "ry_then_rx"],
        default="rx_then_ry",
        help="Order of rotation application (default: rx_then_ry).",
    )
    
    # Rendering
    parser.add_argument(
        "--znear",
        type=float,
        default=0.01,
        help="Near clipping plane (default: 0.01).",
    )
    parser.add_argument(
        "--zfar",
        type=float,
        default=1e6,
        help="Far clipping plane (default: 1e6).",
    )
    
    # Output control
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to disk.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots on screen (matplotlib).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (required if --save is set).",
    )
    
    args = parser.parse_args()
    
    # Validate output arguments
    if not args.save and not args.show:
        parser.error("At least one of --save or --show must be specified.")
    
    if args.save and args.out_dir is None:
        parser.error("--out_dir is required when --save is set.")
    
    return args


def rotation_matrix_x(angle_rad):
    """Create a 3x3 rotation matrix around X axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c],
    ], dtype=np.float64)


def rotation_matrix_y(angle_rad):
    """Create a 3x3 rotation matrix around Y axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def main():
    args = parse_args()

    # =========================================================================
    # EGL Platform Setup
    # =========================================================================
    if args.use_egl:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        print("[INFO] Using EGL platform for headless rendering.")

    # Import pyrender and trimesh after potential EGL setup
    import pyrender
    import trimesh

    # =========================================================================
    # Step 1: Load GT mask
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Load GT mask")
    print("=" * 60)

    mask_path = Path(args.mask_dir) / f"{args.idx:03d}.png"
    if not mask_path.exists():
        print(f"[ERROR] GT mask not found: {mask_path}")
        sys.exit(1)

    gt_mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask_raw is None:
        print(f"[ERROR] Failed to load GT mask: {mask_path}")
        sys.exit(1)

    # Resize to (width, height) using nearest-neighbor if needed
    if gt_mask_raw.shape[0] != args.height or gt_mask_raw.shape[1] != args.width:
        gt_mask_raw = cv2.resize(
            gt_mask_raw,
            (args.width, args.height),
            interpolation=cv2.INTER_NEAREST,
        )
        print(f"[INFO] Resized GT mask to ({args.width}, {args.height}).")

    # Binarize with threshold > 127 → {0, 1}
    gt_mask = (gt_mask_raw > 127).astype(np.uint8)
    print(f"[INFO] Loaded GT mask: {mask_path}")
    print(f"       Shape: {gt_mask.shape}, Non-zero pixels: {np.sum(gt_mask)}")

    # =========================================================================
    # Step 2: Load and decompose projection matrix
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Load and decompose projection matrix")
    print("=" * 60)

    cameras_path = Path(args.cameras_npz)
    if not cameras_path.exists():
        print(f"[ERROR] Cameras file not found: {cameras_path}")
        sys.exit(1)

    cameras_data = np.load(str(cameras_path))
    world_mat_key = f"world_mat_{args.idx}"

    if world_mat_key not in cameras_data:
        available_keys = [k for k in cameras_data.keys() if k.startswith("world_mat_")]
        print(f"[ERROR] Key '{world_mat_key}' not found in cameras file.")
        print(f"        Available world_mat keys: {available_keys[:10]}...")
        sys.exit(1)

    world_mat = cameras_data[world_mat_key]
    print(f"[INFO] Loaded {world_mat_key} with shape: {world_mat.shape}")

    # Extract projection matrix P = world_mat[:3, :4]
    P = world_mat[:3, :4].astype(np.float64)
    print(f"[INFO] Projection matrix P:\n{P}")

    # Decompose using cv2.decomposeProjectionMatrix
    K, R, C_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

    # Normalize intrinsics so that K[2,2] = 1
    K = K / K[2, 2]

    # Compute camera center C in world coordinates (from homogeneous C_h)
    C = C_h[:3, 0] / C_h[3, 0]

    # Compute translation t = -R @ C so that X_cam = R @ X_world + t
    t = -R @ C

    print(f"\n[INFO] Intrinsics K (normalized):\n{K}")
    print(f"\n[INFO] Rotation R:\n{R}")
    print(f"\n[INFO] Camera center C (world coords): {C}")
    print(f"\n[INFO] Translation t: {t}")

    # Extract focal lengths and principal point
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    print(f"\n[INFO] Intrinsic parameters: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")

    # =========================================================================
    # Step 3: Load mesh
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Load mesh")
    print("=" * 60)

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"[ERROR] Mesh file not found: {mesh_path}")
        sys.exit(1)

    # Load the NeuS mesh using trimesh with process=False
    # Do NOT modify or normalize the mesh
    mesh_trimesh = trimesh.load(str(mesh_path), process=False)
    
    # Handle case where loaded object is a trimesh.Scene
    if isinstance(mesh_trimesh, trimesh.Scene):
        mesh_trimesh = trimesh.util.concatenate(tuple(mesh_trimesh.geometry.values()))
    
    print(f"[INFO] Loaded mesh: {mesh_path}")
    print(f"       Vertices: {len(mesh_trimesh.vertices)}, Faces: {len(mesh_trimesh.faces)}")
    print(f"       Bounding box min: {mesh_trimesh.bounds[0]}")
    print(f"       Bounding box max: {mesh_trimesh.bounds[1]}")

    # =========================================================================
    # Step 4: Build base camera-to-world matrix (OpenCV convention)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Build base camera-to-world matrix")
    print("=" * 60)

    # Construct camera-to-world matrix:
    # R_wc = R.T
    # t_wc = -R.T @ t
    R_wc = R.T
    t_wc = -R.T @ t

    # Assemble homogeneous matrix cam2world_cv_base
    cam2world_cv_base = np.eye(4, dtype=np.float64)
    cam2world_cv_base[:3, :3] = R_wc
    cam2world_cv_base[:3, 3] = t_wc

    print(f"[INFO] Base camera-to-world (OpenCV convention):\n{cam2world_cv_base}")

    # =========================================================================
    # Step 5: Build delta transform in camera frame
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Build delta transform in camera frame")
    print("=" * 60)

    # Convert degrees to radians
    rx_rad = np.deg2rad(args.rx_deg)
    ry_rad = np.deg2rad(args.ry_deg)

    print(f"[INFO] Perturbation parameters:")
    print(f"       dz = {args.dz}")
    print(f"       rx_deg = {args.rx_deg} ({rx_rad:.6f} rad)")
    print(f"       ry_deg = {args.ry_deg} ({ry_rad:.6f} rad)")
    print(f"       rot_order = {args.rot_order}")

    # Define rotation matrices Rx(rx) and Ry(ry)
    Rx = rotation_matrix_x(rx_rad)
    Ry = rotation_matrix_y(ry_rad)

    # Combine according to --rot_order
    if args.rot_order == "rx_then_ry":
        R_delta = Ry @ Rx  # Apply Rx first, then Ry
    else:  # ry_then_rx
        R_delta = Rx @ Ry  # Apply Ry first, then Rx

    # Translation vector is [0, 0, dz] in camera coordinates
    t_delta = np.array([0.0, 0.0, args.dz], dtype=np.float64)

    # Assemble Delta_cam_cv (4x4) that maps camera frame to camera frame
    Delta_cam_cv = np.eye(4, dtype=np.float64)
    Delta_cam_cv[:3, :3] = R_delta
    Delta_cam_cv[:3, 3] = t_delta

    print(f"\n[INFO] Delta rotation matrix:\n{R_delta}")
    print(f"\n[INFO] Delta translation: {t_delta}")
    print(f"\n[INFO] Delta_cam_cv:\n{Delta_cam_cv}")

    # =========================================================================
    # Step 6: Apply delta to camera pose
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 6: Apply delta to camera pose")
    print("=" * 60)

    # Because delta is expressed in the camera frame, update pose as:
    # cam2world_cv = cam2world_cv_base @ Delta_cam_cv
    cam2world_cv = cam2world_cv_base @ Delta_cam_cv

    print(f"[INFO] Perturbed camera-to-world (OpenCV convention):\n{cam2world_cv}")

    # =========================================================================
    # Step 7: Convert OpenCV → OpenGL camera axes
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 7: Convert OpenCV → OpenGL camera axes")
    print("=" * 60)

    # Apply axis conversion:
    # OpenCV: X right, Y down, Z forward
    # OpenGL: X right, Y up, Z backward
    CV_TO_GL = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=np.float64)

    # Final pose: cam2world_gl = cam2world_cv @ CV_TO_GL
    cam2world_gl = cam2world_cv @ CV_TO_GL

    print(f"[INFO] CV_TO_GL conversion matrix:\n{CV_TO_GL}")
    print(f"\n[INFO] Camera-to-world (OpenGL convention):\n{cam2world_gl}")

    # =========================================================================
    # Step 8: Render silhouette (pyrender)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 8: Render silhouette (pyrender)")
    print("=" * 60)

    # Create pyrender scene
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])

    # Add mesh to scene
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh)
    scene.add(mesh_pyrender)

    # Create camera with intrinsics from K
    camera = pyrender.IntrinsicsCamera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        znear=args.znear,
        zfar=args.zfar,
    )

    # Add camera to scene with the computed pose
    scene.add(camera, pose=cam2world_gl)

    print(f"[INFO] Created IntrinsicsCamera with fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    print(f"       znear={args.znear}, zfar={args.zfar}")

    # Create offscreen renderer
    renderer = pyrender.OffscreenRenderer(
        viewport_width=args.width,
        viewport_height=args.height,
    )

    try:
        # Render depth only
        depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        if isinstance(depth, (tuple, list)):
            depth = depth[-1]
    finally:
        # Clean up renderer
        renderer.delete()

    # Convert depth to binary silhouette using depth > 0
    rendered_silhouette = (depth > 0).astype(np.uint8)

    print(f"[INFO] Rendered silhouette shape: {rendered_silhouette.shape}")
    print(f"       Non-zero pixels: {np.sum(rendered_silhouette)}")
    print(f"       Depth range: [{depth[depth > 0].min() if np.any(depth > 0) else 'N/A'}, "
          f"{depth.max() if np.any(depth > 0) else 'N/A'}]")

    # =========================================================================
    # Step 9: Compute IoU
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 9: Compute IoU")
    print("=" * 60)

    intersection = np.logical_and(gt_mask, rendered_silhouette).sum()
    union = np.logical_or(gt_mask, rendered_silhouette).sum()

    if union == 0:
        iou = 0.0
        print("[WARNING] Union is zero - both masks are empty!")
    else:
        iou = intersection / union

    print(f"[INFO] Intersection: {intersection}")
    print(f"[INFO] Union: {union}")
    print(f"[INFO] IoU: {iou:.6f}")

    # =========================================================================
    # Step 10: Output results
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 10: Output results")
    print("=" * 60)

    # Print concise summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"  idx:          {args.idx}")
    print(f"  dz:           {args.dz}")
    print(f"  rx_deg:       {args.rx_deg}")
    print(f"  ry_deg:       {args.ry_deg}")
    print(f"  rot_order:    {args.rot_order}")
    print(f"  IoU:          {iou:.6f}")
    print(f"  Intersection: {intersection}")
    print(f"  Union:        {union}")
    print("-" * 60)

    # Save results if --save is set
    if args.save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save gt_{idx:03d}.png
        gt_out_path = out_dir / f"gt_{args.idx:03d}.png"
        cv2.imwrite(str(gt_out_path), gt_mask * 255)
        print(f"[INFO] Saved GT mask: {gt_out_path}")

        # Save pred_{idx:03d}.png
        pred_out_path = out_dir / f"pred_{args.idx:03d}.png"
        cv2.imwrite(str(pred_out_path), rendered_silhouette * 255)
        print(f"[INFO] Saved predicted silhouette: {pred_out_path}")

        # Save overlay_{idx:03d}_iou_{iou:.4f}.png
        # Overlay convention: Red channel = GT mask, Green channel = rendered silhouette
        overlay = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        overlay[:, :, 2] = gt_mask * 255  # Red channel = GT mask
        overlay[:, :, 1] = rendered_silhouette * 255  # Green channel = rendered silhouette
        overlay_out_path = out_dir / f"overlay_{args.idx:03d}_iou_{iou:.4f}.png"
        cv2.imwrite(str(overlay_out_path), overlay)
        print(f"[INFO] Saved overlay: {overlay_out_path}")

        # Save pose_{idx:03d}.npz
        pose_out_path = out_dir / f"pose_{args.idx:03d}.npz"
        np.savez(
            str(pose_out_path),
            P=P,
            K=K,
            R=R,
            t=t,
            C=C,
            cam2world_cv_base=cam2world_cv_base,
            Delta_cam_cv=Delta_cam_cv,
            cam2world_cv=cam2world_cv,
        )
        print(f"[INFO] Saved pose parameters: {pose_out_path}")

        # Save metrics_{idx:03d}.json
        metrics = {
            "iou": float(iou),
            "intersection": int(intersection),
            "union": int(union),
            "dz": float(args.dz),
            "rx_deg": float(args.rx_deg),
            "ry_deg": float(args.ry_deg),
            "rot_order": args.rot_order,
        }
        metrics_out_path = out_dir / f"metrics_{args.idx:03d}.json"
        with open(metrics_out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved metrics: {metrics_out_path}")

    # Show results if --show is set
    if args.show:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # GT mask
        axes[0].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("GT Mask")
        axes[0].axis("off")

        # Predicted silhouette
        axes[1].imshow(rendered_silhouette, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Predicted Silhouette")
        axes[1].axis("off")

        # Overlay (Red=GT, Green=Pred)
        overlay_rgb = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        overlay_rgb[:, :, 0] = gt_mask * 255  # Red channel = GT mask
        overlay_rgb[:, :, 1] = rendered_silhouette * 255  # Green channel = Pred
        axes[2].imshow(overlay_rgb)
        axes[2].set_title(f"Overlay (IoU={iou:.4f})")
        axes[2].axis("off")

        fig.suptitle(
            f"Frame {args.idx} | dz={args.dz}, rx={args.rx_deg}°, ry={args.ry_deg}° | IoU={iou:.4f}",
            fontsize=14,
        )

        plt.tight_layout()
        plt.show()

    print("\n[DONE] Pose perturbation test completed.")


if __name__ == "__main__":
    main()
