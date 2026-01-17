#!/usr/bin/env python3
"""
run_pose_perturbed.py

Single-frame pose perturbation test for a 3D object pose pipeline (NO optimization).

Purpose:
    Starting from the ground-truth camera pose (from SPE3R world_mat_{idx}),
    apply a user-specified perturbation (dz, rx_deg, ry_deg) using STI-Pose Option B
    reduced pose parametrization:
    
    - ry_deg: azimuth angle (rotation around world Y axis)
    - rx_deg: elevation angle (pitch up/down)
    - dz: delta on radial distance (camera-object distance)
    
    The camera always looks at the target (object center at origin),
    keeping the object on the optical axis (x=y=0 in camera coords).
    
    This validates that pose composition works and that IoU is sensitive to pose changes.

Usage:
    python scripts/run_pose_perturbed.py \
        --mesh /path/to/mesh.ply \
        --cameras_npz /path/to/cameras_spe3r.npz \
        --mask_dir /path/to/masks \
        --idx 0 \
        --dz 0.1 \
        --rx_deg 5.0 \
        --ry_deg 10.0 \
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
    
    # Perturbation parameters (spherical coordinates around object)
    parser.add_argument(
        "--dz",
        type=float,
        default=0.0,
        help="Delta on radial distance (camera-object distance): radius = radius_base + dz.",
    )
    parser.add_argument(
        "--rx_deg",
        type=float,
        default=0.0,
        help="Elevation angle in degrees (pitch up/down from horizontal plane).",
    )
    parser.add_argument(
        "--ry_deg",
        type=float,
        default=0.0,
        help="Azimuth angle in degrees (rotation around world Y axis).",
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
    # Step 4: Compute base camera-to-world and base radius
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Compute base camera-to-world and base radius")
    print("=" * 60)

    # Construct base camera-to-world matrix (for reference):
    # R_wc = R.T
    # t_wc = -R.T @ t
    R_wc = R.T
    t_wc = -R.T @ t

    # Assemble homogeneous matrix cam2world_cv_base
    cam2world_cv_base = np.eye(4, dtype=np.float64)
    cam2world_cv_base[:3, :3] = R_wc
    cam2world_cv_base[:3, 3] = t_wc

    print(f"[INFO] Base camera-to-world (OpenCV convention):\n{cam2world_cv_base}")

    # Target point = object center in world coordinates (mesh bounding box center)
    target = mesh_trimesh.bounds.mean(axis=0).astype(np.float64)
    print(f"\n[INFO] Target (mesh center): {target}")

    # Compute base radius and GT spherical angles from GT camera center
    v = C - target
    radius_base = np.linalg.norm(v)
    print(f"[INFO] Base radius (GT camera distance): {radius_base:.6f}")

    # Compute GT spherical parameters using the same convention as the perturbation
    # elevation (rx0): angle above horizontal plane (arcsin of normalized Y component)
    rx0 = np.arcsin(v[1] / radius_base)
    # azimuth (ry0): rotation around world +Y axis
    # Must account for cos(rx) scaling in the forward map:
    #   x = r * sin(ry) * cos(rx)
    #   z = r * cos(ry) * cos(rx)
    # So: ry = atan2(x / (r*cos(rx)), z / (r*cos(rx)))
    cos_rx0 = np.cos(rx0)
    if abs(cos_rx0) < 1e-8:
        # Camera looking straight up or down, azimuth is undefined
        ry0 = 0.0
    else:
        ry0 = np.arctan2(v[0] / (radius_base * cos_rx0),
                         v[2] / (radius_base * cos_rx0))

    print(f"[INFO] GT spherical angles:")
    print(f"       rx0 (elevation) = {np.rad2deg(rx0):.4f} deg ({rx0:.6f} rad)")
    print(f"       ry0 (azimuth)   = {np.rad2deg(ry0):.4f} deg ({ry0:.6f} rad)")

    # =========================================================================
    # Step 5: Compute perturbed camera position on sphere (look-at)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Compute perturbed camera position on sphere")
    print("=" * 60)

    # Convert perturbation degrees to radians
    drx_rad = np.deg2rad(args.rx_deg)  # delta elevation (pitch)
    dry_rad = np.deg2rad(args.ry_deg)  # delta azimuth (around Y axis)

    print(f"[INFO] Perturbation parameters (RELATIVE to GT):")
    print(f"       dz = {args.dz} (delta on radial distance)")
    print(f"       rx_deg = {args.rx_deg} (delta elevation, {drx_rad:.6f} rad)")
    print(f"       ry_deg = {args.ry_deg} (delta azimuth, {dry_rad:.6f} rad)")

    # Apply perturbations RELATIVE to GT spherical angles
    rx = rx0 + drx_rad  # final elevation
    ry = ry0 + dry_rad  # final azimuth
    radius = max(1e-6, radius_base + args.dz)  # final radius

    print(f"\n[INFO] Final spherical parameters:")
    print(f"       rx (elevation) = {np.rad2deg(rx):.4f} deg ({rx:.6f} rad)")
    print(f"       ry (azimuth)   = {np.rad2deg(ry):.4f} deg ({ry:.6f} rad)")
    print(f"       radius = {radius:.6f}")

    # Compute camera position on sphere using spherical coordinates
    # Convention:
    #   - ry (azimuth) rotates around world +Y axis
    #   - rx (elevation) tilts up/down from the horizontal plane
    #   - Positive ry rotates camera position counterclockwise when viewed from above
    #   - Positive rx moves camera up (positive Y)
    # NOTE: When dz=rx_deg=ry_deg=0, cam_pos == C (GT camera center)
    cam_pos = target + radius * np.array([
        np.sin(ry) * np.cos(rx),  # X
        np.sin(rx),                # Y
        np.cos(ry) * np.cos(rx),  # Z
    ], dtype=np.float64)

    print(f"[INFO] New camera position: {cam_pos}")
    print(f"[DEBUG] ||C - cam_pos|| = {np.linalg.norm(C - cam_pos):.6e}")

    # =========================================================================
    # Step 6: Compute look-at rotation (world->camera, OpenCV convention)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 6: Compute look-at rotation")
    print("=" * 60)

    # Build camera axes for OpenCV convention (x right, y down, z forward)
    # Forward axis (camera +Z) points from camera to target
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    # Use GT camera up vector so that with zero perturbations, R_lookat == R
    # In OpenCV convention, camera +Y points down, so camera up is -R.T[:, 1]
    world_up = (-R.T[:, 1]).astype(np.float64)  # GT camera up (since +Y in OpenCV is down)
    world_up = world_up / np.linalg.norm(world_up)

    # Right axis (camera +X) = up x forward (normalized)
    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Camera is looking straight up or down, use fallback
        print("[WARNING] Camera looking along Y axis, using fallback right vector.")
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / right_norm

    # True up axis = forward x right
    up = np.cross(forward, right)

    # Build rotation matrix for world->camera (R_cw)
    # OpenCV camera has +Y down, so we use -up for the Y row
    # Rows of R are camera axes expressed in world coords:
    #   row 0 = right (camera +X)
    #   row 1 = -up (camera +Y points down)
    #   row 2 = forward (camera +Z)
    R_lookat = np.stack([right, -up, forward], axis=0)

    print(f"[INFO] Forward (camera +Z): {forward}")
    print(f"[INFO] Right (camera +X): {right}")
    print(f"[INFO] Up (world up projected): {up}")
    print(f"[INFO] GT camera up (world_up): {world_up}")
    print(f"\n[INFO] Look-at rotation R (world->camera):\n{R_lookat}")
    print(f"[DEBUG] ||R - R_lookat||_F = {np.linalg.norm(R - R_lookat):.6e}")
    print(f"[DEBUG] det(R_lookat) = {np.linalg.det(R_lookat):.6f}")

    # Compute translation: t = -R @ cam_pos
    t_lookat = -R_lookat @ cam_pos
    print(f"[INFO] Translation t_lookat: {t_lookat}")

    # Build camera-to-world matrix from (R_lookat, t_lookat)
    # R_wc = R_lookat.T
    # t_wc = -R_lookat.T @ t_lookat = cam_pos
    R_wc_new = R_lookat.T
    t_wc_new = -R_lookat.T @ t_lookat

    # Assemble cam2world_cv
    cam2world_cv = np.eye(4, dtype=np.float64)
    cam2world_cv[:3, :3] = R_wc_new
    cam2world_cv[:3, 3] = t_wc_new

    print(f"\n[INFO] Perturbed camera-to-world (OpenCV convention):\n{cam2world_cv}")

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
    print(f"  rx_deg:       {args.rx_deg} (elevation)")
    print(f"  ry_deg:       {args.ry_deg} (azimuth)")
    print(f"  radius_base:  {radius_base:.6f}")
    print(f"  radius:       {radius:.6f}")
    print(f"  cam_pos:      {cam_pos}")
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
            cam2world_cv=cam2world_cv,
            # Spherical look-at parameters
            target=target,
            radius_base=radius_base,
            radius=radius,
            cam_pos=cam_pos,
            R_lookat=R_lookat,
            t_lookat=t_lookat,
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
            "radius_base": float(radius_base),
            "radius": float(radius),
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
