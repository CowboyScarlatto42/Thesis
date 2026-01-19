#!/usr/bin/env python3
"""
Minimal STI-Pose runner (single frame) with:
- automatic z_near / z_far from mesh bbox diagonal
- pose correction using STI-Pose internal recentering deltaP

Outputs:
  <out>/pose_T_C_O.txt   (4x4, object-in-camera for the ORIGINAL mesh coords)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.append("external/sti_pose")
from SilhouettePE import Process


def parse_args():
    p = argparse.ArgumentParser(description="Minimal STI-Pose runner (single frame).")
    p.add_argument("--mesh", type=str, required=True, help="Mesh path (.ply with normals).")
    p.add_argument("--cameras_npz", type=str, required=True, help="cameras_spe3r.npz")
    p.add_argument("--mask", type=str, required=True, help="GT mask PNG for this frame.")
    p.add_argument("--idx", type=int, required=True, help="Frame index for world_mat_{idx}.")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--particles", type=int, default=20)
    p.add_argument("--th", type=float, default=0.02)
    p.add_argument("--out", type=str, required=True, help="Output dir.")
    return p.parse_args()


def load_intrinsics_K(cameras_npz: str, idx: int) -> np.ndarray:
    data = np.load(cameras_npz)
    key = f"world_mat_{idx}"
    if key not in data:
        keys = [k for k in data.keys() if k.startswith("world_mat_")]
        raise KeyError(f"Key '{key}' not found. Example keys: {keys[:5]}")
    P = data[key][:3, :4].astype(np.float64)
    K, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    K = K / K[2, 2]
    return K


def load_mask(mask_path: str, width: int, height: int) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to load mask: {mask_path}")
    if m.shape[0] != height or m.shape[1] != width:
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return ((m > 127).astype(np.uint8)) * 255


def compute_z_range_from_mesh(mesh_path: str) -> tuple[float, float, float]:
    """Returns (z_near, z_far, diagonal) using axis-aligned bbox diagonal."""
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not hasattr(mesh, "bounds") or mesh.bounds is None:
        raise RuntimeError(f"Could not read bounds for mesh: {mesh_path}")
    bounds = mesh.bounds  # shape (2,3): min, max
    extent = bounds[1] - bounds[0]
    diagonal = float(np.linalg.norm(extent))
    z_near = 1.5 * diagonal
    z_far = 10.0 * diagonal
    return z_near, z_far, diagonal


def main():
    args = parse_args()

    mesh_path = str(Path(args.mesh))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intrinsics + mask
    K = load_intrinsics_K(args.cameras_npz, args.idx)
    mask = load_mask(args.mask, args.width, args.height)

    # z-range from mesh
    z_near, z_far, diag = compute_z_range_from_mesh(mesh_path)
    print(f"[INFO] bbox diagonal: {diag:.6f}  -> z_near={z_near:.6f}, z_far={z_far:.6f}")

    # STI-Pose
    proc = Process((args.width, args.height), K, 1)
    proc.set_model(mesh_path)

    # IMPORTANT: STI-Pose internal recenter (mean of vertices) stored as renderer.deltaP
    try:
        deltaP = np.array(proc.renderer.deltaP, dtype=np.float64).reshape(3,)
    except Exception as e:
        raise RuntimeError(
            "Could not access proc.renderer.deltaP. "
            "This runner is intentionally written without fallback."
        ) from e

    proc.set_ref(mask)

    # Pose estimation (returns pose for the internally-centered mesh)
    T_centered, pose_internal = proc.pose_es(z_near, z_far, args.iters, args.particles, args.th)

    # Correct translation back to ORIGINAL mesh coordinates:
    # If STI-Pose centered mesh by: X' = X - c
    # Then: (R X' + t_centered) = (R (X - c) + t_centered) = (R X + (t_centered - R c))
    R = T_centered[:3, :3]
    t_centered = T_centered[:3, 3]
    t_orig = t_centered - R @ deltaP

    T_C_O = np.eye(4, dtype=np.float64)
    T_C_O[:3, :3] = R
    T_C_O[:3, 3] = t_orig

    # Save corrected pose
    np.savetxt(out_dir / "pose_T_C_O.txt", T_C_O, fmt="%.10f")
    print(f"[INFO] Saved: {out_dir / 'pose_T_C_O.txt'}")

    # Save raw pose_internal (robust: numpy array or JSON)
    if isinstance(pose_internal, np.ndarray):
        np.savetxt(out_dir / "pose_internal.txt", pose_internal, fmt="%.10f")
        print(f"[INFO] Saved: {out_dir / 'pose_internal.txt'}")
    else:
        try:
            arr = np.array(pose_internal, dtype=np.float64)
            np.savetxt(out_dir / "pose_internal.txt", arr, fmt="%.10f")
            print(f"[INFO] Saved: {out_dir / 'pose_internal.txt'}")
        except (ValueError, TypeError):
            with open(out_dir / "pose_internal.json", "w") as f:
                json.dump(pose_internal, f, indent=2)
            with open(out_dir / "pose_internal.txt", "w") as f:
                f.write("# See pose_internal.json\n")
            print(f"[INFO] Saved: {out_dir / 'pose_internal.json'} (with pose_internal.txt pointer)")


if __name__ == "__main__":
    main()
