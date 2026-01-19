#!/usr/bin/env python3
"""
add_normals_to_ply.py

Adds vertex normals to a triangle mesh (.ply or .obj) and saves a new .ply file.
Required for STI-Pose VisPy renderer.
"""

import argparse
from pathlib import Path
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add vertex normals to a mesh and save as PLY."
    )
    parser.add_argument(
        "--in_mesh",
        type=str,
        required=True,
        help="Input mesh (.ply or .obj).",
    )
    parser.add_argument(
        "--out_ply",
        type=str,
        required=True,
        help="Output mesh (.ply) with vertex normals.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Save PLY in ASCII format (default: binary).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    in_path = Path(args.in_mesh)
    out_path = Path(args.out_ply)

    if not in_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {in_path}")

    mesh = o3d.io.read_triangle_mesh(str(in_path))

    print(f"[INFO] Loaded mesh: {in_path}")
    print(f"       Vertices: {len(mesh.vertices)}")
    print(f"       Triangles: {len(mesh.triangles)}")

    if len(mesh.triangles) == 0:
        raise RuntimeError(
            "Mesh has no faces. STI-Pose requires a triangle mesh."
        )

    # Compute and normalize vertex normals
    mesh.compute_vertex_normals()
    mesh.normalize_normals()

    # Save
    o3d.io.write_triangle_mesh(
        str(out_path),
        mesh,
        write_ascii=args.ascii,
    )

    print(f"[INFO] Saved mesh with normals: {out_path}")


if __name__ == "__main__":
    main()
