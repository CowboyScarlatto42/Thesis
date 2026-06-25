import json
import os
import argparse

import numpy as np
import trimesh


"""
Build the NeuS normalization matrix from a ground-truth mesh.

NeuS expects scenes to be represented in a normalized coordinate system, usually
close to the unit sphere. This script computes a single `scale_mat` from the
OBJ mesh bounding box and saves it as JSON so it can later be inserted into
`cameras_sphere.npz` by `spe3r_to_neus_data.py`.

The matrix follows the IDR/NeuS convention:

    x_world = scale_mat @ x_normalized

so the inverse maps world/CORTO coordinates into the normalized NeuS frame.
"""


def load_geometry(geometry_json_path):
    """Load camera and body positions from CORTO geometry.json."""
    with open(geometry_json_path, "r") as f:
        geometry = json.load(f)

    cam_positions = np.asarray(geometry["camera"]["position"], dtype=np.float64)
    body_positions = np.asarray(geometry["body"]["position"], dtype=np.float64)

    if cam_positions.ndim != 2 or cam_positions.shape[1] != 3:
        raise ValueError("geometry['camera']['position'] deve avere shape [N, 3]")
    if body_positions.ndim != 2 or body_positions.shape[1] != 3:
        raise ValueError("geometry['body']['position'] deve avere shape [N, 3]")

    return cam_positions, body_positions


def load_obj_vertices(obj_path):
    """Load mesh vertices from the OBJ used as the normalization reference."""
    mesh = trimesh.load(obj_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Il file OBJ non e' stato caricato come mesh: {obj_path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        raise ValueError(f"Nessun vertice trovato nell'OBJ: {obj_path}")

    return vertices


def compute_scale_mat_from_obj(vertices, margin=1.05):
    """Compute center and radius so the object fits in the NeuS unit sphere."""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)

    radius = np.linalg.norm(vertices - center, axis=1).max()
    radius *= float(margin)

    if radius <= 0:
        raise ValueError("Raggio non valido")

    scale_mat = np.eye(4, dtype=np.float64)
    scale_mat[:3, :3] = np.eye(3, dtype=np.float64) * radius
    scale_mat[:3, 3] = center

    return scale_mat, center, radius


def check_camera_distances(cam_positions, center, radius):
    """Report camera distances after applying the proposed normalization."""
    cam_dists = np.linalg.norm(cam_positions - center, axis=1)
    cam_dists_norm = cam_dists / radius

    return {
        "min": float(cam_dists_norm.min()),
        "max": float(cam_dists_norm.max()),
        "mean": float(cam_dists_norm.mean()),
    }


def save_scale_mat(output_json_path, scale_mat):
    """Save the 4x4 scale matrix in a small JSON wrapper."""
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump({"scale_mat": scale_mat.tolist()}, f, indent=2)


def main(geometry_json_path, obj_path, output_json_path, margin=1.05):
    """Compute and write the scale matrix, printing basic sanity diagnostics."""
    cam_positions, _ = load_geometry(geometry_json_path)
    vertices = load_obj_vertices(obj_path)

    scale_mat, center, radius = compute_scale_mat_from_obj(vertices, margin=margin)
    cam_stats = check_camera_distances(cam_positions, center, radius)

    print("=" * 60)
    print("scale_mat computation")
    print("=" * 60)
    print(f"OBJ path:                {obj_path}")
    print(f"Centro bounding box:      {center}")
    print(f"Raggio oggetto + margine: {radius:.6f}")
    print(f"Camera dist min norm:     {cam_stats['min']:.6f}")
    print(f"Camera dist max norm:     {cam_stats['max']:.6f}")
    print(f"Camera dist mean norm:    {cam_stats['mean']:.6f}")

    if cam_stats["min"] < 1.05:
        print("[WARN] Alcune camere sono molto vicine alla sfera unitaria")
    if cam_stats["max"] > 6.0:
        print("[WARN] Alcune camere sono molto lontane dalla sfera unitaria")

    save_scale_mat(output_json_path, scale_mat)
    print(f"Salvato: {output_json_path}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a NeuS scale_mat from geometry.json and an OBJ mesh")
    parser.add_argument("--geometry", required=True, help="Path to geometry.json")
    parser.add_argument("--obj", required=True, help="Path to the OBJ mesh used to define center and radius")
    parser.add_argument("--output", required=True, help="Path where scale_mat.json will be saved")
    parser.add_argument(
        "--margin",
        type=float,
        default=1.05,
        help="Extra margin applied to the object radius (default: 1.05)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        geometry_json_path=args.geometry,
        obj_path=args.obj,
        output_json_path=args.output,
        margin=args.margin,
    )
