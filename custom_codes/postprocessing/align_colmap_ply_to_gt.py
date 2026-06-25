#!/usr/bin/env python3
"""
align_colmap_ply_to_gt.py

Apply a fixed COLMAP -> GT 4x4 transform to a PLY file and save the result.

The script preserves all vertex attributes and any non-vertex elements.
It only updates vertex coordinates (x, y, z).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


T_COLMAP_TO_GT = np.array([
    [ 6.832738,  0.175016,  1.494359, -0.695939],
    [-0.012087,  6.955097, -0.759300,  0.022482],
    [-1.504524,  0.738953,  6.792674, -0.097791],
    [ 0.000000,  0.000000,  0.000000,  1.000000],
], dtype=np.float64)


def transform_vertices(vertex_data, transform: np.ndarray):
    """Apply a homogeneous transform to PLY vertex coordinates only."""
    names = vertex_data.dtype.names or ()
    if not all(axis in names for axis in ("x", "y", "z")):
        raise ValueError("The PLY vertex element must contain x, y, z fields.")

    xyz = np.column_stack([
        np.asarray(vertex_data["x"], dtype=np.float64),
        np.asarray(vertex_data["y"], dtype=np.float64),
        np.asarray(vertex_data["z"], dtype=np.float64),
        np.ones(len(vertex_data), dtype=np.float64),
    ])

    xyz_t = (transform @ xyz.T).T

    transformed = np.empty(len(vertex_data), dtype=vertex_data.dtype)
    for name in names:
        transformed[name] = vertex_data[name]

    transformed["x"] = xyz_t[:, 0].astype(transformed["x"].dtype, copy=False)
    transformed["y"] = xyz_t[:, 1].astype(transformed["y"].dtype, copy=False)
    transformed["z"] = xyz_t[:, 2].astype(transformed["z"].dtype, copy=False)
    return transformed


def transform_ply(input_path: Path, output_path: Path, transform: np.ndarray) -> None:
    """Read a PLY file, transform its vertices, and preserve all other elements."""
    ply = PlyData.read(str(input_path))

    if "vertex" not in ply:
        raise ValueError("The input PLY does not contain a vertex element.")

    out_elements = []
    for element in ply.elements:
        if element.name == "vertex":
            transformed_vertices = transform_vertices(element.data, transform)
            out_elements.append(PlyElement.describe(transformed_vertices, "vertex"))
        else:
            out_elements.append(element)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(out_elements, text=ply.text, byte_order=ply.byte_order).write(str(output_path))


def _as_4x4_matrix(values: np.ndarray, source: str) -> np.ndarray:
    """Validate and reshape a flat matrix representation into 4x4 form."""
    mat = np.asarray(values, dtype=np.float64)
    if mat.size != 16:
        raise ValueError(f"Matrix from {source} must contain exactly 16 values, found {mat.size}.")
    mat = mat.reshape(4, 4)
    if not np.isclose(mat[3, 3], 1.0):
        raise ValueError(f"Invalid homogeneous transform from {source}: element (4,4) must be 1.")
    return mat


def load_matrix_from_file(path: Path) -> np.ndarray:
    """Load a 4x4 transform from JSON or whitespace/comma-separated text."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Matrix file is empty: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("matrix", "transform", "T", "t_colmap_to_gt"):
                if key in data:
                    return _as_4x4_matrix(np.asarray(data[key], dtype=np.float64), f"{path}:{key}")
            raise ValueError(f"JSON matrix file must contain one of keys: matrix, transform, T, t_colmap_to_gt ({path})")
        return _as_4x4_matrix(np.asarray(data, dtype=np.float64), str(path))

    cleaned = text.replace(",", " ").replace(";", " ")
    values = np.fromstring(cleaned, sep=" ", dtype=np.float64)
    return _as_4x4_matrix(values, str(path))


def resolve_transform(args) -> np.ndarray:
    """Choose the transform source, giving CLI values priority over files/defaults."""
    if args.matrix_values is not None:
        return _as_4x4_matrix(np.asarray(args.matrix_values, dtype=np.float64), "--matrix_values")
    if args.matrix_file is not None:
        return load_matrix_from_file(args.matrix_file)
    return T_COLMAP_TO_GT


def parse_args():
    """Parse command-line arguments for PLY frame conversion."""
    parser = argparse.ArgumentParser(description="Transform a COLMAP PLY into the GT frame.")
    parser.add_argument("--input", type=Path, required=True, help="Input PLY in COLMAP frame")
    parser.add_argument("--output", type=Path, required=True, help="Output PLY in GT frame")
    parser.add_argument(
        "--matrix_values",
        type=float,
        nargs=16,
        default=None,
        help="Optional 4x4 transform passed as 16 numbers in row-major order",
    )
    parser.add_argument(
        "--matrix_file",
        type=Path,
        default=None,
        help="Optional path to matrix file (.txt/.json). Ignored if --matrix_values is provided",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for applying the COLMAP-to-GT PLY transform."""
    args = parse_args()
    transform = resolve_transform(args)
    transform_ply(args.input, args.output, transform)
    print(f"Saved transformed PLY to: {args.output}")


if __name__ == "__main__":
    main()
