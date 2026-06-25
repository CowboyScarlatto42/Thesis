#!/usr/bin/env python3
"""
Generate diagnostic plots from align_colmap_orbits_to_corto.py outputs.

The plots compare the CORTO ground-truth camera trajectory with the COLMAP
trajectory after Sim(3) alignment. They are used to inspect whether the aligned
poses are geometrically plausible before building the combined NeuS dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FULL_GEOMETRIES = {
    "orbit1": REPO_ROOT / "corto" / "input" / "S10_Spacecraft_Complex_Light" / "geometry" / "geometry.json",
    "orbit2": REPO_ROOT / "corto" / "input" / "S10_Spacecraft_Orbit_2" / "geometry" / "geometry.json",
}


@dataclass
class OrbitDiagnostics:
    tag: str
    names: List[str]
    gt_centers: np.ndarray
    gt_centers_full: np.ndarray
    aligned_centers: np.ndarray
    position_errors: np.ndarray
    view_errors_deg: np.ndarray
    sparse_points: Optional[np.ndarray]


def read_json(path: Path) -> object:
    """Read a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_ascii_ply(path: Path) -> Optional[np.ndarray]:
    """Read an ASCII PLY point cloud if it exists."""
    if not path.is_file():
        return None
    points: List[List[float]] = []
    in_header = True
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if in_header:
                if line == "end_header":
                    in_header = False
                continue
            if line:
                x, y, z = line.split()[:3]
                points.append([float(x), float(y), float(z)])
    return np.asarray(points, dtype=float).reshape(-1, 3)


def read_geometry_camera_positions(path: Path) -> np.ndarray:
    """Load camera positions from a CORTO geometry.json file."""
    payload = read_json(path)
    try:
        positions = np.asarray(payload["camera"]["position"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Expected geometry['camera']['position'] in {path}") from exc
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected an Nx3 camera position array in {path}, got {positions.shape}")
    return positions


def load_full_gt_centers(
    orbit_dir: Path,
    filtered_gt_centers: np.ndarray,
    full_geometry_path: Optional[Path] = None,
) -> np.ndarray:
    """Prefer a full geometry trajectory for plotting when available."""
    if full_geometry_path is not None and full_geometry_path.is_file():
        return read_geometry_camera_positions(full_geometry_path)

    summary_path = orbit_dir / "alignment_summary.json"
    if not summary_path.is_file():
        return filtered_gt_centers

    summary = read_json(summary_path)
    if not isinstance(summary, dict) or "geometry_path" not in summary:
        return filtered_gt_centers

    geometry_path = Path(str(summary["geometry_path"])).expanduser()
    if not geometry_path.is_absolute():
        geometry_path = (summary_path.parent / geometry_path).resolve()
    if not geometry_path.is_file():
        return filtered_gt_centers

    return read_geometry_camera_positions(geometry_path)


def load_orbit(
    alignment_root: Path,
    tag: str,
    full_geometry_path: Optional[Path] = None,
) -> OrbitDiagnostics:
    """Load all diagnostics for one aligned orbit."""
    orbit_dir = alignment_root / tag
    records = read_json(orbit_dir / "aligned_poses_all_fit.json")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Invalid or empty aligned pose list: {orbit_dir}")
    gt_centers = np.asarray([row["camera_center_gt"] for row in records], dtype=float)
    return OrbitDiagnostics(
        tag=tag,
        names=[str(row["stem"]) for row in records],
        gt_centers=gt_centers,
        gt_centers_full=load_full_gt_centers(orbit_dir, gt_centers, full_geometry_path),
        aligned_centers=np.asarray([row["camera_center_aligned"] for row in records], dtype=float),
        position_errors=np.asarray([row["position_error"] for row in records], dtype=float),
        view_errors_deg=np.asarray([row["view_direction_error_deg"] for row in records], dtype=float),
        sparse_points=read_ascii_ply(orbit_dir / "aligned_sparse_points_all_fit.ply"),
    )


def downsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Randomly downsample a point cloud for readable plots."""
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=max_points, replace=False)]


def trajectory_line_points(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Break the line at duplicated endpoints and anomalous trajectory jumps."""
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if len(reference) > 1 and np.allclose(reference[0], reference[-1], rtol=1e-5, atol=1e-8):
        points = points[:-1]
        reference = reference[:-1]

    if len(reference) < 3:
        return points

    step_lengths = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    positive_steps = step_lengths[step_lengths > 1e-12]
    if not len(positive_steps):
        return points

    median_step = float(np.median(positive_steps))
    mad = float(np.median(np.abs(positive_steps - median_step)))
    jump_threshold = max(5.0 * median_step, median_step + 10.0 * 1.4826 * mad)
    jump_indices = np.flatnonzero(step_lengths > jump_threshold) + 1

    if not len(jump_indices):
        return points

    pieces = np.split(points, jump_indices)
    separator = np.full((1, 3), np.nan)
    line_parts = [pieces[0]]
    for piece in pieces[1:]:
        line_parts.extend([separator, piece])
    return np.vstack(line_parts)


def plot_trajectories(
    results: Sequence[OrbitDiagnostics], output: Path, max_sparse_points: int
) -> None:
    """Plot GT and aligned camera trajectories in the CORTO frame."""
    colors = {
        "orbit1": "#0072B2",
        "orbit2": "#D55E00",
    }
    labels = {
        "orbit1": "safety ellipse 1",
        "orbit2": "safety ellipse 2",
    }

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for result in results:
        color = colors.get(result.tag)
        label = labels.get(result.tag, result.tag)
        gt_line = trajectory_line_points(result.gt_centers_full, result.gt_centers_full)
        aligned_line = trajectory_line_points(result.aligned_centers, result.gt_centers)
        ax.plot(
            *gt_line.T,
            color=color,
            linewidth=1.0,
            alpha=0.5,
            marker="o",
            markersize=2.5,
            markeredgewidth=0.0,
            label=f"{label} GT",
        )
        ax.plot(
            *aligned_line.T,
            color=color,
            linewidth=2.2,
            linestyle="--",
            marker="x",
            markersize=5,
            label=f"{label} COLMAP aligned",
        )
    ax.scatter([0], [0], [0], color="#D62728", s=70, depthshade=False, label="target", zorder=5)
    ax.set(
        xlabel="V-BAR X [m]",
        ylabel="-H-BAR Y [m]",
        zlabel="R-BAR Z [m]",
        title="Safety ellipses in CORTO frame",
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_errors(results: Sequence[OrbitDiagnostics], output: Path, kind: str) -> None:
    """Plot position or view-direction residuals per registered frame."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for result in results:
        values = result.position_errors if kind == "position" else result.view_errors_deg
        frame_indices = np.arange(len(values))
        ax.plot(frame_indices, values, marker="o", markersize=3, linewidth=1, label=result.tag)
    if kind == "position":
        ax.set_ylabel("camera-centre error [CORTO units]")
        ax.set_title("Camera-centre alignment residuals")
    else:
        ax.set_ylabel("view-direction error [deg]")
        ax.set_title("View-direction errors after alignment")
    ax.set_xlabel("registered frame index (filtered labels.json order)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    """Load alignment outputs and write trajectory/error figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-sparse-points", type=int, default=5000)
    parser.add_argument(
        "--orbit1-full-geometry",
        type=Path,
        default=DEFAULT_FULL_GEOMETRIES["orbit1"],
        help="Complete geometry.json for orbit1 GT plotting.",
    )
    parser.add_argument(
        "--orbit2-full-geometry",
        type=Path,
        default=DEFAULT_FULL_GEOMETRIES["orbit2"],
        help="Complete geometry.json for orbit2 GT plotting.",
    )
    args = parser.parse_args()

    alignment_root = args.alignment_root.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else alignment_root
    output.mkdir(parents=True, exist_ok=True)
    results = [
        load_orbit(alignment_root, "orbit1", args.orbit1_full_geometry.expanduser().resolve()),
        load_orbit(alignment_root, "orbit2", args.orbit2_full_geometry.expanduser().resolve()),
    ]

    plot_trajectories(results, output / "trajectories_all_fit.png", args.max_sparse_points)
    plot_errors(results, output / "position_errors_all_fit.png", "position")
    plot_errors(results, output / "view_direction_errors_all_fit.png", "view")
    print(f"Plots written to: {output}")


if __name__ == "__main__":
    main()
