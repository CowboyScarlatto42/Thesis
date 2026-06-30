#!/usr/bin/env python3
"""
Generate thesis figures from CORTO geometry.json files.

Outputs:
  - orbit_2_two_suns.png: Orbit 2 camera trajectory with sun markers 3 m behind
    the frame 70 and 80 cameras.
  - roe_trajectory_comparison.png: Complex Light and Orbit 2 trajectories together.
  - safety_ellipse_1.png: Complex Light trajectory only.

All figures are plotted in LVLH coordinates. The RTN-to-LVLH conversion matrix is
kept here to make the convention explicit:
  x_LVLH = +T_RTN (V-BAR), y_LVLH = -N_RTN (-H-BAR), z_LVLH = -R_RTN (R-BAR).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache_dir = Path(tempfile.gettempdir()) / "fontconfig-cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GEOMETRY_ROOT = REPO_ROOT / "corto" / "input"
DEFAULT_OUTPUT_DIR = Path.home() / "Desktop" / "thesis_plots"
RTN_TO_LVLH = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ]
)

CASES = {
    "complex_light": {
        "case": "S10_Spacecraft_Complex_Light",
        "label": "safety ellipse",
        "color": "#0072B2",
    },
    "orbit_2": {
        "case": "S10_Spacecraft_Orbit_2",
        "label": "secondary trajectory",
        "color": "#D55E00",
    },
    "orbit_2_70": {
        "case": "S10_Spacecraft_Orbit_2_70",
        "label": "illumination frame 70",
        "sun_color": "#F0E442",
    },
    "orbit_2_80": {
        "case": "S10_Spacecraft_Orbit_2",
        "label": "illumination frame 80",
        "sun_color": "#E69F00",
    },
}

FRAME_MARKERS = {
    0: "#0072B2",
    25: "#009E73",
    50: "#CC79A7",
    75: "#8C564B",
}


def load_geometry(case_name: str, geometry_root: Path) -> dict[str, np.ndarray]:
    """Load camera, sun, and target positions from one CORTO geometry.json."""
    path = geometry_root / case_name / "geometry" / "geometry.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing geometry file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return {
        "camera": np.asarray(payload["camera"]["position"], dtype=float),
        "sun": np.asarray(payload["sun"]["position"], dtype=float),
        "body": np.asarray(payload["body"]["position"], dtype=float),
        "path": path,
    }


def unique_position(positions: np.ndarray) -> np.ndarray:
    """Return the constant target/sun position stored as an Nx3 array."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected an Nx3 position array, got shape {positions.shape}")
    return positions[0]


def rtn_to_lvlh(points: np.ndarray) -> np.ndarray:
    """Convert RTN coordinates into the LVLH convention used by the figures."""
    points = np.asarray(points, dtype=float)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected positions with last dimension 3, got shape {points.shape}")
    return points @ RTN_TO_LVLH.T


def sun_position_behind_camera(case_data: dict[str, np.ndarray], frame_idx: int, distance: float) -> np.ndarray:
    """Place a synthetic sun marker behind a selected camera frame."""
    camera = case_data["camera"][frame_idx]
    target = case_data["body"][frame_idx]
    camera_from_target = camera - target
    norm = np.linalg.norm(camera_from_target)
    if norm == 0.0:
        raise ValueError(f"Camera and target overlap at frame {frame_idx}; cannot place sun behind camera.")
    return camera + distance * camera_from_target / norm


def set_equal_3d_axes(ax, points: np.ndarray, margin: float = 0.12) -> float:
    """Set equal 3D plot limits around all points and return the plot radius."""
    points = np.asarray(points, dtype=float)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    radius = max(radius * (1.0 + margin), 1.0)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    return radius


def style_3d_axis(ax, title: str) -> None:
    """Apply common axis labels, title styling, grid, and view angle."""
    ax.set_xlabel("V-BAR X [m]")
    ax.set_ylabel("-H-BAR Y [m]")
    ax.set_zlabel("")
    ax.text2D(1.04, 0.53, "R-BAR Z [m]", transform=ax.transAxes, rotation=90, va="center")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.45)
    ax.view_init(elev=25, azim=-62)


def plot_lvh_reference_axes(ax, length: float) -> None:
    """Draw dashed LVLH reference axes centered at the target."""
    axes = (
        ((-length, length), (0, 0), (0, 0), "#D62728"),
        ((0, 0), (-length, length), (0, 0), "#2CA02C"),
        ((0, 0), (0, 0), (-length, length), "#FF7F0E"),
    )
    for xs, ys, zs, color in axes:
        ax.plot(xs, ys, zs, color=color, linestyle="--", linewidth=1.2, alpha=0.9)


def scatter_target_and_suns(ax, body: np.ndarray, suns: list[tuple[str, np.ndarray, str]]) -> list[np.ndarray]:
    """Plot the target and one or more sun markers, returning their positions."""
    target = unique_position(body)
    ax.scatter(*target, color="#D62728", s=70, depthshade=False, label="target", zorder=5)

    points = [target]
    for label, sun, color in suns:
        ax.scatter(*sun, color=color, edgecolor="#6E5A00", linewidth=0.6, s=75, depthshade=False, label=label)
        points.append(sun)
    return points


def plot_frame_markers(ax, positions: np.ndarray, target: np.ndarray, prefix: str = "camera frame") -> None:
    """Highlight selected camera frames and connect them to the target."""
    for idx, color in FRAME_MARKERS.items():
        if idx >= len(positions):
            continue
        point = positions[idx]
        ax.scatter(*point, color=color, s=55, depthshade=False, label=f"{prefix} {idx}", zorder=6)
        ax.plot(
            [point[0], target[0]],
            [point[1], target[1]],
            [point[2], target[2]],
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.45,
        )


def plot_sun_camera_offsets(
    ax,
    camera_positions: np.ndarray,
    sun_positions: list[tuple[int, np.ndarray, str]],
) -> None:
    """Draw dotted offsets from selected camera frames to synthetic sun markers."""
    for frame_idx, sun, color in sun_positions:
        camera = camera_positions[frame_idx]
        ax.plot(
            [camera[0], sun[0]],
            [camera[1], sun[1]],
            [camera[2], sun[2]],
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )


def save_figure(fig, out_path: Path, dpi: int) -> None:
    """Save a Matplotlib figure and close it to release resources."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_orbit_2_with_two_suns(data: dict[str, dict[str, np.ndarray]], out_dir: Path, dpi: int) -> None:
    """Generate the Orbit 2 figure with both illumination-frame sun markers."""
    orbit_data = data["orbit_2"]
    orbit = orbit_data["camera"]
    target = unique_position(orbit_data["body"])
    sun_70 = sun_position_behind_camera(data["orbit_2_70"], frame_idx=70, distance=3.0)
    sun_80 = sun_position_behind_camera(data["orbit_2_80"], frame_idx=80, distance=3.0)

    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        orbit[:, 0],
        orbit[:, 1],
        orbit[:, 2],
        color=CASES["orbit_2"]["color"],
        linewidth=2.2,
        label="secondary trajectory",
    )
    plot_frame_markers(ax, orbit, target)
    plot_sun_camera_offsets(
        ax,
        orbit,
        [
            (70, sun_70, CASES["orbit_2_70"]["sun_color"]),
            (80, sun_80, CASES["orbit_2_80"]["sun_color"]),
        ],
    )
    extra_points = scatter_target_and_suns(
        ax,
        orbit_data["body"],
        [
            ("sun frame 70", sun_70, CASES["orbit_2_70"]["sun_color"]),
            ("sun frame 80", sun_80, CASES["orbit_2_80"]["sun_color"]),
        ],
    )

    limits_points = np.vstack([orbit, np.asarray(extra_points)])
    set_equal_3d_axes(ax, limits_points)
    style_3d_axis(ax, "Secondary trajectory with both illumination conditions")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    save_figure(fig, out_dir / "orbit_2_two_suns.png", dpi)


def plot_trajectory_comparison(data: dict[str, dict[str, np.ndarray]], out_dir: Path, dpi: int) -> None:
    """Generate the figure comparing the two camera safety ellipses."""
    complex_data = data["complex_light"]
    orbit_data = data["orbit_2"]
    target = unique_position(orbit_data["body"])

    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    for key in ("complex_light", "orbit_2"):
        case_cfg = CASES[key]
        camera = data[key]["camera"]
        ax.plot(
            camera[:, 0],
            camera[:, 1],
            camera[:, 2],
            color=case_cfg["color"],
            linewidth=2.2,
            label=case_cfg["label"],
        )

    ax.scatter(*target, color="#D62728", s=70, depthshade=False, label="target", zorder=5)

    limits_points = np.vstack([complex_data["camera"], orbit_data["camera"], target])
    radius = set_equal_3d_axes(ax, limits_points)
    plot_lvh_reference_axes(ax, radius * 0.45)
    style_3d_axis(ax, "Safety ellipse and secondary trajectory")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    save_figure(fig, out_dir / "roe_trajectory_comparison.png", dpi)


def plot_safety_ellipse_1(data: dict[str, dict[str, np.ndarray]], out_dir: Path, dpi: int) -> None:
    """Generate the single-orbit Complex Light trajectory figure."""
    case_data = data["complex_light"]
    camera = case_data["camera"]
    target = unique_position(case_data["body"])

    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        camera[:, 0],
        camera[:, 1],
        camera[:, 2],
        color=CASES["complex_light"]["color"],
        linewidth=2.2,
        label="safety ellipse",
    )
    ax.scatter(*target, color="#D62728", s=70, depthshade=False, label="target", zorder=5)

    limits_points = np.vstack([camera, target])
    radius = set_equal_3d_axes(ax, limits_points)
    plot_lvh_reference_axes(ax, radius * 0.45)
    style_3d_axis(ax, "Safety ellipse")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    save_figure(fig, out_dir / "safety_ellipse_1.png", dpi)


def parse_args() -> argparse.Namespace:
    """Parse geometry root, output directory, and figure DPI options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=DEFAULT_GEOMETRY_ROOT,
        help=f"Directory containing case folders. Default: {DEFAULT_GEOMETRY_ROOT}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for PNG figures. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI.")
    return parser.parse_args()


def main() -> None:
    """Load all configured CORTO cases and generate the thesis figures."""
    args = parse_args()
    data = {
        key: load_geometry(config["case"], args.geometry_root)
        for key, config in CASES.items()
    }

    plot_orbit_2_with_two_suns(data, args.out_dir, args.dpi)
    plot_trajectory_comparison(data, args.out_dir, args.dpi)
    plot_safety_ellipse_1(data, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
