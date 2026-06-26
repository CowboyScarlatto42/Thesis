#!/usr/bin/env python3
"""
Filter a CORTO rendered sequence by Sun phase angle.

The script reads a CORTO `geometry.json`, computes the Sun phase angle for each
frame, and exports the frame indices whose phase angle is below a user-defined
threshold. This is used to keep views where the target is sufficiently
illuminated before converting the render sequence to the later preprocessing
format.

For a Blender Sun light, position is not physically meaningful here: the light
direction is defined by the Sun object orientation. The vector toward the light
source is the local +Z axis rotated into the world frame.

The phase angle is computed between:
  - the vector toward the Sun source;
  - the target-to-camera vector, i.e. normalize(camera_pos - body_pos).

Outputs:
  - accepted_frames.npy: 1D array with accepted frame indices;
  - phase_angle_plot.png: diagnostic phase-angle plot.

Example:
    python filter_by_phase_angle.py \\
        --geometry geometry.json \\
        --output_dir /path/to/output \\
        --threshold 90.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Geometry helpers
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Return a unit vector and reject degenerate inputs."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"Near-zero vector: {v}")
    return v / n


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert a scalar-first quaternion to a 3x3 rotation matrix."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2),   2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),   1-2*(x**2+z**2),   2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x), 1-2*(x**2+y**2)],
    ])


def sun_toward_vector(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Return the unit world-frame vector pointing toward the Sun source.

    Blender Sun rays travel along the object's local -Z axis, so the opposite
    local +Z axis is the direction from the scene toward the source.
    """
    R = quat_wxyz_to_rotmat(q_wxyz)
    return normalize(R @ np.array([0.0, 0.0, 1.0]))


# ============================================================
# Phase-angle computation
# ============================================================

def compute_phase_angles(geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Sun phase angle for every camera pose in a CORTO geometry file.

    The CORTO scenarios used here keep the Sun orientation fixed, so the first
    Sun quaternion is enough to define the lighting direction.
    """
    q_sun = np.array(geometry["sun"]["orientation"][0], dtype=np.float64)
    toward_sun = sun_toward_vector(q_sun)

    camera_positions = np.array(geometry["camera"]["position"], dtype=np.float64)
    body_positions   = np.array(geometry["body"]["position"],   dtype=np.float64)

    n_frames = len(camera_positions)
    phase_angles = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        # Small angle means Sun and camera are on the same side of the target.
        body_to_cam = normalize(camera_positions[i] - body_positions[i])
        cos_angle = np.clip(np.dot(toward_sun, body_to_cam), -1.0, 1.0)
        phase_angles[i] = np.degrees(np.arccos(cos_angle))

    return phase_angles, toward_sun


# ============================================================
# Plotting and reporting
# ============================================================

def plot_phase_angles(
    phase_angles: np.ndarray,
    threshold: float,
    accepted: np.ndarray,
    rejected: np.ndarray,
    toward_sun: np.ndarray,
    output_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 5))

    frame_indices = np.arange(len(phase_angles))

    ax.plot(frame_indices, phase_angles, color="steelblue", linewidth=1.4,
            zorder=3, label="Phase angle")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold}°)")

    ax.fill_between(frame_indices, phase_angles, threshold,
                    where=(phase_angles > threshold),
                    alpha=0.25, color="red", label=f"Rejected ({len(rejected)})")
    ax.fill_between(frame_indices, phase_angles, threshold,
                    where=(phase_angles <= threshold),
                    alpha=0.20, color="green", label=f"Accepted ({len(accepted)})")

    # Add compact accept/reject markers near the x-axis for quick inspection.
    ax.scatter(accepted, np.zeros(len(accepted)) + 2,
               color="green", s=12, zorder=4, linewidths=0)
    ax.scatter(rejected, np.zeros(len(rejected)) + 2,
               color="red", s=12, zorder=4, linewidths=0)

    ax.set_xlabel("Frame index", fontsize=11)
    ax.set_ylabel("Phase angle (degrees)", fontsize=11)
    ax.set_xlim(0, len(phase_angles) - 1)
    ax.set_ylim(0, 185)
    ax.set_yticks(range(0, 181, 30))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    elev = np.degrees(np.arcsin(np.clip(toward_sun[2], -1, 1)))
    azim = np.degrees(np.arctan2(toward_sun[1], toward_sun[0]))
    ax.set_title(
        f"Sun Phase Angle  |  threshold={threshold}°  |  "
        f"accepted={len(accepted)}/{len(phase_angles)} ({100*len(accepted)/len(phase_angles):.1f}%)  |  "
        f"sun elev={elev:.1f}° azim={azim:.1f}°",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Plot salvato: {output_path}")


def print_summary(phase_angles, accepted, rejected, threshold, toward_sun):
    """Print a compact diagnostic summary of the filtering result."""
    elev = np.degrees(np.arcsin(np.clip(toward_sun[2], -1, 1)))
    azim = np.degrees(np.arctan2(toward_sun[1], toward_sun[0]))

    print("\n" + "=" * 60)
    print("RIEPILOGO PHASE ANGLE FILTERING")
    print("=" * 60)
    print(f"  Direzione verso sole:     {np.round(toward_sun, 4)}")
    print(f"  Elevazione sole:          {elev:.1f}°")
    print(f"  Azimuth sole:             {azim:.1f}°")
    print(f"  Threshold:                {threshold:.1f}°")
    print(f"\n  Frame totali:    {len(phase_angles)}")
    print(f"  Frame accettati: {len(accepted)} ({100*len(accepted)/len(phase_angles):.1f}%)")
    print(f"  Frame scartati:  {len(rejected)} ({100*len(rejected)/len(phase_angles):.1f}%)")
    print(f"\n  Phase angle (deg) — tutti i frame:")
    print(f"    min:    {phase_angles.min():.2f}°")
    print(f"    max:    {phase_angles.max():.2f}°")
    print(f"    mean:   {phase_angles.mean():.2f}°")
    print(f"    median: {np.median(phase_angles):.2f}°")
    if len(accepted) > 0:
        print(f"\n  Phase angle (deg) — frame accettati:")
        print(f"    min:  {phase_angles[accepted].min():.2f}°")
        print(f"    max:  {phase_angles[accepted].max():.2f}°")
        print(f"    mean: {phase_angles[accepted].mean():.2f}°")
    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Sun phase angle and save accepted frame indices."
    )
    parser.add_argument("--geometry",   required=True, type=Path,
                        help="Path to geometry.json")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Output directory for accepted_frames.npy and plot")
    parser.add_argument("--threshold",  type=float, default=90.0,
                        help="Threshold in degrees. Frames above it are rejected.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.geometry) as f:
        geometry = json.load(f)

    phase_angles, toward_sun = compute_phase_angles(geometry)

    accepted = np.where(phase_angles <= args.threshold)[0]
    rejected = np.where(phase_angles >  args.threshold)[0]

    print_summary(phase_angles, accepted, rejected, args.threshold, toward_sun)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = args.output_dir / "accepted_frames.npy"
    np.save(npy_path, accepted)
    print(f"\n  accepted_frames.npy salvato: {npy_path}")
    print(f"  Indici: {accepted.tolist()}")

    plot_phase_angles(
        phase_angles, args.threshold, accepted, rejected,
        toward_sun, args.output_dir / "phase_angle_plot.png"
    )
