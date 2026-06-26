import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation as R

"""
Visualize and check Sun/camera geometry from a CORTO `geometry.json` file.

The script generates a GIF and a static plot showing camera positions, target
positions, Sun positions, camera viewing directions, and camera-to-Sun segments.
It also reports whether the Sun is behind the camera according to the same
view-direction convention used by the CORTO spacecraft geometry scripts.

Blender/CORTO cameras look along their local -Z axis. Quaternions in
`geometry.json` are expected in scalar-first [w, x, y, z] order.
"""


def load_geometry(geometry_path):
    """Load camera, Sun, and target positions from a CORTO geometry file."""
    with open(geometry_path, "r") as f:
        geom = json.load(f)

    cam_pos = np.asarray(geom["camera"]["position"], dtype=float)
    cam_ori_wxyz = np.asarray(geom["camera"]["orientation"], dtype=float)
    sun_pos = np.asarray(geom["sun"]["position"], dtype=float)

    if "body" in geom and "position" in geom["body"]:
        body_pos = np.asarray(geom["body"]["position"], dtype=float)
    else:
        body_pos = np.zeros((len(cam_pos), 3), dtype=float)

    n = min(len(cam_pos), len(cam_ori_wxyz), len(sun_pos), len(body_pos))
    cam_pos = cam_pos[:n]
    cam_ori_wxyz = cam_ori_wxyz[:n]
    sun_pos = sun_pos[:n]
    body_pos = body_pos[:n]

    if n < 2:
        raise ValueError("At least two poses are required.")

    return cam_pos, cam_ori_wxyz, sun_pos, body_pos


def camera_rot_mats_from_wxyz(cam_ori_wxyz):
    """Convert scalar-first CORTO camera quaternions to rotation matrices."""
    rot_mats = []
    for q_wxyz in cam_ori_wxyz:
        w, x, y, z = q_wxyz
        r = R.from_quat([x, y, z, w])  # scipy uses [x, y, z, w].
        rot_mats.append(r.as_matrix())
    return np.asarray(rot_mats)


def compute_metrics(cam_pos, sun_pos, body_pos, cam_rot_mats):
    """Compute view/Sun/body angles and Sun-behind-camera diagnostics."""
    # Same convention as sun_behind_camera(): view_dir = -Rcw[:, 2].
    view_dirs = np.asarray([-Rcw[:, 2] for Rcw in cam_rot_mats], dtype=float)

    v_cam_to_sun = sun_pos - cam_pos
    n_sun = np.linalg.norm(v_cam_to_sun, axis=1, keepdims=True)
    v_cam_to_sun_u = v_cam_to_sun / np.maximum(n_sun, 1e-12)

    v_cam_to_body = body_pos - cam_pos
    n_body = np.linalg.norm(v_cam_to_body, axis=1, keepdims=True)
    v_cam_to_body_u = v_cam_to_body / np.maximum(n_body, 1e-12)

    dot_view_sun = np.sum(view_dirs * v_cam_to_sun_u, axis=1)
    dot_view_sun = np.clip(dot_view_sun, -1.0, 1.0)
    angle_view_sun_deg = np.degrees(np.arccos(dot_view_sun))

    # True when the Sun lies behind the camera with respect to the view vector.
    is_behind = dot_view_sun < 0.0

    # Check consistency between quaternion-derived view direction and target.
    dot_view_body = np.sum(view_dirs * v_cam_to_body_u, axis=1)
    dot_view_body = np.clip(dot_view_body, -1.0, 1.0)
    angle_view_body_deg = np.degrees(np.arccos(dot_view_body))

    # Signed Sun distance along the behind-camera axis (-view_dir).
    signed_behind_dist = np.sum(v_cam_to_sun * (-view_dirs), axis=1)

    return {
        "view_dirs": view_dirs,
        "angle_view_sun_deg": angle_view_sun_deg,
        "is_behind": is_behind,
        "angle_view_body_deg": angle_view_body_deg,
        "signed_behind_dist": signed_behind_dist,
    }


def set_axes_equal_3d(ax, pts):
    """Set equal visual scale for a 3D matplotlib axis."""
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_gif(out_gif, cam_pos, sun_pos, body_pos, metrics, fps=15, step=1):
    """Save an animated diagnostic view of the CORTO camera/Sun geometry."""
    idx = np.arange(0, len(cam_pos), step)

    fig = plt.figure(figsize=(8, 7))
    ax3d = fig.add_subplot(1, 1, 1, projection="3d")

    pts = np.vstack([cam_pos, sun_pos, body_pos])
    set_axes_equal_3d(ax3d, pts)

    cam_sc = ax3d.scatter([], [], [], s=45, color="tab:blue", label="camera")
    sun_sc = ax3d.scatter([], [], [], s=45, color="gold", label="sole")
    body_sc = ax3d.scatter([], [], [], s=60, color="tab:red", label="body")

    # Full camera trajectory plus the part already reached in the animation.
    traj_pts = cam_pos[idx]
    ax3d.plot(
        traj_pts[:, 0],
        traj_pts[:, 1],
        traj_pts[:, 2],
        color="0.6",
        linestyle="--",
        linewidth=1.2,
        label="_nolegend_",
    )
    line_traj_done, = ax3d.plot([], [], [], color="tab:cyan", linewidth=2.2, label="_nolegend_")

    line_view, = ax3d.plot([], [], [], color="tab:green", linewidth=2, label="_nolegend_")
    line_cam_sun, = ax3d.plot([], [], [], color="tab:orange", linewidth=2, label="_nolegend_")
    line_pointing, = ax3d.plot([], [], [], color="tab:purple", linestyle="--", linewidth=2.2, label="_nolegend_")

    title = ax3d.set_title("")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend(loc="upper left")

    view_dirs = metrics["view_dirs"]

    def update(k):
        i = idx[k]
        c = cam_pos[i]
        s = sun_pos[i]
        b = body_pos[i]
        vd = view_dirs[i]

        cam_sc._offsets3d = ([c[0]], [c[1]], [c[2]])
        sun_sc._offsets3d = ([s[0]], [s[1]], [s[2]])
        body_sc._offsets3d = ([b[0]], [b[1]], [b[2]])

        # Progressive camera trajectory up to the current frame.
        traj_done = cam_pos[idx[: k + 1]]
        line_traj_done.set_data(traj_done[:, 0], traj_done[:, 1])
        line_traj_done.set_3d_properties(traj_done[:, 2])

        # Camera viewing direction.
        p2 = c + vd * np.linalg.norm(s - c) * 0.5
        line_view.set_data([c[0], p2[0]], [c[1], p2[1]])
        line_view.set_3d_properties([c[2], p2[2]])

        # Camera-to-Sun segment.
        line_cam_sun.set_data([c[0], s[0]], [c[1], s[1]])
        line_cam_sun.set_3d_properties([c[2], s[2]])

        # Dashed camera-to-target pointing segment.
        line_pointing.set_data([c[0], b[0]], [c[1], b[1]])
        line_pointing.set_3d_properties([c[2], b[2]])

        title.set_text(f"Frame {i}")

        return cam_sc, sun_sc, body_sc, line_traj_done, line_view, line_cam_sun, line_pointing, title

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000.0 / fps, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_static_frame_plot(out_png, cam_pos, sun_pos, body_pos, frame_ids):
    """Save a static plot for selected frame IDs."""
    frame_ids = [idx for idx in frame_ids if 0 <= idx < len(cam_pos)]
    if not frame_ids:
        raise ValueError("No valid frame for the static plot.")

    camera_colors = ["tab:blue", "tab:green", "tab:purple", "tab:brown"]

    fig = plt.figure(figsize=(8, 7))
    ax3d = fig.add_subplot(1, 1, 1, projection="3d")

    pts = np.vstack([
        cam_pos[frame_ids],
        sun_pos[frame_ids],
        body_pos[frame_ids],
    ])
    set_axes_equal_3d(ax3d, pts)

    ax3d.plot(
        cam_pos[:, 0],
        cam_pos[:, 1],
        cam_pos[:, 2],
        color="tab:cyan",
        linewidth=2.2,
        label="camera orbit",
    )

    for color_idx, frame_idx in enumerate(frame_ids):
        c = cam_pos[frame_idx]
        s = sun_pos[frame_idx]
        b = body_pos[frame_idx]
        camera_color = camera_colors[color_idx % len(camera_colors)]

        ax3d.scatter(
            c[0], c[1], c[2],
            s=70,
            color=camera_color,
            label=f"camera frame {frame_idx}",
        )
        ax3d.scatter(
            s[0], s[1], s[2],
            s=55,
            color="gold",
            edgecolors="0.25",
            linewidths=0.5,
            label="sun" if color_idx == 0 else "_nolegend_",
        )
        ax3d.scatter(
            b[0], b[1], b[2],
            s=70,
            color="tab:red",
            label="target" if color_idx == 0 else "_nolegend_",
        )

        ax3d.plot(
            [c[0], b[0]],
            [c[1], b[1]],
            [c[2], b[2]],
            color=camera_color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
        )
        ax3d.plot(
            [c[0], s[0]],
            [c[1], s[1]],
            [c[2], s[2]],
            color="tab:orange",
            linewidth=1.0,
            alpha=0.45,
        )

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Check Sun/camera geometry from geometry.json")
    parser.add_argument(
        "--geometry",
        default="input/S10_Spacecraft_Simple_Light/geometry/geometry.json",
        help="Path geometry.json",
    )
    parser.add_argument(
        "--outdir",
        default="output/S10_Spacecraft_Simple_Light/sun_check",
        help="Output directory",
    )
    parser.add_argument("--fps", type=int, default=12, help="FPS gif")
    parser.add_argument("--step", type=int, default=1, help="Frame stride in the GIF")
    parser.add_argument(
        "--plot_frames",
        type=int,
        nargs="+",
        default=[0, 25, 50, 75],
        help="Frames to show in the static plot",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cam_pos, cam_ori_wxyz, sun_pos, body_pos = load_geometry(args.geometry)
    cam_rot_mats = camera_rot_mats_from_wxyz(cam_ori_wxyz)
    metrics = compute_metrics(cam_pos, sun_pos, body_pos, cam_rot_mats)

    out_gif = outdir / "sun_behind_check.gif"
    plot_frames_name = "_".join(str(frame_id) for frame_id in args.plot_frames)
    out_static_plot = outdir / f"static_positions_{plot_frames_name}.png"

    save_gif(out_gif, cam_pos, sun_pos, body_pos, metrics, fps=args.fps, step=args.step)
    save_static_frame_plot(
        out_static_plot,
        cam_pos,
        sun_pos,
        body_pos,
        frame_ids=args.plot_frames,
    )

    behind = metrics["is_behind"]
    print(f"Frame totali: {len(behind)}")
    print(f"Frame con sole dietro: {behind.sum()} ({100.0 * behind.mean():.1f}%)")
    print(f"Sempre dietro: {'SI' if np.all(behind) else 'NO'}")
    print(f"GIF:  {out_gif}")
    print(f"Plot: {out_static_plot}")

    # Extra check: how well the quaternion view direction aligns with the target.
    ang_body = metrics["angle_view_body_deg"]
    print(
        "Allineamento view_dir vs camera->body: "
        f"mean={np.mean(ang_body):.3f} deg, max={np.max(ang_body):.3f} deg"
    )


if __name__ == "__main__":
    main()
