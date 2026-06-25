#!/usr/bin/env python3
"""
Analyze residual COLMAP pose errors for the realistic-scenario datasets.

The errors computed here are residuals after GT-assisted Sim(3) alignment of
COLMAP camera centers to the common metric CORTO world frame. They measure the
geometric consistency of the aligned poses used by NeuS, not the accuracy of a
fully autonomous registration pipeline.

Edit the USER SETTINGS section before running in Colab or as a local .py file.
The script only reads existing inputs and writes CSV/TXT/TEX/PNG outputs.
"""

import os
import re
import csv
import json
import zipfile
import io
import argparse
from pathlib import Path
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

OUT_DIR = str(Path.home() / "Desktop" / "pose_error_analysis_realistic")

# NeuS/IDR projection decomposition returns a camera-to-world matrix whose
# camera-space +Z axis is the forward ray direction used by models/dataset.py
# when rays_v = pose[:3, :3] @ inv(K) @ [u, v, 1].
CAMERA_FORWARD_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
BLENDER_CAMERA_TO_CV_CAMERA = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

DPI = 150
FIG_SIZE = (9, 5)
LINE_WIDTH = 1.8
MARKER = "o"
MARKER_SIZE = 3
GRID_ALPHA = 0.3

SEQUENCE_COLORS = {
    "Orbit 1": "#0072B2",
    "Orbit 2 -- Sun frame 70": "#D55E00",
    "Orbit 2 -- Sun frame 80": "#009E73",
}


@dataclass
class SequenceConfig:
    name: str
    gt_cameras_path: str = None
    aligned_colmap_cameras_path: str = None
    original_frame_mapping_path: str = None
    accepted_frames_path: str = None
    gt_geometry_path: str = None
    combined_dataset_zip_path: str = None
    combined_dataset_orbit_tag: str = None
    poses_already_in_corto_frame: bool = True
    use_scale_mat_for_poses: bool = False
    sim3_transform_path: str = None


# These defaults use the files provided next to the current desktop session:
# combined NeuS datasets are read directly from their zip files, and GT CORTO
# poses are read from the full geometry.json files.
SEQUENCES = [
    SequenceConfig(
        name="Orbit 1",
        gt_geometry_path="/Users/martino/Desktop/geometry_first.json",
        combined_dataset_zip_path="/Users/martino/Desktop/dataset_70.zip",
        combined_dataset_orbit_tag="orbit1",
        accepted_frames_path="/Users/martino/Desktop/accepted_frames_first.npy",
    ),
    SequenceConfig(
        name="Orbit 2 -- Sun frame 70",
        gt_geometry_path="/Users/martino/Desktop/geometry_70.json",
        combined_dataset_zip_path="/Users/martino/Desktop/dataset_70.zip",
        combined_dataset_orbit_tag="orbit2",
        accepted_frames_path="/Users/martino/Desktop/accepted_frames_70.npy",
    ),
    SequenceConfig(
        name="Orbit 2 -- Sun frame 80",
        gt_geometry_path="/Users/martino/Desktop/geometry_80.json",
        combined_dataset_zip_path="/Users/martino/Desktop/dataset_80.zip",
        combined_dataset_orbit_tag="orbit2",
        accepted_frames_path="/Users/martino/Desktop/accepted_frames_80.npy",
    ),
]


COMBINED_DATASETS = [
    ("05 -- Single-orbit realistic", ["Orbit 1"]),
    ("06 -- Two-orbit, Sun frame 70", ["Orbit 1", "Orbit 2 -- Sun frame 70"]),
    ("07 -- Two-orbit, Sun frame 80", ["Orbit 1", "Orbit 2 -- Sun frame 80"]),
]


def parse_args():
    """Parse optional path overrides for the pose-error analysis inputs."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze residual aligned-COLMAP pose errors for the realistic datasets. "
            "All path arguments are optional; omitted values fall back to USER SETTINGS."
        )
    )
    parser.add_argument("--out-dir", default=OUT_DIR, help="Directory where CSV/TEX/TXT/PNG outputs are written.")
    parser.add_argument(
        "--dataset-70-zip",
        help="Combined NeuS dataset zip or extracted directory for Orbit 1 + Orbit 2 sun frame 70.",
    )
    parser.add_argument(
        "--dataset-80-zip",
        help="Combined NeuS dataset zip or extracted directory for Orbit 1 + Orbit 2 sun frame 80.",
    )
    parser.add_argument(
        "--dataset-70-dir",
        help="Alias for --dataset-70-zip when the combined NeuS dataset is an extracted directory.",
    )
    parser.add_argument(
        "--dataset-80-dir",
        help="Alias for --dataset-80-zip when the combined NeuS dataset is an extracted directory.",
    )
    parser.add_argument("--geometry-first", help="Full CORTO geometry.json for Orbit 1.")
    parser.add_argument("--geometry-70", help="Full CORTO geometry.json for Orbit 2 sun frame 70.")
    parser.add_argument("--geometry-80", help="Full CORTO geometry.json for Orbit 2 sun frame 80.")
    parser.add_argument("--accepted-first", help="accepted_frames.npy for Orbit 1.")
    parser.add_argument("--accepted-70", help="accepted_frames.npy for Orbit 2 sun frame 70.")
    parser.add_argument("--accepted-80", help="accepted_frames.npy for Orbit 2 sun frame 80.")
    parser.add_argument(
        "--use-scale-mat-for-poses",
        action="store_true",
        help="Decompose world_mat @ scale_mat instead of world_mat. Default is False for CORTO-frame aligned poses.",
    )
    return parser.parse_args()


def configs_from_args(args):
    """Merge command-line overrides with the default sequence configuration."""
    defaults = {config.name: config for config in SEQUENCES}

    first = defaults["Orbit 1"]
    sun70 = defaults["Orbit 2 -- Sun frame 70"]
    sun80 = defaults["Orbit 2 -- Sun frame 80"]

    return [
        SequenceConfig(
            name="Orbit 1",
            gt_geometry_path=args.geometry_first or first.gt_geometry_path,
            combined_dataset_zip_path=args.dataset_70_dir or args.dataset_70_zip or first.combined_dataset_zip_path,
            combined_dataset_orbit_tag="orbit1",
            accepted_frames_path=args.accepted_first or first.accepted_frames_path,
            use_scale_mat_for_poses=args.use_scale_mat_for_poses,
        ),
        SequenceConfig(
            name="Orbit 2 -- Sun frame 70",
            gt_geometry_path=args.geometry_70 or sun70.gt_geometry_path,
            combined_dataset_zip_path=args.dataset_70_dir or args.dataset_70_zip or sun70.combined_dataset_zip_path,
            combined_dataset_orbit_tag="orbit2",
            accepted_frames_path=args.accepted_70 or sun70.accepted_frames_path,
            use_scale_mat_for_poses=args.use_scale_mat_for_poses,
        ),
        SequenceConfig(
            name="Orbit 2 -- Sun frame 80",
            gt_geometry_path=args.geometry_80 or sun80.gt_geometry_path,
            combined_dataset_zip_path=args.dataset_80_dir or args.dataset_80_zip or sun80.combined_dataset_zip_path,
            combined_dataset_orbit_tag="orbit2",
            accepted_frames_path=args.accepted_80 or sun80.accepted_frames_path,
            use_scale_mat_for_poses=args.use_scale_mat_for_poses,
        ),
    ]


# ============================================================
# NeuS camera decomposition
# ============================================================

def load_K_Rt_from_P(filename, P=None):
    """Local copy of the NeuS/IDR projection-matrix decomposition."""
    if P is None:
        lines = Path(filename).read_text(encoding="utf-8").splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (line.split(" ") for line in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def exact_world_mat_ids(camera_dict):
    """Return sorted integer ids for exact world_mat_<id> camera entries."""
    ids = []
    pattern = re.compile(r"^world_mat_(\d+)$")
    for key in camera_dict.files:
        match = pattern.match(key)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def project_to_so3(R, label="rotation"):
    """Project a near-rotation matrix onto SO(3) using SVD."""
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0.0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt
    det = float(np.linalg.det(R_proj))
    if not np.isfinite(det) or not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(f"{label}: projected SO(3) determinant is invalid: {det}")
    return R_proj


def validate_rotation(R, label):
    """Validate that a matrix is finite, orthogonal, and right-handed."""
    if not np.all(np.isfinite(R)):
        raise ValueError(f"{label}: rotation contains NaN or infinite values")
    err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    if err > 1e-4:
        raise ValueError(f"{label}: rotation is not orthogonal; Frobenius error={err:.6g}")
    det = float(np.linalg.det(R))
    if not np.isclose(det, 1.0, atol=1e-4):
        raise ValueError(f"{label}: rotation determinant is {det:.9f}, expected +1")


def load_cameras_sphere_neus(path, use_scale_mat_for_poses=False):
    """Load NeuS poses from a cameras_sphere.npz file on disk."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing cameras_sphere.npz: {path}")

    camera_dict = np.load(path)
    ids = exact_world_mat_ids(camera_dict)
    if not ids:
        raise ValueError(f"No exact world_mat_<id> keys found in {path}")

    poses = {}
    for camera_id in ids:
        world_mat = np.asarray(camera_dict[f"world_mat_{camera_id}"], dtype=np.float64)
        if world_mat.shape != (4, 4):
            raise ValueError(f"{path}: world_mat_{camera_id} has shape {world_mat.shape}, expected (4, 4)")

        P = world_mat
        if use_scale_mat_for_poses:
            scale_key = f"scale_mat_{camera_id}"
            if scale_key not in camera_dict:
                raise KeyError(f"{path}: missing {scale_key} required by use_scale_mat_for_poses=True")
            P = world_mat @ np.asarray(camera_dict[scale_key], dtype=np.float64)

        if not np.all(np.isfinite(P)):
            raise ValueError(f"{path}: world_mat_{camera_id} contains NaN or infinite values")

        _, pose = load_K_Rt_from_P(None, P[:3, :4])
        pose[:3, :3] = project_to_so3(pose[:3, :3], f"{path}: pose {camera_id}")
        validate_rotation(pose[:3, :3], f"{path}: pose {camera_id}")
        if not np.all(np.isfinite(pose[:3, 3])):
            raise ValueError(f"{path}: pose {camera_id} center contains NaN or infinite values")
        poses[camera_id] = pose

    return poses


def load_cameras_sphere_neus_from_bytes(payload, label, use_scale_mat_for_poses=False, allowed_ids=None):
    """Load NeuS poses from an in-memory cameras_sphere.npz payload."""
    camera_dict = np.load(io.BytesIO(payload))
    ids = exact_world_mat_ids(camera_dict)
    if allowed_ids is not None:
        allowed_ids = set(int(x) for x in allowed_ids)
        ids = [camera_id for camera_id in ids if camera_id in allowed_ids]
    if not ids:
        raise ValueError(f"No exact world_mat_<id> keys found in {label}")

    poses = {}
    for camera_id in ids:
        world_mat = np.asarray(camera_dict[f"world_mat_{camera_id}"], dtype=np.float64)
        if world_mat.shape != (4, 4):
            raise ValueError(f"{label}: world_mat_{camera_id} has shape {world_mat.shape}, expected (4, 4)")

        P = world_mat
        if use_scale_mat_for_poses:
            scale_key = f"scale_mat_{camera_id}"
            if scale_key not in camera_dict:
                raise KeyError(f"{label}: missing {scale_key} required by use_scale_mat_for_poses=True")
            P = world_mat @ np.asarray(camera_dict[scale_key], dtype=np.float64)

        if not np.all(np.isfinite(P)):
            raise ValueError(f"{label}: world_mat_{camera_id} contains NaN or infinite values")

        _, pose = load_K_Rt_from_P(None, P[:3, :4])
        pose[:3, :3] = project_to_so3(pose[:3, :3], f"{label}: pose {camera_id}")
        validate_rotation(pose[:3, :3], f"{label}: pose {camera_id}")
        if not np.all(np.isfinite(pose[:3, 3])):
            raise ValueError(f"{label}: pose {camera_id} center contains NaN or infinite values")
        poses[camera_id] = pose

    return poses


def quat_wxyz_to_rotmat(q):
    """Convert a wxyz quaternion from geometry.json into a rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("Zero-norm quaternion in geometry.json")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_geometry_corto_camera_poses(path_value, accepted_frames=None):
    """Load GT camera poses from a CORTO geometry.json file."""
    path = require_input_file(path_value, "GT CORTO geometry.json")
    with path.open("r", encoding="utf-8") as handle:
        geometry = json.load(handle)

    positions = np.asarray(geometry["camera"]["position"], dtype=np.float64)
    orientations = np.asarray(geometry["camera"]["orientation"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{path}: camera.position must have shape Nx3, got {positions.shape}")
    if orientations.ndim != 2 or orientations.shape[1] != 4:
        raise ValueError(f"{path}: camera.orientation must have shape Nx4, got {orientations.shape}")
    if len(positions) != len(orientations):
        raise ValueError(f"{path}: camera.position and camera.orientation lengths differ")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(orientations)):
        raise ValueError(f"{path}: geometry contains NaN or infinite values")

    if accepted_frames is not None and len(positions) == len(accepted_frames):
        frame_ids = [int(frame_id) for frame_id in accepted_frames]
    else:
        frame_ids = list(range(len(positions)))

    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"{path}: GT frame ids contain duplicates")

    poses = {}
    # geometry.camera.orientation is q_camera_to_world [w, x, y, z] in the
    # Blender camera convention. Convert it to the CV convention used by NeuS:
    # +X right, +Y down, +Z forward.
    for frame_id, center, quat in zip(frame_ids, positions, orientations):
        pose = np.eye(4, dtype=np.float64)
        R_cw_blender = quat_wxyz_to_rotmat(quat)
        pose[:3, :3] = project_to_so3(
            R_cw_blender @ BLENDER_CAMERA_TO_CV_CAMERA,
            f"{path}: GT frame {frame_id}",
        )
        validate_rotation(pose[:3, :3], f"{path}: GT frame {frame_id}")
        pose[:3, 3] = center
        poses[int(frame_id)] = pose
    return poses


# ============================================================
# Mapping and optional fallback transform
# ============================================================

def is_todo_path(value):
    """Treat empty or TODO-prefixed paths as intentionally unconfigured."""
    return value is None or str(value).strip() == "" or str(value).startswith("TODO/")


def require_input_file(path_value, description):
    """Resolve a configured path and require it to be an existing file."""
    if is_todo_path(path_value):
        raise ValueError(f"{description} is not configured. Replace TODO in the USER SETTINGS section.")
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def load_frame_mapping(path_value):
    """Load an explicit local-COLMAP-to-original-frame mapping."""
    path = require_input_file(path_value, "explicit local-COLMAP-to-original-frame mapping")
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 1:
            mapping = {int(i): int(frame_id) for i, frame_id in enumerate(arr.tolist())}
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            mapping = {int(row[0]): int(row[1]) for row in arr}
        else:
            raise ValueError(f"{path}: mapping .npy must be 1D or Nx2+, got shape {arr.shape}")
    elif suffix == ".csv":
        df = pd.read_csv(path)
        mapping = mapping_from_dataframe(df, path)
    else:
        raise ValueError(
            f"{path}: unsupported mapping extension. Use registered_original_frames.npy, "
            "frame_mapping.npy, frame_mapping.csv, or alignment_report.csv."
        )

    validate_mapping(mapping, path)
    return mapping


def mapping_from_dataframe(df, path):
    """Build a frame mapping from a CSV-like dataframe with known column names."""
    local_candidates = [
        "local_colmap_camera_id",
        "local_camera_id",
        "camera_id",
        "colmap_camera_id",
        "source_index",
        "source_index_within_orbit",
        "index",
    ]
    original_candidates = [
        "original_corto_frame_id",
        "original_frame_id",
        "corto_frame_id",
        "frame_id",
        "source_frame_id",
        "original_stem",
        "original_filename",
        "original_image_name",
    ]

    local_col = first_existing_column(df, local_candidates)
    original_col = first_existing_column(df, original_candidates)
    if original_col is None:
        raise ValueError(
            f"{path}: CSV mapping must contain an explicit original CORTO frame id column. "
            "Columns such as stem/image_name are intentionally not accepted because they can be filtered local ids. "
            f"Available columns: {list(df.columns)}"
        )

    mapping = {}
    for row_index, row in df.iterrows():
        if local_col is None:
            local_id = int(row_index)
        else:
            local_id = parse_int_from_value(row[local_col], f"{path}:{local_col}")
        original_id = parse_int_from_value(row[original_col], f"{path}:{original_col}")
        mapping[int(local_id)] = int(original_id)
    return mapping


def first_existing_column(df, candidates):
    """Return the first candidate column present in a dataframe."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def parse_int_from_value(value, label):
    """Extract an integer frame id from numeric values or filename-like strings."""
    if pd.isna(value):
        raise ValueError(f"{label}: missing integer value")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"{label}: cannot extract integer frame id from {value!r}")
    return int(match.group(1))


def validate_mapping(mapping, path):
    """Validate uniqueness and non-negativity of local/original frame ids."""
    if not mapping:
        raise ValueError(f"{path}: frame mapping is empty")
    keys = list(mapping.keys())
    values = list(mapping.values())
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate local COLMAP camera ids in mapping")
    if len(set(values)) != len(values):
        raise ValueError(f"{path}: duplicate original CORTO frame ids in mapping")
    if any(v < 0 for v in values) or any(k < 0 for k in keys):
        raise ValueError(f"{path}: mapping ids must be non-negative")


def load_accepted_frames(path_value):
    """Load the retained original CORTO frame ids from accepted_frames.npy."""
    if is_todo_path(path_value):
        return None
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"accepted_frames_path is configured but missing: {path}")
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f"{path}: accepted_frames.npy must be 1D, got shape {arr.shape}")
    frames = [int(x) for x in arr.tolist()]
    if len(set(frames)) != len(frames):
        raise ValueError(f"{path}: accepted_frames.npy contains duplicate frame ids")
    return frames


def require_input_file_or_dir(path_value, description):
    """Resolve a configured path and require it to be a file or directory."""
    if is_todo_path(path_value):
        raise ValueError(f"{description} is not configured. Replace TODO in the USER SETTINGS section or pass a CLI path.")
    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"{description} is neither a file nor a directory: {path}")
    return path


def load_combined_dataset_manifest(dataset_path_value):
    """Read source_manifest.csv from an extracted or zipped combined dataset."""
    path = require_input_file_or_dir(dataset_path_value, "combined NeuS dataset zip or directory")
    if path.is_dir():
        manifest_path = path / "source_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{path}: missing source_manifest.csv")
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        with zipfile.ZipFile(path) as archive:
            if "source_manifest.csv" not in archive.namelist():
                raise FileNotFoundError(f"{path}: missing source_manifest.csv")
            with archive.open("source_manifest.csv") as handle:
                rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))
    if not rows:
        raise ValueError(f"{path}: source_manifest.csv is empty")
    return rows


def parse_stem_index_1based(stem, label):
    """Extract the 1-based filtered frame index encoded in an image stem."""
    match = re.search(r"(\d+)", str(stem))
    if not match:
        raise ValueError(f"{label}: cannot extract filtered frame index from stem {stem!r}")
    value = int(match.group(1))
    if value <= 0:
        raise ValueError(f"{label}: expected 1-based filtered frame index, got {value}")
    return value


def load_est_poses_and_mapping_from_combined_zip(config, accepted_frames):
    """Load estimated poses and frame mapping from a combined NeuS dataset."""
    dataset_path = require_input_file_or_dir(
        config.combined_dataset_zip_path,
        f"{config.name} combined NeuS dataset zip or directory",
    )
    if is_todo_path(config.combined_dataset_orbit_tag):
        raise ValueError(f"{config.name}: combined_dataset_orbit_tag is not configured")
    if accepted_frames is None:
        raise ValueError(
            f"{config.name}: accepted_frames.npy is required to map filtered stems from the combined zip "
            "back to original CORTO frame ids."
        )

    rows = [
        row for row in load_combined_dataset_manifest(config.combined_dataset_zip_path)
        if row.get("orbit") == config.combined_dataset_orbit_tag
    ]
    if not rows:
        raise ValueError(f"{dataset_path}: no manifest rows found for orbit={config.combined_dataset_orbit_tag!r}")

    required = {"neus_index", "source_index_within_orbit", "source_stem"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{dataset_path}: source_manifest.csv is missing columns {sorted(missing)}")

    rows = sorted(rows, key=lambda row: int(row["source_index_within_orbit"]))
    neus_ids = [int(row["neus_index"]) for row in rows]
    local_ids = [int(row["source_index_within_orbit"]) for row in rows]
    if local_ids != list(range(len(local_ids))):
        raise ValueError(
            f"{config.name}: source_index_within_orbit must be contiguous local ids 0..N-1, got {local_ids[:10]}..."
        )
    if len(set(neus_ids)) != len(neus_ids):
        raise ValueError(f"{config.name}: duplicate neus_index values in source_manifest.csv")

    mapping = {}
    for row in rows:
        local_id = int(row["source_index_within_orbit"])
        filtered_index = parse_stem_index_1based(row["source_stem"], f"{config.name}: source_stem")
        accepted_index = filtered_index - 1
        if accepted_index >= len(accepted_frames):
            raise ValueError(
                f"{config.name}: source_stem={row['source_stem']!r} points to accepted index "
                f"{accepted_index}, but accepted_frames has length {len(accepted_frames)}"
            )
        mapping[local_id] = int(accepted_frames[accepted_index])

    validate_mapping(mapping, f"{dataset_path}:{config.combined_dataset_orbit_tag}")

    if dataset_path.is_dir():
        cameras_path = dataset_path / "cameras_sphere.npz"
        if not cameras_path.is_file():
            raise FileNotFoundError(f"{dataset_path}: missing cameras_sphere.npz")
        all_poses = load_cameras_sphere_neus(
            cameras_path,
            use_scale_mat_for_poses=config.use_scale_mat_for_poses,
        )
        all_poses = {camera_id: pose for camera_id, pose in all_poses.items() if camera_id in set(neus_ids)}
    else:
        with zipfile.ZipFile(dataset_path) as archive:
            if "cameras_sphere.npz" not in archive.namelist():
                raise FileNotFoundError(f"{dataset_path}: missing cameras_sphere.npz")
            cameras_payload = archive.read("cameras_sphere.npz")
        all_poses = load_cameras_sphere_neus_from_bytes(
            cameras_payload,
            f"{dataset_path}:cameras_sphere.npz",
            use_scale_mat_for_poses=config.use_scale_mat_for_poses,
            allowed_ids=neus_ids,
        )

    est_poses = {}
    for row in rows:
        local_id = int(row["source_index_within_orbit"])
        neus_id = int(row["neus_index"])
        if neus_id not in all_poses:
            raise KeyError(f"{config.name}: cameras_sphere.npz is missing world_mat_{neus_id}")
        est_poses[local_id] = all_poses[neus_id]

    return est_poses, mapping


def load_sim3_transform(path_value):
    """Load an optional fallback Sim(3) transform from an NPZ file."""
    path = require_input_file(path_value, "Sim(3) fallback transform")
    data = np.load(path)
    if "scale" not in data or "rotation" not in data or "translation" not in data:
        raise ValueError(f"{path}: Sim(3) npz must contain scale, rotation, and translation arrays")
    scale = float(np.asarray(data["scale"]).reshape(()))
    R = np.asarray(data["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(data["translation"], dtype=np.float64).reshape(3)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{path}: Sim(3) scale must be positive, got {scale}")
    validate_rotation(project_to_so3(R, f"{path}: sim3 rotation"), f"{path}: sim3 rotation")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{path}: Sim(3) translation contains NaN or infinite values")
    return scale, R, t


def apply_sim3_to_pose(pose, scale, R_sim3, t_sim3):
    """Apply a similarity transform to a camera-to-world pose."""
    out = np.array(pose, dtype=np.float64, copy=True)
    out[:3, 3] = scale * (R_sim3 @ pose[:3, 3]) + t_sim3
    out[:3, :3] = project_to_so3(R_sim3 @ pose[:3, :3], "Sim(3)-transformed pose")
    return out


# ============================================================
# Error metrics and summaries
# ============================================================

def rotation_angle_deg(R_gt, R_est):
    """Compute angular distance between two camera rotations in degrees."""
    delta = R_gt.T @ R_est
    value = (float(np.trace(delta)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def vector_angle_deg(a, b):
    """Compute the angle between two non-zero vectors in degrees."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        raise ValueError("Cannot compute angle for a zero-length vector")
    value = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def compute_pose_errors(sequence_name, gt_poses, est_poses, mapping):
    """Compute per-view position, rotation, and view-direction residual errors."""
    rows = []
    est_ids = sorted(est_poses)

    if set(est_ids) != set(mapping):
        missing_mapping = sorted(set(est_ids) - set(mapping))
        missing_pose = sorted(set(mapping) - set(est_ids))
        raise ValueError(
            f"{sequence_name}: exact matching failed between local pose ids and mapping. "
            f"Missing mapping for poses: {missing_mapping[:20]}; mapping without pose: {missing_pose[:20]}"
        )

    for local_id in est_ids:
        original_frame_id = int(mapping[local_id])
        if original_frame_id not in gt_poses:
            raise KeyError(
                f"{sequence_name}: original CORTO frame {original_frame_id} from mapping "
                "is missing in the GT cameras_sphere.npz"
            )

        gt = gt_poses[original_frame_id]
        est = est_poses[local_id]
        C_gt = gt[:3, 3]
        C_est = est[:3, 3]
        R_gt = gt[:3, :3]
        R_est = est[:3, :3]

        f_gt = R_gt @ CAMERA_FORWARD_AXIS
        f_est = R_est @ CAMERA_FORWARD_AXIS

        rows.append(
            {
                "sequence": sequence_name,
                "local_colmap_camera_id": int(local_id),
                "original_corto_frame_id": int(original_frame_id),
                "aligned_position_error_m": float(np.linalg.norm(C_gt - C_est)),
                "aligned_rotation_error_deg": rotation_angle_deg(R_gt, R_est),
                "aligned_view_direction_error_deg": vector_angle_deg(f_gt, f_est),
                "gt_x": float(C_gt[0]),
                "gt_y": float(C_gt[1]),
                "gt_z": float(C_gt[2]),
                "aligned_est_x": float(C_est[0]),
                "aligned_est_y": float(C_est[1]),
                "aligned_est_z": float(C_est[2]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"{sequence_name}: no per-view rows were produced")
    if not np.all(np.isfinite(df.select_dtypes(include=[np.number]).to_numpy())):
        raise ValueError(f"{sequence_name}: per-view errors contain NaN or infinite values")
    if df["original_corto_frame_id"].duplicated().any():
        raise ValueError(f"{sequence_name}: duplicate original CORTO frame ids in per-view output")
    return df


def summarize_errors(values):
    """Summarize one scalar error distribution."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty error distribution")
    if not np.all(np.isfinite(values)):
        raise ValueError("Cannot summarize values containing NaN or infinite values")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def make_summary_row(label, df, retained_frames=None):
    """Build one summary row for a sequence or combined dataset."""
    pos = summarize_errors(df["aligned_position_error_m"].to_numpy())
    rot = summarize_errors(df["aligned_rotation_error_deg"].to_numpy())
    view = summarize_errors(df["aligned_view_direction_error_deg"].to_numpy())

    registered = int(len(df))
    retained = retained_frames if retained_frames is not None else registered
    rejected = int(retained - registered)
    if rejected < 0:
        raise ValueError(f"{label}: registered frames exceed retained frames ({registered} > {retained})")

    return {
        "Sequence": label,
        "Retained frames": int(retained),
        "Registered frames": registered,
        "Rejected retained frames": rejected,
        "Position mean [m]": pos["mean"],
        "Position std [m]": pos["std"],
        "Position median [m]": pos["median"],
        "Position p95 [m]": pos["p95"],
        "Position max [m]": pos["max"],
        "Rotation mean [deg]": rot["mean"],
        "Rotation std [deg]": rot["std"],
        "Rotation median [deg]": rot["median"],
        "Rotation p95 [deg]": rot["p95"],
        "Rotation max [deg]": rot["max"],
        "View-direction mean [deg]": view["mean"],
        "View-direction std [deg]": view["std"],
        "View-direction median [deg]": view["median"],
        "View-direction p95 [deg]": view["p95"],
        "View-direction max [deg]": view["max"],
    }


def compact_summary(df):
    """Select the columns used in compact thesis/report tables."""
    return df[
        [
            "Sequence",
            "Registered frames",
            "Position mean [m]",
            "Position p95 [m]",
            "Position max [m]",
            "Rotation mean [deg]",
            "Rotation p95 [deg]",
            "Rotation max [deg]",
        ]
    ].copy()


def format_for_latex(df):
    """Convert float columns to fixed-decimal strings before LaTeX export."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.6f}")
    return out


def dataframe_to_markdown(df):
    """Render a dataframe as Markdown when tabulate is available."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


# ============================================================
# Plots and outputs
# ============================================================

def safe_slug(text):
    """Convert display text into a filesystem-friendly lowercase slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def plot_with_frame_gaps(ax, x, y, label=None, color=None):
    """Plot frame-indexed values while breaking lines at missing frame gaps."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    breaks = np.where(np.diff(x) > 1.0)[0] + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(x)]

    for segment_index, (start, end) in enumerate(zip(starts, ends)):
        ax.plot(
            x[start:end],
            y[start:end],
            marker=MARKER,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=label if segment_index == 0 else None,
            color=color,
        )


def plot_metric(df, y_col, ylabel, title, out_path, color=None):
    """Save one per-sequence error curve as a PNG figure."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_with_frame_gaps(
        ax,
        df["original_corto_frame_id"],
        df[y_col],
        color=color,
    )
    ax.set_xlabel("Original CORTO frame id")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_metric_overlay(sequence_frames, y_col, ylabel, title, out_path):
    """Save an overlay plot comparing the same metric across sequences."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for name, df in sequence_frames.items():
        color = SEQUENCE_COLORS.get(name)
        plot_with_frame_gaps(
            ax,
            df["original_corto_frame_id"],
            df[y_col],
            label=name,
            color=color,
        )
    ax.set_xlabel("Original CORTO frame id")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(df, title, out_path):
    """Plot GT and aligned COLMAP camera centers in 3D."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(df["gt_x"], df["gt_y"], df["gt_z"], color="#0072B2", linewidth=LINE_WIDTH, label="GT camera centers")
    ax.plot(
        df["aligned_est_x"],
        df["aligned_est_y"],
        df["aligned_est_z"],
        color="#D55E00",
        linewidth=LINE_WIDTH,
        label="aligned COLMAP camera centers",
    )
    ax.scatter(df["gt_x"], df["gt_y"], df["gt_z"], color="#0072B2", s=8)
    ax.scatter(df["aligned_est_x"], df["aligned_est_y"], df["aligned_est_z"], color="#D55E00", s=8)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def write_tables(summary_df, csv_path, tex_path, compact_tex_path):
    """Write full and compact summaries to CSV/LaTeX files."""
    summary_df.to_csv(csv_path, index=False)
    format_for_latex(summary_df).to_latex(tex_path, index=False, escape=True)
    format_for_latex(compact_summary(summary_df)).to_latex(compact_tex_path, index=False, escape=True)


def save_plots(sequence_frames, out_dir):
    """Generate all per-sequence and overlay diagnostic plots."""
    metrics = [
        ("aligned_position_error_m", "Position error [m]", "Position error vs frame", "position_error_vs_frame"),
        ("aligned_rotation_error_deg", "Rotation error [deg]", "Rotation error vs frame", "rotation_error_vs_frame"),
        (
            "aligned_view_direction_error_deg",
            "View-direction error [deg]",
            "View-direction error vs frame",
            "view_direction_error_vs_frame",
        ),
    ]

    for name, df in sequence_frames.items():
        slug = safe_slug(name)
        color = SEQUENCE_COLORS.get(name)
        for y_col, ylabel, title, suffix in metrics:
            plot_metric(df, y_col, ylabel, f"{name}: {title}", out_dir / f"{slug}_{suffix}.png", color=color)
        plot_trajectory(df, f"{name}: GT vs aligned COLMAP trajectory", out_dir / f"{slug}_gt_vs_aligned_colmap_trajectory.png")

    for y_col, ylabel, title, suffix in metrics:
        plot_metric_overlay(sequence_frames, y_col, ylabel, title, out_dir / f"all_sequences_{suffix}.png")


def generated_files(out_dir):
    """Return the expected report/table files plus generated PNG figures."""
    names = [
        "pose_errors_per_view.csv",
        "pose_errors_per_sequence_summary.csv",
        "pose_errors_per_sequence_summary.tex",
        "pose_errors_per_sequence_compact.tex",
        "pose_errors_combined_neus_datasets_summary.csv",
        "pose_errors_combined_neus_datasets_summary.tex",
        "pose_errors_combined_neus_datasets_compact.tex",
        "pose_alignment_report.txt",
    ]
    return [out_dir / name for name in names] + sorted(out_dir.glob("*.png"))


def build_report(per_sequence_summary, combined_summary, sequence_rows, mode_label, out_dir):
    """Build the plain-text analysis report written next to the outputs."""
    lines = []
    lines.append("REALISTIC COLMAP POSE ERROR ANALYSIS")
    lines.append("=" * 72)
    lines.append(
        "Residual pose errors are computed after GT-assisted similarity alignment. "
        "They should not be interpreted as autonomous registration accuracy."
    )
    lines.append("")
    lines.append(f"Mode: {mode_label}")
    lines.append("")
    lines.append("Frame counts:")
    for row in sequence_rows:
        lines.append(
            f"- {row['Sequence']}: retained={row['Retained frames']}, "
            f"registered={row['Registered frames']}, rejected={row['Rejected retained frames']}"
        )
    lines.append("")
    lines.append("Compact per-sequence summary:")
    lines.append(dataframe_to_markdown(compact_summary(per_sequence_summary)))
    lines.append("")
    lines.append("Compact combined-NeuS-dataset summary:")
    lines.append(dataframe_to_markdown(compact_summary(combined_summary)))
    lines.append("")
    lines.append("Generated files:")
    for path in generated_files(out_dir):
        lines.append(f"- {path}")
    lines.append("")
    lines.append(
        "Residual errors are measured after GT-assisted similarity alignment and should not be interpreted "
        "as autonomous registration accuracy."
    )
    return "\n".join(lines)


# ============================================================
# Main workflow
# ============================================================

def process_sequence(config):
    """Load one sequence, compute residual pose errors, and summarize them."""
    accepted_frames = load_accepted_frames(config.accepted_frames_path)

    if not is_todo_path(config.gt_geometry_path):
        gt_poses = load_geometry_corto_camera_poses(config.gt_geometry_path, accepted_frames=accepted_frames)
    else:
        gt_path = require_input_file(config.gt_cameras_path, f"{config.name} GT cameras_sphere.npz")
        gt_poses = load_cameras_sphere_neus(gt_path, use_scale_mat_for_poses=config.use_scale_mat_for_poses)

    if not is_todo_path(config.combined_dataset_zip_path):
        est_poses, mapping = load_est_poses_and_mapping_from_combined_zip(config, accepted_frames)
    else:
        est_path = require_input_file(config.aligned_colmap_cameras_path, f"{config.name} aligned COLMAP cameras_sphere.npz")
        mapping = load_frame_mapping(config.original_frame_mapping_path)
        est_poses = load_cameras_sphere_neus(est_path, use_scale_mat_for_poses=config.use_scale_mat_for_poses)

    if not config.poses_already_in_corto_frame:
        scale, R_sim3, t_sim3 = load_sim3_transform(config.sim3_transform_path)
        est_poses = {camera_id: apply_sim3_to_pose(pose, scale, R_sim3, t_sim3) for camera_id, pose in est_poses.items()}

    if len(mapping) != len(est_poses):
        raise ValueError(
            f"{config.name}: mapping length ({len(mapping)}) must exactly match registered COLMAP poses ({len(est_poses)})."
        )

    retained_count = len(accepted_frames) if accepted_frames is not None else len(mapping)
    if accepted_frames is not None:
        registered_original = set(mapping.values())
        accepted_original = set(accepted_frames)
        if not registered_original.issubset(accepted_original):
            extra = sorted(registered_original - accepted_original)
            raise ValueError(
                f"{config.name}: mapping contains registered frames not present in accepted_frames.npy: {extra[:20]}"
            )

    df = compute_pose_errors(config.name, gt_poses, est_poses, mapping)
    summary = make_summary_row(config.name, df, retained_frames=retained_count)
    return df, summary


def main(sequences=None, out_dir_value=OUT_DIR):
    """Run the full pose-error analysis over all configured sequences."""
    if sequences is None:
        sequences = SEQUENCES

    out_dir = Path(out_dir_value).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    sequence_frames = {}
    sequence_rows = []
    modes = set()

    loaded_orbit1 = 0
    for config in sequences:
        if config.name == "Orbit 1":
            loaded_orbit1 += 1
        mode = "already aligned CORTO-frame mode" if config.poses_already_in_corto_frame else "raw COLMAP + external Sim(3) fallback mode"
        modes.add(mode)
        df, row = process_sequence(config)
        sequence_frames[config.name] = df
        sequence_rows.append(row)

    if loaded_orbit1 != 1:
        raise ValueError(f"Orbit 1 must be configured and loaded exactly once, got {loaded_orbit1}")

    per_view = pd.concat([sequence_frames[name] for name in sequence_frames], ignore_index=True)
    per_view.to_csv(out_dir / "pose_errors_per_view.csv", index=False)

    per_sequence_summary = pd.DataFrame(sequence_rows)
    write_tables(
        per_sequence_summary,
        out_dir / "pose_errors_per_sequence_summary.csv",
        out_dir / "pose_errors_per_sequence_summary.tex",
        out_dir / "pose_errors_per_sequence_compact.tex",
    )

    combined_rows = []
    for dataset_name, sequence_names in COMBINED_DATASETS:
        missing = [name for name in sequence_names if name not in sequence_frames]
        if missing:
            raise ValueError(f"{dataset_name}: missing sequence frames for {missing}")
        combined = pd.concat([sequence_frames[name] for name in sequence_names], ignore_index=True)
        combined_rows.append(make_summary_row(dataset_name, combined, retained_frames=len(combined)))

    combined_summary = pd.DataFrame(combined_rows)
    write_tables(
        combined_summary,
        out_dir / "pose_errors_combined_neus_datasets_summary.csv",
        out_dir / "pose_errors_combined_neus_datasets_summary.tex",
        out_dir / "pose_errors_combined_neus_datasets_compact.tex",
    )

    save_plots(sequence_frames, out_dir)

    mode_label = ", ".join(sorted(modes))
    report = build_report(per_sequence_summary, combined_summary, sequence_rows, mode_label, out_dir)
    (out_dir / "pose_alignment_report.txt").write_text(report, encoding="utf-8")

    print(report)
    print("")
    print("Per-sequence summary:")
    print(dataframe_to_markdown(per_sequence_summary))
    print("")
    print("Combined NeuS dataset summary:")
    print(dataframe_to_markdown(combined_summary))
    print("")
    print(
        "Residual errors are measured after GT-assisted similarity alignment and should not be interpreted "
        "as autonomous registration accuracy."
    )


if __name__ == "__main__":
    cli_args = parse_args()
    main(sequences=configs_from_args(cli_args), out_dir_value=cli_args.out_dir)
