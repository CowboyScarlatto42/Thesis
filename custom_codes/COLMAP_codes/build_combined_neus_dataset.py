#!/usr/bin/env python3
"""
Build a combined NeuS dataset from two independently reconstructed COLMAP orbits
after they have been aligned to the common CORTO/Tango frame.

This script does NOT use the original SPE3R ground-truth labels to build poses.
It reads the COLMAP-aligned poses produced by align_colmap_orbits_to_corto.py:

    <alignment_root>/orbit1/aligned_poses_all_fit.json
    <alignment_root>/orbit2/aligned_poses_all_fit.json

For every retained image it:
  1. reads the aligned COLMAP world-to-camera pose;
  2. constructs world_mat = intrinsic @ w2c;
  3. assigns the same GT-derived scale_mat to every view;
  4. copies images and masks to NeuS-style numeric filenames;
  5. saves cameras_sphere.npz and source manifests.

Expected output:
    OUTPUT/
    ├── image/
    │   ├── 000.png
    │   └── ...
    ├── mask/
    │   ├── 000.png
    │   └── ...
    ├── cameras_sphere.npz
    ├── source_manifest.csv
    ├── source_manifest.json
    └── dataset_summary.json

Example:
python build_combined_neus_dataset.py \
  --orbit1-aligned-poses "/path/to/Merged_Orbit/orbit1/aligned_poses_all_fit.json" \
  --orbit1-images "/path/to/First_Orbit/colmap_output_pruned/images" \
  --orbit1-masks "/path/to/First_Orbit/colmap_output_pruned/masks" \
  --orbit2-aligned-poses "/path/to/Merged_Orbit/orbit2/aligned_poses_all_fit.json" \
  --orbit2-images "/path/to/Second_Orbit/colmap_output/images" \
  --orbit2-masks "/path/to/Second_Orbit/colmap_output/masks" \
  --camera-json "/path/to/camera.json" \
  --scale-mat-json "/path/to/scale_mat.json" \
  --output "/path/to/Combined_Orbits_NeuS" \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class OrbitInput:
    tag: str
    aligned_poses_path: Path
    images_dir: Path
    masks_dir: Path


@dataclass
class FrameRecord:
    orbit: str
    source_index: int
    stem: str
    source_filename: str
    image_id: int
    camera_id: int
    q_wc_wxyz_aligned: np.ndarray
    t_wc_aligned: np.ndarray
    camera_center_aligned: np.ndarray
    position_error: Optional[float]
    view_direction_error_deg: Optional[float]
    source_image_path: Path
    source_mask_path: Optional[Path]


def natural_key(text: str) -> List[object]:
    """Natural sorting: img2 comes before img10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def quat_wxyz_to_rotmat(q: Sequence[float]) -> np.ndarray:
    """Quaternion [w, x, y, z] -> proper 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < EPS:
        raise ValueError("Quaternion with near-zero norm")
    w, x, y, z = q / norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )

    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("Quaternion produced a non-orthonormal rotation matrix")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise ValueError("Quaternion produced an improper rotation matrix")
    return R


def load_intrinsic(camera_json_path: Path) -> np.ndarray:
    """Load camera intrinsics as a 4x4 homogeneous matrix."""
    with camera_json_path.open("r", encoding="utf-8") as handle:
        cam = json.load(handle)

    if "cameraMatrix" in cam:
        K = np.asarray(cam["cameraMatrix"], dtype=np.float64)
    elif "K" in cam:
        raw = cam["K"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        K = np.asarray(raw, dtype=np.float64)
    else:
        fx_raw = cam.get("fx", cam.get("focal"))
        if fx_raw is None:
            raise ValueError("camera.json does not contain cameraMatrix, K, fx, or focal")
        fx = float(fx_raw)
        fy = float(cam.get("fy", fx))
        cx_raw = cam.get("cx", cam.get("ccx", cam.get("ppx")))
        cy_raw = cam.get("cy", cam.get("ccy", cam.get("ppy")))
        if cx_raw is None or cy_raw is None:
            raise ValueError("camera.json does not contain cx/cy principal point values")
        K = np.array(
            [
                [fx, 0.0, float(cx_raw)],
                [0.0, fy, float(cy_raw)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    if K.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsic matrix, found shape {K.shape}")
    if not np.all(np.isfinite(K)):
        raise ValueError("Intrinsic matrix contains non-finite values")
    if abs(float(np.linalg.det(K))) < EPS:
        raise ValueError("Intrinsic matrix is singular")

    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[:3, :3] = K
    return intrinsic


def load_scale_mat(scale_mat_json_path: Path) -> np.ndarray:
    """Load the shared NeuS scale_mat used by all combined views."""
    with scale_mat_json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw = data["scale_mat"] if isinstance(data, dict) and "scale_mat" in data else data
    scale_mat = np.asarray(raw, dtype=np.float64)

    if scale_mat.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 scale_mat, found shape {scale_mat.shape}")
    if not np.all(np.isfinite(scale_mat)):
        raise ValueError("scale_mat contains non-finite values")
    if abs(float(np.linalg.det(scale_mat))) < EPS:
        raise ValueError("scale_mat is singular")
    return scale_mat


def read_aligned_pose_records(path: Path) -> List[dict]:
    """Read and validate aligned pose records produced by the alignment script."""
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list) or not records:
        raise ValueError(f"Aligned pose JSON is empty or is not a list: {path}")

    required = {"filename", "stem", "image_id", "camera_id", "q_wc_wxyz_aligned", "t_wc_aligned"}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} in {path} is not a JSON object")
        missing = required - set(record)
        if missing:
            raise ValueError(f"Record {index} in {path} is missing fields: {sorted(missing)}")

    return sorted(records, key=lambda item: natural_key(str(item["stem"])))


def find_source_image(images_dir: Path, filename: str, stem: str) -> Path:
    """Find the source image by exact filename or unique stem match."""
    exact = images_dir / filename
    if exact.is_file():
        return exact

    candidates = sorted(
        [path for path in images_dir.glob(f"{stem}.*") if path.is_file()],
        key=lambda path: natural_key(path.name),
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Missing source image for {filename} below {images_dir}")
    raise RuntimeError(f"Ambiguous image stem {stem} below {images_dir}: {candidates}")


def find_source_mask(masks_dir: Path, filename: str, stem: str) -> Optional[Path]:
    """Find the optional source mask by exact filename or unique stem match."""
    preferred = [
        masks_dir / f"{stem}.png",
        masks_dir / filename,
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate

    candidates = sorted(
        [path for path in masks_dir.glob(f"{stem}.*") if path.is_file()],
        key=lambda path: natural_key(path.name),
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise RuntimeError(f"Ambiguous mask stem {stem} below {masks_dir}: {candidates}")


def build_frames(orbit_input: OrbitInput) -> List[FrameRecord]:
    """Create frame records for one orbit, preserving aligned-pose ordering."""
    raw_records = read_aligned_pose_records(orbit_input.aligned_poses_path)
    frames: List[FrameRecord] = []

    for source_index, record in enumerate(raw_records):
        filename = str(record["filename"])
        stem = str(record["stem"])
        source_image = find_source_image(orbit_input.images_dir, filename, stem)
        source_mask = find_source_mask(orbit_input.masks_dir, filename, stem)

        q = np.asarray(record["q_wc_wxyz_aligned"], dtype=np.float64).reshape(4)
        t = np.asarray(record["t_wc_aligned"], dtype=np.float64).reshape(3)
        R_wc = quat_wxyz_to_rotmat(q)
        center_from_pose = -R_wc.T @ t

        if "camera_center_aligned" in record:
            center_saved = np.asarray(record["camera_center_aligned"], dtype=np.float64).reshape(3)
            delta = float(np.linalg.norm(center_from_pose - center_saved))
            if delta > 1e-5:
                raise ValueError(
                    f"{orbit_input.tag}/{stem}: aligned center mismatch {delta:.6g}. "
                    "The aligned pose JSON is internally inconsistent."
                )
            center = center_saved
        else:
            center = center_from_pose

        frames.append(
            FrameRecord(
                orbit=orbit_input.tag,
                source_index=source_index,
                stem=stem,
                source_filename=filename,
                image_id=int(record["image_id"]),
                camera_id=int(record["camera_id"]),
                q_wc_wxyz_aligned=q,
                t_wc_aligned=t,
                camera_center_aligned=center,
                position_error=float(record["position_error"]) if "position_error" in record else None,
                view_direction_error_deg=(
                    float(record["view_direction_error_deg"])
                    if "view_direction_error_deg" in record
                    else None
                ),
                source_image_path=source_image,
                source_mask_path=source_mask,
            )
        )

    return frames


def prepare_output(output_dir: Path, overwrite: bool, check_only: bool) -> None:
    """Create or reset the output folder unless the script runs in check-only mode."""
    if output_dir.exists() and not check_only:
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Pass --overwrite to replace it explicitly."
            )
        shutil.rmtree(output_dir)

    if not check_only:
        (output_dir / "image").mkdir(parents=True, exist_ok=True)
        (output_dir / "mask").mkdir(parents=True, exist_ok=True)


def write_png(source_path: Path, destination_path: Path) -> Tuple[int, int, int]:
    """Copy an image through OpenCV and return its shape."""
    data = cv.imread(str(source_path), cv.IMREAD_UNCHANGED)
    if data is None:
        raise RuntimeError(f"OpenCV cannot read: {source_path}")
    if not cv.imwrite(str(destination_path), data):
        raise RuntimeError(f"OpenCV cannot write: {destination_path}")

    if data.ndim == 2:
        h, w = data.shape
        channels = 1
    else:
        h, w, channels = data.shape
    return int(h), int(w), int(channels)


def write_white_mask(destination_path: Path, height: int, width: int) -> None:
    """Generate a full-foreground mask when this is explicitly requested."""
    mask = np.full((height, width), 255, dtype=np.uint8)
    if not cv.imwrite(str(destination_path), mask):
        raise RuntimeError(f"OpenCV cannot write generated white mask: {destination_path}")


def summarize_optional(values: Iterable[Optional[float]]) -> Optional[dict]:
    """Summarize optional numeric diagnostics, ignoring missing values."""
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "std": float(array.std()),
    }


def camera_distance_stats(frames: Sequence[FrameRecord], scale_mat: np.ndarray) -> dict:
    """Report camera distances after mapping CORTO centers to NeuS space."""
    scale_inv = np.linalg.inv(scale_mat)
    normalized_centers: List[np.ndarray] = []

    for frame in frames:
        center_h = np.concatenate([frame.camera_center_aligned, [1.0]])
        normalized = scale_inv @ center_h
        normalized_centers.append(normalized[:3] / normalized[3])

    distances = np.linalg.norm(np.vstack(normalized_centers), axis=1)
    return {
        "min": float(distances.min()),
        "max": float(distances.max()),
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
    }


def make_w2c(frame: FrameRecord) -> np.ndarray:
    """Build a world-to-camera matrix from an aligned COLMAP pose record."""
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = quat_wxyz_to_rotmat(frame.q_wc_wxyz_aligned)
    w2c[:3, 3] = frame.t_wc_aligned
    return w2c


def make_manifest_row(index: int, frame: FrameRecord, output_filename: str) -> dict:
    """Create one traceability row linking NeuS index back to source orbit/image."""
    center = frame.camera_center_aligned
    return {
        "neus_index": index,
        "output_filename": output_filename,
        "orbit": frame.orbit,
        "source_index_within_orbit": frame.source_index,
        "source_filename": frame.source_filename,
        "source_stem": frame.stem,
        "source_image_path": str(frame.source_image_path),
        "source_mask_path": str(frame.source_mask_path) if frame.source_mask_path is not None else None,
        "colmap_image_id": frame.image_id,
        "colmap_camera_id": frame.camera_id,
        "aligned_camera_center_x": float(center[0]),
        "aligned_camera_center_y": float(center[1]),
        "aligned_camera_center_z": float(center[2]),
        "alignment_position_error": frame.position_error,
        "alignment_view_direction_error_deg": frame.view_direction_error_deg,
    }


def write_manifest_csv(path: Path, rows: Sequence[dict]) -> None:
    """Write the source manifest as CSV for inspection in spreadsheets."""
    if not rows:
        raise ValueError("Cannot write an empty manifest")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build the combined two-orbit NeuS dataset."""
    parser = argparse.ArgumentParser(
        description="Build a combined NeuS dataset from aligned COLMAP poses for two orbits."
    )
    parser.add_argument("--orbit1-aligned-poses", required=True, type=Path)
    parser.add_argument("--orbit1-images", required=True, type=Path)
    parser.add_argument("--orbit1-masks", required=True, type=Path)
    parser.add_argument("--orbit2-aligned-poses", required=True, type=Path)
    parser.add_argument("--orbit2-images", required=True, type=Path)
    parser.add_argument("--orbit2-masks", required=True, type=Path)
    parser.add_argument("--camera-json", required=True, type=Path)
    parser.add_argument("--scale-mat-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--white-mask-if-missing",
        action="store_true",
        help="Generate a white mask if a source mask is missing. Default: fail.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and print the summary without writing output files.",
    )
    args = parser.parse_args()

    orbit_inputs = [
        OrbitInput(
            tag="orbit1",
            aligned_poses_path=args.orbit1_aligned_poses.resolve(),
            images_dir=args.orbit1_images.resolve(),
            masks_dir=args.orbit1_masks.resolve(),
        ),
        OrbitInput(
            tag="orbit2",
            aligned_poses_path=args.orbit2_aligned_poses.resolve(),
            images_dir=args.orbit2_images.resolve(),
            masks_dir=args.orbit2_masks.resolve(),
        ),
    ]

    for item in orbit_inputs:
        if not item.aligned_poses_path.is_file():
            raise FileNotFoundError(f"Aligned pose JSON not found: {item.aligned_poses_path}")
        if not item.images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {item.images_dir}")
        if not item.masks_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {item.masks_dir}")

    camera_json_path = args.camera_json.resolve()
    scale_mat_json_path = args.scale_mat_json.resolve()
    output_dir = args.output.resolve()

    if not camera_json_path.is_file():
        raise FileNotFoundError(f"camera.json not found: {camera_json_path}")
    if not scale_mat_json_path.is_file():
        raise FileNotFoundError(f"scale_mat.json not found: {scale_mat_json_path}")

    intrinsic = load_intrinsic(camera_json_path)
    scale_mat = load_scale_mat(scale_mat_json_path)
    scale_mat_inv = np.linalg.inv(scale_mat)

    frames_by_orbit: Dict[str, List[FrameRecord]] = {}
    for orbit_input in orbit_inputs:
        frames_by_orbit[orbit_input.tag] = build_frames(orbit_input)

    frames = frames_by_orbit["orbit1"] + frames_by_orbit["orbit2"]
    if not frames:
        raise ValueError("No aligned frames found")

    missing_masks = [frame for frame in frames if frame.source_mask_path is None]
    if missing_masks and not args.white_mask_if_missing:
        preview = ", ".join(f"{frame.orbit}/{frame.stem}" for frame in missing_masks[:20])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} masks. Examples: {preview}. "
            "Pass --white-mask-if-missing only if this is intentional."
        )

    prepare_output(output_dir, overwrite=args.overwrite, check_only=args.check_only)

    cam_dict: Dict[str, np.ndarray] = {}
    manifest_rows: List[dict] = []

    if not args.check_only:
        for index, frame in enumerate(frames):
            output_filename = f"{index:03d}.png"
            destination_image = output_dir / "image" / output_filename
            destination_mask = output_dir / "mask" / output_filename

            h, w, _ = write_png(frame.source_image_path, destination_image)

            if frame.source_mask_path is not None:
                mask_h, mask_w, _ = write_png(frame.source_mask_path, destination_mask)
                if (mask_h, mask_w) != (h, w):
                    raise ValueError(
                        f"Mask/image resolution mismatch for {frame.orbit}/{frame.stem}: "
                        f"image={(h, w)}, mask={(mask_h, mask_w)}"
                    )
            else:
                write_white_mask(destination_mask, h, w)

            w2c = make_w2c(frame)
            world_mat = intrinsic @ w2c

            if not np.all(np.isfinite(world_mat)):
                raise ValueError(f"Non-finite world_mat for {frame.orbit}/{frame.stem}")
            if abs(float(np.linalg.det(world_mat))) < EPS:
                raise ValueError(f"Singular world_mat for {frame.orbit}/{frame.stem}")

            # NeuS expects the same IDR-style camera keys for every view. The
            # aligned COLMAP pose provides world_mat; the GT-derived scale_mat
            # provides the shared normalized object frame.
            cam_dict[f"camera_mat_{index}"] = intrinsic.astype(np.float32)
            cam_dict[f"camera_mat_inv_{index}"] = np.linalg.inv(intrinsic).astype(np.float32)
            cam_dict[f"world_mat_{index}"] = world_mat.astype(np.float32)
            cam_dict[f"world_mat_inv_{index}"] = np.linalg.inv(world_mat).astype(np.float32)
            cam_dict[f"scale_mat_{index}"] = scale_mat.astype(np.float32)
            cam_dict[f"scale_mat_inv_{index}"] = scale_mat_inv.astype(np.float32)

            manifest_rows.append(make_manifest_row(index, frame, output_filename))

        np.savez(output_dir / "cameras_sphere.npz", **cam_dict)
        write_manifest_csv(output_dir / "source_manifest.csv", manifest_rows)
        with (output_dir / "source_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest_rows, handle, indent=2)

    summary = {
        "output": str(output_dir),
        "check_only": bool(args.check_only),
        "camera_json": str(camera_json_path),
        "scale_mat_json": str(scale_mat_json_path),
        "total_frames": len(frames),
        "orbit_counts": {tag: len(items) for tag, items in frames_by_orbit.items()},
        "missing_masks": len(missing_masks),
        "white_mask_if_missing": bool(args.white_mask_if_missing),
        "intrinsic_4x4": intrinsic.tolist(),
        "scale_mat": scale_mat.tolist(),
        "normalized_camera_distance_stats": camera_distance_stats(frames, scale_mat),
        "alignment_position_error": summarize_optional(frame.position_error for frame in frames),
        "alignment_view_direction_error_deg": summarize_optional(
            frame.view_direction_error_deg for frame in frames
        ),
        "ordering": "orbit1 natural stem order, followed by orbit2 natural stem order",
        "pose_source": "aligned COLMAP world-to-camera poses; original SPE3R GT labels are not used",
    }

    if not args.check_only:
        with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print("=" * 78)
    print("COMBINED NeuS DATASET BUILDER")
    print("=" * 78)
    print(f"Orbit 1 frames:             {len(frames_by_orbit['orbit1'])}")
    print(f"Orbit 2 frames:             {len(frames_by_orbit['orbit2'])}")
    print(f"Total frames:               {len(frames)}")
    print(f"Missing masks:              {len(missing_masks)}")
    print(
        "Normalized camera distance: "
        f"min={summary['normalized_camera_distance_stats']['min']:.6f}, "
        f"max={summary['normalized_camera_distance_stats']['max']:.6f}, "
        f"mean={summary['normalized_camera_distance_stats']['mean']:.6f}"
    )
    if summary["alignment_position_error"] is not None:
        print(
            "Alignment position error:   "
            f"mean={summary['alignment_position_error']['mean']:.6f}, "
            f"max={summary['alignment_position_error']['max']:.6f}"
        )
    if summary["alignment_view_direction_error_deg"] is not None:
        print(
            "View-direction error [deg]: "
            f"mean={summary['alignment_view_direction_error_deg']['mean']:.6f}, "
            f"max={summary['alignment_view_direction_error_deg']['max']:.6f}"
        )

    if args.check_only:
        print("[CHECK ONLY] Validation completed. No files were written.")
    else:
        print(f"Output:                     {output_dir}")
        print(f"NPZ:                        {output_dir / 'cameras_sphere.npz'}")
        print(f"Manifest:                   {output_dir / 'source_manifest.csv'}")
        print(f"Summary:                    {output_dir / 'dataset_summary.json'}")
        print("[DONE] NeuS combined dataset created.")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
