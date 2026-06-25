#!/usr/bin/env python3
"""
Align two independent COLMAP reconstructions to the common CORTO/Tango frame.

STEP A ONLY: this script performs alignment and diagnostics. It does NOT build a
NeuS dataset and does NOT overwrite COLMAP outputs.

Inputs per orbit:
  - COLMAP sparse model directory or colmap_output root
  - filtered SPE3R-style labels.json (used for frame names and ordering)
  - filtered geometry.json (camera positions in the CORTO/Tango frame)

Outputs per orbit:
  - similarity_all_fit.npz
  - aligned_poses_all_fit.json
  - alignment_report.csv
  - summary.json
  - aligned_sparse_points_all_fit.ply (when points3D.txt/bin is available)

Optional RANSAC outputs (with --run-ransac):
  - similarity_ransac_fit.npz
  - aligned_poses_ransac_fit.json
  - aligned_sparse_points_ransac_fit.ply

Combined output:
  - run_summary.json

Use plot_alignment_diagnostics.py to generate trajectory and error plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


@dataclass
class ColmapPose:
    image_id: int
    image_name: str
    stem: str
    camera_id: int
    qvec_wxyz: np.ndarray
    tvec: np.ndarray
    R_wc: np.ndarray
    C_w: np.ndarray


@dataclass
class Similarity:
    """Sim(3) transform from COLMAP coordinates into the CORTO/Tango frame."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply_points(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        return self.scale * (xyz @ self.rotation.T) + self.translation

def normalize(v: np.ndarray) -> np.ndarray:
    """Return a unit vector and reject degenerate inputs."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < EPS:
        raise ValueError("Cannot normalize a zero-length vector")
    return v / norm


def quat_to_rotmat(q: Sequence[float], order: str = "wxyz") -> np.ndarray:
    """Convert a quaternion to a rotation matrix."""
    q = np.asarray(q, dtype=float).reshape(4)
    if order == "xyzw":
        x, y, z, w = q
    elif order == "wxyz":
        w, x, y, z = q
    else:
        raise ValueError(f"Unsupported quaternion order: {order}")

    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < EPS:
        raise ValueError("Zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a scalar-first quaternion."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0:  # deterministic sign
        q *= -1
    return q


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angle between two vectors in degrees."""
    a = normalize(a)
    b = normalize(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def save_json(path: Path, obj: object) -> None:
    """Write indented JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def find_sparse_model_dir(root: Path) -> Path:
    """Find the COLMAP sparse model directory from a scene root or model path."""
    root = root.expanduser().resolve()
    candidates = [root, root / "sparse" / "0", root / "sparse"]
    for candidate in candidates:
        if (candidate / "images.txt").exists() or (candidate / "images.bin").exists():
            return candidate

    # Fallback: find sparse/0-like directories recursively, prefer shortest path.
    recursive = sorted(
        {p.parent for p in root.rglob("images.txt")} | {p.parent for p in root.rglob("images.bin")},
        key=lambda p: (len(p.parts), str(p)),
    )
    if not recursive:
        raise FileNotFoundError(
            f"No COLMAP sparse model found below {root}. Expected images.txt or images.bin."
        )
    return recursive[0]


def ensure_text_model(model_dir: Path, converted_dir: Path, colmap_exe: str) -> Path:
    """Return a model directory containing images.txt, converting BIN -> TXT if needed."""
    if (model_dir / "images.txt").exists():
        return model_dir
    if not (model_dir / "images.bin").exists():
        raise FileNotFoundError(f"Neither images.txt nor images.bin found in {model_dir}")

    converted_dir.mkdir(parents=True, exist_ok=True)
    command = [
        colmap_exe,
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(converted_dir),
        "--output_type",
        "TXT",
    ]
    print("[COLMAP] Converting sparse model BIN -> TXT:")
    print("         " + " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"COLMAP executable not found: {colmap_exe}. "
            "Pass --colmap-exe or convert the model to TXT manually."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"COLMAP model_converter failed with exit code {exc.returncode}") from exc
    return converted_dir


def read_colmap_images_txt(path: Path) -> Dict[str, ColmapPose]:
    """Read registered image poses from COLMAP images.txt.

    Header lines contain exactly 10 tokens. POINTS2D lines are ignored.
    """
    poses: Dict[str, ColmapPose] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 10:
                continue  # POINTS2D line
            try:
                image_id = int(parts[0])
                q = np.asarray([float(v) for v in parts[1:5]], dtype=float)
                t = np.asarray([float(v) for v in parts[5:8]], dtype=float)
                camera_id = int(parts[8])
            except ValueError:
                continue
            image_name = parts[9]
            stem = Path(image_name).stem
            R_wc = quat_to_rotmat(q, order="wxyz")
            C_w = -R_wc.T @ t
            if stem in poses:
                raise ValueError(f"Duplicate COLMAP image stem in {path}: {stem}")
            poses[stem] = ColmapPose(
                image_id=image_id,
                image_name=image_name,
                stem=stem,
                camera_id=camera_id,
                qvec_wxyz=q,
                tvec=t,
                R_wc=R_wc,
                C_w=C_w,
            )
    if not poses:
        raise ValueError(f"No registered image poses parsed from {path}")
    return poses


def read_label_stems(path: Path) -> List[str]:
    """Read labels.json and return frame stems in filtered dataset order."""
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")

    ordered_stems: List[str] = []
    for index, record in enumerate(records):
        try:
            filename = str(record["filename"])
            stem = Path(filename).stem
        except Exception as exc:
            raise ValueError(f"Invalid label record #{index} in {path}: {record}") from exc
        if stem in ordered_stems:
            raise ValueError(f"Duplicate label filename stem in {path}: {stem}")
        ordered_stems.append(stem)
    if not ordered_stems:
        raise ValueError(f"No labels found in {path}")
    return ordered_stems


def read_filtered_geometry_camera_positions(path: Path) -> np.ndarray:
    """Load filtered CORTO camera positions used as alignment targets."""
    with path.open("r", encoding="utf-8") as f:
        geometry = json.load(f)
    try:
        positions = np.asarray(geometry["camera"]["position"], dtype=float)
    except Exception as exc:
        raise ValueError(f"Expected geometry['camera']['position'] in {path}") from exc
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Invalid camera positions shape in {path}: {positions.shape}")
    return positions


def map_geometry_centers_to_stems(
    ordered_stems: Sequence[str], geometry_positions: np.ndarray
) -> Dict[str, np.ndarray]:
    """Associate each filtered label stem with its CORTO camera center."""
    if len(ordered_stems) != len(geometry_positions):
        raise ValueError(
            "Filtered labels.json and geometry.json must contain the same frames in the same order. "
            f"Got labels={len(ordered_stems)}, geometry={len(geometry_positions)}."
        )
    return {stem: geometry_positions[index] for index, stem in enumerate(ordered_stems)}


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Similarity:
    """Least-squares similarity dst ~= scale * R @ src + t."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError("At least 3 point correspondences are required for Sim(3) alignment")

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    covariance = (dst_c.T @ src_c) / n
    U, singular_values, Vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1
    R = U @ correction @ Vt
    var_src = float(np.mean(np.sum(src_c**2, axis=1)))
    if var_src < EPS:
        raise ValueError("Degenerate source trajectory: near-zero variance")
    scale = float(np.sum(singular_values * np.diag(correction)) / var_src)
    t = mu_dst - scale * (R @ mu_src)
    return Similarity(scale=scale, rotation=R, translation=t)


def residuals(similarity: Similarity, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute Euclidean residuals after applying a similarity transform."""
    return np.linalg.norm(similarity.apply_points(src) - dst, axis=1)


def automatic_ransac_threshold(initial_residuals: np.ndarray) -> float:
    """Choose a robust residual threshold from median absolute deviation."""
    initial_residuals = np.asarray(initial_residuals, dtype=float)
    median = float(np.median(initial_residuals))
    mad = float(np.median(np.abs(initial_residuals - median)))
    robust_sigma = 1.4826 * mad
    # Keep a small absolute floor in CORTO units while avoiding a zero threshold.
    return max(0.05, median + 3.0 * robust_sigma)


def ransac_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> Tuple[Similarity, np.ndarray]:
    """Fit a robust Sim(3) transform with minimal 3-point RANSAC samples."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)
    if n < 3:
        raise ValueError("RANSAC requires at least 3 correspondences")
    rng = np.random.default_rng(seed)
    best_inliers: Optional[np.ndarray] = None
    best_score: Optional[Tuple[int, float]] = None

    for _ in range(iterations):
        ids = rng.choice(n, size=3, replace=False)
        try:
            model = umeyama_similarity(src[ids], dst[ids])
        except (ValueError, np.linalg.LinAlgError):
            continue
        err = residuals(model, src, dst)
        inliers = err <= threshold
        count = int(np.sum(inliers))
        if count < 3:
            continue
        score = (count, -float(np.median(err[inliers])))
        if best_score is None or score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None:
        print("[WARNING] RANSAC did not find a valid model; falling back to all-point fit")
        model = umeyama_similarity(src, dst)
        return model, np.ones(n, dtype=bool)

    # Refine and reclassify until stable.
    inliers = best_inliers
    for _ in range(10):
        model = umeyama_similarity(src[inliers], dst[inliers])
        new_inliers = residuals(model, src, dst) <= threshold
        if int(np.sum(new_inliers)) < 3:
            break
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    model = umeyama_similarity(src[inliers], dst[inliers])
    return model, inliers


def transform_colmap_pose(pose: ColmapPose, similarity: Similarity) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned (R_wc, t_wc, C_w) in the CORTO/Tango frame."""
    C_aligned = similarity.apply_points(pose.C_w.reshape(1, 3))[0]
    R_cw_src = pose.R_wc.T
    R_cw_aligned = similarity.rotation @ R_cw_src
    R_wc_aligned = R_cw_aligned.T
    t_aligned = -R_wc_aligned @ C_aligned
    return R_wc_aligned, t_aligned, C_aligned


def read_points3d_txt(path: Path) -> np.ndarray:
    """Read XYZ points from COLMAP points3D.txt."""
    points: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    return np.asarray(points, dtype=float).reshape(-1, 3)


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    """Write a minimal ASCII PLY point cloud."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.10g} {y:.10g} {z:.10g}\n")


def summarize_errors(values: np.ndarray) -> dict:
    """Return common summary statistics for residual arrays."""
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def save_similarity(path: Path, similarity: Similarity, extra: Optional[dict] = None) -> None:
    """Save a Sim(3) transform and optional metadata to .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scale": np.asarray(similarity.scale),
        "rotation": similarity.rotation,
        "translation": similarity.translation,
    }
    if extra:
        for key, value in extra.items():
            payload[key] = np.asarray(value)
    np.savez(path, **payload)


def pose_to_record(
    stem: str,
    pose: ColmapPose,
    gt_center: np.ndarray,
    similarity: Similarity,
    is_ransac_inlier: Optional[bool],
) -> dict:
    """Convert one COLMAP pose into an aligned output JSON record."""
    R_wc, t_wc, center = transform_colmap_pose(pose, similarity)
    forward = R_wc.T @ np.array([0.0, 0.0, 1.0])
    ideal_forward = -gt_center
    return {
        "filename": pose.image_name,
        "stem": stem,
        "image_id": int(pose.image_id),
        "camera_id": int(pose.camera_id),
        "q_wc_wxyz_aligned": rotmat_to_quat_wxyz(R_wc).tolist(),
        "t_wc_aligned": t_wc.tolist(),
        "camera_center_aligned": center.tolist(),
        "camera_center_gt": gt_center.tolist(),
        "position_error": float(np.linalg.norm(center - gt_center)),
        "view_direction_error_deg": float(angle_deg(forward, ideal_forward)),
        "ransac_inlier": is_ransac_inlier,
    }


def align_one_orbit(
    tag: str,
    colmap_root: Path,
    labels_path: Path,
    geometry_path: Path,
    output_root: Path,
    colmap_exe: str,
    run_ransac: bool,
    ransac_threshold: Optional[float],
    ransac_iterations: int,
    seed: int,
) -> dict:
    """Align one COLMAP orbit to its filtered CORTO camera trajectory."""
    print("\n" + "=" * 78)
    print(f"ALIGNMENT: {tag}")
    print("=" * 78)

    orbit_out = output_root / tag
    orbit_out.mkdir(parents=True, exist_ok=True)

    model_dir = find_sparse_model_dir(colmap_root)
    text_model_dir = ensure_text_model(model_dir, output_root / "_converted_models" / tag, colmap_exe)
    images_txt = text_model_dir / "images.txt"
    points_txt = text_model_dir / "points3D.txt"

    print(f"[INPUT] COLMAP model: {model_dir}")
    print(f"[INPUT] Text model:   {text_model_dir}")
    print(f"[INPUT] Labels:       {labels_path}")
    print(f"[INPUT] Geometry:     {geometry_path}")

    colmap_poses = read_colmap_images_txt(images_txt)
    ordered_label_stems = read_label_stems(labels_path)
    geometry_positions = read_filtered_geometry_camera_positions(geometry_path)
    gt_by_stem = map_geometry_centers_to_stems(ordered_label_stems, geometry_positions)

    # Preserve the filtered labels/geometry order, which is the trajectory order.
    common_stems = [stem for stem in ordered_label_stems if stem in colmap_poses]
    registered_without_label = sorted(set(colmap_poses) - set(gt_by_stem))
    labels_not_registered = [stem for stem in ordered_label_stems if stem not in colmap_poses]
    if registered_without_label:
        raise ValueError(
            f"{tag}: registered COLMAP images without labels: {registered_without_label[:20]}"
        )
    if len(common_stems) < 3:
        raise ValueError(f"{tag}: only {len(common_stems)} COLMAP-label correspondences found")

    src_centers = np.vstack([colmap_poses[s].C_w for s in common_stems])
    gt_centers = np.vstack([gt_by_stem[s] for s in common_stems])

    similarity_all = umeyama_similarity(src_centers, gt_centers)
    similarity_ransac: Optional[Similarity] = None
    ransac_inliers: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    if run_ransac:
        threshold = (
            float(ransac_threshold)
            if ransac_threshold is not None
            else automatic_ransac_threshold(residuals(similarity_all, src_centers, gt_centers))
        )
        similarity_ransac, ransac_inliers = ransac_similarity(
            src=src_centers,
            dst=gt_centers,
            threshold=threshold,
            iterations=ransac_iterations,
            seed=seed,
        )

    records_all = [
        pose_to_record(
            stem,
            colmap_poses[stem],
            gt_by_stem[stem],
            similarity_all,
            bool(ransac_inliers[index]) if ransac_inliers is not None else None,
        )
        for index, stem in enumerate(common_stems)
    ]
    records_ransac = (
        [
            pose_to_record(
                stem,
                colmap_poses[stem],
                gt_by_stem[stem],
                similarity_ransac,
                bool(ransac_inliers[index]),
            )
            for index, stem in enumerate(common_stems)
        ]
        if similarity_ransac is not None and ransac_inliers is not None
        else None
    )

    errors_all = np.asarray([row["position_error"] for row in records_all], dtype=float)
    view_all = np.asarray([row["view_direction_error_deg"] for row in records_all], dtype=float)
    aligned_centers_all = np.asarray([row["camera_center_aligned"] for row in records_all], dtype=float)

    save_similarity(
        orbit_out / "similarity_all_fit.npz",
        similarity_all,
        extra={"registered_names": np.asarray(common_stems)},
    )
    save_json(orbit_out / "aligned_poses_all_fit.json", records_all)
    if similarity_ransac is not None and ransac_inliers is not None and records_ransac is not None:
        save_similarity(
            orbit_out / "similarity_ransac_fit.npz",
            similarity_ransac,
            extra={"registered_names": np.asarray(common_stems), "ransac_inliers": ransac_inliers},
        )
        save_json(orbit_out / "aligned_poses_ransac_fit.json", records_ransac)

    with (orbit_out / "alignment_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stem",
                "image_name",
                "ransac_inlier",
                "position_error_all_fit",
                "view_direction_error_deg_all_fit",
                "gt_cx",
                "gt_cy",
                "gt_cz",
                "aligned_all_cx",
                "aligned_all_cy",
                "aligned_all_cz",
            ]
        )
        for i, stem in enumerate(common_stems):
            writer.writerow(
                [
                    stem,
                    colmap_poses[stem].image_name,
                    bool(ransac_inliers[i]) if ransac_inliers is not None else "",
                    errors_all[i],
                    view_all[i],
                    *gt_centers[i].tolist(),
                    *aligned_centers_all[i].tolist(),
                ]
            )

    sparse_points_all: Optional[np.ndarray] = None
    if points_txt.exists():
        sparse_src = read_points3d_txt(points_txt)
        if len(sparse_src):
            sparse_points_all = similarity_all.apply_points(sparse_src)
            write_ascii_ply(orbit_out / "aligned_sparse_points_all_fit.ply", sparse_points_all)
            if similarity_ransac is not None:
                write_ascii_ply(
                    orbit_out / "aligned_sparse_points_ransac_fit.ply",
                    similarity_ransac.apply_points(sparse_src),
                )
            print(f"[POINTS] Transformed sparse points: {len(sparse_src)}")

    summary = {
        "tag": tag,
        "colmap_model_dir": str(model_dir),
        "text_model_dir": str(text_model_dir),
        "labels_path": str(labels_path),
        "geometry_path": str(geometry_path),
        "registered_colmap_images": len(colmap_poses),
        "label_records": len(ordered_label_stems),
        "matched_registered_images": len(common_stems),
        "registered_without_label": registered_without_label,
        "labels_not_registered": labels_not_registered,
        "frame_order": "filtered labels.json order, restricted to COLMAP-registered images",
        "ground_truth_source": "geometry.camera.position mapped by filtered labels.json order",
        "all_fit_similarity": {
            "scale": similarity_all.scale,
            "rotation": similarity_all.rotation.tolist(),
            "translation": similarity_all.translation.tolist(),
        },
        "ransac_enabled": run_ransac,
        "position_error_all_fit": summarize_errors(errors_all),
        "view_direction_error_deg_all_fit": summarize_errors(view_all),
        "sparse_points_transformed": int(len(sparse_points_all)) if sparse_points_all is not None else 0,
    }
    if similarity_ransac is not None and ransac_inliers is not None and records_ransac is not None:
        errors_ransac = np.asarray([row["position_error"] for row in records_ransac], dtype=float)
        view_ransac = np.asarray([row["view_direction_error_deg"] for row in records_ransac], dtype=float)
        summary["ransac_fit_similarity"] = {
            "scale": similarity_ransac.scale,
            "rotation": similarity_ransac.rotation.tolist(),
            "translation": similarity_ransac.translation.tolist(),
            "threshold": threshold,
            "iterations": ransac_iterations,
            "inliers": int(np.sum(ransac_inliers)),
            "outliers": int(np.sum(~ransac_inliers)),
            "outlier_stems": [s for s, ok in zip(common_stems, ransac_inliers) if not ok],
        }
        summary["position_error_ransac_fit_all_images"] = summarize_errors(errors_ransac)
        summary["position_error_ransac_fit_inliers_only"] = summarize_errors(errors_ransac[ransac_inliers])
        summary["view_direction_error_deg_ransac_fit"] = summarize_errors(view_ransac)
    save_json(orbit_out / "summary.json", summary)

    print(f"[COLMAP] Registered images:        {len(colmap_poses)}")
    print(f"[MATCH]  COLMAP-label matches:     {len(common_stems)}")
    print(f"[MATCH]  Labels not registered:    {len(labels_not_registered)}")
    print(f"[SIM3]   All-fit scale:            {similarity_all.scale:.10g}")
    if ransac_inliers is not None and threshold is not None:
        print(f"[SIM3]   RANSAC threshold:         {threshold:.6g}")
        print(f"[SIM3]   RANSAC inliers:           {int(np.sum(ransac_inliers))}/{len(ransac_inliers)}")
    print(
        "[ERROR]  Position all-fit:          "
        f"mean={np.mean(errors_all):.6g}, median={np.median(errors_all):.6g}, max={np.max(errors_all):.6g}"
    )
    print(
        "[ERROR]  View direction all-fit:    "
        f"mean={np.mean(view_all):.6g} deg, median={np.median(view_all):.6g} deg, max={np.max(view_all):.6g} deg"
    )
    if labels_not_registered:
        print(f"[INFO]   Labels not registered: {labels_not_registered}")
    if ransac_inliers is not None and int(np.sum(~ransac_inliers)):
        print(f"[INFO]   RANSAC candidate outliers: {[s for s, ok in zip(common_stems, ransac_inliers) if not ok]}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align two independent COLMAP reconstructions to the common CORTO/Tango frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--orbit1-colmap", type=Path, required=True, help="First orbit colmap_output root or sparse model directory")
    parser.add_argument("--orbit1-labels", type=Path, required=True, help="Filtered labels.json defining frame names and ordering for orbit 1")
    parser.add_argument("--orbit1-geometry", type=Path, required=True, help="Filtered geometry.json containing CORTO camera positions for orbit 1")
    parser.add_argument("--orbit2-colmap", type=Path, required=True, help="Second orbit colmap_output root or sparse model directory")
    parser.add_argument("--orbit2-labels", type=Path, required=True, help="Filtered labels.json defining frame names and ordering for orbit 2")
    parser.add_argument("--orbit2-geometry", type=Path, required=True, help="Filtered geometry.json containing CORTO camera positions for orbit 2")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for aligned poses and reports")
    parser.add_argument("--colmap-exe", default="colmap", help="COLMAP executable used only when BIN -> TXT conversion is needed")

    parser.add_argument("--run-ransac", action="store_true", help="Also estimate a secondary RANSAC alignment for outlier diagnostics")
    parser.add_argument("--ransac-threshold", type=float, default=None, help="Candidate outlier threshold in CORTO units; default is estimated from all-fit residuals")
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_path(path: Optional[Path], label: str) -> Optional[Path]:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    orbit1_colmap = validate_path(args.orbit1_colmap, "orbit1 COLMAP root")
    orbit1_labels = validate_path(args.orbit1_labels, "orbit1 labels")
    orbit1_geometry = validate_path(args.orbit1_geometry, "orbit1 geometry")
    orbit2_colmap = validate_path(args.orbit2_colmap, "orbit2 COLMAP root")
    orbit2_labels = validate_path(args.orbit2_labels, "orbit2 labels")
    orbit2_geometry = validate_path(args.orbit2_geometry, "orbit2 geometry")
    assert orbit1_colmap and orbit1_labels and orbit1_geometry
    assert orbit2_colmap and orbit2_labels and orbit2_geometry

    print("=" * 78)
    print("COLMAP ORBIT ALIGNMENT TO CORTO/TANGO - STEP A ONLY")
    print("=" * 78)
    print(f"Output: {output}")
    print("This script does not create a NeuS dataset and does not modify COLMAP outputs.")

    common_kwargs = dict(
        output_root=output,
        colmap_exe=args.colmap_exe,
        run_ransac=args.run_ransac,
        ransac_threshold=args.ransac_threshold,
        ransac_iterations=args.ransac_iterations,
        seed=args.seed,
    )

    orbit1 = align_one_orbit(
        tag="orbit1",
        colmap_root=orbit1_colmap,
        labels_path=orbit1_labels,
        geometry_path=orbit1_geometry,
        **common_kwargs,
    )
    orbit2 = align_one_orbit(
        tag="orbit2",
        colmap_root=orbit2_colmap,
        labels_path=orbit2_labels,
        geometry_path=orbit2_geometry,
        **common_kwargs,
    )

    run_summary = {
        "step": "A_alignment",
        "output": str(output),
        "orbit1": orbit1,
        "orbit2": orbit2,
        "next_step": (
            "Run plot_alignment_diagnostics.py to inspect plots, then build the combined NeuS dataset."
        ),
    }
    save_json(output / "run_summary.json", run_summary)

    print("\n" + "=" * 78)
    print("COMPLETED: STEP A ALIGNMENT")
    print("=" * 78)
    print(f"Results: {output}")
    print("Inspect:")
    print(f"  - {output / 'run_summary.json'}")
    print(f"  - {output / 'orbit1' / 'alignment_report.csv'}")
    print(f"  - {output / 'orbit2' / 'alignment_report.csv'}")
    print("Generate plots separately with plot_alignment_diagnostics.py.")
    print("No NeuS dataset has been created.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
