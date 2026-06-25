#!/usr/bin/env python3
"""
Prune registered images from a COLMAP sparse model without modifying the original.

The script:
1. Converts sparse/0 from BIN to TXT.
2. Removes selected registered images from images.txt.
3. Removes their observations from points3D.txt.
4. Drops 3D points observed by fewer than --min-track-length retained images.
5. Replaces references to removed 3D points with -1 in retained image observations.
6. Converts the pruned TXT model back to BIN in OUTPUT_ROOT/sparse/0.
7. Optionally copies images and masks, excluding the pruned files.
8. Saves a JSON summary and retained/excluded image lists.

The output is suitable for:
- rerunning an existing alignment script unchanged;
- rerunning imgs2poses.py with a pose_utils.py implementation that supports
  non-contiguous COLMAP IMAGE_ID values.

Example:
python prune_colmap_model.py \
  --input-root "/path/to/colmap_output" \
  --output-root "/path/to/colmap_output_pruned" \
  --exclude img000002 img000003 \
  --copy-assets
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class ImageRecord:
    image_id: int
    header_tokens: List[str]
    name: str
    points2d: List[Tuple[str, str, int]]


@dataclass
class Point3DRecord:
    point3d_id: int
    fixed_tokens: List[str]
    track: List[Tuple[int, int]]


def normalize_image_key(value: str) -> str:
    """Compare names robustly whether or not an extension/path is supplied."""
    return Path(value).stem


def run_command(command: Sequence[str]) -> None:
    """Run a COLMAP command and surface stdout/stderr on failure."""
    print("[RUN]", " ".join(command))
    result = subprocess.run(command, text=True, capture_output=True)
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr)
        raise RuntimeError(
            f"Command failed with return code {result.returncode}: {' '.join(command)}"
        )


def convert_model(input_path: Path, output_path: Path, output_type: str) -> None:
    """Convert a COLMAP model between BIN and TXT representations."""
    output_path.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(input_path),
            "--output_path",
            str(output_path),
            "--output_type",
            output_type,
        ]
    )


def read_non_comment_lines(path: Path) -> List[str]:
    """Read a COLMAP TXT file while skipping comment lines."""
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.rstrip("\n")
            if stripped.startswith("#"):
                continue
            lines.append(stripped)
    return lines


def read_images_txt(path: Path) -> Tuple[List[str], Dict[int, ImageRecord]]:
    """Parse COLMAP images.txt into image records and POINTS2D tracks."""
    comments: List[str] = []
    data_lines: List[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("#"):
                comments.append(line)
            else:
                data_lines.append(line)

    if len(data_lines) % 2 != 0:
        raise ValueError(
            f"{path} contains an odd number of non-comment lines. "
            "Expected two lines per registered image."
        )

    records: Dict[int, ImageRecord] = {}
    for idx in range(0, len(data_lines), 2):
        header_line = data_lines[idx].strip()
        points_line = data_lines[idx + 1].strip()

        if not header_line:
            raise ValueError(f"Unexpected empty image header in {path} at pair {idx // 2}.")

        header_tokens = header_line.split()
        if len(header_tokens) < 10:
            raise ValueError(f"Malformed image header in {path}: {header_line}")

        image_id = int(header_tokens[0])
        name = header_tokens[9]

        point_tokens = points_line.split() if points_line else []
        if len(point_tokens) % 3 != 0:
            raise ValueError(
                f"Malformed POINTS2D line for IMAGE_ID={image_id}: expected triples."
            )

        points2d: List[Tuple[str, str, int]] = []
        for j in range(0, len(point_tokens), 3):
            points2d.append((point_tokens[j], point_tokens[j + 1], int(point_tokens[j + 2])))

        records[image_id] = ImageRecord(
            image_id=image_id,
            header_tokens=header_tokens,
            name=name,
            points2d=points2d,
        )

    return comments, records


def read_points3d_txt(path: Path) -> Tuple[List[str], Dict[int, Point3DRecord]]:
    """Parse COLMAP points3D.txt and keep each 3D point track."""
    comments: List[str] = []
    records: Dict[int, Point3DRecord] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("#"):
                comments.append(line)
                continue
            if not line.strip():
                continue

            tokens = line.split()
            if len(tokens) < 8:
                raise ValueError(f"Malformed points3D row in {path}: {line}")

            fixed_tokens = tokens[:8]
            track_tokens = tokens[8:]
            if len(track_tokens) % 2 != 0:
                raise ValueError(
                    f"Malformed TRACK for POINT3D_ID={tokens[0]}: expected IMAGE_ID/POINT2D_IDX pairs."
                )

            point3d_id = int(tokens[0])
            track: List[Tuple[int, int]] = []
            for j in range(0, len(track_tokens), 2):
                track.append((int(track_tokens[j]), int(track_tokens[j + 1])))

            records[point3d_id] = Point3DRecord(
                point3d_id=point3d_id,
                fixed_tokens=fixed_tokens,
                track=track,
            )

    return comments, records


def write_images_txt(path: Path, comments: Sequence[str], records: Dict[int, ImageRecord]) -> None:
    """Write a valid COLMAP images.txt file from parsed records."""
    with path.open("w", encoding="utf-8") as handle:
        for line in comments:
            handle.write(line + "\n")

        for image_id in sorted(records):
            record = records[image_id]
            handle.write(" ".join(record.header_tokens) + "\n")

            flat: List[str] = []
            for x, y, point3d_id in record.points2d:
                flat.extend([x, y, str(point3d_id)])
            handle.write(" ".join(flat) + "\n")


def write_points3d_txt(
    path: Path, comments: Sequence[str], records: Dict[int, Point3DRecord]
) -> None:
    """Write a valid COLMAP points3D.txt file from parsed records."""
    with path.open("w", encoding="utf-8") as handle:
        for line in comments:
            handle.write(line + "\n")

        for point3d_id in sorted(records):
            record = records[point3d_id]
            track_tokens: List[str] = []
            for image_id, point2d_idx in record.track:
                track_tokens.extend([str(image_id), str(point2d_idx)])

            row = record.fixed_tokens + track_tokens
            handle.write(" ".join(row) + "\n")


def copy_assets(
    input_root: Path,
    output_root: Path,
    excluded_keys: set[str],
    folder_names: Iterable[str] = ("images", "masks"),
) -> Dict[str, int]:
    """Copy images/masks while excluding the pruned image stems."""
    copied_counts: Dict[str, int] = {}

    for folder_name in folder_names:
        source_dir = input_root / folder_name
        if not source_dir.exists():
            print(f"[ASSETS] Skip missing folder: {source_dir}")
            copied_counts[folder_name] = 0
            continue

        destination_dir = output_root / folder_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for source_file in sorted(source_dir.iterdir()):
            if not source_file.is_file():
                continue
            if normalize_image_key(source_file.name) in excluded_keys:
                continue
            shutil.copy2(source_file, destination_dir / source_file.name)
            copied += 1

        copied_counts[folder_name] = copied
        print(f"[ASSETS] Copied {copied} files: {source_dir} -> {destination_dir}")

    return copied_counts


def count_observations(points: Dict[int, Point3DRecord]) -> int:
    """Count total IMAGE_ID/POINT2D observations across all 3D tracks."""
    return sum(len(point.track) for point in points.values())


def write_list(path: Path, values: Iterable[str]) -> None:
    """Write one string per line."""
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(value + "\n")


def main() -> None:
    """Prune registered images and rebuild a consistent COLMAP sparse model."""
    parser = argparse.ArgumentParser(
        description="Prune selected registered images from a COLMAP sparse model."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Original COLMAP scene root containing sparse/0.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Destination scene root. The original model is never modified.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        required=True,
        help="Image names or stems to remove, e.g. img000002 img000003.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=2,
        help="Drop 3D points observed by fewer retained images after pruning. Default: 2.",
    )
    parser.add_argument(
        "--copy-assets",
        action="store_true",
        help="Copy images/ and masks/ into the output root, excluding pruned files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete OUTPUT_ROOT first if it already exists.",
    )
    args = parser.parse_args()

    input_root: Path = args.input_root.resolve()
    output_root: Path = args.output_root.resolve()
    input_model = input_root / "sparse" / "0"
    output_model = output_root / "sparse" / "0"

    if not input_model.exists():
        raise FileNotFoundError(f"COLMAP sparse model not found: {input_model}")

    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP executable not found in PATH.")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}\n"
                "Use --overwrite to replace it explicitly."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    excluded_requested = {normalize_image_key(value) for value in args.exclude}

    print("=" * 78)
    print("COLMAP SPARSE MODEL PRUNING")
    print("=" * 78)
    print(f"Input root:       {input_root}")
    print(f"Output root:      {output_root}")
    print(f"Requested prune:  {sorted(excluded_requested)}")
    print(f"Min track length: {args.min_track_length}")

    with tempfile.TemporaryDirectory(prefix="colmap_prune_") as temp_dir:
        temp_path = Path(temp_dir)
        input_txt = temp_path / "input_txt"
        output_txt = temp_path / "output_txt"

        convert_model(input_model, input_txt, "TXT")
        output_txt.mkdir(parents=True, exist_ok=True)

        cameras_txt = input_txt / "cameras.txt"
        images_txt = input_txt / "images.txt"
        points3d_txt = input_txt / "points3D.txt"

        if not cameras_txt.exists() or not images_txt.exists() or not points3d_txt.exists():
            raise FileNotFoundError("Converted COLMAP TXT model is incomplete.")

        image_comments, images = read_images_txt(images_txt)
        point_comments, points = read_points3d_txt(points3d_txt)

        original_image_ids = set(images)
        original_point_ids = set(points)
        observations_before = count_observations(points)

        excluded_image_ids = {
            image_id
            for image_id, record in images.items()
            if normalize_image_key(record.name) in excluded_requested
        }
        excluded_found_keys = {
            normalize_image_key(images[image_id].name) for image_id in excluded_image_ids
        }
        excluded_missing = sorted(excluded_requested - excluded_found_keys)

        if excluded_missing:
            print(f"[WARN] Requested images not registered in the model: {excluded_missing}")

        retained_images: Dict[int, ImageRecord] = {
            image_id: record
            for image_id, record in images.items()
            if image_id not in excluded_image_ids
        }

        # Remove deleted IMAGE_ID observations from each point track.
        candidate_points: Dict[int, Point3DRecord] = {}
        for point3d_id, point in points.items():
            new_track = [
                (image_id, point2d_idx)
                for image_id, point2d_idx in point.track
                if image_id in retained_images
            ]
            if len(new_track) >= args.min_track_length:
                candidate_points[point3d_id] = Point3DRecord(
                    point3d_id=point.point3d_id,
                    fixed_tokens=point.fixed_tokens,
                    track=new_track,
                )

        retained_point_ids = set(candidate_points)

        # Replace dangling point references in POINTS2D with -1.
        for record in retained_images.values():
            updated_points2d: List[Tuple[str, str, int]] = []
            for x, y, point3d_id in record.points2d:
                if point3d_id != -1 and point3d_id not in retained_point_ids:
                    updated_points2d.append((x, y, -1))
                else:
                    updated_points2d.append((x, y, point3d_id))
            record.points2d = updated_points2d

        shutil.copy2(cameras_txt, output_txt / "cameras.txt")
        write_images_txt(output_txt / "images.txt", image_comments, retained_images)
        write_points3d_txt(output_txt / "points3D.txt", point_comments, candidate_points)

        convert_model(output_txt, output_model, "BIN")

        # Keep a human-readable copy next to the BIN model.
        readable_model = output_root / "sparse_txt" / "0"
        readable_model.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_txt / "cameras.txt", readable_model / "cameras.txt")
        shutil.copy2(output_txt / "images.txt", readable_model / "images.txt")
        shutil.copy2(output_txt / "points3D.txt", readable_model / "points3D.txt")

    copied_assets: Dict[str, int] = {}
    if args.copy_assets:
        copied_assets = copy_assets(input_root, output_root, excluded_requested)

    retained_names = sorted(record.name for record in retained_images.values())
    excluded_registered_names = sorted(images[image_id].name for image_id in excluded_image_ids)

    write_list(output_root / "retained_images.txt", retained_names)
    write_list(output_root / "excluded_images.txt", excluded_registered_names)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "requested_exclusions": sorted(excluded_requested),
        "registered_exclusions_found": excluded_registered_names,
        "requested_exclusions_not_registered": excluded_missing,
        "min_track_length": args.min_track_length,
        "images_before": len(original_image_ids),
        "images_after": len(retained_images),
        "images_removed": len(original_image_ids) - len(retained_images),
        "points3D_before": len(original_point_ids),
        "points3D_after": len(retained_point_ids),
        "points3D_removed": len(original_point_ids) - len(retained_point_ids),
        "observations_before": observations_before,
        "observations_after": count_observations(candidate_points),
        "assets_copied": copied_assets,
        "note": (
            "IMAGE_ID values are intentionally preserved. "
            "Use a pose_utils.py implementation that supports non-contiguous IDs "
            "when running imgs2poses.py."
        ),
    }

    with (output_root / "pruning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("-" * 78)
    print(f"[DONE] Registered images: {summary['images_before']} -> {summary['images_after']}")
    print(f"[DONE] 3D points:          {summary['points3D_before']} -> {summary['points3D_after']}")
    print(f"[DONE] Observations:       {summary['observations_before']} -> {summary['observations_after']}")
    print(f"[DONE] Pruned BIN model:   {output_model}")
    print(f"[DONE] Summary:            {output_root / 'pruning_summary.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
