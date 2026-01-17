#!/usr/bin/env python3
"""
SPE3R -> NeuS converter (single model)

Usage:
    python convert_spe3r_to_neus.py --spe3r-root /path/to/SPE3R \
        --model-name hubble --out /path/to/neus/data

This script is an improved, CLI-friendly version of the user's proposal.
Features:
- argparse-based CLI
- safer path validation and helpful error messages
- writes `cameras_spe3r.npz` with keys `world_mat_0..N-1` and `scale_mat_0..N-1`
- copies/renames images and masks to `image/` and `mask/` as `000.png`, `001.png`, ...

Dependencies:
    pip install numpy scipy opencv-python

"""

from pathlib import Path
import zipfile
import tempfile
import argparse
import json
import shutil
import sys
import logging

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_camera_json(camera_path: Path) -> np.ndarray:
    with camera_path.open("r") as f:
        cam = json.load(f)
    K = np.array(cam["cameraMatrix"], dtype=np.float32)
    return K


def load_labels(labels_path: Path):
    with labels_path.open("r") as f:
        labels = json.load(f)
    labels_sorted = sorted(labels, key=lambda x: x["filename"])
    return labels_sorted


def build_world_and_scale_mats(labels, K: np.ndarray):
    world_mats = []
    scale_mats = []
    for entry in labels:
        q = entry["q_vbs2tango_true"]
        t = entry["r_Vo2To_vbs_true"]

        quat_xyzw = [q[1], q[2], q[3], q[0]]
        rot = R.from_quat(quat_xyzw)
        R_w2c = rot.as_matrix().astype(np.float32)

        t_w2c = np.array(t, dtype=np.float32).reshape(3, 1)
        # SPEED+/SPE3R provides extrinsics as X_cam = R * X_world + t
        # so the correct 3x4 extrinsic block is [R | t] (not [R | -R t]).
        Rt = np.concatenate([R_w2c, t_w2c], axis=1)

        P = K @ Rt

        world_mat = np.eye(4, dtype=np.float32)
        world_mat[:3, :4] = P

        scale_mat = np.eye(4, dtype=np.float32)

        world_mats.append(world_mat)
        scale_mats.append(scale_mat)

    world_mats = np.stack(world_mats, axis=0) if len(world_mats) > 0 else np.zeros((0, 4, 4), dtype=np.float32)
    scale_mats = np.stack(scale_mats, axis=0) if len(scale_mats) > 0 else np.zeros((0, 4, 4), dtype=np.float32)
    return world_mats, scale_mats


def rename_and_copy_images_and_masks(labels, src_img_dir: Path, src_mask_dir: Path, dst_img_dir: Path, dst_mask_dir: Path):
    ensure_dir(dst_img_dir)
    ensure_dir(dst_mask_dir)

    for i, entry in enumerate(labels):
        basename = entry["filename"]

        # accept multiple possible extensions for images/masks
        img_candidates = [src_img_dir / f"{basename}.jpg",
                          src_img_dir / f"{basename}.jpeg",
                          src_img_dir / f"{basename}.png"]
        mask_candidates = [src_mask_dir / f"{basename}.png",
                           src_mask_dir / f"{basename}.jpg",
                           src_mask_dir / f"{basename}.jpeg"]

        src_img = next((p for p in img_candidates if p.is_file()), None)
        src_mask = next((p for p in mask_candidates if p.is_file()), None)

        if src_img is None:
            raise FileNotFoundError(f"Immagine non trovata (cercati: {img_candidates}): {basename}")
        if src_mask is None:
            raise FileNotFoundError(f"Maschera non trovata (cercati: {mask_candidates}): {basename}")

        new_name = f"{i:03d}.png"
        dst_img = dst_img_dir / new_name
        dst_mask = dst_mask_dir / new_name

        img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Errore nel leggere immagine: {src_img}")
        cv2.imwrite(str(dst_img), img)

        mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Errore nel leggere maschera: {src_mask}")
        cv2.imwrite(str(dst_mask), mask)

    logging.info("Copiate e rinominate %d immagini e maschere.", len(labels))


def parse_args():
    p = argparse.ArgumentParser(description="Convert SPE3R (single model) to NeuS-like folder")
    p.add_argument("--spe3r-root", type=Path,
                   default=Path("/Users/martino/Desktop/Tesi/codes/SPE3R"),
                   help="root folder of SPE3R dataset (contains camera.json and model subfolders)")
    p.add_argument("--model-name", type=str, default="hst", help="model subfolder name (e.g. hst)")
    p.add_argument("--out", type=Path, default=Path.cwd() / "spe3r_neus",
                   help="output root where the NeuS case folder will be created")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    spe3r_root = args.spe3r_root
    model_dir = spe3r_root / args.model_name

    camera_path = spe3r_root / "camera.json"
    labels_path = model_dir / "labels.json"
    src_img_dir = model_dir / "images"
    src_mask_dir = model_dir / "masks"
    # try to find a normalized model obj inside model_dir (some SPE3R variants name it differently)
    src_model_obj = None
    # common candidate path
    candidate = model_dir / "models" / "model_normalized.obj"
    if candidate.exists():
        src_model_obj = candidate
    else:
        # search for any file containing 'model_normalized.obj' in the filename
        for p in model_dir.glob("**/*model_normalized*.obj"):
            src_model_obj = p
            break

    # If images/masks dirs are not present, try to find zip archives and extract them
    if not src_img_dir.exists():
        # look for *_images.zip in model_dir
        zip_candidates = list(model_dir.glob("*_images.zip"))
        if zip_candidates:
            z = zip_candidates[0]
            logging.info("Estraggo immagini da %s -> %s", z, src_img_dir)
            ensure_dir(src_img_dir)
            with zipfile.ZipFile(z, 'r') as zf:
                zf.extractall(str(src_img_dir))

    if not src_mask_dir.exists():
        zip_candidates = list(model_dir.glob("*_masks.zip"))
        if zip_candidates:
            z = zip_candidates[0]
            logging.info("Estraggo maschere da %s -> %s", z, src_mask_dir)
            ensure_dir(src_mask_dir)
            with zipfile.ZipFile(z, 'r') as zf:
                zf.extractall(str(src_mask_dir))

    # final existence checks
    checks = [camera_path, labels_path]
    if src_img_dir.exists():
        checks.append(src_img_dir)
    else:
        logging.error("Cartella immagini non trovata: %s", src_img_dir)
        sys.exit(2)
    if src_mask_dir.exists():
        checks.append(src_mask_dir)
    else:
        logging.error("Cartella maschere non trovata: %s", src_mask_dir)
        sys.exit(2)

    if src_model_obj is None or not src_model_obj.exists():
        # try find any .obj in model_dir
        any_obj = next(model_dir.glob("**/*.obj"), None)
        if any_obj is None:
            logging.error("Mesh modello non trovata in %s", model_dir)
            sys.exit(2)
        src_model_obj = any_obj

    case_name = f"{args.model_name}_neus"
    case_dir = args.out / case_name
    dst_img_dir = case_dir / "image"
    dst_mask_dir = case_dir / "mask"
    dst_model_dir = case_dir / "model"

    ensure_dir(case_dir)
    ensure_dir(dst_model_dir)

    logging.info("Carico intrinseci da %s", camera_path)
    K = load_camera_json(camera_path)

    logging.info("Carico labels da %s", labels_path)
    labels = load_labels(labels_path)

    logging.info("Costruisco world_mat e scale_mat per %d viste...", len(labels))
    world_mats, scale_mats = build_world_and_scale_mats(labels, K)

    cameras_npz_path = case_dir / "cameras_spe3r.npz"
    logging.info("Salvo %s", cameras_npz_path)
    N = world_mats.shape[0]
    save_dict = {}
    for i in range(N):
        save_dict[f"world_mat_{i}"] = world_mats[i].astype(np.float32)
        save_dict[f"scale_mat_{i}"] = scale_mats[i].astype(np.float32)
    np.savez(cameras_npz_path, **save_dict)

    logging.info("Copio e rinomino immagini e maschere in %s e %s", dst_img_dir, dst_mask_dir)
    rename_and_copy_images_and_masks(labels, src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir)

    dst_model_obj = dst_model_dir / "model_normalized.obj"
    logging.info("Copio mesh normalizzata in %s", dst_model_obj)
    shutil.copy2(src_model_obj, dst_model_obj)

    logging.info("✅ Conversione completata. Cartella caso NeuS: %s", case_dir)


if __name__ == "__main__":
    main()
