import argparse
import json
import os

import cv2 as cv
import numpy as np


"""
Convert the intermediate SPE3R-like folder into the NeuS dataset layout.

Input folder:

- `images/` and `masks/` with matching `img000001.png`-style names;
- `camera.json` with intrinsics;
- `labels.json` with target pose in the camera frame;
- `scale_mat.json` produced by `scale_mat_builder.py`.

Output folder:

- `image/000.png`, `mask/000.png`, ...;
- `cameras_sphere.npz` with the IDR/NeuS keys expected by `models/dataset.py`.

The script treats each label transform as world-to-camera for the normalized
object frame and constructs `world_mat_i = K @ w2c`. The same `scale_mat` is
stored for every view so NeuS uses one consistent normalized coordinate system.
"""


def quat_wxyz_to_rotmat(q):
    """Quaternion [w, x, y, z] -> rotation matrix 3x3."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Quaternion con norma quasi nulla")
    q = q / n

    w, x, y, z = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def load_intrinsic(camera_json_path):
    """Load camera intrinsics and return a 4x4 homogeneous matrix."""
    with open(camera_json_path, "r") as f:
        cam = json.load(f)

    if "cameraMatrix" in cam:
        K = np.array(cam["cameraMatrix"], dtype=np.float64)
    elif "K" in cam:
        K = np.array(cam["K"], dtype=np.float64)
        if K.shape == ():
            K = np.array(json.loads(cam["K"]), dtype=np.float64)
    else:
        fx = float(cam.get("fx", cam.get("focal")))
        fy = float(cam.get("fy", fx))
        cx = float(cam.get("cx", cam.get("ccx", cam.get("ppx"))))
        cy = float(cam.get("cy", cam.get("ccy", cam.get("ppy"))))
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[:3, :3] = K.astype(np.float32)
    return intrinsic


def load_scale_mat(scale_mat_json_path):
    """Load the 4x4 NeuS normalization matrix."""
    with open(scale_mat_json_path, "r") as f:
        data = json.load(f)

    if "scale_mat" in data:
        scale_mat = np.array(data["scale_mat"], dtype=np.float32)
    else:
        scale_mat = np.array(data, dtype=np.float32)

    if scale_mat.shape != (4, 4):
        raise ValueError(f"scale_mat deve essere 4x4, trovato {scale_mat.shape}")
    return scale_mat


def load_labels(labels_json_path):
    """Load labels sorted by filename so image order and camera order match."""
    with open(labels_json_path, "r") as f:
        labels = json.load(f)

    if not isinstance(labels, list) or len(labels) == 0:
        raise ValueError("labels.json vuoto o non in formato lista")

    return sorted(labels, key=lambda x: x["filename"])


def build_w2c_from_label(label):
    """
    Build the world-to-camera matrix from one SPE3R-like label entry.

    Expected fields:
    - `q_vbs2tango_true` in [w, x, y, z] order;
    - `r_Vo2To_vbs_true` as a 3D translation.
    """
    q = label["q_vbs2tango_true"]
    t = np.asarray(label["r_Vo2To_vbs_true"], dtype=np.float64).reshape(3)

    R = quat_wxyz_to_rotmat(q)

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R.astype(np.float32)
    w2c[:3, 3] = t.astype(np.float32)
    return w2c


def ensure_dirs(out_dir):
    """Create the NeuS output directory structure."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)


def copy_images_and_masks(images_dir, masks_dir, out_dir, labels):
    """Copy images/masks into NeuS numeric naming order."""
    out_img_dir = os.path.join(out_dir, "image")
    out_mask_dir = os.path.join(out_dir, "mask")

    sample_shape = None
    missing_masks = []

    for i, label in enumerate(labels):
        stem = label["filename"]
        src_img = os.path.join(images_dir, stem + ".png")
        src_mask = os.path.join(masks_dir, stem + ".png")

        dst_img = os.path.join(out_img_dir, f"{i:03d}.png")
        dst_mask = os.path.join(out_mask_dir, f"{i:03d}.png")

        img = cv.imread(src_img, cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Immagine mancante/non leggibile: {src_img}")

        if sample_shape is None:
            sample_shape = img.shape

        cv.imwrite(dst_img, img)

        mask = cv.imread(src_mask, cv.IMREAD_UNCHANGED)
        if mask is None:
            missing_masks.append(dst_mask)
        else:
            cv.imwrite(dst_mask, mask)

    if missing_masks:
        if len(sample_shape) == 2:
            white = np.full(sample_shape, 255, dtype=np.uint8)
        else:
            white = np.full(sample_shape[:2], 255, dtype=np.uint8)

        for dst_mask in missing_masks:
            cv.imwrite(dst_mask, white)

        print(f"[WARN] Maschere mancanti: {len(missing_masks)} -> create bianche")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build NeuS cameras_sphere.npz from a SPE3R-like folder"
    )
    parser.add_argument(
        "--spe3r_dir",
        required=True,
        help="Directory con images/, masks/, camera.json, labels.json, scale_mat.json",
    )
    parser.add_argument("--out_dir", required=True, help="Directory output NeuS")
    parser.add_argument(
        "--scale_mat",
        default=None,
        help="Path esplicito a scale_mat.json (default: <spe3r_dir>/scale_mat.json)",
    )
    return parser.parse_args()


def main():
    """Create cameras_sphere.npz and copy the image/mask files."""
    args = parse_args()

    spe3r_dir = args.spe3r_dir
    out_dir = args.out_dir

    images_dir = os.path.join(spe3r_dir, "images")
    masks_dir = os.path.join(spe3r_dir, "masks")
    camera_json_path = os.path.join(spe3r_dir, "camera.json")
    labels_json_path = os.path.join(spe3r_dir, "labels.json")
    scale_mat_json_path = args.scale_mat or os.path.join(spe3r_dir, "scale_mat.json")

    ensure_dirs(out_dir)

    intrinsic = load_intrinsic(camera_json_path)
    scale_mat = load_scale_mat(scale_mat_json_path)
    labels = load_labels(labels_json_path)

    cam_dict = {}

    for i, label in enumerate(labels):
        w2c = build_w2c_from_label(label)
        world_mat = intrinsic @ w2c

        cam_dict[f"camera_mat_{i}"] = intrinsic.astype(np.float32)
        cam_dict[f"camera_mat_inv_{i}"] = np.linalg.inv(intrinsic).astype(np.float32)

        cam_dict[f"world_mat_{i}"] = world_mat.astype(np.float32)
        cam_dict[f"world_mat_inv_{i}"] = np.linalg.inv(world_mat).astype(np.float32)

        cam_dict[f"scale_mat_{i}"] = scale_mat.astype(np.float32)
        cam_dict[f"scale_mat_inv_{i}"] = np.linalg.inv(scale_mat).astype(np.float32)

    copy_images_and_masks(images_dir, masks_dir, out_dir, labels)

    npz_path = os.path.join(out_dir, "cameras_sphere.npz")
    np.savez(npz_path, **cam_dict)

    print("====================================")
    print("Process done!")
    print("OUT_DIR:", out_dir)
    print("NPZ:", npz_path)
    print("Numero frames:", len(labels))
    print("====================================")


if __name__ == "__main__":
    main()
