#!/usr/bin/env python3
"""
verify_silhouette.py

Purpose:
- Load `cameras_spe3r.npz`, `image/{index:03d}.png`, `model/model_normalized.obj` from a NeuS case
- Project mesh, rasterize silhouette, compute IoU against ground-truth mask `mask/{index:03d}.png`
- Save overlay and predicted mask if requested, and print IoU to stdout

Usage:
  python verify_silhouette.py --case /path/to/case --index 0 --save-overlay out.png --save-mask mask.png

Only minimal dependencies: `numpy`, `opencv-python`
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import cv2
import random


def parse_args():
    p = argparse.ArgumentParser(description="Compute IoU between projected mesh silhouette and ground-truth mask")
    p.add_argument("--case", type=Path, required=True, help="case folder (contains cameras_spe3r.npz, image/, mask/, model/)")
    p.add_argument("--index", type=int, default=None, help="view index (0-based). If omitted, a random available view is chosen")
    p.add_argument("--seed", type=int, default=None, help="optional random seed for reproducible selection when index is omitted")
    p.add_argument("--all", action="store_true", help="compute IoU for all available views and write a CSV report")
    p.add_argument("--overlay-index", type=int, default=None, help="if set, use this index for saving the overlay; otherwise choose random")
    p.add_argument("--iou-csv", type=Path, default=None, help="optional path to save the per-view IoU CSV (default: <case>/iou_report.csv)")
    p.add_argument("--save-overlay", type=Path, default=None, help="path to save overlay image (PNG): silhouette overlaid on the input image")
    p.add_argument("--save-mask", type=Path, default=None, help="path to save predicted mask (PNG)")
    return p.parse_args()


def load_obj_minimal(path: Path):
    verts = []
    faces = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idxs = []
                for token in parts:
                    v_idx = token.split("/")[0]
                    if v_idx:
                        idxs.append(int(v_idx) - 1)
                if len(idxs) >= 3:
                    for i in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[i], idxs[i + 1]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def project_vertices(P, verts):
    V = verts.shape[0]
    homo = np.concatenate([verts, np.ones((V, 1), dtype=np.float32)], axis=1)
    proj = (P @ homo.T).T
    zs = proj[:, 2:3].copy()
    eps = 1e-8
    zs_safe = np.where(np.abs(zs) < eps, eps, zs)
    xy = proj[:, :2] / zs_safe
    return xy, proj[:, 2]


def rasterize_triangles(width, height, pts2d, depths, faces):
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    for f in faces:
        ia, ib, ic = f
        pa = pts2d[ia]
        pb = pts2d[ib]
        pc = pts2d[ic]
        za, zb, zc = depths[ia], depths[ib], depths[ic]
        minx = int(max(0, np.floor(min(pa[0], pb[0], pc[0]))))
        maxx = int(min(width - 1, np.ceil(max(pa[0], pb[0], pc[0]))))
        miny = int(max(0, np.floor(min(pa[1], pb[1], pc[1]))))
        maxy = int(min(height - 1, np.ceil(max(pa[1], pb[1], pc[1]))))
        denom = ((pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1]))
        if abs(denom) < 1e-8:
            continue
        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                w0 = ((pb[1] - pc[1]) * (x - pc[0]) + (pc[0] - pb[0]) * (y - pc[1])) / denom
                w1 = ((pc[1] - pa[1]) * (x - pc[0]) + (pa[0] - pc[0]) * (y - pc[1])) / denom
                w2 = 1 - w0 - w1
                if w0 >= -1e-6 and w1 >= -1e-6 and w2 >= -1e-6:
                    z = w0 * za + w1 * zb + w2 * zc
                    if z < zbuf[y, x]:
                        zbuf[y, x] = z
                        mask[y, x] = 255
    return mask


def overlay_mask_on_image(img, mask, color=(0, 0, 255), alpha=0.5):
    overlay = img.copy()
    colored = np.zeros_like(img)
    colored[:, :] = color
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)[mask_bool]
    return overlay


def main():
    args = parse_args()
    case = args.case
    idx = args.index

    cameras_npz = case / "cameras_spe3r.npz"
    model_obj = case / "model" / "model_normalized.obj"

    # Basic existence checks for required files
    for p in [cameras_npz, model_obj]:
        if not p.exists():
            print(f"Missing required file: {p}", file=sys.stderr)
            sys.exit(2)

    # Load camera npz to determine available view indices
    data = np.load(str(cameras_npz))
    available_keys = [k for k in data.files if k.startswith("world_mat_")]
    if len(available_keys) == 0:
        print(f"No world_mat_* keys found in {cameras_npz}", file=sys.stderr)
        sys.exit(2)
    available_indices = sorted([int(k.split("world_mat_")[-1]) for k in available_keys])

    # Prepare mesh once
    verts, faces = load_obj_minimal(model_obj)

    # Choose overlay index: explicit > args.overlay_index > random choice
    if args.overlay_index is not None:
        overlay_idx = int(args.overlay_index)
    else:
        if args.seed is not None:
            random.seed(int(args.seed))
        overlay_idx = int(random.choice(available_indices))

    # If not computing all, choose the index to evaluate (either provided or random)
    if not args.all:
        if args.index is None:
            if args.seed is not None:
                random.seed(int(args.seed))
            idx = int(random.choice(available_indices))
        else:
            idx = int(args.index)

        key = f"world_mat_{idx}"
        if key not in data:
            print(f"world_mat_{idx} not found in {cameras_npz}", file=sys.stderr)
            sys.exit(2)
        world_mat = data[key]
        P = world_mat[:3, :4]

        img_path = case / "image" / f"{idx:03d}.png"
        gt_mask_path = case / "mask" / f"{idx:03d}.png"

        for p in [img_path, gt_mask_path]:
            if not p.exists():
                print(f"Missing required file: {p}", file=sys.stderr)
                sys.exit(2)

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        pts2d, depths = project_vertices(P, verts)
        mask = rasterize_triangles(w, h, pts2d, depths, faces)

        # IoU for single view
        gt = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt > 127).astype(np.uint8)
        pred_bin = (mask > 0).astype(np.uint8)
        inter = int(np.logical_and(gt_bin, pred_bin).sum())
        union = int(np.logical_or(gt_bin, pred_bin).sum())
        iou = float(inter) / union if union > 0 else (1.0 if inter == 0 else 0.0)
        print(f"IoU: {iou:.6f} (inter={inter}, union={union})")

        # Save predicted mask and overlay if requested
        if args.save_mask:
            args.save_mask.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_mask), mask)

        if args.save_overlay:
            # append index to filename if not present
            overlay_path = Path(str(args.save_overlay))
            stem = overlay_path.stem
            suffixed = overlay_path.with_name(f"{stem}_{idx:03d}" + overlay_path.suffix)
            suffixed.parent.mkdir(parents=True, exist_ok=True)
            overlay = overlay_mask_on_image(img, mask)
            cv2.imwrite(str(suffixed), overlay)

        return

    # If here: args.all is True -> compute IoU for all available indices
    results = []
    for idx in available_indices:
        key = f"world_mat_{idx}"
        if key not in data:
            continue
        world_mat = data[key]
        P = world_mat[:3, :4]
        img_path = case / "image" / f"{idx:03d}.png"
        gt_mask_path = case / "mask" / f"{idx:03d}.png"
        if not img_path.exists() or not gt_mask_path.exists():
            continue
        img_tmp = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        h, w = img_tmp.shape[:2]
        pts2d, depths = project_vertices(P, verts)
        mask_tmp = rasterize_triangles(w, h, pts2d, depths, faces)
        gt = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt > 127).astype(np.uint8)
        pred_bin = (mask_tmp > 0).astype(np.uint8)
        inter = int(np.logical_and(gt_bin, pred_bin).sum())
        union = int(np.logical_or(gt_bin, pred_bin).sum())
        iou = float(inter) / union if union > 0 else (1.0 if inter == 0 else 0.0)
        results.append((idx, iou, inter, union))

    # Save CSV
    import csv
    csv_path = args.iou_csv if args.iou_csv is not None else (case / "iou_report.csv")
    with open(str(csv_path), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "iou", "inter", "union"])
        for r in results:
            writer.writerow(r)

    # Print summary
    ious = [r[1] for r in results]
    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    print(f"Computed IoU for {len(results)} views. Mean IoU: {mean_iou:.6f}. CSV saved to {csv_path}")

    # Save overlay for chosen overlay_idx (only one)
    if args.save_overlay:
        idx = overlay_idx
        key = f"world_mat_{idx}"
        if key in data:
            world_mat = data[key]
            P = world_mat[:3, :4]
            img_path = case / "image" / f"{idx:03d}.png"
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            pts2d, depths = project_vertices(P, verts)
            mask_tmp = rasterize_triangles(w, h, pts2d, depths, faces)
            # make filename include index
            overlay_path = Path(str(args.save_overlay))
            stem = overlay_path.stem
            suffixed = overlay_path.with_name(f"{stem}_{idx:03d}" + overlay_path.suffix)
            suffixed.parent.mkdir(parents=True, exist_ok=True)
            overlay = overlay_mask_on_image(img, mask_tmp)
            cv2.imwrite(str(suffixed), overlay)
    
    # Done for --all: exit main to avoid redundant processing below
    return


if __name__ == "__main__":
    main()
