"""
gen_cameras_with_masks.py
=========================
Minimal adaptation of the original NeuS gen_cameras.py script.

The camera-matrix generation logic is intentionally kept unchanged. The custom
part is the image/mask copy step: instead of copying every input image, the
script reads sparse_txt/images.txt and keeps only the views that COLMAP actually
registered, in the same sorted-by-name order used by pose_utils.py for poses.npy.
Masks are copied with the same stem as their corresponding image.

Usage:
    python gen_cameras_with_masks.py <images_dir> <masks_dir> [out_dir] [colmap_dir]

    images_dir : directory containing original images
    masks_dir  : directory containing masks with matching image stems
    out_dir    : optional output directory. Default: <colmap_dir>/preprocessed
    colmap_dir : optional directory with poses.npy, sparse_points_interest.ply,
                 and sparse_txt/. Default: images_dir

Colab examples:
    # Everything in the same folder.
    python gen_cameras_with_masks.py /content/my_data /content/my_data/masks

    # Images, masks, and COLMAP outputs in separate folders.
    python gen_cameras_with_masks.py \
        /content/my_data/images \
        /content/my_data/masks \
        /content/NeuS_thesis/public_data/my_object \
        /content/my_data/colmap_output
"""

import numpy as np
import trimesh
import cv2 as cv
import os
import argparse
from pathlib import Path


def read_images_txt_names(path):
    """
    Read sparse_txt/images.txt and return registered image names sorted by name.

    This matches the np.argsort ordering used by pose_utils.py when writing
    poses.npy, so copied images/masks stay aligned with camera matrices.
    """
    names = []
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip() != '']
    for i in range(0, len(lines), 2):   # paired lines: pose metadata + 2D points
        parts = lines[i].strip().split()
        names.append(parts[9])          # NAME field
    return sorted(names)                # equivalent to the np.argsort used by pose_utils.py


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate NeuS cameras and copy only COLMAP-registered images/masks."
        )
    )
    parser.add_argument(
        "images_dir",
        help="Directory containing the original images",
    )
    parser.add_argument(
        "masks_dir",
        help="Directory containing masks with the same names/stems as the images",
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=None,
        help="Output directory. Default: <colmap_dir>/preprocessed",
    )
    parser.add_argument(
        "colmap_dir",
        nargs="?",
        default=None,
        help="Directory with poses.npy, sparse_points_interest.ply and sparse_txt/. Default: images_dir",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    images_dir = args.images_dir
    masks_dir = args.masks_dir
    out_dir = args.out_dir
    colmap_dir = args.colmap_dir if args.colmap_dir is not None else images_dir

    # Default output directory: <colmap_dir>/preprocessed.
    if out_dir is None:
        out_dir = os.path.join(colmap_dir, 'preprocessed')

    print(f'images_dir : {images_dir}')
    print(f'masks_dir  : {masks_dir}')
    print(f'colmap_dir : {colmap_dir}')
    print(f'out_dir    : {out_dir}')

    # Camera conversion block kept identical to the original NeuS gen_cameras.py.
    poses_hwf = np.load(os.path.join(colmap_dir, 'poses.npy'))  # (N, 3, 5)
    poses_raw = poses_hwf[:, :, :4]
    hwf       = poses_hwf[:, :, 4]

    cam_dict = dict()
    n_images = len(poses_raw)

    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] =  1.0
    convert_mat[1, 0] =  1.0
    convert_mat[2, 2] = -1.0
    convert_mat[3, 3] =  1.0

    for i in range(n_images):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :4] = poses_raw[i]
        pose = pose @ convert_mat
        h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
        intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
        intrinsic[0, 2] = (w - 1) * 0.5
        intrinsic[1, 2] = (h - 1) * 0.5
        w2c       = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)]     = intrinsic
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)]      = world_mat
        cam_dict['world_mat_inv_{}'.format(i)]  = np.linalg.inv(world_mat)

    pcd      = trimesh.load(os.path.join(colmap_dir, 'sparse_points_interest.ply'))
    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center   = (bbox_max + bbox_min) * 0.5
    radius   = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center

    for i in range(n_images):
        cam_dict['scale_mat_{}'.format(i)]     = scale_mat
        cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mask'),  exist_ok=True)
    # End of the original camera conversion block.

    # Custom part: copy registered images from images.txt instead of globbing all inputs.
    images_txt   = os.path.join(colmap_dir, 'sparse_txt', 'images.txt')
    colmap_names = read_images_txt_names(images_txt)

    assert len(colmap_names) == n_images, \
        f"ERROR: {len(colmap_names)} names in images.txt vs {n_images} poses in poses.npy!"

    print(f'Example: image [{colmap_names[0]}] -> expected mask [{Path(colmap_names[0]).stem + ".png"}]')

    missing_masks = []
    for i, fname in enumerate(colmap_names):
        # Image.
        img = cv.imread(os.path.join(images_dir, fname))
        cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), img)

        # Mask with the same image stem, forced to .png.
        mask_fname = Path(fname).stem + '.png'
        src_mask = os.path.join(masks_dir, mask_fname)
        if os.path.exists(src_mask):
            mask = cv.imread(src_mask)
            cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), mask)
        else:
            missing_masks.append((i, fname))

    # Missing masks are replaced by white masks, equivalent to a no-mask run.
    if missing_masks:
        sample = cv.imread(os.path.join(out_dir, 'image', '000.png'))
        white  = np.ones_like(sample) * 255
        for i, fname in missing_masks:
            cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), white)
            print(f'[WARN] missing mask for {fname} -> generated white mask')
    else:
        print(f'All {n_images} masks found and copied.')

    np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
    print('Process done!')
