#!/usr/bin/env python3
"""
Create NeuS view subsets (1 / 4 / 8 views) from the full HST dataset.

Logica:
- Si scelgono 8 indici random (senza rimpiazzamento) tra tutte le immagini disponibili.
- Da questi 8:
    - subset 1-view usa 1 indice (il primo della lista ordinata)
    - subset 4-views usa i primi 4 indici
    - subset 8-views usa tutti e 8 gli indici

Quindi: 1 ⊂ 4 ⊂ 8, tutti scelti a partire dallo stesso set random di 8 viste.
"""

import os
import shutil
import numpy as np
from pathlib import Path

# === CONFIG ===

# Root dataset dir (quella con image/, mask/, cameras_spe3r.npz, model/)
BASE_DIR = Path("/Users/martino/Desktop/Tesi/codes/myCodes/hst_neus")

# Seed per rendere le scelte riproducibili
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Usa symlink invece di copiare i file (consigliato)
USE_SYMLINKS = True


def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if USE_SYMLINKS:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def filter_cameras_npz(src_npz: Path, dst_npz: Path, indices: list[int]):
    """
    NUOVA VERSIONE, ADATTATA A cameras_spe3r.npz DEL TUO HST.

    In cameras_spe3r.npz hai chiavi del tipo:
        world_mat_0, scale_mat_0, world_mat_1, scale_mat_1, ...

    Qui:
    - prendo solo le (world_mat_i, scale_mat_i) per gli indici richiesti
    - le rinomino in modo compatto: 0..(N-1)

      Esempio:
        se indices = [411, 513, 521],
        allora nel nuovo npz avrai:
            world_mat_0 ← world_mat_411
            scale_mat_0 ← scale_mat_411
            world_mat_1 ← world_mat_513
            scale_mat_1 ← scale_mat_513
            world_mat_2 ← world_mat_521
            scale_mat_2 ← scale_mat_521
    """
    print(f"  Filtering cameras_spe3r.npz → {dst_npz.name}")
    data = np.load(src_npz)
    out = {}

    for new_i, old_i in enumerate(indices):
        wm_key_old = f"world_mat_{old_i}"
        sm_key_old = f"scale_mat_{old_i}"

        if wm_key_old not in data or sm_key_old not in data:
            raise KeyError(
                f"Manca {wm_key_old} o {sm_key_old} in {src_npz}. "
                f"Indice richiesto: {old_i}"
            )

        wm = data[wm_key_old]
        sm = data[sm_key_old]

        wm_key_new = f"world_mat_{new_i}"
        sm_key_new = f"scale_mat_{new_i}"

        out[wm_key_new] = wm
        out[sm_key_new] = sm

        print(f"    keeping view {old_i} → new index {new_i}")

    np.savez(dst_npz, **out)


def create_subset_dirs(num_views: int, indices: list[int]):
    subset_name = f"{BASE_DIR.name}_{num_views}views"
    out_root = BASE_DIR.parent / subset_name

    print(f"\n=== Creating subset: {subset_name} with indices {indices} ===")

    img_out = out_root / "image"
    mask_out = out_root / "mask"
    model_out = out_root / "model"

    make_dir(img_out)
    make_dir(mask_out)
    make_dir(model_out)

    # Symlink modello GT
    model_src = BASE_DIR / "model" / "model_normalized.obj"
    link_or_copy(model_src, model_out / "model_normalized.obj")

    # Symlink immagini e maschere selezionate
    for idx in indices:
        fname = f"{idx:03d}.png"
        img_src = BASE_DIR / "image" / fname
        mask_src = BASE_DIR / "mask" / fname

        if not img_src.exists():
            raise FileNotFoundError(f"Image file not found: {img_src}")
        if not mask_src.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_src}")

        link_or_copy(img_src, img_out / fname)
        link_or_copy(mask_src, mask_out / fname)

    # Filtra cameras_spe3r.npz
    cams_src = BASE_DIR / "cameras_spe3r.npz"
    cams_dst = out_root / "cameras_spe3r.npz"
    filter_cameras_npz(cams_src, cams_dst, indices)

    print(f"Subset written to: {out_root}")


def main():
    # Conta le immagini disponibili
    images = sorted((BASE_DIR / "image").glob("*.png"))
    total_imgs = len(images)
    print(f"Found {total_imgs} total images in {BASE_DIR / 'image'}")

    if total_imgs < 8:
        raise ValueError("Not enough images to select 8 views")

    # Scegli 8 indici random (0 .. total_imgs-1), poi ordinali
    rand_idx_8 = np.random.choice(total_imgs, size=8, replace=False)
    rand_idx_8 = sorted(map(int, rand_idx_8))

    # Crea subset annidati
    indices_1 = [rand_idx_8[0]]        # 1-view: primo indice
    indices_4 = rand_idx_8[:4]         # 4-views: primi 4
    indices_8 = rand_idx_8             # 8-views: tutti e 8

    print(f"\nRandom 8-view set (seed={RANDOM_SEED}): {indices_8}")
    print(f"1-view subset (from those 8): {indices_1}")
    print(f"4-view subset (from those 8): {indices_4}")

    # Crea le tre cartelle subset
    create_subset_dirs(1, indices_1)
    create_subset_dirs(4, indices_4)
    create_subset_dirs(8, indices_8)


if __name__ == "__main__":
    main()
