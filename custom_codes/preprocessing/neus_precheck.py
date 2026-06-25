import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2 as cv
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


"""
Run final sanity checks on a NeuS-formatted dataset.

This script validates the folder layout, image/mask consistency,
`cameras_sphere.npz`, camera placement in normalized NeuS coordinates, optional
mesh reprojection, and compatibility with the local `models.dataset.Dataset`
loader. It is intended as a pre-flight check before starting long Colab
training runs.
"""


DEPTH_EPS = 1e-8
INV_ATOL = 1e-3
WORLD_MAT_WARN_ATOL = 5e-3


def parse_args():
    """Parse paths and reporting options for the precheck."""
    parser = argparse.ArgumentParser(
        description="Final sanity checks for a NeuS-formatted dataset."
    )
    parser.add_argument(
        "--neus_dir",
        required=True,
        help="Directory NeuS con image/, mask/ e cameras_sphere.npz",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory output per report e immagini diagnostiche",
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=4,
        help="Numero di viste campione da salvare negli overlay",
    )
    parser.add_argument(
        "--mesh",
        default=None,
        help="Path opzionale a una mesh OBJ/PLY per overlay reproiettato",
    )
    return parser.parse_args()


def list_pngs(folder):
    """Return sorted PNG filenames from a directory."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".png") and os.path.isfile(os.path.join(folder, f))
    )


def load_npz(npz_path):
    """Load a .npz file into a normal dictionary."""
    return dict(np.load(npz_path))


def count_cameras(cam_dict):
    """Count sequential world_mat_i entries in cameras_sphere.npz."""
    i = 0
    while f"world_mat_{i}" in cam_dict:
        i += 1
    return i


def load_k_rt_from_p(P):
    """Decompose a projection matrix using the same convention as NeuS/IDR."""
    P = np.asarray(P, dtype=np.float64)
    K, R, t, _, _, _, _ = cv.decomposeProjectionMatrix(P)
    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.T
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return K, pose


def project_points(P, points_3d):
    """Project 3D homogeneous points and return pixels plus raw depths."""
    points_3d = np.asarray(points_3d, dtype=np.float64)
    points_h = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
    proj = (P @ points_h.T).T

    depth = proj[:, 2]
    pixels = np.full((len(points_3d), 2), np.nan, dtype=np.float64)
    valid = np.abs(depth) > DEPTH_EPS
    pixels[valid, 0] = proj[valid, 0] / depth[valid]
    pixels[valid, 1] = proj[valid, 1] / depth[valid]
    return pixels, depth


def ensure_out_dir(out_dir):
    """Create the report directory."""
    os.makedirs(out_dir, exist_ok=True)


def print_section(title):
    """Print a readable section separator."""
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def load_mesh_vertices(mesh_path, max_points=5000):
    """Load OBJ/PLY vertices and optionally subsample them for plotting."""
    ext = Path(mesh_path).suffix.lower()

    if ext == ".obj":
        vertices = []
        with open(mesh_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        vertices = np.asarray(vertices, dtype=np.float64)
    elif ext == ".ply":
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("Per file PLY serve trimesh") from exc
        geom = trimesh.load(mesh_path)

        if isinstance(geom, trimesh.Scene):
            collected = []
            for item in geom.geometry.values():
                if hasattr(item, "vertices"):
                    verts = np.asarray(item.vertices, dtype=np.float64)
                    if verts.size > 0:
                        collected.append(verts)
            if not collected:
                raise ValueError(f"Nessun vertice trovato in {mesh_path}")
            vertices = np.concatenate(collected, axis=0)
        elif hasattr(geom, "vertices"):
            vertices = np.asarray(geom.vertices, dtype=np.float64)
        else:
            raise ValueError(
                f"PLY non supportato o senza vertici leggibili: {mesh_path}"
            )
    else:
        raise ValueError(f"Formato mesh non supportato: {ext}")

    if vertices.size == 0:
        raise ValueError(f"Nessun vertice trovato in {mesh_path}")

    if len(vertices) > max_points:
        idx = np.random.choice(len(vertices), max_points, replace=False)
        vertices = vertices[idx]

    return vertices


def check_required_layout(neus_dir):
    """Check that image/, mask/, and cameras_sphere.npz exist."""
    print_section("CHECK 1: Struttura cartelle")

    image_dir = os.path.join(neus_dir, "image")
    mask_dir = os.path.join(neus_dir, "mask")
    npz_path = os.path.join(neus_dir, "cameras_sphere.npz")

    ok = True

    if not os.path.isdir(image_dir):
        print(f"[FAIL] Cartella mancante: {image_dir}")
        ok = False
    else:
        print(f"[OK] image/: {image_dir}")

    if not os.path.isdir(mask_dir):
        print(f"[FAIL] Cartella mancante: {mask_dir}")
        ok = False
    else:
        print(f"[OK] mask/:  {mask_dir}")

    if not os.path.isfile(npz_path):
        print(f"[FAIL] File mancante: {npz_path}")
        ok = False
    else:
        print(f"[OK] NPZ:    {npz_path}")

    return ok, image_dir, mask_dir, npz_path


def check_images_and_masks(image_dir, mask_dir, out_dir, n_views):
    """Validate image/mask naming, shapes, readability, and mask occupancy."""
    print_section("CHECK 2: File immagini e maschere")

    image_files = list_pngs(image_dir)
    mask_files = list_pngs(mask_dir)

    ok = True

    if not image_files:
        print("[FAIL] Nessuna immagine trovata")
        return False, image_files, mask_files, None

    if not mask_files:
        print("[FAIL] Nessuna maschera trovata")
        return False, image_files, mask_files, None

    print(f"Immagini trovate: {len(image_files)}")
    print(f"Maschere trovate: {len(mask_files)}")

    if len(image_files) != len(mask_files):
        print("[FAIL] Numero immagini e maschere diverso")
        ok = False

    if image_files != mask_files:
        missing_in_mask = sorted(set(image_files) - set(mask_files))
        missing_in_img = sorted(set(mask_files) - set(image_files))
        if missing_in_mask:
            print(f"[FAIL] File immagine senza maschera: {missing_in_mask[:5]}")
        if missing_in_img:
            print(f"[FAIL] File maschera senza immagine: {missing_in_img[:5]}")
        ok = False

    shapes = []
    mask_ratios = []
    sample_count = min(len(image_files), len(mask_files))

    for i in range(sample_count):
        img_path = os.path.join(image_dir, image_files[i])
        mask_path = os.path.join(mask_dir, mask_files[i])

        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)

        if img is None:
            print(f"[FAIL] Immagine non leggibile: {img_path}")
            ok = False
            continue
        if mask is None:
            print(f"[FAIL] Maschera non leggibile: {mask_path}")
            ok = False
            continue

        if img.ndim != 3 or img.shape[2] < 3:
            print(f"[FAIL] Immagine non RGB-like: {img_path} shape={img.shape}")
            ok = False

        mask_gray = mask if mask.ndim == 2 else mask[..., 0]
        if mask_gray.shape[:2] != img.shape[:2]:
            print(
                f"[FAIL] Shape diversa img/mask per {image_files[i]}: "
                f"{img.shape[:2]} vs {mask_gray.shape[:2]}"
            )
            ok = False

        shapes.append(img.shape[:2])
        mask_ratios.append(float(mask_gray.mean()) / 255.0)

    if shapes:
        unique_shapes = sorted(set(shapes))
        print(f"Shape immagini trovate: {unique_shapes}")
        if len(unique_shapes) != 1:
            print("[FAIL] Non tutte le immagini hanno la stessa risoluzione")
            ok = False

    if mask_ratios:
        mask_ratios = np.asarray(mask_ratios, dtype=np.float64)
        print(
            f"Occupazione mask min/mean/max: "
            f"{mask_ratios.min():.4f} / {mask_ratios.mean():.4f} / {mask_ratios.max():.4f}"
        )
        fully_black = int(np.sum(mask_ratios < 0.01))
        fully_white = int(np.sum(mask_ratios > 0.99))
        if fully_black > 0:
            print(f"[FAIL] Maschere quasi nere: {fully_black}")
            ok = False
        if fully_white == len(mask_ratios):
            print("[WARN] Tutte le maschere sono quasi bianche")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(mask_ratios, linewidth=1.0)
        ax.set_title("Mask occupancy ratio")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Foreground ratio")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "mask_ratio_plot.png"), dpi=160)
        plt.close(fig)

    view_indices = np.linspace(0, sample_count - 1, min(n_views, sample_count), dtype=int)
    if len(view_indices) > 0:
        fig, axes = plt.subplots(1, len(view_indices), figsize=(5 * len(view_indices), 5))
        if len(view_indices) == 1:
            axes = [axes]

        for col, idx in enumerate(view_indices):
            img = cv.imread(os.path.join(image_dir, image_files[idx]), cv.IMREAD_COLOR)
            mask = cv.imread(os.path.join(mask_dir, mask_files[idx]), cv.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            overlay = img.copy()
            overlay[mask > 127] = (
                0.5 * overlay[mask > 127] + 0.5 * np.array([0, 255, 0])
            ).astype(np.uint8)
            axes[col].imshow(overlay)
            axes[col].set_title(image_files[idx])
            axes[col].axis("off")

        fig.suptitle("Overlay maschere su immagini")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "mask_overlay_samples.png"), dpi=160)
        plt.close(fig)

    return ok, image_files, mask_files, shapes[0] if shapes else None


def check_npz(cam_dict, expected_count):
    """Validate expected NeuS camera keys, matrix shapes, and inverses."""
    print_section("CHECK 3: Integrita cameras_sphere.npz")

    ok = True
    has_only_world_mat_warn = True
    n_cams = count_cameras(cam_dict)
    print(f"Camere nel NPZ: {n_cams}")
    print(f"File immagine:   {expected_count}")

    if n_cams != expected_count:
        print("[FAIL] Numero camere nel NPZ diverso dal numero di immagini")
        ok = False

    required_prefixes = [
        "camera_mat",
        "camera_mat_inv",
        "world_mat",
        "world_mat_inv",
        "scale_mat",
        "scale_mat_inv",
    ]

    for i in range(n_cams):
        for prefix in required_prefixes:
            key = f"{prefix}_{i}"
            if key not in cam_dict:
                print(f"[FAIL] Chiave mancante: {key}")
                ok = False
                continue
            mat = np.asarray(cam_dict[key])
            if mat.shape != (4, 4):
                print(f"[FAIL] Shape errata per {key}: {mat.shape}")
                ok = False
            if not np.all(np.isfinite(mat)):
                print(f"[FAIL] NaN/Inf in {key}")
                ok = False

    if n_cams > 0 and "scale_mat_0" in cam_dict:
        scale0 = np.asarray(cam_dict["scale_mat_0"], dtype=np.float64)
        same_scale = all(
            np.allclose(np.asarray(cam_dict[f"scale_mat_{i}"], dtype=np.float64), scale0)
            for i in range(n_cams)
        )
        print(f"scale_mat unico per tutte le viste: {same_scale}")
        if not same_scale:
            ok = False

    for prefix, inv_prefix in [
        ("camera_mat", "camera_mat_inv"),
        ("world_mat", "world_mat_inv"),
        ("scale_mat", "scale_mat_inv"),
    ]:
        for i in range(n_cams):
            a = np.asarray(cam_dict[f"{prefix}_{i}"], dtype=np.float64)
            b = np.asarray(cam_dict[f"{inv_prefix}_{i}"], dtype=np.float64)
            eye = a @ b
            residual = float(np.max(np.abs(eye - np.eye(4))))
            cond = float(np.linalg.cond(a))
            if not np.allclose(eye, np.eye(4), atol=INV_ATOL):
                if prefix == "world_mat" and residual <= WORLD_MAT_WARN_ATOL:
                    print(
                        f"[WARN] Inversa approssimata per {prefix}_{i} "
                        f"(max_residual={residual:.3e}, cond={cond:.3e})"
                    )
                else:
                    print(
                        f"[FAIL] Inversa incoerente per {prefix}_{i} "
                        f"(max_residual={residual:.3e}, cond={cond:.3e})"
                    )
                    ok = False
                    has_only_world_mat_warn = False
                break

    if ok:
        if has_only_world_mat_warn:
            print("[OK] Struttura NPZ coerente")

    return ok, n_cams


def check_camera_geometry(cam_dict, image_shape, out_dir):
    """Inspect camera positions and projection consistency in normalized space."""
    print_section("CHECK 4: Geometria camere e proiezioni")

    ok = True
    n_cams = count_cameras(cam_dict)
    H, W = image_shape

    positions = []
    distances = []
    center_errors = []
    valid_depth_count = 0
    near_zero_depth_count = 0

    target_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.0, 0.15, 0.0],
            [0.0, 0.0, 0.15],
        ],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i in range(n_cams):
        world_mat = np.asarray(cam_dict[f"world_mat_{i}"], dtype=np.float64)
        scale_mat = np.asarray(cam_dict[f"scale_mat_{i}"], dtype=np.float64)
        camera_mat = np.asarray(cam_dict[f"camera_mat_{i}"], dtype=np.float64)

        P = world_mat @ scale_mat
        # Decompose the full world_mat @ scale_mat projection, as done by the
        # NeuS dataset loader. This checks the exact matrices that training will
        # consume, not only the raw intrinsic matrix.
        K_from_p, pose = load_k_rt_from_p(P[:3, :4])
        cam_pos = pose[:3, 3]
        positions.append(cam_pos)

        dist = float(np.linalg.norm(cam_pos))
        distances.append(dist)

        if dist <= 1.0:
            print(f"[FAIL] Camera {i} dentro o sul bordo della unit sphere: dist={dist:.4f}")
            ok = False

        if not np.allclose(K_from_p, camera_mat[:3, :3], atol=1e-3):
            print(f"[FAIL] camera_mat_{i} non coerente con world_mat_{i} @ scale_mat_{i}")
            ok = False

        pixels, depths = project_points(P[:3, :4], target_points)
        px0, py0 = pixels[0]
        z0 = depths[0]

        if np.isfinite(z0) and z0 > DEPTH_EPS and np.all(np.isfinite([px0, py0])):
            valid_depth_count += 1
        else:
            if np.isfinite(z0) and abs(z0) <= DEPTH_EPS:
                near_zero_depth_count += 1
                print(
                    f"[WARN] Origine normalizzata quasi sul piano camera {i}: "
                    f"depth={z0:.3e}"
                )
            else:
                print(f"[WARN] Origine normalizzata dietro la camera {i}: depth={z0:.3e}")
            continue

        cx = camera_mat[0, 2]
        cy = camera_mat[1, 2]
        center_err = float(np.linalg.norm([px0 - cx, py0 - cy]))
        center_errors.append(center_err)

        if center_err > max(W, H) * 0.1:
            print(
                f"[WARN] Camera {i}: origine proiettata lontana dal principal point "
                f"({center_err:.2f}px)"
            )

    positions = np.asarray(positions, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    center_errors = np.asarray(center_errors, dtype=np.float64) if center_errors else np.array([])

    print(
        f"Distanza camere min/mean/max: "
        f"{distances.min():.4f} / {distances.mean():.4f} / {distances.max():.4f}"
    )
    if center_errors.size > 0:
        print(
            f"Errore origine vs principal point min/mean/max: "
            f"{center_errors.min():.2f} / {center_errors.mean():.2f} / {center_errors.max():.2f} px"
        )
    print(f"Origine validamente proiettabile: {valid_depth_count}/{n_cams}")
    print(f"Origine quasi sul piano camera:   {near_zero_depth_count}/{n_cams}")
    print("Nota: il check sull'origine e' informativo se scale_mat usa il centro bbox.")

    axes[0].plot(distances, linewidth=1.0)
    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.6)
    axes[0].set_title("Distanza camere dall'origine")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Distance")
    axes[0].grid(True, alpha=0.3)

    if center_errors.size > 0:
        axes[1].plot(center_errors, linewidth=1.0, color="darkorange")
        axes[1].set_title("Errore reproiezione origine")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Pixel error")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Nessun dato", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "camera_metrics.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=18, c=distances, cmap="viridis")
    ax.scatter(0, 0, 0, c="red", s=100, marker="x")
    ax.set_title("Camera positions in normalized space")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    lim = max(1.5, float(distances.max()) * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "camera_positions_3d.png"), dpi=160)
    plt.close(fig)

    return ok


def check_mesh_reprojection(cam_dict, neus_dir, mesh_path, out_dir, n_views):
    """Project an optional GT/reconstruction mesh onto sample images."""
    print_section("CHECK 5: Mesh reprojection")

    if mesh_path is None:
        print("[SKIP] Mesh non fornita")
        return True

    if not os.path.isfile(mesh_path):
        print(f"[FAIL] Mesh non trovata: {mesh_path}")
        return False

    try:
        vertices_world = load_mesh_vertices(mesh_path, max_points=3000)
    except Exception as exc:
        print(f"[FAIL] Impossibile caricare la mesh: {exc}")
        return False

    scale_mat = np.asarray(cam_dict["scale_mat_0"], dtype=np.float64)
    scale_inv = np.linalg.inv(scale_mat)
    # The optional mesh is assumed to be in the original world/CORTO frame.
    # NeuS projections operate in normalized space, so map vertices with
    # scale_mat^{-1} before applying world_mat @ scale_mat.
    verts_h = np.concatenate([vertices_world, np.ones((len(vertices_world), 1))], axis=1)
    verts_norm = (scale_inv @ verts_h.T).T[:, :3]

    radii = np.linalg.norm(verts_norm, axis=1)
    print(
        f"Raggio mesh norm min/mean/max: "
        f"{radii.min():.4f} / {radii.mean():.4f} / {radii.max():.4f}"
    )
    if radii.max() > 1.05:
        print("[WARN] La mesh normalizzata esce dalla unit sphere")

    image_dir = os.path.join(neus_dir, "image")
    image_files = list_pngs(image_dir)
    if not image_files:
        print("[FAIL] Nessuna immagine disponibile")
        return False

    view_indices = np.linspace(0, len(image_files) - 1, min(n_views, len(image_files)), dtype=int)
    fig, axes = plt.subplots(1, len(view_indices), figsize=(5 * len(view_indices), 5))
    if len(view_indices) == 1:
        axes = [axes]

    for col, idx in enumerate(view_indices):
        img = cv.imread(os.path.join(image_dir, image_files[idx]), cv.IMREAD_COLOR)
        if img is None:
            continue

        P = (
            np.asarray(cam_dict[f"world_mat_{idx}"], dtype=np.float64)
            @ np.asarray(cam_dict[f"scale_mat_{idx}"], dtype=np.float64)
        )[:3, :4]

        pixels, depths = project_points(P, verts_norm)
        valid = (
            (depths > DEPTH_EPS)
            & np.isfinite(pixels[:, 0])
            & np.isfinite(pixels[:, 1])
        )

        H, W = img.shape[:2]
        in_frame = (
            valid
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < W)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < H)
        )

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        axes[col].imshow(img_rgb)
        axes[col].scatter(
            pixels[in_frame, 0],
            pixels[in_frame, 1],
            s=0.3,
            c="lime",
            alpha=0.5,
        )
        axes[col].set_title(f"{image_files[idx]} ({int(in_frame.sum())} pts)")
        axes[col].axis("off")

    fig.suptitle("Mesh reprojection overlay")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mesh_overlay_samples.png"), dpi=160)
    plt.close(fig)

    print(f"[OK] Overlay salvato: {os.path.join(out_dir, 'mesh_overlay_samples.png')}")
    return True


def check_dataset_loader_compatibility(neus_dir):
    """Check the subset of keys and file counts required by models.dataset."""
    print_section("CHECK 6: Compatibilita con Dataset di NeuS")

    ok = True
    image_dir = os.path.join(neus_dir, "image")
    mask_dir = os.path.join(neus_dir, "mask")
    npz_path = os.path.join(neus_dir, "cameras_sphere.npz")

    image_files = list_pngs(image_dir)
    mask_files = list_pngs(mask_dir)
    cam_dict = dict(np.load(npz_path))
    n_cams = count_cameras(cam_dict)

    if len(image_files) != n_cams:
        print("[FAIL] `Dataset` carichera' un numero di immagini diverso dalle camere")
        ok = False

    if len(mask_files) != n_cams:
        print("[FAIL] `Dataset` carichera' un numero di maschere diverso dalle camere")
        ok = False

    for idx in range(n_cams):
        for prefix in ["world_mat", "scale_mat"]:
            if f"{prefix}_{idx}" not in cam_dict:
                print(f"[FAIL] Chiave attesa da Dataset mancante: {prefix}_{idx}")
                ok = False

    if ok:
        print("[OK] Il dataset e' compatibile con il loader attuale di NeuS")

    return ok


def write_summary(out_dir, results):
    """Write a compact pass/fail text report."""
    summary_path = os.path.join(out_dir, "summary.txt")
    all_ok = all(results.values())

    lines = []
    lines.append("NeuS dataset precheck\n")
    for name, passed in results.items():
        lines.append(f"[{'OK' if passed else 'FAIL'}] {name}\n")
    lines.append("\n")
    lines.append("FINAL RESULT: PASS\n" if all_ok else "FINAL RESULT: FAIL\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print_section("SOMMARIO")
    for name, passed in results.items():
        print(f"[{'OK' if passed else 'FAIL'}] {name}")
    print(f"\nRisultato finale: {'PASS' if all_ok else 'FAIL'}")
    print(f"Report salvato in: {summary_path}")

    return all_ok


def main():
    """Run all precheck stages and return a failing exit code on errors."""
    args = parse_args()
    neus_dir = os.path.abspath(args.neus_dir)
    out_dir = os.path.abspath(args.out or os.path.join(neus_dir, "precheck_report"))
    ensure_out_dir(out_dir)

    print("NeuS final precheck")
    print(f"Dataset: {neus_dir}")
    print(f"Output:  {out_dir}")
    print(f"Mesh:    {args.mesh}")

    results = {}

    ok_layout, image_dir, mask_dir, npz_path = check_required_layout(neus_dir)
    results["layout"] = ok_layout

    if not ok_layout:
        write_summary(out_dir, results)
        raise SystemExit(1)

    ok_files, image_files, mask_files, image_shape = check_images_and_masks(
        image_dir, mask_dir, out_dir, args.n_views
    )
    results["images_and_masks"] = ok_files

    cam_dict = load_npz(npz_path)
    ok_npz, _ = check_npz(cam_dict, len(image_files))
    results["npz"] = ok_npz

    if image_shape is not None:
        results["camera_geometry"] = check_camera_geometry(cam_dict, image_shape, out_dir)
    else:
        print("[FAIL] Geometria camere non verificata: shape immagini non disponibile")
        results["camera_geometry"] = False

    results["mesh_reprojection"] = check_mesh_reprojection(
        cam_dict, neus_dir, args.mesh, out_dir, args.n_views
    )

    results["dataset_loader_compatibility"] = check_dataset_loader_compatibility(neus_dir)

    passed = write_summary(out_dir, results)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
