
import os
import sys
import json
import shutil
import argparse
from pathlib import Path


"""
Run COLMAP on a SPE3R-like image/mask folder and export NeuS-compatible poses.

This script is a convenience wrapper used in the realistic-pose pipeline. It:

1. reads camera intrinsics from `camera.json`;
2. copies a selected range of images and masks into a COLMAP workspace;
3. runs the local COLMAP wrapper with fixed intrinsics and optional masks;
4. calls `imgs2poses.py` to produce `poses.npy` and `sparse_points.ply`.

The generated COLMAP model can then be pruned, aligned to the CORTO frame, and
converted into a NeuS dataset by the other scripts in this folder.
"""


def load_camera_params(spe3r_path):
    """
    Load camera parameters from camera.json.
    
    Supported formats:
    1. SPE3R-style `cameraMatrix`;
    2. flat `fx`, `fy`, `cx`, `cy` fields.
    
    Returns:
        Dictionary with fx, fy, cx, cy.
    """
    camera_json_path = Path(spe3r_path) / "camera.json"
    
    if not camera_json_path.exists():
        raise FileNotFoundError(f"camera.json not found in: {spe3r_path}")
    
    with open(camera_json_path, 'r') as f:
        camera_data = json.load(f)
    
    # Extract intrinsics.
    if 'cameraMatrix' in camera_data:
        # SPE3R format with a cameraMatrix field.
        K = camera_data['cameraMatrix']
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        
        width = camera_data.get('Nu', None)
        height = camera_data.get('Nv', None)
        
        print(f"Camera parameters from camera.json (cameraMatrix):")
        print(f"   fx: {fx}")
        print(f"   fy: {fy}")
        print(f"   cx: {cx}")
        print(f"   cy: {cy}")
        if width and height:
            print(f"   Resolution: {width}x{height}")
        
    elif 'fx' in camera_data:
        # Flat format.
        fx = camera_data['fx']
        fy = camera_data['fy']
        cx = camera_data['cx']
        cy = camera_data['cy']
        
        print(f"Camera parameters from camera.json (flat):")
        print(f"   fx: {fx}")
        print(f"   fy: {fy}")
        print(f"   cx: {cx}")
        print(f"   cy: {cy}")
        
    else:
        raise ValueError(
            "camera.json must contain 'cameraMatrix' or the fields 'fx', 'fy', 'cx', 'cy'"
        )
    
    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }


def prepare_images(spe3r_path, satellite, output_path, start_idx=1, num_images=500):
    """Copy a contiguous 1-based image range and matching masks to COLMAP input."""
    spe3r_path = Path(spe3r_path)
    output_path = Path(output_path)

    source_img_dir = spe3r_path / f"images"
    source_mask_dir = spe3r_path / f"masks"

    if not source_img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {source_img_dir}")
    if not source_mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {source_mask_dir}")

    all_image_files = sorted(
        list(source_img_dir.glob("*.png")) + list(source_img_dir.glob("*.jpg"))
    )

    if len(all_image_files) == 0:
        raise FileNotFoundError(f"No images found in: {source_img_dir}")

    print(f"Found {len(all_image_files)} total images in {source_img_dir}")

    end_idx = start_idx - 1 + num_images
    selected_files = all_image_files[start_idx - 1:end_idx]

    if len(selected_files) == 0:
        raise ValueError(
            f"No images found in range [{start_idx}:{end_idx}]. "
            f"Total available: {len(all_image_files)}"
        )

    print(f"Selected images from {start_idx} to {start_idx + len(selected_files) - 1}")
    print(f"   (total: {len(selected_files)} images)")

    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print("Copying images and masks...")
    for img_file in selected_files:
        shutil.copy2(img_file, images_dir / img_file.name)

        mask_file = source_mask_dir / img_file.name
        if not mask_file.exists():
            raise FileNotFoundError(f"Missing mask for {img_file.name}: {mask_file}")
        shutil.copy2(mask_file, masks_dir / mask_file.name)

    print(f"Images copied to: {images_dir}")
    print(f"Masks copied to: {masks_dir}")
    return len(selected_files)

def run_colmap_simple(basedir, neus_path, camera_params, match_type='exhaustive_matcher', use_gpu=False):
    """Run the local COLMAP wrapper with fixed intrinsics and mask support."""
    print("\n" + "="*70)
    print("RUNNING COLMAP")
    print("="*70)

    neus_preprocess_path = Path(neus_path) / "preprocess_custom_data" / "colmap_preprocess"
    sys.path.insert(0, str(neus_preprocess_path))

    from colmap_wrapper_with_intrinsics import run_colmap

    run_colmap(
        basedir=str(basedir),
        match_type=match_type,
        camera_params=camera_params,
        use_gpu=use_gpu,
        mask_path=str(Path(basedir) / "masks")
    )

    print("\nCOLMAP completed")


def generate_poses(basedir, neus_path, match_type):
    """Run NeuS imgs2poses.py after COLMAP has produced sparse reconstruction."""
    import subprocess

    neus_preprocess_path = Path(neus_path) / "preprocess_custom_data" / "colmap_preprocess"
    imgs2poses_script = neus_preprocess_path / "imgs2poses.py"

    if not imgs2poses_script.exists():
        raise FileNotFoundError(f"Script not found: {imgs2poses_script}")

    print("Running imgs2poses.py...")
    result = subprocess.run(
        ['python', str(imgs2poses_script), str(basedir), '--match_type', match_type],
        cwd=str(neus_preprocess_path),
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"imgs2poses.py failed with return code {result.returncode}")

    poses_file = Path(basedir) / "poses.npy"
    ply_file = Path(basedir) / "sparse_points.ply"

    if not poses_file.exists():
        raise RuntimeError(f"poses.npy was not generated in: {basedir}")
    if not ply_file.exists():
        raise RuntimeError(f"sparse_points.ply was not generated in: {basedir}")


def main():
    parser = argparse.ArgumentParser(
        description="COLMAP runner for SPE3R-like datasets"
    )
    
    parser.add_argument(
        "--spe3r-path",
        required=True,
        help="Path to the SPE3R-like satellite directory"
    )
    
    parser.add_argument(
        "--satellite",
        required=True,
        help="Satellite name used only for logging"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory"
    )
    
    parser.add_argument(
        "--neus-path",
        required=True,
        help="Path to the NeuS_thesis repository"
    )
    
    parser.add_argument(
        "--start-idx",
        type=int,
        default=1,
        help="First image index to use, 1-based"
    )
    
    parser.add_argument(
        "--num-images",
        type=int,
        default=500,
        help="Number of images to use"
    )
    
    parser.add_argument(
        "--camera-model",
        default="PINHOLE",
        choices=["PINHOLE", "SIMPLE_PINHOLE"],
        help="COLMAP camera model"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for COLMAP; CPU is usually more stable in Colab"
    )
    
    parser.add_argument(
        "--skip-colmap",
        action="store_true",
        help="Skip COLMAP and only regenerate poses from an existing reconstruction"
    )

    parser.add_argument(
    "--match-type",
    default="exhaustive_matcher",
    choices=["exhaustive_matcher", "sequential_matcher"],
    help="COLMAP matcher type"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    print("="*70)
    print("COLMAP RUNNER")
    print("="*70)
    print(f"Dataset: {args.spe3r_path}")
    print(f"Satellite: {args.satellite}")
    print(f"Output: {args.output}")
    print(f"NeuS path: {args.neus_path}")
    print(f"Image range: {args.start_idx} to {args.start_idx + args.num_images - 1}")
    print(f"GPU: {'ON' if args.use_gpu else 'OFF (CPU)'}")
    print(f"Matcher: {args.match_type}")
    print("="*70)
    
    # Step 0: load camera intrinsics.
    print("\n" + "="*70)
    print("STEP 0: LOADING CAMERA PARAMETERS")
    print("="*70)
    camera_params = load_camera_params(args.spe3r_path)
    camera_params['model'] = args.camera_model
    
    print("\nFinal camera parameters:")
    for key, val in camera_params.items():
        print(f"   {key}: {val}")
    
    # Steps 1-2: run COLMAP unless the reconstruction already exists.
    if args.skip_colmap:
        print("\n" + "="*70)
        print("SKIP: COLMAP already completed")
        print("="*70)
        print(f"Using existing results in: {output_path}")
        
        # Verify that the required COLMAP files exist.
        required_files = [
            output_path / "database.db",
            output_path / "sparse" / "0" / "cameras.bin",
            output_path / "sparse" / "0" / "images.bin",
            output_path / "sparse" / "0" / "points3D.bin",
        ]
        
        missing = [f for f in required_files if not f.exists()]
        if missing:
            print("\nERROR: Missing COLMAP files:")
            for f in missing:
                print(f"   - {f}")
            print("\nRemove --skip-colmap to run COLMAP from scratch")
            return
        
        print("All COLMAP files are present")
        num_images = len(list((output_path / "images").glob("*.jpg"))) + \
                     len(list((output_path / "images").glob("*.png")))
    else:
        # Step 1: prepare images.
        print("\n" + "="*70)
        print("STEP 1: PREPARING IMAGES")
        print("="*70)
        num_images = prepare_images(
            spe3r_path=args.spe3r_path,
            satellite=args.satellite,
            output_path=output_path,
            start_idx=args.start_idx,
            num_images=args.num_images
        )
        
        # Step 2: COLMAP
        print("\n" + "="*70)
        print("STEP 2: COLMAP (FEATURE EXTRACTION + MATCHING + SFM)")
        print("="*70)
        run_colmap_simple(
            basedir=output_path,
            neus_path=args.neus_path,
            camera_params=camera_params,
            match_type=args.match_type,
            use_gpu=args.use_gpu
        )
            
    # Step 3: generate poses.
    print("\n" + "="*70)
    print("STEP 3: GENERATING POSES AND SPARSE POINTS")
    print("="*70)
    generate_poses(
        basedir=output_path,
        neus_path=args.neus_path,
        match_type=args.match_type
    )
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETED")
    print("="*70)
    print(f"Output: {output_path}")
    print(f"Processed images: {num_images}")
    
    print("\nGenerated files:")
    files_to_check = [
        "database.db",
        "sparse/0/cameras.bin",
        "sparse/0/images.bin",
        "sparse/0/points3D.bin",
        "poses.npy",
        "sparse_points.ply",
    ]
    
    for file_path in files_to_check:
        full_path = output_path / file_path
        status = "✅" if full_path.exists() else "❌"
        print(f"   {status} {file_path}")
    
if __name__ == "__main__":
    main()
