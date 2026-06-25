import os
import subprocess


"""
Run COLMAP with fixed camera intrinsics and optional masks.

This wrapper is used by the realistic-pose preprocessing pipeline. Compared with
the original NeuS COLMAP wrapper, it can force a known camera model/intrinsic
matrix, pass foreground masks to COLMAP, and use parameter values that were
stable for the thesis datasets in a Colab/headless environment.
"""


def run_colmap(basedir, match_type, camera_params=None, use_gpu=False, mask_path=None):
    """
    Run COLMAP with validated parameters for the thesis datasets.
    
    Args:
        basedir: Directory containing the images/ subfolder.
        match_type: COLMAP matcher, usually 'exhaustive_matcher'.
        camera_params: Optional dictionary with known intrinsics:
            {
                'model': 'PINHOLE',
                'fx': 1277.37,
                'fy': 1277.37,
                'cx': 128.0,
                'cy': 128.0
            }
        use_gpu: If True, enable GPU SIFT; CPU is usually more stable in Colab.
        mask_path: Optional directory with masks matching image filenames.
    """
    
    # Configure COLMAP/Qt for headless environments such as Google Colab.
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['DISPLAY'] = ''
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-root'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
    os.makedirs('/tmp/runtime-root', exist_ok=True)
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    print("\n" + "="*70)
    print("COLMAP - VALIDATED PARAMETERS")
    print("="*70)
    
    # ========================================================================
    # Feature extraction with fixed, reproducible parameters.
    # ========================================================================
    feature_extractor_args = [
        'colmap', 'feature_extractor', 
        '--database_path', os.path.join(basedir, 'database.db'), 
        '--image_path', os.path.join(basedir, 'images'),
        '--ImageReader.single_camera', '1',
    ]

    # Use masks when available.
    if mask_path is None:
        mask_path = os.path.join(basedir, 'masks')

    if os.path.exists(mask_path):
        feature_extractor_args.extend([
            '--ImageReader.mask_path', mask_path
        ])
        print(f"Masks enabled: {mask_path}")
    else:
        print("No masks found")
    
    # Force known intrinsics when provided.
    if camera_params is not None:
        model = camera_params.get('model', 'PINHOLE')
        feature_extractor_args.extend([
            '--ImageReader.camera_model', model,
        ])
        
        if model == 'PINHOLE':
            params_str = '{},{},{},{}'.format(
                camera_params['fx'],
                camera_params['fy'],
                camera_params['cx'],
                camera_params['cy']
            )
        elif model == 'SIMPLE_PINHOLE':
            params_str = '{},{},{}'.format(
                camera_params.get('f', camera_params.get('fx')),
                camera_params['cx'],
                camera_params['cy']
            )
        else:
            raise ValueError(f"Unsupported camera model: {model}")
        
        feature_extractor_args.extend([
            '--ImageReader.camera_params', params_str,
        ])
        
        print("Camera intrinsics fixed:")
        print(f"   Model: {model}")
        print(f"   Parameters: {params_str}")
    else:
        feature_extractor_args.extend([
            '--ImageReader.camera_model', 'SIMPLE_RADIAL',
        ])
        print("Camera intrinsics not fixed; COLMAP will estimate them")
    
    gpu_flag = '1' if use_gpu else '0'
    feature_extractor_args.extend([
        '--SiftExtraction.use_gpu', gpu_flag,
        '--SiftExtraction.num_threads', '2',
        '--SiftExtraction.max_num_features', '8192',
        '--SiftExtraction.peak_threshold', '0.005',  
        '--SiftExtraction.edge_threshold', '10',      
        '--SiftExtraction.max_image_size', '1024',
    ])

    #'--SiftExtraction.max_num_features', '20000',
    #'--SiftExtraction.peak_threshold', '0.004',
    #'--SiftExtraction.edge_threshold', '20',
    
    print("\n[1/3] Feature Extraction...")
    print(f"   - GPU: {'ON' if use_gpu else 'OFF (CPU)'}")
    print("   - peak_threshold: 0.005")
    print("   - edge_threshold: 10")
    print("   - max_features: 8192")
    
    feat_output = subprocess.check_output(
        feature_extractor_args, 
        universal_newlines=True
    )
    logfile.write(feat_output)
    print('Features extracted')

    # ========================================================================
    # Feature matching with guided matching.
    # ========================================================================
    matcher_args = [
        'colmap', match_type, 
        '--database_path', os.path.join(basedir, 'database.db'),
        '--SiftMatching.use_gpu', gpu_flag,
        '--SiftMatching.guided_matching', '1',
        '--SiftMatching.max_num_matches', '50000',
        '--SiftMatching.max_ratio', '0.75', # stricter than default 0.8 to reduce outliers
        '--SiftMatching.max_error', '3', # stricter than default 4 to reduce outliers
    ]
    
    print("\n[2/3] Feature Matching...")
    print(f"   - GPU: {'ON' if use_gpu else 'OFF (CPU)'}")
    print(f"   - Matcher: {match_type}")
    print("   - guided_matching: ON")
    
    match_output = subprocess.check_output(
        matcher_args, 
        universal_newlines=True
    )
    logfile.write(match_output)
    print('Features matched')
    
    # ========================================================================
    # Sparse reconstruction.
    # ========================================================================
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    mapper_args = [
        'colmap', 'mapper',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--output_path', os.path.join(basedir, 'sparse'),
        '--Mapper.num_threads', '16',
        '--Mapper.multiple_models', '0',
        '--Mapper.extract_colors', '0',

        '--Mapper.init_min_num_inliers', '50',
        '--Mapper.abs_pose_min_num_inliers', '25',
        '--Mapper.abs_pose_min_inlier_ratio', '0.08',
        '--Mapper.min_num_matches', '25',
        '--Mapper.abs_pose_max_error', '8',
        '--Mapper.filter_max_reproj_error', '2',
        # More permissive values can register more images but also produce more outliers.
        #'--Mapper.init_min_num_inliers', '30',
        #'--Mapper.abs_pose_min_num_inliers', '15',
        #'--Mapper.abs_pose_min_inlier_ratio', '0.05',
        #'--Mapper.min_num_matches', '15',
    ]
    
    # If intrinsics are fixed, prevent COLMAP bundle adjustment from changing them.
    if camera_params is not None:
        mapper_args.extend([
            '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_principal_point', '0',
            '--Mapper.ba_refine_extra_params', '0',
        ])
        print("\n[3/3] Mapper...")
        print("   - Intrinsics refinement: disabled")
    else:
        print("\n[3/3] Mapper...")
        print("   - Intrinsics refinement: enabled")
    
    print("   - init_min_num_inliers: 30")
    print("   - abs_pose_min_num_inliers: 15")
    print("   - min_num_matches: 15")

    map_output = subprocess.check_output(
        mapper_args, 
        universal_newlines=True
    )
    logfile.write(map_output)
    logfile.close()
    print('Sparse map created')
    
    print(f"\nCOLMAP completed, log written to: {logfile_name}")
    print("="*70)
