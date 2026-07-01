# Thesis: Multi-View Neural 3D Reconstruction of Unknown Spacecraft Targets in Autonomous Rendezvous

This repository contains the code used for the thesis experiments based on
NeuS. It is a unified version of the two working folders used during
development:

- `Ablation_test`: preprocessing, COLMAP/CORTO utilities, mesh metrics,
  ablation-related postprocessing, and thesis plots.
- `BayesSDF_test`: experimental geometric sensitivity/uncertainty proxy
  inspired by BayesSDF.

This repository is based on the official NeuS implementation by Peng Wang et al. 
Original code is distributed under the MIT License. 
The thesis-specific additions include CORTO preprocessing, 
COLMAP alignment utilities, mesh-evaluation scripts, and the geometry-oriented uncertainty proxy.

## Main Components

### NeuS core

- `exp_runner.py`: main NeuS training, validation, rendering, and mesh
  extraction entry point.
- `models/`: NeuS model, dataset, fields, and renderer implementation.
- `confs/`: configuration files for the experiments.
- `preprocess_custom_data/`: original NeuS custom-data preprocessing utilities.

The renderer in `models/renderer.py` is the extended version from the
BayesSDF-inspired branch. When no deformation grid is passed, it behaves as the
standard NeuS renderer.

### Custom preprocessing and COLMAP utilities

- `custom_codes/preprocessing/`: dataset conversion, filtering, scale matrix
  construction, and NeuS input checks.
- `custom_codes/COLMAP_codes/`: COLMAP execution, pruning, orbit alignment, and
  combined NeuS dataset construction.

### Postprocessing, metrics, and plots

- `custom_codes/postprocessing/mesh_metrics.py`: mesh-to-mesh evaluation.
- `custom_codes/postprocessing/plot_mesh_metrics.py`: metric plotting and
  table generation.
- `custom_codes/postprocessing/thesis_plots.py`: thesis plot generation.
- `custom_codes/postprocessing/analyze_realistic_colmap_pose_errors.py`:
  Residual pose-error analysis for the realistic COLMAP-based datasets.
- `custom_codes/postprocessing/colormap.py`: colormap utilities used by the
  postprocessing scripts.

### Geometric sensitivity/uncertainty proxy

- `models/deformation_grid.py`: dense trilinear deformation grid in normalized
  NeuS coordinates.
- `custom_codes/uncertainty/`: scripts for the BayesSDF-inspired sensitivity
  proxy.

The uncertainty pipeline introduces an optional deformation grid in the NeuS
renderer and measures how much rendered RGB values change under infinitesimal
local geometric perturbations. The accumulated quantity is `H`, a
Laplace-inspired local sensitivity/curvature proxy. Low `H` values indicate a
flatter local response and are visualized through the uncertainty-oriented score
`U = -log10(H + eps)`. This is not a calibrated Bayesian variance.

Useful scripts:

- `accumulate_hessian_proxy.py`
- `accumulate_multiview_geometry_proxy.py`
- `export_geometry_proxy_colored_mesh.py`
- `validate_geometry_proxy_on_mesh.py`
- `correlate_mesh_metrics_distance_proxy.py`

### Custom CORTO scripts

- `corto_custom/`: custom scripts developed on top of the original CORTO
  repository for spacecraft scenario generation, Sun/phase-angle diagnostics,
  Blender mask rendering and the custom CORTO tutorial used in this work.

This folder intentionally contains only the custom code to be added to an
existing CORTO checkout. It does not duplicate the full CORTO repository,
virtual environments, generated outputs, or cache files.

