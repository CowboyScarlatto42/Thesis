import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if 'MPLCONFIGDIR' not in os.environ:
    mpl_cache_dir = Path(tempfile.gettempdir()) / 'matplotlib'
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(mpl_cache_dir)

import matplotlib.pyplot as plt
from matplotlib import cm, colors

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
POSTPROCESSING_DIR = os.path.join(REPO_ROOT, 'custom_codes', 'postprocessing')
for path in [REPO_ROOT, POSTPROCESSING_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from metrics_utils import load_mesh
from custom_codes.uncertainty.validate_geometry_proxy_on_mesh import interpolate_hessian_grid


"""
Export a colored mesh that visualizes the uncertainty-oriented score on NeuS vertices.

The input scalar grid stores H and is sampled at reconstruction-mesh vertices.
The exported mesh is colored by U = -log10(H + eps), where lower H means lower
local geometry sensitivity and therefore higher uncertainty. Geometry is not
modified; only vertex colors are changed.
"""


def valid_grid_values(values, inside):
    """Return values that are both inside the grid bounds and finite."""
    finite = np.isfinite(values)
    return inside & finite


def stats(values):
    """Compute basic finite-value statistics for metadata."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'min': None, 'max': None, 'mean': None, 'median': None}
    return {
        'min': float(values.min()),
        'max': float(values.max()),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
    }


def percentile_limits(values, lower_percentile, upper_percentile):
    """Compute robust visualization limits from percentiles."""
    if values.size == 0:
        raise RuntimeError('no valid values available for percentile color scaling')
    lower = float(np.percentile(values, lower_percentile))
    upper = float(np.percentile(values, upper_percentile))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise RuntimeError('non-finite percentile limits')
    if upper <= lower:
        delta = max(abs(lower) * 1e-6, 1e-12)
        lower -= delta
        upper += delta
    return lower, upper


def resolve_color_limits(values, lower_percentile, upper_percentile, manual_min, manual_max, name):
    """Choose manual or percentile-based color limits for one visualization."""
    if (manual_min is None) != (manual_max is None):
        raise ValueError('both --{}_vmin and --{}_vmax must be provided together'.format(name, name))
    if manual_min is not None:
        lower = float(manual_min)
        upper = float(manual_max)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError('{} manual color limits must be finite'.format(name))
        if upper <= lower:
            raise ValueError('--{}_vmax must be greater than --{}_vmin'.format(name, name))
        return lower, upper, 'manual'

    lower, upper = percentile_limits(values, lower_percentile, upper_percentile)
    return lower, upper, 'percentile'


def values_to_rgba(values, valid_mask, lower, upper, cmap_name, neutral_rgba=(160, 160, 160, 255)):
    """Map scalar values to RGBA colors, using neutral gray for invalid vertices."""
    rgba = np.zeros((len(values), 4), dtype=np.uint8)
    rgba[:] = np.asarray(neutral_rgba, dtype=np.uint8)
    norm = colors.Normalize(vmin=lower, vmax=upper, clip=True)
    cmap = cm.get_cmap(cmap_name)
    mapped = cmap(norm(values[valid_mask]))
    rgba[valid_mask] = (mapped * 255.0).round().astype(np.uint8)
    return rgba


def save_vertex_csv(path, vertices, inside, hessian, uncertainty_score):
    """Write per-vertex proxy values used for coloring and inspection."""
    fieldnames = [
        'vertex_index',
        'x_norm',
        'y_norm',
        'z_norm',
        'inside_bounds',
        'hessian_raw',
        'uncertainty_score',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (vertex, is_inside, h_value, u_value) in enumerate(
            zip(vertices, inside, hessian, uncertainty_score)
        ):
            writer.writerow({
                'vertex_index': idx,
                'x_norm': vertex[0],
                'y_norm': vertex[1],
                'z_norm': vertex[2],
                'inside_bounds': bool(is_inside),
                'hessian_raw': h_value,
                'uncertainty_score': u_value,
            })


def save_colorbar(path, cmap_name, lower, upper, title, label):
    """Save a standalone horizontal colorbar for one mesh visualization."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    fig.subplots_adjust(bottom=0.45)
    norm = colors.Normalize(vmin=lower, vmax=upper)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap_name))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_label(label)
    ax.set_title(title)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_json(path, payload):
    """Write an indented JSON metadata file."""
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def main():
    """CLI entry point for coloring a reconstruction mesh with U values."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--hessian_grid', type=str, required=True)
    parser.add_argument('--reconstruction_mesh', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--lower_percentile', type=float, default=5.0)
    parser.add_argument('--upper_percentile', type=float, default=95.0)
    parser.add_argument('--u_vmin', type=float, default=None)
    parser.add_argument('--u_vmax', type=float, default=None)
    parser.add_argument('--eps', type=float, default=1e-12)
    args = parser.parse_args()

    if args.upper_percentile <= args.lower_percentile:
        raise ValueError('--upper_percentile must be greater than --lower_percentile')
    if args.eps <= 0.0:
        raise ValueError('--eps must be positive')

    os.makedirs(args.output_dir, exist_ok=True)

    hessian_grid = np.load(args.hessian_grid).astype(np.float64)
    if hessian_grid.ndim != 3 or len(set(hessian_grid.shape)) != 1:
        raise ValueError('expected scalar Hessian grid with shape [R, R, R], got {}'.format(hessian_grid.shape))

    mesh = load_mesh(args.reconstruction_mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    hessian_raw, inside = interpolate_hessian_grid(hessian_grid, vertices)

    vertex_count = int(len(vertices))
    inside_count = int(inside.sum())
    outside_count = int(vertex_count - inside_count)
    inside_fraction = float(inside_count / vertex_count) if vertex_count > 0 else 0.0

    print('vertex_count:', vertex_count)
    print('inside_bounds_count:', inside_count)
    print('outside_bounds_count:', outside_count)
    print('inside_bounds_fraction:', inside_fraction)

    if inside_count == 0:
        raise RuntimeError('all mesh vertices are outside [-1, 1]^3')
    if inside_fraction < 0.99:
        raise RuntimeError(
            'inside bounds fraction {:.6f} < 0.99; the reconstruction mesh is likely not in normalized NeuS coordinates'.format(
                inside_fraction
            )
        )

    valid_mask = valid_grid_values(hessian_raw, inside)
    if not np.any(valid_mask):
        raise RuntimeError('no finite inside-bounds Hessian values were interpolated')
    if np.any(hessian_raw[valid_mask] < 0.0):
        raise RuntimeError('interpolated Hessian contains negative values inside bounds')

    uncertainty_score = np.full(vertex_count, np.nan, dtype=np.float64)
    uncertainty_score[valid_mask] = -np.log10(hessian_raw[valid_mask] + args.eps)

    if not np.all(np.isfinite(uncertainty_score[valid_mask])):
        raise RuntimeError('non-finite uncertainty_score values')

    u_lower, u_upper, u_limit_source = resolve_color_limits(
        uncertainty_score[valid_mask],
        args.lower_percentile,
        args.upper_percentile,
        args.u_vmin,
        args.u_vmax,
        'u',
    )

    uncertainty_rgba = values_to_rgba(
        uncertainty_score,
        valid_mask,
        u_lower,
        u_upper,
        'magma',
    )

    uncertainty_mesh = mesh.copy()
    uncertainty_mesh.visual.vertex_colors = uncertainty_rgba

    uncertainty_ply = os.path.join(args.output_dir, 'mesh_geometry_uncertainty_score.ply')
    uncertainty_mesh.export(uncertainty_ply)

    vertex_csv = os.path.join(args.output_dir, 'vertex_proxy_values.csv')
    save_vertex_csv(vertex_csv, vertices, inside, hessian_raw, uncertainty_score)

    uncertainty_colorbar = os.path.join(args.output_dir, 'uncertainty_score_colorbar.png')
    save_colorbar(
        uncertainty_colorbar,
        'magma',
        u_lower,
        u_upper,
        'Uncertainty score',
        r'$U = -\log_{10}(H + \epsilon)$',
    )

    h_positive_count = int((hessian_raw[valid_mask] > 0.0).sum())
    metadata = {
        'hessian_grid_path': args.hessian_grid,
        'reconstruction_mesh_path': args.reconstruction_mesh,
        'grid_shape': list(hessian_grid.shape),
        'vertex_count': vertex_count,
        'inside_bounds_count': inside_count,
        'outside_bounds_count': outside_count,
        'fraction_inside_bounds': inside_fraction,
        'hessian_gt_0_count': h_positive_count,
        'hessian_gt_0_fraction_valid': float(h_positive_count / int(valid_mask.sum())),
        'raw_hessian_stats_valid_inside': stats(hessian_raw[valid_mask]),
        'visualization_transform': {
            'uncertainty_score': '-log10(hessian_raw + eps)',
        },
        'eps': float(args.eps),
        'percentile_settings': {
            'lower_percentile': float(args.lower_percentile),
            'upper_percentile': float(args.upper_percentile),
        },
        'manual_color_limits': {
            'u_vmin': args.u_vmin,
            'u_vmax': args.u_vmax,
        },
        'color_limit_source': {
            'uncertainty_score': u_limit_source,
        },
        'effective_percentile_limits': {
            'uncertainty_score': {
                'lower': u_lower,
                'upper': u_upper,
            },
        },
        'colormaps': {
            'uncertainty_score': 'magma',
            'outside_bounds_color': 'neutral gray rgba(160,160,160,255)',
        },
        'output_filenames': {
            'uncertainty_score_ply': os.path.basename(uncertainty_ply),
            'vertex_proxy_values_csv': os.path.basename(vertex_csv),
            'uncertainty_score_colorbar_png': os.path.basename(uncertainty_colorbar),
        },
        'note': (
            'Percentile clipping affects visualization colors only. Raw hessian_raw and '
            'uncertainty_score values are preserved in CSV. U is an uncertainty-oriented '
            'score, not calibrated uncertainty.'
        ),
    }
    save_json(os.path.join(args.output_dir, 'colored_mesh_metadata.json'), metadata)

    print('vertices with H > 0:', h_positive_count)
    print('uncertainty score PLY:', uncertainty_ply)
    print('output_dir:', args.output_dir)


if __name__ == '__main__':
    main()
