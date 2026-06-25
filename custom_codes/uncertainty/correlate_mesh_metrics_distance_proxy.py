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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
POSTPROCESSING_DIR = os.path.join(REPO_ROOT, 'custom_codes', 'postprocessing')
for path in [REPO_ROOT, POSTPROCESSING_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)


"""
Correlate mesh-metrics pred-to-GT distance with U = -log10(H + eps).

This validation uses the same nearest-neighbor distance style as mesh_metrics:
sample points on the predicted mesh, sample points on the GT mesh, and compute
pred-to-GT nearest-neighbor distances. H values are interpolated at the same
predicted-mesh samples and converted to U before computing Spearman and
sparsification diagnostics.
"""


def finite_stats(values):
    """Compute descriptive statistics after removing non-finite values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'std': float('nan'),
            'p95': float('nan'),
            'max': float('nan'),
        }
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'p95': float(np.quantile(values, 0.95)),
        'max': float(np.max(values)),
    }


def resolve_conf_path(conf_path):
    """Resolve a config path relative to the repository when possible."""
    if os.path.isabs(conf_path):
        return conf_path
    candidate = os.path.join(REPO_ROOT, conf_path)
    if os.path.isfile(candidate):
        return candidate
    return conf_path


def load_scale_mat_from_conf(conf_path, case):
    """Load scale_mat_0 from the NeuS cameras file referenced by a config."""
    from pyhocon import ConfigFactory

    conf_path = resolve_conf_path(conf_path)
    with open(conf_path) as f:
        conf_text = f.read().replace('CASE_NAME', case)
    conf = ConfigFactory.parse_string(conf_text)
    data_dir = conf['dataset.data_dir'].replace('CASE_NAME', case)
    cameras_name = conf.get_string('dataset.render_cameras_name')
    cameras = np.load(os.path.join(data_dir, cameras_name))
    return cameras['scale_mat_0'].astype(np.float64)


def transform_gt_mesh_to_normalized(mesh, scale_mat):
    """Map a GT-frame mesh into normalized NeuS coordinates."""
    normalized = mesh.copy()
    scale = float(scale_mat[0, 0])
    translation = scale_mat[:3, 3].astype(np.float64)
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError('invalid scale_mat[0, 0]: {}'.format(scale))
    vertices = np.asarray(normalized.vertices, dtype=np.float64)
    normalized.vertices = (vertices - translation[None, :]) / scale
    return normalized


def spearman_payload(error, proxy):
    """Compute Spearman correlation with finite-value filtering."""
    from scipy.stats import spearmanr

    mask = np.isfinite(error) & np.isfinite(proxy)
    if int(mask.sum()) < 2:
        return {'rho': float('nan'), 'p_value': float('nan'), 'n': int(mask.sum())}
    rho, p_value = spearmanr(error[mask], proxy[mask])
    return {'rho': float(rho), 'p_value': float(p_value), 'n': int(mask.sum())}


def uncertainty_score_from_hessian(hessian, eps):
    """Convert raw H values to U = -log10(H + eps)."""
    return -np.log10(hessian + eps)


def mean_remaining_errors(error, order, fractions_removed):
    """Compute mean remaining error after removing ranked samples."""
    n = len(error)
    curve = []
    for fraction in fractions_removed:
        remove_count = min(int(round(fraction * n)), n - 1)
        keep = order[remove_count:]
        curve.append(float(np.mean(error[keep])))
    return np.asarray(curve, dtype=np.float64)


def sparsification_curve(error, uncertainty, seed, num_random_permutations=50, steps=101):
    """Build sparsification curves by removing highest-uncertainty samples first."""
    n = len(error)
    fractions_removed = np.linspace(0.0, 0.99, steps)
    rng = np.random.default_rng(seed)
    proxy_order = np.argsort(-uncertainty)
    oracle_order = np.argsort(-error)
    random_orders = [rng.permutation(n) for _ in range(num_random_permutations)]

    proxy_curve = mean_remaining_errors(error, proxy_order, fractions_removed)
    oracle_curve = mean_remaining_errors(error, oracle_order, fractions_removed)
    random_curves = np.asarray([
        mean_remaining_errors(error, order, fractions_removed)
        for order in random_orders
    ], dtype=np.float64)

    random_mean = random_curves.mean(axis=0)
    random_std = random_curves.std(axis=0)
    random_p05 = np.percentile(random_curves, 5, axis=0)
    random_p95 = np.percentile(random_curves, 95, axis=0)

    rows = []
    for idx, fraction in enumerate(fractions_removed):
        rows.append({
            'removed_fraction': float(fraction),
            'proxy_error': float(proxy_curve[idx]),
            'random_mean_error': float(random_mean[idx]),
            'random_std_error': float(random_std[idx]),
            'random_p05_error': float(random_p05[idx]),
            'random_p95_error': float(random_p95[idx]),
            'oracle_error': float(oracle_curve[idx]),
        })
    return rows, {
        'fractions': fractions_removed,
        'proxy_curve': proxy_curve,
        'oracle_curve': oracle_curve,
        'random_curves': random_curves,
        'random_mean_curve': random_mean,
    }


def integrate_curve_with_endpoint(fractions, values, max_removed_fraction):
    """Integrate a curve up to a selected removed-fraction endpoint."""
    fractions = np.asarray(fractions, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if max_removed_fraction <= 0.0 or max_removed_fraction > fractions.max():
        raise ValueError('max_removed_fraction must be in (0, {}]'.format(fractions.max()))

    mask = fractions <= max_removed_fraction
    x = fractions[mask]
    y = values[mask]
    if x[-1] < max_removed_fraction:
        endpoint_value = float(np.interp(max_removed_fraction, fractions, values))
        x = np.concatenate([x, np.asarray([max_removed_fraction], dtype=np.float64)])
        y = np.concatenate([y, np.asarray([endpoint_value], dtype=np.float64)])
    trapezoid = getattr(np, 'trapezoid', np.trapz)
    return float(trapezoid(y, x))


def compute_auc_metrics(fractions, proxy_curve, oracle_curve, random_curves, max_removed_fraction):
    """Compute proxy, random, and oracle AUC summaries for sparsification."""
    auc_proxy = integrate_curve_with_endpoint(fractions, proxy_curve, max_removed_fraction)
    auc_oracle = integrate_curve_with_endpoint(fractions, oracle_curve, max_removed_fraction)
    random_mean_curve = random_curves.mean(axis=0)
    auc_random_mean_curve = integrate_curve_with_endpoint(fractions, random_mean_curve, max_removed_fraction)
    random_aucs = np.asarray([
        integrate_curve_with_endpoint(fractions, curve, max_removed_fraction)
        for curve in random_curves
    ], dtype=np.float64)

    return {
        'max_removed_fraction': float(max_removed_fraction),
        'auc_proxy': float(auc_proxy),
        'auc_random_mean_curve': float(auc_random_mean_curve),
        'auc_oracle': float(auc_oracle),
        'auc_random_permutation_mean': float(random_aucs.mean()),
        'auc_random_permutation_std': float(random_aucs.std()),
        'auc_random_permutation_p05': float(np.percentile(random_aucs, 5)),
        'auc_random_permutation_p95': float(np.percentile(random_aucs, 95)),
        'relative_auc_improvement_vs_random': float(
            (auc_random_mean_curve - auc_proxy) / auc_random_mean_curve
        ) if auc_random_mean_curve != 0.0 else float('nan'),
        'normalized_auc_proxy': float(auc_proxy / max_removed_fraction),
        'normalized_auc_random': float(auc_random_mean_curve / max_removed_fraction),
        'normalized_auc_oracle': float(auc_oracle / max_removed_fraction),
    }


def write_rows_csv(path, rows):
    """Write a list of row dictionaries to CSV."""
    if len(rows) == 0:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_sparsification(path, rows):
    """Save the sparsification curve with random baseline and oracle reference."""
    x = np.asarray([row['removed_fraction'] for row in rows], dtype=np.float64)
    proxy = np.asarray([row['proxy_error'] for row in rows], dtype=np.float64)
    random_mean = np.asarray([row['random_mean_error'] for row in rows], dtype=np.float64)
    random_p05 = np.asarray([row['random_p05_error'] for row in rows], dtype=np.float64)
    random_p95 = np.asarray([row['random_p95_error'] for row in rows], dtype=np.float64)
    oracle = np.asarray([row['oracle_error'] for row in rows], dtype=np.float64)

    plt.figure(figsize=(7, 5))
    plt.plot(x, proxy, label='Uncertainty proxy')
    plt.plot(x, random_mean, label='Random')
    plt.fill_between(x, random_p05, random_p95, alpha=0.25, label='_nolegend_')
    plt.plot(x, oracle, label='Oracle')
    plt.xlabel('Fraction of removed points')
    plt.ylabel('Mean pred-to-GT NN distance of remaining points')
    plt.title('Sparsification curve')
    plt.xlim(0.0, 0.95)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def interpolate_hessian_grid(hessian_grid, points_norm):
    """Trilinearly interpolate a cubic proxy grid at normalized NeuS points."""
    if hessian_grid.ndim != 3 or hessian_grid.shape[0] != hessian_grid.shape[1] or hessian_grid.shape[1] != hessian_grid.shape[2]:
        raise ValueError('hessian grid must have shape [R, R, R]')

    resolution = hessian_grid.shape[0]
    points = np.asarray(points_norm, dtype=np.float64)
    normalized = (points + 1.0) * 0.5 * (resolution - 1)
    x = normalized[:, 0]
    y = normalized[:, 1]
    z = normalized[:, 2]

    inside = (
        (x >= 0.0) & (x <= resolution - 1) &
        (y >= 0.0) & (y <= resolution - 1) &
        (z >= 0.0) & (z <= resolution - 1)
    )
    values = np.zeros(points.shape[0], dtype=np.float64)
    if not np.any(inside):
        return values, inside

    xi = x[inside]
    yi = y[inside]
    zi = z[inside]
    x0 = np.floor(xi).astype(np.int64)
    y0 = np.floor(yi).astype(np.int64)
    z0 = np.floor(zi).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, resolution - 1)
    y1 = np.clip(y0 + 1, 0, resolution - 1)
    z1 = np.clip(z0 + 1, 0, resolution - 1)
    wx = xi - x0
    wy = yi - y0
    wz = zi - z0

    c000 = hessian_grid[z0, y0, x0]
    c001 = hessian_grid[z0, y0, x1]
    c010 = hessian_grid[z0, y1, x0]
    c011 = hessian_grid[z0, y1, x1]
    c100 = hessian_grid[z1, y0, x0]
    c101 = hessian_grid[z1, y0, x1]
    c110 = hessian_grid[z1, y1, x0]
    c111 = hessian_grid[z1, y1, x1]

    c00 = c000 * (1.0 - wx) + c001 * wx
    c01 = c010 * (1.0 - wx) + c011 * wx
    c10 = c100 * (1.0 - wx) + c101 * wx
    c11 = c110 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c01 * wy
    c1 = c10 * (1.0 - wy) + c11 * wy
    values[inside] = c0 * (1.0 - wz) + c1 * wz
    return values, inside


def write_csv(path, points, distances, hessian, uncertainty_score):
    """Write sampled points, distances, raw H, and U values."""
    fieldnames = [
        'x_norm',
        'y_norm',
        'z_norm',
        'pred_to_gt_nn_distance',
        'hessian',
        'uncertainty_score',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(points.shape[0]):
            writer.writerow({
                'x_norm': points[idx, 0],
                'y_norm': points[idx, 1],
                'z_norm': points[idx, 2],
                'pred_to_gt_nn_distance': distances[idx],
                'hessian': hessian[idx],
                'uncertainty_score': uncertainty_score[idx],
            })


def save_json(path, payload):
    """Write an indented JSON summary."""
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def main():
    """CLI entry point for mesh-metrics-style proxy correlation."""
    parser = argparse.ArgumentParser(
        description=(
            'Correlate mesh_metrics-style pred->gt nearest-neighbor distances with '
            'U = -log10(H + eps) on the same normalized reconstruction samples.'
        )
    )
    parser.add_argument('--hessian_grid', type=str, required=True)
    parser.add_argument('--pred_mesh_normalized', type=str, required=True)
    parser.add_argument('--gt_mesh', type=str, required=True)
    parser.add_argument(
        '--gt_mesh_frame',
        choices=['normalized', 'gt'],
        default='normalized',
        help=(
            'Use "normalized" if --gt_mesh is already in the NeuS normalized frame. '
            'Use "gt" if --gt_mesh is in the original GT/CORTO frame and must be '
            'mapped with the inverse NeuS scale_mat.'
        ),
    )
    parser.add_argument('--conf', type=str, default='./confs/long_test.conf')
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--num_surface_points', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--num_random_permutations', type=int, default=50)
    parser.add_argument('--auc_max_removed_fraction', type=float, default=0.95)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    if args.num_surface_points <= 0:
        raise ValueError('--num_surface_points must be positive')
    if args.eps <= 0.0:
        raise ValueError('--eps must be positive')
    if args.num_random_permutations <= 0:
        raise ValueError('--num_random_permutations must be positive')
    if args.auc_max_removed_fraction <= 0.0 or args.auc_max_removed_fraction > 0.99:
        raise ValueError('--auc_max_removed_fraction must be in (0, 0.99]')

    from metrics_utils import load_mesh, nn_distances, sample_surface_points

    os.makedirs(args.output_dir, exist_ok=True)

    hessian_grid = np.load(args.hessian_grid).astype(np.float64)
    pred_mesh = load_mesh(args.pred_mesh_normalized)
    gt_mesh = load_mesh(args.gt_mesh)
    scale_mat = None
    if args.gt_mesh_frame == 'gt':
        scale_mat = load_scale_mat_from_conf(args.conf, args.case)
        gt_mesh = transform_gt_mesh_to_normalized(gt_mesh, scale_mat)

    pred_seed = None if args.seed == -1 else int(args.seed) + 1
    gt_seed = None if args.seed == -1 else int(args.seed)
    pred_points = sample_surface_points(pred_mesh, args.num_surface_points, seed=pred_seed)
    gt_points = sample_surface_points(gt_mesh, args.num_surface_points, seed=gt_seed)

    pred_to_gt = nn_distances(pred_points, gt_points)
    hessian, inside = interpolate_hessian_grid(hessian_grid, pred_points)

    finite = np.isfinite(pred_to_gt) & np.isfinite(hessian) & inside
    if int(finite.sum()) < 2:
        raise RuntimeError('not enough finite inside-bounds samples for correlation')

    pred_points = pred_points[finite]
    pred_to_gt = pred_to_gt[finite]
    hessian = hessian[finite]
    if np.any(hessian < 0.0):
        raise RuntimeError('interpolated H values must be non-negative')
    uncertainty_score = uncertainty_score_from_hessian(hessian, args.eps)
    if not np.all(np.isfinite(uncertainty_score)):
        raise RuntimeError('non-finite U values')

    spearman_u = spearman_payload(pred_to_gt, uncertainty_score)
    sparsification_rows, sparsification_payload = sparsification_curve(
        pred_to_gt,
        uncertainty_score,
        args.seed,
        num_random_permutations=args.num_random_permutations,
    )
    auc_payload = compute_auc_metrics(
        sparsification_payload['fractions'],
        sparsification_payload['proxy_curve'],
        sparsification_payload['oracle_curve'],
        sparsification_payload['random_curves'],
        args.auc_max_removed_fraction,
    )

    np.save(os.path.join(args.output_dir, 'pred_points_normalized.npy'), pred_points)
    np.save(os.path.join(args.output_dir, 'gt_points_normalized.npy'), gt_points)
    np.save(os.path.join(args.output_dir, 'pred_to_gt_nn_distance.npy'), pred_to_gt)
    np.save(os.path.join(args.output_dir, 'hessian_on_pred_points.npy'), hessian)
    np.save(os.path.join(args.output_dir, 'uncertainty_score_on_pred_points.npy'), uncertainty_score)
    write_csv(
        os.path.join(args.output_dir, 'pred_points_distance_proxy.csv'),
        pred_points,
        pred_to_gt,
        hessian,
        uncertainty_score,
    )
    write_rows_csv(
        os.path.join(args.output_dir, 'sparsification_curve.csv'),
        sparsification_rows,
    )
    plot_sparsification(
        os.path.join(args.output_dir, 'sparsification_curve.png'),
        sparsification_rows,
    )

    summary = {
        'hessian_grid': args.hessian_grid,
        'pred_mesh_normalized': args.pred_mesh_normalized,
        'gt_mesh': args.gt_mesh,
        'gt_mesh_frame': args.gt_mesh_frame,
        'conf': args.conf if args.gt_mesh_frame == 'gt' else None,
        'case': args.case if args.gt_mesh_frame == 'gt' else None,
        'num_requested_surface_points': int(args.num_surface_points),
        'num_valid_samples': int(len(pred_to_gt)),
        'fraction_inside_hessian_bounds_before_filter': float(np.mean(inside)),
        'distance_method': 'mesh_metrics.sample_surface_points + mesh_metrics.nn_distances(pred_points, gt_points)',
        'coordinate_note': (
            'Distances are computed in the NeuS normalized frame. If gt_mesh_frame="gt", '
            'GT vertices are mapped with v_norm = (v_gt - scale_mat[:3, 3]) / scale_mat[0, 0]. '
            'Hessian values are interpolated at the sampled predicted-mesh points.'
        ),
        'scale_mat_0': scale_mat.tolist() if scale_mat is not None else None,
        'pred_to_gt_distance_stats': finite_stats(pred_to_gt),
        'hessian_stats': finite_stats(hessian),
        'uncertainty_score': 'U = -log10(H + eps)',
        'uncertainty_score_stats': finite_stats(uncertainty_score),
        'spearman_pred_to_gt_distance_vs_uncertainty_score': spearman_u,
        'sparsification': {
            'uncertainty_proxy': 'U = -log10(H + eps)',
            'random_baseline': {
                'num_permutations': int(args.num_random_permutations),
                'seed': int(args.seed),
            },
            'auc': auc_payload,
        },
    }
    save_json(os.path.join(args.output_dir, 'spearman_mesh_metrics_distance_summary.json'), summary)

    print('valid samples:', len(pred_to_gt))
    print('inside-bounds fraction before filter:', summary['fraction_inside_hessian_bounds_before_filter'])
    print('pred->gt distance mean:', summary['pred_to_gt_distance_stats']['mean'])
    print('Spearman distance vs U:', spearman_u)
    print('AUC proxy:', auc_payload['auc_proxy'])
    print('AUC random mean:', auc_payload['auc_random_mean_curve'])
    print('AUC oracle:', auc_payload['auc_oracle'])
    print('output_dir:', args.output_dir)


if __name__ == '__main__':
    main()
