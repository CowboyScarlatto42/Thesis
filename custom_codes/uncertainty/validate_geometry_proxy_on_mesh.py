import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from pyhocon import ConfigFactory
from scipy.stats import spearmanr

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

from metrics_utils import load_mesh, sample_surface_points


"""
Validate U = -log10(H + eps) against GT surface distance on sampled mesh points.

The proxy grid is interpolated in normalized NeuS coordinates. Sampled points
are then mapped to the GT/CORTO frame with scale_mat_0, and their geometric
error is measured as point-to-surface distance to the GT mesh. The script writes
pointwise Spearman correlation and sparsification/AUC diagnostics for U.
"""


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
    conf_path = resolve_conf_path(conf_path)
    with open(conf_path) as f:
        conf_text = f.read().replace('CASE_NAME', case)
    conf = ConfigFactory.parse_string(conf_text)
    data_dir = conf['dataset.data_dir'].replace('CASE_NAME', case)
    cameras_name = conf.get_string('dataset.render_cameras_name')
    cameras = np.load(os.path.join(data_dir, cameras_name))
    scale_mat = cameras['scale_mat_0'].astype(np.float64)
    return scale_mat


def normalized_to_gt_frame(points_norm, scale_mat):
    # Same convention used by Runner.validate_mesh(world_space=True):
    # vertices = vertices * scale_mat[0, 0] + scale_mat[:3, 3].
    return points_norm * scale_mat[0, 0] + scale_mat[:3, 3][None, :]


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

    # Saved scalar Hessian grid is indexed [z, y, x], while points are (x, y, z).
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


def point_to_surface_distance(points, mesh):
    """Compute point-to-surface distances using trimesh proximity queries."""
    try:
        closest, distances, triangle_id = mesh.nearest.on_surface(points)
    except BaseException as exc:
        raise RuntimeError(
            'trimesh point-to-surface query failed. Install runtime dependencies such as rtree, '
            'or run in an environment where trimesh proximity queries are available.'
        ) from exc
    return np.asarray(distances, dtype=np.float64)


def spearman_payload(error, hessian):
    """Compute Spearman correlation with finite-value filtering."""
    mask = np.isfinite(error) & np.isfinite(hessian)
    if int(mask.sum()) < 2:
        return {'rho': float('nan'), 'p_value': float('nan'), 'n': int(mask.sum())}
    rho, p_value = spearmanr(error[mask], hessian[mask])
    return {'rho': float(rho), 'p_value': float(p_value), 'n': int(mask.sum())}


def write_sampled_points_csv(path, points_norm, points_gt, hessian, uncertainty_score, error):
    """Write sampled points, raw H, U, and GT distances."""
    fieldnames = [
        'x_norm',
        'y_norm',
        'z_norm',
        'x_gt',
        'y_gt',
        'z_gt',
        'hessian',
        'uncertainty_score',
        'gt_distance',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p_norm, p_gt, h, u, d in zip(points_norm, points_gt, hessian, uncertainty_score, error):
            writer.writerow({
                'x_norm': p_norm[0],
                'y_norm': p_norm[1],
                'z_norm': p_norm[2],
                'x_gt': p_gt[0],
                'y_gt': p_gt[1],
                'z_gt': p_gt[2],
                'hessian': h,
                'uncertainty_score': u,
                'gt_distance': d,
            })


def mean_remaining_errors(error, order, fractions_removed):
    """Compute mean error after progressively removing ranked samples."""
    n = len(error)
    curve = []
    for fraction in fractions_removed:
        remove_count = min(int(round(fraction * n)), n - 1)
        keep = order[remove_count:]
        curve.append(float(np.mean(error[keep])))
    return np.asarray(curve, dtype=np.float64)


def integrate_curve_with_endpoint(fractions, values, max_removed_fraction):
    """Integrate a curve up to a chosen removed-fraction endpoint."""
    fractions = np.asarray(fractions, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if max_removed_fraction <= 0.0 or max_removed_fraction > fractions.max():
        raise ValueError('auc_max_removed_fraction must be in (0, {}]'.format(fractions.max()))

    mask = fractions <= max_removed_fraction
    x = fractions[mask]
    y = values[mask]
    if x[-1] < max_removed_fraction:
        endpoint_value = float(np.interp(max_removed_fraction, fractions, values))
        x = np.concatenate([x, np.asarray([max_removed_fraction], dtype=np.float64)])
        y = np.concatenate([y, np.asarray([endpoint_value], dtype=np.float64)])
    trapezoid = getattr(np, 'trapezoid', np.trapz)
    return float(trapezoid(y, x))


def compute_auc_metrics(fractions, proxy_curve, oracle_curve, random_curves, auc_max_removed_fraction, tolerance=1e-12):
    """Compute AUC metrics for proxy, oracle, and random sparsification curves."""
    auc_proxy = integrate_curve_with_endpoint(fractions, proxy_curve, auc_max_removed_fraction)
    auc_oracle = integrate_curve_with_endpoint(fractions, oracle_curve, auc_max_removed_fraction)
    random_mean_curve = random_curves.mean(axis=0)
    auc_random_mean_curve = integrate_curve_with_endpoint(fractions, random_mean_curve, auc_max_removed_fraction)
    random_aucs = np.asarray([
        integrate_curve_with_endpoint(fractions, curve, auc_max_removed_fraction)
        for curve in random_curves
    ], dtype=np.float64)

    auc_values = np.concatenate([
        np.asarray([auc_proxy, auc_oracle, auc_random_mean_curve], dtype=np.float64),
        random_aucs,
    ])
    if not np.all(np.isfinite(auc_values)):
        raise RuntimeError('non-finite AUC value detected')
    if auc_oracle > auc_proxy + tolerance:
        raise RuntimeError('oracle AUC {} is greater than proxy AUC {}'.format(auc_oracle, auc_proxy))
    if auc_proxy > auc_random_mean_curve + tolerance:
        print(
            'WARNING: proxy AUC {} is greater than random-mean AUC {}; lower AUC is better'.format(
                auc_proxy,
                auc_random_mean_curve,
            )
        )

    return {
        'max_removed_fraction': float(auc_max_removed_fraction),
        'auc_proxy': auc_proxy,
        'auc_random_mean_curve': auc_random_mean_curve,
        'auc_oracle': auc_oracle,
        'auc_random_permutation_mean': float(random_aucs.mean()),
        'auc_random_permutation_std': float(random_aucs.std()),
        'auc_random_permutation_p05': float(np.percentile(random_aucs, 5)),
        'auc_random_permutation_p95': float(np.percentile(random_aucs, 95)),
        'relative_auc_improvement_vs_random': float(
            (auc_random_mean_curve - auc_proxy) / auc_random_mean_curve
        ) if auc_random_mean_curve != 0.0 else float('nan'),
        'normalized_auc_proxy': float(auc_proxy / auc_max_removed_fraction),
        'normalized_auc_random': float(auc_random_mean_curve / auc_max_removed_fraction),
        'normalized_auc_oracle': float(auc_oracle / auc_max_removed_fraction),
    }


def sparsification_curve(error, uncertainty_score, seed, num_random_permutations=50, steps=101):
    """Build sparsification curves by removing highest-U points first."""
    n = len(error)
    fractions_removed = np.linspace(0.0, 0.99, steps)
    rng = np.random.default_rng(seed)
    proxy_order = np.argsort(-uncertainty_score)
    oracle_order = np.argsort(-error)
    random_orders = [rng.permutation(n) for _ in range(num_random_permutations)]

    proxy_curve = mean_remaining_errors(error, proxy_order, fractions_removed)
    oracle_curve = mean_remaining_errors(error, oracle_order, fractions_removed)
    random_curves = np.asarray([
        mean_remaining_errors(error, order, fractions_removed)
        for order in random_orders
    ], dtype=np.float64)
    if random_curves.shape != (num_random_permutations, len(fractions_removed)):
        raise RuntimeError(
            'unexpected random curve matrix shape {}, expected {}'.format(
                random_curves.shape,
                (num_random_permutations, len(fractions_removed)),
            )
        )
    if not np.all(np.isfinite(random_curves)):
        raise RuntimeError('random sparsification curves contain NaN or Inf values')

    random_mean = random_curves.mean(axis=0)
    random_std = random_curves.std(axis=0)
    random_p05 = np.percentile(random_curves, 5, axis=0)
    random_p95 = np.percentile(random_curves, 95, axis=0)

    rows = []
    for idx, fraction in enumerate(fractions_removed):
        rows.append({
            'removed_fraction': float(fraction),
            'estimated_proxy_error': float(proxy_curve[idx]),
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


def write_csv(path, rows, fieldnames=None):
    """Write generic row dictionaries to CSV."""
    if len(rows) == 0:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_sparsification_curves(rows, tolerance=1e-12):
    """Validate curve finiteness and oracle monotonicity before plotting."""
    estimated = np.asarray([row['estimated_proxy_error'] for row in rows], dtype=np.float64)
    random_mean = np.asarray([row['random_mean_error'] for row in rows], dtype=np.float64)
    random_p05 = np.asarray([row['random_p05_error'] for row in rows], dtype=np.float64)
    random_p95 = np.asarray([row['random_p95_error'] for row in rows], dtype=np.float64)
    oracle = np.asarray([row['oracle_error'] for row in rows], dtype=np.float64)

    if estimated.shape != random_mean.shape or estimated.shape != random_p05.shape or estimated.shape != random_p95.shape or estimated.shape != oracle.shape:
        raise RuntimeError(
            'sparsification curves have mismatched shapes'
        )
    if not (
        np.all(np.isfinite(estimated)) and
        np.all(np.isfinite(random_mean)) and
        np.all(np.isfinite(random_p05)) and
        np.all(np.isfinite(random_p95)) and
        np.all(np.isfinite(oracle))
    ):
        raise RuntimeError('sparsification curves contain NaN or Inf values')

    initial = np.asarray([estimated[0], random_mean[0], random_p05[0], random_p95[0], oracle[0]], dtype=np.float64)
    if np.max(np.abs(initial - initial[0])) > tolerance:
        raise RuntimeError(
            'sparsification curves do not share the same initial mean error: {}'.format(initial.tolist())
        )
    if np.any(np.diff(oracle) > tolerance):
        raise RuntimeError('oracle sparsification curve is not non-increasing within tolerance {}'.format(tolerance))

    return estimated, random_mean, random_p05, random_p95, oracle


def plot_sparsification(path, rows):
    """Save the proxy, random-baseline, and oracle sparsification curves."""
    x = np.asarray([row['removed_fraction'] for row in rows], dtype=np.float64)
    estimated, random_mean, random_p05, random_p95, oracle = validate_sparsification_curves(rows)

    plt.figure(figsize=(7, 5))
    plt.plot(x, estimated, label='Sparsification curve')
    plt.plot(x, random_mean, label='Random')
    plt.fill_between(x, random_p05, random_p95, alpha=0.25, label='_nolegend_')
    plt.plot(x, oracle, label='Oracle')
    plt.xlabel('Fraction of removed points')
    plt.ylabel('Mean geometric error of remaining points')
    plt.title('Sparsification curve')
    plt.xlim(0.0, 0.95)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def quartile_summary(error, uncertainty_score):
    """Summarize geometric error across quartiles of U."""
    quantiles = np.quantile(uncertainty_score, [0.0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    for idx in range(4):
        lo = quantiles[idx]
        hi = quantiles[idx + 1]
        if idx == 3:
            mask = (uncertainty_score >= lo) & (uncertainty_score <= hi)
        else:
            mask = (uncertainty_score >= lo) & (uncertainty_score < hi)
        values = error[mask]
        rows.append({
            'quartile': idx + 1,
            'uncertainty_score_min': float(lo),
            'uncertainty_score_max': float(hi),
            'count': int(mask.sum()),
            'mean_error': float(np.mean(values)) if len(values) > 0 else float('nan'),
            'median_error': float(np.median(values)) if len(values) > 0 else float('nan'),
        })
    return rows


def save_json(path, payload):
    """Write an indented JSON summary."""
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def main():
    """CLI entry point for proxy-vs-GT validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--hessian_grid', type=str, required=True)
    parser.add_argument('--reconstruction_mesh', type=str, required=True)
    parser.add_argument('--gt_mesh', type=str, required=True)
    parser.add_argument('--num_surface_points', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--conf', type=str, default='./confs/long_test.conf')
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--num_random_permutations', type=int, default=50)
    parser.add_argument('--auc_max_removed_fraction', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=1e-12)
    args = parser.parse_args()
    if args.num_random_permutations <= 0:
        raise ValueError('--num_random_permutations must be positive')
    if args.auc_max_removed_fraction <= 0.0 or args.auc_max_removed_fraction > 0.99:
        raise ValueError('--auc_max_removed_fraction must be in (0, 0.99]')
    if args.eps <= 0.0:
        raise ValueError('--eps must be positive')

    os.makedirs(args.output_dir, exist_ok=True)
    hessian_grid = np.load(args.hessian_grid).astype(np.float64)
    recon_mesh = load_mesh(args.reconstruction_mesh)
    gt_mesh = load_mesh(args.gt_mesh)

    points_norm = sample_surface_points(recon_mesh, args.num_surface_points, seed=args.seed)
    hessian, inside = interpolate_hessian_grid(hessian_grid, points_norm)
    scale_mat = load_scale_mat_from_conf(args.conf, args.case)
    points_gt = normalized_to_gt_frame(points_norm, scale_mat)
    error = point_to_surface_distance(points_gt, gt_mesh)

    finite = np.isfinite(hessian) & np.isfinite(error)
    if int(finite.sum()) < 2:
        raise RuntimeError('not enough finite samples for validation')
    points_norm = points_norm[finite]
    points_gt = points_gt[finite]
    hessian = hessian[finite]
    error = error[finite]

    if np.any(hessian < 0.0):
        raise RuntimeError('interpolated H values must be non-negative')
    uncertainty_score = -np.log10(hessian + args.eps)
    if not np.all(np.isfinite(uncertainty_score)):
        raise RuntimeError('non-finite U values')

    positive = hessian > 0.0
    spearman_u = spearman_payload(error, uncertainty_score)

    sampled_csv = os.path.join(args.output_dir, 'sampled_mesh_points_with_proxy.csv')
    write_sampled_points_csv(sampled_csv, points_norm, points_gt, hessian, uncertainty_score, error)

    curve_rows, curve_payload = sparsification_curve(
        error,
        uncertainty_score,
        args.seed,
        num_random_permutations=args.num_random_permutations,
    )
    auc_payload = compute_auc_metrics(
        curve_payload['fractions'],
        curve_payload['proxy_curve'],
        curve_payload['oracle_curve'],
        curve_payload['random_curves'],
        args.auc_max_removed_fraction,
    )
    write_csv(os.path.join(args.output_dir, 'sparsification_curve.csv'), curve_rows)
    plot_sparsification(os.path.join(args.output_dir, 'sparsification_curve.png'), curve_rows)

    quartile_rows = quartile_summary(error, uncertainty_score)
    write_csv(os.path.join(args.output_dir, 'quartile_error_summary.csv'), quartile_rows)

    summary = {
        'hessian_grid': args.hessian_grid,
        'reconstruction_mesh': args.reconstruction_mesh,
        'gt_mesh': args.gt_mesh,
        'num_sampled_points': int(len(error)),
        'num_requested_surface_points': int(args.num_surface_points),
        'fraction_hessian_gt_0': float(np.mean(positive)),
        'fraction_inside_hessian_bounds_before_finite_filter': float(np.mean(inside)),
        'uncertainty_score': 'U = -log10(H + eps)',
        'eps': float(args.eps),
        'spearman_error_vs_uncertainty_score': spearman_u,
        'mean_geometric_error': float(np.mean(error)),
        'median_geometric_error': float(np.median(error)),
        'uncertainty_score_quartile_error_summary': quartile_rows,
        'random_baseline': {
            'num_permutations': int(args.num_random_permutations),
            'seed': int(args.seed),
        },
        'sparsification_auc': auc_payload,
        'scale_mat_0': scale_mat.tolist(),
        'coordinate_note': (
            'Hessian interpolation is performed in normalized NeuS coordinates before '
            'mapping sampled points to the CORTO/GT frame with the NeuS scale_mat convention.'
        ),
    }
    save_json(os.path.join(args.output_dir, 'spearman_summary.json'), summary)

    print('sampled points:', len(error))
    print('fraction H > 0:', summary['fraction_hessian_gt_0'])
    print('Spearman error vs U:', spearman_u)
    print('mean geometric error:', summary['mean_geometric_error'])
    print('AUC proxy:', auc_payload['auc_proxy'])
    print('AUC random mean:', auc_payload['auc_random_mean_curve'])
    print('AUC oracle:', auc_payload['auc_oracle'])
    print('mean error by quartile:')
    for row in quartile_rows:
        print('  Q{}: {}'.format(row['quartile'], row['mean_error']))
    print('output_dir:', args.output_dir)


if __name__ == '__main__':
    main()
