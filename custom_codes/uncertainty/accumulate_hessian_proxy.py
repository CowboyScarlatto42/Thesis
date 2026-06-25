import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exp_runner import Runner
from models.deformation_grid import DenseDeformationGrid


"""
Estimate a geometry-oriented Hessian proxy from a single NeuS validation image.

The script freezes a trained NeuS model, attaches a dense deformation grid to
the geometry branch, renders selected foreground rays, and accumulates squared
RGB gradients with respect to the grid offsets. The resulting 3D grid stores
the raw proxy H, interpreted in a Laplace-inspired way as a local curvature /
geometry-sensitivity proxy. Larger H means that rendered RGB is more sensitive
to small geometry perturbations in that region. Lower H means a flatter local
response, a wider admissible perturbation interval, and therefore higher
geometry uncertainty. The uncertainty-oriented score U is derived later as
U = -log10(H + eps); it is not accumulated directly by this script.
"""


def set_seed(seed):
    """Set Python, NumPy, and Torch seeds for reproducible ray sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_neus(runner):
    """Freeze all trained NeuS networks so only grid-offset gradients are measured."""
    modules = [
        runner.nerf_outside,
        runner.sdf_network,
        runner.deviation_network,
        runner.color_network,
    ]
    for module in modules:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)


def choose_image_index(dataset, requested_idx):
    """Use the requested image, or choose the image with the largest foreground mask."""
    if requested_idx >= 0:
        return requested_idx

    best_idx = 0
    best_count = -1
    for idx in range(dataset.n_images):
        mask = dataset.masks[idx][..., 0]
        count = int((mask > 0.5).sum().item())
        if count > best_count:
            best_idx = idx
            best_count = count
    return best_idx


def rays_from_pixels(dataset, img_idx, pixels_y, pixels_x):
    """Build ray origins and directions for explicit pixel coordinates."""
    pixels_x = pixels_x.float()
    pixels_y = pixels_y.float()
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
    p = torch.matmul(dataset.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()
    rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
    rays_d = torch.matmul(dataset.pose_all[img_idx, None, :3, :3], rays_d[:, :, None]).squeeze()
    rays_o = dataset.pose_all[img_idx, None, :3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def candidate_rays_from_mask(dataset, img_idx, candidate_count, mask_threshold):
    """Sample candidate rays from foreground mask pixels."""
    mask = dataset.masks[img_idx][..., 0]
    coords = torch.nonzero(mask > mask_threshold, as_tuple=False)
    if len(coords) == 0:
        return None, None, None, 0

    perm = torch.randperm(len(coords))[:candidate_count]
    selected = coords[perm]
    pixels_y = selected[:, 0]
    pixels_x = selected[:, 1]
    rays_o, rays_d = rays_from_pixels(dataset, img_idx, pixels_y, pixels_x)
    return rays_o, rays_d, selected, len(coords)


def random_candidate_rays(dataset, img_idx, candidate_count):
    """Fallback candidate-ray sampler used when no foreground mask pixels exist."""
    pixels_x = torch.randint(low=0, high=dataset.W, size=[candidate_count])
    pixels_y = torch.randint(low=0, high=dataset.H, size=[candidate_count])
    coords = torch.stack([pixels_y, pixels_x], dim=-1)
    rays_o, rays_d = rays_from_pixels(dataset, img_idx, pixels_y, pixels_x)
    return rays_o, rays_d, coords


def render_batch(runner, rays_o, rays_d, deformation_grid=None):
    """Render a batch of rays with optional geometry deformation grid."""
    near, far = runner.dataset.near_far_from_sphere(rays_o.cpu(), rays_d.cpu())
    near = near.to(runner.device)
    far = far.to(runner.device)
    background_rgb = torch.ones([1, 3], device=runner.device) if runner.use_white_bkgd else None
    return runner.renderer.render(
        rays_o,
        rays_d,
        near,
        far,
        perturb_overwrite=0,
        background_rgb=background_rgb,
        cos_anneal_ratio=runner.get_cos_anneal_ratio(),
        deformation_grid=deformation_grid,
    )


def render_for_selection(runner, rays_o, rays_d):
    # The baseline selection render only needs detached opacity values. The NeuS
    # renderer still needs autograd internally for SDF normals, so temporarily
    # re-enable grad inside no_grad and detach immediately after the render.
    with torch.no_grad():
        with torch.enable_grad():
            render_out = render_batch(runner, rays_o, rays_d, deformation_grid=None)
        return {
            'weight_sum': render_out['weight_sum'].detach(),
        }


def opacity_stats(values):
    """Return min/max/mean opacity statistics for ray filtering."""
    return {
        'min': float(values.min().item()),
        'max': float(values.max().item()),
        'mean': float(values.mean().item()),
    }


def tensor_stats(values):
    """Return min/max/mean statistics for a tensor-like object."""
    return {
        'min': float(values.min().item()),
        'max': float(values.max().item()),
        'mean': float(values.mean().item()),
    }


def nonzero_stats(values, threshold):
    """Count values above a threshold and report their fraction."""
    count = int((values > threshold).sum().item())
    total = int(values.numel())
    return {
        'count': count,
        'fraction': float(count / total) if total > 0 else 0.0,
    }


def ensure_finite_tensor(name, values, ray_idx=None, channel_idx=None, pixel_coord=None):
    """Fail fast when an intermediate tensor contains NaN or Inf values."""
    if torch.isfinite(values).all():
        return

    details = [name]
    if ray_idx is not None:
        details.append('ray_idx={}'.format(ray_idx))
    if channel_idx is not None:
        details.append('channel_idx={}'.format(channel_idx))
    if pixel_coord is not None:
        details.append('pixel_yx={}'.format(tuple(int(v) for v in pixel_coord)))
    raise RuntimeError('non-finite value detected: ' + ', '.join(details))


def parse_prefix_counts(raw_value, num_rays):
    """Parse optional cumulative ray counts for intermediate proxy snapshots."""
    if raw_value is None or raw_value.strip() == '':
        return []

    prefix_counts = []
    for item in raw_value.split(','):
        item = item.strip()
        if item == '':
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError('--prefix_counts must contain comma-separated integers') from exc
        if value <= 0:
            raise ValueError('--prefix_counts values must be positive integers')
        prefix_counts.append(value)

    prefix_counts = sorted(set(prefix_counts))
    too_large = [value for value in prefix_counts if value > num_rays]
    if too_large:
        raise ValueError(
            'requested prefix counts exceed --num_rays ({}): {}'.format(num_rays, too_large)
        )
    return prefix_counts


def identify_checkpoint_path(runner):
    """Return the loaded checkpoint path when it follows the standard NeuS name."""
    candidate = os.path.join(
        runner.base_exp_dir,
        'checkpoints',
        'ckpt_{:0>6d}.pth'.format(runner.iter_step),
    )
    if os.path.isfile(candidate):
        return candidate
    return None


def save_json(path, data):
    """Write a JSON file with indentation for inspection."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_per_ray_csv(path, rows):
    """Write per-ray contribution diagnostics to CSV."""
    fieldnames = [
        'ray_order_index',
        'pixel_y',
        'pixel_x',
        'pixel_linear_index',
        'opacity',
        'total_contribution',
        'max_single_voxel_trace_contribution',
        'r_contribution',
        'g_contribution',
        'b_contribution',
        'finite',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prefix_metadata(prefix_count, components, trace):
    """Summarize an intermediate cumulative proxy grid."""
    ensure_finite_tensor('prefix_components', components)
    ensure_finite_tensor('prefix_trace', trace)
    trace_stats = tensor_stats(trace)
    trace_sum = float(trace.sum().item())
    trace_max = trace_stats['max']
    gt0 = nonzero_stats(trace, 0.0)
    gt_eps = nonzero_stats(trace, 1e-12)
    return {
        'ray_count': int(prefix_count),
        'component_shape': list(components.shape),
        'trace_shape': list(trace.shape),
        'trace_min': trace_stats['min'],
        'trace_max': trace_max,
        'trace_mean': trace_stats['mean'],
        'trace_sum': trace_sum,
        'trace_sum_per_ray': float(trace_sum / prefix_count),
        'trace_max_per_ray': float(trace_max / prefix_count),
        'cumulative_trace_sum': trace_sum,
        'cumulative_trace_sum_per_ray': float(trace_sum / prefix_count),
        'cumulative_trace_max': trace_max,
        'cumulative_trace_max_per_ray': float(trace_max / prefix_count),
        'trace_voxels_gt_0': gt0['count'],
        'trace_fraction_gt_0': gt0['fraction'],
        'trace_voxels_gt_1e-12': gt_eps['count'],
        'trace_fraction_gt_1e-12': gt_eps['fraction'],
        'finite': True,
        'finite_validation': True,
    }


def percentile_from_sorted(values, fraction):
    """Compute a percentile from an already sorted 1D array."""
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * fraction
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def per_ray_summary(rows):
    """Summarize how much each selected ray contributed to the proxy."""
    totals = np.array([row['total_contribution'] for row in rows], dtype=np.float64)
    finite_mask = np.isfinite(totals)
    finite_totals = totals[finite_mask]
    finite_count = int(finite_mask.sum())
    non_finite_count = int(len(totals) - finite_count)
    if finite_count == 0:
        raise RuntimeError('no finite per-ray contributions were computed')
    total_proxy = float(finite_totals.sum())
    if total_proxy == 0.0:
        raise RuntimeError('final total Hessian proxy contribution is zero')

    sorted_totals = np.sort(finite_totals)
    descending = np.sort(finite_totals)[::-1]
    top5_count = min(5, len(descending))
    top10_count = min(10, len(descending))
    return {
        'finite_rays': finite_count,
        'non_finite_rays': non_finite_count,
        'min': float(sorted_totals[0]),
        'median': percentile_from_sorted(sorted_totals, 0.5),
        'mean': float(finite_totals.mean()),
        'max': float(sorted_totals[-1]),
        'largest_fraction': float(descending[0] / total_proxy),
        'top5_fraction': float(descending[:top5_count].sum() / total_proxy),
        'top10_fraction': float(descending[:top10_count].sum() / total_proxy),
        'total_proxy': total_proxy,
    }


def print_top_rays(rows, limit=10):
    """Print the rays with the largest accumulated proxy contribution."""
    sorted_rows = sorted(rows, key=lambda row: row['total_contribution'], reverse=True)
    for rank, row in enumerate(sorted_rows[:min(limit, len(sorted_rows))], start=1):
        print(
            'rank: {rank} | ray_order_index: {ray_order_index} | pixel_yx: ({pixel_y}, {pixel_x}) | '
            'opacity: {opacity:.8g} | total: {total_contribution:.8g} | '
            'max_voxel: {max_single_voxel_trace_contribution:.8g} | '
            'R: {r_contribution:.8g} | G: {g_contribution:.8g} | B: {b_contribution:.8g}'.format(
                rank=rank,
                **row,
            )
        )


def main():
    """CLI entry point for single-image Hessian-proxy accumulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--img_idx', type=int, default=-1)
    parser.add_argument('--candidate_rays', type=int, default=128)
    parser.add_argument('--num_rays', type=int, default=8)
    parser.add_argument('--grid_resolution', type=int, default=16)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mask_threshold', type=float, default=0.5)
    parser.add_argument('--opacity_threshold', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prefix_counts', type=str, default='')
    args = parser.parse_args()

    start_time = time.time()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    prefix_counts = parse_prefix_counts(args.prefix_counts, args.num_rays)

    runner = Runner(args.conf, mode='validate_image', case=args.case, is_continue=True)
    freeze_neus(runner)

    img_idx = choose_image_index(runner.dataset, args.img_idx)
    candidate_count = max(args.candidate_rays, args.num_rays)
    rays_o, rays_d, pixel_coords, mask_count = candidate_rays_from_mask(
        runner.dataset,
        img_idx,
        candidate_count,
        args.mask_threshold,
    )
    source = 'mask'
    if rays_o is None:
        rays_o, rays_d, pixel_coords = random_candidate_rays(runner.dataset, img_idx, candidate_count)
        mask_count = 0
        source = 'random'

    rays_o = rays_o.to(runner.device)
    rays_d = rays_d.to(runner.device)

    baseline_candidates = render_for_selection(runner, rays_o, rays_d)
    candidate_opacity = baseline_candidates['weight_sum'].reshape(-1).detach()
    candidate_stats = opacity_stats(candidate_opacity)
    valid = torch.nonzero(candidate_opacity > args.opacity_threshold, as_tuple=False).reshape(-1)
    if len(valid) == 0:
        raise RuntimeError(
            'no candidate rays exceeded the opacity threshold; '
            'increase --candidate_rays or lower --opacity_threshold'
        )
    selected = valid[:min(args.num_rays, len(valid))]
    selected_cpu = selected.cpu()

    rays_o = rays_o[selected]
    rays_d = rays_d[selected]
    selected_pixel_coords = pixel_coords[selected_cpu].cpu()
    selected_opacity = candidate_opacity[selected].cpu()
    selected_stats = opacity_stats(selected_opacity)

    deformation_grid = DenseDeformationGrid(resolution=args.grid_resolution).to(runner.device)
    grid_param = deformation_grid.offsets
    hessian_components = torch.zeros_like(grid_param.detach()[0], dtype=torch.float32, device=runner.device)

    render_out = render_batch(runner, rays_o, rays_d, deformation_grid=deformation_grid)
    color = render_out['color_fine']
    ensure_finite_tensor('rendered_color', color)

    num_selected = color.shape[0]
    if prefix_counts and prefix_counts[-1] > num_selected:
        raise RuntimeError(
            'requested prefix count {} exceeds selected valid rays {}'.format(prefix_counts[-1], num_selected)
        )
    num_channels = color.shape[1]
    if num_channels != 3:
        raise RuntimeError('expected rendered RGB with 3 channels, got {}'.format(num_channels))
    num_scalars = num_selected * num_channels
    prefix_set = set(prefix_counts)
    prefix_stats = {}
    per_ray_rows = []
    progress_interval = 1 if num_selected <= 16 else 8

    # Compute each d RGB[ray, channel] / d grid separately. Squaring gradients
    # after summing RGB first would introduce cross terms, which is not the
    # requested trace-style Hessian proxy.
    scalar_index = 0
    for ray_idx in range(num_selected):
        pixel_coord = selected_pixel_coords[ray_idx].tolist()
        ray_components = torch.zeros_like(hessian_components)
        channel_contributions = []
        for channel_idx in range(num_channels):
            scalar_index += 1
            retain_graph = scalar_index < num_scalars
            scalar = color[ray_idx, channel_idx]
            ensure_finite_tensor(
                'rgb_scalar',
                scalar,
                ray_idx=ray_idx,
                channel_idx=channel_idx,
                pixel_coord=pixel_coord,
            )
            grad = torch.autograd.grad(
                outputs=scalar,
                inputs=grid_param,
                retain_graph=retain_graph,
                create_graph=False,
                only_inputs=True,
            )[0]
            ensure_finite_tensor(
                'grid_gradient',
                grad,
                ray_idx=ray_idx,
                channel_idx=channel_idx,
                pixel_coord=pixel_coord,
            )
            squared = grad.detach()[0].float() ** 2
            hessian_components += squared
            ray_components += squared
            channel_contributions.append(float(squared.sum().item()))

        ray_trace = ray_components.sum(dim=0)
        ray_total = float(ray_components.sum().item())
        ray_max_voxel = float(ray_trace.max().item())
        ray_values = torch.tensor(
            [ray_total, ray_max_voxel] + channel_contributions,
            dtype=torch.float32,
            device=runner.device,
        )
        ensure_finite_tensor('per_ray_contribution', ray_values, ray_idx=ray_idx, pixel_coord=pixel_coord)
        per_ray_rows.append({
            'ray_order_index': int(ray_idx),
            'pixel_y': int(pixel_coord[0]),
            'pixel_x': int(pixel_coord[1]),
            'pixel_linear_index': int(pixel_coord[0] * runner.dataset.W + pixel_coord[1]),
            'opacity': float(selected_opacity[ray_idx].item()),
            'total_contribution': ray_total,
            'max_single_voxel_trace_contribution': ray_max_voxel,
            'r_contribution': channel_contributions[0],
            'g_contribution': channel_contributions[1],
            'b_contribution': channel_contributions[2],
            'finite': True,
        })

        processed_rays = ray_idx + 1
        if processed_rays in prefix_set:
            prefix_components = hessian_components.detach().clone()
            prefix_trace = prefix_components.sum(dim=0)
            components_prefix_path = os.path.join(
                args.output_dir,
                'hessian_proxy_components_prefix_{}.npy'.format(processed_rays),
            )
            trace_prefix_path = os.path.join(
                args.output_dir,
                'hessian_proxy_trace_prefix_{}.npy'.format(processed_rays),
            )
            np.save(components_prefix_path, prefix_components.cpu().numpy())
            np.save(trace_prefix_path, prefix_trace.cpu().numpy())
            prefix_stats[str(processed_rays)] = prefix_metadata(
                processed_rays,
                prefix_components.cpu(),
                prefix_trace.cpu(),
            )

        if processed_rays == num_selected or processed_rays % progress_interval == 0:
            elapsed = time.time() - start_time
            cumulative_trace_sum = float(hessian_components.sum().item())
            print(
                'processed rays: {}/{} | elapsed: {:.2f} s | cumulative trace sum: {:.8g}'.format(
                    processed_rays,
                    num_selected,
                    elapsed,
                    cumulative_trace_sum,
                )
            )

    hessian_trace = hessian_components.sum(dim=0)
    ensure_finite_tensor('hessian_components', hessian_components)
    ensure_finite_tensor('hessian_trace', hessian_trace)
    components_cpu = hessian_components.detach().cpu()
    trace_cpu = hessian_trace.detach().cpu()

    components_path = os.path.join(args.output_dir, 'hessian_proxy_components.npy')
    trace_path = os.path.join(args.output_dir, 'hessian_proxy_trace.npy')
    metadata_path = os.path.join(args.output_dir, 'hessian_proxy_metadata.json')
    selected_rays_path = os.path.join(args.output_dir, 'selected_rays_metadata.json')
    per_ray_csv_path = os.path.join(args.output_dir, 'per_ray_hessian_contributions.csv')
    per_ray_json_path = os.path.join(args.output_dir, 'per_ray_hessian_contributions.json')

    np.save(components_path, components_cpu.numpy())
    np.save(trace_path, trace_cpu.numpy())

    trace_nonzero_gt0 = nonzero_stats(trace_cpu, 0.0)
    trace_nonzero_gt_eps = nonzero_stats(trace_cpu, 1e-12)
    elapsed_time = time.time() - start_time
    checkpoint_path = identify_checkpoint_path(runner)
    ray_summary = per_ray_summary(per_ray_rows)

    metadata = {
        'conf': args.conf,
        'case': args.case,
        'checkpoint_path': checkpoint_path,
        'image_index': int(img_idx),
        'seed': int(args.seed),
        'candidate_source': source,
        'mask_pixels_above_threshold': int(mask_count),
        'candidate_rays': int(len(candidate_opacity)),
        'candidate_valid_rays': int(len(valid)),
        'selected_rays': int(num_selected),
        'grid_resolution': int(args.grid_resolution),
        'component_shape': list(components_cpu.shape),
        'trace_shape': list(trace_cpu.shape),
        'components_stats': tensor_stats(components_cpu),
        'trace_stats': tensor_stats(trace_cpu),
        'trace_voxels_gt_0': trace_nonzero_gt0,
        'trace_voxels_gt_1e-12': trace_nonzero_gt_eps,
        'prefix_counts': prefix_counts,
        'prefix_stats': prefix_stats,
        'per_ray_contribution_summary': ray_summary,
        'candidate_opacity_stats': candidate_stats,
        'selected_opacity_stats': selected_stats,
        'elapsed_time_seconds': float(elapsed_time),
        'device': str(runner.device),
    }
    save_json(metadata_path, metadata)
    save_per_ray_csv(per_ray_csv_path, per_ray_rows)
    save_json(per_ray_json_path, per_ray_rows)

    selected_rays_metadata = {
        'image_index': int(img_idx),
        'seed': int(args.seed),
        'num_rays': int(num_selected),
        'pixel_coordinates_yx': selected_pixel_coords.tolist(),
        'pixel_linear_indices': (
            selected_pixel_coords[:, 0] * runner.dataset.W + selected_pixel_coords[:, 1]
        ).tolist(),
        'opacity': [float(v) for v in selected_opacity.tolist()],
    }
    save_json(selected_rays_path, selected_rays_metadata)

    pct_gt0 = 100.0 * trace_nonzero_gt0['fraction']
    pct_gt_eps = 100.0 * trace_nonzero_gt_eps['fraction']
    print('selected_rays:', num_selected)
    print('finite_rays:', ray_summary['finite_rays'])
    print('non_finite_rays:', ray_summary['non_finite_rays'])
    print('image_index:', img_idx)
    print('output_dir:', args.output_dir)
    print('component_grid_shape:', tuple(components_cpu.shape))
    print('trace_grid_shape:', tuple(trace_cpu.shape))
    print('trace_min:', metadata['trace_stats']['min'])
    print('trace_max:', metadata['trace_stats']['max'])
    print('trace_mean:', metadata['trace_stats']['mean'])
    print('per_ray_contribution_min:', ray_summary['min'])
    print('per_ray_contribution_median:', ray_summary['median'])
    print('per_ray_contribution_mean:', ray_summary['mean'])
    print('per_ray_contribution_max:', ray_summary['max'])
    print('largest_ray_fraction:', ray_summary['largest_fraction'])
    print('top_5_rays_fraction:', ray_summary['top5_fraction'])
    print('top_10_rays_fraction:', ray_summary['top10_fraction'])
    print('trace_voxels_gt_0:', trace_nonzero_gt0['count'], '({:.6f}%)'.format(pct_gt0))
    print('trace_voxels_gt_1e-12:', trace_nonzero_gt_eps['count'], '({:.6f}%)'.format(pct_gt_eps))
    print('top contributing rays:')
    print_top_rays(per_ray_rows, limit=10)
    print('elapsed_time_seconds:', elapsed_time)


if __name__ == '__main__':
    main()
