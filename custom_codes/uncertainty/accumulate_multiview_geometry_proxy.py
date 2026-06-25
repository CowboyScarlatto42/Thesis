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
Accumulate a geometry-oriented Hessian proxy from multiple NeuS images.

This is the multi-view version of the single-image proxy estimator. It samples a
fixed number of foreground rays from each requested image, accumulates squared
RGB gradients with respect to a dense deformation grid, and stores both the raw
scalar grid H and the per-axis component grid. H is interpreted in a
Laplace-inspired way as a local curvature / geometry-sensitivity proxy: lower
values indicate a flatter local response, a wider admissible perturbation
interval, and therefore higher geometry uncertainty. The derived score
U = -log10(H + eps) is computed later by visualization/validation scripts.
"""


def set_seed(seed):
    """Set Python, NumPy, and Torch seeds for deterministic sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_image_indices(raw_value):
    """Parse a comma-separated list of image indices."""
    indices = []
    for item in raw_value.split(','):
        item = item.strip()
        if item:
            indices.append(int(item))
    if len(indices) == 0:
        raise ValueError('--image_indices must contain at least one index')
    return indices


def orbit_for_image(img_idx):
    """Assign dataset-specific orbit labels used in summary metadata."""
    if 0 <= img_idx <= 60:
        return 'orbit_1'
    if 61 <= img_idx <= 127:
        return 'orbit_2'
    return 'unknown'


def freeze_neus(runner):
    """Freeze trained NeuS modules before measuring grid-offset gradients."""
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


def identify_checkpoint_path(runner):
    """Return the standard checkpoint path if it exists for the loaded runner."""
    candidate = os.path.join(
        runner.base_exp_dir,
        'checkpoints',
        'ckpt_{:0>6d}.pth'.format(runner.iter_step),
    )
    if os.path.isfile(candidate):
        return candidate
    return None


def rays_from_pixels(dataset, img_idx, pixels_y, pixels_x):
    """Construct camera rays for explicit pixel coordinates."""
    pixels_x = pixels_x.float()
    pixels_y = pixels_y.float()
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
    p = torch.matmul(dataset.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()
    rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
    rays_d = torch.matmul(dataset.pose_all[img_idx, None, :3, :3], rays_d[:, :, None]).squeeze()
    rays_o = dataset.pose_all[img_idx, None, :3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def select_foreground_pixels(dataset, img_idx, rays_per_image, rng):
    """Randomly select foreground pixels from one image mask."""
    mask = dataset.masks[img_idx][..., 0].detach().cpu().numpy()
    coords = np.argwhere(mask > 0.5)
    if len(coords) < rays_per_image:
        raise RuntimeError(
            'image {} has {} foreground pixels, fewer than rays_per_image={}'.format(
                img_idx, len(coords), rays_per_image
            )
        )
    selected_ids = rng.choice(len(coords), size=rays_per_image, replace=False)
    selected = coords[selected_ids]
    pixels_y = torch.from_numpy(selected[:, 0]).long()
    pixels_x = torch.from_numpy(selected[:, 1]).long()
    rays_o, rays_d = rays_from_pixels(dataset, img_idx, pixels_y, pixels_x)
    return rays_o, rays_d, selected, len(coords)


def render_geometry_only(runner, rays_o, rays_d, deformation_grid):
    """Render rays with the deformation grid attached to the geometry branch."""
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


def ensure_finite(name, values, img_idx=None, ray_idx=None, channel_idx=None, pixel_yx=None):
    """Raise an informative error if a tensor contains non-finite values."""
    if torch.isfinite(values).all():
        return
    details = [name]
    if img_idx is not None:
        details.append('image_index={}'.format(img_idx))
    if ray_idx is not None:
        details.append('ray_idx={}'.format(ray_idx))
    if channel_idx is not None:
        details.append('channel_idx={}'.format(channel_idx))
    if pixel_yx is not None:
        details.append('pixel_yx={}'.format(tuple(int(v) for v in pixel_yx)))
    raise RuntimeError('non-finite value detected: ' + ', '.join(details))


def array_stats(values):
    """Return min/max/mean/sum statistics for a tensor or array."""
    return {
        'min': float(values.min().item()),
        'max': float(values.max().item()),
        'mean': float(values.mean().item()),
        'sum': float(values.sum().item()),
    }


def nonzero_stats(values, threshold):
    """Count values above a threshold and report their fraction."""
    count = int((values > threshold).sum().item())
    total = int(values.numel())
    return {
        'count': count,
        'fraction': float(count / total) if total > 0 else 0.0,
    }


def save_json(path, payload):
    """Write an indented JSON file."""
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def save_per_image_stats(path, rows):
    """Write per-image accumulation statistics to CSV."""
    fieldnames = [
        'image_index',
        'orbit',
        'foreground_pixels',
        'selected_rays',
        'image_trace_sum',
        'cumulative_trace_sum',
        'elapsed_time_seconds',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    """CLI entry point for multi-view geometry-proxy accumulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--image_indices', type=str, required=True)
    parser.add_argument('--rays_per_image', type=int, default=64)
    parser.add_argument('--grid_resolution', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    start_time = time.time()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    image_indices = parse_image_indices(args.image_indices)
    runner = Runner(args.conf, mode='validate_image', case=args.case, is_continue=True)
    freeze_neus(runner)

    deformation_grid = DenseDeformationGrid(resolution=args.grid_resolution).to(runner.device)
    grid_param = deformation_grid.offsets
    hessian_components = torch.zeros_like(grid_param.detach()[0], dtype=torch.float32, device=runner.device)

    selected_metadata = []
    per_image_rows = []
    processed_rays = 0

    for image_order, img_idx in enumerate(image_indices, start=1):
        rays_o, rays_d, selected_pixels, foreground_count = select_foreground_pixels(
            runner.dataset,
            img_idx,
            args.rays_per_image,
            rng,
        )
        rays_o = rays_o.to(runner.device)
        rays_d = rays_d.to(runner.device)

        render_out = render_geometry_only(runner, rays_o, rays_d, deformation_grid)
        color = render_out['color_fine']
        ensure_finite('rendered_color', color, img_idx=img_idx)
        if color.shape != (args.rays_per_image, 3):
            raise RuntimeError('unexpected color shape for image {}: {}'.format(img_idx, tuple(color.shape)))

        image_components = torch.zeros_like(hessian_components)
        num_scalars = args.rays_per_image * 3
        scalar_index = 0
        for ray_idx in range(args.rays_per_image):
            pixel_yx = selected_pixels[ray_idx].tolist()
            for channel_idx in range(3):
                scalar_index += 1
                retain_graph = scalar_index < num_scalars
                scalar = color[ray_idx, channel_idx]
                ensure_finite('rgb_scalar', scalar, img_idx=img_idx, ray_idx=ray_idx,
                              channel_idx=channel_idx, pixel_yx=pixel_yx)
                grad = torch.autograd.grad(
                    outputs=scalar,
                    inputs=grid_param,
                    retain_graph=retain_graph,
                    create_graph=False,
                    only_inputs=True,
                )[0]
                ensure_finite('grid_gradient', grad, img_idx=img_idx, ray_idx=ray_idx,
                              channel_idx=channel_idx, pixel_yx=pixel_yx)
                squared = grad.detach()[0].float() ** 2
                hessian_components += squared
                image_components += squared

        processed_rays += args.rays_per_image
        image_trace_sum = float(image_components.sum().item())
        cumulative_trace_sum = float(hessian_components.sum().item())
        elapsed_time = time.time() - start_time
        orbit = orbit_for_image(img_idx)

        selected_metadata.append({
            'image_index': int(img_idx),
            'orbit': orbit,
            'foreground_pixels': int(foreground_count),
            'selected_pixels_yx': selected_pixels.astype(int).tolist(),
            'selected_pixel_linear_indices': (
                selected_pixels[:, 0] * runner.dataset.W + selected_pixels[:, 1]
            ).astype(int).tolist(),
        })
        per_image_rows.append({
            'image_index': int(img_idx),
            'orbit': orbit,
            'foreground_pixels': int(foreground_count),
            'selected_rays': int(args.rays_per_image),
            'image_trace_sum': image_trace_sum,
            'cumulative_trace_sum': cumulative_trace_sum,
            'elapsed_time_seconds': elapsed_time,
        })
        print(
            'image_index: {} | processed images: {}/{} | processed rays: {} | '
            'cumulative trace sum: {:.8g} | elapsed: {:.2f} s'.format(
                img_idx,
                image_order,
                len(image_indices),
                processed_rays,
                cumulative_trace_sum,
                elapsed_time,
            )
        )

    hessian_raw = hessian_components.sum(dim=0)
    ensure_finite('hessian_components', hessian_components)
    ensure_finite('hessian_raw', hessian_raw)
    if float(hessian_raw.sum().item()) == 0.0:
        raise RuntimeError('final hessian geometry grid is entirely zero')

    components_cpu = hessian_components.detach().cpu()
    raw_cpu = hessian_raw.detach().cpu()
    np.save(os.path.join(args.output_dir, 'hessian_geometry_components.npy'), components_cpu.numpy())
    np.save(os.path.join(args.output_dir, 'hessian_geometry_raw.npy'), raw_cpu.numpy())
    save_per_image_stats(os.path.join(args.output_dir, 'per_image_stats.csv'), per_image_rows)

    orbit_assignment = {str(idx): orbit_for_image(idx) for idx in image_indices}
    metadata = {
        'config': args.conf,
        'case': args.case,
        'checkpoint_path': identify_checkpoint_path(runner),
        'image_indices': image_indices,
        'orbit_assignment': orbit_assignment,
        'rays_per_image': int(args.rays_per_image),
        'total_selected_rays': int(processed_rays),
        'grid_resolution': int(args.grid_resolution),
        'grid_bounds': [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        'seed': int(args.seed),
        'raw_grid_stats': array_stats(raw_cpu),
        'voxels_gt_0': nonzero_stats(raw_cpu, 0.0),
        'voxels_gt_1e-12': nonzero_stats(raw_cpu, 1e-12),
        'elapsed_time_seconds': float(time.time() - start_time),
        'device': str(runner.device),
        'finite_value_validation': True,
    }
    save_json(os.path.join(args.output_dir, 'multiview_geometry_proxy_metadata.json'), metadata)
    save_json(os.path.join(args.output_dir, 'selected_rays_metadata.json'), {
        'seed': int(args.seed),
        'rays_per_image': int(args.rays_per_image),
        'images': selected_metadata,
    })

    print('saved output_dir:', args.output_dir)
    print('raw grid sum:', metadata['raw_grid_stats']['sum'])
    print('raw grid max:', metadata['raw_grid_stats']['max'])


if __name__ == '__main__':
    main()
