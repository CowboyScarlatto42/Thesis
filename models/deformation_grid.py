import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Dense deformation grid used for BayesSDF-inspired sensitivity analysis.

The grid is not part of standard NeuS training. It is introduced only by the
uncertainty scripts to create differentiable local geometric perturbations in
the normalized NeuS coordinate system. With zero offsets, rendered RGB should
match the baseline model; gradients with respect to the offsets measure how
sensitive the render is to local geometry changes.
"""


class DenseDeformationGrid(nn.Module):
    """
    Dense trilinear deformation grid over a configured axis-aligned box.

    The learnable tensor has shape [1, 3, D, H, W]. Channels are offsets in
    world-coordinate order [dx, dy, dz]. For torch.grid_sample on a 5D tensor,
    the spatial dimensions are ordered [D, H, W], while each query coordinate is
    ordered [x, y, z]. Therefore offsets[:, :, z, y, x] is sampled by passing
    normalized coordinates grid[..., 0] = x, grid[..., 1] = y, grid[..., 2] = z.
    """

    def __init__(self, resolution=16, bound_min=(-1.0, -1.0, -1.0), bound_max=(1.0, 1.0, 1.0)):
        """Create a learnable offset grid over the given 3D bounds."""
        super().__init__()
        if isinstance(resolution, int):
            resolution = (resolution, resolution, resolution)
        if len(resolution) != 3:
            raise ValueError('resolution must be an int or a 3-tuple')

        self.resolution = tuple(int(v) for v in resolution)
        if any(v < 2 for v in self.resolution):
            raise ValueError('all grid resolution dimensions must be >= 2')

        self.offsets = nn.Parameter(torch.zeros(1, 3, *self.resolution))
        # Bounds are buffers, not trainable parameters. They move with the
        # module across devices and define the coordinate normalization used by
        # `grid_sample`.
        self.register_buffer('bound_min', torch.tensor(bound_min, dtype=torch.float32))
        self.register_buffer('bound_max', torch.tensor(bound_max, dtype=torch.float32))

    def forward(self, points):
        """
        Interpolate offsets for points with trailing shape [..., 3].

        Points outside the configured bounds receive exactly zero deformation via
        an explicit inside-bounds mask after grid_sample interpolation.
        """
        if points.shape[-1] != 3:
            raise ValueError('points must have trailing dimension 3')

        original_shape = points.shape[:-1]
        flat_points = points.reshape(-1, 3)
        bound_min = self.bound_min.to(device=flat_points.device, dtype=flat_points.dtype)
        bound_max = self.bound_max.to(device=flat_points.device, dtype=flat_points.dtype)

        inside = ((flat_points >= bound_min) & (flat_points <= bound_max)).all(dim=-1, keepdim=True)
        # Convert from NeuS coordinates to the [-1, 1] coordinates expected by
        # torch.grid_sample.
        normalized = 2.0 * (flat_points - bound_min) / (bound_max - bound_min) - 1.0
        sample_grid = normalized.reshape(1, -1, 1, 1, 3)

        sampled = F.grid_sample(
            self.offsets.to(device=flat_points.device, dtype=flat_points.dtype),
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        sampled = sampled[0, :, :, 0, 0].transpose(0, 1)
        sampled = sampled * inside.to(sampled.dtype)
        return sampled.reshape(*original_shape, 3)


def self_check(device='cpu'):
    """Run minimal interpolation and out-of-bounds checks for the grid module."""
    grid = DenseDeformationGrid(resolution=4).to(device)
    inside_points = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.25, -0.5, 0.75],
            [1.0, 1.0, 1.0],
        ],
        device=device,
    )
    zero_offsets = grid(inside_points)
    assert torch.allclose(zero_offsets, torch.zeros_like(zero_offsets)), 'zero grid produced non-zero offsets'

    constant = torch.tensor([0.1, -0.2, 0.3], device=device)
    with torch.no_grad():
        grid.offsets[:] = constant.view(1, 3, 1, 1, 1)
    constant_offsets = grid(inside_points)
    expected = constant.expand_as(constant_offsets)
    assert torch.allclose(constant_offsets, expected, atol=1e-6), 'constant grid did not interpolate constant offsets'

    outside_points = torch.tensor(
        [
            [-1.1, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 0.0, 1.1],
            [2.0, 2.0, 2.0],
        ],
        device=device,
    )
    outside_offsets = grid(outside_points)
    assert torch.allclose(outside_offsets, torch.zeros_like(outside_offsets), atol=1e-6), \
        'outside points did not receive zero deformation'

    print('DenseDeformationGrid self-check passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    self_check(device=args.device)
