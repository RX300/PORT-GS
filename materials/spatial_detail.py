"""Explicit PyTorch multiresolution grid residuals with higher-order gradients.

The installed tiny-cuda-nn binary requires a newer host glibc. This module is
the chosen implementation, not a runtime fallback or an environment change.
Coordinates use the SDF's normalized [-1,1]^3 domain.
"""

import math

import torch
from torch import nn


class MultiResolutionGrid(nn.Module):
    def __init__(self, levels=12, base_resolution=16, max_resolution=256,
                 log2_table_size=17, device='cpu'):
        super().__init__()
        self.levels = levels
        self.table_size = 2**log2_table_size
        self.output_dim = 2*levels
        resolutions = [round(base_resolution*math.exp(math.log(max_resolution/base_resolution)*i/max(1, levels-1)))
                       for i in range(levels)]
        sizes = [min(r**3, self.table_size) for r in resolutions]
        offsets = [sum(sizes[:i]) for i in range(levels)]
        self.register_buffer('resolutions', torch.tensor(resolutions, device=device))
        self.register_buffer('offsets', torch.tensor(offsets, device=device))
        self.register_buffer('dense', torch.tensor([r**3 <= self.table_size for r in resolutions], device=device))
        self.register_buffer('corners', torch.cartesian_prod(*[torch.tensor([0, 1], device=device) for _ in range(3)]))
        self.table = nn.Parameter(torch.empty(sum(sizes), 2, device=device))
        nn.init.uniform_(self.table, -1e-4, 1e-4)

    def forward(self, points):
        shape = points.shape[:-1]
        unit = ((points.reshape(-1, 3)+1)*.5).clamp(0, 1)
        coordinate = unit[:, None]*(self.resolutions-1)[None, :, None]
        lower = coordinate.floor().long()
        fraction = coordinate-lower
        vertex = torch.minimum(lower[:, :, None]+self.corners[None, None],
                               (self.resolutions-1)[None, :, None, None])
        x, y, z = vertex.unbind(-1)
        linear = (x*self.resolutions[None, :, None]+y)*self.resolutions[None, :, None]+z
        hashed = (x ^ (y*2654435761) ^ (z*805459861)) % self.table_size
        index = torch.where(self.dense[None, :, None], linear, hashed)+self.offsets[None, :, None]
        factor = torch.where(self.corners[None, None].bool(), fraction[:, :, None], 1-fraction[:, :, None])
        weight = factor[..., 0]*factor[..., 1]*factor[..., 2]
        features = (self.table[index]*weight[..., None]).sum(-2)
        return features.reshape(*shape, self.output_dim)


class SpatialDetail(nn.Module):
    """Zero-initialized residual; optional C1-zero envelope at the SDF boundary."""
    def __init__(self, outputs, boundary_zero=False, device='cpu', **grid_options):
        super().__init__()
        self.grid = MultiResolutionGrid(device=device, **grid_options)
        self.projection = nn.Linear(self.grid.output_dim, outputs, bias=False, device=device)
        nn.init.zeros_(self.projection.weight)
        self.boundary_zero = boundary_zero

    def forward(self, points):
        result = self.projection(self.grid(points))
        if self.boundary_zero:
            axis = (1-points.square()).clamp_min(0).square()
            result = result*axis[..., :1]*axis[..., 1:2]*axis[..., 2:3]
        return result
