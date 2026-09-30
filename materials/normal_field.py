"""Position-only detail correction for the existing bounded shading normal."""
import math

import torch
from torch import nn


class NormalResidualField(nn.Module):
    def __init__(self, center, radius):
        super().__init__()
        self.register_buffer('center', center.detach().clone())
        self.register_buffer('radius', center.new_tensor(radius))
        self.register_buffer('frequencies', 2.**torch.arange(8, device=center.device)*math.pi)
        layers = []
        for incoming in (51, 128, 128):
            layers.extend((nn.Linear(incoming, 128), nn.ReLU()))
        layers.append(nn.Linear(128, 3))
        self.network = nn.Sequential(*layers).to(center.device)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, world_points):
        points = (world_points-self.center)/self.radius
        angles = points[..., None]*self.frequencies
        encoded = torch.cat((points, angles.sin().flatten(-2), angles.cos().flatten(-2)), -1)
        return self.network(encoded)
