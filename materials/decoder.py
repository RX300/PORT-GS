"""Six-dimensional neural BRDF prior adapted from Yu et al., SIGGRAPH 2026.

The evaluation and directional-albedo MLPs each have four 64-wide hidden
layers. Direction coordinates are isotropic invariants; the log half-angle
coordinate is our adaptation for resolving narrow lobes with a small MLP.
No position, illumination intensity, object identity or image enters a decoder.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, outputs):
    layers = []
    for _ in range(4):
        layers.extend((nn.Linear(inputs, 64), nn.ReLU()))
        inputs = 64
    layers.append(nn.Linear(64, outputs))
    return nn.Sequential(*layers)


def angular_coordinates(wi, wo, normal=None):
    half = F.normalize(wi + wo, dim=-1)
    if normal is None:
        mu_i, mu_o, mu_h = wi[..., 2:3], wo[..., 2:3], half[..., 2:3]
    else:
        mu_i = (normal * wi).sum(-1, keepdim=True)
        mu_o = (normal * wo).sum(-1, keepdim=True)
        mu_h = (normal * half).sum(-1, keepdim=True)
    mu_i, mu_o, mu_h = (value.clamp(0., 1.) for value in (mu_i, mu_o, mu_h))
    mu_d = (wi * half).sum(-1, keepdim=True).clamp(0., 1.)
    # log1p retains detail around the half-vector pole without acos gradients.
    narrow = torch.log1p((1 - mu_h) * 1e5) / math.log1p(1e5)
    return torch.cat((mu_i, mu_o, mu_h, mu_d, narrow), dim=-1)


class MaterialEncoder(nn.Module):
    """Used only for procedural pretraining and the initial material code."""

    def __init__(self):
        super().__init__()
        self.network = mlp(22, 6)

    def forward(self, parameters):
        normalized = parameters.clone()
        # Resolve the three decades of GGX roughness before compression.
        normalized[..., [7, 8, 21]] = parameters[..., [7, 8, 21]].log() / math.log(.001)
        return self.network(normalized).sigmoid()


class MaterialDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.brdf = mlp(6 + 5, 3)
        self.albedo = mlp(6 + 1, 4)
        nn.init.normal_(self.brdf[-1].weight, std=.01)
        nn.init.constant_(self.brdf[-1].bias, math.log(.05))

    def forward(self, latent, wi, wo, normal=None):
        angles = angular_coordinates(wi, wo, normal)
        non_diffuse = self.brdf(torch.cat((latent, angles), dim=-1)).exp()
        # This network consumes incident angle only: T must not depend on wo.
        energy = self.albedo(torch.cat((latent, angles[..., :1]), dim=-1)).sigmoid()
        return non_diffuse, energy[..., :3], energy[..., 3:]
