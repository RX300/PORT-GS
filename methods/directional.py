"""Material-conditioned directional exchange within learned spatial ports."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .base import TransportBase


class DirectionalTransport(TransportBase):
    defaults = dict(TransportBase.defaults, dir_dim=4, dir_width=32,
                    direction_basis="constant_spherical_gaussian",
                    matrix_parameterization="softplus", direction_axis_init="xyz",
                    direction_kappa_init=1.0, matrix_diagonal_init=0.25,
                    matrix_offdiagonal_init=0.001)
    cli_fields = TransportBase.cli_fields + ("dir_dim", "dir_width")

    def __init__(self, feature_dim=32, width=128, rank=512, light_scale=1.0,
                 dir_dim=4, dir_width=32, direction_basis="constant_spherical_gaussian",
                 matrix_parameterization="softplus", direction_axis_init="xyz",
                 direction_kappa_init=1.0, matrix_diagonal_init=0.25,
                 matrix_offdiagonal_init=0.001):
        if dir_dim != 4 or direction_basis != "constant_spherical_gaussian":
            raise ValueError("directional_port_v1 requires one constant and three spherical Gaussians")
        if matrix_parameterization != "softplus" or direction_axis_init != "xyz":
            raise ValueError("Unsupported directional parameterization")
        super().__init__(feature_dim, width, rank, light_scale)
        self.direction_net = nn.Sequential(
            nn.Linear(feature_dim, dir_width), nn.SiLU(),
            nn.Linear(dir_width, dir_width), nn.SiLU(),
            nn.Linear(dir_width, 12),
        )
        nn.init.normal_(self.direction_net[-1].weight, std=0.001)
        with torch.no_grad():
            bias = self.direction_net[-1].bias.view(3, 4)
            bias[:, :3].copy_(torch.eye(3))
            bias[:, 3].fill_(math.log(math.expm1(direction_kappa_init)))
        matrix = torch.full((rank, 3, dir_dim, dir_dim), matrix_offdiagonal_init)
        matrix.diagonal(dim1=-2, dim2=-1).fill_(matrix_diagonal_init)
        self.raw_C = nn.Parameter(matrix.expm1().log())

    def direction_basis(self, features, direction):
        parameters = self.direction_net(features).reshape(-1, 3, 4)
        axes = F.normalize(parameters[..., :3], dim=-1)
        kappa = F.softplus(parameters[..., 3])
        cosine = (axes * direction[:, None]).sum(-1).clamp(-1, 1)
        lobes = torch.exp(kappa * (cosine - 1))
        return torch.cat((torch.ones_like(lobes[:, :1]), lobes), dim=-1)

    def port_state(self, source_incident, log_partition, exchange_logits, mass, basis):
        # Normalize over sources in log space, then retain the RGB denominator.
        weights = (log_partition + mass.log()[:, None]).T.contiguous().softmax(-1)
        fraction = exchange_logits.sigmoid()
        denominator = (weights @ fraction).clamp_min(torch.finfo(weights.dtype).tiny)
        signal = basis[:, :, None] * (fraction * source_incident)[:, None, :]
        pooled = (weights @ signal.flatten(1)).reshape(-1, 4, 3)
        z = pooled / denominator[:, None, :]
        return torch.einsum("rcoi,ric->roc", F.softplus(self.raw_C), z)

    def receiver_partition(self, xyz):
        return self.partition(xyz)

    def exchange_radiance(self, source, receiver):
        y = self.port_state(
            source.incident, self.partition(source.xyz), self.exchange(source.features), source.mass,
            self.direction_basis(source.features, source.direction))
        h = (self.receiver_partition(receiver.xyz).exp() @ y.flatten(1)).reshape(-1, 4, 3)
        basis = self.direction_basis(receiver.features, receiver.direction)
        fraction = self.exchange(receiver.features).sigmoid()
        direct = receiver.incident * receiver.response
        return (1 - fraction) * direct + fraction * (basis[:, :, None] * h).sum(1)
