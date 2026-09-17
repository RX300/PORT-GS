"""Material-conditioned directional exchange within learned spatial ports."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from transport import Transport, direction_encoding, quadrature_mass


class DirectionalTransport(Transport):
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

    def forward(self, gaussians, receivers, eye, light_pos, light_intensity,
                source_visibility, port_active=True):
        delta = light_pos - receivers["means"]
        incident = light_intensity[None] / self.light_scale / delta.square().sum(-1, keepdim=True)
        incident = incident * receivers["visibility"][:, None]
        light_dir = F.normalize(delta, dim=-1)
        view_dir = F.normalize(eye - receivers["means"], dim=-1)
        xyz = (receivers["means"] - gaussians.center) / gaussians.radius
        inputs = torch.cat((
            receivers["features"], direction_encoding(light_dir), direction_encoding(view_dir),
            direction_encoding(F.normalize(light_dir + view_dir, dim=-1)),
            (light_dir * view_dir).sum(-1, keepdim=True), direction_encoding(xyz, bands=8),
        ), dim=-1)
        direct = incident * F.softplus(receivers["base"] + self.local(inputs))
        if not port_active:
            return direct
        source = gaussians.params
        source_delta = light_pos - source["means"]
        source_incident = (light_intensity[None] / self.light_scale
                           / source_delta.square().sum(-1, keepdim=True)
                           * source_visibility[:, None])
        source_xyz = (source["means"] - gaussians.center) / gaussians.radius
        y = self.port_state(
            source_incident, self.partition(source_xyz), self.exchange(source["features"]),
            quadrature_mass(source),
            self.direction_basis(source["features"], F.normalize(source_delta, dim=-1)),
        )
        h = (self.partition(xyz).exp() @ y.flatten(1)).reshape(-1, 4, 3)
        basis = self.direction_basis(receivers["features"], view_dir)
        fraction = self.exchange(receivers["features"]).sigmoid()
        return (1 - fraction) * direct + fraction * (basis[:, :, None] * h).sum(1)


def build_transport(config, light_scale):
    options = {name: config[name] for name in ("feature_dim", "width", "rank")}
    representation = config.get("representation", "learned_anchor_exchange")
    if representation == "learned_anchor_exchange":
        return Transport(**options, light_scale=light_scale)
    if representation != "directional_port_v1":
        raise ValueError(f"Unknown representation: {representation}")
    names = ("dir_dim", "dir_width", "direction_basis", "matrix_parameterization",
             "direction_axis_init", "direction_kappa_init", "matrix_diagonal_init",
             "matrix_offdiagonal_init")
    return DirectionalTransport(**options, light_scale=light_scale,
                                **{name: config[name] for name in names})
