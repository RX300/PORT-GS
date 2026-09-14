"""Material-conditioned conservative exchange of point-light irradiance."""

import torch
import tinycudann as tcnn
from torch import nn
from torch.nn import functional as F


def direction_encoding(direction, bands=4):
    frequency = 2 ** torch.arange(bands, device=direction.device, dtype=direction.dtype)
    phase = direction[..., None] * frequency * torch.pi
    return torch.cat((direction, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)


def quadrature_mass(params):
    """Normalized opacity-weighted mean projected-area surrogate.

    This discretizes an exchange measure, rather than a measured surface area.
    Detaching the measure keeps its normalization outside geometry optimization.
    """
    scales = params["scales"].exp()
    area = scales[:, 0] * scales[:, 1] + scales[:, 1] * scales[:, 2] + scales[:, 2] * scales[:, 0]
    mass = (params["opacities"].sigmoid() * area).detach()
    return mass / mass.sum()


def exchange_irradiance(
    source_incident, source_partition, source_exchange_logits, mass,
    incident, log_partition, exchange_logits,
):
    """Exchange Gaussian-source light with continuous material receivers.

    Source-node queries recover the mass-conservative discrete operator.
    Arbitrary receiver locations query the same source integral.
    """
    fraction = source_exchange_logits.sigmoid()
    source_logits = (source_partition + mass.log()[:, None]).T.contiguous()
    source_weights = source_logits.softmax(dim=-1)
    # Factor the channel-dependent normalization into two small port matrices.
    # This computes the same m*a*f average without an N x R x RGB tensor.
    pooled = (source_weights @ (fraction * source_incident)) / (source_weights @ fraction)
    received = log_partition.exp() @ pooled
    fraction = exchange_logits.sigmoid()
    return (1 - fraction) * incident + fraction * received


class Transport(nn.Module):
    """A continuous receiver field over Gaussian-source irradiance exchange.

    The exchange operator conserves irradiance in its discrete measure. The
    learned angular response and approximate Gaussian visibility are separate
    modeling assumptions; the complete renderer has no energy-conservation claim.
    """

    def __init__(self, feature_dim=32, width=128, rank=512, light_scale=1.0, *, hash_encoding, seed=0):
        super().__init__()
        self.register_buffer("light_scale", torch.tensor(float(light_scale)))
        self.local = nn.Sequential(
            nn.Linear(feature_dim + 82 + 51, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 3),
        )
        nn.init.normal_(self.local[-1].weight, std=0.001)
        nn.init.zeros_(self.local[-1].bias)
        self.exchange = nn.Linear(feature_dim, 3)
        nn.init.zeros_(self.exchange.weight)
        nn.init.constant_(self.exchange.bias, -2.0)
        self.spatial_encoding = tcnn.Encoding(
            n_input_dims=3, encoding_config=hash_encoding, seed=seed, dtype=torch.float32,
        )
        self.partition_head = nn.Linear(self.spatial_encoding.n_output_dims, rank)

    def partition(self, xyz):
        # Map the camera-derived initialization cube [-1, 1]^3 to grid coordinates.
        # Do not clamp positions: geometry needs gradients through the native encoding.
        encoded = self.spatial_encoding((xyz + 1) * 0.5)
        return F.log_softmax(self.partition_head(encoded), dim=-1)

    def forward(
        self, gaussians, receivers, eye, light_pos, light_intensity, source_visibility,
        port_active=True,
    ):
        delta = light_pos - receivers["means"]
        incident = light_intensity[None] / self.light_scale / delta.square().sum(-1, keepdim=True)
        incident = incident * receivers["visibility"][:, None]
        light_dir = F.normalize(delta, dim=-1)
        view_dir = F.normalize(eye - receivers["means"], dim=-1)
        xyz = (receivers["means"] - gaussians.center) / gaussians.radius
        material_inputs = torch.cat(
            (
                receivers["features"], direction_encoding(light_dir), direction_encoding(view_dir),
                direction_encoding(F.normalize(light_dir + view_dir, dim=-1)),
                (light_dir * view_dir).sum(-1, keepdim=True),
                direction_encoding(xyz, bands=8),
            ), dim=-1,
        )
        response = F.softplus(receivers["base"] + self.local(material_inputs))
        if port_active:
            source = gaussians.params
            source_delta = light_pos - source["means"]
            source_incident = (
                light_intensity[None] / self.light_scale
                / source_delta.square().sum(-1, keepdim=True) * source_visibility[:, None]
            )
            source_xyz = (source["means"] - gaussians.center) / gaussians.radius
            incident = exchange_irradiance(
                source_incident, self.partition(source_xyz), self.exchange(source["features"]),
                quadrature_mass(source),
                incident, self.partition(xyz), self.exchange(receivers["features"]),
            )
        return response * incident
