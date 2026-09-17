"""Shared point-light shading and the transport method interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SourceLight:
    xyz: torch.Tensor
    features: torch.Tensor
    incident: torch.Tensor
    direction: torch.Tensor
    mass: torch.Tensor


@dataclass
class ReceiverLight:
    xyz: torch.Tensor
    features: torch.Tensor
    incident: torch.Tensor
    direction: torch.Tensor
    response: torch.Tensor


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


def spatial_partition(xyz, centers, log_width):
    distance2 = (xyz[:, None] - centers[None]).square().sum(-1)
    return F.log_softmax(-distance2 / (2 * log_width.exp().square()), dim=-1)


class TransportBase(nn.Module, ABC):
    """Minimal contract for methods, including implementations without ports.

    Subclasses declare constructor defaults and implement linear pixel shading.
    Shared scene geometry, observations and optimizers remain outside this class.
    """
    defaults = dict(feature_dim=32)
    cli_fields = ("feature_dim",)

    def __init__(self, light_scale=1.0):
        super().__init__()
        self.register_buffer("light_scale", torch.tensor(float(light_scale)))

    @abstractmethod
    def forward(self, gaussians, receivers, eye, light_pos, light_intensity,
                source_visibility, port_active=True):
        """Return [M, 3] linear foreground radiance for the pixel receivers."""


class PortTransport(TransportBase):
    """Return linear foreground RGB at the renderer's pixel receivers.

    Subclasses implement exchange_radiance(source, receiver). The shared forward
    owns point-light illumination and direct shading. No image-space observation
    transform or alpha composition belongs in a transport method.
    """

    defaults = dict(TransportBase.defaults, width=128, rank=512)
    cli_fields = TransportBase.cli_fields + ("width", "rank")

    def __init__(self, feature_dim=32, width=128, rank=512, light_scale=1.0):
        super().__init__(light_scale)
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
        self.anchor_centers = nn.Parameter(torch.empty(rank, 3).uniform_(-0.5, 0.5))
        widths = torch.full((rank,), 0.5)
        widths[: rank // 2] = 0.15
        self.log_width = nn.Parameter(widths.log())

    def partition(self, xyz):
        return spatial_partition(xyz, self.anchor_centers, self.log_width)

    def material_directions(self, features, light_dir, view_dir):
        return light_dir, view_dir

    def direct_response(self, receivers, xyz, light_dir, view_dir):
        light_dir, view_dir = self.material_directions(receivers["features"], light_dir, view_dir)
        inputs = torch.cat((
            receivers["features"], direction_encoding(light_dir), direction_encoding(view_dir),
            direction_encoding(F.normalize(light_dir + view_dir, dim=-1)),
            (light_dir * view_dir).sum(-1, keepdim=True), direction_encoding(xyz, bands=8),
        ), dim=-1)
        return F.softplus(receivers["base"] + self.local(inputs))

    def point_light(self, points, light_pos, intensity, visibility):
        delta = light_pos - points
        incident = intensity[None] / self.light_scale / delta.square().sum(-1, keepdim=True)
        return incident * visibility[:, None], F.normalize(delta, dim=-1)

    def forward(self, gaussians, receivers, eye, light_pos, light_intensity,
                source_visibility, port_active=True):
        incident, light_dir = self.point_light(
            receivers["means"], light_pos, light_intensity, receivers["visibility"])
        view_dir = F.normalize(eye - receivers["means"], dim=-1)
        xyz = (receivers["means"] - gaussians.center) / gaussians.radius
        response = self.direct_response(receivers, xyz, light_dir, view_dir)
        if not port_active:
            return incident * response
        params = gaussians.params
        source_incident, source_direction = self.point_light(
            params["means"], light_pos, light_intensity, source_visibility)
        source = SourceLight(
            (params["means"] - gaussians.center) / gaussians.radius,
            params["features"], source_incident, source_direction, quadrature_mass(params))
        receiver = ReceiverLight(xyz, receivers["features"], incident, view_dir, response)
        return self.exchange_radiance(source, receiver)

    @abstractmethod
    def exchange_radiance(self, source: SourceLight, receiver: ReceiverLight):
        """Combine direct and nonlocal light, returning [receivers, 3] linear RGB."""
