"""Signed light-conditioned foreground correction on fixed GS receivers."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from materials.spatial_detail import SpatialDetail
from sdf_volume import encode_direction


def spherical_gaussian_kernel(center, half, scales):
    """Spherical Gaussian with scales in radians and finite endpoint gradients."""
    cosine = (center*half).sum(-1, keepdim=True).clamp(-1, 1)
    return torch.exp((cosine-1)/scales.square())


class RadianceResidual(nn.Module):
    """Zero initial correction, scaled by explicit point-light irradiance."""

    def __init__(self, center, radius, normal_source='geometry', interaction='none',
                 angular_bank='none'):
        super().__init__()
        if interaction not in ('none', 'add', 'multiply'):
            raise ValueError(f'Unknown residual interaction: {interaction}')
        if angular_bank not in ('none', 'wide', 'narrow'):
            raise ValueError(f'Unknown residual angular bank: {angular_bank}')
        if angular_bank != 'none' and (interaction != 'multiply' or normal_source != 'geometry'):
            raise ValueError('Residual angular bank requires multiply interaction and geometry normals')
        self.register_buffer('center', center.detach().clone())
        self.register_buffer('radius', center.new_tensor(radius))
        self.normal_source = normal_source
        self.interaction = interaction
        self.angular_bank = angular_bank
        self.network = nn.Sequential(nn.Linear(94, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.to(center.device)
        self.detail = SpatialDetail(128, device=center.device)
        if interaction != 'none':
            self.interaction_spatial = nn.Linear(24, 16, bias=False, device=center.device)
            self.interaction_angular = nn.Linear(67, 16, bias=False, device=center.device)
            self.interaction_projection = nn.Linear(16, 128, bias=False, device=center.device)
        if angular_bank != 'none':
            self.center_network = nn.Sequential(nn.Linear(51, 32, device=center.device), nn.ReLU(),
                                                nn.Linear(32, 3, device=center.device))
            nn.init.zeros_(self.center_network[-1].weight)
            nn.init.zeros_(self.center_network[-1].bias)
            self.angular_projection = nn.Linear(8, 128, bias=False, device=center.device)
            lower, upper = (8., 128.) if angular_bank == 'wide' else (2., 32.)
            # One CPU construction preserves exact checkpoint scales across devices.
            scales = torch.deg2rad(torch.logspace(math.log10(lower), math.log10(upper), 8,
                                                device='cpu'))
            self.register_buffer('angular_scales', scales.to(center.device))

    def normalized(self, points):
        return (points-self.center)/self.radius

    def forward(self, points, normals, eye, light_pos, light_intensity, light_scale,
                return_stats=False):
        points, normals = points.detach(), normals.detach()
        delta = light_pos.detach()-points
        wi = F.normalize(delta, dim=-1)
        wo = F.normalize(eye.detach()-points, dim=-1)
        half = F.normalize(wi+wo, dim=-1)
        xyz = self.normalized(points)
        light = self.normalized(light_pos.detach())
        cosines = torch.cat([(normals*direction).sum(-1, keepdim=True)
                             for direction in (wi, wo, half)] +
                            [(wi*half).sum(-1, keepdim=True)], -1)
        bands = points.new_tensor([100., 1000., 10000.])
        hints = torch.log1p((1-cosines[..., 2:3].clamp(0, 1))*bands)/torch.log1p(bands)
        position_features = encode_direction(xyz)
        features = torch.cat((position_features, encode_direction(wi), encode_direction(wo),
                              normals, cosines, hints.detach(), light.expand_as(points)), -1)
        stats = {}
        if self.interaction != 'none':
            grid = self.detail.grid(xyz)
            spatial = self.interaction_spatial(grid)
            angular = self.interaction_angular(features[..., 27:])
            factors = spatial+angular if self.interaction == 'add' else spatial*angular
            interaction = self.interaction_projection(factors)
            if return_stats:
                for name, value in (('grid', grid), ('spatial', spatial),
                                    ('angular', angular), ('output', interaction)):
                    value = value.detach()
                    stats[f'interaction_{name}_abs_mean'] = value.abs().mean() if value.numel() else value.new_zeros(())
                    stats[f'interaction_{name}_square_mean'] = value.square().mean() if value.numel() else value.new_zeros(())
        if self.angular_bank != 'none':
            offset = self.center_network(torch.cat((position_features, grid), -1))
            raw_center = normals+offset
            center = F.normalize(raw_center, dim=-1)
            kernels = spherical_gaussian_kernel(center, half, self.angular_scales)
            angular_output = self.angular_projection(kernels)
            if return_stats:
                for index, values in enumerate(kernels.detach().unbind(-1)):
                    for moment, value in (('mean', values), ('square_mean', values.square()),
                                          ('zero_fraction', (values == 0).to(values.dtype))):
                        stats[f'angular_kernel_{index}_{moment}'] = value.mean() if value.numel() else value.new_zeros(())
                frozen_center = center.detach()
                original_center = F.normalize(normals, dim=-1)
                rotation = torch.rad2deg(torch.atan2(
                    torch.linalg.cross(original_center, frozen_center, dim=-1).norm(dim=-1),
                    (original_center*frozen_center).sum(-1)))
                raw_norm = raw_center.detach().norm(dim=-1)
                stats['center_raw_norm_min'] = raw_norm.min() if raw_norm.numel() else raw_norm.new_zeros(())
                for name, value in (('center_rotation_deg_mean', rotation),
                                    ('center_rotation_deg_square_mean', rotation.square()),
                                    ('center_raw_norm_below_01_fraction', (raw_norm < .1).to(raw_norm.dtype)),
                                    ('center_offset_square_mean', offset.detach().square()),
                                    ('angular_output_square_mean', angular_output.detach().square())):
                    stats[name] = value.mean() if value.numel() else value.new_zeros(())
        for index, layer in enumerate(self.network):
            features = layer(features)
            if index == 0:
                if self.interaction == 'none':
                    features = features+self.detail(xyz)
                else:
                    features = features+self.detail.projection(grid)+interaction
                if self.angular_bank != 'none':
                    features = features+angular_output
        incident = light_intensity.detach()/light_scale.detach()/delta.square().sum(-1, keepdim=True)
        output = features*incident
        return (output, stats) if return_stats else output
