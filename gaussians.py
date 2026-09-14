"""Gaussian geometry initialized only from training cameras."""

import math

import torch
from torch import nn


def camera_bounds(samples):
    poses = torch.stack([s["c2w"] for s in samples])
    origins, directions = poses[:, :3, 3], -poses[:, :3, 2]
    directions = torch.nn.functional.normalize(directions, dim=-1)
    projectors = (
        torch.eye(3, device=poses.device)[None] - directions[:, :, None] * directions[:, None, :]
    )
    center = torch.linalg.solve(
        projectors.sum(0), (projectors @ origins[..., None]).sum(0)
    ).squeeze(-1)
    intrinsics = torch.stack([sample["K"] for sample in samples])
    sizes = poses.new_tensor([sample["image"].shape[:2][::-1] for sample in samples])
    angles = (sizes / (2 * intrinsics[:, (0, 1), (0, 1)])).amax(-1)
    radii = (origins - center).norm(dim=-1) * angles
    return center, radii.median().item() * 1.2


class Gaussians(nn.Module):
    def __init__(self, count, center, radius, feature_dim=32, device="cuda"):
        super().__init__()
        means = (torch.rand(count, 3, device=device) * 2 - 1) * radius
        means += center.to(device)
        quats = torch.zeros(count, 4, device=device)
        quats[:, 0] = 1
        self.params = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "scales": nn.Parameter(
                    torch.full((count, 3), math.log(radius * 1.5 / count ** (1 / 3)), device=device)
                ),
                "quats": nn.Parameter(quats),
                "opacities": nn.Parameter(torch.full((count,), -2.2, device=device)),
                "features": nn.Parameter(torch.randn(count, feature_dim, device=device) * 0.01),
                "base": nn.Parameter(torch.full((count, 3), -1.5, device=device)),
            }
        )
        self.register_buffer("center", center.to(device))
        self.radius = radius

    def optimizers(self, position_scale=None):
        position_scale = self.radius if position_scale is None else position_scale
        rates = {
            "means": 1.6e-4 * position_scale,
            "scales": 0.005,
            "quats": 0.001,
            "opacities": 0.05,
            "features": 0.0025,
            "base": 0.0025,
        }
        return {
            name: torch.optim.Adam([value], lr=rates[name], eps=1e-15)
            for name, value in self.params.items()
        }

    def raster_inputs(self):
        p = self.params
        return dict(
            means=p["means"],
            quats=p["quats"],
            scales=p["scales"].exp(),
            opacities=p["opacities"].sigmoid(),
        )

    @torch.no_grad()
    def project_geometry(self, opacity_cap=0.99, min_scale=1e-4, max_scale=0.1):
        """Project into a numerically trainable opacity/scale range."""
        if opacity_cap < 1:
            self.params["opacities"].clamp_(max=math.log(opacity_cap / (1 - opacity_cap)))
        if min_scale > 0:
            self.params["scales"].clamp_(min=math.log(self.radius * min_scale))
        if max_scale > 0:
            self.params["scales"].clamp_(max=math.log(self.radius * max_scale))
