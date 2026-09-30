"""Continuous surface for mutual 2DGS supervision and optional PBR normals."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


def sdf_ray_weights(signed_distances, sharpness):
    """NeuS-style logistic-CDF compositing on ordered interval boundaries.

    Positive SDF is outside. Only decreasing CDF creates opacity, so the back
    surface cannot add a second opaque layer. Values are queried at actual bin
    edges, rather than NeuS's annealed midpoint-gradient extrapolation. This
    primitive alone is not the complete NeuS renderer or a photometric branch.
    """
    log_cdf = F.logsigmoid(signed_distances * sharpness)
    log_survival = (log_cdf[..., 1:] - log_cdf[..., :-1]).clamp_max(0)
    before = F.pad(log_survival[..., :-1].cumsum(-1), (1, 0))
    weights = -log_survival.expm1() * before.exp()
    opacity = -log_survival.sum(-1).expm1()
    return weights, opacity


@torch.no_grad()
def silhouette_rays(sample, field):
    """Fixed camera rays through the SDF's normalized [-1,1]^3 domain."""
    h, w = sample['alpha'].shape[:2]
    y, x = torch.meshgrid(torch.arange(h, device=field.center.device)+.5,
                          torch.arange(w, device=field.center.device)+.5, indexing='ij')
    pixels = torch.stack((x, y, torch.ones_like(x)), -1).reshape(-1, 3)
    direction = F.normalize((pixels @ torch.linalg.inv(sample['K']).T)
                            @ sample['viewmat'][:3, :3], dim=-1)
    origin = field.normalized(sample['c2w'][:3, 3])
    t0, t1 = (-1-origin)/direction, (1-origin)/direction
    near = torch.minimum(t0, t1).amax(-1).clamp_min(0)
    far = torch.maximum(t0, t1).amin(-1)
    valid = far > near
    mask = (sample['alpha'][..., 0] > .5).float()[None, None]
    boundary = (F.max_pool2d(mask, 3, 1, 1)+F.max_pool2d(-mask, 3, 1, 1))[0, 0] > 0
    return {'origin':origin, 'direction':direction, 'near':near, 'far':far,
            'valid':valid.nonzero().flatten(),
            'boundary':(boundary.flatten() & valid).nonzero().flatten(),
            'target':sample['alpha'][..., 0].flatten(), 'shape':(h, w)}


def silhouette_logits(field, rays, indices, ray_samples=128, softness=.005):
    """Soft occupancy of a ray, from its sampled minimum signed distance.

    A closed surface is hit when some point along the ray is inside. Cameras,
    sampled positions and ray bounds stay fixed; only the SDF receives gradients.
    This is a silhouette constraint, not a radiance or full NeuS renderer.
    """
    near, far = rays['near'][indices], rays['far'][indices]
    t = near[:, None]+(far-near)[:, None]*torch.linspace(
        0, 1, ray_samples, device=near.device, dtype=near.dtype)
    points = rays['origin']+rays['direction'][indices, None]*t[..., None]
    values = field(points.reshape(-1, 3)).reshape(len(indices), ray_samples)
    return -values.amin(-1)/softness


class SurfaceSDF(nn.Module):
    """Distances and coordinates use the fixed Gaussian scene radius as unit."""
    def __init__(self, center, radius, detail=False):
        super().__init__()
        self.register_buffer("center", center.detach().clone())
        self.register_buffer("radius", center.new_tensor(radius))
        self.register_buffer("frequencies", 2. ** torch.arange(4, device=center.device) * math.pi)
        layers = []
        for incoming in (27, 64, 64):
            layers.extend((nn.Linear(incoming, 64), nn.Softplus(beta=100)))
        layers.append(nn.Linear(64, 1))
        self.network = nn.Sequential(*layers).to(center.device)
        nn.init.normal_(self.network[-1].weight, std=1e-4)
        nn.init.zeros_(self.network[-1].bias)
        self.detail = None
        if detail:
            from materials.spatial_detail import SpatialDetail
            self.detail = SpatialDetail(1, boundary_zero=True, device=center.device)

    def forward(self, points):
        # Input points are already normalized. The sphere provides an oriented
        # initial field; the residual learns the object's continuous surface.
        angles = points[..., None] * self.frequencies
        encoded = torch.cat((points, angles.sin().flatten(-2), angles.cos().flatten(-2)), -1)
        value = points.norm(dim=-1) - .6 + self.network(encoded).squeeze(-1)
        return value if self.detail is None else value+self.detail(points).squeeze(-1)

    def normalized(self, world_points):
        return (world_points - self.center) / self.radius

    def shading_normals(self, world_points, eye):
        """Differentiate the field in training; compute the same normals at inference."""
        differentiable = torch.is_grad_enabled()
        with torch.enable_grad():
            points = self.normalized(world_points).requires_grad_()
            gradient = torch.autograd.grad(self(points).sum(), points,
                                           create_graph=differentiable)[0]
            normals = F.normalize(gradient, dim=-1)
        toward_camera = eye-world_points.detach()
        sign = torch.where((normals.detach()*toward_camera).sum(-1, keepdim=True) >= 0, 1., -1.)
        return normals*sign

    def fit_losses(self, points, normals, toward_camera):
        """2DGS -> SDF: no gradients to the geometry providing observations."""
        points, normals, toward_camera = points.detach(), normals.detach(), toward_camera.detach()
        points = points.requires_grad_()
        value = self(points)
        gradient = torch.autograd.grad(value.sum(), points, create_graph=True)[0]
        offset = torch.rand_like(value) * .015 + .005
        outside = self(points + offset[:, None] * normals)
        inside = self(points - offset[:, None] * normals)
        # Only a narrow signed band is trusted behind the observed surface.
        signed = .5 * ((outside-offset).abs().mean() + (inside+offset).abs().mean())
        front = points + (torch.rand_like(value) * .2 + .05)[:, None] * toward_camera
        free = F.relu(.01-self(front)).mean()
        random_points = torch.rand_like(points) * 2 - 1
        probes = torch.cat((points.detach() + offset[:, None]*normals, random_points), 0).requires_grad_()
        probe_gradient = torch.autograd.grad(self(probes).sum(), probes, create_graph=True)[0]
        return {
            "sdf_fit": value.abs().mean() + signed,
            "sdf_orientation": .05 * (1-(F.normalize(gradient, dim=-1)*normals).sum(-1)).mean(),
            "sdf_eikonal": .1 * (probe_gradient.norm(dim=-1)-1).square().mean(),
            "sdf_free_space": .1 * free,
        }

    def geometry_losses(self, points, normals, distance_weight, normal_weight):
        """SDF -> 2DGS: freeze the field, retain position/normal gradients."""
        weights = {name: value.detach() for name, value in self.named_parameters()}
        value = functional_call(self, weights, (points,))
        gradient = torch.autograd.grad(value.sum(), points, retain_graph=True)[0].detach()
        target = F.normalize(gradient, dim=-1)
        return {
            "sdf_surface": distance_weight * value.abs().mean(),
            "sdf_normal": normal_weight * (1-(normals*target).sum(-1)).mean(),
        }

    def primitive_loss(self, means):
        """Pull primitive centers toward the frozen zero set in scene-radius units."""
        weights = {name: value.detach() for name, value in self.named_parameters()}
        return functional_call(self, weights, (self.normalized(means),)).abs().mean()


@torch.no_grad()
def visible_primitives(means, opacity, info, alpha, sample, radius):
    """Approximate visibility: opaque centers near the rendered front surface."""
    camera = means @ sample['viewmat'][:3, :3].T + sample['viewmat'][:3, 3]
    indices = ((opacity > .5) & (camera[:, 2] > 0)).nonzero().flatten()
    camera = camera[indices]
    pixels = camera @ sample['K'].T
    uv = pixels[:, :2] / pixels[:, 2:]
    h, w = alpha.shape[:2]
    grid = (uv / uv.new_tensor([w, h])*2-1)[None, None]
    valid = (alpha[..., 0] > .8) & (info['surface_depth'][..., 0] > 0)
    if sample['alpha'] is not None:
        valid &= sample['alpha'][..., 0] > .9
    interior = F.avg_pool2d(valid.float()[None, None], 3, 1, 1)
    foreground = F.grid_sample(interior, grid, align_corners=False)[0, 0, 0] > .999
    depth = F.grid_sample(info['surface_depth'].permute(2, 0, 1)[None], grid,
                          align_corners=False)[0, 0, 0]
    near_surface = (camera[:, 2]-depth).abs() < .05*radius
    return indices[foreground & near_surface]


def sample_surface(info, alpha, sample, field, count, generator):
    """Select foreground-interior samples; reconstruct world points from camera Z."""
    depth = info["surface_depth"][..., 0]
    valid = (alpha[..., 0].detach() > .8) & (depth.detach() > 0)
    if sample["alpha"] is not None:
        valid &= sample["alpha"][..., 0] > .9
    valid = F.avg_pool2d(valid.float()[None, None], 3, 1, 1)[0, 0] == 1
    indices = valid.nonzero()
    if len(indices) == 0:
        return None
    indices = indices[torch.randint(len(indices), (count,), device=indices.device, generator=generator)]
    y, x = indices.unbind(-1)
    pixels = torch.stack((x+.5, y+.5, torch.ones_like(x)), -1).to(depth.dtype)
    camera = (pixels @ torch.linalg.inv(sample["K"]).T) * depth[y, x, None]
    world = (camera-sample["viewmat"][:3, 3]) @ sample["viewmat"][:3, :3]
    toward = F.normalize(sample["c2w"][:3, 3]-world.detach(), dim=-1)
    normals = F.normalize(info["surface_normals"][y, x], dim=-1)
    sign = torch.where((normals.detach()*toward).sum(-1, keepdim=True) >= 0, 1., -1.)
    return field.normalized(world), normals*sign, toward
