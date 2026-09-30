"""Light-conditioned SDF volume branch for independent image supervision.

Distances are normalized by the fixed scene radius. CDF bin compositing is
NeuS-style; bins query actual SDF values rather than midpoint extrapolation.
The radiance head is scene-trained and is not an intrinsic-material estimator.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from sdf import sdf_ray_weights, silhouette_rays


def encode_direction(value):
    phase = value[..., None] * value.new_tensor([1., 2., 4., 8.]) * math.pi
    return torch.cat((value, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)


def encode_hints(value):
    phase = value[..., None] * value.new_tensor([1., 2., 4., 8.])
    return torch.cat((phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)


class SDFRadiance(nn.Module):
    """A positive light-conditioned point response, times explicit irradiance."""
    def __init__(self, device='cpu', detail=False, hint_encoding=False):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(94, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))
        nn.init.normal_(self.network[-1].weight, std=.001)
        nn.init.constant_(self.network[-1].bias, -2.)
        self.log_sharpness = nn.Parameter(torch.tensor(math.log(64.)))
        self.to(device)
        self.detail = None
        if detail:
            from materials.spatial_detail import SpatialDetail
            self.detail = SpatialDetail(128, device=device)
        self.hint_encoding = None
        if hint_encoding:
            self.hint_encoding = nn.Linear(24, 128, bias=False, device=device)
            nn.init.zeros_(self.hint_encoding.weight)

    def forward(self, points, normals, wi, wo, light_position):
        half = F.normalize(wi+wo, dim=-1)
        cosines = torch.cat([(normals*direction).sum(-1, keepdim=True)
                             for direction in (wi, wo, half)] +
                            [(wi*half).sum(-1, keepdim=True)], -1)
        # As in the hint-based relighting formulation, these sharp features
        # guide appearance; normals themselves still pass image gradients.
        bands = points.new_tensor([100., 1000., 10000.])
        hints = torch.log1p((1-cosines[..., 2:3].clamp(0, 1))*bands)/torch.log1p(bands)
        hints = hints.detach()
        features = torch.cat((encode_direction(points), encode_direction(wi),
                              encode_direction(wo), normals, cosines, hints,
                              light_position.expand_as(points)), -1)
        for index, layer in enumerate(self.network):
            features = layer(features)
            if index == 0 and self.detail is not None:
                features = features+self.detail(points)
            if index == 0 and self.hint_encoding is not None:
                features = features+self.hint_encoding(encode_hints(hints))
        return F.softplus(features)


@torch.no_grad()
def select_rays(rays, count, generator, peak_mask=None, peak_fraction=0.,
                context_mask=None, context_fraction=0.):
    """GT peak/context quotas, then half foreground/valid; reweighted objective.

    Unused quotas from empty pools return to the foreground/valid mixture.
    GT masks are sampling supervision and never rendering inputs.
    """
    choices = []
    budget = count
    if peak_fraction:
        peaks = rays['valid'][peak_mask.reshape(-1)[rays['valid']]]
        npeak = int(count*peak_fraction) if len(peaks) else 0
        if npeak:
            choices.append(peaks[torch.randint(len(peaks), (npeak,), device=peaks.device, generator=generator)])
            count -= npeak
    if context_fraction:
        context = rays['valid'][context_mask.reshape(-1)[rays['valid']]]
        ncontext = int(budget*context_fraction) if len(context) else 0
        if ncontext:
            choices.append(context[torch.randint(len(context), (ncontext,), device=context.device, generator=generator)])
            count -= ncontext
    foreground = rays['valid'][rays['target'][rays['valid']] > .9]
    nforeground = count//2 if len(foreground) else 0
    for pool, n in [(foreground, nforeground), (rays['valid'], count-nforeground)]:
        if n:
            choices.append(pool[torch.randint(len(pool), (n,), device=pool.device, generator=generator)])
    return torch.cat(choices)


@torch.no_grad()
def interval_edges(field, rays, indices, samples, sharpness):
    near, far = rays['near'][indices], rays['far'][indices]
    edges = near[:, None]+(far-near)[:, None]*torch.linspace(0, 1, samples+1, device=near.device, dtype=near.dtype)
    points = rays['origin']+rays['direction'][indices, None]*edges[..., None]
    values = field(points.reshape(-1, 3)).reshape_as(edges)
    weights, _ = sdf_ray_weights(values, sharpness)
    # A small uniform component keeps sampling exploratory on currently empty
    # rays. Sampling positions are detached from the field optimization.
    pdf = (weights+.01/samples)/(weights.sum(-1, keepdim=True)+.01)
    cdf = F.pad(pdf.cumsum(-1), (1, 0))
    # Equal coarse/fine counts place uniform-PDF samples at bin midpoints.
    # Half as many midpoint quantiles would duplicate coarse bin boundaries.
    n = samples
    u = ((torch.arange(n, device=near.device, dtype=near.dtype)+.5)/n).expand(len(indices), -1).contiguous()
    upper = torch.searchsorted(cdf.contiguous(), u, right=True).clamp_max(samples)
    lower = upper-1
    c0, c1 = cdf.gather(1, lower), cdf.gather(1, upper)
    z0, z1 = edges.gather(1, lower), edges.gather(1, upper)
    extra = z0+(u-c0)/(c1-c0).clamp_min(1e-8)*(z1-z0)
    return torch.cat((edges, extra), -1).sort(-1).values


def render_rays(field, radiance, sample, rays, indices, light_scale,
                samples=64, background=0., detach_field=False):
    differentiable = torch.is_grad_enabled()
    sharpness = radiance.log_sharpness.exp()
    edges = interval_edges(field, rays, indices, samples, sharpness.detach())
    direction = rays['direction'][indices]
    edge_points = rays['origin']+direction[:, None]*edges[..., None]
    with torch.set_grad_enabled(differentiable and not detach_field):
        signed = field(edge_points.reshape(-1, 3)).reshape_as(edges)
    weights, opacity = sdf_ray_weights(signed, sharpness)
    middle = .5*(edges[:, 1:]+edges[:, :-1])
    points = (rays['origin']+direction[:, None]*middle[..., None]).reshape(-1, 3)
    with torch.enable_grad():
        points = points.detach().requires_grad_()
        gradient = torch.autograd.grad(field(points).sum(), points,
                                      create_graph=differentiable and not detach_field)[0]
    if detach_field or not differentiable:
        gradient = gradient.detach()
    normals = F.normalize(gradient, dim=-1)
    light = field.normalized(sample['light_pos']).detach()
    wi = F.normalize(light-points, dim=-1)
    wo = -direction[:, None].expand(-1, middle.shape[1], -1).reshape(-1, 3)
    response = radiance(points.detach(), normals, wi.detach(), wo, light)
    distance2 = ((light-points.detach())*field.radius).square().sum(-1, keepdim=True)
    incident = sample['light_intensity'].detach()/light_scale/distance2
    color = (weights[..., None]*(response*incident).reshape(*middle.shape, 3)).sum(-2)
    color = color+background*(1-opacity[:, None])
    camera_cosine = direction @ sample['viewmat'][2, :3].detach()
    depth = (weights*middle).sum(-1)/opacity.clamp_min(1e-8)*field.radius*camera_cosine
    normal = F.normalize((weights[..., None]*normals.reshape(*middle.shape, 3)).sum(-2), dim=-1)
    return {'linear':color, 'alpha':opacity[:, None], 'depth':depth, 'normal':normal,
            'eikonal':(gradient.norm(dim=-1)-1).square().mean()}


@torch.no_grad()
def render_image(field, radiance, sample, light_scale, samples=64, background=0., chunk=512):
    rays = silhouette_rays(sample, field)
    h, w = rays['shape']
    color = field.center.new_full((h*w, 3), background)
    alpha = field.center.new_zeros((h*w, 1))
    depth = field.center.new_zeros(h*w)
    normal = field.center.new_zeros((h*w, 3))
    for indices in rays['valid'].split(chunk):
        result = render_rays(field, radiance, sample, rays, indices, light_scale,
                             samples=samples, background=background)
        color[indices], alpha[indices] = result['linear'], result['alpha']
        depth[indices], normal[indices] = result['depth'], result['normal']
    return color.reshape(h, w, 3), alpha.reshape(h, w, 1), {
        'surface_depth':depth.reshape(h, w, 1), 'surface_normals':normal.reshape(h, w, 3)}


def ray_geometry_losses(result, indices, info, alpha, sample, radius, depth_weight, normal_weight):
    """Two-way rendered depth/normal consistency, with explicit stopped targets."""
    gs_depth = info['surface_depth'].reshape(-1)[indices]
    gs_normal = F.normalize(info['surface_normals'].reshape(-1, 3)[indices], dim=-1)
    target = sample['alpha'][..., 0] > .9
    interior = F.avg_pool2d(target.float()[None, None], 3, 1, 1)[0, 0] == 1
    valid = (interior.reshape(-1)[indices] & (alpha.detach().reshape(-1)[indices] > .8)
             & (result['alpha'].detach()[:, 0] > .8) & (gs_depth.detach() > 0))
    if not valid.any():
        return {}
    gs_depth, gs_normal = gs_depth[valid], gs_normal[valid]
    sdf_depth, sdf_normal = result['depth'][valid], result['normal'][valid]
    return {
        'sdf_ray_depth_to_field':depth_weight*(sdf_depth-gs_depth.detach()).abs().mean()/radius,
        'sdf_ray_depth_to_gs':depth_weight*(gs_depth-sdf_depth.detach()).abs().mean()/radius,
        'sdf_ray_normal_to_field':normal_weight*(1-(sdf_normal*gs_normal.detach()).sum(-1)).mean(),
        'sdf_ray_normal_to_gs':normal_weight*(1-(gs_normal*sdf_normal.detach()).sum(-1)).mean(),
    }
