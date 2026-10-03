"""LiSA: light-space neural transport atlases for point-light relighting.

A point light delivers the same flux I*dw through every solid-angle element, so
splatting the Gaussians from the light with ordinary alpha compositing deposits
flux I*dw*T_i*alpha_i on Gaussian i without any normal. The light pass stores
learned per-Gaussian flux features and the first two depth moments of that
deposition; a Gaussian pyramid of the map provides world-scale kernels. Every
camera receiver reads the pyramid at its light-space projection and shades

    L = E * (V * rho + sum_k sum_c W[k, :, c] * Phi[k, c]),   E = I / r^2,

with rho a local neural material, V a moment shadow test with a learned
residual, and W light-space transfer kernels that are linear in the gathered
flux (translucency, subsurface bleeding and nearby bounce light). All gather
inputs are light-relative, so the operator moves with the light.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from renderer import _rasterize, covariance_normals, light_view
from .base import TransportBase, direction_encoding


def _mlp(inputs, width, outputs, hidden):
    layers = []
    for _ in range(hidden):
        layers.extend((nn.Linear(inputs, width), nn.SiLU()))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


def _binomial_blur(x):
    """Separable [1 4 6 4 1]/16 blur; zero padding keeps outside-frustum texels empty."""
    channels = x.shape[1]
    kernel = x.new_tensor((1., 4., 6., 4., 1.)) / 16
    x = F.conv2d(x, kernel.view(1, 1, 1, 5).expand(channels, 1, 1, 5), padding=(0, 2), groups=channels)
    return F.conv2d(x, kernel.view(1, 1, 5, 1).expand(channels, 1, 5, 1), padding=(2, 0), groups=channels)


def _normal_cdf(x):
    return 0.5 * (1 + torch.erf(x * (0.5 ** 0.5)))


class LightAtlasTransport(TransportBase):
    geometry = "3dgs"
    requires_normals = True
    # The renderer skips per-Gaussian deep shadows and passes its shadow switch.
    light_space = True
    STATS = 5  # coverage, normalized offset, depth offset, log spread, moment visibility
    # Local material heads. compact: the first six-scene run (2026-10-01). reflect: adds a
    # reflection-vector encoding, finer direction bands, a sharper hint and one more hidden
    # layer. spatial: reflect plus a positional encoding for spatially varying glossy
    # reflections (metals mirroring nearby lit surfaces).
    # svbrdf: reflect plus position-dependent but light-independent coefficients that scale
    # `basis` light-dependent responses computed without position (a rank-limited
    # position-light interaction), so moving shadows and transport cannot be baked locally.
    HEADS = {
        "compact": dict(direction_bands=3, position_bands=0, reflection=False, hidden=3,
                        lobes=(8., 32., 128., 512.)),
        "reflect": dict(direction_bands=4, position_bands=0, reflection=True, hidden=4,
                        lobes=(8., 32., 128., 512., 2048.)),
        "spatial": dict(direction_bands=4, position_bands=8, reflection=True, hidden=4,
                        lobes=(8., 32., 128., 512., 2048.)),
        "svbrdf": dict(direction_bands=4, position_bands=8, reflection=True, hidden=4,
                       lobes=(8., 32., 128., 512., 2048.), basis=8),
    }
    defaults = dict(TransportBase.defaults, width=128, flux_dim=8, atlas_resolution=512, atlas_levels=7,
                    light_transport="atlas", visibility_model="neural", atlas_coverage=0.999,
                    material_head="compact")
    cli_fields = TransportBase.cli_fields + ("width", "flux_dim", "atlas_resolution", "atlas_levels",
                                             "light_transport", "visibility_model", "atlas_coverage",
                                             "material_head")

    def __init__(self, feature_dim=32, width=128, flux_dim=8, atlas_resolution=512, atlas_levels=7,
                 light_transport="atlas", visibility_model="neural", atlas_coverage=0.999,
                 material_head="compact", light_scale=1.0):
        super().__init__(light_scale)
        if light_transport not in ("atlas", "none"):
            raise ValueError("light_transport must be atlas or none")
        if visibility_model not in ("neural", "moment", "gaussian", "none"):
            raise ValueError("visibility_model must be neural, moment, gaussian or none")
        if material_head not in self.HEADS:
            raise ValueError(f"material_head must be one of {sorted(self.HEADS)}")
        if atlas_levels < 1 or atlas_resolution % (1 << (atlas_levels - 1)):
            raise ValueError("atlas resolution must be divisible by 2^(levels-1)")
        if not 0.5 < atlas_coverage <= 1:
            raise ValueError("atlas_coverage must lie in (0.5, 1]")
        self.flux_dim = flux_dim
        self.atlas_resolution = atlas_resolution
        self.atlas_levels = atlas_levels
        self.light_transport = light_transport
        self.visibility_model = visibility_model
        # Ablation: the renderer splats per-Gaussian deep-shadow visibility instead (as in
        # per-Gaussian shadow methods); the atlas then only feeds the transfer term.
        self.per_gaussian_visibility = visibility_model == "gaussian"
        self.atlas_coverage = atlas_coverage
        self.material_head = material_head
        head = self.HEADS[material_head]
        self.direction_bands = head["direction_bands"]
        self.position_bands = head["position_bands"]
        self.reflection = head["reflection"]
        self.basis = head.get("basis", 0)
        self.register_buffer("lobe_sharpness", torch.tensor(head["lobes"]))
        stats = self.STATS * atlas_levels
        # Per-Gaussian flux features: latent code, raw base and light incidence cosine.
        self.flux = _mlp(feature_dim + 4, 32, flux_dim, 2)
        nn.init.normal_(self.flux[-1].weight, std=0.001)
        nn.init.constant_(self.flux[-1].bias, math.log(math.expm1(1.0)))
        # Local neural material: code, normal, directional encodings (light, view, half and
        # optionally reflection), 5 cosines, highlight hints, optional positional encoding.
        encoding = 3 + 6 * self.direction_bands
        inputs = feature_dim + 3 + (4 if self.reflection else 3) * encoding + 5 + len(head["lobes"])
        position = 3 + 6 * self.position_bands if self.position_bands else 0
        if self.basis:
            self.material = _mlp(inputs, width, 3 * (1 + self.basis), head["hidden"])
            self.coefficients = _mlp(feature_dim + position, 64, self.basis, 2)
            nn.init.zeros_(self.coefficients[-1].weight)
            nn.init.zeros_(self.coefficients[-1].bias)
        else:
            self.material = _mlp(inputs + position, width, 3, head["hidden"])
        nn.init.normal_(self.material[-1].weight, std=0.001)
        nn.init.zeros_(self.material[-1].bias)
        self.visibility = _mlp(stats + feature_dim + 1, 64, 1, 2)
        nn.init.zeros_(self.visibility[-1].weight)
        nn.init.zeros_(self.visibility[-1].bias)
        self.kernel = _mlp(stats + feature_dim + 2, width, atlas_levels * 3 * flux_dim, 2)
        nn.init.normal_(self.kernel[-1].weight, std=0.001)
        nn.init.constant_(self.kernel[-1].bias, -7.0)

    # ----- light pass -------------------------------------------------------
    def light_atlas(self, gaussians, light_pos):
        """Splat flux features and depth moments from the light; return the pyramid and its frame."""
        inputs = gaussians.raster_inputs()
        radius = gaussians.radius
        view = light_view(light_pos, gaussians.center)
        means = inputs["means"]
        camera = means @ view[:3, :3].T + view[:3, 3]
        depth_center = (gaussians.center - light_pos).norm()
        with torch.no_grad():
            # Fit the frustum to the splatted extent of almost all Gaussians; distant floaters may fall outside.
            support = 3 * inputs["scales"].amax(-1)
            near = camera[:, 2] - support
            front = near > 1e-3 * radius
            tangent = (camera[:, :2].abs().amax(-1) + support)[front] / near[front]
            if len(tangent):
                k = max(1, min(len(tangent), int(math.ceil(self.atlas_coverage * len(tangent)))))
                extent = tangent.kthvalue(k).values * 1.05
            else:
                extent = tangent.new_tensor(1.0)
            extent = extent.clamp(1e-3, math.tan(math.radians(75)))
        resolution = self.atlas_resolution
        focal = 0.5 * resolution / extent
        K = torch.zeros(3, 3, device=means.device, dtype=means.dtype)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = K[1, 2] = 0.5 * resolution
        K[2, 2] = 1
        normals = covariance_normals(gaussians, light_pos)
        cosine = (normals * F.normalize(light_pos - means, dim=-1)).sum(-1, keepdim=True).clamp(0, 1)
        params = gaussians.params
        phi = F.softplus(self.flux(torch.cat((params["features"], params["base"], cosine), -1)))
        depth = (camera[:, 2:] - depth_center) / radius
        colors = torch.cat((phi, depth, depth.square()), -1)
        rendered, alpha, _ = _rasterize(
            "3dgs", **inputs, colors=colors, viewmats=view[None], Ks=K[None],
            width=resolution, height=resolution, backgrounds=colors.new_zeros((1, colors.shape[-1])),
            packed=False)
        atlas = torch.cat((rendered[0], alpha[0]), -1).permute(2, 0, 1)[None].contiguous()
        levels = [atlas]
        for _ in range(1, self.atlas_levels):
            levels.append(F.avg_pool2d(_binomial_blur(levels[-1]), 2))
        texel = (2 * extent * depth_center / (resolution * radius)).detach()
        return {"view": view, "focal": focal, "depth_center": depth_center, "levels": levels,
                "texel": texel, "extent": extent}

    def gather(self, atlas, points, radius):
        """Per-level light-space statistics and flux features at the receivers' projections."""
        view = atlas["view"]
        camera = points @ view[:3, :3].T + view[:3, 3]
        z = camera[:, 2:].clamp_min(1e-6)
        grid = (camera[:, :2] / z * (2 * atlas["focal"] / self.atlas_resolution)).view(1, 1, -1, 2)
        depth = (camera[:, 2:] - atlas["depth_center"]) / radius
        stats, fluxes = [], []
        c = self.flux_dim
        for k, level in enumerate(atlas["levels"]):
            sampled = F.grid_sample(level, grid, mode="bilinear", padding_mode="zeros",
                                    align_corners=False)[0, :, 0].T
            phi, m1, m2, m0 = sampled[:, :c], sampled[:, c:c + 1], sampled[:, c + 1:c + 2], sampled[:, c + 2:]
            coverage = m0.clamp(0, 1)
            safe = m0.clamp_min(1e-4)
            mean = m1 / safe
            variance = (m2 / safe - mean.square()).clamp_min(0)
            tau = atlas["texel"] * (2 ** k)
            spread = (variance + tau.square()).sqrt()
            offset = depth - mean
            normalized = offset / spread
            moment = 1 - coverage * _normal_cdf(normalized - 3)
            stats.append(torch.cat(((coverage), (normalized / 4).clamp(-2, 2), (4 * offset).clamp(-4, 4),
                                    spread.log(), moment), -1))
            fluxes.append(phi)
        return torch.cat(stats, -1), torch.stack(fluxes, 1)

    # ----- shading ----------------------------------------------------------
    def local_response(self, receivers, normal, light_dir, view_dir, half, xyz=None):
        cosines = torch.stack(((normal * light_dir).sum(-1), (normal * view_dir).sum(-1),
                               (normal * half).sum(-1), (view_dir * half).sum(-1),
                               (light_dir * view_dir).sum(-1)), -1)
        hints = torch.exp(self.lobe_sharpness * (cosines[:, 2:3] - 1))
        bands = self.direction_bands
        parts = [receivers["features"], normal, direction_encoding(light_dir, bands),
                 direction_encoding(view_dir, bands), direction_encoding(half, bands)]
        if self.reflection:
            reflected = 2 * cosines[:, 1:2] * normal - view_dir
            parts.append(direction_encoding(reflected, bands))
        parts.extend((cosines, hints))
        if self.basis:
            # Position enters only through light-independent coefficients of light-dependent bases.
            out = self.material(torch.cat(parts, -1))
            coefficients = self.coefficients(torch.cat(
                (receivers["features"], direction_encoding(xyz, self.position_bands)), -1))
            response = out[:, :3] + (out[:, 3:].view(-1, self.basis, 3) * coefficients[..., None]).sum(1)
            return F.softplus(receivers["base"] + response), cosines
        if self.position_bands:
            parts.append(direction_encoding(xyz, self.position_bands))
        return F.softplus(receivers["base"] + self.material(torch.cat(parts, -1))), cosines

    def receiver_visibility(self, stats, features, n_dot_l):
        prior = stats[:, self.STATS - 1:self.STATS]  # level-0 moment test
        if self.visibility_model == "moment":
            return prior
        logit = torch.logit(prior.clamp(1e-4, 1 - 1e-4))
        return torch.sigmoid(logit + self.visibility(torch.cat((stats, features, n_dot_l), -1)))

    def transfer(self, stats, fluxes, features, cosines):
        weights = F.softplus(self.kernel(torch.cat((stats, features, cosines[:, :2]), -1)))
        weights = weights.view(-1, self.atlas_levels, 3, self.flux_dim)
        return torch.einsum("mkrc,mkc->mr", weights, fluxes)

    def forward(self, gaussians, receivers, eye, light_pos, light_intensity, source_visibility,
                port_active=True, shadow=True):
        points = receivers["means"]
        delta = light_pos - points
        incident = light_intensity[None] / self.light_scale / delta.square().sum(-1, keepdim=True)
        light_dir = F.normalize(delta, dim=-1)
        view_dir = F.normalize(eye - points, dim=-1)
        half = F.normalize(light_dir + view_dir, dim=-1)
        xyz = (points - gaussians.center) / gaussians.radius
        rho, cosines = self.local_response(receivers, receivers["normals"], light_dir, view_dir, half, xyz)
        use_shadow = shadow and self.visibility_model != "none"
        if use_shadow and self.per_gaussian_visibility:
            # Splatted per-Gaussian deep-shadow visibility from the renderer (ablation).
            rho = rho * receivers["visibility"][:, None]
            use_shadow = False
        use_transport = port_active and self.light_transport == "atlas"
        if not (use_shadow or use_transport):
            return incident * rho
        atlas = self.light_atlas(gaussians, light_pos)
        stats, fluxes = self.gather(atlas, points, gaussians.radius)
        features = receivers["features"]
        radiance = rho * (self.receiver_visibility(stats, features, cosines[:, :1]) if use_shadow else 1)
        if use_transport:
            radiance = radiance + self.transfer(stats, fluxes, features, cosines)
        return incident * radiance
