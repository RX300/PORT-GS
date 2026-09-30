"""Procedural eight-lobe non-diffuse teacher from Yu et al., SIGGRAPH 2026.

Reference: https://blaire9989.github.io/assets/4_DataEnhance/supplemental.pdf
The 22 channels follow its Table 4: eta_d, eta_r RGB, eta_i RGB, alpha_core,
alpha_haze, mix_core, metalness, w_dust, t_dust, w_clear, eta_clear, w_inner,
t_inner, scatter RGB, w_sub, alpha_sub. Directions are unit vectors in the
positive local-z hemisphere; returned f excludes both diffuse and cosine.

Adaptations: online themed samples replace the paper's large texture grids;
directional albedos use MaterialX analytic fits (Schlick F0, whereas BRDF
Fresnel is exact). Core weights are consistent in both S.2 expressions,
correcting their apparent printed interchange. Incident-only layering follows
S.1-S.6 and does not impose reciprocity on the composed lobes. See NOTICE.md
for the adapted MaterialX formulas, coefficients, and license.
"""

import math

import torch
import torch.nn.functional as F


def _uniform(shape, device, generator):
    # Open endpoints avoid singular IORs and precisely tangent directions.
    return torch.rand(shape, device=device, generator=generator).clamp(1e-6, 1 - 1e-6)


def sample_materials(count, device, generator=None):
    """Return [N,22] parameters and classes: haze, dust, coat, scatter, infill.

    Each class has probability 20%; infill samples free artist controls
    independently. Themes use 70% dielectric / 30% conductor, except scatter,
    which is dielectric. Metal RGB includes common gold/silver/copper colors.
    """
    u = _uniform((count, 28), device, generator)
    classes = (u[:, 0] * 5).long()
    infill = classes[:, None] == 4

    def normal(column, mean, std, low, high):
        a = math.erf((low - mean) / (std * math.sqrt(2)))
        b = math.erf((high - mean) / (std * math.sqrt(2)))
        return mean + std * math.sqrt(2) * torch.erfinv(a + (b - a) * u[:, column:column + 1])

    eta = torch.where(infill, 1 + u[:, 1:2], normal(1, 1.5, .2, 1., 2.))
    color = u[:, 2:5]
    metals = u.new_tensor([[.95, .78, .35], [.97, .96, .91], [.95, .64, .54]])
    common = metals[(u[:, 25] * 3).long()] * (.8 + .2 * u[:, 26:27])
    color = torch.where((~infill) & (u[:, 27:28] < .5), common, color)
    # Gulbrandsen (2014), equations 2 and 12, white edge tint g=1.
    eta_r = (1 - color) / (1 + color)
    eta_i = 2 * color.sqrt() / (1 + color)
    core = torch.where(infill, .001 + .199 * u[:, 5:6],
                       (.001 ** .5 + (.2 ** .5 - .001 ** .5) * u[:, 5:6]).square())
    haze = torch.where(infill, .005 + .495 * u[:, 6:7], (core + .3 * u[:, 6:7]).clamp_min(.005))
    mix = torch.where(infill, u[:, 7:8], normal(7, .5, .2, 0., 1.))
    metal = torch.where(infill, u[:, 8:9], (u[:, 8:9] > .7).float())
    metal = torch.where(classes[:, None] == 3, 0., metal)
    dust = u[:, 9:10] * ((u[:, 10:11] < .1) | (classes[:, None] == 1) | infill)
    coat = ((u[:, 12:13] < .2) | (classes[:, None] == 2)).float()
    coat = torch.where(infill, (u[:, 12:13] < .5).float(), coat)
    eta_coat = torch.where(infill, 1.3 + .4 * u[:, 13:14], normal(13, 1.5, .08, 1.3, 1.7))
    inner = u[:, 14:15] * ((u[:, 15:16] < .2) | (classes[:, None] == 3) | infill)
    sub = ((u[:, 20:21] < .2) | (classes[:, None] == 3)).float()
    sub = torch.where(infill, (u[:, 20:21] < .5).float(), sub)
    alpha_sub = torch.where(infill, .1 + .85 * u[:, 21:22],
                            (haze + .05 + .4 * u[:, 21:22]).clamp_min(.1))
    decorated = (classes[:, None] != 0).float()
    params = torch.cat((eta, eta_r, eta_i, core, haze, mix, metal,
                        dust * decorated, u[:, 11:12], coat * decorated, eta_coat,
                        inner * decorated, u[:, 16:17], u[:, 17:20],
                        sub * decorated, alpha_sub), -1)
    return params, classes


def sample_directions(count, device, generator=None, *, params=None):
    """50% broad coordinates, 50% material-matched GGX half-vector proposals.

    Difference angle is sampled inside its exact valid interval, so both
    directions lie above the horizon without replacement/fallback directions.
    Optional [N,22] params selects core/haze/active-coat GGX roughness; without
    it, the proposal uses alpha=.1. This is a training distribution, not a
    full multi-lobe importance sampler or a Monte Carlo integration PDF.
    """
    u = _uniform((count, 6), device, generator)
    theta_h = u[:, 0] * (math.pi / 2 - 1e-5)
    if params is None:
        alpha = u.new_full((count,), .1)
    else:
        alpha = torch.where(u[:, 4] < params[:, 9], params[:, 7], params[:, 8])
        alpha = torch.where(u[:, 4] < .25 * params[:, 13], .005, alpha)
    ggx_theta = torch.atan(alpha * torch.sqrt(u[:, 0] / (1 - u[:, 0])))
    theta_h = torch.where(u[:, 5] < .5, ggx_theta, theta_h)
    phi_d = 2 * math.pi * u[:, 1]
    sh, ch = theta_h.sin(), theta_h.cos()
    max_difference = torch.atan2(ch, sh * phi_d.cos().abs())
    theta_d = u[:, 2] * max_difference * (1 - 1e-5)
    dx = theta_d.sin() * phi_d.cos()
    dy = theta_d.sin() * phi_d.sin()
    dz = theta_d.cos()
    phi_h = 2 * math.pi * u[:, 3]
    cp, sp = phi_h.cos(), phi_h.sin()

    def rotate(sign):
        x, y = ch * (sign * dx) + sh * dz, sign * dy
        z = ch * dz - sh * (sign * dx)
        return torch.stack((cp * x - sp * y, sp * x + cp * y, z), -1)

    return rotate(1), rotate(-1)


def _fresnel_dielectric(c, eta):
    g = (eta.square() + c.square() - 1).clamp_min(0).sqrt()
    rs = ((c - g) / (c + g)).square()
    rp = ((eta.square() * c - g) / (eta.square() * c + g)).square()
    return .5 * (rs + rp)


def _fresnel_conductor(c, eta, k):
    c2, s2 = c.square(), 1 - c.square()
    t0 = eta.square() - k.square() - s2
    ab = (t0.square() + 4 * eta.square() * k.square()).sqrt()
    # Stable real part of sqrt(t0 + 2j*eta*k), including near-white metals.
    root = (.5 * (ab + t0.abs())).sqrt()
    a = torch.where(t0 >= 0, root, eta * k / root)
    t1, t2 = ab + c2, 2 * a * c
    rs = (t1 - t2) / (t1 + t2)
    t3, t4 = c2 * ab + s2.square(), t2 * s2
    rp = rs * (t3 - t4) / (t3 + t4)
    return .5 * (rs + rp)


def _ggx(ni, no, nh, alpha):
    a2 = alpha.square()
    # Algebraic form avoids cancellation at nh=1 for alpha=.001.
    denom = 1 - nh.square() + a2 * nh.square()
    ndf = a2 / (math.pi * denom.square())
    li = (a2 + (1 - a2) * ni.square()).sqrt()
    lo = (a2 + (1 - a2) * no.square()).sqrt()
    return ndf / (2 * (li * no + lo * ni))


def _ggx_albedo(cosine, alpha, f0):
    """MaterialX rational GGX directional-albedo fit, F90=1."""
    x, y = cosine, alpha
    coefficients = x.new_tensor([
        [.1003, .9345, 1., 1.], [-.6303, -2.323, -1.765, .2281],
        [9.748, 2.229, 8.263, 15.94], [-2.038, -3.748, 11.53, -55.83],
        [29.34, 1.424, 28.96, 13.08], [-8.245, -.7684, -7.507, 41.26],
        [-26.44, 1.436, -36.11, 54.9], [19.99, .2913, 15.86, 300.2],
        [-5.448, .6286, 33.37, -285.1]])
    basis = torch.cat((torch.ones_like(x), x, y, x * y, x.square(), y.square(),
                       x.square() * y, x * y.square(), x.square() * y.square()), -1)
    r = basis @ coefficients
    ab = (r[..., :2] / r[..., 2:]).clamp(0, 1)
    return (f0 * ab[..., :1] + ab[..., 1:2]).clamp(0, 1)


def _sheen(ni, no, nh, thickness):
    inverse = 1 / thickness.clamp_min(.005)
    ndf = (2 + inverse) * (1 - nh.square()).clamp_min(0).pow(.5 * inverse) / (2 * math.pi)
    return ndf / (4 * (ni + no - ni * no))


def _sheen_albedo(cosine, thickness):
    x, y = cosine, thickness.clamp_min(.005)
    numerator = 13.673 - 68.78018 * x + 799.08825 * y - 905.00061 * x * y + 60.28956 * x.square() + 1086.96473 * y.square()
    denominator = 1 + 61.57746 * x + 442.78211 * y + 2597.49308 * x * y + 121.81241 * x.square() + 3045.55075 * y.square()
    return (numerator / denominator).clamp(0, 1)


def enhanced_brdf(params, wi, wo):
    """Return RGB f_non, RGB incident transmission, and reflected luminance.

    Reflection is accumulated separately from transmission: absorption in
    metals and tinted scatter must not be labeled reflected energy. Albedo
    fits give approximate energy accounting, not an exact integral of f_non.
    """
    eta, nr, ki = params[..., :1], params[..., 1:4], params[..., 4:7]
    core, haze, mix, metal = (params[..., i:i + 1] for i in range(7, 11))
    dust, td, coat, ec, inner, ti = (params[..., i:i + 1] for i in range(11, 17))
    tint, sub, alpha_sub = params[..., 17:20], params[..., 20:21], params[..., 21:22]
    half = F.normalize(wi + wo, dim=-1)
    ni, no = wi[..., 2:3], wo[..., 2:3]
    nh = half[..., 2:3].clamp(0, 1)
    ih = (wi * half).sum(-1, keepdim=True).clamp(0, 1)
    fd = _fresnel_dielectric(ih, eta)
    fc = _fresnel_conductor(ih, nr, ki)
    f0d = ((eta - 1) / (eta + 1)).square()
    f0c = ((nr - 1).square() + ki.square()) / ((nr + 1).square() + ki.square())
    shape = mix * _ggx(ni, no, nh, core) + (1 - mix) * _ggx(ni, no, nh, haze)
    ad = mix * _ggx_albedo(ni, core, f0d) + (1 - mix) * _ggx_albedo(ni, haze, f0d)
    ac = mix * _ggx_albedo(ni, core, f0c) + (1 - mix) * _ggx_albedo(ni, haze, f0c)

    # S.1: tint absorbs the unreflected portion of each scatter event.
    ai, asc = _sheen_albedo(ni, ti), _ggx_albedo(ni, alpha_sub, f0d)
    t_inner = 1 - inner * ai
    bottom = tint * (inner * _sheen(ni, no, nh, ti) + t_inner * sub * fd * _ggx(ni, no, nh, alpha_sub))
    r_bottom = tint * (inner * ai + t_inner * sub * asc)
    t_bottom = t_inner * (1 - sub * asc)
    # S.2-S.4: conductors reflect or absorb; only dielectric transmits.
    f = metal * fc * shape + (1 - metal) * (fd * shape + (1 - ad) * bottom)
    reflected = metal * ac + (1 - metal) * (ad + (1 - ad) * r_bottom)
    transmitted = (1 - metal) * (1 - ad) * t_bottom
    # S.5-S.6: clearcoat below white dust.
    alpha_coat = torch.full_like(ec, .005)
    a_coat = _ggx_albedo(ni, alpha_coat, ((ec - 1) / (ec + 1)).square())
    f_coat = _fresnel_dielectric(ih, ec) * _ggx(ni, no, nh, alpha_coat)
    t_coat = 1 - coat * a_coat
    f = coat * f_coat + t_coat * f
    reflected = coat * a_coat + t_coat * reflected
    transmitted = t_coat * transmitted
    a_dust = _sheen_albedo(ni, td)
    t_dust = 1 - dust * a_dust
    f = dust * _sheen(ni, no, nh, td) + t_dust * f
    reflected = dust * a_dust + t_dust * reflected
    transmitted = (t_dust * transmitted).expand_as(f)
    luminance = (reflected * params.new_tensor([.2126, .7152, .0722])).sum(-1, keepdim=True)
    return f, transmitted, luminance
