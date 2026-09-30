"""Six directly optimized parameters for a two-lobe isotropic GGX material.

RGB F0, two log-spaced GGX alpha values and a mixture weight replace the
neural latent. Directional albedo uses the same MaterialX fit as the teacher.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .procedural import _ggx, _ggx_albedo


class GGXMaterial(nn.Module):
    def forward(self, latent, wi, wo, normal=None):
        half = F.normalize(wi + wo, dim=-1)
        if normal is None:
            ni, no, nh = wi[..., 2:3], wo[..., 2:3], half[..., 2:3]
        else:
            ni, no, nh = ((normal * direction).sum(-1, keepdim=True)
                          for direction in (wi, wo, half))
        visible = (ni > 0) & (no > 0)
        ni, no, nh = ni.clamp(1e-6, 1), no.clamp(1e-6, 1), nh.clamp(0, 1)
        ih = (wi * half).sum(-1, keepdim=True).clamp(0, 1)
        f0 = latent[..., :3]
        alpha = (.001 * torch.exp(math.log(1000) * latent[..., 3:5]))
        mix = latent[..., 5:6]
        fresnel = f0 + (1 - f0) * (1 - ih).pow(5)
        shape = mix * _ggx(ni, no, nh, alpha[..., :1]) + (1 - mix) * _ggx(ni, no, nh, alpha[..., 1:])
        albedo = mix * _ggx_albedo(ni, alpha[..., :1], f0) + (1 - mix) * _ggx_albedo(ni, alpha[..., 1:], f0)
        reflected = (albedo * latent.new_tensor([.2126, .7152, .0722])).sum(-1, keepdim=True)
        return fresnel * shape * visible, 1 - albedo, reflected

    @staticmethod
    def initial_latent(reference):
        return reference.new_tensor([.04, .04, .04, math.log(100)/math.log(1000),
                                     math.log(300)/math.log(1000), .9])
