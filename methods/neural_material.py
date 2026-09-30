"""Frozen universal BRDF, Gaussian material codes and directional transport."""

import math

import torch
from torch.nn import functional as F

from materials import MaterialDecoder
from materials.ggx import GGXMaterial
from .base import SourceLight, ReceiverLight, quadrature_mass
from .directional import DirectionalTransport


class NeuralMaterialTransport(DirectionalTransport):
    geometry = "2dgs"
    requires_normals = True
    defaults = dict(DirectionalTransport.defaults, material_decoder="", material_model="neural")
    cli_fields = DirectionalTransport.cli_fields + ("material_decoder", "material_model")

    def __init__(self, material_decoder="", material_model="neural", **kwargs):
        super().__init__(**kwargs)
        if self.direction_net[0].in_features < 9:
            raise ValueError("neural_material requires 6 material and 3 shading-normal channels")
        del self.local
        self.material_decoder = material_decoder
        self.material_model = material_model
        if material_model not in {"neural", "ggx"}:
            raise ValueError("material_model must be neural or ggx")
        self.decoder = (MaterialDecoder() if material_model == "neural" else GGXMaterial()).requires_grad_(False)

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.decoder.requires_grad_(False)
        return self

    @torch.no_grad()
    def initialize_material(self, gaussians, reset_normal=True):
        # Fresh training loads a prior once. Evaluation restores embedded weights
        # directly from transport.state_dict(), without accessing this path.
        if self.material_model == "neural":
            saved = torch.load(self.material_decoder, map_location="cpu", weights_only=False)
            self.decoder.load_state_dict(saved["decoder"], strict=True)
            initial = saved["initial_latent"]
        else:
            initial = self.decoder.initial_latent(gaussians.params["features"])
        gaussians.params["features"][:, :6].copy_(torch.logit(initial))
        if reset_normal:
            gaussians.params["features"][:, 6:9].zero_()

    @staticmethod
    def material_normal(receivers):
        normal=receivers['normals']
        code = receivers['features'][:,6:9]
        if 'normal_residual' in receivers:
            code = code+receivers['normal_residual']
        offset=.5*code.tanh()
        offset=offset-(offset*normal).sum(-1,keepdim=True)*normal
        return F.normalize(normal+offset,dim=-1)

    def material_response(self, receivers, light_dir, view_dir):
        normal = self.material_normal(receivers)
        latent = receivers["features"][:, :6].sigmoid()
        non_diffuse, transmission, _ = self.decoder(latent, light_dir, view_dir, normal)
        diffuse = receivers["base"].sigmoid() / math.pi
        cosine = (normal * light_dir).sum(-1, keepdim=True).clamp_min(0.)
        return non_diffuse * cosine, transmission * diffuse * cosine

    def direct_response(self, receivers, xyz, light_dir, view_dir):
        specular, diffuse = self.material_response(receivers, light_dir, view_dir)
        return specular + diffuse

    @staticmethod
    def transport_features(features):
        # Transport may condition on material, but must not move its codes to
        # fit a separate unconstrained RGB branch.
        return torch.cat((features[:, :9].detach(), features[:, 9:]), dim=-1)

    def forward(self, gaussians, receivers, eye, light_pos, light_intensity,
                source_visibility, port_active=True):
        incident, light_dir = self.point_light(
            receivers['means'], light_pos, light_intensity, receivers['visibility'])
        view_dir = F.normalize(eye - receivers['means'], dim=-1)
        specular, diffuse = self.material_response(receivers, light_dir, view_dir)
        if not port_active:
            return incident * (specular + diffuse)
        p = gaussians.params
        source_incident, source_direction = self.point_light(
            p['means'], light_pos, light_intensity, source_visibility)
        source = SourceLight((p['means']-gaussians.center)/gaussians.radius,
                             self.transport_features(p['features']), source_incident,
                             source_direction, quadrature_mass(p))
        receiver = ReceiverLight((receivers['means']-gaussians.center)/gaussians.radius,
                                 self.transport_features(receivers['features']), incident,
                                 view_dir, diffuse)
        # The inherited gate only mixes diffuse and nonlocal transport. It
        # cannot attenuate or recolor the direct non-diffuse material response.
        return incident * specular + self.exchange_radiance(source, receiver)
