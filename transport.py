"""Direct, light-conditioned radiance queries using NVIDIA HashGrid features."""

import torch
import tinycudann as tcnn
from torch import nn
from torch.nn import functional as F


def direction_encoding(direction, bands=4):
    frequency = 2 ** torch.arange(bands, device=direction.device, dtype=direction.dtype)
    phase = direction[..., None] * frequency * torch.pi
    return torch.cat((direction, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)


class Transport(nn.Module):
    """Decode RGB at each receiver, with no source integral or exchange channels.

    The decoder predicts a positive response conditioned on spatial hash
    features, material features, light/view geometry and shadow visibility.
    Point-light intensity and inverse-square falloff set the radiometric scale.
    Visibility is an input, not a hard multiplier, so occluded queries can learn
    nonzero responses. There is no discrete conservation/reversibility claim.
    """

    def __init__(self, feature_dim=32, width=128, light_scale=1.0, *, hash_encoding, seed=0):
        super().__init__()
        self.register_buffer("light_scale", torch.tensor(float(light_scale)))
        self.spatial_encoding = tcnn.Encoding(
            n_input_dims=3, encoding_config=hash_encoding, seed=seed, dtype=torch.float32,
        )
        # Material + hash + angular (82) + light position (27) + distance + visibility.
        inputs = feature_dim + self.spatial_encoding.n_output_dims + 82 + 27 + 2
        self.decoder = nn.Sequential(
            nn.Linear(inputs, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 3),
        )
        nn.init.normal_(self.decoder[-1].weight, std=0.001)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, receivers, center, radius, eye, light_pos, light_intensity):
        xyz = (receivers["means"] - center) / radius
        spatial = self.spatial_encoding((xyz + 1) * 0.5)
        delta = light_pos - receivers["means"]
        distance = delta.norm(dim=-1, keepdim=True)
        light_dir = delta / distance
        view_dir = F.normalize(eye - receivers["means"], dim=-1)
        light_position = direction_encoding((light_pos - center) / radius)
        inputs = torch.cat(
            (
                receivers["features"], spatial,
                direction_encoding(light_dir), direction_encoding(view_dir),
                direction_encoding(F.normalize(light_dir + view_dir, dim=-1)),
                (light_dir * view_dir).sum(-1, keepdim=True),
                light_position[None].expand(len(xyz), -1),
                torch.log1p(distance / radius), receivers["visibility"][:, None],
            ), dim=-1,
        )
        response = F.softplus(receivers["base"] + self.decoder(inputs))
        return response * (light_intensity[None] / self.light_scale / distance.square())
