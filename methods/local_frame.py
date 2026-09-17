"""Learn a material shading frame for direct light; retain directional exchange."""
import torch
from torch import nn
from torch.nn import functional as F

from .directional import DirectionalTransport


class LocalFrameTransport(DirectionalTransport):
    defaults = dict(DirectionalTransport.defaults, frame_width=32)
    cli_fields = DirectionalTransport.cli_fields + ("frame_width",)

    def __init__(self, frame_width=32, **kwargs):
        super().__init__(**kwargs)
        self.frame_net = nn.Sequential(
            nn.Linear(self.exchange.in_features, frame_width), nn.SiLU(),
            nn.Linear(frame_width, 6),
        )
        nn.init.zeros_(self.frame_net[-1].weight)
        with torch.no_grad():
            self.frame_net[-1].bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def shading_frame(self, features):
        first, second = self.frame_net(features).chunk(2, dim=-1)
        tangent = F.normalize(first, dim=-1)
        bitangent = F.normalize(second - (second * tangent).sum(-1, keepdim=True) * tangent, dim=-1)
        normal = torch.linalg.cross(tangent, bitangent, dim=-1)
        return torch.stack((tangent, bitangent, normal), dim=-1)

    def material_directions(self, features, light_dir, view_dir):
        frame = self.shading_frame(features)
        directions = frame.transpose(1, 2) @ torch.stack((light_dir, view_dir), dim=-1)
        return directions.unbind(-1)
