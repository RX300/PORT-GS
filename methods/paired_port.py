"""Separate the input and output spatial support of each directional mode."""
from torch import nn

from .base import spatial_partition
from .directional import DirectionalTransport


class PairedPortTransport(DirectionalTransport):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Equal initial supports recover the directional baseline exactly.
        self.output_centers = nn.Parameter(self.anchor_centers.detach().clone())
        self.output_log_width = nn.Parameter(self.log_width.detach().clone())

    def receiver_partition(self, xyz):
        return spatial_partition(xyz, self.output_centers, self.output_log_width)
