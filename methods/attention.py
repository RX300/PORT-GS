"""Illumination-linear cross attention from source cells to pixel receivers.

Fixed spatial binning only reduces source count. No learned port centers,
spherical-Gaussian lobes or per-port transfer matrices are used.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .base import NeuralTransport, direction_encoding


def pool_sources(source, signal, grid_size):
    """Area-weighted source quadrature in occupied fixed scene cells."""
    coordinates = ((source.xyz.detach()+1)*(.5*grid_size)).floor().long().clamp(0,grid_size-1)
    cell = coordinates[:,0]*grid_size**2 + coordinates[:,1]*grid_size + coordinates[:,2]
    mass = source.mass
    attributes = torch.cat((source.xyz,source.features,source.direction,signal),dim=-1)
    total = mass.new_zeros(grid_size**3).index_add(0,cell,mass)
    pooled = attributes.new_zeros((grid_size**3,attributes.shape[-1])).index_add(
        0,cell,attributes*mass[:,None])
    occupied = total>0
    values = pooled[occupied]/total[occupied,None]
    xyz, features = values[:,:3], values[:,3:-6]
    direction, radiance = F.normalize(values[:,-6:-3],dim=-1), values[:,-3:]
    return xyz,features,direction,radiance,total[occupied]


class SurfaceAttention(NeuralTransport):
    geometry = "2dgs"
    defaults = dict(NeuralTransport.defaults, grid_size=8, attention_dim=32, attention_score="dot")
    cli_fields = NeuralTransport.cli_fields + ("grid_size", "attention_dim", "attention_score")

    def __init__(self, feature_dim=32, width=128, light_scale=1., grid_size=8, attention_dim=32,
                 attention_score="dot"):
        super().__init__(feature_dim, width, light_scale)
        if attention_score not in ("dot", "cosine"):
            raise ValueError("attention_score must be dot or cosine")
        self.grid_size = grid_size
        self.attention_dim = attention_dim
        self.attention_score = attention_score
        inputs = feature_dim + 54  # position and direction Fourier features
        self.query = nn.Sequential(nn.Linear(inputs, attention_dim), nn.SiLU(),
                                   nn.Linear(attention_dim, attention_dim))
        self.key = nn.Sequential(nn.Linear(inputs, attention_dim), nn.SiLU(),
                                 nn.Linear(attention_dim, attention_dim))
        self.source_response = nn.Linear(feature_dim, 3)
        nn.init.zeros_(self.source_response.weight)
        nn.init.constant_(self.source_response.bias, math.log(math.expm1(1.)))

    def pooled_sources(self, source):
        signal = F.softplus(self.source_response(source.features))*source.incident
        return pool_sources(source,signal,self.grid_size)

    def source_tokens(self, source):
        xyz,features,direction,radiance,mass = self.pooled_sources(source)
        keys = self.key(torch.cat((features,direction_encoding(xyz),direction_encoding(direction)),dim=-1))
        return keys,radiance,mass

    def exchange_radiance(self, source, receiver):
        keys,values,mass = self.source_tokens(source)
        queries = self.query(torch.cat((receiver.features,direction_encoding(receiver.xyz),
                                       direction_encoding(receiver.direction)),dim=-1))
        scale = None
        if self.attention_score == "cosine":
            # Bound angular logits without letting feature norms shut off source gradients.
            queries, keys = F.normalize(queries,dim=-1), F.normalize(keys,dim=-1)
            scale = self.attention_dim**.5
        # Q/K and the area prior never consume RGB illumination. Only V carries light.
        # Padding V allows the native CUDA memory-efficient attention kernel.
        values = F.pad(values,(0,self.attention_dim-3))
        attended = F.scaled_dot_product_attention(
            queries[None,None],keys[None,None],values[None,None],
            attn_mask=mass.log()[None,None,None,:],dropout_p=0.,scale=scale)[0,0,:,:3]
        fraction = self.exchange(receiver.features).sigmoid()
        return (1-fraction)*receiver.incident*receiver.response + fraction*attended

