"""Explicit registry shared by training, evaluation and checkpoints."""
import argparse

from .base import PortTransport, TransportBase
from .directional import DirectionalTransport
from .attention import SurfaceAttention
from .neural_material import NeuralMaterialTransport


METHODS = {
    "directional_port_v1": DirectionalTransport,
    "surface_attention": SurfaceAttention,
    "neural_material": NeuralMaterialTransport,
}
DEFAULT_METHOD = "directional_port_v1"


def method_class(config):
    name = config["representation"]
    if name not in METHODS:
        raise ValueError(f"Unknown transport method: {name}")
    return METHODS[name]


def resolve_config(config):
    cls = method_class(config)
    geometry = '3dgs' if config.get('init_geometry_format') == 'gggs' else cls.geometry
    return {**cls.defaults, **config, "geometry": geometry}


def build_transport(config, light_scale):
    config = resolve_config(config)
    cls = method_class(config)
    return cls(**{key: config[key] for key in cls.defaults}, light_scale=light_scale)


def add_method_arguments(parser, argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    choices = METHODS
    selector.add_argument("--representation", choices=choices, default=DEFAULT_METHOD)
    selection, _ = selector.parse_known_args(argv)
    parser.add_argument("--representation", choices=choices, default=DEFAULT_METHOD,
                        help="registered transport method; use METHOD --help for its options")
    cls = choices[selection.representation]
    group = parser.add_argument_group(f"{selection.representation} parameters")
    for field in cls.cli_fields:
        default = cls.defaults[field]
        group.add_argument("--" + field.replace("_", "-"), type=type(default), default=default)
