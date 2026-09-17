"""Explicit registry shared by training, evaluation and checkpoints."""
import argparse

from .anchor import AnchorTransport
from .base import TransportBase
from .directional import DirectionalTransport


METHODS = {
    "learned_anchor_exchange": AnchorTransport,
    "directional_port_v1": DirectionalTransport,
}
DEFAULT_METHOD = "directional_port_v1"


def method_class(config):
    # Pre-directional checkpoints did not record representation.
    name = config.get("representation", "learned_anchor_exchange")
    if name not in METHODS:
        raise ValueError(f"Unknown transport method: {name}")
    return METHODS[name]


def resolve_config(config):
    return {**method_class(config).defaults, **config}


def build_transport(config, light_scale):
    cls = method_class(config)
    return cls(**{key: config[key] for key in cls.defaults}, light_scale=light_scale)


def add_method_arguments(parser, argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--representation", choices=METHODS, default=DEFAULT_METHOD)
    selection, _ = selector.parse_known_args(argv)
    parser.add_argument("--representation", choices=METHODS, default=DEFAULT_METHOD,
                        help="registered transport method; use METHOD --help for its options")
    cls = METHODS[selection.representation]
    group = parser.add_argument_group(f"{selection.representation} parameters")
    for field in cls.cli_fields:
        default = cls.defaults[field]
        group.add_argument("--" + field.replace("_", "-"), type=type(default), default=default)
