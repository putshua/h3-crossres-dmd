from __future__ import annotations

import argparse

from h3dmd.config import H3Config, load_config, load_experiment


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        help="H3DMD YAML configuration (optionally composed from registered defaults)",
    )
    source.add_argument(
        "--experiment", help="registered experiment name, e.g. h3_synthetic_audio_dmd"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable dotted YAML override",
    )


def config_from_args(args: argparse.Namespace) -> H3Config:
    if args.experiment:
        return load_experiment(args.experiment, args.set)
    return load_config(args.config, args.set)


__all__ = ["add_config_arguments", "config_from_args"]
