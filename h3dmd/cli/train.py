from __future__ import annotations

import argparse

from h3dmd.cli.common import add_config_arguments, config_from_args
from h3dmd.trainers import build_trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MiniMax-H3 with native conditional DMD/RF objectives"
    )
    add_config_arguments(parser)
    args = parser.parse_args()
    build_trainer(config_from_args(args)).train()


if __name__ == "__main__":
    main()
