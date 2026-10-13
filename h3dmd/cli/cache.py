from __future__ import annotations

import argparse

import torch

from h3dmd.cache import H3OfflineCacheEncoder, cache_manifest
from h3dmd.cli.common import add_config_arguments, config_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and cache H3 Qwen layer-50/video-VAE/audio-VAE features")
    add_config_arguments(parser)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = config_from_args(args)
    encoder = H3OfflineCacheEncoder.from_pretrained(
        config.model.pretrained_path,
        device=args.device,
        dtype=torch.bfloat16,
    )
    output = cache_manifest(
        args.raw_manifest,
        args.output_dir,
        encoder,
        text_layer=config.cache.text_encoder_layer,
        video_seed=config.cache.video_encode_seed,
        overwrite=config.cache.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
