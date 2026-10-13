"""Pretrain/validate a mapper before launching the independent crossres DMD run."""
import argparse
import importlib
import os
from pathlib import Path

import torch
import torch.distributed as dist

from imaginaire.utils.config_helper import get_config_module, override
from rcm.configs.h3 import native_config
from h3dmd.mapper_pretraining import pretrain


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", default="rcm/configs/registry_h3.py")
    p.add_argument("--output", required=True)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()
    config = native_config(override(importlib.import_module(get_config_module(a.config)).make_config(),
                                    a.opts).model.config.h3)
    if config.method != "h3_crossres_dmd":
        raise ValueError("Use the registered cross-resolution experiment for mapper pretraining")
    torch.set_num_threads(2)
    local, rank, world = (int(os.environ.get(k, "0" if k != "WORLD_SIZE" else "1"))
                          for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    torch.manual_seed(config.training.seed)
    try:
        pretrain(config, Path(a.output), device, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
