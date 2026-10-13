from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import torch


class H3Logger:
    def __init__(self, output_dir: str | Path, *, enabled: bool, tensorboard: bool, jsonl: bool,
                 console_sink: Callable[[str], None] | None = None) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.python_logger = logging.getLogger("h3dmd")
        self.console_sink = console_sink or self.python_logger.info
        if not self.python_logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        self.json_file = None
        self.writer = None
        if enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if jsonl:
                self.json_file = (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8")
            if tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter

                    self.writer = SummaryWriter(self.output_dir / "tensorboard")
                except ImportError:
                    self.python_logger.warning("tensorboard is unavailable; continuing with JSON/stdout logging")

    def log(self, step: int, values: dict[str, Any]) -> None:
        if not self.enabled:
            return
        scalars = {
            key: float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in values.items()
        }
        self.console_sink(f"step={step} " + " ".join(f"{k}={v:.6g}" for k, v in scalars.items()))
        if self.json_file:
            self.json_file.write(json.dumps({"step": step, **scalars}) + "\n")
            self.json_file.flush()
        if self.writer:
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        if self.json_file:
            self.json_file.close()
        if self.writer:
            self.writer.close()


__all__ = ["H3Logger"]
