from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from h3dmd.config import CheckpointConfig, H3Config
from h3dmd.distributed import DistributedContext, barrier
from h3dmd.models import H3ModelBundle


class H3CheckpointManager:
    def __init__(self, config: CheckpointConfig, context: DistributedContext) -> None:
        self.config = config
        self.context = context
        self.root = Path(config.output_dir).expanduser().resolve() / "checkpoints"
        if context.is_main:
            self.root.mkdir(parents=True, exist_ok=True)
        barrier()

    def _state(
        self,
        models: H3ModelBundle,
        optimizers: dict[str, torch.optim.Optimizer],
        step: int,
    ) -> dict[str, Any]:
        def portable(model):
            return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        return {
            "student": portable(models.student),
            "critic": portable(models.critic),
            "student_optimizer": optimizers["student"],
            "critic_optimizer": optimizers["critic"],
            "step": torch.tensor(step, dtype=torch.int64),
        }

    def save(
        self,
        models: H3ModelBundle,
        optimizers: dict[str, torch.optim.Optimizer],
        step: int,
        full_config: H3Config,
    ) -> Path:
        destination = self.root / f"step_{step:09d}"
        state = self._state(models, optimizers, step)
        if dist.is_available() and dist.is_initialized():
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict

            for role in ("student", "critic"):
                state[role], state[f"{role}_optimizer"] = get_state_dict(
                    state[role], optimizers[role]
                )
            dcp.save(state, checkpoint_id=str(destination))
        elif self.context.is_main:
            destination.mkdir(parents=True, exist_ok=False)
            temporary = destination / f"state.tmp.{os.getpid()}.pt"
            torch.save(
                {
                    "student": models.student.state_dict(),
                    "critic": models.critic.state_dict(),
                    "student_optimizer": optimizers["student"].state_dict(),
                    "critic_optimizer": optimizers["critic"].state_dict(),
                    "step": step,
                },
                temporary,
            )
            temporary.replace(destination / "state.pt")
        barrier()
        if self.context.is_main:
            (destination / "config.json").write_text(
                json.dumps(full_config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            latest_tmp = self.root / f"latest.tmp.{os.getpid()}"
            latest_tmp.write_text(destination.name + "\n", encoding="utf-8")
            latest_tmp.replace(self.root / "latest")
            self._prune()
        barrier()
        return destination

    def load(
        self,
        path: str | Path,
        models: H3ModelBundle,
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> int:
        source = Path(path).expanduser().resolve()
        if source.is_dir() and (source / "latest").is_file():
            source = source / (source / "latest").read_text(encoding="utf-8").strip()
        state = self._state(models, optimizers, 0)
        if dist.is_available() and dist.is_initialized():
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

            roles = {role: state[role] for role in ("student", "critic")}
            for role, model in roles.items():
                state[role], state[f"{role}_optimizer"] = get_state_dict(model, optimizers[role])
            dcp.load(state, checkpoint_id=str(source))
            for role, model in roles.items():
                set_state_dict(
                    model, optimizers[role], model_state_dict=state[role],
                    optim_state_dict=state[f"{role}_optimizer"],
                )
            return int(state["step"].item())
        payload = torch.load(source / "state.pt", map_location=self.context.device, weights_only=False)
        models.student.load_state_dict(payload["student"], strict=True)
        models.critic.load_state_dict(payload["critic"], strict=True)
        optimizers["student"].load_state_dict(payload["student_optimizer"])
        optimizers["critic"].load_state_dict(payload["critic_optimizer"])
        return int(payload["step"])

    def _prune(self) -> None:
        if self.config.keep_last <= 0:
            return
        checkpoints = sorted(path for path in self.root.glob("step_*"))
        for path in checkpoints[: -self.config.keep_last]:
            if path.parent == self.root and path.name.startswith("step_"):
                shutil.rmtree(path)


__all__ = ["H3CheckpointManager"]
