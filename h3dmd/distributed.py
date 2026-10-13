from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from h3dmd.config import DistributedConfig
from h3dmd.models import H3ModelBundle


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(config: DistributedConfig) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
        if world_size > 1 and config.backend == "nccl":
            raise RuntimeError("NCCL distributed training requires CUDA")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=config.backend, init_method="env://")
    return DistributedContext(rank, world_size, local_rank, device)


def seed_everything(seed: int, rank: int = 0) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _enable_sequence_parallel(model: nn.Module, size: int, world_size: int) -> None:
    if size <= 1:
        return
    if size != world_size:
        raise NotImplementedError(
            "Phase 1 Diffusers context parallelism uses the complete process group; "
            "sequence_parallel_size must equal WORLD_SIZE."
        )
    transformer = getattr(model, "transformer", model)
    if not hasattr(transformer, "enable_parallelism"):
        raise NotImplementedError("installed Diffusers H3 model has no context-parallel API")
    try:
        from diffusers.models._modeling_parallel import ContextParallelConfig
    except ImportError as exc:
        raise ImportError("sequence parallelism requires a recent Diffusers build") from exc
    transformer.enable_parallelism(
        config=ContextParallelConfig(
            ring_degree=1,
            ulysses_degree=size,
            ulysses_anything=True,
        )
    )


def _apply_fsdp2(model: nn.Module, config: DistributedConfig) -> nn.Module:
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    world_size = dist.get_world_size()
    shard_size = min(config.fsdp_size or world_size, world_size)
    if world_size % shard_size:
        raise ValueError(f"WORLD_SIZE={world_size} must be divisible by FSDP size={shard_size}")
    if shard_size == world_size:
        mesh = init_device_mesh("cuda", (shard_size,), mesh_dim_names=("shard",))
    else:
        mesh = init_device_mesh(
            "cuda", (world_size // shard_size, shard_size), mesh_dim_names=("replicate", "shard")
        )

    reduce_dtype = torch.float32 if config.reduce_dtype == "float32" else torch.bfloat16
    # param_dtype=None is required: H3 intentionally mixes FP32 projections
    # and BF16 blocks. FSDP gathers each parameter in its stored dtype.
    policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=reduce_dtype, output_dtype=None)
    transformer = getattr(model, "transformer", model)
    repeated: list[nn.Module] = []
    repeated.extend(list(getattr(transformer, "transformer_blocks", [])))
    token_refiner = getattr(transformer, "token_refiner", None)
    repeated.extend(list(getattr(token_refiner, "refiner_blocks", [])))
    # FSDP2 requires a uniform dtype among trainable parameters per group.
    # Full H3 training exposes FP32 input/time/output modules alongside the
    # BF16 root parameters; shard those modules separately, preserving dtype.
    for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"):
        module = getattr(transformer, name, None)
        if module is not None:
            repeated.append(module)
    for block in repeated:
        _materialize_cpu_states(block)
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=config.fsdp_reshard_after_forward,
        )
    _materialize_cpu_states(transformer)
    fully_shard(
        transformer,
        mesh=mesh,
        mp_policy=policy,
        reshard_after_forward=config.fsdp_reshard_after_forward,
    )
    return model


@torch.no_grad()
def _materialize_cpu_states(module: nn.Module) -> None:
    """Detach CPU storage from NFS safetensors mappings before CUDA copies.

    Work one FSDP group at a time to bound temporary host memory. Previously
    sharded children already live on CUDA and are left untouched. Pageable
    CUDA copies directly from large mapped files can spend minutes waiting
    for NFS pages, even when only a small parameter view is needed.
    """
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.device.type == "cpu":
            tensor.data = tensor.detach().clone()


def configure_parallel_models(
    models: H3ModelBundle,
    config: DistributedConfig,
    context: DistributedContext,
) -> H3ModelBundle:
    for model in (models.student, models.teacher, models.critic):
        _enable_sequence_parallel(model, config.sequence_parallel_size, context.world_size)
    if config.strategy == "none" or context.world_size == 1:
        return models
    if config.strategy == "fsdp2":
        models.student = _apply_fsdp2(models.student, config)  # type: ignore[assignment]
        models.teacher = _apply_fsdp2(models.teacher, config)  # type: ignore[assignment]
        models.critic = _apply_fsdp2(models.critic, config)  # type: ignore[assignment]
        return models
    if config.strategy == "ddp":
        device_ids = [context.local_rank] if context.device.type == "cuda" else None
        models.student = torch.nn.parallel.DistributedDataParallel(  # type: ignore[assignment]
            models.student, device_ids=device_ids, broadcast_buffers=False
        )
        models.critic = torch.nn.parallel.DistributedDataParallel(  # type: ignore[assignment]
            models.critic, device_ids=device_ids, broadcast_buffers=False
        )
        return models
    raise ValueError(f"unknown distributed strategy {config.strategy!r}")


def configure_parallel_model(
    model: nn.Module,
    role: str,
    config: DistributedConfig,
    context: DistributedContext,
) -> nn.Module:
    """Immediately shard one role so three full H3 copies never coexist.

    This is the construction path used by the trainer. A rank temporarily
    materializes one checkpoint copy, shards it, then proceeds to the next
    role. It is materially lower peak memory than building the bundle first.
    """

    _enable_sequence_parallel(model, config.sequence_parallel_size, context.world_size)
    if config.strategy == "none" or context.world_size == 1:
        return model
    if config.strategy == "fsdp2":
        return _apply_fsdp2(model, config)
    if config.strategy == "ddp" and role != "teacher":
        device_ids = [context.local_rank] if context.device.type == "cuda" else None
        return torch.nn.parallel.DistributedDataParallel(
            model, device_ids=device_ids, broadcast_buffers=False
        )
    if config.strategy == "ddp":
        return model
    raise ValueError(f"unknown distributed strategy {config.strategy!r}")


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    # FSDP2 utilities such as clip_grad_norm_ may return a scalar DTensor.
    # Materialize that global scalar before using the c10d Tensor collective;
    # c10d.all_reduce has no DTensor sharding rule.
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(value, DTensor):
            value = value.full_tensor()
    except ImportError:
        pass
    result = value.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result / dist.get_world_size()


__all__ = [
    "DistributedContext",
    "barrier",
    "configure_parallel_models",
    "configure_parallel_model",
    "init_distributed",
    "reduce_mean",
    "seed_everything",
]
