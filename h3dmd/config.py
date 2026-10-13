from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, get_origin, get_type_hints

import yaml


@dataclass
class ModelConfig:
    pretrained_path: str = "MiniMaxAI/MiniMax-H3"
    transformer_subfolder: str = "transformer"
    task: Literal["t2va", "fl2va", "ref2va"] = "t2va"
    loader: Literal["diffusers", "legacy_fl2va"] = "diffusers"
    dtype: Literal["bfloat16"] = "bfloat16"
    gradient_checkpointing: bool = True
    attention_backend: str | None = None
    lora_path: str | None = None
    lora_rank: int = 0
    lora_alpha: int = 0
    critic_lora_rank: int = 0
    critic_lora_alpha: int = 0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"]
    )


@dataclass
class GeometryConfig:
    video_channels: int = 24
    audio_channels: int = 2
    audio_latent_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    spatial_compression_ratio: int = 16
    fps: int = 24
    audio_latents_per_second: int = 40
    frames_per_chunk: int = 17
    latents_per_chunk: int = 5
    packed_alignment: int | None = None


@dataclass
class ScheduleConfig:
    video_shift: float = 12.0
    audio_shift: float = 3.0
    # None preserves old checkpoints: score training uses video_shift.
    train_video_shift: float | None = None
    # Independent score-noising audio shift; None keeps rollout audio shift.
    train_audio_shift: float | None = None
    # Optional unshifted shared q grid, strictly decreasing to zero. Video and
    # audio sigmas are derived from it with the configured modality shifts.
    inference_q: list[float] | None = None
    num_inference_steps: int = 50
    train_q_min: float = 0.02
    train_q_max: float = 0.98


@dataclass
class DMDConfig:
    enabled: bool = True
    conditional_only: bool = True
    data_mode: Literal["data", "rollout", "mixed"] = "mixed"
    video_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    rf_loss_weight: float = 1.0
    dmd_loss_weight: float = 1.0
    critic_loss_weight: float = 1.0
    critic_update_frequency: int = 1
    rollout_steps: int = 4
    rollout_grad_steps: int = 1
    normalize_dmd_gradient: bool = True
    normalization_eps: float = 1e-5
    update_schedule: Literal["joint", "alternating"] = "joint"
    student_update_frequency: int = 5
    rollout_solver: Literal["euler", "renoise"] = "euler"
    cycle_rollout_steps: bool = False
    loss_reduction: Literal["mean", "sum"] = "mean"


@dataclass
class PCMConfig:
    enabled: bool = False
    loss_weight: float = 1.0
    # Shared unshifted RF coordinate; each modality applies its native shift.
    phase_boundaries_q: list[float] = field(
        default_factory=lambda: [1.0, 0.75, 0.5, 0.25, 0.0]
    )
    steps_per_phase: int = 24
    # Shift the phase edges first, then subdivide each modality in RF sigma.
    # "q" reproduces the pre-2026-10-12 teacher grid.
    grid_interpolation: Literal["sigma", "q"] = "sigma"
    skipping_interval_steps: int = 1
    data_mode: Literal["data", "rollout"] = "data"
    warmup_steps: int = 0
    warmup_only: bool = False


@dataclass
class StagedDistillationConfig:
    pcm_gradient_accumulation_steps: int = 1
    fake_warmup_steps: int = 100
    pcm_lr_warmup_updates: int = 50
    dmd_lr_warmup_updates: int = 20
    pcm_lr_min_ratio: float = 0.1


@dataclass
class CrossResolutionConfig:
    # Used only by the independently registered h3_crossres_dmd algorithm.
    score_height: int = 480
    score_width: int = 768
    # One-based student forward at which the rollout switches to the score
    # resolution. Zero retains the original all-low-resolution algorithm.
    rollout_high_resolution_from_step: int = 0
    mapper_kind: Literal["dysample", "carafe", "liif", "rcan", "bicubic", "ltx"] = "dysample"
    mapper_width: int = 64
    mapper_groups: int = 4
    mapper_blocks_per_stage: int = 4
    mapper_gradient_checkpointing: bool = False
    mapper_checkpoint: str = ""
    mapper_sha256: str = ""
    paired_manifest: str = "data/h3_crossres_320_480_v1/pairs.jsonl"
    pretrain_steps: int = 2000
    max_pretrain_steps: int = 4000
    train_samples: int = 1024
    validation_samples: int = 64
    crop_latent_frames: int = 5
    learning_rate: float = 2e-4
    validation_every: int = 250
    mse_ratio_limit: float = 0.995
    temporal_ratio_limit: float = 1.0


@dataclass
class DataConfig:
    manifest: str = ""
    # Optional prompt-only source after a staged real-AV PCM warmup.
    dmd_manifest: str = ""
    cache_root: str = ""
    batch_size: int = 1
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    frame_buckets: list[int] = field(default_factory=list)
    # Optional per-sample spatial schedule for mixed-resolution training.
    # Entries use ``{width}x{height}``; the sampler repeats the schedule in
    # order and keeps batch size one so each layout remains packed safely.
    resolution_schedule: list[str] = field(default_factory=list)
    # Exact cached AV layout buckets: WIDTHxHEIGHTxFRAMES.
    size_schedule: list[str] = field(default_factory=list)


@dataclass
class CacheConfig:
    text_encoder_layer: int = 50
    video_encode_seed: int = 42
    overwrite: bool = False


@dataclass
class DistributedConfig:
    backend: Literal["nccl", "gloo"] = "nccl"
    strategy: Literal["none", "ddp", "fsdp2"] = "fsdp2"
    sequence_parallel_size: int = 1
    tensor_parallel_size: int = 1
    fsdp_reshard_after_forward: bool = True
    fsdp_size: int | None = None
    reduce_dtype: Literal["float32", "bfloat16"] = "float32"


@dataclass
class CheckpointConfig:
    output_dir: str = "outputs/h3dmd"
    resume: str | None = None
    save_every: int = 1000
    keep_last: int = 3
    save_on_exit: bool = True


@dataclass
class LoggingConfig:
    log_every: int = 10
    tensorboard: bool = True
    jsonl: bool = True


@dataclass
class TrainingConfig:
    seed: int = 42
    max_steps: int = 100_000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    critic_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    grad_clip_enabled: bool = True
    # Match the inherited trainer, and carry the same policy into standalone inference.
    allow_tf32: bool = True
    autocast: bool = True


@dataclass
class H3Config:
    method: str = "h3_distill"
    trainer: str = "distillation"
    model: ModelConfig = field(default_factory=ModelConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    dmd: DMDConfig = field(default_factory=DMDConfig)
    pcm: PCMConfig = field(default_factory=PCMConfig)
    staged: StagedDistillationConfig = field(default_factory=StagedDistillationConfig)
    crossres: CrossResolutionConfig = field(default_factory=CrossResolutionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.method in {"h3_pcm_warmup_dmd", "h3_pcm_warmup_crossres_dmd"}:
            if (not self.pcm.enabled or not self.pcm.warmup_only or self.pcm.warmup_steps <= 0
                    or not self.dmd.enabled or self.dmd.dmd_loss_weight <= 0
                    or self.dmd.rf_loss_weight != 0 or self.dmd.data_mode != "rollout"
                    or self.dmd.update_schedule != "alternating"):
                raise ValueError("staged training requires PCM-only warmup and alternating pure rollout DMD")
            if (self.staged.pcm_gradient_accumulation_steps < 1
                    or self.staged.fake_warmup_steps <= 0 or self.staged.pcm_lr_warmup_updates <= 0
                    or self.staged.dmd_lr_warmup_updates <= 0
                    or not 0 < self.staged.pcm_lr_min_ratio <= 1):
                raise ValueError("invalid staged warmup/scheduler configuration")
        if self.data.dmd_manifest and self.method != "h3_pcm_warmup_crossres_dmd":
            raise ValueError("separate dmd_manifest requires h3_pcm_warmup_crossres_dmd")
        if self.method in {"h3_crossres_dmd", "h3_pcm_warmup_crossres_dmd", "h3_multires_dmd"}:
            c = self.crossres
            if ((self.pcm.enabled and self.method in {"h3_crossres_dmd", "h3_multires_dmd"})
                    or self.dmd.rf_loss_weight != 0 or self.dmd.data_mode != "rollout"):
                raise ValueError("h3_crossres_dmd requires pure DMD with generated-only fake targets")
            if not self.dmd.enabled or self.dmd.dmd_loss_weight <= 0:
                raise ValueError("h3_crossres_dmd requires an enabled DMD objective")
            if self.dmd.loss_reduction != "mean":
                raise ValueError("cross-resolution experiments use explicit per-modality mean reduction")
            if self.data.width is None or self.data.height is None:
                raise ValueError("cross-resolution DMD requires one fixed student resolution")
            if (c.score_width % 32 or c.score_height % 32 or
                    c.score_width <= self.data.width or c.score_height <= self.data.height):
                raise ValueError("score dimensions must be larger positive multiples of 32")
            if c.score_width * self.data.height != c.score_height * self.data.width:
                raise ValueError("cross-resolution score and rollout must preserve aspect ratio")
            if c.mapper_kind not in {"dysample", "carafe", "liif", "rcan", "bicubic", "ltx"}:
                raise ValueError("invalid neural upsampler kind")
            if c.mapper_kind == "bicubic" and (c.mapper_checkpoint or c.mapper_sha256):
                raise ValueError("fixed bicubic mapping must not bind a neural checkpoint")
            if c.mapper_width < 1 or c.mapper_groups < 1 or c.mapper_width % c.mapper_groups:
                raise ValueError("mapper feature width must be divisible by its positive group count")
            if c.mapper_kind == "ltx" and (c.mapper_width % 32 or c.mapper_blocks_per_stage < 1
                    or 2*c.score_height != 3*self.data.height or 2*c.score_width != 3*self.data.width):
                raise ValueError("LTX mapper requires width divisible by32, positive blocks and exact 1.5x geometry")
            if c.mapper_sha256 and (len(c.mapper_sha256) != 64 or any(
                    v not in "0123456789abcdef" for v in c.mapper_sha256)):
                raise ValueError("mapper_sha256 must be a lowercase SHA256 digest")
            if (min(c.pretrain_steps, c.train_samples, c.validation_samples,
                    c.crop_latent_frames, c.validation_every) < 1 or
                    c.max_pretrain_steps < c.pretrain_steps or
                    not math.isfinite(c.learning_rate) or c.learning_rate <= 0 or
                    not 0 < c.mse_ratio_limit <= 1 or not 0 < c.temporal_ratio_limit <= 1):
                raise ValueError("invalid mapper pretraining settings")
        high_from = self.crossres.rollout_high_resolution_from_step
        if self.method == "h3_multires_dmd":
            if (not 2 <= high_from <= self.dmd.rollout_steps
                    or self.dmd.rollout_solver != "renoise"):
                raise ValueError("multires DMD requires renoise and a valid high-resolution transition step")
        elif high_from != 0:
            raise ValueError("rollout resolution transitions require h3_multires_dmd")
        if self.dmd.data_mode not in {"data", "rollout", "mixed"}:
            raise ValueError("invalid dmd.data_mode")
        if self.dmd.update_schedule not in {"joint", "alternating"}:
            raise ValueError("invalid dmd.update_schedule")
        if self.dmd.rollout_solver not in {"euler", "renoise"}:
            raise ValueError("invalid dmd.rollout_solver")
        if self.dmd.loss_reduction not in {"mean", "sum"}:
            raise ValueError("invalid dmd.loss_reduction")
        if self.dmd.student_update_frequency < 1:
            raise ValueError("student_update_frequency must be positive")
        if (
            self.dmd.update_schedule == "alternating"
            and self.dmd.critic_update_frequency != 1
        ):
            raise ValueError("alternating updates require critic_update_frequency=1")
        if self.pcm.data_mode not in {"data", "rollout"}:
            raise ValueError("invalid pcm.data_mode")
        edges = self.pcm.phase_boundaries_q
        if (
            len(edges) < 2
            or edges[0] != 1
            or edges[-1] != 0
            or any(not math.isfinite(q) or not 0 <= q <= 1 for q in edges)
            or any(a <= b for a, b in zip(edges, edges[1:]))
        ):
            raise ValueError("PCM phase_boundaries_q must strictly descend from 1 to 0")
        if self.pcm.grid_interpolation not in {"sigma", "q"}:
            raise ValueError("PCM grid_interpolation must be sigma or q")
        if not 0 < self.pcm.skipping_interval_steps <= self.pcm.steps_per_phase:
            raise ValueError(
                "PCM requires 0 < skipping_interval_steps <= steps_per_phase"
            )
        if (
            self.pcm.warmup_steps < 0
            or not math.isfinite(self.pcm.loss_weight)
            or self.pcm.loss_weight < 0
        ):
            raise ValueError(
                "PCM warmup_steps/loss_weight must be nonnegative and finite"
            )
        if self.pcm.warmup_only and self.pcm.warmup_steps == 0:
            raise ValueError("PCM warmup_only requires positive warmup_steps")
        if self.pcm.enabled and self.pcm.loss_weight <= 0:
            raise ValueError("enabled PCM requires positive loss_weight")
        if (
            self.pcm.enabled
            and self.pcm.warmup_only
            and not (
                self.dmd.enabled
                and self.dmd.dmd_loss_weight > 0
                or self.dmd.rf_loss_weight > 0
            )
        ):
            raise ValueError("PCM warmup_only requires an objective after warmup")
        if self.model.task != "t2va":
            raise NotImplementedError(
                "Phase 1 supports the FL2VA checkpoint in T2VA mode only; "
                "FL2VA/Ref2VA extension points exist but are not enabled."
            )
        if not self.dmd.conditional_only:
            raise ValueError(
                "MiniMax-H3 phase 1 is conditional-only; CFG/unconditional branches are invalid."
            )
        for name in ("video_shift", "audio_shift", "train_video_shift", "train_audio_shift"):
            value = getattr(self.schedule, name)
            if value is None and name in {"train_video_shift", "train_audio_shift"}:
                continue
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"schedule.{name} must be finite and positive")
        if self.model.dtype != "bfloat16":
            raise ValueError(
                "H3 block weights are BF16 with selected FP32 modules; use dtype=bfloat16."
            )
        if self.geometry.audio_channels != 2:
            raise ValueError(
                "MiniMax-H3 audio latents are stereo and packed channel-major."
            )
        if self.geometry.patch_size != (1, 2, 2):
            raise ValueError("The released FL2VA transformer uses patch_size=(1,2,2).")
        if (
            self.geometry.video_channels != 24
            or self.geometry.audio_latent_channels != 32
        ):
            raise ValueError(
                "The released FL2VA checkpoint requires 24 video and 32 audio latent channels."
            )
        if self.geometry.spatial_compression_ratio != 16:
            raise ValueError(
                "MiniMax-H3's video VAE uses a spatial compression ratio of 16."
            )
        if self.geometry.fps != 24 or self.geometry.audio_latents_per_second != 40:
            raise ValueError(
                "MiniMax-H3 requires 24 fps video and 40 Hz audio latents."
            )
        if (self.geometry.frames_per_chunk, self.geometry.latents_per_chunk) != (17, 5):
            raise ValueError(
                "MiniMax-H3's temporal VAE contract is 17 pixel frames to 5 latent frames."
            )
        if self.cache.text_encoder_layer != 50:
            raise ValueError(
                "The released H3 checkpoint is conditioned on Qwen3-VL hidden_states[50]."
            )
        if self.data.batch_size < 1:
            raise ValueError("batch_size must be positive")
        for name in ("height", "width"):
            value = getattr(self.data, name)
            if value is not None and (value <= 0 or value % 32):
                raise ValueError(
                    f"data.{name} must be a positive multiple of 32 (VAE + patch)"
                )
        if self.data.num_frames is not None and (
            self.data.num_frames < 5 or (self.data.num_frames - 5) % 17
        ):
            raise ValueError("data.num_frames must follow 17*n+5")
        if self.data.frame_buckets:
            if self.data.num_frames is not None:
                raise ValueError("use num_frames or frame_buckets, not both")
            if any(n < 5 or (n - 5) % 17 for n in self.data.frame_buckets):
                raise ValueError("frame_buckets must follow 17*n+5")
            if (
                self.data.batch_size != 1
                or self.training.gradient_accumulation_steps != 1
            ):
                raise ValueError(
                    "alternating frame buckets currently require batch_size=1 and accumulation=1"
                )
        if self.data.resolution_schedule:
            if self.data.height is not None or self.data.width is not None:
                raise ValueError(
                    "mixed-resolution data requires data.height/data.width=null"
                )
            if self.data.batch_size != 1 or self.training.gradient_accumulation_steps != 1:
                raise ValueError(
                    "mixed-resolution training currently requires batch_size=1 and accumulation=1"
                )
            for spec in self.data.resolution_schedule:
                try:
                    width_text, height_text = spec.lower().split("x", 1)
                    width, height = int(width_text), int(height_text)
                except (AttributeError, ValueError) as exc:
                    raise ValueError(
                        f"resolution_schedule entries must be WIDTHxHEIGHT, got {spec!r}"
                    ) from exc
                if width <= 0 or height <= 0 or width % 32 or height % 32:
                    raise ValueError(
                        f"resolution_schedule entry must use positive multiples of 32, got {spec!r}"
                    )
        if self.data.size_schedule:
            if self.data.resolution_schedule:
                raise ValueError("use size_schedule or resolution_schedule, not both")
            if self.data.height is not None or self.data.width is not None or self.data.num_frames is not None:
                raise ValueError("size_schedule requires data.height/data.width/data.num_frames=null")
            if self.data.batch_size != 1 or self.training.gradient_accumulation_steps != 1:
                raise ValueError("size_schedule requires batch_size=1 and accumulation=1")
            for spec in self.data.size_schedule:
                try:
                    width, height, frames = (int(part) for part in spec.lower().split("x"))
                except (AttributeError, ValueError) as exc:
                    raise ValueError(f"size_schedule entries must be WIDTHxHEIGHTxFRAMES, got {spec!r}") from exc
                if width <= 0 or height <= 0 or width % 32 or height % 32 or frames < 5 or (frames - 5) % 17:
                    raise ValueError(f"invalid size_schedule entry {spec!r}")
        if self.distributed.fsdp_size is not None and self.distributed.fsdp_size < 1:
            raise ValueError("fsdp_size must be positive")
        if self.geometry.packed_alignment is not None:
            raise NotImplementedError(
                "the current Diffusers H3 forward has no padded-attention input; keep "
                "geometry.packed_alignment=null for active training/inference"
            )
        if not 0 <= self.schedule.train_q_min < self.schedule.train_q_max <= 1:
            raise ValueError("training q range must satisfy 0 <= min < max <= 1")
        if (
            self.dmd.rollout_grad_steps < 1
            or self.dmd.rollout_grad_steps > self.dmd.rollout_steps
        ):
            raise ValueError("rollout_grad_steps must be in [1, rollout_steps]")
        if self.dmd.rollout_steps < 1:
            raise ValueError("rollout_steps must be positive")
        loss_weights = (
            self.dmd.video_loss_weight,
            self.dmd.audio_loss_weight,
            self.dmd.rf_loss_weight,
            self.dmd.dmd_loss_weight,
            self.dmd.critic_loss_weight,
        )
        if any(weight < 0 for weight in loss_weights):
            raise ValueError("loss weights cannot be negative")
        if self.schedule.inference_q is not None:
            q = self.schedule.inference_q
            if (
                len(q) < 2
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in q)
                or q[-1] != 0
                or any(left <= right for left, right in zip(q, q[1:]))
            ):
                raise ValueError(
                    "inference_q must contain finite [0,1] values, be strictly decreasing, "
                    "contain at least two points, and end at 0"
                )
            if len(q) - 1 < self.dmd.rollout_grad_steps:
                raise ValueError(
                    "inference_q has fewer forwards than rollout_grad_steps"
                )
            if len(q) - 1 != self.dmd.rollout_steps:
                raise ValueError(
                    "inference_q forward count must equal dmd.rollout_steps"
                )
        if self.dmd.critic_update_frequency < 1:
            raise ValueError("critic_update_frequency must be positive")
        if self.model.lora_rank < 0 or self.model.critic_lora_rank < 0:
            raise ValueError("LoRA ranks cannot be negative")
        if self.model.lora_rank > 0 and self.model.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive when lora_rank is enabled")
        if self.model.critic_lora_rank > 0 and self.model.critic_lora_alpha <= 0:
            raise ValueError(
                "critic_lora_alpha must be positive when critic_lora_rank is enabled"
            )
        if self.schedule.inference_q is None and self.schedule.num_inference_steps < 2:
            raise ValueError(
                "num_inference_steps counts q grid points and must be at least 2"
            )
        if self.distributed.sequence_parallel_size < 1:
            raise ValueError("sequence_parallel_size must be positive")
        if self.training.max_steps < 1 or self.training.gradient_accumulation_steps < 1:
            raise ValueError(
                "max_steps and gradient_accumulation_steps must be positive"
            )
        if self.checkpoint.save_every < 1 or self.logging.log_every < 1:
            raise ValueError("save_every and log_every must be positive")
        if self.distributed.tensor_parallel_size != 1:
            raise NotImplementedError(
                "Phase 1 supports FSDP data sharding plus sequence parallelism, not TP."
            )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate_cached_data(self, *, shape_only: bool) -> None:
        if not shape_only:
            return
        if self.dmd.rf_loss_weight > 0 or self.dmd.data_mode != "rollout":
            raise ValueError(
                "shape_only requires rf_loss_weight=0 and data_mode=rollout"
            )
        if self.pcm.enabled and self.pcm.data_mode != "rollout":
            raise ValueError(
                "shape_only caches require pcm.data_mode=rollout; PCM data mode needs real AV latents"
            )


T = TypeVar("T")


def _coerce_dataclass(cls: type[T], values: dict[str, Any], path: str = "") -> T:
    if not isinstance(values, dict):
        raise TypeError(f"{path or cls.__name__} must be a mapping")
    hints = get_type_hints(cls)
    known = {item.name for item in dataclasses.fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise KeyError(f"unknown config keys at {path or cls.__name__}: {unknown}")
    kwargs: dict[str, Any] = {}
    for item in dataclasses.fields(cls):
        if item.name not in values:
            continue
        value = values[item.name]
        hinted = hints.get(item.name)
        if isinstance(hinted, type) and dataclasses.is_dataclass(hinted):
            value = _coerce_dataclass(hinted, value, f"{path}.{item.name}".strip("."))
        elif get_origin(hinted) is tuple and isinstance(value, list):
            value = tuple(value)
        elif hinted is float and isinstance(value, (str, int)):
            value = float(value)
        kwargs[item.name] = value
    return cls(**kwargs)


def _parse_override(value: str) -> Any:
    return yaml.safe_load(value)


def _apply_override(mapping: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must be key=value, got {override!r}")
    dotted, raw = override.split("=", 1)
    cursor = mapping
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = _parse_override(raw)


def config_from_dict(
    raw: dict[str, Any], overrides: list[str] | None = None
) -> H3Config:
    from h3dmd.configs import compose_config

    raw = compose_config(raw)
    for override in overrides or []:
        _apply_override(raw, override)
    config = _coerce_dataclass(H3Config, raw)
    config.validate()
    return config


def load_config(path: str | Path, overrides: list[str] | None = None) -> H3Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if raw.get("model", {}).get("_target_") == "rcm.models.h3_distill.H3DistillModel":
        from omegaconf import OmegaConf
        raw = OmegaConf.to_container(OmegaConf.create(raw).model.config.h3, resolve=True)
    return config_from_dict(raw, overrides)


def load_experiment(name: str, overrides: list[str] | None = None) -> H3Config:
    return config_from_dict({"experiment": name}, overrides)
