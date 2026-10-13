from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from h3dmd.cache import H3OfflineCacheEncoder
from h3dmd.config import H3Config
from h3dmd.h3.adapter import H3ModelAdapter
from h3dmd.h3.layout import build_packed_layout, unpack_audio
from h3dmd.h3.scheduler import H3DualNoiseScheduler, H3NoiseLevel, truncated_q_grid
from h3dmd.registry import PIPELINES
from h3dmd.precision import configure_h3_precision


def align_num_frames(
    num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5
) -> int:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    chunks = max(
        0, (num_frames - latents_per_chunk + frames_per_chunk - 1) // frames_per_chunk
    )
    return chunks * frames_per_chunk + latents_per_chunk


def video_latent_num_frames(
    num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5
) -> int:
    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError("num_frames must be aligned to 17*n+5")
    return (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2


def audio_latent_num_frames(
    num_frames: int, fps: int = 24, latent_rate: int = 40
) -> int:
    return int(round(num_frames / fps * latent_rate))


@dataclass
class H3InferenceOutput:
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    videos: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    sampling_rate: int | None = None
    num_forwards: int = 0


@PIPELINES.register("h3_t2va")
class H3InferencePipeline:
    """Conditional-only MiniMax-H3 T2VA inference with coupled schedules."""

    def __init__(
        self,
        model: H3ModelAdapter,
        config: H3Config,
        encoders: H3OfflineCacheEncoder | None = None,
        score_mapper: torch.nn.Module | None = None,
    ) -> None:
        config.validate()
        configure_h3_precision(config)
        self.model = model
        self.config = config
        self.encoders = encoders
        self.score_mapper = score_mapper
        if config.method == "h3_multires_dmd" and self.score_mapper is None:
            from h3dmd.upsampling import load_frozen_mapper
            self.score_mapper = load_frozen_mapper(config, device=next(model.parameters()).device)
        self.scheduler = H3DualNoiseScheduler(
            config.schedule.video_shift, config.schedule.audio_shift
        )

    @classmethod
    def from_pretrained(
        cls,
        config: H3Config,
        *,
        device: str | torch.device = "cuda",
        load_encoders: bool = False,
        encoder_device: str | torch.device = "cpu",
    ) -> "H3InferencePipeline":
        model = H3ModelAdapter.from_pretrained(
            config.model.pretrained_path,
            subfolder=config.model.transformer_subfolder,
            loader=config.model.loader,
            dtype=torch.bfloat16,
            strict=True,
            device=device,
        )
        if config.model.attention_backend:
            if not hasattr(model.transformer, "set_attention_backend"):
                raise NotImplementedError(
                    "this Diffusers H3 build has no selectable attention backend"
                )
            model.transformer.set_attention_backend(config.model.attention_backend)
        model.configure_lora(
            path=config.model.lora_path,
            rank=config.model.lora_rank,
            alpha=config.model.lora_alpha,
            target_modules=config.model.lora_target_modules,
            trainable=False,
        )
        encoders = (
            H3OfflineCacheEncoder.from_pretrained(
                config.model.pretrained_path,
                device=encoder_device,
                dtype=torch.bfloat16,
            )
            if load_encoders
            else None
        )
        return cls(model.eval(), config, encoders)

    def load_student_checkpoint(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint).expanduser().resolve()
        if (path / "state.pt").is_file():
            payload = torch.load(
                path / "state.pt", map_location="cpu", weights_only=False
            )
            self.model.load_state_dict(payload["student"], strict=True)
            return
        try:
            import torch.distributed.checkpoint as dcp
            upstream = path / "model" if (path / "model" / ".metadata").is_file() else path
            if (upstream / ".metadata").is_file():
                metadata = dcp.FileSystemReader(str(upstream)).read_metadata()
                if any(k.startswith("net.") for k in metadata.state_dict_metadata):
                    state = {"net." + k: v for k, v in self.model.state_dict().items()}
                    dcp.load(state, checkpoint_id=str(upstream))
                    self.model.load_state_dict({k[4:]: v for k, v in state.items()}, strict=True)
                    return
            dcp.load({"student": self.model}, checkpoint_id=str(path))
        except Exception as exc:
            raise RuntimeError(
                f"could not load distilled H3 checkpoint from {path}"
            ) from exc

    @staticmethod
    def _cpu_generator(generator: torch.Generator | None) -> torch.Generator:
        if generator is None:
            return torch.Generator(device="cpu").manual_seed(42)
        if str(generator.device) != "cpu":
            raise ValueError("SGLang-aligned H3 noise requires a CPU torch.Generator")
        clone = torch.Generator(device="cpu")
        clone.set_state(generator.get_state())
        return clone

    @torch.no_grad()
    def generate_latents(
        self,
        prompt_embeds: torch.Tensor,
        text_token_tags: torch.Tensor,
        *,
        height: int = 768,
        width: int = 1344,
        num_frames: int = 124,
        num_inference_steps: int | None = None,
        q_schedule: list[float] | None = None,
        generator: torch.Generator | None = None,
        video_latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        solver: str | None = None,
        clean_video_transforms: dict[int, torch.nn.Module] | None = None,
    ) -> H3InferenceOutput:
        if height % 32 or width % 32:
            raise ValueError("H3 height and width must be multiples of 32")
        num_frames = align_num_frames(
            num_frames,
            self.config.geometry.frames_per_chunk,
            self.config.geometry.latents_per_chunk,
        )
        minimum_frames = align_num_frames(5 * self.config.geometry.fps)
        maximum_frames = align_num_frames(15 * self.config.geometry.fps)
        if not minimum_frames <= num_frames <= maximum_frames:
            raise ValueError("released MiniMax-H3 generates 5 to 15 seconds at 24 fps")
        batch = prompt_embeds.shape[0]
        pcm_only = self.config.pcm.enabled and not (
            self.config.dmd.enabled and self.config.dmd.dmd_loss_weight > 0
        )
        solver = solver or ("euler" if pcm_only else self.config.dmd.rollout_solver)
        if solver not in {"euler", "renoise"}:
            raise ValueError("solver must be euler or renoise")
        multires = self.config.method == "h3_multires_dmd"
        if clean_video_transforms:
            if solver != "renoise" or multires:
                raise ValueError("Explicit clean transforms require renoise without implicit multires mapping")
        if multires and (solver != "renoise" or (height, width) != (
                self.config.data.height, self.config.data.width)):
            raise ValueError("multires inference must start at the configured low resolution with renoise")
        video_generator = self._cpu_generator(generator)
        audio_generator = self._cpu_generator(generator)
        device = next(self.model.parameters()).device
        latent_t = video_latent_num_frames(
            num_frames,
            self.config.geometry.frames_per_chunk,
            self.config.geometry.latents_per_chunk,
        )
        latent_h = height // self.config.geometry.spatial_compression_ratio
        latent_w = width // self.config.geometry.spatial_compression_ratio
        audio_t = audio_latent_num_frames(
            num_frames,
            self.config.geometry.fps,
            self.config.geometry.audio_latents_per_second,
        )
        if video_latents is None:
            # Current SGLang H3 uses two independently re-seeded CPU streams:
            # raw video tensor first in its stream, packed audio rows first in
            # the other. Preserve those shapes before moving to the model.
            video_latents = torch.randn(
                (
                    batch,
                    self.config.geometry.video_channels,
                    latent_t,
                    latent_h,
                    latent_w,
                ),
                generator=video_generator,
                dtype=torch.float32,
            ).to(device)
        else:
            video_latents = video_latents.to(device=device, dtype=torch.float32)
        if audio_latents is None:
            audio_rows = torch.randn(
                (
                    batch,
                    self.config.geometry.audio_channels * audio_t,
                    self.config.geometry.audio_latent_channels,
                ),
                generator=audio_generator,
                dtype=torch.float32,
            )
            audio_latents = unpack_audio(
                audio_rows,
                self.config.geometry.audio_channels,
                self.config.geometry.audio_latent_channels,
            ).to(device)
        else:
            audio_latents = audio_latents.to(device=device, dtype=torch.float32)
        expected_video = (
            batch,
            self.config.geometry.video_channels,
            latent_t,
            latent_h,
            latent_w,
        )
        expected_audio = (
            batch,
            self.config.geometry.audio_channels,
            self.config.geometry.audio_latent_channels,
            audio_t,
        )
        if (
            tuple(video_latents.shape) != expected_video
            or tuple(audio_latents.shape) != expected_audio
        ):
            raise ValueError(
                f"initial latent shapes must be video={expected_video}, audio={expected_audio}; "
                f"got {tuple(video_latents.shape)}, {tuple(audio_latents.shape)}"
            )
        layout = build_packed_layout(
            text_token_tags=text_token_tags,
            num_latent_frames=latent_t,
            latent_height=latent_h,
            latent_width=latent_w,
            num_audio_latents=audio_t,
            patch_size=self.config.geometry.patch_size,
            audio_channels=self.config.geometry.audio_channels,
            alignment=self.config.geometry.packed_alignment,
        ).to(device)
        # Fewer forwards use exactly the same input times as training rollout.
        # Explicit q_schedule remains available for deliberately different grids.
        effective_q = q_schedule
        if effective_q is None:
            configured_q = (self.config.pcm.phase_boundaries_q if pcm_only
                            else self.config.schedule.inference_q)
            if configured_q is not None:
                effective_q = (list(configured_q) if num_inference_steps is None else
                               truncated_q_grid(configured_q, num_inference_steps - 1))
        schedule = self.scheduler.schedule(
            num_inference_steps or self.config.schedule.num_inference_steps,
            q=effective_q,
            device=device,
        )
        if clean_video_transforms and any(
            not isinstance(i, int) or not 1 <= i < schedule.num_forwards
            for i in clean_video_transforms
        ):
            raise ValueError("Clean transform keys must be interior target grid indices")
        prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)
        for index in range(schedule.num_forwards):
            level = H3NoiseLevel(
                q=schedule.q[index],
                video_sigma=schedule.video_sigmas[index],
                audio_sigma=schedule.audio_sigmas[index],
                video_timestep=schedule.video_timesteps[index],
                audio_timestep=schedule.audio_timesteps[index],
            )
            velocity = self.model(
                video_latents,
                audio_latents,
                prompt_embeds,
                layout=layout,
                video_timestep=level.video_timestep,
                audio_timestep=level.audio_timestep,
            )
            if solver == "renoise":
                video_latents = self.scheduler.to_x0(
                    video_latents.float(), velocity.video.float(), level.video_sigma
                )
                audio_latents = self.scheduler.to_x0(
                    audio_latents.float(), velocity.audio.float(), level.audio_sigma
                )
                # A boundary belongs to the next noisy state: resize the current
                # clean prediction first, then draw noise at the target geometry.
                if clean_video_transforms and index + 1 in clean_video_transforms:
                    with torch.autocast(device.type, enabled=False):
                        video_latents = clean_video_transforms[index + 1](video_latents.float())
                    if (video_latents.ndim != 5 or video_latents.shape[:3] !=
                            (batch, self.config.geometry.video_channels, latent_t)):
                        raise ValueError("Clean transform must preserve video batch, channels and time")
                    if any(s % patch for s, patch in zip(
                            video_latents.shape[-2:], self.config.geometry.patch_size[-2:])):
                        raise ValueError("Transformed latent geometry must align to video patches")
                    layout = build_packed_layout(
                        text_token_tags=text_token_tags, num_latent_frames=latent_t,
                        latent_height=video_latents.shape[-2], latent_width=video_latents.shape[-1],
                        num_audio_latents=audio_t, patch_size=self.config.geometry.patch_size,
                        audio_channels=self.config.geometry.audio_channels,
                        alignment=self.config.geometry.packed_alignment,
                    ).to(device)
                # Map clean x0, never a noisy latent. The next forward then
                # receives fresh high-resolution noise at its original time.
                if (multires and index < schedule.num_forwards - 1
                        and index + 2 == self.config.crossres.rollout_high_resolution_from_step):
                    with torch.autocast(device.type, enabled=False):
                        video_latents = self.score_mapper(video_latents.float())
                    layout = build_packed_layout(
                        text_token_tags=text_token_tags, num_latent_frames=latent_t,
                        latent_height=video_latents.shape[-2], latent_width=video_latents.shape[-1],
                        num_audio_latents=audio_t, patch_size=self.config.geometry.patch_size,
                        audio_channels=self.config.geometry.audio_channels,
                        alignment=self.config.geometry.packed_alignment,
                    ).to(device)
                if index < schedule.num_forwards - 1:
                    video_noise = torch.randn(
                        video_latents.shape,
                        generator=video_generator,
                        dtype=torch.float32,
                    ).to(device)
                    audio_noise = unpack_audio(
                        torch.randn(
                            (
                                batch,
                                self.config.geometry.audio_channels * audio_t,
                                self.config.geometry.audio_latent_channels,
                            ),
                            generator=audio_generator,
                            dtype=torch.float32,
                        ),
                        self.config.geometry.audio_channels,
                        self.config.geometry.audio_latent_channels,
                    ).to(device)
                    video_latents = self.scheduler.add_noise(
                        video_latents, video_noise, schedule.video_sigmas[index + 1]
                    )
                    audio_latents = self.scheduler.add_noise(
                        audio_latents, audio_noise, schedule.audio_sigmas[index + 1]
                    )
                continue
            video_latents = self.scheduler.euler_step(
                video_latents,
                velocity.video,
                schedule.video_sigmas[index],
                schedule.video_sigmas[index + 1],
            )
            audio_latents = self.scheduler.euler_step(
                audio_latents,
                velocity.audio,
                schedule.audio_sigmas[index],
                schedule.audio_sigmas[index + 1],
            )
        if multires and video_latents.shape[-2:] == (latent_h, latent_w):
            # Early exits use the same mapped clean distribution as DMD score.
            with torch.autocast(device.type, enabled=False):
                video_latents = self.score_mapper(video_latents.float())
        return H3InferenceOutput(
            video_latents, audio_latents, num_forwards=schedule.num_forwards
        )

    @torch.no_grad()
    def decode(self, output: H3InferenceOutput) -> H3InferenceOutput:
        if self.encoders is None:
            raise RuntimeError(
                "decoding requires load_encoders=True or an H3OfflineCacheEncoder"
            )
        video_vae = self.encoders.video_vae
        audio_vae = self.encoders.audio_vae
        device = next(video_vae.parameters()).device
        video_mean = torch.tensor(video_vae.config.latents_mean, device=device).view(
            1, -1, 1, 1, 1
        )
        video_std = torch.tensor(video_vae.config.latents_std, device=device).view(
            1, -1, 1, 1, 1
        )
        normalized = output.video_latents.to(device) * video_std + video_mean
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            videos = video_vae.decode(normalized, return_dict=False)[0]
        pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(
            1, 3, 1, 1, 1
        )
        pixel_std = torch.tensor((0.229, 0.224, 0.225), device=device).view(
            1, 3, 1, 1, 1
        )
        videos = (videos.float() * pixel_std + pixel_mean).clamp(0, 1)

        audio_device = next(audio_vae.parameters()).device
        audio_mean = torch.tensor(
            audio_vae.config.latents_mean, device=audio_device
        ).view(1, 1, -1, 1)
        audio_std = torch.tensor(
            audio_vae.config.latents_std, device=audio_device
        ).view(1, 1, -1, 1)
        audio_latents = output.audio_latents.to(audio_device) * audio_std + audio_mean
        batch, channels, latent_channels, frames = audio_latents.shape
        decoded = audio_vae.decode(
            audio_latents.reshape(batch * channels, latent_channels, frames),
            return_dict=False,
        )[0]
        audio = decoded.reshape(batch, channels, -1).float()
        sampling_rate = int(
            getattr(
                audio_vae.config,
                "sample_rate",
                getattr(audio_vae.config, "sampling_rate", 32000),
            )
        )
        output.videos = videos
        output.audio = audio
        output.sampling_rate = sampling_rate
        return output

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        *,
        decode: bool = True,
        **kwargs: Any,
    ) -> H3InferenceOutput:
        if self.encoders is None:
            raise RuntimeError(
                "prompt inference requires load_encoders=True; otherwise call generate_latents"
            )
        prompt_embeds, tags = self.encoders.encode_text(
            prompt, self.config.cache.text_encoder_layer
        )
        output = self.generate_latents(prompt_embeds[None], tags, **kwargs)
        return self.decode(output) if decode else output


__all__ = [
    "H3InferenceOutput",
    "H3InferencePipeline",
    "align_num_frames",
    "audio_latent_num_frames",
    "video_latent_num_frames",
]
