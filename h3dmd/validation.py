from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from h3dmd.cache import H3OfflineCacheEncoder
from h3dmd.config import H3Config
from h3dmd.data import H3Batch
from h3dmd.dmd import H3DMDMethod
from h3dmd.distributed import configure_parallel_model, init_distributed
from h3dmd.h3.adapter import H3ModelAdapter
from h3dmd.h3.layout import build_packed_layout
from h3dmd.h3.scheduler import H3DualNoiseScheduler
from h3dmd.inference import H3InferencePipeline
from h3dmd.models import H3ModelBundle, build_model_adapter


@dataclass(frozen=True)
class TensorComparison:
    shape: tuple[int, ...]
    max_abs: float
    mean_abs: float
    max_rel: float
    cosine: float
    passed: bool


def load_tensor_mapping(path: str | Path) -> dict[str, Any]:
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"validation fixture must be a tensor mapping: {path}")
    return value


def compare_tensor(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> TensorComparison:
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: actual={actual.shape}, expected={expected.shape}")
    actual_f = actual.detach().float().cpu()
    expected_f = expected.detach().float().cpu()
    difference = (actual_f - expected_f).abs()
    denominator = expected_f.abs().clamp_min(1e-8)
    cosine = torch.nn.functional.cosine_similarity(actual_f.flatten(), expected_f.flatten(), dim=0)
    passed = bool(torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol))
    return TensorComparison(
        shape=tuple(actual.shape),
        max_abs=float(difference.max()),
        mean_abs=float(difference.mean()),
        max_rel=float((difference / denominator).max()),
        cosine=float(cosine),
        passed=passed,
    )


def batch_from_fixture(values: dict[str, Any], device: torch.device | str) -> H3Batch:
    prompt = values["prompt_embeds"]
    video = values["video_latents"]
    audio = values["audio_latents"]
    if prompt.ndim == 2:
        prompt = prompt[None]
    if video.ndim == 4:
        video = video[None]
    if audio.ndim == 3:
        audio = audio[None]
    return H3Batch(
        prompt_embeds=prompt.to(device),
        text_token_tags=values["text_token_tags"].flatten().long().to(device),
        video_latents=video.float().to(device),
        audio_latents=audio.float().to(device),
        sample_ids=[str(values.get("sample_id", "validation"))],
    )


class H3ValidationSuite:
    """Ordered GPU acceptance checks against exported SGLang oracle tensors."""

    def __init__(self, config: H3Config, device: str | torch.device = "cuda") -> None:
        self.config = config
        self.context = init_distributed(config.distributed)
        requested = torch.device(device)
        self.device = self.context.device if requested.type == "cuda" else requested

    def _training_bundle(self) -> H3ModelBundle:
        load_device = (
            "cpu"
            if self.config.distributed.strategy == "fsdp2" and self.context.world_size > 1
            else self.device
        )
        roles: dict[str, Any] = {}
        for role in ("student", "teacher", "critic"):
            adapter = build_model_adapter(self.config, role, load_device)
            roles[role] = configure_parallel_model(
                adapter,
                role,
                self.config.distributed,
                self.context,
            )
        return H3ModelBundle(roles["student"], roles["teacher"], roles["critic"])

    def strict_load(self) -> tuple[H3ModelAdapter, dict[str, Any]]:
        model = H3ModelAdapter.from_pretrained(
            self.config.model.pretrained_path,
            subfolder=self.config.model.transformer_subfolder,
            loader=self.config.model.loader,
            dtype=torch.bfloat16,
            strict=True,
            device=self.device,
        ).eval()
        if self.config.model.attention_backend:
            model.transformer.set_attention_backend(self.config.model.attention_backend)
        dtypes: dict[str, int] = {}
        for parameter in model.parameters():
            key = str(parameter.dtype).removeprefix("torch.")
            dtypes[key] = dtypes.get(key, 0) + parameter.numel()
        return model, {"passed": True, "parameters_by_dtype": dtypes}

    @torch.no_grad()
    def forward_parity(
        self,
        model: H3ModelAdapter,
        fixture: dict[str, Any],
        oracle: dict[str, Any],
        *,
        atol: float = 2e-2,
        rtol: float = 2e-2,
    ) -> dict[str, Any]:
        batch = batch_from_fixture(fixture, self.device)
        layout = build_packed_layout(
            text_token_tags=batch.text_token_tags,
            num_latent_frames=batch.video_latents.shape[2],
            latent_height=batch.video_latents.shape[3],
            latent_width=batch.video_latents.shape[4],
            num_audio_latents=batch.audio_latents.shape[3],
            patch_size=self.config.geometry.patch_size,
            audio_channels=self.config.geometry.audio_channels,
        ).to(self.device)
        level = H3DualNoiseScheduler(12, 3).level(float(fixture.get("q", 0.5)), self.device)
        output = model(
            batch.video_latents,
            batch.audio_latents,
            batch.prompt_embeds.to(torch.bfloat16),
            layout=layout,
            video_timestep=level.video_timestep,
            audio_timestep=level.audio_timestep,
        )
        video = compare_tensor(output.video, oracle["video_velocity"], atol=atol, rtol=rtol)
        audio = compare_tensor(output.audio, oracle["audio_velocity"], atol=atol, rtol=rtol)
        return {
            "passed": video.passed and audio.passed,
            "video": asdict(video),
            "audio": asdict(audio),
        }

    @torch.no_grad()
    def dense_parity(
        self,
        model: H3ModelAdapter,
        fixture: dict[str, Any],
        oracle: dict[str, Any],
        *,
        atol: float = 3e-2,
        rtol: float = 3e-2,
    ) -> dict[str, Any]:
        batch = batch_from_fixture(fixture, self.device)
        pipeline = H3InferencePipeline(model, self.config)
        output = pipeline.generate_latents(
            batch.prompt_embeds,
            batch.text_token_tags,
            height=int(fixture["height"]),
            width=int(fixture["width"]),
            num_frames=int(fixture["num_frames"]),
            num_inference_steps=50,
            q_schedule=torch.linspace(1.0, 0.0, 50, dtype=torch.float32).tolist(),
            video_latents=fixture["initial_video_latents"],
            audio_latents=fixture["initial_audio_latents"],
        )
        if output.num_forwards != 49:
            raise AssertionError(f"dense H3 inference must make 49 forwards, got {output.num_forwards}")
        video = compare_tensor(output.video_latents, oracle["video_latents"], atol=atol, rtol=rtol)
        audio = compare_tensor(output.audio_latents, oracle["audio_latents"], atol=atol, rtol=rtol)
        return {
            "passed": video.passed and audio.passed,
            "num_forwards": output.num_forwards,
            "video": asdict(video),
            "audio": asdict(audio),
        }

    def rf_backward(self, models: H3ModelBundle, fixture: dict[str, Any]) -> dict[str, Any]:
        batch = batch_from_fixture(fixture, self.device)
        method = H3DMDMethod(self.config, models)
        result = method.rf_loss(batch)
        result.loss.backward()
        finite = True
        total_norm = torch.zeros((), device=self.device)
        count = 0
        for parameter in models.student_parameters():
            if parameter.grad is not None:
                finite = finite and bool(torch.isfinite(parameter.grad).all())
                total_norm += parameter.grad.float().square().sum()
                count += 1
        passed = finite and count > 0 and float(total_norm) > 0
        return {
            "passed": passed,
            "loss": float(result.loss.detach()),
            "gradient_tensors": count,
            "gradient_norm": float(total_norm.sqrt()),
        }

    def dmd_one_step(self, models: H3ModelBundle, fixture: dict[str, Any]) -> dict[str, Any]:
        batch = batch_from_fixture(fixture, self.device)
        method = H3DMDMethod(self.config, models)
        critic_optimizer = torch.optim.AdamW(models.critic_parameters(), lr=1e-6)
        student_optimizer = torch.optim.AdamW(models.student_parameters(), lr=1e-6)
        critic = method.critic_loss(batch, 0)
        critic.loss.backward()
        critic_optimizer.step()
        critic_optimizer.zero_grad(set_to_none=True)
        student = method.student_loss(batch)
        student.loss.backward()
        student_optimizer.step()
        return {
            "passed": bool(torch.isfinite(critic.loss) and torch.isfinite(student.loss)),
            "critic_loss": float(critic.loss.detach()),
            "student_loss": float(student.loss.detach()),
        }

    def end_to_end(
        self,
        checkpoint: str | Path,
        prompt: str,
        *,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        pipeline = H3InferencePipeline.from_pretrained(
            self.config,
            device=self.device,
            load_encoders=False,
        )
        pipeline.load_student_checkpoint(checkpoint)
        if not self.context.is_main:
            return {"passed": True, "worker_skipped_decode": True}
        pipeline.encoders = H3OfflineCacheEncoder.from_pretrained(
            self.config.model.pretrained_path,
            device="cpu",
            dtype=torch.bfloat16,
        )
        output = pipeline(
            prompt,
            height=768,
            width=1344,
            num_frames=124,
            generator=torch.Generator(device="cpu").manual_seed(self.config.training.seed),
        )
        if output.videos is None or output.audio is None:
            raise AssertionError("end-to-end H3 inference did not decode both modalities")
        passed = bool(
            torch.isfinite(output.videos).all()
            and torch.isfinite(output.audio).all()
            and output.audio.shape[1] == 2
            and output.num_forwards
            == (
                len(self.config.schedule.inference_q) - 1
                if self.config.schedule.inference_q is not None
                else self.config.schedule.num_inference_steps - 1
            )
        )
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "videos": output.videos.cpu(),
                    "audio": output.audio.cpu(),
                    "sampling_rate": output.sampling_rate,
                    "video_latents": output.video_latents.cpu(),
                    "audio_latents": output.audio_latents.cpu(),
                    "num_forwards": output.num_forwards,
                },
                destination,
            )
        return {
            "passed": passed,
            "video_shape": list(output.videos.shape),
            "audio_shape": list(output.audio.shape),
            "sampling_rate": output.sampling_rate,
            "num_forwards": output.num_forwards,
        }

    def run_ordered(
        self,
        fixture_path: str | Path,
        oracle_path: str | Path,
        *,
        include_training: bool = False,
        distilled_checkpoint: str | Path | None = None,
        prompt: str | None = None,
        e2e_output: str | Path | None = None,
    ) -> dict[str, Any]:
        fixture = load_tensor_mapping(fixture_path)
        oracle = load_tensor_mapping(oracle_path)
        report: dict[str, Any] = {}
        model, report["01_strict_load"] = self.strict_load()
        report["02_forward_sglang"] = self.forward_parity(model, fixture, oracle)
        report["03_dense_49_sglang"] = self.dense_parity(model, fixture, oracle)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if include_training:
            models = self._training_bundle()
            report["04_rf_backward"] = self.rf_backward(models, fixture)
            for parameter in models.student.parameters():
                parameter.grad = None
            report["05_dmd_one_step"] = self.dmd_one_step(models, fixture)
            del models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if distilled_checkpoint is not None:
            if not prompt:
                raise ValueError("a prompt is required for end-to-end checkpoint validation")
            report["06_distilled_e2e_av"] = self.end_to_end(
                distilled_checkpoint,
                prompt,
                output_path=e2e_output,
            )
        report["passed"] = all(
            stage.get("passed", False) for stage in report.values() if isinstance(stage, dict)
        )
        return report


def save_report(report: dict[str, Any], path: str | Path | None = None) -> str:
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if path is not None:
        Path(path).write_text(serialized + "\n", encoding="utf-8")
    return serialized


__all__ = [
    "H3ValidationSuite",
    "TensorComparison",
    "batch_from_fixture",
    "compare_tensor",
    "load_tensor_mapping",
    "save_report",
]
