from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from h3dmd.config import H3Config
from h3dmd.h3.adapter import H3ModelAdapter


class H3ModelBundle(nn.Module):
    """Explicit ownership of the student, frozen teacher, and fake-score critic."""

    def __init__(self, student: H3ModelAdapter, teacher: H3ModelAdapter, critic: H3ModelAdapter) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher.freeze()
        self.critic = critic

    def train(self, mode: bool = True) -> "H3ModelBundle":
        self.student.train(mode)
        self.critic.train(mode)
        self.teacher.eval()
        return self

    def student_parameters(self):
        return (parameter for parameter in self.student.parameters() if parameter.requires_grad)

    def critic_parameters(self):
        return (parameter for parameter in self.critic.parameters() if parameter.requires_grad)


def build_model_adapter(
    config: H3Config,
    role: str,
    device: str | torch.device | None = None,
) -> H3ModelAdapter:
    if role not in {"student", "teacher", "critic"}:
        raise ValueError(f"unknown H3 model role {role!r}")
    adapter = H3ModelAdapter.from_pretrained(
        pretrained_path=config.model.pretrained_path,
        subfolder=config.model.transformer_subfolder,
        loader=config.model.loader,
        dtype=torch.bfloat16,
        strict=True,
        device=device,
    )
    if config.model.attention_backend:
        if not hasattr(adapter.transformer, "set_attention_backend"):
            raise NotImplementedError("this Diffusers H3 build has no selectable attention backend")
        adapter.transformer.set_attention_backend(config.model.attention_backend)
    if role in {"student", "critic"} and config.model.gradient_checkpointing:
        adapter.enable_gradient_checkpointing()
    if role == "student":
        adapter.configure_lora(
            path=config.model.lora_path,
            rank=config.model.lora_rank,
            alpha=config.model.lora_alpha,
            target_modules=config.model.lora_target_modules,
        )
    elif role == "critic":
        adapter.configure_lora(
            rank=config.model.critic_lora_rank,
            alpha=config.model.critic_lora_alpha,
            target_modules=config.model.lora_target_modules,
        )
    else:
        adapter.freeze()
    return adapter


def build_model_bundle(config: H3Config, device: str | torch.device | None = None) -> H3ModelBundle:
    student = build_model_adapter(config, "student", device)
    teacher = build_model_adapter(config, "teacher", device)
    critic = build_model_adapter(config, "critic", device)
    return H3ModelBundle(student, teacher, critic)


__all__ = ["H3ModelBundle", "build_model_adapter", "build_model_bundle"]
