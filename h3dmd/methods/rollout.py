from __future__ import annotations

import torch
from h3dmd.data import H3Batch
from h3dmd.h3.scheduler import H3NoiseLevel, truncated_q_grid
from h3dmd.methods.types import H3Latents


class FewStepRolloutMixin:
    """Coupled RF Euler or x0/re-noise simulation with truncated gradients."""

    def prepare_rollout_transition(self, clean, batch, next_forward):
        """Optional clean-space resolution change before the next re-noising."""
        return clean, batch

    def rollout(
        self,
        batch: H3Batch,
        *,
        requires_grad: bool,
        initial: H3Latents | None = None,
        num_steps: int | None = None,
    ) -> H3Latents:
        """Run the student's coupled few-step trajectory with truncated BPTT."""

        layout = self.build_layout(batch)
        count = num_steps if num_steps is not None else self.config.dmd.rollout_steps
        if not 1 <= count <= self.config.dmd.rollout_steps:
            raise ValueError("num_steps must be in [1, rollout_steps]")
        grid = self.config.schedule.inference_q
        if grid is None:
            grid = torch.linspace(1, 0, self.config.dmd.rollout_steps + 1).tolist()
        # Bidirectional uses a prefix of its configured denoising times,
        # followed by a direct prediction of x0 on the final forward.
        grid = truncated_q_grid(grid, count)
        schedule = self.scheduler.schedule(
            count + 1,
            q=grid,
            device=batch.video_latents.device,
        )
        if initial is None:
            state = H3Latents(
                torch.randn_like(batch.video_latents, dtype=torch.float32),
                torch.randn_like(batch.audio_latents, dtype=torch.float32),
            )
        else:
            state = initial
        grad_from = max(0, schedule.num_forwards - self.config.dmd.rollout_grad_steps)
        for index in range(schedule.num_forwards):
            use_grad = requires_grad and index >= grad_from
            context = torch.enable_grad() if use_grad else torch.no_grad()
            with context:
                level = H3NoiseLevel(
                    q=schedule.q[index],
                    video_sigma=schedule.video_sigmas[index],
                    audio_sigma=schedule.audio_sigmas[index],
                    video_timestep=schedule.video_timesteps[index],
                    audio_timestep=schedule.audio_timesteps[index],
                )
                velocity = self._call("student", state, batch, layout, level)
                if self.config.dmd.rollout_solver == "renoise":
                    clean = H3Latents(
                        self.scheduler.to_x0(
                            state.video.float(),
                            velocity.video.float(),
                            level.video_sigma,
                        ),
                        self.scheduler.to_x0(
                            state.audio.float(),
                            velocity.audio.float(),
                            level.audio_sigma,
                        ),
                    )
                    if index < schedule.num_forwards - 1:
                        clean, next_batch = self.prepare_rollout_transition(
                            clean, batch, next_forward=index + 2)
                        if next_batch is not batch:
                            batch = next_batch
                            layout = self.build_layout(batch)
                    state = (
                        clean
                        if index == schedule.num_forwards - 1
                        else self._noise(
                            clean, self.scheduler.level(schedule.q[index + 1])
                        )[0]
                    )
                else:
                    state = H3Latents(
                        self.scheduler.euler_step(
                            state.video,
                            velocity.video,
                            schedule.video_sigmas[index],
                            schedule.video_sigmas[index + 1],
                        ),
                        self.scheduler.euler_step(
                            state.audio,
                            velocity.audio,
                            schedule.audio_sigmas[index],
                            schedule.audio_sigmas[index + 1],
                        ),
                    )
            if not use_grad:
                state = state.detach()
        return state
