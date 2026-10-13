"""H3 metrics and clipping using the upstream callback lifecycle."""
from pathlib import Path
import time
import torch
from imaginaire.utils.callback import Callback
from imaginaire.utils import distributed, log
from h3dmd.distributed import reduce_mean
from h3dmd.logging import H3Logger


class H3TrainingCallback(Callback):
    def on_train_start(self, model, iteration=0):
        self.logger = H3Logger(Path(self.config.job.path_local) / "logs",
            enabled=distributed.is_rank0(), tensorboard=model.h3.logging.tensorboard,
            jsonl=model.h3.logging.jsonl, console_sink=log.info)
        self.grad_norms = {}
        self.pending_metrics = {}
        self._timed_iteration = None

    def on_training_step_start(self, model, data, iteration=0):
        if getattr(self, "_timed_iteration", None) != iteration:
            self.started = time.monotonic()
            self._timed_iteration = iteration

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        grad_scaler.unscale_(optimizer)
        parameters = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
        max_norm = model.h3.training.max_grad_norm if model.h3.training.grad_clip_enabled else float("inf")
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm,
                                              error_if_nonfinite=True)
        key = "student" if optimizer is model.optimizer_dict["net"] else "critic"
        self.grad_norms[f"grad/{key}"] = reduce_mean(norm).detach()
        coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.)
        self.grad_norms[f"grad_clip_coefficient/{key}"] = reduce_mean(coefficient).detach()

    def on_training_step_end(self, model, data, output_batch, loss, iteration=0):
        # An interval divisible by the student update frequency can otherwise
        # log only critic losses. Keep the latest observation of each metric
        # within this interval, detached so no computation graph is retained.
        self.pending_metrics.update({k: v.detach() for k, v in output_batch.items()})
        if iteration % self.config.trainer.logging_iter:
            return
        metrics = {k: reduce_mean(v) for k, v in self.pending_metrics.items()}
        metrics.update(self.grad_norms)
        metrics["time/step_seconds"] = time.monotonic() - self.started
        metrics["data/num_frames"] = 17 * ((data["video_latents"].shape[2] - 2) // 5) + 5
        for key, opt in model.optimizer_dict.items():
            metrics[f"lr/{key}"] = opt.param_groups[0]["lr"]
        if torch.cuda.is_available():
            metrics["memory/peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        self.logger.log(iteration, metrics)
        self.pending_metrics.clear()
        self.grad_norms.clear()

    def on_train_end(self, model, iteration=0):
        self.logger.close()
