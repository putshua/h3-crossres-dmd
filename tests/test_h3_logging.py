"""Metrics must reach the trainer's stdout/file sinks under preconfigured logging."""
import json
import logging
from types import SimpleNamespace

import torch

from h3dmd.logging import H3Logger
from imaginaire.utils import log
from rcm.callbacks.h3 import H3TrainingCallback


def test_callback_logs_to_upstream_stdout_file_and_json_with_warning_root(tmp_path, monkeypatch, capsys):
    # Hydra/third-party imports can configure the Python root logger before H3.
    # basicConfig(INFO) is then a no-op; the trainer uses independent Loguru sinks.
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [logging.NullHandler()])
    monkeypatch.setattr(root, "level", logging.WARNING)
    monkeypatch.setattr(logging.getLogger("h3dmd"), "level", logging.NOTSET)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    stdout_file = tmp_path / "stdout.log"
    log.init_loguru_stdout()
    log.init_loguru_file(str(stdout_file))
    callback = H3TrainingCallback()
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)),
                                      trainer=SimpleNamespace(logging_iter=1))
    model = SimpleNamespace(h3=SimpleNamespace(logging=SimpleNamespace(tensorboard=False, jsonl=True)),
                            optimizer_dict={})
    try:
        callback.on_train_start(model)
        callback.on_training_step_start(model, {})
        callback.on_training_step_end(model, {"video_latents": torch.zeros(1, 24, 37, 2, 2)},
                                      {"student/dmd/loss": torch.tensor(.25)}, None, iteration=7)
        callback.on_train_end(model)
        log.logger.complete()
        assert "step=7" in capsys.readouterr().out
        assert "student/dmd/loss=0.25" in stdout_file.read_text()
        row = json.loads((tmp_path / "logs/metrics.jsonl").read_text())
        assert row["step"] == 7 and row["student/dmd/loss"] == .25
    finally:
        log.logger.remove()


def test_disabled_logger_does_not_emit_or_write(tmp_path):
    messages = []
    logger = H3Logger(tmp_path / "rank1", enabled=False, tensorboard=False,
                      jsonl=True, console_sink=messages.append)
    logger.log(1, {"loss": 1.})
    logger.close()
    assert messages == [] and not (tmp_path / "rank1").exists()


def test_ten_step_logging_keeps_student_and_critic_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    callback = H3TrainingCallback()
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)),
                                      trainer=SimpleNamespace(logging_iter=10))
    model = SimpleNamespace(h3=SimpleNamespace(logging=SimpleNamespace(tensorboard=False, jsonl=True)),
                            optimizer_dict={})
    callback.on_train_start(model)
    data = {"video_latents": torch.zeros(1, 24, 2, 2, 2)}
    for step in range(1, 21):
        callback.on_training_step_start(model, data)
        role = "student/dmd" if (step - 1) % 5 == 0 else "critic/rf"
        callback.on_training_step_end(model, data,
            {f"{role}/loss": torch.tensor(float(step), requires_grad=True)}, None, iteration=step)
        assert all(v.grad_fn is None and not v.requires_grad for v in callback.pending_metrics.values())
    callback.on_train_end(model)
    rows = [json.loads(line) for line in (tmp_path / "logs/metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [10, 20]
    assert [r["student/dmd/loss"] for r in rows] == [6, 16]
    assert [r["critic/rf/loss"] for r in rows] == [10, 20]
    assert callback.pending_metrics == {}
