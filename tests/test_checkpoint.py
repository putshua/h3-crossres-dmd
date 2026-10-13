import torch
from torch import nn

from h3dmd.checkpoint import H3CheckpointManager
from h3dmd.config import CheckpointConfig, H3Config
from h3dmd.distributed import DistributedContext
from h3dmd.models import H3ModelBundle


class Role(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(value)))

    def freeze(self):
        self.requires_grad_(False)
        return self


def test_single_process_checkpoint_round_trip_and_prune(tmp_path) -> None:
    config = H3Config()
    config.checkpoint = CheckpointConfig(output_dir=str(tmp_path), save_every=1, keep_last=1)
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    manager = H3CheckpointManager(config.checkpoint, context)
    models = H3ModelBundle(Role(1), Role(2), Role(3))
    optimizers = {
        "student": torch.optim.AdamW(models.student_parameters(), lr=1e-3),
        "critic": torch.optim.AdamW(models.critic_parameters(), lr=1e-3),
    }
    manager.save(models, optimizers, 1, config)
    with torch.no_grad():
        models.student.weight.fill_(99)
    manager.save(models, optimizers, 2, config)
    assert not (tmp_path / "checkpoints" / "step_000000001").exists()
    with torch.no_grad():
        models.student.weight.fill_(-1)
    step = manager.load(tmp_path / "checkpoints", models, optimizers)
    assert step == 2
    assert models.student.weight == 99
