from types import SimpleNamespace

import torch

from h3dmd.methods import H3DistillMethod, TrainingClosure, H3LossOutput, build_method
from h3dmd.registry import METHODS
from test_dmd import make_batch, make_method


def test_registered_method_and_closures_allow_new_objective():
    method, models = make_method()

    @METHODS.register("test_hook_extension")
    class Extension(H3DistillMethod):
        def student_objectives(self, ctx):
            yield TrainingClosure(
                "extension",
                "student",
                lambda: H3LossOutput(
                    self.models.student.scale.square(),
                    {"iteration": torch.tensor(ctx.iteration)},
                ),
            )

    config = method.config
    config.method = "test_hook_extension"
    config.dmd.enabled = False
    built = build_method(config, models)
    closures = list(built.training_step_closures(make_batch(), 7))
    assert len(closures) == 1 and closures[0].name == "extension"
    result = closures[0].compute()
    result.loss.backward()
    assert models.student.scale.grad is not None
    assert result.metrics["iteration"] == 7

