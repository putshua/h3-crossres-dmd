"""Standalone/legacy entrypoint for the same registered Hydra experiment."""
from h3dmd.registry import EXPERIMENTS
from rcm.configs.experiments.h3.crossres import EXPERIMENT_NAME, experiment_node


@EXPERIMENTS.register(EXPERIMENT_NAME)
def h3_crossres_pure_dmd():
    node = experiment_node()
    native = node["model"]["config"]["h3"]
    native["defaults"] = ["model/h3_full", "distributed/fsdp32"]
    native["training"].update(learning_rate=node["optimizer"]["lr"],
                              critic_learning_rate=node["model"]["config"]["optimizer_fake_score"]["lr"],
                              max_steps=node["trainer"]["max_iter"])
    native["logging"] = {"log_every": 1}
    native["checkpoint"] = {"output_dir": "outputs/h3-crossres-pure-dmd-320to480-traj4-vshift6",
                            "save_every": 500}
    return native
