from h3dmd.registry import CONFIG_GROUPS


@CONFIG_GROUPS.register("method/pure_dmd")
def pure_dmd():
    return {
        "method": "h3_distill",
        "dmd": {
            "enabled": True,
            "conditional_only": True,
            "data_mode": "rollout",
            "rf_loss_weight": 0.0,
            "dmd_loss_weight": 1.0,
            "critic_loss_weight": 1.0,
            "critic_update_frequency": 1,
            "rollout_steps": 4,
            "rollout_grad_steps": 1,
            "update_schedule": "alternating",
            "student_update_frequency": 5,
            "rollout_solver": "renoise",
            "cycle_rollout_steps": True,
            "loss_reduction": "mean",
        },
        "pcm": {
            "enabled": False,
            "loss_weight": 1.0,
            "phase_boundaries_q": [1.0, 0.75, 0.5, 0.25, 0.0],
            "steps_per_phase": 24,
            "skipping_interval_steps": 1,
            "warmup_steps": 0,
            "warmup_only": False,
        },
    }


@CONFIG_GROUPS.register("method/pcm_dmd")
def pcm_dmd():
    config = pure_dmd()
    config["pcm"]["enabled"] = True
    return config


@CONFIG_GROUPS.register("method/pcm")
def pcm():
    config = pcm_dmd()
    config["dmd"]["dmd_loss_weight"] = 0.0
    config["dmd"]["rollout_solver"] = "euler"
    config["dmd"]["cycle_rollout_steps"] = False
    return config
