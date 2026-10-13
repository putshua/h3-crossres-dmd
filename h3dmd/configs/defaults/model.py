from h3dmd.registry import CONFIG_GROUPS


@CONFIG_GROUPS.register("model/h3_full")
def h3_full():
    return {
        "model": {
            "pretrained_path": "/mnt/data/butong/mmh3/MiniMax-H3/FL2VA",
            "loader": "legacy_fl2va",
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
            "lora_rank": 0,
            "critic_lora_rank": 0,
        }
    }
