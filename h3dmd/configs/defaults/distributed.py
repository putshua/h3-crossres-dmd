from h3dmd.registry import CONFIG_GROUPS


@CONFIG_GROUPS.register("distributed/fsdp32")
def fsdp32():
    return {
        "distributed": {
            "backend": "nccl",
            "strategy": "fsdp2",
            "fsdp_size": 32,
            "fsdp_reshard_after_forward": True,
            "sequence_parallel_size": 1,
            "reduce_dtype": "float32",
        }
    }
