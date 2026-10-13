"""H3 network factory: preserve native mixed dtypes and shard each role early."""
from h3dmd.models import build_model_adapter
from h3dmd.distributed import configure_parallel_model


def build_h3_network(config, role, context):
    device = "cpu" if context.world_size > 1 else context.device
    adapter = build_model_adapter(config, "student" if role == "ema" else role, device=device)
    if role == "ema":
        adapter.float().requires_grad_(False)
    return configure_parallel_model(adapter, role, config.distributed, context)
