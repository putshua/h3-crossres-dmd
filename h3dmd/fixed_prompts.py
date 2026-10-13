"""Versioned, pre-encoded H3 fixed prompts shared by preparation and evaluation."""
import hashlib
import json
from pathlib import Path

import torch


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def read_prompts(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("fixed prompts must have unique IDs and cannot be empty")
    for row in rows:
        if row["prompt_sha256"] != prompt_hash(row["prompt"]):
            raise ValueError(f"prompt hash mismatch: {row['id']}")
    return rows


def load_fixed_prompts(manifest, *, expected_count=32, expected_layer=50):
    """Read all embeddings on CPU; reject stale or non-H3 caches before training."""
    from safetensors import safe_open
    manifest = Path(manifest)
    rows = read_prompts(manifest)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} fixed prompts, got {len(rows)}")
    sources = {row.get("prompt_source") for row in rows}
    if sources != {None}:
        if len(sources) != 1 or None in sources:
            raise ValueError("fixed prompt cache must reference one consistent PE source")
        source = read_prompts(manifest.parent / sources.pop())
        if [(r["id"], r["prompt_sha256"]) for r in rows] != [(r["id"], r["prompt_sha256"]) for r in source]:
            raise ValueError("PE source changed: re-encode the fixed prompts")
    for row in rows:
        path = manifest.parent / row["cache_path"]
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            if metadata.get("prompt_sha256") != row["prompt_sha256"]:
                raise ValueError(f"stale prompt embedding: {path}")
            if metadata.get("text_encoder_layer") != str(expected_layer):
                raise ValueError(f"wrong H3 text encoder layer: {path}")
            embeddings = f.get_tensor("prompt_embeds")
            tags = f.get_tensor("text_token_tags")
        if (embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] != 5120
                or embeddings.dtype != torch.bfloat16 or not torch.isfinite(embeddings).all()):
            raise ValueError(f"expected finite BF16 H3 embeddings [L,5120]: {path}")
        if tags.dtype != torch.int64 or tags.shape != embeddings.shape[:1] or not (tags == 1).all():
            raise ValueError(f"expected all-text H3 token tags: {path}")
        row["prompt_embeds"], row["text_token_tags"] = embeddings, tags
    return rows


def rank_prompt_slots(count, world_size, rank):
    """Equal forward counts on every FSDP rank, including a padded last round."""
    if count <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid prompt count or distributed rank")
    for start in range(0, count, world_size):
        index = start + rank
        yield index % count, index < count
