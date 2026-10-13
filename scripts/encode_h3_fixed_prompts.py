"""Encode H3 PE JSONL once; no DiT, VAE, or online training dependency on Qwen."""
import argparse
import json
import os
from pathlib import Path

from h3dmd.fixed_prompts import load_fixed_prompts, read_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="test_prompts/h3_high_motion_32/prompts.jsonl")
    parser.add_argument("--output-dir", default="test_prompts/h3_high_motion_32/encoded")
    parser.add_argument("--pretrained-path", default="/mnt/data/butong/mmh3/MiniMax-H3/FL2VA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = read_prompts(args.prompts)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "manifest.jsonl"
    if manifest.exists() and not args.overwrite:
        existing = load_fixed_prompts(manifest, expected_count=len(rows), expected_layer=args.layer)
        if [r["prompt_sha256"] for r in existing] != [r["prompt_sha256"] for r in rows]:
            raise ValueError("existing embeddings are for different prompts; use --overwrite")
        print(f"Already encoded and validated: {manifest}", flush=True)
        return
    from h3dmd.cache import H3TextEncoder
    from safetensors.torch import save_file
    print(f"Loading H3 text encoder on {args.device}", flush=True)
    encoder = H3TextEncoder.from_pretrained(args.pretrained_path, device=args.device)
    for i, row in enumerate(rows):
        embeddings, tags = encoder.encode_text(row["prompt"], args.layer)
        row = rows[i] = dict(row, cache_path=f"{row['id']}.safetensors",
                             text_encoder_layer=args.layer, num_tokens=len(tags),
                             prompt_source=os.path.relpath(Path(args.prompts).resolve(), destination.resolve()),
                             pretrained_path=str(Path(args.pretrained_path).resolve()))
        metadata = {"prompt_sha256": row["prompt_sha256"], "text_encoder_layer": str(args.layer),
                    "pretrained_path": row["pretrained_path"], "format": "h3_text_v1"}
        path = destination / row["cache_path"]
        temp = path.with_suffix(".tmp")
        save_file({"prompt_embeds": embeddings.contiguous(), "text_token_tags": tags}, str(temp), metadata=metadata)
        temp.replace(path)
        print(f"Encoded {i + 1}/{len(rows)}: {row['id']} {tuple(embeddings.shape)}", flush=True)
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    load_fixed_prompts(temporary, expected_count=len(rows), expected_layer=args.layer)
    temporary.replace(manifest)
    print(f"Validated: {manifest}", flush=True)


if __name__ == "__main__":
    main()
