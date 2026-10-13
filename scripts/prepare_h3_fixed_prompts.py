"""Prepare the reviewed high-motion H3 PE set, preserving source order and provenance."""
import argparse
import hashlib
import json
import re
from pathlib import Path

FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
DEFAULT_SOURCE = "test_prompts/h3_high_motion_proportional_32_extended.json"
DEFAULT_OUTPUT = "test_prompts/h3_high_motion_32"


def parse_h3_prompt(prompt):
    matches = list(re.finditer(r"(?m)^(integrated_multimodal_description|overall_soundscape|non_diegetic_music):[ \t]*", prompt))
    if tuple(m.group(1) for m in matches) != FIELDS or prompt[:matches[0].start()].strip():
        raise ValueError("expected exactly the three ordered H3 PE fields")
    fields = {m.group(1): prompt[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(prompt)].strip()
              for i, m in enumerate(matches)}
    if not all(fields.values()):
        raise ValueError("H3 PE fields must be nonempty")
    return fields


def prepare(source_path, output_dir):
    source_path, root = Path(source_path), Path(output_dir)
    source = source_path.read_text()
    bundle = json.loads(source)
    prompts = bundle["prompts"]
    annotations = json.loads((root / "audio_annotations.json").read_text())
    if len(prompts) != 32 or len(annotations) != 32:
        raise ValueError("expected 32 high-motion prompts and matching audio annotations")
    if [r["index"] for r in prompts] != list(range(32)):
        raise ValueError("source indices must preserve the ordered range 0..31")
    rows = []
    for index, (row, audio) in enumerate(zip(prompts, annotations)):
        fields = parse_h3_prompt(row["h3_prompt"])
        fields["integrated_multimodal_description"] += "\nAudio: " + audio["audio"]
        for key in FIELDS[1:]:
            if key in audio:
                fields[key] = audio[key]
        # A visible drum performance and the block-collapse scene have no score.
        if index in (5, 10, 29):
            fields["non_diegetic_music"] = "N/A"
        else:
            fields["non_diegetic_music"] += " Instrumental only, mixed below foreground scene sounds."
        prompt = "\n\n".join(f"{k}: {fields[k]}" for k in FIELDS)
        rows.append({"id": f"high_motion_{index + 1:02d}", "source_index": row["index"],
                     "source_line": row["source_line"], "source_prompt": row["source_prompt"],
                     "source_h3_prompt": row["h3_prompt"], "h3_fields": fields, "prompt": prompt,
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
    (root / "prompts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (root / "provenance.json").write_text(json.dumps({
        "source": str(source_path), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "count": 32, "duration_seconds": bundle["metadata"]["duration_seconds"],
        "source_metadata": bundle["metadata"],
        "review": "Preserve all source visual rewrites and order. Add action-synchronized diegetic audio; keep existing scores instrumental and below scene sound; no separate score for visible drums/dragon or wooden blocks. Clarify underwater/poolside perspectives, gym floor contacts and nonverbal crew reactions.",
        "audio_annotations_sha256": hashlib.sha256((root / "audio_annotations.json").read_bytes()).hexdigest(),
        "format": "Three-field H3 PE, retaining source 10-second shot timestamps; direct tokenization, no chat template.",
    }, ensure_ascii=False, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = prepare(args.source, args.output_dir)
    print(f"Prepared {len(rows)} prompts: {args.output_dir}/prompts.jsonl", flush=True)


if __name__ == "__main__":
    main()
