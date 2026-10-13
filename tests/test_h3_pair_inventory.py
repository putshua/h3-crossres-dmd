"""Source identity, bucket selection and leakage checks for a full pair cache."""
import json

import pytest

from scripts.prepare_h3_upsampler_pairs import normalize, prepare


def test_inventory_deduplicates_and_preserves_source_split(tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    rows = [dict(id="five", source_video=str(source), time_range=[0, 5.125],
                 target_frames=124),
            dict(id="ten", source_video=str(source), time_range=[0, 10.083333],
                 target_frames=243)]
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows + rows[:1]))
    output = tmp_path / "pairs"
    report = prepare([manifest], output)
    saved = [json.loads(line) for line in (output / "input.jsonl").read_text().splitlines()]
    assert report["total_pairs"] == 2
    assert report["unique_source_videos"] == 1
    assert saved[0]["group"] == saved[1]["group"]
    assert saved[0]["split"] == saved[1]["split"]
    with pytest.raises(FileExistsError):
        prepare([manifest], output)


def test_rendered_fallback_keeps_original_group_and_time(tmp_path):
    clip = tmp_path / "rendered.mp4"
    clip.touch()
    row = dict(source_video=str(tmp_path / "missing.mp4"), video=str(clip),
               time_range=[20, 25.125], target_frames=124)
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    output = tmp_path / "pairs"
    report = prepare([manifest], output)
    saved = json.loads((output / "input.jsonl").read_text())
    assert report["media_origins"] == {"reviewed_rendered_clip": 1}
    assert saved["source_time_range"] == [0, 124 / 24]
    assert saved["original_time_range"] == [20, 25.125]
    assert saved["group"] == normalize(row, manifest)["group"]


def test_invalid_or_rejected_clips_are_not_silently_accepted(tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    base = dict(source_video=str(source), time_range=[0, 5.125], target_frames=124)
    rows = [base, dict(base, accepted=False), dict(base, target_frames=120),
            dict(base, repair_issues=["unresolved"])]
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    report = prepare([manifest], tmp_path / "pairs")
    assert report["total_pairs"] == 1
    assert report["rejected"] == 3
