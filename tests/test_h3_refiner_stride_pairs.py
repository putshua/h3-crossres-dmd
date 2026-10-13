import pytest

from h3dmd.stride_pairs import source_windows, stride_for_fps, window_starts


def test_refiner_stride_map_and_playback_speed():
    for fps, stride in [(24,1),(25,1),(30,1),(48,2),(50,2),(60,2),(72,3),
                        (96,4),(100,4),(120,5),(29.97,1),(59.94,2),(23.976,1)]:
        result = stride_for_fps(fps)
        assert result["source_stride"] == stride
        assert result["speed_factor"] == pytest.approx(24 * stride / fps)
    for fps in (0, float("nan"), 15, 90):
        with pytest.raises(ValueError):
            stride_for_fps(fps)


def test_reserve_frames_are_required_and_never_padded():
    assert list(window_starts(125,1,124)) == []
    assert list(window_starts(126,1,124)) == [0]
    assert list(window_starts(250,1,124)) == [0,124]
    assert list(window_starts(250,2,124)) == []
    assert list(window_starts(251,2,124)) == [0]


def test_blacklist_uses_original_seconds_including_reserve():
    row = dict(status="ok",width=1920,height=1080,fps=30.,duration=20.,
               native_frames_estimate=600,blacklist_policy="partial",
               blocked_intervals=[[4.15,4.20]])
    windows = list(source_windows(row,5))
    assert [w["start_native_frame"] for w in windows] == [248,372]
    assert windows[0]["num_frames"] == 124
    assert windows[0]["source_decode_frames"] == 126
    assert windows[0]["speed_factor"] == .8
    row["blocked_intervals"] = []
    with pytest.raises(ValueError):
        list(source_windows(row,5))
