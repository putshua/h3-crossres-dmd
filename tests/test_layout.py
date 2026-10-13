import torch

from h3dmd.h3.layout import (
    AUDIO_TAG,
    PADDING_TAG,
    build_packed_layout,
    pack_audio,
    patchify_video,
    unpack_audio,
    unpatchify_video,
)


def test_t2va_layout_matches_sglang_row_contract() -> None:
    layout = build_packed_layout(
        text_token_tags=torch.tensor([1, 0, 1]),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=4,
        alignment=64,
    )
    assert layout.used_sequence_length == 19
    assert layout.sequence_length == 64
    assert layout.cu_seqlens.tolist() == [0, 19, 64]
    assert layout.text_indices.tolist() == [0, 1, 2]
    assert layout.audio_indices.tolist() == list(range(3, 11))
    assert layout.video_indices.tolist() == list(range(11, 19))
    assert torch.all(layout.token_tags[layout.audio_indices] == AUDIO_TAG)
    assert torch.all(layout.token_tags[layout.video_indices] == 0)
    assert torch.all(layout.token_tags[19:] == PADDING_TAG)
    assert layout.position_ids.dtype == torch.float64
    assert layout.position_ids[:3, 0].tolist() == [0.0, 1.0, 2.0]
    assert layout.position_ids[3:11, 0].tolist() == [3, 4, 5, 6, 3, 4, 5, 6]
    assert layout.position_ids[3:7, 2].tolist() == [0, 0, 0, 0]
    assert layout.position_ids[7:11, 2].tolist() == [16, 16, 16, 16]
    assert layout.position_ids[11, 0] == 3
    assert torch.isclose(layout.position_ids[15, 0], torch.tensor(14 / 3, dtype=torch.float64))


def test_video_and_audio_pack_round_trip() -> None:
    layout = build_packed_layout(
        text_token_tags=torch.ones(2, dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=6,
        num_audio_latents=5,
    )
    video = torch.arange(2 * 24 * 2 * 4 * 6).reshape(2, 24, 2, 4, 6).float()
    video_rows = patchify_video(video, (1, 2, 2))
    assert video_rows.shape == (2, 12, 96)
    assert torch.equal(unpatchify_video(video_rows, layout, 24), video)

    audio = torch.arange(2 * 2 * 32 * 5).reshape(2, 2, 32, 5).float()
    audio_rows = pack_audio(audio)
    assert audio_rows.shape == (2, 10, 32)
    assert torch.equal(unpack_audio(audio_rows, 2, 32), audio)


def test_row_timestep_table_uses_float_t_not_integer_indices() -> None:
    layout = build_packed_layout(
        text_token_tags=torch.ones(2, dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
    )
    unique, inverse = layout.build_row_timesteps(0.125, 0.75)
    assert torch.equal(unique, torch.tensor([0.125, 0.75]))
    assert torch.all(inverse[layout.audio_indices] == 1)
    assert torch.all(inverse[layout.video_indices] == 0)
    assert torch.all(inverse[layout.text_indices] == 0)
