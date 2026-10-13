from __future__ import annotations

import argparse
from pathlib import Path

import torch

from h3dmd.cli.common import add_config_arguments, config_from_args
from h3dmd.inference import H3InferencePipeline


def _parse_q(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value.split(",")]


def _save_output(path: Path, output, fps: int) -> None:
    if path.suffix.lower() == ".pt":
        torch.save(
            {
                "videos": output.videos.cpu() if output.videos is not None else None,
                "audio": output.audio.cpu() if output.audio is not None else None,
                "sampling_rate": output.sampling_rate,
                "video_latents": output.video_latents.cpu(),
                "audio_latents": output.audio_latents.cpu(),
                "num_forwards": output.num_forwards,
            },
            path,
        )
        return
    if path.suffix.lower() != ".mp4":
        raise ValueError("inference output must end in .pt or .mp4")
    if output.videos is None or output.audio is None:
        raise ValueError("MP4 export requires decoded video and audio")
    try:
        from torchvision.io import write_video
    except ImportError as exc:
        raise ImportError("MP4 export requires h3dmd[media] (torchvision + PyAV)") from exc
    frames = (output.videos[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    write_video(
        str(path),
        frames,
        fps=fps,
        video_codec="libx264",
        audio_array=output.audio[0].cpu(),
        audio_fps=output.sampling_rate,
        audio_codec="aac",
        options={"crf": "18"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax-H3 conditional-only T2VA inference")
    add_config_arguments(parser)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", help="distilled student checkpoint directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--encoder-device",
        default="cpu",
        help="Qwen/video-VAE/audio-VAE device; kept separate from the denoiser by default",
    )
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--q", help="comma-separated shared q grid, e.g. 1,.7,.35,0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = config_from_args(args)
    pipeline = H3InferencePipeline.from_pretrained(
        config,
        device=args.device,
        load_encoders=True,
        encoder_device=args.encoder_device,
    )
    if args.checkpoint:
        pipeline.load_student_checkpoint(args.checkpoint)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    output = pipeline(
        args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        q_schedule=_parse_q(args.q),
        generator=generator,
    )
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_output(destination, output, config.geometry.fps)
    print(destination)


if __name__ == "__main__":
    main()
