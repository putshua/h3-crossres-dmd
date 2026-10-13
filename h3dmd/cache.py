from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


class H3TextEncoder:
    """Text-only H3 conditioner; never loads the DiT or either VAE."""

    def __init__(self, text_encoder: Any, tokenizer: Any, processor: Any) -> None:
        self.text_encoder = text_encoder.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        self.processor = processor

    @classmethod
    def from_pretrained(cls, path: str, *, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        root = Path(path).expanduser()
        # Place weights directly on the destination; avoid a second full CPU copy.
        text_encoder = AutoModelForImageTextToText.from_pretrained(
            root / "text_encoder", dtype=dtype, low_cpu_mem_usage=True,
            device_map={"": str(device)}, local_files_only=True,
        )
        return cls(text_encoder, AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True),
                   AutoProcessor.from_pretrained(root / "processor", local_files_only=True))

    def to(self, device):
        self.text_encoder.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    @torch.no_grad()
    def encode_text_batch(self, prompts: list[str], layer: int = 50):
        """The existing offline pipeline's right-padded PE batching contract."""
        tokens = [self.tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
        if not tokens or any(not x or len(x) > 4096 for x in tokens):
            raise ValueError("H3 text batch requires 1–4096 tokens per prompt")
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        inputs = torch.full((len(tokens), max(map(len,tokens))), pad,
                            dtype=torch.long, device=self.device)
        mask = torch.zeros_like(inputs)
        for i,ids in enumerate(tokens):
            inputs[i,:len(ids)] = torch.tensor(ids,device=self.device)
            mask[i,:len(ids)] = 1
        mm = torch.tensor(self.processor.create_mm_token_type_ids(inputs.tolist()),
                          dtype=torch.long,device=self.device)
        output = self.text_encoder.model(input_ids=inputs, attention_mask=mask,
            mm_token_type_ids=mm, use_cache=False, output_hidden_states=True)
        values = output.hidden_states[layer].to(torch.bfloat16).cpu()
        return [(values[i,:len(ids)].clone().contiguous(),
                 torch.ones(len(ids),dtype=torch.long)) for i,ids in enumerate(tokens)]

    @torch.no_grad()
    def encode_text(self, prompt: str, layer: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("T2VA prompt must be a non-empty string")
        token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        num_layers = self.text_encoder.config.text_config.num_hidden_layers
        if num_layers <= layer:
            raise ValueError(f"Qwen3-VL needs more than {layer} layers, got {num_layers}")
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        mm_ids = torch.tensor(
            self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=self.device
        )
        outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        embeddings = outputs.hidden_states[layer].to(torch.bfloat16).cpu()[0]
        return embeddings, torch.ones(len(token_ids), dtype=torch.long)


class H3OfflineCacheEncoder(H3TextEncoder):
    """Frozen T2VA encoders matching the Diffusers MiniMax-H3 contracts."""

    def __init__(self, text_encoder: Any, tokenizer: Any, processor: Any, video_vae: Any, audio_vae: Any) -> None:
        self.text_encoder = text_encoder.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.video_vae = video_vae.requires_grad_(False).eval()
        self.audio_vae = audio_vae.requires_grad_(False).eval()

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "H3OfflineCacheEncoder":
        local_root = Path(path).expanduser()
        if local_root.is_dir() and not (local_root / "modular_model_index.json").is_file():
            return cls._from_legacy_fl2va(local_root, device=device, dtype=dtype)
        try:
            from diffusers import ModularPipeline
        except ImportError as exc:
            raise ImportError("offline H3 caching requires Diffusers ModularPipeline") from exc
        pipe = ModularPipeline.from_pretrained(path)
        available = set(pipe.pretrained_component_names)
        video_name = "video_vae" if "video_vae" in available else "vae"
        required = ["text_encoder", "tokenizer", "processor", video_name, "audio_vae"]
        missing = set(required) - available
        if missing:
            raise RuntimeError(f"MiniMax-H3 checkpoint lacks encoder components: {sorted(missing)}")
        # Load only frozen/offline components. In particular, never allocate
        # the 61.7 GB transformer in a cache worker.
        pipe.load_components(names=required, dtype=dtype)
        return cls(
            pipe.text_encoder,
            pipe.tokenizer,
            pipe.processor,
            getattr(pipe, video_name),
            pipe.audio_vae,
        ).to(device)

    @classmethod
    def _from_legacy_fl2va(
        cls,
        root: Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> "H3OfflineCacheEncoder":
        """Load only conditioner/VAEs from the original local FL2VA tree."""

        try:
            from diffusers import AutoModel as DiffusersAutoModel
            from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise ImportError("legacy FL2VA caching requires recent Diffusers and Transformers") from exc
        text_encoder = AutoModelForImageTextToText.from_pretrained(
            root / "text_encoder", dtype=dtype, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
        processor = AutoProcessor.from_pretrained(root / "processor")
        # The released local tree declares local AutoModel implementations in
        # auto_map. This is scoped to the user-supplied checkpoint directory.
        video_vae = DiffusersAutoModel.from_pretrained(
            root / "video_vae", trust_remote_code=True, dtype=dtype
        )
        audio_vae = DiffusersAutoModel.from_pretrained(
            root / "audio_vae", trust_remote_code=True, dtype=dtype
        )
        return cls(text_encoder, tokenizer, processor, video_vae, audio_vae).to(device)

    def to(self, device: str | torch.device) -> "H3OfflineCacheEncoder":
        self.text_encoder.to(device)
        self.video_vae.to(device)
        self.audio_vae.to(device)
        return self

    @torch.no_grad()
    def encode_video(self, pixels: torch.Tensor, seed: int = 42) -> torch.Tensor:
        """Encode uint8 video ``[3,T,H,W]`` into normalized data latents."""

        if pixels.ndim != 4 or pixels.shape[0] != 3 or pixels.dtype != torch.uint8:
            raise ValueError("video pixels must be uint8 [3,T,H,W]")
        _, frames, height, width = pixels.shape
        if frames % 17 != 5:
            raise ValueError("T2VA target frames must follow 17*n+5")
        if height % 16 or width % 16:
            raise ValueError("T2VA target height and width must be divisible by 16")
        device = next(self.video_vae.parameters()).device
        values = pixels[None].to(device)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1, 1)
        values = (values.float().div(255.0) - mean) / std
        posterior = self.video_vae.encode(values, return_dict=False)[0]
        latents = posterior.sample(generator=torch.Generator().manual_seed(seed))
        latents = latents.to(torch.float16).float().cpu()
        latent_mean = torch.tensor(self.video_vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latent_std = torch.tensor(self.video_vae.config.latents_std).view(1, -1, 1, 1, 1)
        return ((latents - latent_mean) / latent_std)[0]

    @torch.no_grad()
    def encode_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Encode stereo audio ``[2,N]`` into normalized ``[2,32,T]`` latents."""

        if waveform.ndim != 2 or waveform.shape[0] != 2:
            raise ValueError("audio waveform must be stereo [2,num_samples]")
        expected_rate = getattr(
            self.audio_vae.config,
            "sampling_rate",
            getattr(self.audio_vae.config, "sample_rate", 32_000),
        )
        if sample_rate != expected_rate:
            raise ValueError(
                f"audio must be decoded at the Audio VAE rate {expected_rate}, got {sample_rate}"
            )
        device = next(self.audio_vae.parameters()).device
        posterior = self.audio_vae.encode(waveform.to(device)[:, None], return_dict=False)[0]
        latents = posterior.mode().float().cpu()
        mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, -1, 1)
        std = torch.tensor(self.audio_vae.config.latents_std).view(1, -1, 1)
        return (latents - mean) / std


def load_cached_input(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def cache_manifest(
    manifest: str | Path,
    output_dir: str | Path,
    encoder: H3OfflineCacheEncoder,
    *,
    text_layer: int = 50,
    video_seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Cache a JSONL manifest whose raw video/audio fields point to .pt tensors."""

    manifest_path = Path(manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for index, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        entry = json.loads(line)
        sample_id = str(entry.get("id", f"{index:09d}"))
        cache_path = output / f"{sample_id}.pt"
        if overwrite or not cache_path.exists():
            prompt_embeds, tags = encoder.encode_text(entry["prompt"], text_layer)
            if "audio_sample_rate" not in entry:
                raise KeyError(f"raw manifest sample {sample_id} has no audio_sample_rate")
            pixels = load_cached_input(entry["video_tensor"])
            waveform = load_cached_input(entry["audio_tensor"])
            if isinstance(pixels, dict):
                pixels = pixels.get("video", pixels.get("pixels"))
            if isinstance(waveform, dict):
                waveform = waveform.get("audio", waveform.get("waveform"))
            payload = {
                "prompt_embeds": prompt_embeds,
                "text_token_tags": tags,
                "video_latents": encoder.encode_video(pixels, video_seed),
                "audio_latents": encoder.encode_audio(waveform, int(entry["audio_sample_rate"])),
            }
            temporary = cache_path.with_suffix(f".tmp.{os.getpid()}.pt")
            torch.save(payload, temporary)
            temporary.replace(cache_path)
        entries.append({"id": sample_id, "cache_path": cache_path.name})
    cached_manifest = output / "manifest.jsonl"
    temporary_manifest = output / f"manifest.tmp.{os.getpid()}.jsonl"
    temporary_manifest.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries), encoding="utf-8"
    )
    temporary_manifest.replace(cached_manifest)
    return cached_manifest


__all__ = ["H3OfflineCacheEncoder", "cache_manifest"]
