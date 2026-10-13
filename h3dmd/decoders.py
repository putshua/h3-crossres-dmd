"""VAE-only loading for H3 sampling, including the original FL2VA export."""
import json
import hashlib
import importlib
import importlib.machinery
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch


class LegacyVAEDecoder(torch.nn.Module):
    """Adapt the released raw-tensor VAEs to the Diffusers decode contract."""
    def __init__(self, model, config, *, video):
        super().__init__()
        self.model = model
        # Normalization constants live in the outer export config, not the
        # legacy network's inner architecture config.
        self.config = SimpleNamespace(**config)
        self.video = video

    def decode(self, latents, return_dict=False):
        if return_dict:
            raise ValueError("H3 legacy decoder supports return_dict=False")
        # decode_base performs temporal chunking/blending for 17*n+5 frames.
        # Calling the raw legacy video decode would bypass this contract.
        samples = self.model.decode_base(latents) if self.video else self.model.decode(latents)
        return (samples,)


def _local_vae_class(component_dir, entrypoint):
    """Import a trusted local VAE bundle without HF's shared flat code cache.

    Diffusers copies local modules such as conv.py into the same directory on
    every rank without an inter-process lock. Import directly from an isolated
    package per resolved component path: no shared copy/truncate race and no
    collision between different local VAE bundles' relative imports.
    """
    component_dir = Path(component_dir).resolve()
    module_name, class_name = entrypoint.rsplit(".", 1)
    if not all(part.isidentifier() for part in module_name.split(".")) or not class_name.isidentifier():
        raise ValueError(f"Invalid local VAE entrypoint: {entrypoint!r}")
    package_name = "_h3_local_vae_" + hashlib.sha256(str(component_dir).encode()).hexdigest()[:24]
    package = ModuleType(package_name)
    package.__path__ = [str(component_dir)]
    package.__package__ = package_name
    package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
    sys.modules.setdefault(package_name, package)
    module = importlib.import_module(f"{package_name}.{module_name}")
    return getattr(module, class_name)


def load_h3_vaes(path, device, components=("video_vae", "audio_vae")):
    root = Path(path)
    names = tuple(components)
    if not names or any(name not in {"video_vae", "audio_vae"} for name in names):
        raise ValueError("Unknown H3 VAE component")
    components = {}
    for name in names:
        config = json.loads((root / name / "config.json").read_text())
        if "source_safetensors_path" in config:
            model_class = _local_vae_class(root / name, config["auto_map"]["AutoModel"])
            model = model_class.from_pretrained(str(root / name), dtype=torch.float32)
            model = LegacyVAEDecoder(model, config, video=name == "video_vae")
        else:
            from diffusers import AutoModel
            model = AutoModel.from_pretrained(str(root / name), trust_remote_code=True, dtype=torch.float32)
        components[name] = model.requires_grad_(False).eval().to(device)
    return SimpleNamespace(**components)
