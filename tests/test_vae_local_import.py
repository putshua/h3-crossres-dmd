import json
import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

from h3dmd.decoders import load_h3_vaes


def test_local_vae_relative_imports_isolate_bundles_and_bypass_corrupt_hf_cache(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Legacy VAE must not enter HF dynamic module loader")
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        AutoModel=SimpleNamespace(from_pretrained=fail)))
    # Reproduce the observed symptom: the shared module lacks the class.
    poisoned = ModuleType("diffusers_modules.local.conv")
    monkeypatch.setitem(sys.modules, poisoned.__name__, poisoned)
    roots = []
    for index in range(4):
        root = tmp_path / str(index)
        roots.append(root)
        for component in ("video_vae", "audio_vae"):
            path = root / component
            path.mkdir(parents=True)
            (path / "config.json").write_text(json.dumps({
                "source_safetensors_path": "unused", "auto_map": {"AutoModel": "entry.VAE"}}))
            (path / "conv.py").write_text(f"class SpatialParallelConv3d:\n    value = {index}\n")
            (path / "entry.py").write_text(
                "from .conv import SpatialParallelConv3d\n"
                "import torch\n"
                "class VAE(torch.nn.Linear):\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, path, **kwargs):\n"
                "        assert isinstance(path, str)\n"
                "        model = cls(1, 1)\n"
                "        model.marker = SpatialParallelConv3d.value\n"
                "        return model\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda root: load_h3_vaes(root, "cpu"), roots))
    assert [v.video_vae.model.marker for v in results] == list(range(4))
    assert [v.audio_vae.model.marker for v in results] == list(range(4))
    assert len({type(v.video_vae.model).__module__ for v in results}) == 4
    assert all(not p.requires_grad for v in results for p in v.video_vae.parameters())
    assert sys.modules[poisoned.__name__] is poisoned
