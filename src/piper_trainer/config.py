"""Project layout, tier definitions, and language metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Quality tiers. piper1-gpl's VitsModel defaults ARE the medium tier, so medium
# needs no --model.* overrides at all. Values for other tiers should ultimately
# be read from the base checkpoint's own hyper_parameters; these are the
# published architectures and serve as the default.
TIERS: dict[str, dict] = {
    "low": {
        "sample_rate": 16000,
        "model_args": {
            "resblock": "2",
            "resblock_kernel_sizes": "[3,5,7]",
            "resblock_dilation_sizes": "[[1,2],[2,6],[3,12]]",
            "upsample_rates": "[8,8,4]",
            "upsample_initial_channel": "128",
            "upsample_kernel_sizes": "[16,16,8]",
            "hidden_channels": "96",
            "inter_channels": "96",
            "filter_channels": "384",
            "n_layers": "6",
        },
    },
    "medium": {
        "sample_rate": 22050,
        "model_args": {},  # defaults == medium
    },
    "high": {
        "sample_rate": 22050,
        "model_args": {
            "resblock": "1",
            "resblock_kernel_sizes": "[3,7,11]",
            "resblock_dilation_sizes": "[[1,3,5],[1,3,5],[1,3,5]]",
            "upsample_rates": "[8,8,2,2]",
            "upsample_initial_channel": "512",
            "upsample_kernel_sizes": "[16,16,4,4]",
        },
    },
}

# espeak voice -> the language block wyoming-piper expects in the .onnx.json.
LANGUAGES: dict[str, dict] = {
    "en-us": {"code": "en_US", "family": "en", "region": "US",
              "name_native": "English", "name_english": "English",
              "country_english": "United States"},
    "en-gb": {"code": "en_GB", "family": "en", "region": "GB",
              "name_native": "English", "name_english": "English",
              "country_english": "Great Britain"},
    "en-gb-x-rp": {"code": "en_GB", "family": "en", "region": "GB",
                   "name_native": "English", "name_english": "English",
                   "country_english": "Great Britain"},
    "de": {"code": "de_DE", "family": "de", "region": "DE",
           "name_native": "Deutsch", "name_english": "German",
           "country_english": "Germany"},
    "fr-fr": {"code": "fr_FR", "family": "fr", "region": "FR",
              "name_native": "Français", "name_english": "French",
              "country_english": "France"},
    "es": {"code": "es_ES", "family": "es", "region": "ES",
           "name_native": "Español", "name_english": "Spanish",
           "country_english": "Spain"},
}


def language_block(espeak_voice: str) -> dict:
    """Best-effort language metadata for an espeak voice name."""
    if espeak_voice in LANGUAGES:
        return LANGUAGES[espeak_voice]
    # fall back to the base voice (en-gb-x-gbclan -> en-gb -> en)
    parts = espeak_voice.split("-")
    for n in range(len(parts) - 1, 0, -1):
        candidate = "-".join(parts[:n])
        if candidate in LANGUAGES:
            return LANGUAGES[candidate]
    fam = parts[0]
    return {"code": fam, "family": fam, "region": "",
            "name_native": fam, "name_english": fam, "country_english": ""}


@dataclass
class Project:
    """On-disk layout for one voice. Deliberately human-readable and
    toolchain-compatible: everything here works with the bare piper1-gpl CLI
    if this project is ever abandoned."""

    root: Path
    name: str

    @property
    def raw(self) -> Path: return self.root / "raw"
    @property
    def work48k(self) -> Path: return self.root / "work" / "48k"
    @property
    def denoised(self) -> Path: return self.root / "work" / "denoised"
    @property
    def clips(self) -> Path: return self.root / "work" / "clips"
    @property
    def dataset(self) -> Path: return self.root / "dataset"
    @property
    def wavs(self) -> Path: return self.dataset / "wavs"
    @property
    def metadata(self) -> Path: return self.dataset / "metadata.csv"
    @property
    def audit(self) -> Path: return self.dataset / "audit.csv"
    @property
    def checkpoints(self) -> Path: return self.root / "base_checkpoints"
    @property
    def out(self) -> Path: return self.root / "out"

    def cache(self, tier: str) -> Path: return self.root / f"cache-{tier}"
    def runs(self, tier: str) -> Path: return self.root / f"runs-{tier}"

    def ensure(self) -> None:
        for p in (self.raw, self.work48k, self.denoised, self.clips,
                  self.wavs, self.checkpoints, self.out):
            p.mkdir(parents=True, exist_ok=True)

    # ---- project.json --------------------------------------------------
    # Settings that must not be re-specified (and therefore mis-specified) on
    # every invocation. espeak_voice in particular: defaulting it per-command
    # is how a British voice gets trained with American phonemization — the
    # error that trains happily and only reveals itself when you listen.

    @property
    def meta_path(self) -> Path:
        return self.root / "project.json"

    def meta(self) -> dict:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def get(self, key: str, default=None):
        return self.meta().get(key, default)

    def set(self, **kwargs) -> dict:
        data = self.meta()
        data.setdefault("name", self.name)
        data.update({k: v for k, v in kwargs.items() if v is not None})
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(data, indent=2))
        return data

    @classmethod
    def load(cls, root: Path) -> "Project":
        root = Path(root).resolve()
        meta = root / "project.json"
        name = json.loads(meta.read_text())["name"] if meta.exists() else root.name
        return cls(root=root, name=name)

    def save(self, **extra) -> None:
        self.set(**extra)
