"""Export a checkpoint to .onnx and write a COMPLETE .onnx.json.

piper1-gpl's generated config is structurally correct but omits three fields
wyoming-piper needs: dataset, audio.quality, and the language block. The
legacy trainer is worse — it derives dataset/quality from the cache directory
path and writes the espeak voice into language.code.

The .onnx filename stem, the JSON's `dataset` field, and the name the client
requests must all be the same string. A mismatch produces two different
failures: the voice silently never appears, or it appears and then throws
VoiceNotFoundError on use.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .config import Project, TIERS, language_block


def export(project: Project, tier: str, checkpoint: Path,
           voice_name: str | None = None,
           espeak_voice: str = "en-us",
           length_scale: float | None = None,
           noise_scale: float | None = None,
           noise_w: float | None = None) -> tuple[Path, Path]:
    stem = voice_name or f"{project.name}-{tier}"
    project.out.mkdir(parents=True, exist_ok=True)
    onnx_path = project.out / f"{stem}.onnx"
    json_path = project.out / f"{stem}.onnx.json"

    cmd = [sys.executable, "-m", "piper.train.export_onnx",
           "--checkpoint", str(checkpoint),
           "--output-file", str(onnx_path)]
    subprocess.run(cmd, check=True)

    generated = project.out / f"{project.name}-{tier}.config.json"
    if not generated.exists():
        raise FileNotFoundError(
            f"training config not found at {generated}; pass --config explicitly")

    cfg = json.loads(generated.read_text())

    # --- the three fields piper1-gpl omits -----------------------------------
    cfg["dataset"] = stem                      # MUST equal the .onnx stem
    cfg.setdefault("audio", {})["quality"] = tier
    cfg["audio"].setdefault("sample_rate", TIERS[tier]["sample_rate"])
    cfg["language"] = language_block(espeak_voice)
    cfg.setdefault("espeak", {})["voice"] = espeak_voice

    inf = cfg.setdefault("inference", {})
    inf.setdefault("noise_scale", 0.667)
    inf.setdefault("length_scale", 1.0)
    inf.setdefault("noise_w", 0.8)
    # a fresh export always writes defaults; re-apply hand tuning here
    if length_scale is not None:
        inf["length_scale"] = length_scale
    if noise_scale is not None:
        inf["noise_scale"] = noise_scale
    if noise_w is not None:
        inf["noise_w"] = noise_w

    # phoneme_id_map / num_symbols / num_speakers describe the trained
    # embedding table and are never touched.
    json_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    return onnx_path, json_path


def verify(onnx_path: Path, json_path: Path) -> list[str]:
    problems = []
    if onnx_path.stem != json_path.name[: -len(".onnx.json")]:
        problems.append("stem mismatch between .onnx and .onnx.json")
    cfg = json.loads(json_path.read_text())
    if cfg.get("dataset") != onnx_path.stem:
        problems.append(
            f"dataset field {cfg.get('dataset')!r} != onnx stem {onnx_path.stem!r}")
    for key in ("num_symbols", "num_speakers", "phoneme_id_map"):
        if key not in cfg:
            problems.append(f"missing {key}")
    if "quality" not in cfg.get("audio", {}):
        problems.append("missing audio.quality")
    if "code" not in cfg.get("language", {}):
        problems.append("missing language.code")
    size_mb = onnx_path.stat().st_size / 1e6
    if not 20 < size_mb < 200:
        problems.append(f"unexpected model size {size_mb:.0f} MB")
    return problems
