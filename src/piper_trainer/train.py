"""Build and run the piper1-gpl training command."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import Project, TIERS


def build_command(
    project: Project,
    tier: str = "medium",
    espeak_voice: str = "en-us",
    batch_size: int = 32,
    max_epochs: int = 4000,
    num_workers: int = 8,
    warmstart: Path | None = None,
    resume: Path | None = None,
    validation_split: float = 0.02,
    precision: str = "32-true",
    accelerator: str = "gpu",
    check_val_every_n_epoch: int = 25,
    voice_name: str | None = None,
) -> list[str]:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    spec = TIERS[tier]
    lang_code = espeak_voice.split("-")[0]
    name = voice_name or f"{lang_code}-{project.name}-{tier}"

    cmd = [
        sys.executable, "-m", "piper.train", "fit",
        "--data.voice_name", name,
        "--data.csv_path", str(project.metadata),
        "--data.audio_dir", str(project.wavs),
        "--data.espeak_voice", espeak_voice,
        "--data.cache_dir", str(project.cache(tier)),
        "--data.config_path", str(project.out / f"{project.name}-{tier}.config.json"),
        "--data.batch_size", str(batch_size),
        "--data.num_workers", str(num_workers),
        # auditok already applied uniform padding; don't let the trainer
        # re-trim it with its own 0.25s default
        "--data.trim_silence", "false",
        "--data.validation_split", str(validation_split),
        "--model.sample_rate", str(spec["sample_rate"]),
        "--trainer.accelerator", accelerator,
        "--trainer.devices", "1",
        # fp32: gfx1151 has known bf16 bugs, and VITS is small enough that
        # mixed precision buys little
        "--trainer.precision", precision,
        "--trainer.max_epochs", str(max_epochs),
        "--trainer.check_val_every_n_epoch", str(check_val_every_n_epoch),
        # without this, lightning_logs/ lands in the launch directory
        "--trainer.default_root_dir", str(project.runs(tier)),
    ]

    for key, value in spec["model_args"].items():
        cmd += [f"--model.{key}", value]

    if resume:
        # Lightning resume: restores optimizer state AND the epoch counter,
        # so max_epochs must exceed the checkpoint's epoch.
        cmd += ["--ckpt_path", str(resume)]
    elif warmstart:
        # weights-only; starts the epoch count at zero. The right choice when
        # fine-tuning from a different voice.
        cmd += ["--model.warmstart_ckpt", str(warmstart)]

    return cmd


def latest_checkpoint(project: Project, tier: str) -> Path | None:
    runs = project.runs(tier)
    if not runs.exists():
        return None
    cands = sorted(runs.glob("lightning_logs/version_*/checkpoints/*.ckpt"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        return None
    last = [c for c in cands if c.name == "last.ckpt"]
    return last[-1] if last else cands[-1]


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(" \\\n  ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)
