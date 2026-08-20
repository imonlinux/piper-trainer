"""Whisper transcription -> metadata.csv + audit.csv."""
from __future__ import annotations

import csv
import os
import wave
from pathlib import Path

from . import metadata
from .config import Project


def duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def transcribe(project: Project, model_size: str | None = None,
               language: str = "en", device: str = "cpu",
               compute_type: str = "int8") -> dict:
    """Use a LARGE model: this is a batch job with no latency requirement, and
    small-model errors bury you in false audit flags.

    condition_on_previous_text=False is important — the default feeds prior
    text as context, which is where repetition-loop hallucinations come from,
    and these clips are independent utterances anyway.

    vad_filter=False because segmentation already happened; Whisper's own VAD
    can trim audio it thinks is silence and desync transcript from clip.
    """
    from faster_whisper import WhisperModel

    model_size = model_size or os.environ.get(
        "PIPER_TRAINER_WHISPER_MODEL", "large-v3")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    rows, audit = [], []
    for wav in sorted(project.wavs.glob("*.wav")):
        segments, info = model.transcribe(
            str(wav), language=language, beam_size=5,
            vad_filter=False, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        dur = duration(wav)
        cps = len(text) / dur if dur else 0.0
        rows.append([wav.stem, text])
        audit.append([wav.name, f"{dur:.2f}", f"{cps:.1f}",
                      f"{info.language_probability:.3f}", text])

    metadata.write(project.metadata, rows)
    with project.audit.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "duration", "chars_per_sec", "lang_prob", "text"])
        w.writerows(audit)

    return {"clips": len(rows),
            "total_seconds": sum(float(r[1]) for r in audit)}
