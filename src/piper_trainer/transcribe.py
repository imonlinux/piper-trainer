"""Whisper transcription -> metadata.csv + audit.csv.

Resumable: clips that already have a row in metadata.csv are skipped and
their audit rows carried forward, so an interrupted run costs only the clips
it had not reached. --retranscribe forces a full pass.
"""
from __future__ import annotations

import csv
import os
import wave
from collections.abc import Callable
from pathlib import Path

from . import metadata
from .config import Project


def duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def _load_audit(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 5 and row[0]:
                rows[row[0]] = row
    return rows


def transcribe(project: Project, model_size: str | None = None,
               language: str = "en", device: str = "cpu",
               compute_type: str = "int8",
               retranscribe: bool = False,
               on_progress: Callable[[int, int, str], None] | None = None
               ) -> dict:
    """Use a LARGE model: this is a batch job with no latency requirement, and
    small-model errors bury you in false audit flags.

    condition_on_previous_text=False is important — the default feeds prior
    text as context, which is where repetition-loop hallucinations come from,
    and these clips are independent utterances anyway.

    vad_filter=False because segmentation already happened; Whisper's own VAD
    can trim audio it thinks is silence and desync transcript from clip.

    on_progress(done, total, filename) fires after each clip — transcribed
    or skipped — so a long batch can stream a live counter instead of
    going silent for the whole run.
    """
    wavs = sorted(project.wavs.glob("*.wav"))
    existing: dict[str, str] = {}
    existing_problems: list = []
    if not retranscribe and project.metadata.exists():
        existing, existing_problems = metadata.read(project.metadata)
        existing = dict(existing)
    audit_old = _load_audit(project.audit)

    model = None
    if not retranscribe:
        todo = [w for w in wavs if w.stem not in existing]
    else:
        todo = list(wavs)
    if todo:
        from faster_whisper import WhisperModel
        model_size = model_size or os.environ.get(
            "PIPER_TRAINER_WHISPER_MODEL", "large-v3")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

    rows, audit, transcribed, skipped, total_seconds = [], [], 0, 0, 0.0
    for i, wav in enumerate(wavs):
        dur = duration(wav)
        total_seconds += dur
        if not retranscribe and wav.stem in existing:
            text = existing[wav.stem]
            skipped += 1
            if wav.name in audit_old:
                # carry the original audit row forward verbatim
                audit.append(audit_old[wav.name])
                rows.append([wav.stem, text])
                if on_progress:
                    on_progress(i + 1, len(wavs), wav.name)
                continue
            info = None
        else:
            segments, info = model.transcribe(
                str(wav), language=language, beam_size=5,
                vad_filter=False, condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in segments).strip()
            transcribed += 1
        if on_progress:
            on_progress(i + 1, len(wavs), wav.name)
        cps = len(text) / dur if dur else 0.0
        prob = f"{info.language_probability:.3f}" if info is not None else ""
        rows.append([wav.stem, text])
        audit.append([wav.name, f"{dur:.2f}", f"{cps:.1f}", prob, text])

    # Malformed lines from the previous metadata are carried through the
    # rewrite (appended after the new rows, in original order) rather than
    # dropped — validate and clean exist to deal with them afterwards.
    preserve = {len(rows) + 1 + i: p.raw
                for i, p in enumerate(existing_problems)}
    metadata.write(project.metadata, rows, raw_lines=preserve)
    with project.audit.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "duration", "chars_per_sec", "lang_prob", "text"])
        w.writerows(audit)

    return {"clips": len(rows), "transcribed": transcribed, "skipped": skipped,
            "malformed_preserved": len(preserve),
            "total_seconds": total_seconds}
