"""Dataset rows for the Audit screen (design doc §6.3).

metadata.csv is the source of truth for transcripts; audit.csv (written by
transcribe) carries duration / chars-per-second / lang_prob per clip. The
two are joined by clip stem here rather than merged at write time, so an
edited transcript never shows stale text and a re-transcribe never loses
scores. A clip missing from audit.csv gets its duration probed straight
from the WAV (stdlib wave, fast) and cps derived; lang_prob stays null.
"""
from __future__ import annotations

import csv
import wave

from .. import clean, metadata
from ..config import Project
from ..lock import project_lock


class ClipNotFound(LookupError):
    pass


class BadText(ValueError):
    pass


def _audit_scores(project: Project) -> dict[str, dict]:
    """audit.csv keyed by clip stem: duration / cps / lang_prob."""
    out: dict[str, dict] = {}
    if not project.audit.exists():
        return out
    with project.audit.open(newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            stem = (row.get("file") or "").removesuffix(".wav")
            if not stem:
                continue
            def _f(key):
                try:
                    return float(row[key])
                except (TypeError, ValueError, KeyError):
                    return None
            out[stem] = {"duration": _f("duration"),
                         "cps": _f("chars_per_sec"),
                         "lang_prob": _f("lang_prob")}
    return out


def _probe_duration(path) -> float | None:
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001 — unreadable WAV shows as null, validate explains
        return None


def rows(project: Project) -> list[dict]:
    """One row per metadata entry, joined with scores and quarantine state."""
    out: list[dict] = []
    if not project.metadata.exists():
        return out
    scores = _audit_scores(project)
    quarantined = {e["clip_id"] for e in clean._manifest_rows(project)
                   if e.get("action") == "quarantine"}
    text_by_id, problems = metadata.read(project.metadata)
    del problems  # the table renders rows; malformed lines stay validate/clean's job
    for cid, text in text_by_id:
        wav = project.wavs / f"{cid}.wav"
        s = scores.get(cid, {})
        duration = s.get("duration")
        if duration is None and wav.exists():
            duration = _probe_duration(wav)
            if duration is not None:
                s = {**s, "duration": duration}
        cps = s.get("cps")
        if cps is None and duration and text:
            cps = round(len(text) / duration, 1)
        out.append({"id": cid, "text": text, "duration": duration,
                    "cps": cps, "lang_prob": s.get("lang_prob"),
                    "missing": not wav.exists(),
                    "quarantined": cid in quarantined})
    return out


def quarantine(project: Project) -> list[dict]:
    """Quarantine manifest entries, most recent first."""
    return list(reversed(clean._manifest_rows(project)))


def set_text(project: Project, clip_id: str, text: str) -> dict:
    """Edit one transcript in place. Line count is unchanged, so any
    malformed lines keep their recorded positions on write-back."""
    if not text.strip():
        raise BadText("transcript must not be empty")
    if "\n" in text or "\r" in text:
        raise BadText("transcript must be a single line")
    text = text.strip()

    with project_lock(project, "api:edit-text", wait=2.0):
        if not project.metadata.exists():
            raise ClipNotFound(f"no dataset/metadata.csv in {project.name}")
        rows_, problems = metadata.read(project.metadata)
        hit = [i for i, (cid, _t) in enumerate(rows_) if cid == clip_id]
        if not hit:
            raise ClipNotFound(f"no clip {clip_id!r} in metadata.csv")
        rows_[hit[0]] = (clip_id, text)
        raw = {p.line_no: p.raw for p in problems}
        metadata.write(project.metadata, rows_, raw_lines=raw)
    return {"id": clip_id, "text": text}
