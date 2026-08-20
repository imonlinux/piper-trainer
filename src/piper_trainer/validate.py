"""Pre-flight checks. Every rule here exists because it cost someone an hour."""
from __future__ import annotations

import re
import statistics
import subprocess
import wave
from pathlib import Path

from . import metadata
from .config import Project, TIERS

# What espeak-ng will mangle if left in a transcript.
#
# Three parts, kept separate because a naive r"\b[A-Z][a-z]?\." flags every
# sentence ending in a one- or two-letter capitalised word — "...and then I."
# is a normal sentence, not the abbreviation "I.". That false positive fired
# on 42 of 692 real transcripts.
#
#   1. digits and symbols, anywhere
#   2. known abbreviations, named explicitly (mirrors clean.ABBREV)
#   3. unknown short-capital abbreviations, but only mid-sentence, i.e.
#      followed by a lowercase word
KNOWN_ABBREV = ("Mr", "Mrs", "Ms", "Dr", "St", "Prof", "Sr", "Jr",
                "vs", "etc", "Inc", "Ltd", "Co", "Ave", "Rd", "No")
BAD_TEXT = re.compile(
    r"[0-9$%&#@]"
    r"|\b(?:" + "|".join(KNOWN_ABBREV) + r")\."
    r"|\b[ei]\.[g]\.|\bi\.e\."
    r"|\b[A-Z][a-z]?\.(?=\s+[a-z])"
)


# How `clean` may act on each finding code.
#   "repair"    -> fix the text/file in place; deleting would throw away good
#                  audio over a text problem
#   "drop-row"  -> remove the metadata row (and quarantine the WAV if present)
#   "quarantine"-> move clip + row out for review; judgment calls, never silent
#   None        -> informational, or needs a human decision
ACTIONS: dict[str, str] = {
    "crlf": "repair",
    "columns": "repair",
    "blank-row": "repair",
    "unspoken-text": "repair",
    "missing-wav": "drop-row",
    "orphan-wav": "quarantine",   # no row to drop; move the file out
    "unreadable": "quarantine",
    "short-clips": "quarantine",
    "long-clips": "quarantine",
    "cps-outliers": "quarantine",
    "sample-rate": "quarantine",
    "channels": "quarantine",
}


class Finding:
    """A single validation result.

    `ids` names the affected clip stems where the check knows them, which is
    what makes `clean` possible without re-deriving the analysis.
    """

    def __init__(self, level: str, code: str, message: str,
                 ids: list[str] | None = None):
        self.level, self.code, self.message = level, code, message
        self.ids = ids or []

    @property
    def action(self) -> str | None:
        return ACTIONS.get(self.code)

    def __str__(self) -> str:
        icon = {"error": "✗", "warn": "!", "info": "·"}[self.level]
        act = f" ({self.action})" if self.action else ""
        return f"{icon} [{self.code}]{act} {self.message}"


def validate_dataset(project: Project, tier: str = "medium",
                     batch_size: int | None = None,
                     espeak_voice: str | None = None,
                     validation_split: float = 0.02) -> list[Finding]:
    f: list[Finding] = []

    if not project.metadata.exists():
        return [Finding("error", "no-metadata",
                        f"{project.metadata} does not exist")]

    endings = metadata.line_endings(project.metadata)
    if endings == "crlf":
        f.append(Finding("error", "crlf",
                         "metadata.csv has CRLF line endings — strip with "
                         "sed -i 's/\\r$//'"))
    elif endings == "mixed":
        f.append(Finding("error", "crlf",
                         "metadata.csv has mixed line endings (LF and CRLF) "
                         "— normalize to LF (e.g. sed -i 's/\\r$//')"))

    rows, problems = metadata.read(project.metadata)
    for p in (p for p in problems if p.code == "blank-row"):
        f.append(Finding("error", "blank-row", f"line {p.line_no} is blank"))
    bad_cols = [p.line_no for p in problems if p.code == "columns"]
    if bad_cols:
        f.append(Finding("error", "columns",
                         f"{len(bad_cols)} row(s) lack a transcript "
                         f"(lines {bad_cols[:5]}...). Wrong delimiter? "
                         f"piper wants '|', not ','"))
    if not rows:
        return f + [Finding("error", "empty", "no usable rows in metadata.csv")]

    # --- pairing -------------------------------------------------------------
    missing = [i for i, _ in rows if not (project.wavs / f"{i}.wav").exists()]
    if missing:
        f.append(Finding("error", "missing-wav",
                         f"{len(missing)} row(s) reference absent WAVs "
                         f"(e.g. {missing[:3]})", ids=missing))
    ids = {i for i, _ in rows}
    orphans = [w.stem for w in project.wavs.glob("*.wav") if w.stem not in ids]
    if orphans:
        f.append(Finding("warn", "orphan-wav",
                         f"{len(orphans)} WAV(s) have no metadata row "
                         f"(e.g. {orphans[:3]})", ids=orphans))

    # --- audio properties ----------------------------------------------------
    want_rate = TIERS[tier]["sample_rate"]
    rates, chans, durs = set(), set(), []
    bad_rate, bad_chan, per_id_dur = [], [], {}
    for wav in project.wavs.glob("*.wav"):
        try:
            with wave.open(str(wav)) as w:
                rate, nch = w.getframerate(), w.getnchannels()
                dur = w.getnframes() / rate
            rates.add(rate)
            chans.add(nch)
            durs.append(dur)
            per_id_dur[wav.stem] = dur
            if rate != want_rate:
                bad_rate.append(wav.stem)
            if nch != 1:
                bad_chan.append(wav.stem)
        except Exception as exc:  # noqa: BLE001
            f.append(Finding("error", "unreadable", f"{wav.name}: {exc}",
                             ids=[wav.stem]))
    if bad_rate:
        f.append(Finding("error", "sample-rate",
                         f"tier '{tier}' needs {want_rate} Hz; found "
                         f"{sorted(rates)}", ids=bad_rate))
    if bad_chan:
        f.append(Finding("error", "channels", f"non-mono audio: {sorted(chans)}",
                         ids=bad_chan))

    if durs:
        total = sum(durs)
        f.append(Finding("info", "duration",
                         f"{len(durs)} clips, {total/60:.1f} min total, "
                         f"median {statistics.median(durs):.1f}s"))
        if total < 300:
            f.append(Finding("warn", "tiny-dataset",
                             f"only {total/60:.1f} min of audio — expect heavy "
                             f"overfitting. 30-60 min is the practical target"))
        elif total < 1800:
            f.append(Finding("warn", "small-dataset",
                             f"{total/60:.1f} min is below the 30 min comfort "
                             f"threshold; usable but not robust"))
        short = [i for i, d in per_id_dur.items() if d < 1.0]
        long_ = [i for i, d in per_id_dur.items() if d > 15.0]
        if short:
            f.append(Finding("warn", "short-clips",
                             f"{len(short)} clip(s) under 1s destabilize "
                             f"alignment", ids=short))
        if long_:
            f.append(Finding("warn", "long-clips",
                             f"{len(long_)} clip(s) over 15s waste batch memory",
                             ids=long_))

    # --- transcripts ---------------------------------------------------------
    flagged = [(i, t) for i, t in rows if BAD_TEXT.search(t)]
    if flagged:
        f.append(Finding("error", "unspoken-text",
                         f"{len(flagged)} transcript(s) contain digits, symbols "
                         f"or abbreviations — espeak phonemizes these literally. "
                         f"e.g. {flagged[0][0]}: {flagged[0][1][:60]!r}",
                         ids=[i for i, _ in flagged]))

    # chars-per-second outliers: truncated audio, hallucinations, mispairs.
    # per_id_dur was collected in the audio-properties pass above; reuse it.
    cps = [(i, len(t) / d) for i, t in rows
           if (d := per_id_dur.get(i))]
    if len(cps) > 4:
        vals = [c for _, c in cps]
        mu, sd = statistics.mean(vals), statistics.pstdev(vals) or 1.0
        outliers = [i for i, c in cps if abs(c - mu) > 2 * sd]
        if outliers:
            f.append(Finding("warn", "cps-outliers",
                             f"{len(outliers)} clip(s) have unusual "
                             f"chars/sec (e.g. {outliers[:3]}) — check for "
                             f"truncated audio or hallucinated text",
                             ids=outliers))

    # --- batch size ----------------------------------------------------------
    if batch_size:
        train_split = int(len(rows) * (1 - validation_split))
        if batch_size >= train_split:
            f.append(Finding("error", "batch-size",
                             f"batch_size {batch_size} >= train split "
                             f"{train_split}: you get one batch per epoch. "
                             f"Use {max(2, train_split // 3)} or lower"))

    # --- validation split ----------------------------------------------------
    if 0 < validation_split and rows and round(len(rows) * validation_split) < 1:
        f.append(Finding("warn", "validation-split",
                         f"validation_split {validation_split} yields 0 "
                         f"validation clips for {len(rows)} rows — Lightning "
                         f"will skip validation entirely. Use "
                         f"--validation-split 0 or a larger split"))

    # --- espeak voice --------------------------------------------------------
    if espeak_voice:
        try:
            voices = subprocess.run(["espeak-ng", "--voices"],
                                    capture_output=True, text=True,
                                    check=True).stdout
        except FileNotFoundError:
            f.append(Finding("error", "espeak-missing",
                             "espeak-ng binary not found on PATH — "
                             "phonemization (and training) will fail"))
        except subprocess.CalledProcessError as exc:
            f.append(Finding("error", "espeak-missing",
                             f"espeak-ng --voices failed: {exc}"))
        else:
            names = {ln.split()[1] for ln in voices.splitlines()[1:]
                     if len(ln.split()) > 1}
            if espeak_voice not in names:
                f.append(Finding("error", "espeak-voice",
                                 f"'{espeak_voice}' is not an espeak voice. "
                                 f"Run: espeak-ng --voices"))

    return f


def validate_checkpoint(ckpt: Path, tier: str) -> list[Finding]:
    """Confirm a base checkpoint matches the requested tier before a long run."""
    import torch

    f: list[Finding] = []
    if not ckpt.exists():
        return [Finding("error", "no-checkpoint", f"{ckpt} does not exist")]
    try:
        data = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return [Finding("error", "checkpoint-unreadable", str(exc))]

    hp = data.get("hyper_parameters", {}) or {}
    want = TIERS[tier]["sample_rate"]
    got = hp.get("sample_rate")
    if got and got != want:
        f.append(Finding("error", "checkpoint-rate",
                         f"checkpoint is {got} Hz, tier '{tier}' needs {want} Hz"))
    for k, v in data.get("state_dict", {}).items():
        if "emb" in k.lower() and hasattr(v, "shape") and len(v.shape) == 2:
            f.append(Finding("info", "vocab",
                             f"{k}: vocabulary size {v.shape[0]} "
                             f"(piper1-gpl default is 256; 130 = 2023-era "
                             f"checkpoint — use warmstart_ckpt, not ckpt_path)"))
            break
    epoch = data.get("epoch")
    if epoch is not None:
        f.append(Finding("info", "epoch",
                         f"checkpoint is at epoch {epoch} — with --ckpt_path "
                         f"max_epochs must exceed this"))
    return f
