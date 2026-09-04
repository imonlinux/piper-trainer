"""raw audio -> 48 kHz mono -> denoise -> VAD segment -> training-rate WAVs.

Order is load-bearing: DeepFilterNet is 48 kHz only, so denoising must happen
before the final resample, and the low-tier 16 kHz set is built from the
denoised 48 kHz files rather than downsampled from 22.05 kHz.

Idempotency: each stage owns and clears its output directory, and records a
`.stage.json` manifest (params + input fingerprint). Re-running with unchanged
inputs and parameters skips the stage; `--force` overrides. A stage's output
is a pure function of its input and parameters — stale files from previous
parameter sets can never leak into `dataset/wavs/`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import Project, TIERS
from .lock import project_lock

# The one list of processable audio containers: what the pipeline (and
# sources()/peaks.is_audio) recognise. app.py serves playback from
# PLAYABLE_EXT so /files/ stays a playback whitelist, not a second truth.
AUDIO_EXT = {".wav", ".mp4", ".m4a", ".mp3", ".flac", ".ogg", ".aac",
             ".webm", ".mkv", ".opus"}

# Browsers have no <audio> support for Matroska; .mkv is still processed
# end to end, it is just not served back through /files/.
PLAYABLE_EXT = AUDIO_EXT - {".mkv"}

# channel picker -> ffmpeg filter. Shared with peaks.py so the tuner's
# waveform is decoded through the same channel choice the VAD will see.
PAN = {"left": "pan=mono|c0=c0", "right": "pan=mono|c0=c1"}

SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# auditok's energy_threshold lives on 20*log10(RMS) of int16-scaled samples,
# where full scale is 32768: a threshold of 55 rejects every analysis window
# quieter than 55 - this constant (about -35.3) dBFS RMS. Clear-but-quiet
# recordings fail that bar on playback perfectly well, so the level helpers
# below report dBFS figures the UI can put next to the threshold.
INT16_FULL_SCALE_DB = 20.0 * math.log10(32768)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT)


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries",
         "stream=codec_name,sample_rate,channels,bit_rate:format=duration",
         "-of", "default=nw=1:nk=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return dict(
        line.split("=", 1) for line in out.strip().splitlines() if "=" in line
    )


def sources(project: Project) -> list[dict]:
    """Inventory of raw/: name, codec, rate, channels, duration, size."""
    out = []
    for src in sorted(project.raw.iterdir()):
        if not src.is_file() or src.suffix.lower() not in AUDIO_EXT:
            continue
        info = probe(src)
        out.append({"name": src.name,
                    "codec": info.get("codec_name"),
                    "sample_rate": info.get("sample_rate"),
                    "channels": info.get("channels"),
                    "duration": info.get("duration"),
                    "size": src.stat().st_size})
    return out


def delete_sources(project: Project, names: list[str]) -> dict:
    """Move raw/ sources to the workspace .trash (§0 — nothing is destroyed).

    Returns {"moved": [...], "missing": [...]}. A name is a basename:
    sources live in raw/, never at a path, and only files sources()
    would list are eligible. The move sits under the project lock so a
    concurrent prepare/preview cannot watch a file vanish mid-read.
    """
    moved: list[str] = []
    missing: list[str] = []
    targets: list[tuple[str, Path]] = []
    for name in dict.fromkeys(names):  # dedupe, keep order
        safe = Path(name).name
        src = project.raw / safe
        if (not safe or safe in (".", "..")
                or src.suffix.lower() not in AUDIO_EXT
                or not src.is_file()):
            missing.append(name)
            continue
        targets.append((name, src))
    if targets:
        trash = project.root.parent / ".trash"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = trash / f"{project.root.name}-sources-{stamp}"
        dest.mkdir(parents=True, exist_ok=True)
        with project_lock(project, "api:delete-sources", wait=2.0):
            for name, src in targets:
                shutil.move(str(src), str(dest / src.name))
                moved.append(name)
    return {"moved": moved, "missing": missing}


# --------------------------------------------------------------- stage plumbing

def _clear_dir(d: Path) -> None:
    if d.exists():
        for p in d.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    d.mkdir(parents=True, exist_ok=True)


def _fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        st = p.stat()
        h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _stage_matches(out_dir: Path, stage: str, params: dict,
                   inputs: list[Path]) -> bool:
    mf = out_dir / ".stage.json"
    if not mf.exists():
        return False
    try:
        data = json.loads(mf.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (data.get("stage") == stage
            and data.get("params") == params
            and data.get("input_fingerprint") == _fingerprint(inputs))


def _write_stage_manifest(out_dir: Path, stage: str, params: dict,
                          inputs: list[Path], outputs: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".stage.json").write_text(json.dumps(
        {"stage": stage, "params": params,
         "input_fingerprint": _fingerprint(inputs),
         "outputs": outputs,
         "completed_at": datetime.now(timezone.utc).strftime(
             "%Y-%m-%dT%H:%M:%SZ")},
        indent=2) + "\n")


# -------------------------------------------------------------------- naming

def sanitize_stem(stem: str) -> str:
    """Normalize to [A-Za-z0-9._-]; runs of anything else collapse to '_'."""
    return SANITIZE_RE.sub("_", stem) or "_"


def assign_names(srcs: list[Path]) -> dict[Path, str]:
    """Map each source file to a destination stem.

    Sanitize first, then resolve collisions: for any stem claimed by more
    than one source, every member of the group gets `{stem}_{ext}` —
    deterministic given the input set, never order-dependent. A rename that
    still collides (possible when two distinct names sanitize identically
    and share an extension) is separated by index within the sorted group.
    """
    assign: dict[Path, str] = {}
    groups: dict[str, list[Path]] = defaultdict(list)
    for src in srcs:
        groups[sanitize_stem(src.stem)].append(src)
    for stem, members in groups.items():
        if len(members) == 1:
            assign[members[0]] = stem
        else:
            for m in members:
                assign[m] = f"{stem}_{m.suffix.lower().lstrip('.')}"
    for _ in range(8):  # fixpoint; sane sets converge immediately
        by_name: dict[str, list[Path]] = defaultdict(list)
        for src in sorted(assign):
            by_name[assign[src]].append(src)
        dups = {n: ms for n, ms in by_name.items() if len(ms) > 1}
        if not dups:
            return assign
        for name, members in dups.items():
            for i, m in enumerate(sorted(members)):
                if i > 0:  # sorted-first keeps the name; rest get an index
                    assign[m] = f"{name}_{i + 1}"
    raise RuntimeError(f"could not resolve destination names for {srcs!r}")


# ------------------------------------------------------------------ primitives
# One-file operations shared by the full stages above and the preview
# variants (design doc §0: the API layer never re-implements the pipeline).

def convert_one(src: Path, dst: Path, channel: str | None = None) -> None:
    """One source -> 48 kHz mono 16-bit WAV, channel choice respected."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn"]
    if channel in PAN:
        cmd += ["-af", PAN[channel]]
    else:
        cmd += ["-ac", "1"]
    cmd += ["-ar", "48000", "-c:a", "pcm_s16le", str(dst)]
    _run(cmd)


def excerpt(src: Path, dst: Path, seconds: float) -> None:
    """First `seconds` of a WAV, re-encoded 16-bit PCM."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(src), "-t", str(seconds),
          "-c:a", "pcm_s16le", str(dst)])


def denoise_file(src: Path, dst_dir: Path) -> Path:
    """DeepFilterNet3 on one file — the same -D flag (STFT lookahead
    compensation) as the batch stage; without it audio drifts out of
    alignment with its transcript. dst_dir must differ from src's parent."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    _run(["deep-filter", "-D", "-o", str(dst_dir), str(src)])
    return dst_dir / src.name


def split_audio(
    src: Path,
    out_dir: Path,
    energy_threshold: float = 55,
    min_dur: float = 1.5,
    max_dur: float = 10.0,
    max_silence: float = 0.4,
    max_leading_silence: float = 0.15,
    max_trailing_silence: float = 0.15,
    stem: str | None = None,
) -> list[dict]:
    """VAD-split one WAV into out_dir; returns boundary metadata per clip.

    Same auditok call (and the same older-API fallback) as the segment
    stage; `stem` overrides the clip-name stem so previews can name clips
    after the raw source instead of an internal work file.
    """
    import auditok

    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(min_dur=min_dur, max_dur=max_dur, max_silence=max_silence,
                  energy_threshold=energy_threshold)
    region = auditok.load(str(src))
    try:
        events = region.split(
            max_leading_silence=max_leading_silence,
            max_trailing_silence=max_trailing_silence, **kwargs)
    except TypeError:
        # older auditok without the leading/trailing silence parameters
        events = region.split(**kwargs)
    stem = stem or src.stem
    clips = []
    for i, ev in enumerate(events, start=1):
        # auditok moved start/end from ev.meta to the region itself;
        # support both APIs
        start = getattr(ev, "start", None)
        if start is None:
            start, end = ev.meta.start, ev.meta.end
        else:
            end = ev.end
        name = f"{stem}_{i:04d}_{start:.3f}-{end:.3f}.wav"
        ev.save(str(out_dir / name))
        clips.append({"clip": name, "start": round(start, 3),
                      "end": round(end, 3)})
    return clips


def wav_level_dbfs(path: Path, window_seconds: float = 0.2) -> dict:
    """Level profile of a 16-bit mono WAV: peak, overall RMS and speech RMS.

    speech_dbfs is the 95th-percentile RMS across 0.2 s windows, so silence
    gaps do not dilute it; it is the honest comparator for energy_threshold
    (see INT16_FULL_SCALE_DB). Read in chunks so an hour-long source never
    sits in memory whole. numpy rides in with auditok, so no new dependency.
    """
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError("expected a 16-bit mono WAV")
        rate = w.getframerate()
        win = max(1, int(rate * window_seconds))
        peak = 0
        sq_total = 0.0
        n_total = 0
        window_rms: list[float] = []
        while True:
            frames = w.readframes(win)
            if not frames:
                break
            x = np.frombuffer(frames, dtype="<i2").astype(np.float64)
            if x.size == 0:
                continue
            peak = max(peak, int(np.max(np.abs(x))))
            sq_total += float(np.sum(x * x))
            n_total += x.size
            window_rms.append(float(np.sqrt(np.mean(x * x))))

    def db(v: float) -> float:
        return round(20.0 * math.log10(max(v, 1e-10) / 32768.0), 1)

    rms = math.sqrt(sq_total / n_total) if n_total else 0.0
    speech = sorted(window_rms)[int(0.95 * (len(window_rms) - 1))] \
        if window_rms else 0.0
    return {"peak_dbfs": db(float(peak)), "rms_dbfs": db(rms),
            "speech_dbfs": db(speech)}


# --------------------------------------------------------------------- stages

def to_48k(project: Project, channel: str | None = None,
           force: bool = False) -> tuple[int | str, dict[str, str]]:
    """Convert every source file to 48 kHz mono 16-bit WAV.

    channel: None -> downmix (default); 'left'/'right' -> pick one channel.
    Use a single channel when the source has a good mic on one side only;
    a blind downmix averages in the bad channel and can halve SNR.
    """
    srcs = sorted(p for p in project.raw.iterdir()
                  if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    params = {"channel": channel or "downmix"}
    if not force and _stage_matches(project.work48k, "to_48k", params, srcs):
        return "skipped", {}
    _clear_dir(project.work48k)
    names = assign_names(srcs)
    renamed = {}
    for src in srcs:
        dst = project.work48k / f"{names[src]}.wav"
        convert_one(src, dst, channel=channel)
        if dst.stem != src.stem:  # sanitized and/or collision-renamed
            renamed[src.name] = dst.name
    _write_stage_manifest(project.work48k, "to_48k", params, srcs, len(srcs))
    return len(srcs), renamed


def denoise(project: Project, enabled: bool = True,
            force: bool = False) -> int | str:
    """DeepFilterNet3. -D compensates STFT/model lookahead delay; without it
    audio drifts out of alignment with its transcript.

    No postfilter (--pf): it trades naturalness for suppression, and denoisers
    already do their worst damage on breaths, sibilants and plosives — exactly
    the detail VITS needs to learn.
    """
    srcs = sorted(project.work48k.glob("*.wav"))
    if not srcs:
        return 0
    params = {"enabled": enabled}
    if not force and _stage_matches(project.denoised, "denoise", params, srcs):
        return "skipped"
    _clear_dir(project.denoised)
    if not enabled:
        for s in srcs:
            shutil.copy2(s, project.denoised / s.name)
        n = len(srcs)
    else:
        _run(["deep-filter", "-D", "-o", str(project.denoised),
              *[str(s) for s in srcs]])
        # one output per input; count what this invocation wrote, not what
        # happens to be in the directory
        n = len(srcs)
    _write_stage_manifest(project.denoised, "denoise", params, srcs, n)
    return n


def segment(
    project: Project,
    energy_threshold: float = 55,
    min_dur: float = 1.5,
    max_dur: float = 10.0,
    max_silence: float = 0.4,
    max_leading_silence: float = 0.15,
    max_trailing_silence: float = 0.15,
    force: bool = False,
) -> tuple[int | str, dict[str, int]]:
    """Energy-based VAD split. Not a model — the threshold is a per-recording
    dial, and denoised audio usually wants a LOWER value than the default.

    Leading/trailing silence keeps natural onsets and fade-outs (clipped
    plosives teach the model to swallow consonants) and gives every clip the
    same padding, which matters because VITS learns the padding too.

    Returns (total, per-source clip counts). A source with zero clips used to
    vanish silently from the run; now every source's count is logged and a
    zero-count source is called out by name, because nothing downstream
    (transcribe, train) will ever see that audio again.
    """
    srcs = sorted(project.denoised.glob("*.wav"))
    if not srcs:
        return 0, {}
    params = dict(energy_threshold=energy_threshold, min_dur=min_dur,
                  max_dur=max_dur, max_silence=max_silence,
                  max_leading_silence=max_leading_silence,
                  max_trailing_silence=max_trailing_silence)
    if not force and _stage_matches(project.clips, "segment", params, srcs):
        return "skipped", {}
    _clear_dir(project.clips)
    total = 0
    per_source: dict[str, int] = {}
    for src in srcs:
        clips = split_audio(src, project.clips,
                            energy_threshold=energy_threshold,
                            min_dur=min_dur, max_dur=max_dur,
                            max_silence=max_silence,
                            max_leading_silence=max_leading_silence,
                            max_trailing_silence=max_trailing_silence)
        per_source[src.name] = len(clips)
        total += len(clips)
        print(f"segment: {src.name}: {len(clips)} clips", flush=True)
        if not clips:
            print(f"! segment: {src.name}: 0 clips — this source contributes "
                  "NOTHING to the dataset. Try a lower energy threshold "
                  "(Prepare tuner) or check the file's level with the "
                  "denoise A/B preview.", flush=True)
    _write_stage_manifest(project.clips, "segment", params, srcs, total)
    return total, per_source


def finalize(project: Project, tier: str = "medium",
             target: Path | None = None, force: bool = False) -> int | str:
    """Loudness-normalize and resample clips to the tier's training rate."""
    rate = TIERS[tier]["sample_rate"]
    dst_dir = target or project.wavs
    clips = sorted(project.clips.glob("*.wav"))
    if not clips:
        return 0
    params = {"tier": tier}
    if not force and _stage_matches(dst_dir, "finalize", params, clips):
        return "skipped"
    _clear_dir(dst_dir)
    count = 0
    for clip in clips:
        _run(["ffmpeg", "-y", "-i", str(clip),
              "-af", "loudnorm=I=-23:TP=-2:LRA=7,aresample=resampler=soxr",
              "-ar", str(rate), "-ac", "1", "-c:a", "pcm_s16le",
              str(dst_dir / clip.name)])
        count += 1
    _write_stage_manifest(dst_dir, "finalize", params, clips, count)
    return count


def run_all(project: Project, tier: str = "medium", channel: str | None = None,
            denoise_enabled: bool = True, force: bool = False,
            on_stage=None, **seg_kwargs) -> dict:
    """Run the full pipeline. `on_stage(index, name)` — 1-based — lets job
    runners surface per-stage progress without reaching into the stages."""
    project.ensure()
    stats: dict = {}
    if on_stage:
        on_stage(1, "convert")
    stats["converted"], renamed = to_48k(project, channel=channel, force=force)
    if renamed:
        stats["renamed"] = renamed
    if on_stage:
        on_stage(2, "denoise")
    stats["denoised"] = denoise(project, enabled=denoise_enabled, force=force)
    if on_stage:
        on_stage(3, "segment")
    seg, per_source = segment(project, force=force, **seg_kwargs)
    stats["clips"] = seg
    zero = sorted(k for k, v in per_source.items() if v == 0)
    if zero:
        # only surfaced when it matters: a silent source drop is the one
        # segment outcome the operator must never miss
        stats["clips_per_source"] = per_source
        stats["clips_zero_sources"] = zero
    if on_stage:
        on_stage(4, "finalize")
    stats["finalized"] = finalize(project, tier=tier, force=force)
    return stats
