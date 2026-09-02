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
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import Project, TIERS

AUDIO_EXT = {".wav", ".mp4", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ".webm", ".mkv"}

# channel picker -> ffmpeg filter. Shared with peaks.py so the tuner's
# waveform is decoded through the same channel choice the VAD will see.
PAN = {"left": "pan=mono|c0=c0", "right": "pan=mono|c0=c1"}

SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


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
    pan = PAN.get(channel or "")
    names = assign_names(srcs)
    renamed = {}
    for src in srcs:
        dst = project.work48k / f"{names[src]}.wav"
        cmd = ["ffmpeg", "-y", "-i", str(src), "-vn"]
        if pan:
            cmd += ["-af", pan]
        else:
            cmd += ["-ac", "1"]
        cmd += ["-ar", "48000", "-c:a", "pcm_s16le", str(dst)]
        _run(cmd)
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
) -> int | str:
    """Energy-based VAD split. Not a model — the threshold is a per-recording
    dial, and denoised audio usually wants a LOWER value than the default.

    Leading/trailing silence keeps natural onsets and fade-outs (clipped
    plosives teach the model to swallow consonants) and gives every clip the
    same padding, which matters because VITS learns the padding too.
    """
    import auditok

    srcs = sorted(project.denoised.glob("*.wav"))
    if not srcs:
        return 0
    params = dict(energy_threshold=energy_threshold, min_dur=min_dur,
                  max_dur=max_dur, max_silence=max_silence,
                  max_leading_silence=max_leading_silence,
                  max_trailing_silence=max_trailing_silence)
    if not force and _stage_matches(project.clips, "segment", params, srcs):
        return "skipped"
    _clear_dir(project.clips)
    total = 0
    kwargs = dict(min_dur=min_dur, max_dur=max_dur, max_silence=max_silence,
                  energy_threshold=energy_threshold)
    for src in srcs:
        region = auditok.load(str(src))
        try:
            events = region.split(
                max_leading_silence=max_leading_silence,
                max_trailing_silence=max_trailing_silence, **kwargs)
        except TypeError:
            # older auditok without the leading/trailing silence parameters
            events = region.split(**kwargs)
        for i, ev in enumerate(events, start=1):
            # auditok moved start/end from ev.meta to the region itself;
            # support both APIs
            start = getattr(ev, "start", None)
            if start is None:
                start, end = ev.meta.start, ev.meta.end
            else:
                end = ev.end
            name = f"{src.stem}_{i:04d}_{start:.3f}-{end:.3f}.wav"
            ev.save(str(project.clips / name))
            total += 1
    _write_stage_manifest(project.clips, "segment", params, srcs, total)
    return total


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
            **seg_kwargs) -> dict:
    project.ensure()
    stats: dict = {}
    stats["converted"], renamed = to_48k(project, channel=channel, force=force)
    if renamed:
        stats["renamed"] = renamed
    stats["denoised"] = denoise(project, enabled=denoise_enabled, force=force)
    stats["clips"] = segment(project, force=force, **seg_kwargs)
    stats["finalized"] = finalize(project, tier=tier, force=force)
    return stats
