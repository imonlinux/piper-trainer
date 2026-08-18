"""raw audio -> 48 kHz mono -> denoise -> VAD segment -> training-rate WAVs.

Order is load-bearing: DeepFilterNet is 48 kHz only, so denoising must happen
before the final resample, and the low-tier 16 kHz set is built from the
denoised 48 kHz files rather than downsampled from 22.05 kHz.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import Project, TIERS

AUDIO_EXT = {".wav", ".mp4", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ".webm", ".mkv"}


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT)


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate",
         "-of", "default=nw=1:nk=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return dict(
        line.split("=", 1) for line in out.strip().splitlines() if "=" in line
    )


def to_48k(project: Project, channel: str | None = None) -> int:
    """Convert every source file to 48 kHz mono 16-bit WAV.

    channel: None -> downmix (default); 'left'/'right' -> pick one channel.
    Use a single channel when the source has a good mic on one side only;
    a blind downmix averages in the bad channel and can halve SNR.
    """
    project.work48k.mkdir(parents=True, exist_ok=True)
    pan = {"left": "pan=mono|c0=c0", "right": "pan=mono|c0=c1"}.get(channel or "")
    count = 0
    for src in sorted(project.raw.iterdir()):
        if not src.is_file() or src.suffix.lower() not in AUDIO_EXT:
            continue
        dst = project.work48k / f"{src.stem}.wav"
        cmd = ["ffmpeg", "-y", "-i", str(src), "-vn"]
        if pan:
            cmd += ["-af", pan]
        else:
            cmd += ["-ac", "1"]
        cmd += ["-ar", "48000", "-c:a", "pcm_s16le", str(dst)]
        _run(cmd)
        count += 1
    return count


def denoise(project: Project, enabled: bool = True) -> int:
    """DeepFilterNet3. -D compensates STFT/model lookahead delay; without it
    audio drifts out of alignment with its transcript.

    No postfilter (--pf): it trades naturalness for suppression, and denoisers
    already do their worst damage on breaths, sibilants and plosives — exactly
    the detail VITS needs to learn.
    """
    project.denoised.mkdir(parents=True, exist_ok=True)
    sources = sorted(project.work48k.glob("*.wav"))
    if not sources:
        return 0
    if not enabled:
        for s in sources:
            shutil.copy2(s, project.denoised / s.name)
        return len(sources)
    _run(["deep-filter", "-D", "-o", str(project.denoised),
          *[str(s) for s in sources]])
    return len(list(project.denoised.glob("*.wav")))


def segment(
    project: Project,
    energy_threshold: float = 55,
    min_dur: float = 1.5,
    max_dur: float = 10.0,
    max_silence: float = 0.4,
    max_leading_silence: float = 0.15,
    max_trailing_silence: float = 0.15,
) -> int:
    """Energy-based VAD split. Not a model — the threshold is a per-recording
    dial, and denoised audio usually wants a LOWER value than the default.

    Leading/trailing silence keeps natural onsets and fade-outs (clipped
    plosives teach the model to swallow consonants) and gives every clip the
    same padding, which matters because VITS learns the padding too.
    """
    import auditok

    project.clips.mkdir(parents=True, exist_ok=True)
    total = 0
    kwargs = dict(min_dur=min_dur, max_dur=max_dur, max_silence=max_silence,
                  energy_threshold=energy_threshold)
    for src in sorted(project.denoised.glob("*.wav")):
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
    return total


def finalize(project: Project, tier: str = "medium",
             target: Path | None = None) -> int:
    """Loudness-normalize and resample clips to the tier's training rate."""
    rate = TIERS[tier]["sample_rate"]
    dst_dir = target or project.wavs
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for clip in sorted(project.clips.glob("*.wav")):
        _run(["ffmpeg", "-y", "-i", str(clip),
              "-af", "loudnorm=I=-23:TP=-2:LRA=7,aresample=resampler=soxr",
              "-ar", str(rate), "-ac", "1", "-c:a", "pcm_s16le",
              str(dst_dir / clip.name)])
        count += 1
    return count


def run_all(project: Project, tier: str = "medium", channel: str | None = None,
            denoise_enabled: bool = True, **seg_kwargs) -> dict:
    project.ensure()
    stats = {}
    stats["converted"] = to_48k(project, channel=channel)
    stats["denoised"] = denoise(project, enabled=denoise_enabled)
    stats["clips"] = segment(project, **seg_kwargs)
    stats["finalized"] = finalize(project, tier=tier)
    return stats
