"""Server-side waveform peaks (design doc §8, resolved decision 1).

The server already has the audio decoded; a peaks array is far smaller over
the wire than a WAV, and the client canvas stays dumb. Peaks are cached next
to the source file, keyed by name + channel + mtime + size, so a re-ingest
invalidates the cache automatically.

Decoding happens at 8 kHz mono: a peaks envelope is display data, and at the
bucket sizes a screen shows (~0.3 s/bucket for a ten-minute file) 8 kHz
resolves far more than enough. The channel filter matches prepare.to_48k so
the waveform the tuner draws is the waveform the VAD actually segments.
"""
from __future__ import annotations

import array
import json
import subprocess
import sys
from pathlib import Path

from .prepare import AUDIO_EXT, PAN, probe

CHANNELS = ("downmix", "left", "right")
DECODE_RATE = 8000
BUCKETS_MIN, BUCKETS_MAX = 200, 8000


def _bucketize(pcm: bytes, buckets: int) -> list[float]:
    """Max absolute s16 sample per bucket, normalized to 0..1.

    max(chunk) / -min(chunk) run at C speed over array slices; taking the
    larger of the two is the per-bucket peak without a Python-level abs loop.
    """
    samples = array.array("h")
    samples.frombytes(pcm[: (len(pcm) // 2) * 2])
    if sys.byteorder == "big":  # s16le from ffmpeg
        samples.byteswap()
    n = len(samples)
    if not n:
        return [0.0] * buckets
    per = max(1, n // buckets)
    out = []
    for i in range(0, n, per):
        chunk = samples[i:i + per]
        out.append(max(max(chunk), -min(chunk)) / 32768.0)
    out += [0.0] * (buckets - len(out))   # shorter than asked: silent tail
    return out[:buckets]


def _cache_path(src: Path, channel: str) -> Path:
    return src.with_name(f"{src.name}.{channel}.peaks.json")


def _decode(src: Path, channel: str) -> bytes:
    cmd = ["ffmpeg", "-v", "error", "-i", str(src), "-vn"]
    if channel in PAN:
        cmd += ["-af", PAN[channel]]
    cmd += ["-ac", "1", "-ar", str(DECODE_RATE), "-f", "s16le", "pipe:1"]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def compute_peaks(src: Path, channel: str = "downmix",
                  buckets: int = 2000) -> dict:
    """Peaks for one raw source, with a fingerprint-checked disk cache."""
    buckets = max(BUCKETS_MIN, min(buckets, BUCKETS_MAX))
    cache = _cache_path(src, channel)
    st = src.stat()
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if (data.get("mtime_ns") == st.st_mtime_ns
                    and data.get("size") == st.st_size
                    and data.get("buckets") == buckets
                    and len(data.get("peaks", [])) == buckets):
                return {k: v for k, v in data.items()
                        if k not in ("mtime_ns", "size")}
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache — recompute below

    duration = float(probe(src)["duration"])
    peaks = _bucketize(_decode(src, channel), buckets)
    out = {"name": src.name, "channel": channel, "buckets": len(peaks),
           "rate": DECODE_RATE, "duration": duration, "peaks": peaks}
    cache.write_text(json.dumps(
        {**out, "mtime_ns": st.st_mtime_ns, "size": st.st_size}))
    return out


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXT
