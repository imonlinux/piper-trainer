"""Peaks: server-side waveform envelope with a fingerprint-checked cache."""
from __future__ import annotations

import array
import json
import subprocess
import sys

import pytest

from piper_trainer import peaks


def _s16(*samples: int) -> bytes:
    buf = array.array("h", samples)
    if sys.byteorder == "big":
        buf.byteswap()
    return buf.tobytes()


def test_bucketize_takes_per_bucket_peak():
    # buckets of 2 samples: |100| and |-200| dominate their buckets
    out = peaks._bucketize(_s16(0, 100, -200, 50), 2)
    assert out == [100 / 32768, 200 / 32768]


def test_bucketize_pads_short_input():
    assert peaks._bucketize(b"", 4) == [0.0, 0.0, 0.0, 0.0]
    out = peaks._bucketize(_s16(16384), 4)
    assert out == [16384 / 32768, 0.0, 0.0, 0.0]


def test_bucketize_values_are_normalized():
    out = peaks._bucketize(_s16(*([-32768] * 8)), 4)
    assert out == [1.0, 1.0, 1.0, 1.0]


@pytest.fixture
def wav(tmp_path):
    """A real 0.5 s 440 Hz WAV, via the ffmpeg the runtime actually uses.
    The sine source runs at 1/8 amplitude; volume=8 brings it to full scale."""
    path = tmp_path / "take.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=440:d=0.5",
         "-af", "volume=8", "-ar", "8000", "-ac", "1", "-c:a", "pcm_s16le",
         str(path)],
        check=True)
    return path


def test_compute_peaks_shape_and_cache(wav):
    out = peaks.compute_peaks(wav, buckets=500)
    assert out["name"] == "take.wav"
    assert out["channel"] == "downmix"
    assert out["buckets"] == len(out["peaks"]) == 500
    assert 0.4 <= out["duration"] <= 0.6
    assert max(out["peaks"]) > 0.5          # a sine is loud
    assert (wav.parent / "take.wav.downmix.peaks.json").exists()

    # second call comes from the cache (fingerprints still match)
    cached = peaks.compute_peaks(wav, buckets=500)
    assert cached == {k: v for k, v in out.items()}


def test_cache_invalidated_by_touch(wav):
    first = peaks.compute_peaks(wav, buckets=100)
    import os
    st = os.stat(wav)
    os.utime(wav, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    again = peaks.compute_peaks(wav, buckets=100)
    assert again["peaks"] == first["peaks"]     # recomputed, same audio
    data = json.loads((wav.parent / "take.wav.downmix.peaks.json").read_text())
    assert data["mtime_ns"] == st.st_mtime_ns + 1_000_000


def test_channel_changes_cache_key(wav):
    peaks.compute_peaks(wav, channel="left", buckets=100)
    assert (wav.parent / "take.wav.left.peaks.json").exists()
    # single-channel source: left channel equals the mono downmix
    a = peaks.compute_peaks(wav, channel="left", buckets=100)
    b = peaks.compute_peaks(wav, channel="downmix", buckets=100)
    assert a["peaks"] == b["peaks"]
