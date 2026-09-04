"""Instant synthesis preview for an exported voice (§4.6 say).

Runs piper1-gpl's CLI as a subprocess rather than calling its Python API:
the CLI flags are the stable surface, an inference crash stays out of the
API server process, and the wav comes back on stdout ("--output-file -"),
so a preview the operator throws away never touches the filesystem.

Synthesis parameters fall back to the voice's own stored defaults (the
.onnx.json inference block), so `curl .../say -d '{"text": "hi"}'` sounds
like the exported voice, while the UI sends its slider values explicitly.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# A say preview is a sentence, not a batch job — cap it before the
# subprocess, not as a pydantic constraint, so every caller is covered.
MAX_TEXT = 1000
DEFAULT_TIMEOUT = 120.0


class SayError(RuntimeError):
    pass


def synthesize(onnx_path: Path, json_path: Path, text: str,
               length_scale: float | None = None,
               noise_scale: float | None = None,
               noise_w: float | None = None,
               timeout: float = DEFAULT_TIMEOUT) -> bytes:
    """Synthesize text and return the wav bytes. Raises ValueError on bad
    text, SayError when piper is missing, times out, or fails."""
    if not text.strip():
        raise ValueError("nothing to say")
    if len(text) > MAX_TEXT:
        raise ValueError(f"text too long (max {MAX_TEXT} characters)")

    inf = json.loads(json_path.read_text()).get("inference", {})
    ls = inf.get("length_scale", 1.0) if length_scale is None else length_scale
    ns = inf.get("noise_scale", 0.667) if noise_scale is None else noise_scale
    nw = inf.get("noise_w", 0.8) if noise_w is None else noise_w

    cmd = [sys.executable, "-m", "piper",
           "--model", str(onnx_path),
           "--config", str(json_path),
           "--length-scale", str(ls),
           "--noise-scale", str(ns),
           "--noise-w", str(nw),
           "--output-file", "-"]
    try:
        proc = subprocess.run(cmd, input=text.encode("utf-8"),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout)
    except FileNotFoundError:
        raise SayError(
            "piper CLI not found — the training image carries it; "
            "run `piper-trainer doctor`") from None
    except subprocess.TimeoutExpired:
        raise SayError(f"synthesis timed out after {timeout:.0f}s") from None
    if proc.returncode != 0:
        lines = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise SayError(lines[-1] if lines
                       else f"piper exited {proc.returncode}")
    if not proc.stdout.startswith(b"RIFF"):
        raise SayError("piper produced no wav — checkpoint and voice "
                       "config may not match the installed piper")
    return proc.stdout
