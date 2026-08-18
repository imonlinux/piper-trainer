"""Environment self-check. Run this first; it verifies every layer that has
ever silently failed in this stack."""
from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path


def check() -> tuple[list[str], bool]:
    lines, ok = [], True

    def mark(good: bool, msg: str) -> None:
        nonlocal ok
        lines.append(("✓ " if good else "✗ ") + msg)
        if not good:
            ok = False

    # binaries
    for exe in ("ffmpeg", "ffprobe", "espeak-ng", "deep-filter"):
        mark(shutil.which(exe) is not None, f"{exe} on PATH")

    # piper C extensions — both have silently failed to build in the past
    for mod, label in (
        ("piper.espeakbridge", "espeakbridge (CMake extension)"),
        ("piper.train.vits.monotonic_align.monotonic_align.core",
         "monotonic_align (Cython extension)"),
    ):
        try:
            importlib.import_module(mod)
            mark(True, label)
        except Exception as exc:  # noqa: BLE001
            mark(False, f"{label}: {exc}")

    # pipeline libraries
    for mod in ("auditok", "faster_whisper", "onnxscript", "lightning"):
        try:
            importlib.import_module(mod)
            mark(True, f"{mod}")
        except Exception as exc:  # noqa: BLE001
            mark(False, f"{mod}: {exc}")

    # accelerator
    try:
        import torch
        lines.append(f"· torch {torch.__version__}")
        hip = getattr(torch.version, "hip", None)
        cuda = getattr(torch.version, "cuda", None)
        lines.append(f"· backend: {'ROCm ' + hip if hip else 'CUDA ' + str(cuda)}")
        avail = torch.cuda.is_available()
        mark(avail, f"GPU available ({torch.cuda.device_count()} device(s))")
        if avail:
            lines.append(f"· device: {torch.cuda.get_device_name(0)}")
        elif hip:
            lines.append("  hint: ROCm needs /dev/kfd + /dev/dri passed through, "
                         "and the container user in the 'video' group")
        else:
            lines.append("  hint: NVIDIA needs --gpus all and a driver new "
                         "enough for this torch build")
    except Exception as exc:  # noqa: BLE001
        mark(False, f"torch: {exc}")

    # export patch
    try:
        src = Path("/opt/piper1-gpl/src/piper/train/export_onnx.py")
        if src.exists():
            mark("dynamo=False" in src.read_text(),
                 "export_onnx patched with dynamo=False")
    except Exception:  # noqa: BLE001
        pass

    # workspace writability — matters when running as --user
    try:
        probe = Path("/workspace/.piper-trainer-write-test")
        probe.write_text("ok")
        probe.unlink()
        mark(True, "/workspace writable")
    except Exception as exc:  # noqa: BLE001
        mark(False, f"/workspace not writable: {exc}")

    return lines, ok


def espeak_voices(prefix: str = "") -> list[str]:
    out = subprocess.run(["espeak-ng", "--voices"], capture_output=True,
                         text=True, check=True).stdout
    names = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) > 1 and parts[1].startswith(prefix):
            names.append(parts[1])
    return names
