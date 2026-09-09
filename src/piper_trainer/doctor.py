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
        avail = torch.cuda.is_available()
        if cuda is None and hip is None:
            # CPU-only build (the 'cpu' image variant): no GPU is expected,
            # so its absence is informational, not a fault
            lines.append("· CPU-only torch build — GPU check skipped")
        else:
            lines.append(f"· backend: {'ROCm ' + hip if hip else 'CUDA ' + str(cuda)}")
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


def piper_espeak_voices() -> list[str] | None:
    """Voice names from the espeak-ng data piper1-gpl actually phonemizes
    with: the copy bundled inside the installed `piper` package (see
    piper/phonemize_espeak.py). This is not the system espeak-ng — the
    bundled data has no plain en-gb, which is how a British project could
    pass every check and die on the first phonemize. A voice's name is the
    `language` directive of its voice file (what espeak-ng --voices lists
    and SetVoiceByName resolves), falling back to the file stem; the branch
    directory (lang/gmw/) is never part of the name. None means the
    bundled data is not importable here and the caller should fall back
    to the system voice list."""
    try:
        from piper.phonemize_espeak import ESPEAK_DATA_DIR
    except Exception:  # noqa: BLE001 — any import failure means: fall back
        return None
    lang = Path(ESPEAK_DATA_DIR) / "lang"
    if not lang.is_dir():
        return None
    names: set[str] = set()
    for f in lang.rglob("*"):
        if not f.is_file():
            continue
        ident = ""
        try:
            for line in f.read_text(errors="replace").splitlines():
                line = line.strip()
                if line.startswith("language "):
                    # "language en-us 2": the trailing number is a
                    # priority, not part of the voice name — keeping it
                    # produced invalid names like "en-us 2"
                    ident = line.split()[1]
                    break
        except OSError:
            continue
        names.add((ident or f.stem).lower())
    return sorted(names)
