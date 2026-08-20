# piper-trainer

A packaged, reproducible training pipeline for [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) — dataset prep through to a deployable Home Assistant voice.

Existing Piper training GUIs wrap the **archived** `rhasspy/piper` toolchain. This packages the current one, with every Linux build trap pre-solved, and adds a ROCm/gfx1151 variant that doesn't otherwise exist.

## What's in the image

| Layer | Notes |
|---|---|
| Python 3.12 | 3.13/3.14 have no ROCm wheels and break dependency resolution |
| torch | CUDA or ROCm gfx1151 via `TORCH_INDEX_URL` build arg; wheels bundle their own runtime |
| piper1-gpl | with **both** C extensions actually built |
| DeepFilterNet 3 | static musl binary, no Python deps |
| auditok, faster-whisper, ffmpeg, espeak-ng | the dataset pipeline |
| `piper-trainer` CLI | subcommands mapping 1:1 to the planned UI screens |

### The build traps this solves

`pip install -e '.[train]'` reports success while omitting two required C extensions, because `pyproject.toml` declares `build-backend = "setuptools.build_meta"` while `setup.py` uses `skbuild.setup()` — pip never runs the CMake path. The image builds them explicitly and **fails the build** if either artifact is missing:

1. `espeakbridge` (CMake) → `python3 setup.py build_ext --inplace`
2. `monotonic_align` (Cython) → `./build_monotonic_align.sh` (nested package layout)

It also patches `export_onnx.py` with `dynamo=False`, because recent torch defaults to the dynamo exporter, which can't trace VITS's stochastic duration predictor and dies with `GuardOnDataDependentSymNode`.

## Installation

The supported path is the **container** — build it and use `./run.sh`. The
default `pip install piper-trainer` is deliberately dependency-free and will
import fine but fail at the first real call: the pipeline needs `piper1-gpl`
itself (including its two C extensions, `espeakbridge` and `monotonic_align`),
plus `ffmpeg`, `espeak-ng`, and `deep-filter` on `PATH` — none of which are
pip-installable dependencies. A standalone install is at least *possible*
with the `[runtime]` extra (`pip install 'piper-trainer[runtime]'`, covering
`auditok`, `faster-whisper`, `soundfile`, `numpy`), but you are then on the
hook for the rest yourself. `piper-trainer doctor` reports exactly what is
missing in any environment.

## Build

```bash
# NVIDIA
docker compose --profile cuda build

# AMD Strix Halo / gfx1151
docker compose --profile rocm build

# CPU fallback
docker compose --profile cpu build
```

## Run

Use `./run.sh` — it handles the runtime differences that trip up compose files:

```bash
export WORKSPACE=/path/to/voice-training
./run.sh doctor                      # VARIANT=rocm by default
VARIANT=cuda ./run.sh doctor
SHELL_IN=1 ./run.sh                  # drop into a shell for poking around
```

<details>
<summary>What run.sh handles, and why compose can't</summary>

| Situation | Requirement |
|---|---|
| rootless podman | **No `--user`.** Your host UID already maps to container root; adding `--user 1000:1000` maps to a subordinate UID with no write access to the volume — the mount appears read-only. |
| rootful docker | `--user` **is** required, or every file on the volume ends up root-owned. |
| podman + AMD | `--group-add keep-groups`. Podman resolves group *names* inside the container, and Debian images have no `render` group → `Unable to find group render`. |
| docker + AMD | numeric GIDs from `getent group video render`. |
| SELinux (Fedora/Nobara) | `:Z` on the volume, or every write is `Permission denied`. |
| podman + NVIDIA | `--device nvidia.com/gpu=all` (CDI), not `--gpus all`. |

`podman-compose` is a third-party reimplementation with incomplete device and volume handling; the equivalent `podman run` worked when `podman-compose` did not. The compose file is kept for rootful Docker.
</details>

Verified working (Beelink GTR9 Pro, Nobara 44, rootless podman):

```
✓ espeakbridge (CMake extension)
✓ monotonic_align (Cython extension)
· torch 2.10.0+rocm7.13.0a...  · backend: ROCm 7.13
✓ GPU available (1 device(s))  · device: Radeon 8060S Graphics
✓ export_onnx patched with dynamo=False
✓ /workspace writable
```

`doctor` verifies every layer that has silently failed in this stack: binaries on PATH, both C extensions importable, pipeline libraries, GPU visibility, the export patch, and workspace writability. Run it first.

## Pipeline

```bash
R="./run.sh"

$R init /workspace/marvin --name marvin --espeak-voice en-gb-x-rp
# drop source recordings into /workspace/marvin/raw/

$R prepare  /workspace/marvin --tier medium --energy-threshold 55
$R transcribe /workspace/marvin --language en
# review /workspace/marvin/dataset/audit.csv, fix transcripts

$R validate /workspace/marvin --tier medium --batch-size 16 --espeak-voice en-gb-x-rp
$R clean    /workspace/marvin --tier medium              # dry run
$R clean    /workspace/marvin --tier medium --apply
$R train    /workspace/marvin --tier medium --espeak-voice en-gb-x-rp \
            --batch-size 16 --max-epochs 4000 \
            --warmstart /workspace/marvin/base_checkpoints/alan-medium.ckpt
$R export   /workspace/marvin --tier medium --espeak-voice en-gb-x-rp \
            --voice-name marvin-medium --length-scale 1.25
```

Then copy `out/marvin-medium.onnx` and `out/marvin-medium.onnx.json` to your wyoming-piper data directory, restart it, and reload the Wyoming integration.

### espeak voice is a project setting

`init --espeak-voice` records it in `project.json`, and `train`/`export` use it unless overridden. This matters more than it looks: a run that silently defaults to `en-us` on a British voice re-phonemizes the whole dataset against the wrong accent, trains cleanly, and only reveals itself when you listen. Changing it later prints a warning, because checkpoints from the previous voice will not transfer cleanly.

### Resuming

```bash
$R train /workspace/marvin --resume auto      # finds the latest checkpoint
```

`--resume` uses Lightning's `--ckpt_path` (restores optimizer state and the epoch counter, so `--max-epochs` must exceed it). `--warmstart` uses `--model.warmstart_ckpt` (weights only, epoch count starts at zero) and is the right choice when fine-tuning from a *different* voice.

## Guardrails

`validate` runs automatically before training and **refuses to start** on errors:

- delimiter/CRLF/blank-row problems in `metadata.csv`
- rows referencing missing WAVs, and WAVs with no row
- sample rate or channel count wrong for the tier
- digits, symbols, or abbreviations in transcripts (espeak phonemizes literally)
- chars-per-second outliers (truncated audio, hallucinated text, mispairs)
- batch size ≥ train split (one batch per epoch)
- espeak voice not recognised
- dataset shorter than the practical minimum
- with `--checkpoint`: tier/sample-rate mismatch, and the 130-vs-256 vocabulary question

## clean

`clean` acts on validation findings. It is a **dry run unless `--apply`**, and it never deletes anything.

| Finding | Action | Why |
|---|---|---|
| `crlf`, `columns`, `blank-row` | repair | rewrite `metadata.csv` |
| `unspoken-text` | repair | expand `Mr.` → `Mister`, `&` → `and`, single digits → words. Deleting a clip over a *text* problem throws away good audio. Multi-digit numbers are left for a human — "1984" is ambiguous. |
| `missing-wav` | drop row | the audio isn't there |
| `orphan-wav`, `short-clips`, `long-clips`, `cps-outliers`, `sample-rate`, `channels`, `unreadable` | quarantine | judgment calls; moved to `dataset/quarantine/` with a manifest |

```bash
$R clean /workspace/marvin                                  # show the plan
$R clean /workspace/marvin --apply
$R clean /workspace/marvin --apply --only orphan-wav,short-clips
$R clean /workspace/marvin --apply --exclude cps-outliers
$R restore /workspace/marvin                                # undo quarantine
```

Two safety rails:

- **Nothing is deleted.** Clips move to `dataset/quarantine/` and `manifest.csv` records the clip id, the reasons, and its transcript. `restore` moves them back.
- **Mass culls are refused.** If a run would remove more than a third of the dataset it aborts, on the theory that a validation threshold is wrong rather than the data. `--force` overrides.

A `cps-outlier` flag is a hint to *listen*, not proof a clip is bad — `--exclude cps-outliers` is a reasonable default on small datasets.

`export` writes a **complete** `.onnx.json` — piper1-gpl omits `dataset`, `audio.quality`, and the `language` block — and enforces the rule that the `.onnx` stem, the `dataset` field, and the name the client requests are all the same string.

## Project layout on the mounted volume

```
marvin/
├── project.json
├── raw/                  # your source recordings
├── work/
│   ├── 48k/              # ffmpeg output
│   ├── denoised/         # DeepFilterNet
│   └── clips/            # auditok segments
├── dataset/
│   ├── metadata.csv      # id|text
│   ├── audit.csv         # duration, chars/sec, lang_prob, text
│   └── wavs/
├── base_checkpoints/
├── cache-medium/         # trainer's preprocessed tensors
├── runs-medium/          # lightning_logs live here, not the CWD
└── out/                  # .onnx + .onnx.json + training config
```

Deliberately human-readable and toolchain-compatible: every file works with the bare piper1-gpl CLI. The dataset is the durable asset — it outlives Piper, VITS, and this project.

## Status

**ROCm image builds and passes `doctor` on gfx1151.** The CUDA image has
been exercised in-container (work order 03): adversarial `raw/` naming
(spaces, unicode, collisions), idempotent `prepare` against real
ffmpeg/DeepFilterNet/auditok — including skip-on-repeat across container
restarts — and the project lock (concurrent refusal naming the holder, and
immediate release on SIGKILL). `transcribe`/`train`/`export` share the same
code paths but have not been run end to end in-container. The CPU variant
shares the same Dockerfile and is untested.

Next: a FastAPI + worker layer over these same functions, with a waveform view for auditok threshold tuning and a sortable audit table for transcript review.
