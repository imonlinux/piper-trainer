# piper-trainer

A packaged, reproducible training pipeline for [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) — dataset prep through to a deployable Home Assistant voice.

Existing Piper training GUIs wrap the **archived** `rhasspy/piper` toolchain. This packages the current one, with every Linux build trap pre-solved, and adds a ROCm/gfx1151 variant that doesn't otherwise exist.

## What's in the image

| Layer | Notes |
|---|---|
| Python 3.12 | 3.13/3.14 have no ROCm wheels and break dependency resolution |
| torch | CUDA (cu124) or ROCm gfx1151 via `TORCH_INDEX_URL` + `TORCH_VERSION` build args; wheels bundle their own runtime |
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
`auditok` and `faster-whisper`), but you are then on the hook for the rest
yourself. `piper-trainer doctor` reports exactly what is missing in any
environment.

## Build

```bash
# NVIDIA
docker compose --profile cuda build

# AMD Strix Halo / gfx1151
docker compose --profile rocm build

# CPU fallback
docker compose --profile cpu build
```

A raw `docker build` / `podman build` bypasses these profiles. The NVIDIA/CPU defaults are enough on their own, but an ROCm build must pass both args itself — the gfx1151 index is a rolling nightly that has already dropped older generations:

```bash
docker build -t piper-trainer:rocm \
  --build-arg TORCH_INDEX_URL=https://rocm.nightlies.amd.com/v2/gfx1151/ \
  --build-arg TORCH_VERSION=2.10.0 .
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

$R init /workspace/recordings --name marvin --espeak-voice en-gb-x-rp
# drop source recordings into /workspace/recordings/raw/

$R sources    /workspace/recordings          # what ffprobe makes of them
$R prepare    /workspace/recordings --energy-threshold 55
$R transcribe /workspace/recordings --language en
# review /workspace/recordings/dataset/audit.csv, fix transcripts

$R validate /workspace/recordings --batch-size 16
$R clean    /workspace/recordings                            # dry run
$R clean    /workspace/recordings --apply
$R train    /workspace/recordings --batch-size 16 --max-epochs 4000 \
            --warmstart /workspace/recordings/base_checkpoints/alan-medium.ckpt
$R export   /workspace/recordings --voice-name marvin-medium --length-scale 1.25
```

`--espeak-voice`, `--tier`, and `--name` appear only on `init`: all three are recorded in `project.json` and every later command reads them. `--name` is the voice name used in the config, checkpoint, and export filenames — the project directory can be called anything. Pass any of them again only to change them, which prints a warning.

**Do not skip `init`.** A project without `init` has no `project.json`, so there is no saved espeak voice — and every later command then **falls back to `en-us`**, printing a one-line warning to stderr. That is not a labeling error at export time: `train` phonemizes with `en-us` too, so the model itself is trained with the wrong accent and the defect only reveals itself when you listen to the result. If you already trained such a project, re-run `train` with an explicit `--espeak-voice` (which records it in `project.json` for future runs) after deleting the tier's `cache-*/` and `runs-*/` directories — both are phonemized with the old voice and cannot be reused.

Then copy `out/marvin-medium.onnx` and `out/marvin-medium.onnx.json` to your wyoming-piper data directory, restart it, and reload the Wyoming integration.

### Useful flags

| Flag | Command | Effect |
|---|---|---|
| `--force` | `prepare` | re-run stages even when nothing changed |
| `--retranscribe` | `transcribe` | re-do every clip instead of only the missing ones |
| `--add-epochs N` | `train` | train N more epochs from the resumed checkpoint |
| `--validation-split` | `train` | override the 0.02 default (use `0` on small sets) |
| `--files-only` | `restore` | move WAVs back without restoring their metadata rows |
| `--wait N` | any mutating command | wait up to N seconds for the project lock |

### Resuming

```bash
$R train /workspace/recordings --resume auto --add-epochs 1000
```

`--resume auto` finds the latest checkpoint; `--add-epochs N` trains N more epochs from wherever it left off.

Prefer `--add-epochs` over `--max-epochs` when resuming. Lightning's `--trainer.max_epochs` is an **absolute ceiling** and `--ckpt_path` restores the epoch counter, so resuming a checkpoint at epoch 9999 with `--max-epochs 4000` exits immediately having trained nothing. `--add-epochs` does the arithmetic for you, and passing `--max-epochs` below the checkpoint's epoch is now refused with a message naming both numbers.

**`--resume` vs `--warmstart`** — the distinction is worth getting right:

| | `--resume` | `--warmstart` |
|---|---|---|
| Lightning flag | `--ckpt_path` | `--model.warmstart_ckpt` |
| Restores | weights, optimizer state, epoch counter | weights only |
| Epoch count starts at | the checkpoint's epoch | zero |
| Use when | continuing **your own** run | fine-tuning from a **different** voice |

The first run of a tier records `target_epochs` in `project.json`; a resume never overwrites it, so it stays meaningful as a progress denominator.

## Source filenames

`prepare` normalizes source names on the way in, because they flow through clip names into `metadata.csv` and eventually into shell and container arguments.

- Stems are reduced to `[A-Za-z0-9._-]`; other characters become `_`.
- Two sources that would land on the same name are disambiguated by extension, then by index: `foo.wav` and `foo.mp3` become `foo_wav.wav` and `foo_mp3.wav`; `foo bar.wav` and `foo?bar.wav` become `foo_bar_wav.wav` and `foo_bar_wav_2.wav`.
- Naming is deterministic for a given set of sources — the same `raw/` produces the same `48k/` every time.
- The full original → final mapping is printed in the `prepare` output, which is the fastest way to work out where a particular recording went.

## Re-running is cheap

Every `prepare` stage records what produced its output in a `.stage.json` alongside it: the parameters it ran with, and a fingerprint of its inputs. On the next run a stage whose parameters and inputs both match is **skipped**.

```
$ $R prepare /workspace/recordings --energy-threshold 45
converted: skipped
denoised: skipped
clips: 8
finalized: 8
```

Changing `--energy-threshold` re-runs segmentation and everything downstream, but not the conversion and denoising that the change cannot affect. `--force` re-runs regardless.

**Each stage owns its output directory and clears it first.** This matters more than the speed: before, re-segmenting with different parameters left the previous run's clips in place — they had different names, so nothing overwrote them — and `finalize` copied *both sets* into `dataset/wavs/`. A dataset could silently contain the output of two different parameter sets with nothing reporting it. Now a stage's output is exactly what its current parameters produce.

### Tuning segmentation

`--energy-threshold` is the loudness floor above which audio counts as speech, and it is genuinely per-recording. Denoised audio usually wants a **lower** value than the 55 default, because DeepFilterNet has already removed the noise floor the default was calibrated against.

Clip count is **not monotonic** in the threshold. Lowering it makes pauses read as speech, so adjacent utterances merge into fewer, longer clips — on one test set, threshold 55 produced 11 clips and threshold 45 produced 8. Sweep a few values, then *listen*: clips should start and end on word boundaries, with the first consonant intact and no long silent stretches.

`--max-silence` is the companion dial. Threshold decides what counts as speech; max-silence decides how far apart two speech regions can be and still belong to the same clip. Raise it if single sentences are being cut at breath pauses; lower it if clips are merging several sentences.

## One writer per project

Mutating commands take an exclusive lock on the project directory. A second one refuses rather than racing on `metadata.csv`, `dataset/wavs/`, and the checkpoint directory:

```
$ $R train /workspace/recordings
project is locked by 'prepare' (started 2026-08-20T22:22:20Z on c4f98cb14bb7);
'train' cannot run concurrently. Wait for it to finish, stop it, or pass
--wait N to wait up to N seconds.
```

| Locked | Not locked |
|---|---|
| `prepare`, `transcribe`, `clean --apply`, `restore`, `train`, `export` | `validate`, `sources`, `doctor`, `clean` dry run |

Read-only commands never block, so you can validate a dataset while a training run is in progress.

**There is no stale-lock cleanup, because there cannot be a stale lock.** The mechanism is an `fcntl` advisory lock, which the kernel releases when the holding process dies — including on `SIGKILL` and on container teardown. If you kill a run, the next command acquires immediately. Any `.piper-trainer.lock` file left behind is an empty shell, not a lock.

One limitation worth stating: `flock` is per-kernel. This protects concurrent containers and CLI runs on the same host, which is the deployment model. It does **not** protect a project directory shared over NFS between machines.

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
| `crlf` | repair | rewrite `metadata.csv` with LF endings |
| `columns`, `blank-row` | repair | drop the malformed or empty lines |
| `unspoken-text` | repair | expand `Mr.` → `Mister`, `&` → `and`, single digits → words. Deleting a clip over a *text* problem throws away good audio. Multi-digit numbers are left for a human — "1984" is ambiguous. |
| `missing-wav` | drop row | the audio isn't there |
| `orphan-wav`, `short-clips`, `long-clips`, `cps-outliers`, `sample-rate`, `channels`, `unreadable` | quarantine | judgment calls; moved to `dataset/quarantine/` with a manifest |

```bash
$R clean /workspace/recordings                                  # show the plan
$R clean /workspace/recordings --apply
$R clean /workspace/recordings --apply --only orphan-wav,short-clips
$R clean /workspace/recordings --apply --exclude cps-outliers
$R restore /workspace/recordings                                # undo quarantine
```

Three safety rails:

- **Nothing is deleted.** Clips move to `dataset/quarantine/` and `manifest.csv` records the clip id, the reasons, and its transcript. `restore` moves the files back **and re-adds their rows** from that manifest; `--files-only` moves the files alone if you would rather re-transcribe. An existing row is never overwritten — the conflict is reported instead.
- **`--only` and `--exclude` bound the whole operation.** A run that excludes `columns` preserves malformed lines verbatim rather than tidying them away as a side effect. Line-ending normalization is the sole exception and is always applied, because the canonical writer emits LF by construction.
- **Mass culls are refused.** If a run would remove more than a third of the dataset it aborts, on the theory that a validation threshold is wrong rather than the data. `--force` overrides.

A `cps-outlier` flag is a hint to *listen*, not proof a clip is bad — `--exclude cps-outliers` is a reasonable default on small datasets.

`export` writes a **complete** `.onnx.json` — piper1-gpl omits `dataset`, `audio.quality`, and the `language` block — and enforces the rule that the `.onnx` stem, the `dataset` field, and the name the client requests are all the same string.

## Project layout on the mounted volume

```
recordings/
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

## API server and Bones UI

The same pipeline functions the CLI calls, served over HTTP, with a job runner in front of everything long-running. Specified in `docs/piper-trainer-ui-design.md` (§1 jobs, §4 endpoints, §9 "Bones" scope); the API is the product and the UI is just its first client.

```bash
pip install -e '.[api]'
piper-trainer serve                      # http://127.0.0.1:8000/
```

The training images carry the api runtime too, so the same thing works from the image (workspace and GPU passthrough come from compose):

```bash
docker compose --profile rocm run --rm --service-ports trainer-rocm serve --host 0.0.0.0
```

The Bones UI is served at `/` — project list and detail, upload, checkpoint picker, run/cancel buttons, live log tail. It is deliberately ugly static HTML; React arrives with the segment tuner.

The job runner (`api/jobs.py`) makes four decisions worth knowing before you extend it:

- **Jobs are processes, not threads.** Each job is `python -m piper_trainer.api.runner <job-dir>` in its own session; a crash or OOM kill cannot take the API down. `job.json` is the record, written only by the manager (atomic tmp+rename). The runner never touches it — it reports through structured stdout lines (`##TARGET`/`##PROGRESS`/`##RESULT` + JSON) and its exit code, so there is no two-writer race by construction.
- **State lives on disk, and the log file is the source of truth.** Every output line is teed to `jobs/<id>/log.txt` and fanned out to websocket subscribers; a browser refresh mid-run loses nothing.
- **One running job per project** unless `PIPER_ALLOW_PARALLEL` is set — everything else queues in submit order. Cancel is SIGTERM to the process group with a `PIPER_CANCEL_GRACE` (default 30 s) head start so Lightning can write `last.ckpt`, then SIGKILL.
- **Nothing restarts itself.** On startup the manager marks `running` jobs whose PID is gone as `failed: interrupted`. Queued jobs from a previous lifetime are surfaced but never adopted automatically — `POST /api/jobs/{id}/start` adopts one deliberately.

Ingest uploads stream straight into the job's `incoming/` staging dir; the job sanitizes filenames into `raw/` and records the original→stored mapping in its result. The checkpoint picker reads the `rhasspy/piper-checkpoints` catalog live (3 s timeout) and falls back to a bundled snapshot (`api/catalog_snapshot.json`, regenerated by `scripts/build_catalog_snapshot.py`) so the UI works offline.

## Development

```bash
pip install -e '.[dev,api]'
pytest -q
```

The suite is pure logic — no network, no GPU, no audio processing — so it runs anywhere in a couple of seconds. WAV fixtures are generated with the `wave` module inside `tmp_path`; nothing binary is committed. The job tests spawn real stub subprocesses (`python -c ...`) to exercise the supervision loop; `fetch=`-style seams keep the catalog and doctor tests offline.

Two conventions worth keeping if you extend it:

- **`metadata.py` is the only module that reads or writes `metadata.csv`.** Three modules used to implement that format independently and disagreed about line endings, delimiter escaping, and how to report malformed rows. Everything goes through the one reader and the one writer.
- **New tests are mutation-checked**: reintroduce the bug the test exists to catch, confirm exactly one test fails, restore. A test that passes with the bug present is not a test. This has already caught one genuinely inadequate test and one gap in coverage.

## Status

The CLI is feature-complete and verified end to end in the container.

| | |
|---|---|
| Verified with real binaries | filename normalization and collision handling, stage idempotency and skipping, project locking including SIGKILL release |
| Verified by unit tests | metadata format contract, `clean` planning and application, `restore`, training argument construction, export config patching, lock semantics; API surface, job scheduling/cancel/rescan, runner stages, catalog fallback |
| Not yet exercised end to end | a full `transcribe` → `train` → `export` run inside the container; the CPU image variant; a real `train` job through the API (the GPU path is stubbed in tests) |
