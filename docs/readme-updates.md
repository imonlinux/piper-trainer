# README updates for WO1–WO3

Replacement and new sections. Ordered as they should appear in the file.

---

## REPLACE: the `[runtime]` sentence in **Installation**

The extra lists two packages `src/` never imports. Change:

> A standalone install is at least *possible* with the `[runtime]` extra (`pip install 'piper-trainer[runtime]'`, covering `auditok`, `faster-whisper`, `soundfile`, `numpy`), but you are then on the hook for the rest yourself.

to:

> A standalone install is at least *possible* with the `[runtime]` extra (`pip install 'piper-trainer[runtime]'`, covering `auditok` and `faster-whisper`), but you are then on the hook for the rest yourself.

Update `pyproject.toml` to match — `soundfile` and `numpy` arrive transitively and should not be declared.

---

## REPLACE: **Pipeline**

## Pipeline

```bash
R="./run.sh"

$R init /workspace/marvin --name marvin --espeak-voice en-gb-x-rp
# drop source recordings into /workspace/marvin/raw/

$R sources    /workspace/marvin          # what ffprobe makes of them
$R prepare    /workspace/marvin --energy-threshold 55
$R transcribe /workspace/marvin --language en
# review /workspace/marvin/dataset/audit.csv, fix transcripts

$R validate /workspace/marvin --batch-size 16
$R clean    /workspace/marvin                            # dry run
$R clean    /workspace/marvin --apply
$R train    /workspace/marvin --batch-size 16 --max-epochs 4000 \
            --warmstart /workspace/marvin/base_checkpoints/alan-medium.ckpt
$R export   /workspace/marvin --voice-name marvin-medium --length-scale 1.25
```

`--espeak-voice` and `--tier` appear only on `init`: both are recorded in `project.json` and every later command reads them. Pass them again only to change them, which prints a warning.

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

---

## NEW: after **Pipeline**

## Re-running is cheap

Every `prepare` stage records what produced its output in a `.stage.json` alongside it: the parameters it ran with, and a fingerprint of its inputs. On the next run a stage whose parameters and inputs both match is **skipped**.

```
$ $R prepare /workspace/marvin --energy-threshold 45
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

---

## NEW: after **Re-running is cheap**

## One writer per project

Mutating commands take an exclusive lock on the project directory. A second one refuses rather than racing on `metadata.csv`, `dataset/wavs/`, and the checkpoint directory:

```
$ $R train /workspace/marvin
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

---

## NEW: inside **Pipeline**, or immediately after

## Source filenames

`prepare` normalizes source names on the way in, because they flow through clip names into `metadata.csv` and eventually into shell and container arguments.

- Stems are reduced to `[A-Za-z0-9._-]`; other characters become `_`.
- Two sources that would land on the same name are disambiguated by extension, then by index: `foo.wav` and `foo.mp3` become `foo_wav.wav` and `foo_mp3.wav`; `foo bar.wav` and `foo?bar.wav` become `foo_bar_wav.wav` and `foo_bar_wav_2.wav`.
- Naming is deterministic for a given set of sources — the same `raw/` produces the same `48k/` every time.
- The full original → final mapping is printed in the `prepare` output, which is the fastest way to work out where a particular recording went.

---

## REPLACE: **Resuming**

### Resuming

```bash
$R train /workspace/marvin --resume auto --add-epochs 1000
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

---

## REPLACE: the action table and safety rails in **clean**

| Finding | Action | Why |
|---|---|---|
| `crlf` | repair | rewrite `metadata.csv` with LF endings |
| `columns`, `blank-row` | repair | drop the malformed or empty lines |
| `unspoken-text` | repair | expand `Mr.` → `Mister`, `&` → `and`, single digits → words. Deleting a clip over a *text* problem throws away good audio. Multi-digit numbers are left for a human — "1984" is ambiguous. |
| `missing-wav` | drop row | the audio isn't there |
| `orphan-wav`, `short-clips`, `long-clips`, `cps-outliers`, `sample-rate`, `channels`, `unreadable` | quarantine | judgment calls; moved to `dataset/quarantine/` with a manifest |

Three safety rails:

- **Nothing is deleted.** Clips move to `dataset/quarantine/` and `manifest.csv` records the clip id, the reasons, and its transcript. `restore` moves the files back **and re-adds their rows** from that manifest; `--files-only` moves the files alone if you would rather re-transcribe. An existing row is never overwritten — the conflict is reported instead.
- **`--only` and `--exclude` bound the whole operation.** A run that excludes `columns` preserves malformed lines verbatim rather than tidying them away as a side effect. Line-ending normalization is the sole exception and is always applied, because the canonical writer emits LF by construction.
- **Mass culls are refused.** If a run would remove more than a third of the dataset it aborts, on the theory that a validation threshold is wrong rather than the data. `--force` overrides.

A `cps-outlier` flag is a hint to *listen*, not proof a clip is bad — `--exclude cps-outliers` is a reasonable default on small datasets.

---

## NEW: before **Status**

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

The suite is pure logic — no network, no GPU, no audio processing — so it runs anywhere in well under a second. WAV fixtures are generated with the `wave` module inside `tmp_path`; nothing binary is committed.

Two conventions worth keeping if you extend it:

- **`metadata.py` is the only module that reads or writes `metadata.csv`.** Three modules used to implement that format independently and disagreed about line endings, delimiter escaping, and how to report malformed rows. Everything goes through the one reader and the one writer.
- **New tests are mutation-checked**: reintroduce the bug the test exists to catch, confirm exactly one test fails, restore. A test that passes with the bug present is not a test. This has already caught one genuinely inadequate test and one gap in coverage.

---

## REPLACE: **Status**

## Status

The CLI is feature-complete and verified end to end in the container.

| | |
|---|---|
| Verified with real binaries | filename normalization and collision handling, stage idempotency and skipping, project locking including SIGKILL release |
| Verified by unit tests | metadata format contract, `clean` planning and application, `restore`, training argument construction, export config patching, lock semantics |
| Not yet exercised end to end | a full `transcribe` → `train` → `export` run inside the container; the CPU image variant |

Next: the FastAPI layer and job runner, specified in `docs/piper-trainer-ui-design.md` §1 and §4.
