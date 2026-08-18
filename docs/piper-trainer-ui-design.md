# piper-trainer UI — Design Document

**Status:** v1 — approved for implementation
**Scope:** the API and job layer, plus the screen sequence. Deliberately fixes the parts that are expensive to change and leaves layout and styling to iteration.
**Stack:** FastAPI + a worker process; React frontend; no database — state lives on the mounted volume.

---

## 0. Principles

1. **The API is the product; the UI is a client.** Both the web UI and the CLI call the same Python functions in `piper_trainer.*`. Neither shells out to the other. If a capability exists in one, it exists in both.
2. **State lives on the mounted volume, human-readable.** No database. A project directory is fully self-describing and survives the container being deleted, this project being abandoned, or the user dropping to the bare `piper1-gpl` CLI. Same rule as the dataset itself.
3. **Nothing is destroyed.** `clean` quarantines rather than deletes; previews write to scratch; full runs never overwrite a previous run's checkpoints.
4. **Judgment stays with the human.** The tool measures, flags, and previews. It does not decide that a clip is bad or that a voice is finished.
5. **Every long operation is a job.** Uniform lifecycle, uniform log streaming, survives a browser refresh.

---

## 1. Job model

The single most important decision, because training runs take hours and everything else hangs off how they're represented.

### 1.1 What a job is

```jsonc
{
  "id": "20260817T014233Z-train-a1b2",
  "kind": "prepare|transcribe|validate|clean|train|export|preview",
  "stage": "segment",            // preview only: which stage is being previewed
  "project": "hal-9000",
  "params": { "...": "as submitted" },
  "state": "queued|running|succeeded|failed|cancelled",
  "created_at": "2026-08-17T01:42:33Z",
  "started_at": "2026-08-17T01:42:34Z",
  "finished_at": null,
  "exit_code": null,
  "pid": 4711,
  "progress": { "current": 1284, "total": 4000, "unit": "epoch" },
  "result": { "...": "stage-specific summary" },
  "artifacts": ["out/hal-9000-medium.onnx"],
  "error": null
}
```

### 1.2 Lifecycle

```
queued ──► running ──► succeeded
               │
               ├─────► failed      (non-zero exit / exception)
               └─────► cancelled   (user request; SIGTERM then SIGKILL)
```

**Rules:**

- **One running job per project.** They contend for the same GPU and the same directories. Queue the rest. Different projects may run concurrently only if the operator opts in (`allow_parallel`), and the UI warns about GPU contention.
- **Jobs are processes, not threads.** A crashed or OOM-killed training run must not take the API down with it.
- **State is written to disk on every transition**, not held in memory. On container restart the API rescans `jobs/`; any job still marked `running` whose PID is gone is marked `failed` with `error: "interrupted"`.
- **Nothing restarts itself.** Interrupted and queued jobs are surfaced for the user to restart deliberately. Auto-restart risks a crash loop that burns hours of GPU time against a job that was never going to succeed — and for training specifically, a resume is a *different* command than the original run (`--ckpt_path` rather than `--model.warmstart_ckpt`), so replaying the original invocation would be wrong anyway. The UI offers a **Resume** action on an interrupted training job, pre-filled from its last checkpoint.
- **Logs stream to a file and to subscribers.** The file is the source of truth; websocket/SSE clients tail it. A browser refresh mid-run loses nothing.
- **Cancellation is graceful.** SIGTERM first — Lightning's `ModelCheckpoint` writes `last.ckpt` at epoch end, so at most one epoch is lost — then SIGKILL after a timeout.

### 1.3 Progress

Parsed from each stage's output, not invented:

| Kind | Signal |
| --- | --- |
| prepare | files processed / total (per sub-stage) |
| transcribe | clips transcribed / total |
| train | `Epoch N:` from Lightning's progress bar → current/total epochs, plus it/s |
| export | single step; indeterminate |

Training also gets **derived fields** the CLI never showed: steps per epoch, elapsed, and projected completion. These matter because *epochs are a misleading unit* — 4000 epochs on 19 clips is ~8,000 steps; on 500 clips it is ~125,000.

### 1.4 Epochs, absolutely never expressed absolutely

The underlying flag `--trainer.max_epochs` is an **absolute ceiling**, and `--ckpt_path` restores the epoch counter. Set the ceiling below the checkpoint's epoch and training exits immediately having done nothing — a trap that cost real time during development.

The UI never exposes the absolute value. It asks for **"train N more epochs"** and submits `completed + N`.

`project.json` records the original `target_epochs` so progress can be shown as a fraction. The default for N is:

```
remaining = target_epochs - completed_epochs
N_default = remaining if remaining > 0 else 1000     # label changes accordingly
```

Two cases, because they mean different things to the user:

| Condition | Label | Default N |
| --- | --- | --- |
| `completed < target` | *Finish this run* — N more to reach the original target | `target - completed` |
| `completed >= target` | *Continue past the original target* | 1000 |

The second case is not an error — deliberately training past the target after listening is normal. But it is the moment to offer **audition** (§2.3) beside the continue button, because "should I continue at all" is the better question and it has a cheap answer.

---

## 2. Preview

A preview is a **job variant**: same lifecycle, same log streaming, but short-lived, non-destructive, and scoped to a sample. It exists because every stage in this pipeline has at least one parameter that can only be judged by ear.

### 2.1 Rules

- Previews write **only** to `work/preview/<stage>/<preview-id>/`. Nothing enters `dataset/` except via a full run with settled parameters — this preserves the consistency rule that every file in a training set received identical treatment.
- Each preview writes a `preview.json` recording its parameters, so the UI can show a sweep side by side and **promote the winner** to a full run with one action rather than retyping.
- Previews are freely discardable; a `prune` action clears them.

### 2.2 Per-stage contract

| Stage | Sample | Returns | Judge by |
| --- | --- | --- | --- |
| `denoise` | 20–30 s excerpt of one source | original + denoised WAV | sibilants, breaths, plosives — metallic or gated means back off |
| `segment` | one source file | clip count, duration histogram, boundary timestamps, first ~5 clips as audio | do clips start/end on word boundaries; is the count plausible for the duration |
| `finalize` | 3–5 clips | resampled audio + measured loudness | level consistency |
| `transcribe` | 5–10 clips, **small model** | text + `lang_prob` + chars/sec | obvious errors before committing to a large-model batch run |
| `train` | ~50 steps | does it start; step rate; **projected wall clock for the full run** | catches a broken warmstart, a bad batch size, or a mis-set espeak voice in under a minute |
| `audition` | N existing checkpoints | one fixed sentence rendered through each | has the voice plateaued (see §2.3) |

The **train preview** is the highest-value one: a 50-step run answers "will this work and how long will it take" before committing hours. Every multi-hour failure during development would have been caught by it.

### 2.3 Audition

Special case of preview aimed at a question with no other cheap answer: *are more epochs helping?*

- Select N checkpoints (default: the last 3 saved).
- Export each to a temporary ONNX.
- Synthesize the **same held-out sentences** through each — text deliberately *not* in the training data, since resemblance to training text is exactly what overfitting flatters.
- Present as an A/B/C player with the epoch/step count on each.

If the newest is not clearly better than its predecessor, the run has plateaued — stop and spend the effort on more audio instead.

---

## 2.5 Ingest

The pipeline begins at `raw/`, and getting files there is a stage in its own right — not a manual `cp` the tool pretends did not happen.

Ingest is a **job kind** with a `source_type` discriminator. All variants land audio in `raw/`, run `ffprobe` on arrival, and report the same summary (files added, total duration, per-file codec/rate/channels). Only the acquisition differs.

```
POST /projects/{id}/ingest   → job
  { source_type: "upload" | "url" | "media-site" | "hf-dataset", ... }
GET  /projects/{id}/sources  → files with ffprobe metadata
DELETE /projects/{id}/sources/{filename}
```

### 2.5.1 Upload

Multipart, single or multiple files. Streamed to disk rather than buffered — source recordings are routinely hundreds of megabytes.

**Filenames are sanitized on arrival.** Spaces, quotes, and `=` propagate through clip names into `metadata.csv` and eventually into shell and container arguments; a checkpoint named `epoch=6339-step=1647790.ckpt` had to be renamed by hand for exactly this reason during development. Normalize to `[A-Za-z0-9._-]`, preserve the extension, and record the original name in the job result so the user can see the mapping.

### 2.5.2 URL — direct media file

Plain HTTP GET with a content-type check. Identical to upload once the bytes land.

### 2.5.3 Media site (yt-dlp)

The most useful ingest path for the actual use case: character-voice source material overwhelmingly lives on video platforms.

```jsonc
{
  "source_type": "media-site",
  "url": "https://...",
  "audio_only": true,          // -x, always: video is dead weight
  "sections": "*00:04:12-00:09:30",   // optional --download-sections
  "playlist": false            // require an explicit opt-in for playlists
}
```

Implementation is a thin wrapper — `yt-dlp -x --audio-format wav -o raw/%(title)s.%(ext)s` plus progress parsing — so it ships in v1. Three things keep it from growing complexity:

- **No format negotiation in the UI.** Always extract audio, always to WAV. The `prepare` stage normalizes to 48 kHz mono anyway.
- **Section download is exposed** (`--download-sections`) because voice datasets usually want a few minutes from a long video, and fetching the whole thing to discard 95% is wasteful.
- **Playlists require an explicit flag**, so a paste of a playlist URL cannot accidentally pull forty videos.

Two caveats to hold:

1. **yt-dlp updates on its own schedule** and breaks when sites change. Pin a version in the image, treat updating it as a maintenance action, and surface extractor errors verbatim rather than wrapping them — the message is usually the fix.
2. **Rights are the user's responsibility.** The tool fetches what it is pointed at. The README should say the obvious thing about training on material you have the rights to use, and the UI should not pretend the question does not exist.

If yt-dlp proves troublesome in practice, it can be dropped to an optional build arg without disturbing anything else — the job contract is the same as `url`.

### 2.5.4 HuggingFace dataset

Structurally different from the others, and worth special handling because **it may arrive with transcripts already**.

```jsonc
{ "source_type": "hf-dataset", "repo_id": "campwill/HAL-9000-Speech", "split": "train" }
```

HF speech datasets commonly ship as parquet shards with embedded audio, or as an audio directory plus a metadata file. The importer:

1. Fetches with `huggingface_hub` (no git-lfs dance, no `.git` duplicating every byte).
2. Extracts audio to `raw/` and, **if the dataset carries transcripts**, writes them to `dataset/metadata.csv` in `id|text` form.
3. Sets `transcripts_provided: true` in `project.json`.

That flag changes the pipeline graph: the UI marks **transcribe as optional** rather than required, and routes the user from prepare straight to validate. Re-running Whisper over transcripts a human already checked is wasted time — but the option stays available, since an independent transcription is also the cheapest way to *audit* provided ones (transcribe to a scratch file, diff, review only the disagreements).

One thing that does **not** change: provided transcripts still go through validation. A third-party dataset is as likely to contain digits, abbreviations, and symbols as a Whisper pass — arguably more so, since it was written for humans to read.

### 2.5.5 Container additions

Two packages, both small:

```dockerfile
RUN python3 -m pip install "yt-dlp==<pinned>" "huggingface_hub>=0.20"
```

`ffmpeg` is already present and is what yt-dlp uses for extraction. Pinning yt-dlp is deliberate (§2.5.3); updating it is a maintenance action with a rebuild, not an automatic pull.

### 2.5.6 Ingest previews

Fetching 30 seconds before pulling a two-hour video is worth the round trip. `preview` with `stage: "ingest"` fetches a short excerpt so the user can confirm the audio is what they expected — right speaker, usable quality, not a music bed — before committing bandwidth and disk.

---

## 3. Checkpoint picker

The base checkpoint is chosen **first**, at project creation, because it constrains three other settings. Picking it first makes a class of mismatch errors impossible by construction rather than catching them at `validate`.

### 3.1 What the choice determines

| Setting | Derived from | Overridable |
| --- | --- | --- |
| tier | last path segment (`.../alan/medium/` → `medium`) | no — it *is* the checkpoint's architecture |
| architecture params | the checkpoint directory's own `config.json` | no  |
| sample rate | tier (16000 for low, 22050 for medium/high) | no — `prepare` must build the dataset to match |
| espeak voice | locale segment + the checkpoint's `config.json` | **yes** — accent is overwritten by fine-tuning; this is a default, not a constraint |
| warmstart vs resume | embedding shape (see §3.4) | advanced only |

This resolves most of the tier-architecture question: rather than trusting a hardcoded table, the real values come from the checkpoint that will actually be loaded. (The table is still needed for training from scratch, which has no checkpoint — rare enough to stay best-effort with a warning.)

### 3.2 Source: the HF catalog

`huggingface.co/datasets/rhasspy/piper-checkpoints`, laid out as `<family>/<locale>/<voice>/<quality>/`. Note this is a **different repo** from `rhasspy/piper-voices`, which ships only inference `.onnx` files.

Three-level picker: **language → voice → quality**, showing only qualities that actually exist. This matters: `en_GB/alan` has *only* `medium`, and discovering that by 404 during a build is a bad first experience. The picker should show the gap, not offer a selection that isn't there.

Each checkpoint directory also carries files worth surfacing:

- `config.json` — the authoritative architecture for that tier
- `MODEL_CARD` — dataset size, license, speaker details; exactly what you want when choosing a starting point. Render it in a detail pane.
- `train.sh` — the command that produced it (old API, but the hyperparameters translate)

### 3.3 Source: this project's own runs

After one training run, `runs-<tier>/lightning_logs/version_N/checkpoints/` contains a checkpoint that, for a *second* run on the same voice, is a better starting point than any stranger's — same speaker, same vocabulary, same architecture.

So the picker has two tabs:

- **Catalog** — the HF repo, with a download indicator
- **This project** — previous runs, labelled by epoch/step and run date

### 3.4 warmstart vs resume, decided for the user

The single most confusing part of the underlying tool, and the picker can simply make the right call by reading the checkpoint's embedding shape and provenance:

| Situation | Flag | Consequence |
| --- | --- | --- |
| catalog checkpoint (different voice) | `--model.warmstart_ckpt` | weights only; epoch count starts at 0 |
| this project's own earlier run | `--ckpt_path` | restores optimizer state **and the epoch counter**, so `max_epochs` must exceed it |

The UI should never present the raw flag names. It presents "start from this voice" vs "continue this run", sets `max_epochs` sensibly for each (for resume: current epoch + N more, not an absolute), and surfaces the vocabulary size as an informational detail:

> 2023-era checkpoint, 130-symbol vocabulary (current default is 256). Warm start bridges this cleanly.

### 3.5 Caching and fetch

Checkpoints are 800 MB+. Download once to `base_checkpoints/`, keyed by the HF path, with a local manifest mapping `en/en_GB/alan/medium/epoch=6339-step=1647790.ckpt` → local filename. Rename on disk to strip the `=` characters, which cause quoting problems in shells, env vars, and container args.

The fetch is slow enough to be a **job** with progress, not a synchronous request.

### 3.6 Endpoints

```
GET  /checkpoints/catalog                     → tree: languages → voices → qualities
GET  /checkpoints/catalog/{path}              → config.json + MODEL_CARD + file list
POST /projects/{id}/checkpoints/fetch         → job; {catalog_path}
GET  /projects/{id}/checkpoints               → local: catalog-fetched + own runs,
                                                 with tier, epoch, step, vocab size
```

**Fetch live, fall back to a bundled snapshot.** On opening the picker, query the HF tree with a short timeout (~3 s) and cache the result for the session. On failure, serve a snapshot bundled into the image and show a persistent notice — *"showing cached catalog from <date>"* — with a retry action.

Live-first is the right default despite the repo changing rarely: the failure mode of a stale snapshot is a user who *knows* a checkpoint was added and cannot see it, which is more frustrating than a slow page load. The bundled snapshot means the picker still works offline and on first run.

### 3.7 Effect on project creation

Project creation becomes: **name → checkpoint → confirm derived settings**. The confirmation step shows what was inferred (tier, sample rate, espeak voice) with the espeak voice editable, and warns if the intended accent differs from the checkpoint's — legitimate, since fine-tuning overwrites it, but worth stating out loud.

`project.json` then records the choice, so every later stage knows the target sample rate without being told again.

This is already implemented in the CLI: `init --espeak-voice` persists the setting and `train`/`export` read it, warning when an explicit value differs from the saved one. It exists because a run that silently defaults to `en-us` on a British voice re-phonemizes the whole dataset against the wrong accent, trains cleanly, and only reveals itself when you listen.

### 3.8 Transcription device

Related, and worth stating because it is asymmetric between the two image variants:

faster-whisper runs on **CTranslate2, which has a CUDA backend and no ROCm backend**. `device="cuda"` therefore works on the CUDA image and does nothing on the ROCm image.

The UI exposes a device toggle that is **enabled only where it can work**; the ROCm build reports *"GPU transcription unavailable on this build (CTranslate2 has no ROCm backend)"* rather than offering a control that silently falls back. `GET /doctor` reports the capability so the frontend does not have to infer it.

Switching engines to whisper.cpp (whose Vulkan backend does run on gfx1151) would close the gap, but it is a separate piece of work and transcription is not the bottleneck — a full run is minutes, not hours. Deferred.

---

## 4. API surface

FastAPI. All paths relative to `/api`.

### 4.1 System

```
GET  /health                    → {ok, version}
GET  /doctor                    → the doctor checks as structured JSON
GET  /espeak-voices?prefix=en   → ["en-us", "en-gb", "en-gb-x-rp", ...]
GET  /tiers                     → tier names, sample rates, architecture params
```

### 4.2 Projects

```
GET    /projects                → [{name, path, clips, minutes, tiers_trained, last_job}]
POST   /projects                → create {name}
GET    /projects/{id}           → full state: counts per directory, dataset stats,
                                   checkpoints, exported voices, recent jobs
DELETE /projects/{id}           → unlink from the UI only; never rm -rf the volume
GET    /projects/{id}/files/{path}  → serve a WAV for playback (scoped to the project)
```

### 4.3 Dataset

```
GET  /projects/{id}/sources                 → raw files with ffprobe metadata
GET  /projects/{id}/dataset                 → rows joined with per-clip duration,
                                              chars/sec, lang_prob  (the audit table)
PATCH /projects/{id}/dataset/rows/{clip_id} → edit one transcript
GET  /projects/{id}/validate?tier=&batch_size=&espeak_voice=
                                            → [Finding] with ids and action
POST /projects/{id}/clean                   → {only, exclude, apply, force}
                                              apply=false returns the plan
POST /projects/{id}/restore                 → {ids}
GET  /projects/{id}/quarantine              → manifest rows + playable audio
```

`validate` is synchronous — it is fast and the UI wants it inline. Everything that touches audio in bulk is a job.

### 4.4 Jobs

```
POST /projects/{id}/jobs        → {kind, stage?, params} → job
GET  /projects/{id}/jobs        → list, newest first
GET  /jobs/{job_id}             → job
GET  /jobs/{job_id}/log         → full log (text)
WS   /jobs/{job_id}/stream      → {type: "log"|"progress"|"state", ...}
POST /jobs/{job_id}/cancel      → graceful stop
```

### 4.5 Previews

```
POST /projects/{id}/preview     → {stage, params, sample?} → job (kind="preview")
GET  /projects/{id}/previews    → grouped by stage, newest first, with params
POST /projects/{id}/previews/{preview_id}/promote
                                → start a full run using this preview's params
DELETE /projects/{id}/previews  → prune scratch
```

### 4.6 Voices

```
GET  /projects/{id}/voices              → exported .onnx + config summary
POST /projects/{id}/export              → job; {tier, checkpoint, voice_name,
                                           espeak_voice, length_scale, ...}
PATCH /projects/{id}/voices/{stem}      → edit inference params in the .onnx.json
POST /projects/{id}/voices/{stem}/say   → synthesize text with the exported voice
GET  /projects/{id}/voices/{stem}/download
```

`PATCH .../voices/{stem}` plus `say` is the inference-tuning loop — `length_scale`, `noise_scale`, `noise_w` change perceived quality substantially and cost nothing, so they deserve a slider and an instant preview rather than a JSON edit and a service restart.

---

## 5. State layout on the volume

```
<project>/
├── project.json              name, espeak_voice, tier, base checkpoint,
│                             target_epochs, transcripts_provided
│                             (see §1.4, §2.5.4, §3.7)
├── raw/                      source recordings (never written to)
├── work/
│   ├── 48k/  denoised/  clips/
│   └── preview/<stage>/<preview-id>/{preview.json, *.wav, result.json}
├── dataset/
│   ├── metadata.csv          id|text
│   ├── audit.csv             duration, chars/sec, lang_prob, text
│   ├── wavs/
│   ├── quarantine/           + manifest.csv
│   └── clean-log.csv
├── base_checkpoints/
├── cache-<tier>/             trainer's preprocessed tensors
├── runs-<tier>/lightning_logs/version_N/checkpoints/
├── out/                      .onnx + .onnx.json + training config
└── jobs/
    └── <job-id>/{job.json, log.txt}
```

No database. `jobs/` is scanned on startup; `job.json` is rewritten on every state transition. This means a job's history survives everything, and an operator can `cat` it.

---

## 6. Screens

Built and shipped **one at a time**, each used on a real dataset before starting the next.

### 6.1 Bones (step 1)

Project list, project detail with directory counts, and `doctor` output. Proves FastAPI → existing functions → job runner → log streaming end to end. Deliberately ugly.

Includes a first pass at **project creation with the checkpoint picker** (§3), because the derived settings it produces are what every later screen depends on. The picker can start as three plain `<select>` elements over a bundled catalog snapshot; the model-card pane and download progress come later.

### 6.2 Prepare (step 2) — *the screen that justifies the UI*

Source file list with ffprobe metadata and a channel picker (downmix vs left/right — a blind downmix can halve SNR when one channel is a bad mic).

Then the **segment tuner**: waveform of one source with detected regions overlaid, sliders for `energy_threshold`, `min_dur`, `max_dur`, `max_silence`, `pad`. Each adjustment fires a `segment` preview; results appear as a sweep the user can compare. A "promote" button runs the full prepare with those parameters.

This is the screen that cannot be designed on paper — expect it to change after two real uses.

### 6.3 Audit (step 3)

The dataset table: play button, duration, chars/sec, `lang_prob`, transcript (editable inline). Sortable, with quick filters for the validation findings. Validation results render as chips that filter the table to the affected rows. `clean` shows its plan as a diff before `--apply`, and quarantine is a reviewable list with restore.

### 6.4 Train (step 4)

Batch size, max epochs, and checkpoint selection — presented as **"start from this voice"** vs **"continue this run"** rather than `warmstart_ckpt` vs `ckpt_path` (§3.4). Tier, sample rate, and espeak voice are already fixed by the project's checkpoint choice and shown read-only with an edit affordance.

Validation runs first and blocks on errors. A **train preview** button reports projected wall clock before committing.

Live: loss curves, epoch/step progress, elapsed and projected, streaming log, cancel.

### 6.5 Voices (step 5)

Checkpoint list with epoch/step, audition A/B/C, export with voice-name enforcement (the `.onnx` stem, the `dataset` field, and the name the client requests must agree), inference sliders with instant `say` preview, and download.

---

## 7. Non-goals for v1

- Multi-user, auth, or remote access. Bind to localhost; the user already has a VPN.
- Multi-speaker models.
- Dataset recording (Piper Recording Studio's job).
- Anything that hides the filesystem. The layout is a feature.

---

## 8. Resolved decisions

Recorded so they are not relitigated mid-build.

| #   | Question | Decision | Reasoning |
| --- | --- | --- | --- |
| 1   | Waveform rendering | **Server-side peaks JSON** | The server already has the audio decoded; peaks are far smaller over the wire than a WAV, and the client stays simple. Cache peaks next to the source file. |
| 2   | Job recovery on restart | **User-initiated restart only** | Auto-restart risks a crash loop burning GPU hours on a job that was never going to succeed. A training resume is also a *different command* than the original run (`--ckpt_path`, not `--model.warmstart_ckpt`), so replaying the invocation would be wrong. Interrupted jobs get a pre-filled **Resume** action. |
| 3   | Whisper on GPU | **Expose a device toggle, enabled only where it works** | CTranslate2 has a CUDA backend and no ROCm backend, so this is asymmetric between images. The ROCm build says so plainly rather than offering a control that silently falls back. `GET /doctor` reports the capability. See §3.8. |
| 4   | Frontend stack | **React** | Decided by the segment tuner (§6.2): peaks rendering, region overlays, drag-to-adjust, and synchronized playback all have mature React components and painful htmx equivalents. Cost is a build step in the image — annoying but bounded. |
| 5   | Tier architecture params | **Read from the selected checkpoint's `config.json`** | Superseded by the checkpoint picker (§3.1). The hardcoded table in `config.py` survives only for training from scratch, where no checkpoint exists to read. |
| 6   | Catalog freshness | **Fetch live, fall back to a bundled snapshot** | The failure mode of a stale snapshot — a user who knows a checkpoint exists and cannot see it — is worse than a slow page load. Offline and first-run still work. See §3.5. |
| 7   | Resume epoch semantics | **"Train N more epochs", submitted as `completed + N`** | The absolute `max_epochs` ceiling combined with a restored epoch counter is the source of the "training exits immediately" trap. `project.json` keeps `target_epochs` for the progress fraction; the default for N is `target - completed`, or 1000 when already past target. See §1.4. |

---

## 9. Implementation order

Each step is shipped and used on a real dataset before the next begins.

1. **Bones** — FastAPI over the existing `piper_trainer.*` functions, the job runner with disk-backed state and log streaming, project list/detail, `doctor`, and project creation with a basic checkpoint picker. Includes **upload ingest** (§2.5.1), since nothing downstream can be tested without files in `raw/`.
2. **Prepare** — remaining ingest types (URL, yt-dlp, HF dataset), source list, channel picker, and the segment tuner with server-side peaks. Expect the tuner to change after two real uses.
3. **Audit** — dataset table with playback and inline transcript editing; validation chips that filter it; `clean` plan-then-apply; quarantine review with restore.
4. **Train** — checkpoint selection framed as *start from this voice* / *continue this run*, blocking validation, train preview with projected wall clock, live loss curves and log.
5. **Voices** — checkpoint list, audition A/B/C, export with name-agreement enforcement, inference sliders with instant `say` preview, download.

Two existing datasets — HAL-9000 (89 clips) and Marvin (692 clips, 47 min) — serve as the test corpus, and the two build variants (CUDA on one machine, ROCm on another) catch platform assumptions early.
