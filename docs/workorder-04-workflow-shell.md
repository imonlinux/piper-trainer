# Workorder 04 — Workflow Shell & Guided Chains

The UI's job is to make the pipeline's shape visible: where a project is,
what the next decision is, and what a button will do before it is pressed.
Today every page is a flat form and the project page is a 631-line
everything-page. This spec replaces that with a stage-driven shell and,
in a second phase, one-button chains that run the pipeline to the next
human decision.

Two phases, each shippable alone:

- **Phase A — the shell.** Six stages, one readiness endpoint, one
  persistent shell per project. No behavior change to any pipeline job.
- **Phase B — guided chains.** "Run to next decision": the server walks
  prepare → transcribe → validate → train → export and stops at four
  human gates.

## Starting point

The prerequisites this spec depends on are already on `feat/ui`
(6edcdf2, 5f1b2db): fetch caps and the yt-dlp watchdog, websocket
reconnect with backoff plus connected-gated polling, the metrics.csv
whole-file re-read, resume-aware train projection, rescan never reaping
its own mid-spawn jobs, and the audit page's terminal-state refresh.
291 tests pass. Nothing in Phase A touches the runner.

## Principles

1. **Hue is for state, not for chrome.** Buttons and links are neutral
   light-on-dark. Color appears only to answer "is this stage fine?"
2. **Numbers over adjectives.** "2,431 clips · 14 h 32 m audio", never
   "dataset ready".
3. **Every stop names its next action.** Including designed stops.
4. **Never recompute server state for a render.** The shell reads what
   jobs already wrote; it does not run validation to color a dot.
5. **One job at a time, honestly.** The JobManager slot is single; the
   UI shows the running job and links to the rest, it does not pretend
   to run three things.

---

# Phase A — the shell

## A1. Design tokens

Dark-first. Amber is the accent because this project builds a HAL-9000
voice — the eye that watches while it works. Vendored via `@fontsource`,
no runtime CDN.

```css
:root {
  --bg:         #101216;  /* page */
  --surface:    #16191e;  /* cards, bars */
  --surface-2:  #1c2027;  /* inputs, hover */
  --border:     #262b33;
  --fg:         #e8eaed;  /* primary text */
  --fg-dim:     #9aa1ab;  /* secondary text */

  --state-run:  #e39b3c;  /* amber: work in flight, pulsing */
  --state-ok:   #4da76b;  /* stage done */
  --state-attn: #d4574e;  /* failed job or findings need you */
  --state-open: #c8cdd4;  /* ready, never run */
  --state-locked:#4a515c; /* prerequisites missing */

  --radius: 10px;
  --radius-pill: 9999px;
  --font-ui:   "IBM Plex Sans", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, monospace;
}
```

Rules:

- `--state-run` is **reserved for in-flight state** (the running dot,
  the active stage stripe, the ETA tick). It never colors a button,
  link, or focus ring. This is the whole trick: when something is
  amber, something is actually running.
- Interaction is monochrome: buttons are `--surface-2` with `--fg`
  text; primary action inverts (light fill, dark text).
- `--state-attn` covers both "job failed" and "findings need review";
  both mean "you must look".
- A light theme via the same tokens is welcome but not gating; the
  theme toggle must exist on **every** route (see A4), not only shell
  routes.

Typography: stage names and copy in Plex Sans; all numbers that move
(epoch counters, durations, ETAs, file sizes) in Plex Mono with
`font-variant-numeric: tabular-nums`.

## A2. Readiness: `GET /api/projects/{name}/stages`

One endpoint the shell polls. Derived entirely from the project
directory and the job records that already exist — no subprocess, no
re-validation.

```jsonc
{
  "project": { "name": "hal", "voice": "hal-medium" },
  "stages": {
    "sources": {
      "status": "done",
      "requirements": [ { "met": true, "text": "3 source files, 18.2 h" } ],
      "active_job": null,
      "last_job": { "id": "…", "kind": "ingest", "state": "succeeded",
                    "finished_at": "…" }
    },
    "prepare":    { "status": "ready",   "requirements": [ … ], … },
    "transcribe": { "status": "locked",  "requirements": [ … ],
                    "blocked_by": "prepare has not produced clips yet" },
    "audit":      { "status": "attn",    "requirements": [ … ],
                    "findings": { "errors": 12, "warnings": 3,
                                  "stale": false } },
    "train":      { "status": "done",    "requirements": [ … ] },
    "voices":     { "status": "ready",   "requirements": [ … ] }
  },
  "next": { "stage": "audit", "why": "validation found 12 errors" },
  "running": { "job": { "id": "…", "kind": "prepare",
                        "progress": { "unit": "clip", "current": 412,
                                      "total": 2431 } },
               "more": 0 }
}
```

### Stage order and status

Pipeline order: `sources → prepare → transcribe → audit → train → voices`.

Statuses, precedence `active > attn > done > ready > locked`:

| status | meaning | color |
|---|---|---|
| `active` | a job of this stage is running now | amber, pulsing |
| `attn` | last job failed, or audit findings await | red |
| `done` | last job succeeded and nothing supersedes it | green |
| `ready` | prerequisites met, runnable | light neutral |
| `locked` | prerequisites missing | dim neutral |

`requirements` is a list of `{met, text}` checks in display order;
`text` names numbers ("2431 clips in dataset/wavs", "transcribe has not
run since the last prepare"). `blocked_by` is **derived**: the first
unmet requirement's text, or null. It is never hardcoded to a stage
name — if prepare is fine but transcribe never ran, transcribe's
blocker is "not run yet", not "prepare".

### Job kind → stage mapping

| kinds | stage |
|---|---|
| `ingest` | sources |
| `prepare` | prepare |
| `transcribe` | transcribe |
| `validate`, `clean`, `clean-apply` | audit |
| `train` | train |
| `preview` | by `params.stage` (`prepare` / `audit` / `train`) |
| `export`, `audition` | voices |

`active_job` is the **newest** running job mapped to the stage (parallel
jobs are possible; the singular field shows the freshest one and
`running.more` counts the rest).

### The audit read set (important)

The audit stage's `findings` come from `result` of the **newest of**:
the last `validate` job, the last `clean` job, the last `train` job.
These are the jobs that record `errors`/`warnings` counts. It is never
recomputed at request time.

When the newest of those is a `clean-apply` (which returns
moved/dropped/repaired **stats**, not error counts), the findings are
real but out of date:

```jsonc
"findings": { "errors": 12, "warnings": 3, "stale": true,
              "stale_reason": "clean applied after this validation" }
```

The UI then shows the counts dimmed with the line
"clean applied — run validation to refresh these counts". Honest and
cheap; the number just says when it was measured.

### `next` selection

1. Any stage `active` → that stage, why "running now".
2. Else the first stage (pipeline order) with `attn` → fix it.
3. Else the first stage with `ready` → run it.
4. Else `voices`, why "audition your voice".

## A3. Shell components

### `useStages` hook

`GET /stages` every 3 s via `usePoll`, paused while
`document.hidden`. Exposes `{ stages, next, running, loading }`. One
poll per shell, shared with pages through context — pages stop
fetching overlapping data.

### `ProjectShell`

Wraps all `#/project/{name}/…` routes. Renders IdentityBar, Stepper,
JobBar (only while something runs), then the routed page. Owns the
stages context and the theme class on `<html>`.

### `IdentityBar` (46 px)

One row: `← projects` · project name · output voice chip (`hal-medium`,
the `voice_stem()` result — what you get, not the base checkpoint) ·
status line · theme toggle · overflow (doctor, settings).

The status line is the **idle JobBar**: when nothing runs it shows the
one-line truth, e.g. `3/6 stages done · next: audit — validation found
12 errors`. When a job runs, this row yields to the JobBar (below) and
shows `running: prepare · clip 412/2431`.

### `Stepper` (68 px)

Six pills in pipeline order. Each pill: stage name, a status dot, and a
2 px top stripe in the status color — the dot answers "what is this",
the stripe reads at a glance as a progress bar across the six.

**Every pill is always clickable, locked included.** A locked stage
navigates normally and the destination page explains itself with a
dismissible banner (see Gate, below). There is no read-only,
disabled-looking mode anywhere in the shell: a locked page says "here
is what this stage does, here is exactly what's missing, here is where
to go fix it". Locks teach; they don't gate-click.

### `JobBar` (42 px, conditional)

**Exists only while ≥ 1 job is running.** When the last job finishes it
collapses and the IdentityBar status line resumes. Total chrome at rest:
88 px.

Contents: newest running job — kind in user vocabulary ("preparing
clips"), progress `clip 412/2431` (unit first), ETA, cancel button,
`+N more` when parallel jobs exist (links to Activity).

ETA rules: shown only after ≥ 2 progress samples spanning ≥ 2 % of
total, as `~(total − current)/rate`; before that it reads
`estimating…`. Never an absolute wall-clock destination.

### `Gate` and the dismissible banner

Two renderings of the same requirement list:

- **Decision point** (requirements met): a full panel that names the
  numbers and offers the continue action. Used by chains (Phase B) and
  by "you are about to spend hours" moments (train preview).
- **Dismissible banner** (requirements unmet): one line at the top of
  the destination page — `training needs a validated dataset:
  validation hasn't run since transcribe` — expandable to the full
  requirement list, dismissible for the session. The rest of the page
  renders normally underneath.

### `Panel`, `Consequence`, `Help`

- **Panel**: titled bordered section, the only card primitive.
- **Consequence**: one sentence above a destructive or expensive
  action, naming what happens to data: "retranscribe overwrites the
  transcript column for all 2,431 clips".
- **Help**: a `?` affordance per panel with a two-sentence explainer.
  Pipeline jargon gets a plain-language line, once, where it's used.

## A4. Routing

Existing hash routes keep working (bookmarks survive). Additions follow
the same shape:

| route | page | change |
|---|---|---|
| `#/projects` | list | keep; gets theme toggle in a thin header |
| `#/new` | NewProject | keep; live stem preview (A5) |
| `#/doctor` | Doctor | keep |
| `#/project/{name}` | overview | rewritten (A5) |
| `#/project/{name}/activity` | **new** — job history (A5) | |
| `#/sources/{name}` | **new** — carved out of overview | |
| `#/prepare/{name}` | Prepare | keep, add gate/banner |
| `#/transcribe/{name}` | **new** — carved out of overview | |
| `#/audit/{name}` | Audit | keep, add findings header |
| `#/train/{name}` | Train | keep, add preview gate |
| `#/voices/{name}` | Voices | keep |

The old global header (`piper-trainer · react · Projects · New ·
Doctor · vN`) stays on the four non-shell routes and is replaced by the
shell on project routes; the IdentityBar carries the brand and the
`← projects` link, and the overflow menu holds Doctor and theme. The
version chip moves to the overflow. Health text drops (the shell IS the
health indicator; a dead API shows as `stale` in the status line).

The **Activity view is specified and shipped in the same change that
deletes the overview's raw job table** — history is where cancelled
runs, failed fetches, and audition logs go to be found. No deletion
before the replacement exists.

## A5. Per-page changes

### Overview (`#/project/{name}`) — rewritten, target ≤ 200 lines

1. **Next card** — the one thing to do now: `next.stage`, `next.why`,
   one primary button (which becomes "run to next decision" in Phase
   B). Below it, the six-stage mini-map (same data as the Stepper).
2. **Settings panel** — two groups, finally separated:
   - *Output*: voice name (editable pre-export), output location,
     exported voices count.
   - *Started from*: base checkpoint id, warmstart source (read-only
     facts; they describe where the voice's accent comes from).
3. **Recent activity** — last 5 jobs (kind in user vocabulary, state,
   when, duration) + link to Activity.

Everything else moves out (sources → Sources, transcribe → Transcribe)
or away (job table → Activity).

### Sources (`#/sources/{name}`) — new page

Current ingest + source list from the overview, unchanged in behavior:
fetch form (URL, optional section), upload, per-source size/duration,
waveform peaks view, delete with a Consequence line ("removes the
source file; prepared clips already made from it stay"). Fetch caps and
the watchdog are server-side already; the page just reports their
errors verbatim.

### Prepare (`#/prepare/{name}`)

Existing flow plus: readiness banner from stages data; the preview
promote flow stays; a Consequence line on "re-prepare" ("re-segments
sources; existing clips and transcripts are kept for unchanged
sources"). In Phase B this page hosts gate 1.

### Transcribe (`#/transcribe/{name}`) — new page

- Device recording, gated by doctor's `transcribe_devices`: with
  devices, a record panel; without, a one-line explainer and a link to
  `#/doctor` ("no transcription devices found — doctor shows what's
  installed"). The gate is informational, not a lock.
- **Normalize toggle**, default on, labeled "normalize text (Mr. →
  Mister, digits → words)" — parity with the runner's
  `normalize`/`--no-normalize`, passed as `params.normalize`.
- Batch transcription with the retranscribe Consequence line ("overwrites
  the transcript column for all 2,431 clips").
- Last-transcribe summary from the job result: clips, characters,
  duration.

### Audit (`#/audit/{name}`)

Findings header first: counts (dimmed + stale line when
`findings.stale`), then the existing table with apply/restore. Apply
and restore keep the existing refresh-on-terminal behavior. The header's
stale line is the fix-state affordance: "run validation to refresh".

### Train (`#/train/{name}`)

All existing controls stay. Additions: the preview card shows the
resume-aware projection (already shipped: `basis` line included); the
fetch row (already shipped) — warmstart mode carries a catalog-path
input prefilled from the project's `catalog_path` and a fetch button,
so the picker's "fetched (catalog)" group can never again be empty with
no way to fill it; a Consequence line on warmstart vs scratch
("warmstart starts from the base voice's weights — from scratch
ignores them"). In Phase B the preview is gate 3.

### Voices (`#/voices/{name}`)

Unchanged list + say/download; export gets a Consequence line if it
overwrites an existing stem.

### NewProject (`#/new`)

Live stem preview under the form: as name and tier are typed, show
`hal-medium` immediately, including what the sanitizer will do to bad
characters (`my voice!` → `my_voice_-medium`). No guessing at export
time.

### Activity (`#/project/{name}/activity`)

The full job history, the table that used to live on the overview:
kind (user vocabulary), state, started/finished, duration, error text
when failed, log link. Filter row: all / running / failed. This is the
only place raw job vocabulary (`validate`, `clean-apply`, ids) is
allowed to surface; everywhere else speaks stage.

## A6. Phase A acceptance

- `GET /stages` returns the contract for a fresh project, a mid-pipeline
  project, and a project with a failed job (all three statuses exercised).
- Audit `stale` flips true exactly when a clean-apply postdates the last
  validate, and the overview's next card sends the user to audit.
- Every stage pill navigates, locked included; locked pages show the
  dismissible banner with the derived `blocked_by` text.
- JobBar appears on job start, collapses on finish; at rest the chrome
  is IdentityBar + Stepper only.
- The Activity view exists and holds the history before the overview's
  job table is removed.
- Transcribe device gate reflects `transcribe_devices`; the normalize
  toggle reaches the runner as `params.normalize`.
- NewProject stem preview matches `voice_stem()` for tricky names.
- Theme toggle reachable on `#/projects`, `#/new`, `#/doctor`.
- `tsc --noEmit && vite build` clean; pytest suite green.

---

# Phase B — guided chains

Phase A shows where you are. Phase B walks the pipeline for you and
stops where judgment is needed. The four gates are exactly the four
places the operator should look before the pipeline spends time or
mutates data:

1. **Segment check** — after prepare, before transcription (listen/scan
   clips; bad segmentation poisons everything downstream).
2. **Validate findings** — after validation, before any cleaning (you
   decide what's an error).
3. **Train preview** — before a long run (confirm epochs, batch,
   projection; warmstart vs scratch).
4. **Audition** — after export (the held-out sentence through the new
   checkpoints; you judge the voice).

Transcription and export themselves run unattended inside the chain.

## B1. Chain model

One chain per project at a time. State lives in the project directory:

```
projects/hal/chains/{chain_id}.json
{
  "id": "c-20260905-143210",
  "stages": ["prepare", "transcribe", "audit", "train", "voices"],
  "current": "train",
  "state": "running",       // running | waiting_gate | interrupted
                            // | done | cancelled | failed
  "gate": null,             // gate id when waiting_gate
  "created": "…", "updated": "…",
  "jobs": { "prepare": "…job id…", "transcribe": "…" }
}
```

Each job the chain submits records `chain_id` and `chain_stage` in its
state file — a job knows its chain without reverse lookups.

## B2. Controller

Submission and continuation live in `JobManager._supervise`'s `finally`
(the one place every job already passes through on the way to terminal
state): if the finished job carries a `chain_id`, load the chain, and:

- job succeeded and the next stage has a gate → chain `waiting_gate`,
  gate id set. UI shows the Gate panel.
- job succeeded and no gate → submit the next stage's job immediately
  (slot was held by this very chain, so it is free).
- job succeeded and stages exhausted → chain `done`.
- job failed/cancelled → chain `failed`/`cancelled`; the stage page
  shows the error; "resume chain" resubmits from `current`.

Chains respect the single slot: a chain is only submitted when
`_slot_free`, and it owns the slot across its stages. A manual job
submitted mid-chain is rejected server-side with "a guided run is in
progress — cancel it or wait" (the chain's current job can be cancelled
from the JobBar like any job).

### Crash reconciliation

The periodic rescan (10 s tick) gains a chain check: any chain in
`running`/`waiting_gate` whose current stage has **no live job record**
(server died mid-stage) is marked `interrupted`. The next `GET /stages`
carries it; the next card reads "guided run interrupted at train —
resume" and resume resubmits from `current`. Gates survive restarts
because they are just chain state: a `waiting_gate` chain with its
stage job still succeeded is not interrupted — it resumes at the gate.

## B3. Endpoints (per-project)

```
POST /api/projects/{name}/chains                 { "from": "prepare" }  → chain
GET  /api/projects/{name}/chains/{chain_id}      → chain
POST /api/projects/{name}/chains/{chain_id}/resume   # continue past a gate,
                                                 # or resubmit an interrupted stage
POST /api/projects/{name}/chains/{chain_id}/cancel   # cancel current job + chain
```

All under the project path: chains are project state, and URLs stay
greppable per project. `GET /stages` includes the `chain` block so the
shell renders chain state without a second poll.

## B4. The four gates in the UI

Each gate renders as a decision-point Gate panel on the relevant stage
page (and the overview's next card points there):

1. **Segment check** (prepare page): "847 clips · 11 h 03 m · median
   12.4 s". Actions: "listen to samples" (deep-links the existing
   preview UI), "continue", "stop here". Skippable but never silent —
   continuing logs `gate=segment-check accepted` into the chain record.
2. **Validate findings** (audit page): the findings header, plus the
   stale warning if a clean was applied mid-chain. Actions: continue /
   apply suggested clean (then re-validate inside the chain before
   continuing) / stop.
3. **Train preview** (train page): epochs, batch size, the resume-aware
   projection with its `basis` line, warmstart-vs-scratch choice.
   Actions: "start training" / adjust / stop.
4. **Audition** (voices page): the held-out sentence through the new
   checkpoints' audio elements. Actions: "keep this voice" (chain done)
   / "train more epochs" (chains back to train via resume-from) / stop.

## B5. Copy for chains

A chain is a colleague running errands, not a wizard. Progress lines:

- `guided run: preparing clips → clip 412/2431` (JobBar)
- `guided run paused: check your segments before transcribing` (idle
  status line, gate state)
- `guided run interrupted at train — resume` (after a server restart)

Designed stops never say "failed". "Failed" appears only when a job
actually errored, and then the error text is shown verbatim.

## B6. Phase B acceptance

- End-to-end chain on a tiny project passes all four gates to `done`.
- Gate stop → resume continues at the right stage; cancel mid-chain
  cancels the current job and marks the chain cancelled.
- `kill -9` the server mid-stage → on restart, rescan marks the chain
  `interrupted` within one tick, resume resubmits from `current`.
- A manual job submission during a chain is rejected with the
  chain-aware message.
- `GET /stages` chain block drives the next card with no extra polling.
- Full pytest suite green, including the chain reconciliation tests.

---

# Appendix — copy standards

1. **Name the number.** "2,431 clips · 14 h 32 m audio", "12 errors ·
   3 warnings", "epoch 12/40". Adjectives only when there is no number.
2. **Unit first, then fraction.** "clip 412/2431", "epoch 12/40" — the
   unit tells you what is being counted before the numbers land.
3. **Say what to do next.** Every stop names its action: "run
   validation to refresh these counts", "no transcription devices —
   doctor shows what's installed".
4. **Never "failed" for designed stops.** Gates are "waiting for you";
   interruptions are "interrupted — resume". "Failed" + the verbatim
   error is reserved for real errors.
5. **Buttons in user vocabulary.** "Check clips", "Start training",
   "Keep this voice" — stage names live in the Stepper, job kinds live
   in Activity.
6. **Consequence before the irreversible.** Any action that overwrites,
   deletes, or commits hours gets its one-sentence cost above the
   button, with the real number in it.
