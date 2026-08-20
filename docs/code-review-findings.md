# piper-trainer — Code Review Findings

**Date:** 2026-08-19
**Scope:** `src/piper_trainer/*.py` (cli, config, prepare, transcribe, validate, clean, train, export, doctor)
**Status:** findings only — no code changes made.

---

## 1. Critical (happy path broken by default)

### 1.1 `transcribe` writes CRLF into `metadata.csv`

Both `csv.writer` calls in `transcribe.py:49-55` use the default
`lineterminator='\r\n'`. Verified against the stdlib: the writer emits
`clip1|hello world\r\n`.

Consequence: `validate.py:88-91` flags CRLF as an **error**, and `train`
(`cli.py:251-254`) refuses to start when any error-level finding exists.
Every dataset produced by `piper-trainer transcribe` therefore fails its own
validation out of the box.

**Fix:** pass `lineterminator="\n"` to both writers.

### 1.2 `clean --apply` never implements the file-level repairs it advertises

`build_plan` collects `crlf` / `columns` / `blank-row` into
`plan.normalize_file` (`clean.py:114-115`), but `apply()` (`clean.py:166`)
never touches that field. `describe()` prints "normalize metadata.csv
(crlf)", `--apply` changes nothing, and validate still fails — the user is
told to re-run with `--apply` for an action that is a no-op.

(Fixing 1.1 makes the crlf case moot, but the unimplemented action is still a
gap for `columns` and `blank-row`.)

### 1.3 Escape/unescape asymmetry on the `|` delimiter

Writers escape the delimiter (`QUOTE_NONE` + `escapechar="\\"`), but readers
parse it back with a bare `split("|")` and never unescape:

- `clean.read_rows` (`clean.py:84`)
- `validate.py:98`

Any transcript containing a literal `|` (or a trailing backslash) corrupts
the clip id and text on the next clean/validate pass — e.g. `a | b` splits
into id `a \` and text ` b`.

**Fix:** consolidate metadata read/write into one shared helper that handles
escaping and unescaping in one place (used by transcribe, validate, and
clean), or switch to a parser that understands the writer's escaping.

---

## 2. Robustness

### 2.1 `prepare` is not idempotent

`run_all` (`prepare.py:141`) re-runs every stage from scratch on each
invocation — a change to `--energy-threshold` triggers full re-conversion and
re-denoising, though only segmentation was affected. Re-segmenting also
leaves stale clips in `work/clips/` and `dataset/wavs/`, which then surface
as `orphan-wav` findings and accumulate over runs. No invalidation and no
destination-directory cleanup between runs.

### 2.2 Filename collision in `to_48k`

`foo.wav` and `foo.mp3` in `raw/` both map to `work/48k/foo.wav`
(`prepare.py:47`); the later file silently overwrites the earlier one.

### 2.3 `transcribe` has no resumability

Every clip is re-transcribed on every run. A crash, or re-reviewing a large
set with the big Whisper model, starts over from zero. Skipping clips already
present in `metadata.csv` (and re-transcribing only new ones) would be a
cheap, high-value change.

### 2.4 `train --resume auto` can silently no-op

If the checkpoint's epoch is ≥ `max_epochs`, Lightning exits immediately
having done nothing — a known trap (see the UI design doc, §1.4).
`validate_checkpoint` warns about it, but only via the `validate`
subcommand; the auto-resume path in `train` (`cli.py:256-259`) neither
checks the checkpoint's epoch against `max_epochs` nor warns.

### 2.5 Redundant full WAV pass in validate

Every file in `dataset/wavs/` is opened once in the audio-properties loop
(`validate.py:129-144`) and a second time in the cps-outlier loop
(`validate.py:189-196`). `per_id_dur` is already collected in the first
pass; the second open is unnecessary.

### 2.6 Naming inconsistency between train and export

Train's voice name is `{lang_code}-{name}-{tier}` (e.g.
`en-marvin-medium`, `train.py:30`, note `lang_code = "en"`, not `en_us`),
while export's default stem is `{name}-{tier}` (`export.py:29`). This does
not break the documented contract (onnx stem == `dataset` field == requested
name), but the generated config's `data.voice_name` and `dataset` fields end
up as different strings after a default export. A shared naming helper
would remove the foot-gun.

### 2.7 `doctor` fails on the supported CPU variant

"GPU available" is a hard `mark()` failure (`doctor.py:52`), so `doctor`
returns exit code 1 for the supported `cpu` image even when everything
relevant works.

### 2.8 Smaller items

- **Espeak-voice check swallows all exceptions** (`validate.py:230-231`):
  if `espeak-ng` is missing the check silently doesn't run at all.
- **Zero-clip validation split:** `validation_split=0.02`
  (`train.py:20,45`) yields 0–1 validation clips on a ~20-clip dataset;
  validate has no check for this.
- **`denoise` return count globs the output dir** (`prepare.py:77`), so
  stale files from a previous run inflate the reported count.

---

## 3. Code quality / process

### 3.1 No tests, no linter configuration

`pyproject.toml` is build-config only. A small unit-test suite targeting the
pure logic would have caught §1.1–1.3 directly and guards the highest-risk
functions:

- `repair_text` (abbreviations, symbols, single-digit expansion)
- metadata read/write round-trip (the escaping issue, §1.3)
- `build_command` argument construction per tier/resume/warmstart
- `latest_checkpoint` (multi-version `lightning_logs/`)
- `language_block` fallback ladder
- `export.verify` (stem/dataset equality, required fields)

### 3.2 Dead code

- `prepare.probe()` is defined (`prepare.py:23`) but never called.
- `validate.py:4` imports `csv` but never uses it.

### 3.3 `clean` unresolved-reporting granularity

If a multi-id `unspoken-text` finding is only *partly* auto-repairable, the
*entire* finding (all ids) is appended to `plan.unresolved`
(`clean.py:125-126`). The "needs a human" list should name only the specific
clip ids that could not be repaired.

### 3.4 `restore` can re-add rows directly

`restore` (`clean.py:223`) moves WAVs back but requires re-running full
transcribe or hand-editing `metadata.csv` to recover the rows. The quarantine
manifest (`quarantine/manifest.csv`) already records every quarantined row's
text and reasons — row restoration from the manifest is straightforward and
would also work for clips quarantined long ago.

### 3.5 No single-writer guard on project directories

Two concurrent container/CLI runs against the same project volume race on
the same paths (`dataset/wavs/`, `metadata.csv`, checkpoints). The UI design
doc already mandates one running job per project (§1.2); a cheap lockfile
check on project root would protect the CLI path in the meantime.

### 3.6 Undeclared dependencies in `pyproject.toml`

The CLI package declares zero runtime dependencies. This is consistent with
the container-only strategy (the Dockerfile installs everything), but the
constraint should be stated explicitly (docstring or README) so nobody tries
to `pip install` the package standalone and gets a silent import failure.

### 3.7 Forward-looking (UI / job layer, per `docs/piper-trainer-ui-design.md`)

No per-stage progress counters and no log files exist yet. The design doc
makes the log file the source of truth for jobs (§1.2) and parses progress
from stage output (§1.3). Adding that plumbing to the Python layer now —
rather than retrofitting when the FastAPI layer arrives — keeps the
"API and CLI call the same functions" principle (§0.1) clean.

---

## 4. Recommended sequencing

1. **§1.1** — one-line `lineterminator="\n"` fix; unblocks the main flow.
2. **§1.3 + §3.1** — shared metadata helper + minimal test suite; guards the
   format contract that three modules currently each re-implement ad hoc.
3. **§1.2** — implement (or remove) the `normalize_file` actions so `clean`
   never advertises a no-op.
4. **§2.1–2.4** — idempotent `prepare`, collision-safe naming, resumable
   `transcribe`, resume-epoch warning.
5. Rest of §2/§3 as convenient.
