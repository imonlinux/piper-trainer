# Work order 02 — correctness and idempotency

**Prerequisite:** work order 01 merged (`ddec4c0`, `74c4aa0`).
**Theme:** make the CLI functions trustworthy before the API layer wraps them. Everything here is a correctness or idempotency fix, not a feature.

Findings addressed: WO1 review carry-overs, §2.1–2.8, §3.2, §3.3.
Explicitly **deferred** to WO3 (do not implement): §3.4 restore-from-manifest, §3.5 project lockfile, §3.6 dependency declaration.

Tasks are ordered by consequence. Task 2 is the one that silently corrupts training data.

---

## Task 1 — carry-overs from the WO1 review

Both in `clean.apply()`.

### 1a. `normalized` reports 0 when normalization happened

Observed in your own acceptance run:

```
· normalize metadata.csv (crlf)
repaired: 0, quarantined: 0, rows_removed: 0, rows_remaining: 3, normalized: 0
```

The plan says it normalized CRLF; the stat says nothing was normalized. `normalized` counts dropped rows only, so file-level fixes never increment it.

Rename the counter to `malformed_rows_dropped` and add a separate `line_endings_fixed: bool` (true when `line_endings()` was not `"lf"` before the rewrite). Two facts, two fields, neither lying.

### 1b. `--only` does not constrain side effects, and the guard cannot see them

`apply()` always round-trips through `metadata.read`/`write`, so `clean --only cps-outliers --apply` *also* drops every malformed row — work the user did not ask for — and `MAX_FRACTION` never sees those drops.

- Gate the malformed-row drop on `"columns" in plan.normalize_file`. When it is not gated in, malformed rows are **preserved verbatim** in the rewrite.
- Include `len(col_problems)` in the fraction the `MAX_FRACTION` guard evaluates, when they are being dropped.
- Line-ending normalization stays unconditional — that is the point of a canonical writer — but must be reported per 1a.

Preserving a malformed row through a csv round trip needs care: `metadata.read` excludes it from `rows`, so `apply` has no copy. Have `read` retain the raw line on the `Problem` (add a `raw: str` field) so `apply` can write it back untouched.

### 1c. `apply()` assumes the file exists

It calls `metadata.read()` with no existence check; today only `build_plan`'s guard makes that safe. Add an explicit check with a clear error.

---

## Task 2 — prepare is not idempotent (§2.1, §2.2, §2.8c)

**This is the one that silently changes what gets trained on.**

Re-running `prepare` with a different `--energy-threshold` produces differently-named clips (the names embed timestamps), so the old ones remain. `finalize` then globs `clips/*.wav` and copies **both sets** into `dataset/wavs/`. The dataset silently contains output from two parameter sets, and nothing reports it.

### 2a. Each stage owns and clears its output directory

`to_48k`, `denoise`, `segment`, `finalize` each remove their destination directory's contents before writing. No `--keep` escape hatch; a stage's output is a pure function of its input and parameters.

### 2b. Stage manifests

Each stage writes `<output_dir>/.stage.json`:

```jsonc
{
  "stage": "segment",
  "params": { "energy_threshold": 45, "min_dur": 1.5, ... },
  "input_fingerprint": "sha256 of sorted (name, size, mtime_ns) of inputs",
  "outputs": 137,
  "completed_at": "2026-08-20T14:02:11Z"
}
```

Two uses: the UI needs to show what produced the current dataset, and it makes 2c possible.

### 2c. Skip when nothing changed

If the manifest exists and both `params` and `input_fingerprint` match, skip the stage and report `"skipped"`. `--force` overrides. This is what makes re-running the pipeline cheap enough to be routine.

### 2d. Destination collisions in `to_48k` (§2.2)

`foo.wav` and `foo.mp3` both map to `48k/foo.wav`; the second silently wins.

Compute all destinations first. For any stem claimed by more than one source, use `{stem}_{ext}.wav` for **every** member of that group — deterministic given the input set, rather than order-dependent. Report the mapping in the result dict.

### 2e. Sanitize filenames on the way in

Source names flow through clip names into `metadata.csv` and eventually into shell and container arguments. Normalize the destination stem to `[A-Za-z0-9._-]`, collapsing runs of other characters to `_`. Apply after 2d so collision handling sees the final names. Record the original→sanitized mapping in the result.

### 2f. `denoise` count globs the output directory (§2.8c)

`len(list(project.denoised.glob("*.wav")))` counts stale files. With 2a the directory is cleared first, so this is nearly fixed — but count what this invocation actually wrote rather than what happens to be on disk.

---

## Task 3 — `--resume` can silently no-op (§2.4)

`--trainer.max_epochs` is an absolute ceiling and `--ckpt_path` restores the epoch counter, so resuming a checkpoint at epoch 9999 with `--max-epochs 4000` exits immediately having trained nothing. `validate_checkpoint` knows this; the `train` path never asks.

### 3a. Guard

When `--resume` resolves to a checkpoint, read its epoch. If `epoch >= max_epochs`, **refuse to start** with a message naming both numbers and the value that would work.

### 3b. `--add-epochs N`

Add as an alternative to `--max-epochs`, mutually exclusive with it:

```
--add-epochs 1000    →  max_epochs = checkpoint_epoch + 1000
```

Without `--resume`, `--add-epochs` is an error (nothing to add to).

This implements `docs/piper-trainer-ui-design.md` §1.4 at the CLI level, so the API layer can call it directly rather than reimplementing the arithmetic.

### 3c. Record the target

Store `target_epochs` in `project.json` on the first run of a tier (the design doc uses it for the progress fraction). Do not overwrite it on resume.

---

## Task 4 — `transcribe` has no resumability (§2.3)

Every clip is re-transcribed on every run. On a 692-clip dataset with `large-v3` that is a long, wholly avoidable wait after any interruption.

- Read the existing `metadata.csv` if present; skip clips already having a row.
- `--retranscribe` forces a full pass.
- `--only-missing` (default behaviour, stated explicitly for the API) transcribes only clips absent from the metadata.
- Report `transcribed` and `skipped` counts separately.
- `audit.csv` must stay consistent with `metadata.csv` — carry forward existing audit rows for skipped clips rather than dropping them.

---

## Task 5 — shared voice-name helper (§2.6)

`train.build_command` derives `{lang_code}-{name}-{tier}` (e.g. `en-marvin-medium`) while `export` defaults to `{name}-{tier}` (`marvin-medium`). Not a contract break — export sets `dataset` from its own stem — but the training config's `data.voice_name` and the deployed voice name differ, which will confuse anyone reading logs later.

Add one helper in `config.py`, use it in both places:

```python
def voice_stem(name: str, tier: str, espeak_voice: str | None = None) -> str
```

Pick **one** convention and apply it to both. I suggest `{name}-{tier}` (no language prefix): it matches the deployed filename, and the language is already recorded in the config's `language` block. State which you chose.

---

## Task 6 — `clean` unresolved reporting granularity (§3.3)

When an `unspoken-text` finding covers several ids and only some are auto-repairable, the **entire** `Finding` is appended to `plan.unresolved` — once per unrepairable id, so a finding with three bad ids prints three times.

Collect the unrepairable ids, then append **one** new `Finding` carrying only those ids, so `describe()` names exactly the clips needing a human.

---

## Task 7 — small correctness items

### 7a. Redundant WAV pass in validate (§2.5)

Every file in `dataset/wavs/` is opened twice: once in the audio-properties loop, again in the chars-per-second loop. `per_id_dur` is already collected in the first pass. Reuse it.

### 7b. `doctor` fails on the supported CPU variant (§2.7)

"GPU available" is a hard failure, so `doctor` exits 1 on the `cpu` image even when everything relevant works.

Make it informational when the torch build has neither backend (`torch.version.cuda is None and torch.version.hip is None`) — that is a CPU build and the absence of a GPU is expected, not a fault. Keep it a failure when a GPU-capable build cannot see a device, which is the case worth catching.

### 7c. espeak-voice check swallows every exception (§2.8a)

`except Exception: pass` means a missing `espeak-ng` silently skips validation entirely. Catch `FileNotFoundError` and `subprocess.CalledProcessError` specifically, and emit a `Finding("error", "espeak-missing", ...)` rather than staying quiet.

### 7d. Validation split can yield zero clips (§2.8b)

`validation_split=0.02` on a 20-clip dataset gives 0 validation clips. Add a validate check: warn when `round(n_clips * validation_split) < 1`, suggesting `--validation-split 0`.

### 7e. `prepare.probe()` is dead code (§3.2)

Do not delete it — wire it up. Add a `sources` subcommand listing files in `raw/` with codec, sample rate, channels, duration, and size. This is `GET /projects/{id}/sources` from the design doc (§2.5), so building it now means the API layer wraps an existing function instead of growing one.

---

## Task 8 — tests

Extend the existing suite. Same constraints: pure logic, no network, no GPU, no real audio processing. Mock `subprocess` for anything invoking ffmpeg or deep-filter.

- **metadata**: `Problem.raw` round-trips a malformed line unchanged (Task 1b)
- **clean**: `malformed_rows_dropped` and `line_endings_fixed` report accurately in all four combinations; `--only` that excludes `columns` preserves malformed rows verbatim; the `MAX_FRACTION` guard counts malformed drops when they are gated in; `unresolved` contains one finding with only the unrepairable ids (Task 6)
- **prepare**: destination collision produces `{stem}_{ext}.wav` for every group member; sanitization maps expected characters; the stage manifest round-trips; a matching fingerprint skips and `--force` does not; a changed parameter does not skip
- **train**: `--add-epochs` computes `checkpoint_epoch + N`; `--add-epochs` without `--resume` errors; the guard refuses when `epoch >= max_epochs`; `--add-epochs` and `--max-epochs` together error
- **transcribe**: existing rows are skipped; `--retranscribe` does not skip; audit rows for skipped clips are carried forward (mock the Whisper model)
- **validate**: zero-validation-clip warning fires at the boundary; `espeak-missing` finding when the binary is absent
- **config**: `voice_stem` is identical for the train and export call sites

Mutation-check the additions the way you did in WO1 §6 — reintroduce each bug, confirm exactly one test fails.

---

## Acceptance

```bash
pytest -q                                    # all pass
grep -rn "except Exception: *pass" src/      # no hits
```

Behavioural, on a scratch project:

```bash
# idempotency
piper-trainer prepare <proj> --energy-threshold 55   # note clip count
piper-trainer prepare <proj> --energy-threshold 45   # count reflects 45 ONLY
piper-trainer prepare <proj> --energy-threshold 45   # reports "skipped"
piper-trainer prepare <proj> --energy-threshold 45 --force   # re-runs

# resume guard
piper-trainer train <proj> --resume auto --max-epochs 10     # refuses, names the numbers
piper-trainer train <proj> --resume auto --add-epochs 100 --dry-run
```

---

## For review

Same format as WO1 — it worked well:

1. `git diff --stat`
2. `git diff` for `prepare.py` and `cli.py` only
3. `pytest -q` tail
4. the mutation-check result for the new tests
5. disagreements and what you did instead

Do **not** paste whole unchanged files. If a task's specification is internally
contradictory — WO1 had one — say so rather than guessing.
