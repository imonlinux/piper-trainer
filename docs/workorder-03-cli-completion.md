# Work order 03 — CLI completion and concurrency safety

**Prerequisite:** WO2 merged.
**Theme:** finish the CLI layer and make it safe for concurrent access. This is the last work order before the API wraps these functions, so anything that would be awkward to change once an HTTP layer depends on it belongs here.

Contents: three carry-overs from the WO2 review, the three items deferred from WO2 (§3.4, §3.5, §3.6), and the container verification that WO2's stub-based acceptance could not provide.

**Not in scope:** the FastAPI app, the job runner, and log streaming. Those are WO4 and are specified in `docs/piper-trainer-ui-design.md` §1 and §4.

---

## Task 1 — WO2 review carry-overs

### 1a. `transcribe` silently drops malformed metadata lines

WO2 established for `clean` that removing a line requires its code to be in the plan. `transcribe` is not `clean` and so does not violate that contract directly, but a user running `transcribe --only-missing` who loses hand-edited-but-malformed rows gets the same class of surprise.

Carry malformed lines through the rewrite using the `raw_lines` splice added in WO2, and report the count in the result dict. If a malformed line cannot be placed sensibly relative to the new rows, append it at the end rather than dropping it — validate and clean exist to deal with it afterwards.

### 1b. `line_endings_fixed` conflates two different repairs

Counting `"none"` as fixed is defensible — the rewrite does add the missing final newline — but `line_endings_fixed: true` on a file that never had CRLF will puzzle whoever reads it.

Keep the behaviour; split the reporting. `describe()` should say *"converted CRLF line endings to LF"* or *"added missing final newline"*, whichever applies.

### 1c. `--max-epochs` default now lives in two places

Moving the default out of argparse was necessary to detect explicit-vs-default, but the value now appears in both `resolve_max_epochs` and the help string, where it can drift. Define one module-level constant and reference it from both.

---

## Task 2 — restore rows from the quarantine manifest (§3.4)

`restore` moves WAVs back but leaves the user to re-run a full transcribe or hand-edit `metadata.csv` to recover the rows. The manifest already records each quarantined clip's id, reasons, and **transcript** — everything needed to put the row back.

- `restore` re-adds each restored clip's row from `quarantine/manifest.csv`.
- `--files-only` preserves the current behaviour for anyone who wants to re-transcribe instead.
- If a clip has multiple manifest entries (quarantined, restored, quarantined again), use the most recent.
- If a row for that clip id already exists in `metadata.csv`, do not overwrite it; report the conflict.
- Rows are appended, then the file is rewritten through `metadata.write`. Order is not significant to training, but keep it stable — sort by clip id — so diffs stay readable.
- Report `files_restored` and `rows_restored` separately; they can legitimately differ.

Clips quarantined before this change still work, since the manifest format is unchanged.

---

## Task 3 — project lockfile (§3.5)

Two concurrent runs against the same project race on `dataset/wavs/`, `metadata.csv`, the cache directory, and checkpoints. Today nothing prevents it. The API will make this materially more likely — a user can start a job in the UI and run a CLI command in a terminal — and the design doc's "one running job per project" rule needs an enforcement point below the API, or the CLI simply bypasses it.

### Mechanism: `fcntl.flock`, not a PID file

Use an advisory exclusive lock on `<project>/.piper-trainer.lock` via `fcntl.flock(fd, LOCK_EX | LOCK_NB)`.

This matters: **the kernel releases a flock automatically when the holding process dies**, including on SIGKILL and on container teardown. A PID file cannot achieve that here, because a PID recorded inside one container namespace means nothing to another process checking liveness — stale-lock detection would be guesswork.

Write human-readable metadata into the locked file (command, start time, hostname/container id) so the failure message can name what holds it. Treat that content as advisory only; the lock itself is the flock.

### Which commands lock

| Locked (mutating) | Not locked (read-only) |
|---|---|
| `prepare`, `transcribe`, `clean --apply`, `restore`, `train`, `export` | `validate`, `sources`, `doctor`, `clean` dry run |

Read-only commands must never block; the UI will call `validate` while a job runs.

### Behaviour

- Non-blocking acquire. On failure, exit non-zero with a message naming the holding command and its start time.
- `--wait N` optionally blocks up to N seconds before giving up.
- Released on normal exit, on exception, and by the kernel on death — use a context manager, and do not rely on cleanup code running.

### Limitation to document

`flock` is per-kernel. It protects concurrent containers and CLI runs **on the same host**, which is the deployment model. It does not protect a project directory shared over NFS between machines. State this in the docstring rather than implying a guarantee that isn't there.

---

## Task 4 — declare the dependency posture (§3.6)

`pyproject.toml` declares zero runtime dependencies, which is correct for the container-only strategy but silently misleading: `pip install piper-trainer` outside the image yields a package that imports and then fails at the first real call.

- Add a `[project.optional-dependencies] runtime` group listing what the code actually imports (`auditok`, `faster-whisper`, `soundfile`, `numpy`), so a standalone install is at least *possible*.
- Document in the README that the supported path is the container, that `piper1-gpl` and its two C extensions plus `ffmpeg`, `espeak-ng`, and `deep-filter` are **not** pip-installable dependencies, and that `doctor` is the way to find out what's missing.
- Leave the default install dependency-free.

---

## Task 5 — container verification with real binaries

WO2's acceptance ran against stubs because the dev environment lacks `deep-filter`, `auditok`, and `torch`. That proved the control flow — skip/force, refusal messages, epoch arithmetic, mutual exclusion — which was most of the risk. It could not exercise the parts that touch real files and real tools.

Run inside the built container, on a scratch project, and report the output.

### 5a. Adversarial `raw/` directory

Create sources that stress Task 2d/2e from WO2:

```
foo bar.wav          # space
foo?bar.wav          # sanitizes to the same stem as the above
foo.mp3              # same stem, different extension
foo.wav
Ünïcödé námé.m4a     # non-ASCII
a.very.dotted.name.wav
```

Confirm: every source produces exactly one output, names are deterministic (run twice, compare), the original→final mapping is reported, and no file is silently overwritten. Paste the resulting `48k/` listing and the mapping.

### 5b. Idempotency with real audio

The WO2 acceptance sequence, with real `ffmpeg`, `deep-filter`, and `auditok`:

```bash
piper-trainer prepare <proj> --energy-threshold 55
piper-trainer prepare <proj> --energy-threshold 45     # count reflects 45 only
piper-trainer prepare <proj> --energy-threshold 45     # skipped
piper-trainer prepare <proj> --energy-threshold 45 --force
```

The fingerprint uses mtime_ns; confirm a stage does **not** spuriously re-run when inputs are untouched but the container has been restarted.

### 5c. Lock behaviour

Start a long `prepare` in one shell; in another, confirm `train` refuses with a message naming the holder and `validate` still succeeds. Then `kill -9` the first and confirm the lock is immediately available — this is the property a PID file could not give you.

---

## Task 6 — tests

- **transcribe**: malformed lines survive a `--only-missing` rewrite and are counted
- **clean**: `describe()` distinguishes the CRLF conversion from the added-newline case
- **restore**: rows come back from the manifest; `--files-only` restores no rows; the most recent entry wins for a re-quarantined clip; an existing row is not overwritten and the conflict is reported; `files_restored` and `rows_restored` differ where expected
- **lock**: a second acquire in the same process fails; the lock file is created and removed; a read-only command does not acquire; `--wait` times out. Use a real second process (`multiprocessing`) for at least one case — a same-process flock test can pass for the wrong reason, since flock semantics are per file description
- **config**: the max-epochs constant is referenced by both the resolver and the help text (assert they agree)

Mutation-check the additions as in WO1 §6 and WO2.

---

## Acceptance

```bash
pytest -q
grep -rn "os.getpid\|pidfile" src/    # no PID-based locking
```

Plus the three container runs from Task 5, with output pasted.

---

## For review

1. `git diff --stat`
2. `git diff` for the lock module and `cli.py` only
3. `pytest -q` tail
4. mutation-check result
5. **Task 5 output verbatim** — this is the part stubs could not cover, so it is the part worth reading
6. disagreements and what you did instead

If a specification is internally contradictory — WO1 had one, WO2 had two — say so rather than guessing.
