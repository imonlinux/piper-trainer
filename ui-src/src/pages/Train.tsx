import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, get, getText, post, postEmpty } from "../api";
import { lossHistory, percentDone, summarize, summaryText } from "../joblog";
import type { LossPoint } from "../joblog";
import { progressText, useJobStream } from "../hooks";
import type {
  Checkpoint,
  Job,
  ProjectDetail,
  TrainPreview,
} from "../types";

// The Train screen (§6.4). Two deliberate framings, never mixed:
//  - "continue this run"  → add_epochs, resume auto (optimizer + counter)
//  - "start from this voice" → warmstart weights + max_epochs from zero
// The §1.4 ceiling rule is encoded by which param the screen submits;
// the page never shows an absolute epoch ceiling for a resume.
// Everything else is honest projection: the wall-clock estimate exists
// only when a previous succeeded train job measured one.

type Mode = "continue" | "warm" | "scratch";

function fmtDuration(sec: number): string {
  if (!Number.isFinite(sec) || sec <= 0) return "-";
  const d = Math.floor(sec / 86400);
  const h = Math.floor((sec % 86400) / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m`;
  return `${Math.round(sec)}s`;
}

// warmstart picker value → the path the runner accepts. Catalog
// checkpoints record an absolute dir + fetched-file mapping (one .ckpt);
// run checkpoints carry a project-relative path the runner resolves.
function warmstartPath(sel: string, ckpts: Checkpoint[]): string | null {
  if (sel === "") return null;
  const [kind, ...rest] = sel.split(":");
  const key = rest.join(":");
  const c = ckpts.find((x) =>
    kind === "c" ? x.catalog_path === key : x.path === key,
  );
  if (!c) return null;
  if (kind === "c") {
    if (!c.dir || !c.files) return null;
    const ck = Object.values(c.files).find((f) => f.endsWith(".ckpt"));
    return ck ? `${c.dir}/${ck}` : null;
  }
  return c.path ?? null;
}

// Mirror of the server's fetch-checkpoint rule (runner._fetch_checkpoint):
// family/locale/voice/quality, safe charset, no "." or ".." segment.
const CATALOG_PATH_RE = /^[A-Za-z0-9_.-]+(\/[A-Za-z0-9_.-]+){3}$/;

export function validCatalogPath(p: string): boolean {
  return CATALOG_PATH_RE.test(p) &&
    !p.split("/").some((s) => s === "." || s === "..");
}

export function TrainPage({ name }: { name: string }) {
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [ckpts, setCkpts] = useState<Checkpoint[]>([]);
  const [gone, setGone] = useState(false);
  const [mode, setMode] = useState<Mode>("continue");
  const [warmSel, setWarmSel] = useState("");
  const [epochs, setEpochs] = useState("1000");
  const [batch, setBatch] = useState("32");
  const [skip, setSkip] = useState(false);
  const [preview, setPreview] = useState<TrainPreview | null>(null);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [measureId, setMeasureId] = useState<string | null>(null);
  const [fetchId, setFetchId] = useState<string | null>(null);
  const [fetchPath, setFetchPath] = useState("");
  const [measuredAt, setMeasuredAt] = useState(0);
  const [fullLog, setFullLog] = useState("");
  const [now, setNow] = useState(() => Date.now());
  const logPre = useRef<HTMLPreElement>(null);

  const { lines, job, logHref } = useJobStream(runId);
  const sum = useMemo(() => summarize(lines, job), [lines, job]);
  // The §2.2 train preview: a short real run whose measured step rate
  // becomes the projection's basis. Its log stays in the preview job;
  // this page only mirrors the outcome into the projection table.
  const mstream = useJobStream(measureId);
  const measuring =
    mstream.job !== null &&
    (mstream.job.state === "running" || mstream.job.state === "queued");
  // fetch-checkpoint: the job that populates base_checkpoints/ and the
  // picker's "fetched (catalog)" group.
  const fstream = useJobStream(fetchId);
  const fetching =
    fstream.job !== null &&
    (fstream.job.state === "running" || fstream.job.state === "queued");

  // Keep the raw log pinned to the bottom as lines land (same contract
  // as the Project page's stream — this page is now a watch surface too).
  useEffect(() => {
    const pre = logPre.current;
    if (pre) pre.scrollTop = pre.scrollHeight;
  }, [lines]);

  // ------------------------------------------------------------- loading
  useEffect(() => {
    let alive = true;
    Promise.all([
      get<ProjectDetail>(`/projects/${name}`),
      get<Checkpoint[]>(`/projects/${name}/checkpoints`),
    ])
      .then(([d, cks]) => {
        if (!alive) return;
        setDetail(d);
        setCkpts(cks);
        setFetchPath(d.config.catalog_path ?? "");
        // "continue" is the natural default once a run exists; with a
        // catalog voice — fetched or merely chosen at creation — the page
        // opens on warmstart, where the fetch row lives. (Defaulting a
        // chosen-but-unfetched voice to scratch once produced a 4000-epoch
        // from-nothing run; never again.)
        const hasRun = cks.some((c) => c.source === "run");
        if (!hasRun) {
          setMode(
            cks.some((c) => c.source === "catalog") || d.config.catalog_path
              ? "warm"
              : "scratch",
          );
        }
        // resume hint: continue from where the last run stopped
        const ep = Math.max(
          -1,
          ...cks.filter((c) => c.source === "run").map((c) => c.epoch ?? -1),
        );
        setEpochs(ep >= 0 ? "1000" : "4000");
      })
      .catch((e: Error) => {
        if (!alive) return;
        if (e instanceof ApiError && e.status === 404) setGone(true);
        else setError(String(e));
      });
    // Watch the newest train job (any state): the server replays the log
    // tail for terminal jobs, so the curves and progress survive reloads.
    get<Job[]>(`/projects/${name}/jobs`)
      .then((jobs) => {
        if (!alive) return;
        const t = jobs.find((j) => j.kind === "train");
        if (t) setRunId(t.id);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name]);

  // Full log for the loss history (the live tail is capped at 5000
  // lines; a 4000-epoch run blows past that within hours).
  useEffect(() => {
    setFullLog("");
    if (!runId) return;
    let alive = true;
    getText(`/jobs/${runId}/log`)
      .then((t) => {
        if (alive) setFullLog(t);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [runId]);

  // Elapsed clock ticks only while something is running.
  useEffect(() => {
    if (job === null || job.state !== "running") return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [job]);

  // Preview projection, debounced; invalid dials simply clear it.
  // measuredAt rides in the deps so a finished measurement refetches it.
  useEffect(() => {
    const n = parseInt(epochs, 10);
    const b = parseInt(batch, 10);
    if (!Number.isFinite(n) || n < 1 || !Number.isFinite(b) || b < 1) {
      setPreview(null);
      return;
    }
    let alive = true;
    const t = setTimeout(() => {
      get<TrainPreview>(
        `/projects/${name}/train/preview?epochs=${n}&batch_size=${b}`,
      )
        .then((p) => {
          if (alive) setPreview(p);
        })
        .catch(() => {
          if (alive) setPreview(null);
        });
    }, 350);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [name, epochs, batch, measuredAt]);

  useEffect(() => {
    if (mstream.job?.state === "succeeded") setMeasuredAt(Date.now());
  }, [mstream.job?.state]);

  // A finished fetch-checkpoint refreshes the picker and selects what
  // arrived, so the next click is "train", not "find the new entry".
  useEffect(() => {
    const st = fstream.job?.state;
    if (st === "succeeded") {
      get<Checkpoint[]>(`/projects/${name}/checkpoints`)
        .then((cks) => {
          setCkpts(cks);
          const p = fetchPath.trim();
          if (p && cks.some((c) => c.catalog_path === p)) {
            setWarmSel(`c:${p}`);
          }
        })
        .catch(() => undefined);
      setMessage(`base voice fetched — selected in the picker`);
      setFetchId(null);
    } else if (st === "failed") {
      setError(`fetch failed: ${fstream.job?.error ?? "see its log"}`);
      setFetchId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fstream.job?.state]);

  // ------------------------------------------------------------ derived
  const runCkpts = ckpts.filter((c) => c.source === "run");
  const catCkpts = ckpts.filter((c) => c.source === "catalog");
  const lastRunEpoch = runCkpts.length
    ? Math.max(...runCkpts.map((c) => c.epoch ?? 0))
    : null;
  const warmPath = warmstartPath(warmSel, ckpts);
  const n = parseInt(epochs, 10);
  const b = parseInt(batch, 10);
  const ready =
    Number.isFinite(n) && n >= 1 && Number.isFinite(b) && b >= 1 &&
    (mode === "continue"
      ? runCkpts.length > 0
      : mode === "warm"
        ? warmPath !== null
        : true); // scratch: dials only, no base checkpoint required

  const loss = useMemo(
    () => lossHistory([...fullLog.split("\n"), ...lines]),
    [fullLog, lines],
  );
  // Placeholder honesty: "no epochs yet" and "epochs but no loss values
  // in their lines" are different situations and say different things.
  const anyEpoch = useMemo(
    () =>
      [...fullLog.split("\n"), ...lines].some((l) => /\bEpoch \d+:/.test(l)),
    [fullLog, lines],
  );

  // ----------------------------------------------------------- handlers
  async function run(): Promise<void> {
    setError("");
    setMessage("");
    const params: Record<string, unknown> =
      mode === "continue"
        ? { add_epochs: n }
        : mode === "warm"
          ? { warmstart: warmPath, max_epochs: n }
          : { max_epochs: n }; // scratch: fresh weights, counter from zero
    params.batch_size = b;
    if (skip) params.skip_validate = true;
    const what =
      mode === "continue"
        ? `continue: ${n} more epochs`
        : mode === "warm"
          ? `warmstart: ${n} epochs from ${warmPath}`
          : `scratch: ${n} epochs from nothing`;
    if (!confirm(`Start training?\n${what}`)) return;
    try {
      const job2 = await post<Job>(`/projects/${name}/jobs`, {
        kind: "train",
        params,
      });
      setRunId(job2.id);
      setMessage(`train job queued — watch it below`);
      get<Checkpoint[]>(`/projects/${name}/checkpoints`)
        .then(setCkpts)
        .catch(() => undefined);
    } catch (ex) {
      setError(String(ex));
    }
  }

  async function cancel(): Promise<void> {
    if (!runId) return;
    try {
      await postEmpty(`/jobs/${runId}/cancel`);
    } catch (ex) {
      setError(String(ex));
    }
  }

  // Populate base_checkpoints/ from the HF catalog (§3.5). The endpoint
  // existed from the start; nothing in the UI called it, which is why the
  // warmstart picker could be empty forever.
  async function fetchBase(): Promise<void> {
    setError("");
    setMessage("");
    const p = fetchPath.trim();
    if (!validCatalogPath(p)) return;
    try {
      const j = await post<Job>(
        `/projects/${name}/checkpoints/fetch`,
        { catalog_path: p },
      );
      setFetchId(j.id);
    } catch (ex) {
      setError(String(ex));
    }
  }

  // The measured projection basis (§2.2): same mode/dials the real run
  // would use, but Lightning is capped at ~50 steps. Failure here is the
  // point — it is the full run's failure, caught for the cost of a minute.
  async function measure(): Promise<void> {
    setError("");
    const params: Record<string, unknown> =
      mode === "continue"
        ? { add_epochs: n }
        : mode === "warm"
          ? { warmstart: warmPath, max_epochs: n }
          : { max_epochs: n };
    params.batch_size = b;
    if (skip) params.skip_validate = true;
    try {
      const j = await post<Job>(`/projects/${name}/preview`, {
        stage: "train",
        params,
      });
      setMeasureId(j.id);
    } catch (ex) {
      setError(String(ex));
    }
  }

  if (gone) {
    return (
      <>
        <h1>{name}</h1>
        <p className="error">no such project: {name}</p>
        <p>
          <a href="#/projects">back to projects</a>
        </p>
      </>
    );
  }
  if (detail === null) return <p className="muted">loading…</p>;

  const running = job !== null && (job.state === "running" || job.state === "queued");
  const pct = percentDone(sum, job?.state ?? "");
  const elapsed =
    job !== null && job.started_at !== null
      ? ((job.state === "succeeded" || job.state === "failed"
          ? Date.parse(job.finished_at ?? job.started_at)
          : now) -
          Date.parse(job.started_at)) / 1000
      : null;
  const projected =
    preview?.projected_seconds != null ? fmtDuration(preview.projected_seconds) : null;

  return (
    <>
      <h1>Train — {name}</h1>
      <p className="muted">
        tier {detail.config.tier ?? "medium"} · espeak{" "}
        {detail.config.espeak_voice ?? "en-us"} · {detail.dataset.rows} clips
        {" · "}
        <a href={`#/project/${name}`}>back to project</a>
      </p>
      {error && <p className="error">{error}</p>}
      {message && <p className="muted">{message}</p>}

      <h2>Mode</h2>
      <div className="row">
        <label className="inline" title="resume the latest checkpoint: optimizer state and the epoch counter continue where they stopped">
          <input
            type="radio"
            name="mode"
            checked={mode === "continue"}
            disabled={runCkpts.length === 0}
            onChange={() => setMode("continue")}
          />
          continue this run{lastRunEpoch !== null ? ` (at epoch ${lastRunEpoch})` : ""}
        </label>
        <label className="inline" title="weights-only start from a base voice: the epoch count begins at zero">
          <input
            type="radio"
            name="mode"
            checked={mode === "warm"}
            onChange={() => setMode("warm")}
          />
          start from this voice
        </label>
        <label className="inline" title="no base checkpoint: train a fresh voice from nothing">
          <input
            type="radio"
            name="mode"
            checked={mode === "scratch"}
            onChange={() => setMode("scratch")}
          />
          from scratch
        </label>
      </div>
      {mode === "warm" && (
        <div className="row">
          <select
            value={warmSel}
            onChange={(e) => setWarmSel(e.target.value)}
            aria-label="warmstart checkpoint"
          >
            <option value="">— pick a base checkpoint —</option>
            <optgroup label="fetched (catalog)">
              {catCkpts.map((c) => (
                <option key={c.catalog_path} value={`c:${c.catalog_path}`}>
                  {c.catalog_path}
                </option>
              ))}
            </optgroup>
            <optgroup label="this project's runs">
              {runCkpts.map((c) => (
                <option key={c.path} value={`r:${c.path}`}>
                  {c.tier} · {c.name}
                  {c.epoch != null ? ` (epoch ${c.epoch})` : ""}
                </option>
              ))}
            </optgroup>
          </select>
          {warmSel === "" && (
            <span className="muted">a warmstart needs a base checkpoint</span>
          )}
        </div>
      )}
      {mode === "warm" && (
        <>
          <div className="row">
            <input
              value={fetchPath}
              onChange={(e) => setFetchPath(e.target.value)}
              placeholder="family/locale/voice/quality"
              aria-label="catalog checkpoint path"
              spellCheck={false}
              style={{ width: "24em" }}
            />
            <button
              disabled={
                !validCatalogPath(fetchPath.trim()) || fetching || running ||
                measuring
              }
              onClick={() => void fetchBase()}
            >
              fetch base voice
            </button>
            {fetching && fstream.job !== null && (
              <span className="muted">
                {fstream.job.state} · {progressText(fstream.job.progress)}
              </span>
            )}
          </div>
          {catCkpts.length === 0 && (
            <p className="muted">
              nothing fetched yet — paste the catalog path this project was
              created with (prefilled above) and fetch it once; every
              project reuses it after that
            </p>
          )}
        </>
      )}

      <h2>Dials</h2>
      <div className="row">
        <label className="inline">
          {mode === "continue" ? "epochs (more)" : "epochs (total)"}
          <input
            type="number"
            min={1}
            value={epochs}
            style={{ width: "7em" }}
            onChange={(e) => setEpochs(e.target.value)}
          />
        </label>
        <label className="inline">
          batch size
          <input
            type="number"
            min={1}
            value={batch}
            style={{ width: "5em" }}
            onChange={(e) => setBatch(e.target.value)}
          />
        </label>
        <label
          className="inline"
          title="train even when validation reports errors (same as the CLI's --skip-validate)"
        >
          <input
            type="checkbox"
            checked={skip}
            onChange={(e) => setSkip(e.target.checked)}
          />
          skip validation
        </label>
        <button
          disabled={!ready || running || measuring}
          onClick={() => void run()}
        >
          {mode === "continue" ? `train ${n || "?"} more` : `train ${n || "?"} epochs`}
        </button>
      </div>

      <h2>Projection</h2>
      <div className="row">
        <button
          disabled={!ready || running || measuring}
          onClick={() => void measure()}
        >
          {measuring ? "measuring…" : "measure (~50 real steps)"}
        </button>
        <span className="muted">
          runs a short training burst with these dials, then projects the
          full run from the measured speed
        </span>
      </div>
      {measuring && (
        <p className="muted">
          train preview running — its log is in the jobs list
        </p>
      )}
      {mstream.job?.state === "failed" && (
        <p className="error">
          train preview failed: {mstream.job.error ?? "see its log"} — the
          full run would have failed the same way
        </p>
      )}
      {preview === null ? (
        <p className="muted">enter valid dials to see the projection</p>
      ) : (
        <table>
          <tbody>
            <tr>
              <th>clips</th>
              <td className="num">{preview.clips}</td>
              <th>steps / epoch</th>
              <td className="num">{preview.steps_per_epoch ?? "-"}</td>
            </tr>
            <tr>
              <th>total steps</th>
              <td className="num">{preview.total_steps ?? "-"}</td>
              <th>sample rate</th>
              <td className="num">{preview.sample_rate ?? "-"}</td>
            </tr>
            <tr>
              <th>projected wall clock</th>
              <td className="num">{projected ?? "no basis yet"}</td>
              <th>basis</th>
              <td>{preview.basis ?? "first run measures it"}</td>
            </tr>
          </tbody>
        </table>
      )}

      <h2>Current run</h2>
      {job === null ? (
        <p className="muted">no train job yet — configure and start one</p>
      ) : (
        <>
          <p className="row">
            <span className={`state-${job.state}`}>{job.id}</span>
            <span className="muted">{job.kind}</span>
            <span className={`state-${job.state}`}>{job.state}</span>
            <span className="muted">{progressText(job.progress)}</span>
            {elapsed !== null && (
              <span className="muted">elapsed {fmtDuration(elapsed)}</span>
            )}
            {running && projected !== null && (
              <span className="muted">projected total {projected}</span>
            )}
            {running && (
              <button onClick={() => void cancel()}>cancel</button>
            )}
          </p>
          <div className="logstrip">
            <span className="muted">
              {job.state === "succeeded" && sum.total !== null
                ? `epoch ${sum.total}/${sum.total}`
                : summaryText(sum)}
            </span>
            {pct !== null && (
              <span className="bar" aria-hidden="true">
                <span className="bar-fill" style={{ width: `${pct}%` }} />
              </span>
            )}
            {job.error && <span className="error">{job.error}</span>}
          </div>
          <LossCanvas
            points={loss}
            note={
              anyEpoch
                ? "epochs seen, but their log lines carry no loss values yet"
                : undefined
            }
          />
          <p className="muted">
            {loss.length} epochs parsed
            {logHref && (
              <>
                {" · "}
                <a href={logHref}>full log</a>
              </>
            )}
          </p>
          <pre className="log" ref={logPre}>
            {lines.join("\n")}
          </pre>
        </>
      )}
    </>
  );
}

// The loss curve (§6.4): training_loss per epoch solid, validation_loss
// dashed when the run emitted it. Canvas, no chart dependency — the
// points arrive as plain log lines and the redraw is a plain effect.
function LossCanvas({
  points,
  note,
}: {
  points: LossPoint[];
  note?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (cv === null) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth;
    const h = 220;
    cv.width = w * dpr;
    cv.height = h * dpr;
    const ctx = cv.getContext("2d");
    if (ctx === null) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (points.length < 2) {
      ctx.fillStyle = "#777";
      ctx.font = "12px sans-serif";
      ctx.fillText(note ?? "waiting for the first epochs…", 12, 24);
      return;
    }
    const padL = 52;
    const padR = 12;
    const padT = 12;
    const padB = 22;
    const xs = points.map((p) => p.epoch);
    const ys = points.flatMap((p) => [p.train, p.val].filter((v): v is number => v !== null));
    const x0 = Math.min(...xs);
    const x1 = Math.max(...xs);
    const y0 = Math.min(...ys);
    const y1 = Math.max(...ys);
    const yPad = (y1 - y0) * 0.08 || 0.1;
    const X = (e: number) =>
      padL + ((e - x0) / Math.max(1, x1 - x0)) * (w - padL - padR);
    const Y = (v: number) =>
      padT + (1 - (v - (y0 - yPad)) / (y1 - y0 + 2 * yPad)) * (h - padT - padB);
    // horizontal grid + value labels
    ctx.strokeStyle = "#e4e4e7";
    ctx.fillStyle = "#777";
    ctx.font = "10px sans-serif";
    for (let i = 0; i <= 4; i++) {
      const v = y0 - yPad + ((y1 - y0 + 2 * yPad) * i) / 4;
      const y = Y(v);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.fillText(v.toFixed(3), 4, y + 3);
    }
    // x extent labels (epoch numbers)
    ctx.fillText(String(x0), padL, h - 6);
    ctx.fillText(String(x1), w - padR - 18, h - 6);
    const drawSeries = (
      pick: (p: LossPoint) => number | null,
      color: string,
      dash: number[],
    ): void => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      for (const p of points) {
        const v = pick(p);
        if (v === null) continue;
        const x = X(p.epoch);
        const y = Y(v);
        if (started) ctx.lineTo(x, y);
        else {
          ctx.moveTo(x, y);
          started = true;
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    };
    drawSeries((p) => p.val, "#b1651c", [4, 3]);
    drawSeries((p) => p.train, "#4a5fc1", []);
  }, [points]);
  return <canvas ref={ref} style={{ width: "100%", height: "220px" }} />;
}
