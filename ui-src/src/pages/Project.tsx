import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  del,
  get,
  post,
  postEmpty,
  upload,
} from "../api";
import { progressText, useJobStream, usePoll } from "../hooks";
import {
  percentDone,
  resultText,
  summaryText,
  summarize,
} from "../joblog";
import type { LogSummary } from "../joblog";
import type { Job, ProjectDetail, SourceInfo } from "../types";

// Project detail: directory counts, sources, ingest (upload / url /
// media-site / hf-dataset), the job table with live watch, and the job
// actions. Training moved to its own screen (§6.4) — this page links
// there. One page, one poll, one websocket — every timer and socket
// here is owned by an effect and cleaned up on route change (the React
// statement of Bones finding 7).

export function ProjectPage({ name }: { name: string }) {
  // ------------------------------------------------------- hooks (all of
  // them before any early return: 404 and loading states come after, or
  // hook order flips between renders)
  const [p, setP] = useState<ProjectDetail | null>(null);
  const [loadError, setLoadError] = useState<Error | null>(null);
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [selSources, setSelSources] = useState<Set<string>>(new Set());
  const [sourceMsg, setSourceMsg] = useState("");
  const [actionError, setActionError] = useState("");
  const [uploadError, setUploadError] = useState("");
  // Ingest source type (§2.5): upload is staged by the API; url,
  // media-site and hf-dataset are acquired by the runner itself.
  const [ingestMode, setIngestMode] = useState<
    "upload" | "url" | "media-site" | "hf"
  >("upload");
  // Set right after an upload POST: the ingest runs as a job, so the
  // sources list can only be reloaded once that job settles (the poll
  // feeds p.jobs; when we see the terminal state we reload once).
  const [ingestId, setIngestId] = useState<string | null>(null);
  const [watchId, setWatchId] = useState<string | null>(null);
  const autoWatched = useRef(false);
  const { lines, job: watched, logHref } = useJobStream(watchId);
  const logPre = useRef<HTMLPreElement>(null);
  // The clean layer over the raw tail: progress + RESULT folded from the
  // runner's ##directives (and Lightning's Epoch lines for train jobs).
  const watchSum = useMemo(() => summarize(lines, watched), [lines, watched]);

  function load(): void {
    get<ProjectDetail>(`/projects/${name}`)
      .then((d) => {
        setP(d);
        setLoadError(null);
      })
      .catch((e: Error) => setLoadError(e));
  }

  function loadSources(): void {
    get<SourceInfo[]>(`/projects/${name}/sources`)
      .then((s) => {
        setSources(s);
        // a reload can only shrink the eligible set (files moved out from
        // under us) — drop selections that no longer resolve
        setSelSources((prev) => {
          const next = new Set(
            [...prev].filter((n) => s.some((x) => x.name === n)),
          );
          return next.size === prev.size ? prev : next;
        });
      })
      .catch(() => setSources([]));
  }

  useEffect(() => {
    setP(null);
    setLoadError(null);
    setWatchId(null);
    autoWatched.current = false;
    get<ProjectDetail>(`/projects/${name}`)
      .then((d) => {
        setP(d);
        setLoadError(null);
      })
      .catch((e: Error) => setLoadError(e));
    loadSources();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name]);

  // Auto-watch a running job; with none running, watch the newest job
  // anyway — the server replays state + the log tail on connect for
  // finished jobs too, so the pane shows the last run instead of an
  // empty box. Once per page; the table can switch the watch after.
  useEffect(() => {
    if (p !== null && !autoWatched.current) {
      autoWatched.current = true;
      const running = p.jobs.find((j) => j.state === "running");
      setWatchId(running ? running.id : (p.jobs[0]?.id ?? null));
    }
  }, [p]);

  // List refresh pauses only while the watched job is actually live
  // (the websocket carries its state); watching a finished job must NOT
  // pause the poll, or the job table would go stale forever.
  const watchActive =
    watchId !== null &&
    (watched === null || watched.state === "running" || watched.state === "queued");
  const paused = watchActive || loadError !== null;
  usePoll(async () => {
    try {
      const jobs = await get<Job[]>(`/projects/${name}/jobs`);
      setP((prev) => (prev ? { ...prev, jobs } : prev));
    } catch (ex) {
      // The project vanished out from under the open tab: surface the
      // 404 (which also pauses this poll) instead of spinning forever.
      // Anything else is a transient blip; the next tick retries.
      if (ex instanceof ApiError && ex.status === 404) setLoadError(ex);
    }
  }, 2000, paused);

  // Keep the log pinned to the bottom as lines land.
  useEffect(() => {
    const pre = logPre.current;
    if (pre) pre.scrollTop = pre.scrollHeight;
  }, [lines]);

  // The upload POST returns as soon as the ingest job is created, not
  // when the file has landed in raw/ — so reload the sources list when
  // the tracked ingest job reaches a terminal state. If that state is
  // never observed (poll paused while another job is watched), the next
  // live poll tick after un-watching heals it.
  useEffect(() => {
    if (ingestId === null || p === null) return;
    const ing = p.jobs.find((j) => j.id === ingestId);
    if (!ing) return;
    if (["succeeded", "failed", "canceled", "interrupted"].includes(ing.state)) {
      setIngestId(null);
      loadSources();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ingestId, p]);

  // ------------------------------------------------------------- guards
  // A deleted project must say so, not spin: project_or_404 answers 404
  // for any name without project.json, and polling into that is noise.
  if (loadError instanceof ApiError && loadError.status === 404) {
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
  if (loadError) return <p className="error">{String(loadError)}</p>;
  if (p === null) return <p className="muted">loading…</p>;

  // ---------------------------------------------------------- handlers
  function refreshSoon(): void {
    setTimeout(load, 300);
  }

  async function runJob(
    kind: string,
    params: Record<string, unknown> = {},
  ): Promise<void> {
    setActionError("");
    try {
      await post<Job>(`/projects/${name}/jobs`, { kind, params });
      refreshSoon();
    } catch (ex) {
      setActionError(String(ex));
    }
  }

  async function cancelJob(id: string): Promise<void> {
    setActionError("");
    try {
      await postEmpty(`/jobs/${id}/cancel`);
      setTimeout(load, 200);
    } catch (ex) {
      setActionError(String(ex));
    }
  }

  async function startJob(id: string): Promise<void> {
    setActionError("");
    try {
      await postEmpty(`/jobs/${id}/start`);
      setTimeout(load, 200);
    } catch (ex) {
      setActionError(String(ex));
    }
  }

  async function doIngest(
    e: React.FormEvent<HTMLFormElement>,
  ): Promise<void> {
    e.preventDefault();
    setUploadError("");
    const formEl = e.currentTarget;
    try {
      if (ingestMode === "upload") {
        const input = formEl.elements.namedItem(
          "files",
        ) as HTMLInputElement | null;
        const fd = new FormData();
        for (const f of input?.files ?? []) fd.append("files", f);
        if (fd.getAll("files").length === 0) return;
        const job = await upload<Job>(`/projects/${name}/ingest`, fd);
        setIngestId(job.id);
      } else {
        // The other source types (§2.5.2–2.5.4) need no staging: the
        // runner acquires the bytes itself, so a plain job does it.
        const fd = new FormData(formEl);
        const opt = (k: string): string | undefined => {
          const v = String(fd.get(k) ?? "").trim();
          return v === "" ? undefined : v;
        };
        const params: Record<string, unknown> =
          ingestMode === "url"
            ? { source_type: "url", url: opt("url") }
            : ingestMode === "media-site"
              ? {
                  source_type: "media-site",
                  url: opt("url"),
                  sections: opt("sections"),
                  playlist: fd.get("playlist") === "on",
                }
              : {
                  source_type: "hf-dataset",
                  repo_id: opt("repo_id"),
                  split: opt("split"),
                };
        const job = await post<Job>(`/projects/${name}/jobs`, {
          kind: "ingest",
          params,
        });
        setIngestId(job.id);
      }
      formEl.reset();
      refreshSoon();
    } catch (ex) {
      setUploadError(String(ex));
    }
  }

  // §1.4 epoch arithmetic moved to the Train screen (§6.4): this page
  // links there instead of hosting its own epochs box.
  const cfg = p.config ?? {};

  async function doDeleteSources(): Promise<void> {
    const names = [...selSources];
    if (names.length === 0) return;
    if (
      !confirm(
        `Move ${names.length} source file(s) to .trash?\n${names.join("\n")}\nNothing is destroyed.`,
      )
    )
      return;
    setActionError("");
    setSourceMsg("");
    try {
      const res = await post<{ moved: string[]; missing: string[] }>(
        `/projects/${name}/sources/delete`,
        { names },
      );
      setSelSources(new Set());
      setSourceMsg(
        `moved ${res.moved.length} to .trash` +
          (res.missing.length ? ` · not found: ${res.missing.join(", ")}` : ""),
      );
      loadSources();
      load(); // the raw/ directory card counts sources
    } catch (ex) {
      setActionError(String(ex));
    }
  }

  async function doDelete(): Promise<void> {
    if (!confirm(`Move project "${name}" to .trash? Nothing is destroyed.`))
      return;
    setActionError("");
    try {
      await del(`/projects/${name}`);
      location.hash = "#/projects";
    } catch (ex) {
      setActionError(String(ex));
    }
  }

  // -------------------------------------------------------------- view
  const jobs = p.jobs.slice(0, 10).map((j) => (j.id === watched?.id ? watched : j));

  return (
    <>
      <h1>{p.name}</h1>
      <p className="muted">
        {p.clips} clips, {p.minutes ?? "?"} min, tiers trained:{" "}
        {p.tiers_trained.join(", ") || "none"} {watched ? summaryText(watchSum) : ""}
      </p>

      <h2>Directories</h2>
      <div className="grid">
        {Object.entries(p.directories).map(([k, v]) => (
          <div className="cell" key={k}>
            {k}
            <b>{v}</b>
          </div>
        ))}
      </div>

      <h2>Dataset</h2>
      <p>
        {p.dataset.rows} rows, {p.dataset.malformed_lines} malformed lines,
        line endings: {p.dataset.line_endings ?? "-"}
        {cfg.espeak_voice ? `, espeak voice: ${cfg.espeak_voice}` : ""}
        {cfg.tier ? `, tier: ${cfg.tier}` : ""}
      </p>

      <h2>Sources (raw/)</h2>
      {sources.length === 0 ? (
        <p className="muted">no source recordings</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>
                <input
                  type="checkbox"
                  aria-label="select all sources"
                  checked={
                    sources.length > 0 && selSources.size === sources.length
                  }
                  onChange={(e) =>
                    setSelSources(
                      e.target.checked
                        ? new Set(sources.map((s) => s.name))
                        : new Set(),
                    )
                  }
                />
              </th>
              <th>name</th>
              <th>codec</th>
              <th>rate</th>
              <th>ch</th>
              <th>duration</th>
            </tr>
          </thead>
          <tbody>
            {sources.map((s) => (
              <tr key={s.name}>
                <td>
                  <input
                    type="checkbox"
                    aria-label={`select ${s.name}`}
                    checked={selSources.has(s.name)}
                    onChange={(e) =>
                      setSelSources((prev) => {
                        const next = new Set(prev);
                        if (e.target.checked) next.add(s.name);
                        else next.delete(s.name);
                        return next;
                      })
                    }
                  />
                </td>
                <td>{s.name}</td>
                <td>{s.codec ?? "?"}</td>
                <td className="num">{s.sample_rate ?? "?"}</td>
                <td className="num">{s.channels ?? "?"}</td>
                <td className="num">{s.duration ?? "?"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="row">
        <button
          disabled={selSources.size === 0}
          onClick={() => void doDeleteSources()}
        >
          delete selected ({selSources.size})
        </button>
        {sourceMsg && <span className="muted">{sourceMsg}</span>}
      </p>
      <form
        className="row"
        onSubmit={(e) => {
          void doIngest(e);
        }}
      >
        <select
          value={ingestMode}
          aria-label="source type"
          onChange={(e) => setIngestMode(e.target.value as typeof ingestMode)}
        >
          <option value="upload">upload files</option>
          <option value="url">direct url</option>
          <option value="media-site">media site (yt-dlp)</option>
          <option value="hf">huggingface dataset</option>
        </select>
        {ingestMode === "upload" && <input type="file" name="files" multiple />}
        {ingestMode === "url" && (
          <input
            name="url"
            placeholder="https://…/clip.wav"
            style={{ width: "22em" }}
          />
        )}
        {ingestMode === "media-site" && (
          <>
            <input
              name="url"
              placeholder="video page url"
              style={{ width: "18em" }}
            />
            <input
              name="sections"
              placeholder="* 00:10-01:20 (optional)"
              style={{ width: "15em" }}
            />
            <label className="inline" title="download a whole playlist instead of one video">
              <input type="checkbox" name="playlist" />
              playlist
            </label>
          </>
        )}
        {ingestMode === "hf" && (
          <>
            <input
              name="repo_id"
              placeholder="owner/dataset"
              style={{ width: "16em" }}
            />
            <input
              name="split"
              placeholder="split (optional)"
              style={{ width: "10em" }}
            />
          </>
        )}
        <button type="submit">ingest</button>
        {uploadError && <span className="error">{uploadError}</span>}
      </form>
      <p className="muted">
        {ingestMode === "media-site" &&
          "extracts audio as wav; needs yt-dlp in the image, which reports its own errors when a video refuses to download"}
        {ingestMode === "hf" &&
          "audio-directory datasets with a csv/tsv/jsonl transcript file; parquet-embedded audio is refused"}
        {ingestMode === "url" &&
          "one media file over http(s); an HTML error page is refused by content type"}
      </p>

      <h2>Jobs</h2>
      <p>
        <a href={`#/prepare/${name}`}>prepare tuner (segment preview + promote)</a>
        {" · "}
        <a href={`#/audit/${name}`}>audit dataset (transcripts, validation, clean)</a>
        {" · "}
        <a href={`#/train/${name}`}>train screen (warmstart, resume, projection)</a>
      </p>
      <div className="row">
        <button onClick={() => void runJob("prepare")}>run prepare</button>
        <button onClick={() => void runJob("transcribe")}>run transcribe</button>
        <button onClick={() => void runJob("export")}>run export</button>
      </div>
      {actionError && <p className="error">{actionError}</p>}
      <JobsTable
        jobs={jobs}
        watchId={watchId}
        onWatch={setWatchId}
        onCancel={(id) => void cancelJob(id)}
        onStart={(id) => void startJob(id)}
      />
      {logHref && (
        <p>
          <a href={logHref}>full log</a>
        </p>
      )}
      <LogStrip sum={watchSum} job={watched} />
      <pre className="log" ref={logPre}>
        {lines.join("\n")}
      </pre>

      <h2>Danger</h2>
      <p>
        <button onClick={() => void doDelete()}>
          delete (moves to .trash)
        </button>
      </p>
    </>
  );
}

// Progress strip above the raw log: the runner's TARGET/PROGRESS
// directives (plus Lightning's Epoch lines) drive the bar; the RESULT
// directive becomes a one-line summary or error. Nothing here blocks
// the raw tail below — that stays the source of truth.
function LogStrip(props: { sum: LogSummary; job: Job | null }) {
  const { sum, job } = props;
  if (job === null) return null;
  // On success the last "Epoch N:" line under-reports by one (N runs,
  // N+1 are done); show the run's own total instead.
  const text =
    job.state === "succeeded" && sum.total !== null
      ? `${sum.unit ?? ""} ${sum.total}/${sum.total}`.trim()
      : summaryText(sum);
  const pct = percentDone(sum, job.state);
  return (
    <div className="logstrip">
      {text && (
        <span className={job.state === "failed" ? "error" : "muted"}>{text}</span>
      )}
      {pct !== null && (
        <span className="bar" aria-hidden="true">
          <span className="bar-fill" style={{ width: `${pct}%` }} />
        </span>
      )}
      {sum.error !== null ? (
        <span className="error">{sum.error}</span>
      ) : sum.result !== null ? (
        <span className="ok">{resultText(sum.result)}</span>
      ) : null}
    </div>
  );
}

function JobsTable(props: {
  jobs: Job[];
  watchId: string | null;
  onWatch: (id: string) => void;
  onCancel: (id: string) => void;
  onStart: (id: string) => void;
}) {
  if (props.jobs.length === 0) {
    return <p className="muted">no jobs yet</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>id</th>
          <th>kind</th>
          <th>state</th>
          <th>progress</th>
          <th>error</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {props.jobs.map((j) => {
          const active = j.state === "running" || j.state === "queued";
          return (
            <tr key={j.id}>
              <td>{j.id}</td>
              <td>{j.kind}</td>
              <td className={`state-${j.state}`}>{j.state}</td>
              <td>{progressText(j.progress)}</td>
              <td>{j.error ?? ""}</td>
              <td>
                {active && (
                  <button onClick={() => props.onCancel(j.id)}>cancel</button>
                )}{" "}
                {/* every job has a log once it has run; the server
                    replays the tail for terminal jobs on connect */}
                <button onClick={() => props.onWatch(j.id)}>
                  {props.watchId === j.id ? "watching" : "log"}
                </button>{" "}
                {j.state === "queued" && (
                  <button onClick={() => props.onStart(j.id)}>start</button>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
