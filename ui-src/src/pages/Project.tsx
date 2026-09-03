import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  del,
  get,
  post,
  postEmpty,
  upload,
} from "../api";
import { progressText, useJobStream, usePoll } from "../hooks";
import type { Job, ProjectDetail, SourceInfo } from "../types";

// Project detail: directory counts, sources, upload ingest, the job
// table with live watch, and the job actions. One page, one poll, one
// websocket — every timer and socket here is owned by an effect and
// cleaned up on route change (the React statement of Bones finding 7).

export function ProjectPage({ name }: { name: string }) {
  // ------------------------------------------------------- hooks (all of
  // them before any early return: 404 and loading states come after, or
  // hook order flips between renders)
  const [p, setP] = useState<ProjectDetail | null>(null);
  const [loadError, setLoadError] = useState<Error | null>(null);
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [actionError, setActionError] = useState("");
  const [uploadError, setUploadError] = useState("");
  // Set right after an upload POST: the ingest runs as a job, so the
  // sources list can only be reloaded once that job settles (the poll
  // feeds p.jobs; when we see the terminal state we reload once).
  const [ingestId, setIngestId] = useState<string | null>(null);
  const [watchId, setWatchId] = useState<string | null>(null);
  const autoWatched = useRef(false);
  const { lines, job: watched, logHref } = useJobStream(watchId);
  const logPre = useRef<HTMLPreElement>(null);

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
      .then((s) => setSources(s))
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

  async function doUpload(
    e: React.FormEvent<HTMLFormElement>,
  ): Promise<void> {
    e.preventDefault();
    setUploadError("");
    const formEl = e.currentTarget;
    const input = formEl.elements.namedItem("files") as HTMLInputElement | null;
    const fd = new FormData();
    for (const f of input?.files ?? []) fd.append("files", f);
    if (fd.getAll("files").length === 0) return;
    try {
      const job = await upload<Job>(`/projects/${name}/ingest`, fd);
      formEl.reset();
      refreshSoon();
      setIngestId(job.id);
    } catch (ex) {
      setUploadError(String(ex));
    }
  }

  // §1.4: never expose the absolute max_epochs ceiling. A tier with no
  // checkpoint asks for epochs (submitted as max_epochs); a trained tier
  // asks for "N more" (submitted as add_epochs) so a resume can never
  // set the ceiling below the restored epoch counter and exit instantly.
  const cfg = p.config ?? {};
  const tier = cfg.tier ?? "medium";
  const trained = p.tiers_trained.includes(tier);

  async function doTrain(
    epochs: number,
    skipValidate: boolean,
  ): Promise<void> {
    if (!epochs || epochs < 1) {
      setActionError("epochs must be a number >= 1");
      return;
    }
    const params: Record<string, unknown> = trained
      ? { add_epochs: epochs }
      : { max_epochs: epochs };
    if (skipValidate) params.skip_validate = true;
    await runJob("train", params);
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
        {p.tiers_trained.join(", ") || "none"} {watched ? progressText(watched.progress) : ""}
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
      <form
        className="row"
        onSubmit={(e) => {
          void doUpload(e);
        }}
      >
        <input type="file" name="files" multiple />
        <button type="submit">upload</button>
        {uploadError && <span className="error">{uploadError}</span>}
      </form>

      <h2>Jobs</h2>
      <p>
        <a href={`#/prepare/${name}`}>prepare tuner (segment preview + promote)</a>
      </p>
      <div className="row">
        <button onClick={() => void runJob("prepare")}>run prepare</button>
        <button onClick={() => void runJob("transcribe")}>run transcribe</button>
        <TrainControls
          trained={trained}
          onTrain={(n, skip) => void doTrain(n, skip)}
        />
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

function TrainControls(props: {
  trained: boolean;
  onTrain: (epochs: number, skipValidate: boolean) => void;
}) {
  const [epochs, setEpochs] = useState(props.trained ? "1000" : "4000");
  const [skip, setSkip] = useState(false);
  return (
    <>
      <label className="inline">
        {props.trained ? "epochs (more)" : "epochs"}
        <input
          type="number"
          min={1}
          value={epochs}
          style={{ width: "6em" }}
          onChange={(e) => setEpochs(e.target.value)}
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
      <button onClick={() => props.onTrain(parseInt(epochs, 10), skip)}>
        {props.trained ? "train N more" : "run train"}
      </button>
    </>
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
