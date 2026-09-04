// The Audit screen (§6.3): the dataset table with playback and inline
// transcript editing, validation findings as filter chips, clean as
// plan-then-apply, and quarantine review with restore.
//
// Validation and clean are job kinds, not endpoints: the findings come
// from the latest succeeded validate/clean job's result, and every
// mutation here (validate, clean plan, clean apply, restore) submits a
// job exactly like the Project page does.
import { useEffect, useMemo, useRef, useState } from "react";
import { get, patch, post } from "../api";
import { usePoll } from "../hooks";
import type {
  Dataset,
  DatasetRow,
  Finding,
  Job,
  QuarantineEntry,
} from "../types";

type SortKey = "id" | "duration" | "cps" | "lang_prob";

const NUM_FMT = (v: number | null, digits = 2): string =>
  v === null || v === undefined ? "-" : v.toFixed(digits);

function findingChips(findings: Finding[]): {
  code: string;
  level: string;
  ids: string[];
}[] {
  const by = new Map<string, { level: string; ids: string[] }>();
  for (const f of findings) {
    if (!f.ids.length) continue; // dataset-level findings render as messages
    const cur = by.get(f.code);
    if (cur) {
      for (const id of f.ids) cur.ids.push(id);
    } else {
      by.set(f.code, { level: f.level, ids: [...f.ids] });
    }
  }
  return [...by.entries()].map(([code, v]) => ({ code, ...v }));
}

export function AuditPage({ name }: { name: string }) {
  const [data, setData] = useState<Dataset | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>("id");
  const [sortDir, setSortDir] = useState<1 | -1>(1);
  const [activeCode, setActiveCode] = useState<string | null>(null);
  const [editing, setEditing] = useState<{ id: string; draft: string } | null>(
    null,
  );
  const [editError, setEditError] = useState("");
  const [playing, setPlaying] = useState<string | null>(null);
  const [planText, setPlanText] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const planJobId = useRef<string | null>(null);

  useEffect(() => {
    let alive = true;
    get<Dataset>(`/projects/${name}/dataset`)
      .then((d) => {
        if (alive) setData(d);
      })
      .catch((e: Error) => {
        if (alive) setError(e);
      });
    get<Job[]>(`/projects/${name}/jobs`)
      .then((j) => {
        if (alive) setJobs(j);
      })
      .catch(() => {
        // the poll below retries
      });
    return () => {
      alive = false;
      if (audioRef.current) audioRef.current.pause();
    };
  }, [name]);

  usePoll(async () => {
    try {
      setJobs(await get<Job[]>(`/projects/${name}/jobs`));
    } catch {
      // transient; the next tick retries
    }
  }, 2500);

  // A finished plan run (clean without apply) enables apply-clean: its
  // plan text is what the user is agreeing to.
  useEffect(() => {
    const job = jobs.find(
      (j) =>
        j.kind === "clean" &&
        j.state === "succeeded" &&
        !j.params?.apply &&
        j.result?.plan,
    );
    if (job && job.result && job.id !== planJobId.current) {
      planJobId.current = job.id;
      setPlanText(String(job.result.plan));
      setMessage("clean plan ready below — review before applying");
    }
  }, [jobs]);

  // Latest succeeded validate/clean job carries the findings.
  const findings: Finding[] = useMemo(() => {
    const job = jobs.find(
      (j) =>
        (j.kind === "validate" || j.kind === "clean") &&
        j.state === "succeeded" &&
        j.result?.findings,
    );
    return job && job.result ? (job.result.findings as Finding[]) : [];
  }, [jobs]);

  const pendingJob = jobs.find(
    (j) =>
      (j.kind === "validate" || j.kind === "clean" || j.kind === "restore") &&
      (j.state === "running" || j.state === "queued"),
  );

  const chips = useMemo(() => findingChips(findings), [findings]);
  const datasetLevel = useMemo(
    () => findings.filter((f) => !f.ids.length),
    [findings],
  );

  async function runJob(kind: string, params: Record<string, unknown> = {}) {
    setMessage("");
    await post<Job>(`/projects/${name}/jobs`, { kind, params });
    setJobs(await get<Job[]>(`/projects/${name}/jobs`));
  }

  async function applyClean() {
    const n = data?.rows.filter((r) => !r.quarantined).length ?? 0;
    if (!window.confirm(
      `Apply clean? Clips with findings move to dataset/quarantine/ ` +
        `(nothing is deleted; restore brings them back).\n${n} rows in table.`,
    )) return;
    await runJob("clean", { apply: true });
    setMessage("clean apply queued");
  }

  async function restoreClip(clipId: string) {
    await runJob("restore", { ids: [clipId] });
    setMessage(`restore queued for ${clipId}`);
    setData(await get<Dataset>(`/projects/${name}/dataset`));
  }

  function sortOn(key: SortKey) {
    if (key === sortKey) setSortDir((d) => (d === 1 ? -1 : 1));
    else {
      setSortKey(key);
      setSortDir(1);
    }
  }

  const arrow = (key: SortKey) =>
    key === sortKey ? (sortDir === 1 ? " ▲" : " ▼") : "";

  const shown = useMemo(() => {
    if (!data) return [];
    let rows = data.rows;
    if (activeCode) {
      const chip = chips.find((c) => c.code === activeCode);
      const ids = new Set(chip?.ids ?? []);
      rows = rows.filter((r) => ids.has(r.id));
    }
    const val = (r: DatasetRow): number | string | null => r[sortKey];
    return [...rows].sort((a, b) => {
      const va = val(a);
      const vb = val(b);
      if (va === null) return 1; // nulls last, both directions
      if (vb === null) return -1;
      const c = typeof va === "string"
        ? va.localeCompare(vb as string)
        : (va as number) - (vb as number);
      return c * sortDir;
    });
  }, [data, activeCode, chips, sortKey, sortDir]);

  async function saveEdit() {
    if (!editing || !data) return;
    try {
      const updated = await patch<{ id: string; text: string }>(
        `/projects/${name}/dataset/${encodeURIComponent(editing.id)}`,
        { text: editing.draft },
      );
      setData({
        ...data,
        rows: data.rows.map((r) =>
          r.id === updated.id ? { ...r, text: updated.text } : r,
        ),
      });
      setEditing(null);
      setEditError("");
    } catch (e: unknown) {
      setEditError(String(e));
    }
  }

  function togglePlay(row: DatasetRow) {
    if (playing === row.id) {
      audioRef.current?.pause();
      setPlaying(null);
      return;
    }
    audioRef.current?.pause();
    const url =
      `/api/projects/${encodeURIComponent(name)}/files/dataset/wavs/` +
      encodeURIComponent(`${row.id}.wav`);
    const audio = new Audio(url);
    audioRef.current = audio;
    audio.onended = () => setPlaying(null);
    audio.play().catch(() => setPlaying(null));
    setPlaying(row.id);
  }

  if (error) return <p className="error">{String(error)}</p>;
  if (data === null) return <p className="muted">loading…</p>;

  const q = data.quarantine;

  return (
    <>
      <h1>audit — {name}</h1>
      <p className="muted">
        {data.rows.length} rows ·{" "}
        {data.rows.filter((r) => r.missing).length} missing WAVs ·{" "}
        {q.length} quarantine entries ·{" "}
        {datasetLevel.length ? `${datasetLevel.length} dataset-level findings` : "no findings read yet"}
      </p>

      <div className="row">
        <button onClick={() => void runJob("validate")}>run validation</button>
        <button onClick={() => void runJob("clean")}>plan clean</button>
        <button onClick={() => void applyClean()} disabled={planText === null}>
          apply clean
        </button>
        {pendingJob && (
          <span className="muted">
            {pendingJob.kind} {pendingJob.state}…
          </span>
        )}
        {message && <span className="ok">{message}</span>}
      </div>
      <p className="muted">
        findings come from the latest validate/clean run — run validation
        first if the chips below are empty
      </p>

      {datasetLevel.length > 0 && (
        <div>
          {datasetLevel.map((f, i) => (
            <p key={i} className={f.level === "error" ? "error" : "notice"}>
              [{f.code}] {f.message}
            </p>
          ))}
        </div>
      )}

      {chips.length > 0 && (
        <div className="row">
          <button
            className="chip"
            aria-pressed={activeCode === null}
            onClick={() => setActiveCode(null)}
          >
            all ({data.rows.length})
          </button>
          {chips.map((c) => (
            <button
              key={c.code}
              className={`chip ${c.level === "error" ? "chip-error" : "chip-warn"}`}
              aria-pressed={activeCode === c.code}
              onClick={() => setActiveCode(activeCode === c.code ? null : c.code)}
            >
              {c.code} ({c.ids.length})
            </button>
          ))}
        </div>
      )}

      {planText !== null && (
        <>
          <h2>clean plan (dry run)</h2>
          <pre className="log">{planText}</pre>
        </>
      )}

      {editError && <p className="error">{editError}</p>}

      <table>
        <thead>
          <tr>
            <th></th>
            <th className="sortable" onClick={() => sortOn("id")}>
              id{arrow("id")}
            </th>
            <th className="sortable" onClick={() => sortOn("duration")}>
              dur (s){arrow("duration")}
            </th>
            <th className="sortable" onClick={() => sortOn("cps")}>
              chars/sec{arrow("cps")}
            </th>
            <th className="sortable" onClick={() => sortOn("lang_prob")}>
              lang{arrow("lang_prob")}
            </th>
            <th>transcript (click to edit; Enter saves, Esc cancels)</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((r) => (
            <tr
              key={r.id}
              className={[
                playing === r.id ? "playing" : "",
                r.missing ? "row-missing" : "",
                r.quarantined ? "row-quarantined" : "",
              ].join(" ").trim() || undefined}
            >
              <td>
                <button
                  onClick={() => togglePlay(r)}
                  disabled={r.missing || r.quarantined}
                  title={r.quarantined ? "clip is quarantined" : "play"}
                >
                  {playing === r.id ? "■" : "▶"}
                </button>
              </td>
              <td>
                {r.id}
                {r.missing && (
                  <span className="error" title="no WAV in dataset/wavs">
                    {" "}missing
                  </span>
                )}
                {r.quarantined && <span className="muted"> quarantined</span>}
              </td>
              <td className="num">{NUM_FMT(r.duration)}</td>
              <td className="num">{NUM_FMT(r.cps, 1)}</td>
              <td className="num">
                {r.lang_prob === null ? "-" : r.lang_prob.toFixed(2)}
              </td>
              <td
                className="transcript"
                onClick={() =>
                  editing?.id !== r.id &&
                  !r.quarantined &&
                  setEditing({ id: r.id, draft: r.text })}
                style={{
                  cursor:
                    editing?.id === r.id || r.quarantined ? "default" : "text",
                }}
              >
                {editing?.id === r.id ? (
                  <>
                    <textarea
                      value={editing.draft}
                      onChange={(e) =>
                        setEditing({ id: r.id, draft: e.target.value })}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          void saveEdit();
                        } else if (e.key === "Escape") {
                          setEditing(null);
                          setEditError("");
                        }
                      }}
                      autoFocus
                    />
                    <button onClick={() => void saveEdit()}>save</button>
                    <button onClick={() => setEditing(null)}>cancel</button>
                  </>
                ) : (
                  <span>{r.text}</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>quarantine</h2>
      {q.length === 0 ? (
        <p className="muted">
          nothing quarantined (entries land here after an applied clean)
        </p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>clip</th>
              <th>reasons</th>
              <th>when</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {q.map((e: QuarantineEntry, i) => (
              <tr key={i}>
                <td>{e.clip_id}</td>
                <td>{e.reasons}</td>
                <td className="num">{e.timestamp}</td>
                <td>
                  <button onClick={() => void restoreClip(e.clip_id)}>
                    restore
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}
