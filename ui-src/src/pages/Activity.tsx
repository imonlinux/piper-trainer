import { useEffect, useMemo, useState } from "react";
import { ApiError, get, postEmpty } from "../api";
import { durationText, progressText, usePoll } from "../hooks";
import { kindText } from "../shell";
import type { Job } from "../types";

// Activity (workorder-04 A5): the full job history — where cancelled
// runs, failed fetches, and audition logs go to be found. This is the
// only page where raw job vocabulary (validate, clean-apply, ids) is
// allowed to surface; everywhere else speaks stage.

type Filter = "all" | "running" | "failed";

export function ActivityPage({ name }: { name: string }) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [loadError, setLoadError] = useState<Error | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [actionError, setActionError] = useState("");

  useEffect(() => {
    setJobs(null);
    setLoadError(null);
  }, [name]);

  usePoll(async () => {
    try {
      setJobs(await get<Job[]>(`/projects/${encodeURIComponent(name)}/jobs`));
      setLoadError(null);
    } catch (ex) {
      if (ex instanceof ApiError && ex.status === 404) setLoadError(ex);
      // anything else is a transient blip; keep the last list
    }
  }, 2000);

  const shown = useMemo(() => {
    const list = jobs ?? [];
    if (filter === "running")
      return list.filter((j) => j.state === "running" || j.state === "queued");
    if (filter === "failed") return list.filter((j) => j.state === "failed");
    return list;
  }, [jobs, filter]);

  async function cancel(id: string): Promise<void> {
    setActionError("");
    try {
      await postEmpty(`/jobs/${id}/cancel`);
    } catch (ex) {
      setActionError(String(ex));
    }
  }

  return (
    <>
      <h1>activity</h1>
      <p className="row">
        {(["all", "running", "failed"] as Filter[]).map((f) => (
          <button
            key={f}
            type="button"
            className="chip"
            aria-pressed={filter === f}
            onClick={() => setFilter(f)}
          >
            {f}
          </button>
        ))}
      </p>
      {loadError instanceof ApiError && loadError.status === 404 ? (
        <p className="error">no such project: {name}</p>
      ) : shown.length === 0 ? (
        <p className="muted">
          {jobs === null ? "loading…" : `no ${filter === "all" ? "" : filter} jobs`}
        </p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>id</th>
              <th>job</th>
              <th>state</th>
              <th>started</th>
              <th>duration</th>
              <th>progress</th>
              <th>error</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {shown.map((j) => {
              const active = j.state === "running" || j.state === "queued";
              return (
                <tr key={j.id}>
                  <td className="mono">{j.id}</td>
                  <td>
                    {kindText(j.kind)}{" "}
                    {kindText(j.kind) !== j.kind && (
                      <span className="muted">({j.kind})</span>
                    )}
                  </td>
                  <td className={`state-${j.state}`}>{j.state}</td>
                  <td>{j.started_at ?? "—"}</td>
                  <td className="mono">
                    {durationText(j.started_at, j.finished_at) || "—"}
                  </td>
                  <td>{progressText(j.progress)}</td>
                  <td className="error">{j.error ?? ""}</td>
                  <td>
                    {active && (
                      <button onClick={() => void cancel(j.id)}>cancel</button>
                    )}{" "}
                    <a href={`/api/jobs/${j.id}/log`}>log</a>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {actionError && <p className="error">{actionError}</p>}
    </>
  );
}
