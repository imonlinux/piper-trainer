import { useEffect, useState } from "react";
import { ApiError, del, get } from "../api";
import { usePoll } from "../hooks";
import {
  Panel,
  RecentJob,
  Stepper,
  stageHref,
  useStagesData,
} from "../shell";
import type { Job, ProjectDetail } from "../types";

// Overview (workorder-04 A5): the next card, the stage mini-map, the
// Output / Started-from settings split, the last five jobs, and the
// danger zone. Everything else moved out — sources live on their own
// page, transcription on its own page, the job table on Activity.

export function ProjectPage({ name }: { name: string }) {
  const [p, setP] = useState<ProjectDetail | null>(null);
  const [loadError, setLoadError] = useState<Error | null>(null);
  const [actionError, setActionError] = useState("");
  // stages come from the shell's one poll — no second fetch here
  const stages = useStagesData();

  useEffect(() => {
    setP(null);
    setLoadError(null);
    get<ProjectDetail>(`/projects/${encodeURIComponent(name)}`)
      .then(setP)
      .catch((e: Error) => setLoadError(e));
  }, [name]);

  // Only the recent-activity list stays live here; the shell owns the
  // stage poll. A 404 surfaces (and pauses this poll) instead of spin.
  usePoll(async () => {
    try {
      const jobs = await get<Job[]>(`/projects/${encodeURIComponent(name)}/jobs`);
      setP((prev) => (prev ? { ...prev, jobs } : prev));
    } catch (ex) {
      if (ex instanceof ApiError && ex.status === 404) setLoadError(ex);
    }
  }, 2000, loadError !== null);

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

  const cfg = p?.config ?? {};
  const recent = (p?.jobs ?? []).slice(0, 5);

  return (
    <>
      {loadError instanceof ApiError && loadError.status === 404 ? (
        <>
          <h1>{name}</h1>
          <p className="error">no such project: {name}</p>
        </>
      ) : loadError ? (
        <p className="error">{String(loadError)}</p>
      ) : (
        <>
          <h1>{p?.name ?? name}</h1>
          {p && (
            <p className="muted">
              {p.clips} clips · {p.minutes ?? "?"} min ·{" "}
              {p.dataset.rows} dataset rows
              {p.dataset.malformed_lines > 0
                ? `, ${p.dataset.malformed_lines} malformed`
                : ""}
            </p>
          )}

          <Panel
            title="next"
            help="The one thing to do now, derived from where the project actually is. The stepper below is the whole pipeline; every stage is always clickable."
          >
            {stages ? (
              <div className="next-card">
                <p className="next-why">{stages.next.why}</p>
                <a className="button-primary" href={stageHref(stages.next.stage, name)}>
                  go to {stages.next.stage}
                </a>
              </div>
            ) : (
              <p className="muted">loading…</p>
            )}
          </Panel>
          <Stepper name={name} />

          <Panel
            title="settings"
            help="Output describes what you get; Started-from describes where the voice's accent comes from. The base voice never changes after export."
          >
            <div className="settings-cols">
              <div>
                <h3>output</h3>
                <p>
                  voice name: {stages?.project.voice ?? "—"}
                  <br />
                  exported voices: {p?.voices.length ?? 0}
                  <br />
                  location: out/
                </p>
              </div>
              <div>
                <h3>started from</h3>
                <p>
                  {cfg.catalog_path
                    ? `base voice: ${cfg.catalog_path}`
                    : "no base voice selected — training runs from scratch"}
                  {cfg.tier ? (
                    <>
                      <br />
                      tier: {cfg.tier}
                    </>
                  ) : null}
                </p>
              </div>
            </div>
          </Panel>

          <Panel
            title="recent activity"
            help="The five most recent jobs. The full history — cancelled runs, failed fetches, audition logs — lives on the activity page."
          >
            {recent.length === 0 ? (
              <p className="muted">no jobs yet</p>
            ) : (
              <ul className="recent">
                {recent.map((j) => (
                  <RecentJob key={j.id} job={j} />
                ))}
              </ul>
            )}
            <p>
              <a href={`#/project/${name}/activity`}>all activity</a>
            </p>
          </Panel>
        </>
      )}

      <Panel title="danger">
        <button onClick={() => void doDelete()}>delete (moves to .trash)</button>
        {actionError && <span className="error"> {actionError}</span>}
      </Panel>
    </>
  );
}
