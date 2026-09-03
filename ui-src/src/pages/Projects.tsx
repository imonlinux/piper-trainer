import { useEffect, useState } from "react";
import { get } from "../api";
import type { ProjectSummary } from "../types";

export function ProjectsPage() {
  const [rows, setRows] = useState<ProjectSummary[] | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let alive = true;
    get<ProjectSummary[]>("/projects")
      .then((r) => {
        if (alive) setRows(r);
      })
      .catch((e: Error) => {
        if (alive) setError(e);
      });
    return () => {
      alive = false;
    };
  }, []);

  if (error) return <p className="error">{String(error)}</p>;
  if (rows === null) return <p className="muted">loading…</p>;

  return (
    <>
      <h1>Projects</h1>
      {rows.length === 0 ? (
        <p className="muted">no projects yet</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>name</th>
              <th>clips</th>
              <th>minutes</th>
              <th>tiers trained</th>
              <th>last job</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.name}>
                <td>
                  <a href={`#/project/${p.name}`}>{p.name}</a>
                </td>
                <td className="num">{p.clips}</td>
                <td className="num">{p.minutes ?? "-"}</td>
                <td>{p.tiers_trained.join(", ") || "-"}</td>
                <td>
                  {p.last_job
                    ? `${p.last_job.kind} (${p.last_job.state})`
                    : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p>
        <a href="#/new">create a project</a>
      </p>
    </>
  );
}
