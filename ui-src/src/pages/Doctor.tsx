import { useEffect, useState } from "react";
import { get } from "../api";
import type { Doctor } from "../types";

const LABEL: Record<string, string> = {
  ok: "ok",
  error: "FAIL",
  info: "info",
};

export function DoctorPage() {
  const [d, setD] = useState<Doctor | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let alive = true;
    get<Doctor>("/doctor")
      .then((r) => {
        if (alive) setD(r);
      })
      .catch((e: Error) => {
        if (alive) setError(e);
      });
    return () => {
      alive = false;
    };
  }, []);

  if (error) return <p className="error">{String(error)}</p>;
  if (d === null) return <p className="muted">loading…</p>;

  return (
    <>
      <h1>Doctor</h1>
      <p>
        {d.ok ? (
          <span className="ok">environment OK</span>
        ) : (
          <span className="error">problems found</span>
        )}
      </p>
      <p className="muted">
        transcription devices: {d.transcribe_devices.join(", ")}
      </p>
      <table>
        <tbody>
          {d.checks.map((c, i) => (
            <tr key={i}>
              <td className={c.status}>{LABEL[c.status] ?? c.status}</td>
              <td>{c.message}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
