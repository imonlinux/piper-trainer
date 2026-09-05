import { useEffect, useState } from "react";
import { get, post, upload } from "../api";
import { usePoll } from "../hooks";
import { Consequence } from "../shell";
import type { Job, SourceInfo } from "../types";

// Sources (workorder-04 A5): the ingest form and the raw/ list, carved
// out of the old overview with behavior unchanged. Fetch caps and the
// watchdog are server-side; the page reports their errors verbatim.

export function SourcesPage({ name }: { name: string }) {
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [sourceMsg, setSourceMsg] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [actionError, setActionError] = useState("");
  const [ingestMode, setIngestMode] = useState<
    "upload" | "url" | "media-site" | "hf"
  >("upload");
  // The ingest runs as a job; reload the list once it settles.
  const [ingestId, setIngestId] = useState<string | null>(null);

  function loadSources(): void {
    get<SourceInfo[]>(`/projects/${encodeURIComponent(name)}/sources`)
      .then((s) => {
        setSources(s);
        setSel((prev) => {
          const next = new Set(
            [...prev].filter((n) => s.some((x) => x.name === n)),
          );
          return next.size === prev.size ? prev : next;
        });
      })
      .catch(() => setSources([]));
  }

  useEffect(() => {
    loadSources();
    setIngestId(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name]);

  // Watch the tracked ingest job; when it lands, refresh the list.
  usePoll(async () => {
    if (ingestId === null) return;
    try {
      const jobs = await get<Job[]>(
        `/projects/${encodeURIComponent(name)}/jobs`,
      );
      const ing = jobs.find((j) => j.id === ingestId);
      if (!ing) return;
      if (
        ["succeeded", "failed", "canceled", "interrupted"].includes(ing.state)
      ) {
        setIngestId(null);
        loadSources();
      }
    } catch {
      // transient; the next tick retries
    }
  }, 2000);

  async function doIngest(e: React.FormEvent<HTMLFormElement>): Promise<void> {
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
    } catch (ex) {
      setUploadError(String(ex));
    }
  }

  async function doDeleteSources(): Promise<void> {
    const names = [...sel];
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
      setSel(new Set());
      setSourceMsg(
        `moved ${res.moved.length} to .trash` +
          (res.missing.length
            ? ` · not found: ${res.missing.join(", ")}`
            : ""),
      );
      loadSources();
    } catch (ex) {
      setActionError(String(ex));
    }
  }

  return (
    <>
      <h1>sources</h1>
      <p className="muted">
        raw recordings land in raw/ — prepare turns them into clips
      </p>

      <h2>add sources</h2>
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
            <label
              className="inline"
              title="download a whole playlist instead of one video"
            >
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

      <h2>in raw/ ({sources.length})</h2>
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
                    sources.length > 0 && sel.size === sources.length
                  }
                  onChange={(e) =>
                    setSel(
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
                    checked={sel.has(s.name)}
                    onChange={(e) =>
                      setSel((prev) => {
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
      <Consequence>
        deleting removes the source file; prepared clips already made from
        it stay
      </Consequence>
      <p className="row">
        <button
          disabled={sel.size === 0}
          onClick={() => void doDeleteSources()}
        >
          delete selected ({sel.size})
        </button>
        {sourceMsg && <span className="muted">{sourceMsg}</span>}
        {actionError && <span className="error">{actionError}</span>}
      </p>
    </>
  );
}
