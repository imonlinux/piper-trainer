// One fetch wrapper for the whole /api surface. Errors keep the HTTP
// status so pages can branch on 404 (a project deleted out from under
// an open tab must say so, not spin forever).

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, detail: string) {
    super(`${status}: ${detail}`);
    this.status = status;
  }
}

export async function api<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      // body was not JSON; statusText is the best we have
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const get = <T,>(path: string): Promise<T> => api<T>(path);

export function post<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const postEmpty = <T,>(path: string): Promise<T> =>
  api<T>(path, { method: "POST" });

export const del = <T,>(path: string): Promise<T> =>
  api<T>(path, { method: "DELETE" });

// Multipart (upload ingest): never set Content-Type by hand — the
// browser must add the boundary.
export const upload = <T,>(path: string, form: FormData): Promise<T> =>
  api<T>(path, { method: "POST", body: form });

// File URL for playable artifacts (previews, clips).
export const fileUrl = (project: string, dir: string, name: string): string =>
  `/api/projects/${project}/files/${dir}/${encodeURIComponent(name)}`;

export const jobLogUrl = (jobId: string): string => `/api/jobs/${jobId}/log`;
