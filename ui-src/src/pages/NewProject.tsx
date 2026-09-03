import { useEffect, useState } from "react";
import { get, post } from "../api";
import type { Catalog, ProjectSummary } from "../types";

// Project creation with the checkpoint picker (§3): language -> locale ->
// voice -> quality cascades over the HF catalog, live with snapshot
// fallback (decision §8.6). The derived settings written here are what
// every later screen depends on.

export function NewProjectPage() {
  const [cat, setCat] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [espeak, setEspeak] = useState("");
  const [lang, setLang] = useState("");
  const [locale, setLocale] = useState("");
  const [voice, setVoice] = useState("");
  const [quality, setQuality] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let alive = true;
    get<Catalog>("/checkpoints/catalog")
      .then((c) => {
        if (!alive) return;
        setCat(c);
        const langs = Object.keys(c.languages).sort();
        if (langs.length > 0) {
          setLang(langs[0]);
          const locales = Object.keys(c.languages[langs[0]]).sort();
          if (locales.length > 0) {
            setLocale(locales[0]);
            const voices = Object.keys(c.languages[langs[0]][locales[0]]).sort();
            if (voices.length > 0) {
              setVoice(voices[0]);
              setQuality(c.languages[langs[0]][locales[0]][voices[0]][0] ?? "");
            }
          }
        }
      })
      .catch((e: Error) => {
        if (alive) setError(String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  if (error) {
    return (
      <>
        <h1>New project</h1>
        <p className="error">{error}</p>
        <p>
          <a href="#/new">retry</a>
        </p>
      </>
    );
  }
  if (cat === null) return <p className="muted">loading catalog…</p>;

  const locales = Object.keys(cat.languages[lang] ?? {}).sort();
  const voices = Object.keys(cat.languages[lang]?.[locale] ?? {}).sort();
  const qualities = cat.languages[lang]?.[locale]?.[voice] ?? [];
  const c = cat; // narrowed alias for the cascade helpers below

  // Cascade setters must read the catalog with explicit values: setState
  // is async, so reading `lang`/`locale` right after setting them sees
  // the previous render's values.
  function qualitiesFor(l: string, loc: string, v: string): string[] {
    return c.languages[l]?.[loc]?.[v] ?? [];
  }
  function pickVoiceOf(l: string, loc: string): void {
    const vs = Object.keys(c.languages[l]?.[loc] ?? {}).sort();
    const v = vs[0] ?? "";
    setVoice(v);
    setQuality(qualitiesFor(l, loc, v)[0] ?? "");
  }
  function onLang(v: string): void {
    setLang(v);
    const locs = Object.keys(c.languages[v] ?? {}).sort();
    const loc = locs[0] ?? "";
    setLocale(loc);
    pickVoiceOf(v, loc);
  }
  function onLocale(v: string): void {
    setLocale(v);
    pickVoiceOf(lang, v);
  }
  function onVoice(v: string): void {
    setVoice(v);
    setQuality(qualitiesFor(lang, locale, v)[0] ?? "");
  }

  async function submit(e: React.FormEvent): Promise<void> {
    e.preventDefault();
    setError(null);
    // locale is e.g. en_GB; the espeak voice for it is en-gb
    const derived = locale.replace("_", "-").toLowerCase();
    const espeakVoice = espeak || derived;
    setSubmitting(true);
    try {
      const p = await post<ProjectSummary>("/projects", {
        name,
        espeak_voice: espeakVoice,
        tier: quality,
        catalog_path: [lang, locale, voice, quality].join("/"),
      });
      location.hash = `#/project/${p.name}`;
    } catch (ex) {
      setError(String(ex));
    } finally {
      setSubmitting(false);
    }
  }

  async function retryCatalog(): Promise<void> {
    // A plain reload would serve the same cached snapshot for the
    // snapshot TTL; refresh=1 skips the server cache before redraw
    // (review finding 10).
    await get("/checkpoints/catalog?refresh=1").catch(() => null);
    location.reload();
  }

  return (
    <>
      <h1>New project</h1>
      {cat.source === "snapshot" ? (
        <p className="notice">
          showing cached catalog from {cat.generated} (HF unreachable) —{" "}
          <a
            href="#/new"
            onClick={(e) => {
              e.preventDefault();
              void retryCatalog();
            }}
          >
            retry
          </a>
        </p>
      ) : (
        <p className="muted">catalog: live</p>
      )}
      <form
        onSubmit={(e) => {
          void submit(e);
        }}
      >
        <label>name</label>
        <input
          type="text"
          placeholder="hal-9000"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <h2>Base checkpoint</h2>
        <label>language</label>
        <select value={lang} onChange={(e) => onLang(e.target.value)}>
          {Object.keys(cat.languages).sort().map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
        <label>locale</label>
        <select value={locale} onChange={(e) => onLocale(e.target.value)}>
          {locales.map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
        <label>voice</label>
        <select value={voice} onChange={(e) => onVoice(e.target.value)}>
          {voices.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
        <label>quality (= tier)</label>
        <select value={quality} onChange={(e) => setQuality(e.target.value)}>
          {qualities.map((q) => (
            <option key={q} value={q}>
              {q}
            </option>
          ))}
        </select>
        <label>espeak voice (editable)</label>
        <input
          type="text"
          placeholder="en-us"
          value={espeak}
          onChange={(e) => setEspeak(e.target.value)}
        />
        <p>
          <button type="submit" disabled={submitting}>
            create
          </button>
        </p>
      </form>
      {error && <p className="error">{error}</p>}
    </>
  );
}
