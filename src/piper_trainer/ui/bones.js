/* Bones UI: plain hash-routed pages over the /api surface.
   Polling for lists, websocket for job logs. No framework by design. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const main = () => $("#main");

async function api(path, opts) {
  const res = await fetch(`/api${path}`, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "onclick") node.addEventListener("click", v);
    else if (k === "onchange") node.addEventListener("change", v);
    else if (k === "onsubmit") node.addEventListener("submit", v);
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    node.append(c.nodeType ? c : document.createTextNode(c));
  }
  return node;
}

// ------------------------------------------------------------------ pages

async function projectsPage() {
  const rows = await api("/projects");
  const tbl = el("table", {},
    el("tr", {}, ...["name", "clips", "minutes", "tiers trained", "last job"]
      .map(h => el("th", {}, h))));
  for (const p of rows) {
    tbl.append(el("tr", {},
      el("td", {}, el("a", { href: `#/project/${p.name}` }, p.name)),
      el("td", { class: "num" }, String(p.clips)),
      el("td", { class: "num" }, p.minutes == null ? "-" : String(p.minutes)),
      el("td", {}, p.tiers_trained.join(", ") || "-"),
      el("td", {}, p.last_job
        ? `${p.last_job.kind} (${p.last_job.state})` : "-")));
  }
  main().replaceChildren(
    el("h1", {}, "Projects"),
    rows.length ? tbl : el("p", { class: "muted" }, "no projects yet"),
    el("p", {}, el("a", { href: "#/new" }, "create a project")));
}

async function newProjectPage() {
  const cat = await api("/checkpoints/catalog");
  const notice = cat.source === "snapshot"
    ? el("p", { class: "notice" },
        `showing cached catalog from ${cat.generated} (HF unreachable) — `,
        el("a", { href: "#/new", onclick: () => setTimeout(() => location.reload(), 0) }, "retry"))
    : el("p", { class: "muted" }, "catalog: live");

  const langs = Object.keys(cat.languages).sort();
  const langSel = el("select", { onchange: fillLocales });
  const locSel = el("select", { onchange: fillVoices });
  const voiceSel = el("select", { onchange: fillQuality });
  const qualSel = el("select", {});

  function opts(sel, values) {
    sel.replaceChildren(...values.map(v => el("option", { value: v }, v)));
  }
  function locales() { return cat.languages[langSel.value] || {}; }
  function voices() { return locales()[locSel.value] || {}; }
  function qualities() { return voices()[voiceSel.value] || []; }

  function fillLocales() { opts(locSel, Object.keys(locales()).sort()); fillVoices(); }
  function fillVoices() { opts(voiceSel, Object.keys(voices()).sort()); fillQuality(); }
  function fillQuality() { opts(qualSel, qualities()); }

  opts(langSel, langs);
  fillLocales();

  const espeak = el("input", { type: "text", placeholder: "en-us" });
  const name = el("input", { type: "text", placeholder: "hal-9000" });
  const err = el("p", { class: "error" });

  async function submit(e) {
    e.preventDefault();
    err.textContent = "";
    const locale = locSel.value;                 // e.g. en_GB -> en-gb
    const derived = locale.replace("_", "-").toLowerCase();
    if (!espeak.value) espeak.value = derived;
    try {
      const p = await api("/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name.value,
          espeak_voice: espeak.value || derived,
          tier: qualSel.value,
          catalog_path: [langSel.value, locSel.value, voiceSel.value,
                         qualSel.value].join("/"),
        }),
      });
      location.hash = `#/project/${p.name}`;
    } catch (ex) { err.textContent = String(ex); }
  }

  main().replaceChildren(
    el("h1", {}, "New project"),
    notice,
    el("form", { onsubmit: submit },
      el("label", {}, "name"), name,
      el("h2", {}, "Base checkpoint"),
      el("label", {}, "language"), langSel,
      el("label", {}, "locale"), locSel,
      el("label", {}, "voice"), voiceSel,
      el("label", {}, "quality (= tier)"), qualSel,
      el("label", {}, "espeak voice (editable)"), espeak,
      el("p", {}, el("button", { type: "submit" }, "create"))),
    err);
}

async function doctorPage() {
  const d = await api("/doctor");
  main().replaceChildren(
    el("h1", {}, "Doctor"),
    el("p", {}, d.ok
      ? el("span", { class: "ok" }, "environment OK")
      : el("span", { class: "error" }, "problems found")),
    el("p", { class: "muted" },
      `transcription devices: ${d.transcribe_devices.join(", ")}`),
    el("table", {}, ...d.checks.map(c => el("tr", {},
      el("td", { class: c.status }, { ok: "ok", error: "FAIL", info: "info" }[c.status]),
      el("td", {}, c.message)))));
}

async function projectPage(name) {
  const p = await api(`/projects/${name}`);
  const cells = Object.entries(p.directories).map(([k, v]) =>
    el("div", { class: "cell" }, k, el("b", {}, String(v))));

  const upload = el("input", { type: "file", multiple: true });
  const uploadErr = el("span", { class: "error" });
  async function doUpload(e) {
    e.preventDefault();
    uploadErr.textContent = "";
    const fd = new FormData();
    for (const f of upload.files) fd.append("files", f);
    try {
      await api(`/projects/${name}/ingest`, { method: "POST", body: fd });
      setTimeout(() => render(), 300);
    } catch (ex) { uploadErr.textContent = String(ex); }
  }

  async function runJob(kind, params = {}) {
    try {
      await api(`/projects/${name}/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, params }),
      });
      setTimeout(() => render(), 300);
    } catch (ex) { uploadErr.textContent = String(ex); }
  }

  const srcRows = await api(`/projects/${name}/sources`).catch(() => []);
  const srcTbl = el("table", {},
    el("tr", {}, ...["name", "codec", "rate", "ch", "duration"].map(h => el("th", {}, h))),
    ...srcRows.map(s => el("tr", {},
      el("td", {}, s.name),
      el("td", {}, s.codec || "?"),
      el("td", { class: "num" }, s.sample_rate || "?"),
      el("td", { class: "num" }, s.channels || "?"),
      el("td", { class: "num" }, s.duration || "?"))));

  const cfg = p.config || {};
  const log = el("pre", { class: "log" }, "");
  let ws = null;
  let wsJob = null;

  function watch(jobId) {
    if (ws) { ws.close(); ws = null; }
    wsJob = jobId;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/api/jobs/${jobId}/stream`);
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "log_reset") log.textContent = msg.text;
      else if (msg.type === "log") {
        log.textContent += msg.line + "\n";
        log.scrollTop = log.scrollHeight;
      } else if (msg.type === "state") {
        renderJobs([msg.job], false);
        const prog = msg.job.progress;
        progSpan.textContent = prog && prog.total
          ? `epoch ${prog.current}/${prog.total}` : "";
      }
    };
    ws.onclose = () => { if (wsJob === jobId) ws = null; };
  }

  const jobsWrap = el("div", {});
  const progSpan = el("span", { class: "muted" });

  function renderJobs(jobs, clickable = true) {
    const tbl = el("table", {},
      el("tr", {}, ...["id", "kind", "state", "progress", "error", ""]
        .map(h => el("th", {}, h))));
    for (const j of jobs) {
      const prog = j.progress;
      tbl.append(el("tr", {},
        el("td", {}, j.id),
        el("td", {}, j.kind),
        el("td", { class: `state-${j.state}` }, j.state),
        el("td", {}, prog && prog.total
          ? `${prog.unit || ""} ${prog.current ?? "-"}/${prog.total}` : ""),
        el("td", {}, j.error || ""),
        el("td", {},
          (j.state === "running" || j.state === "queued")
            ? el("form", { class: "inline", onsubmit: async (e) => {
                e.preventDefault();
                await api(`/jobs/${j.id}/cancel`, { method: "POST" });
                setTimeout(() => render(), 200);
              } }, el("button", {}, "cancel"))
            : null,
          j.state === "running" && clickable
            ? el("button", { onclick: () => watch(j.id) }, "log") : null,
          j.state === "queued"
            ? el("button", { onclick: async () => {
                await api(`/jobs/${j.id}/start`, { method: "POST" });
                setTimeout(() => render(), 200);
              } }, "start") : null)));
    }
    jobsWrap.replaceChildren(
      jobs.length ? tbl : el("p", { class: "muted" }, "no jobs yet"));
  }

  const deleteErr = el("span", { class: "error" });
  async function doDelete(e) {
    e.preventDefault();
    if (!confirm(`Move project "${name}" to .trash? Nothing is destroyed.`)) return;
    try {
      await api(`/projects/${name}`, { method: "DELETE" });
      location.hash = "#/projects";
    } catch (ex) { deleteErr.textContent = String(ex); }
  }

  // Design doc §1.4: never expose the absolute max_epochs ceiling. A tier
  // with no checkpoint asks for epochs (submitted as max_epochs); a trained
  // tier asks for "N more" (submitted as add_epochs) so a resume can never
  // set the ceiling below the restored epoch counter and exit instantly.
  const tier = cfg.tier || "medium";
  const trained = p.tiers_trained.includes(tier);
  const epochs = el("input", { type: "number", min: "1",
                               value: trained ? "1000" : "4000",
                               style: "width:6em" });
  async function doTrain() {
    const n = parseInt(epochs.value, 10);
    if (!n || n < 1) { uploadErr.textContent = "epochs must be a number >= 1"; return; }
    await runJob("train", trained ? { add_epochs: n } : { max_epochs: n });
  }

  main().replaceChildren(
    el("h1", {}, p.name),
    el("p", { class: "muted" },
      `${p.clips} clips, ${p.minutes ?? "?"} min, tiers trained: ${p.tiers_trained.join(", ") || "none"}`,
      " ", progSpan),
    el("h2", {}, "Directories"), el("div", { class: "grid" }, ...cells),
    el("h2", {}, "Dataset"),
    el("p", {}, `${p.dataset.rows} rows, ${p.dataset.malformed_lines} malformed lines, `,
      `line endings: ${p.dataset.line_endings ?? "-"}`,
      cfg.espeak_voice ? `, espeak voice: ${cfg.espeak_voice}` : "",
      cfg.tier ? `, tier: ${cfg.tier}` : ""),
    el("h2", {}, "Sources (raw/)"),
    srcRows.length ? srcTbl : el("p", { class: "muted" }, "no source recordings"),
    el("form", { onsubmit: doUpload, class: "row" }, upload,
      el("button", { type: "submit" }, "upload"), uploadErr),
    el("h2", {}, "Jobs"),
    el("div", { class: "row" },
      el("button", { onclick: () => runJob("prepare") }, "run prepare"),
      el("button", { onclick: () => runJob("transcribe") }, "run transcribe"),
      el("label", { class: "inline" }, trained ? "epochs (more)" : "epochs",
        epochs),
      el("button", { onclick: doTrain }, trained ? "train N more" : "run train"),
      el("button", { onclick: () => runJob("export") }, "run export")),
    jobsWrap,
    log,
    el("h2", {}, "Danger"),
    el("form", { onsubmit: doDelete, class: "row" },
      el("button", {}, "delete (moves to .trash)"), deleteErr));

  function render() { projectPage(name); }
  renderJobs(p.jobs);
  const running = p.jobs.find(j => j.state === "running");
  if (running) watch(running.id);

  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    if (ws) return; // the websocket already carries live state
    try {
      const jobs = await api(`/projects/${name}/jobs`);
      renderJobs(jobs.slice(0, 10));
    } catch {}
  }, 2000);
}

let pollTimer = null;

// ------------------------------------------------------------------ router

async function route() {
  clearInterval(pollTimer);
  const h = location.hash || "#/projects";
  try {
    if (h === "#/projects") await projectsPage();
    else if (h === "#/new") await newProjectPage();
    else if (h === "#/doctor") await doctorPage();
    else if (h.startsWith("#/project/")) await projectPage(h.slice(10));
    else await projectsPage();
  } catch (ex) {
    main().replaceChildren(el("p", { class: "error" }, String(ex)));
  }
}

(async function init() {
  try {
    const h = await api("/health");
    $("#health").textContent = `v${h.version}`;
  } catch {
    $("#health").textContent = "API unreachable";
  }
  window.addEventListener("hashchange", route);
  route();
})();
