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
    else if (v === false || v == null) {}   // boolean attribute absent
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

// ------------------------------------------------------- prepare tuner (§6.2)

// The screen that justifies the UI: waveform of one source with detected
// regions overlaid, sliders for the VAD parameters, each run producing a
// sweep entry the user can compare and promote to a full prepare run.

function drawWave(cv, data, regions) {
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#f8f8f8";
  ctx.fillRect(0, 0, W, H);
  if (!data) return;
  const dur = data.duration || 1;
  const n = data.peaks.length;
  const bw = W / n;
  ctx.fillStyle = "#888";
  for (let i = 0; i < n; i++) {
    const h = Math.max(1, data.peaks[i] * (H - 10));
    ctx.fillRect(i * bw, (H - h) / 2, Math.max(1, bw), h);
  }
  for (const r of regions) {            // detected speech regions
    const x0 = (r.start / dur) * W, x1 = (r.end / dur) * W;
    ctx.fillStyle = "rgba(35, 134, 54, 0.22)";
    ctx.fillRect(x0, 0, x1 - x0, H);
    ctx.strokeStyle = "#22863a";
    ctx.strokeRect(x0 + 0.5, 0.5, x1 - x0 - 1, H - 1);
  }
}

async function preparePage(name) {
  const srcs = await api(`/projects/${name}/sources`);
  const err = el("p", { class: "error" });

  // -- channel picker (a blind downmix can halve SNR on a bad channel)
  let channel = "downmix";
  const chRadios = ["downmix", "left", "right"].map(c =>
    el("label", { class: "inline" },
      el("input", { type: "radio", name: "chan", value: c,
                    checked: c === "downmix",
                    onchange: () => { channel = c; loadPeaks(); } }), c));

  // -- source select
  const srcSel = el("select", {
    onchange: () => loadPeaks(),
  }, ...srcs.map(s => {
    const o = el("option", { value: s.name }, s.name);
    return o;
  }));

  // -- VAD parameter sliders
  const SLIDERS = [
    ["energy_threshold", "energy threshold", 20, 90, 1, 55],
    ["min_dur", "min duration (s)", 0.5, 5, 0.1, 1.5],
    ["max_dur", "max duration (s)", 2, 20, 0.5, 10],
    ["max_silence", "max silence (s)", 0.1, 1.5, 0.05, 0.4],
    ["pad", "pad (s)", 0, 0.5, 0.01, 0.15],
  ];
  const sliders = {};
  const sliderRows = SLIDERS.map(([key, label, min, max, step, dflt]) => {
    const val = el("span", { class: "muted" }, String(dflt));
    const input = el("input", {
      type: "range", min: String(min), max: String(max), step: String(step),
      value: String(dflt), style: "width:14em",
      oninput: () => { val.textContent = input.value; },
    });
    sliders[key] = input;
    return el("label", { class: "inline" }, label, input, val);
  });
  const dnChk = el("input", { type: "checkbox", checked: true });

  function tunerParams() {
    const p = { source: srcSel.value, channel,
                denoise: dnChk.checked };
    for (const [key] of SLIDERS) p[key] = parseFloat(sliders[key].value);
    return p;
  }

  // -- waveform
  const cv = el("canvas", { width: "1000", height: "160",
                            style: "width:100%" });
  let peaksData = null;
  let selected = null;   // the sweep entry whose regions/clips are shown

  async function loadPeaks() {
    selected = null;
    peaksData = await api(`/projects/${name}/sources/${encodeURIComponent(srcSel.value)}/peaks?channel=${channel}&buckets=2000`).catch(() => null);
    draw();
  }
  function draw() {
    drawWave(cv, peaksData, selected && selected.result.clips
             ? selected.result.clips : []);
  }

  // -- preview actions
  async function runSegmentPreview() {
    err.textContent = "";
    try {
      await api(`/projects/${name}/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stage: "segment", params: tunerParams() }),
      });
      msg.textContent = "segment preview queued\u2026";
    } catch (ex) { err.textContent = String(ex); }
  }
  async function runDenoisePreview() {
    err.textContent = "";
    try {
      await api(`/projects/${name}/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stage: "denoise",
                               params: { source: srcSel.value, channel,
                                         seconds: 25 } }),
      });
      msg.textContent = "denoise preview queued\u2026";
    } catch (ex) { err.textContent = String(ex); }
  }
  const msg = el("span", { class: "muted" });

  // -- sweep (grouped by stage, newest first)
  const segWrap = el("div", {});
  const dnWrap = el("div", {});
  const clipEmpty = () =>
    el("p", { class: "muted" }, "click a preview id below to inspect its clips here");
  const clipWrap = el("div", {}, clipEmpty());
  const fileUrl = (row, f) =>
    `/api/projects/${name}/files/${row.dir}/${encodeURIComponent(f)}`;

  function selectPreview(row) {
    selected = row;
    const clips = (row.result.audio || []).map(f =>
      el("div", { class: "cell" }, f,
        el("audio", { controls: "", src: fileUrl(row, f), style: "width:100%" })));
    const head = row.stage === "denoise"
      ? `denoise A/B from ${row.id} (original vs denoised, ${row.result.seconds ?? "?"}s)`
      : `clips from ${row.id} (${row.result.clip_count ?? "?"} total, `
        + `first ${clips.length} playable)`;
    // A segment preview that found nothing used to render a link with no
    // feedback at all — say which dials to move instead.
    const empty = row.stage === "segment" && row.result.clip_count === 0
      ? el("p", { class: "notice" },
          "0 clips: the splitter rejected every region. Lower the energy "
        + "threshold, lower the min duration for short utterances, or pick "
        + "left/right when only one channel has audio (downmix averages in "
        + "the silent side). Then preview segment again.")
      : null;
    clipWrap.replaceChildren(
      el("h3", {}, head),
      empty || el("div", { class: "grid" }, ...clips));
    draw();   // players first: a draw failure must never eat them
  }

  async function promote(row) {
    err.textContent = "";
    if (!confirm(`Run the full prepare with these parameters?\n${JSON.stringify(row.params)}`)) return;
    try {
      await api(`/projects/${name}/previews/${row.id}/promote`,
                { method: "POST" });
      msg.textContent = "full prepare queued from promoted parameters";
    } catch (ex) { err.textContent = String(ex); }
  }

  function sweepTable(rows, kind) {
    const tbl = el("table", {},
      el("tr", {}, ...["id", "params", kind === "segment" ? "clips" : "seconds",
                       "histogram", ""].map(h => el("th", {}, h))));
    for (const row of rows) {
      const p = row.params || {};
      const sum = `${p.source || "?"} \u00b7 ${p.channel || "downmix"}`
        + (kind === "segment"
           ? ` \u00b7 energy ${p.energy_threshold} \u00b7 dn ${p.denoise !== false}`
           : ` \u00b7 ${p.seconds}s`);
      const hist = (row.result.histogram || []).map(h =>
        `${h.from}-${h.to}s: ${h.count}`).join(", ");
      tbl.append(el("tr", {},
        el("td", {},
          el("a", { href: "#", onclick: (e) => {
            e.preventDefault(); selectPreview(row);
          } }, row.id)),
        el("td", {}, sum),
        el("td", { class: "num" }, String(row.result[kind === "segment"
          ? "clip_count" : "seconds"] ?? "")),
        el("td", {}, kind === "segment" ? hist : ""),
        el("td", {},
          kind === "segment"
            ? el("button", { onclick: () => promote(row) }, "promote")
            : null)));
    }
    return tbl;
  }

  let sweepFp = "";   // row ids seen last; rows are immutable once listed

  async function loadSweep() {
    const rows = await api(`/projects/${name}/previews`).catch(() => []);
    const seg = rows.filter(r => r.stage === "segment");
    const dn = rows.filter(r => r.stage === "denoise");
    // Rebuilding this DOM mid-playback kills the denoise A/B <audio> nodes,
    // so leave the sweep untouched unless the set of previews changed.
    const fp = seg.map(r => r.id).join() + "|" + dn.map(r => r.id).join();
    if (fp === sweepFp) return;
    sweepFp = fp;
    segWrap.replaceChildren(seg.length
      ? sweepTable(seg, "segment") : el("p", { class: "muted" }, "no segment previews yet"));
    dnWrap.replaceChildren(dn.length
      ? sweepTable(dn, "denoise") : el("p", { class: "muted" }, "no denoise previews yet"));
    if (dn.length) {   // denoise A/B: players right in the sweep
      dnWrap.append(el("div", { class: "grid" }, ...dn.slice(0, 3).map(row =>
        el("div", { class: "cell" },
          el("a", { href: "#", onclick: (e) => {
            e.preventDefault(); selectPreview(row);
          } }, row.id),
          ...(row.result.audio || []).map(f =>
            el("audio", { controls: "", src: fileUrl(row, f),
                          style: "width:100%" }))))));
    }
    if (!selected && seg.length) selectPreview(seg[0]);  // newest by default
  }

  async function prune() {
    if (!confirm("Delete all previews? They are freely discardable.")) return;
    await api(`/projects/${name}/previews`, { method: "DELETE" }).catch(() => {});
    selected = null;
    clipWrap.replaceChildren(clipEmpty());
    await loadSweep();
  }

  main().replaceChildren(
    el("h1", {}, `Prepare tuner \u2014 ${name}`),
    el("p", { class: "muted" },
      "adjust the VAD dials, preview against one source, promote the winner ",
      el("a", { href: `#/project/${name}` }, "back to project")),
    err,
    el("h2", {}, "Source"),
    el("div", { class: "row" }, srcSel, ...chRadios),
    srcs.length ? el("div", {}, cv)
      : el("p", { class: "muted" }, "no sources \u2014 upload audio on the project page first"),
    el("h2", {}, "VAD parameters"),
    el("div", { class: "row", style: "flex-wrap:wrap" }, ...sliderRows,
      el("label", { class: "inline", title: "the full pipeline denoises "
                   + "before segmenting; previews should judge the same audio" },
        dnChk, "denoise first")),
    el("div", { class: "row" },
      el("button", { onclick: runSegmentPreview, disabled: srcs.length ? null : "" },
        "preview segment"),
      el("button", { onclick: runDenoisePreview, disabled: srcs.length ? null : "" },
        "preview denoise A/B"),
      msg),
    el("h2", {}, "Clips"),
    clipWrap,
    el("h2", {}, "Segment sweep"),
    segWrap,
    el("h2", {}, "Denoise A/B"),
    dnWrap,
    el("p", {}, el("button", { onclick: prune }, "prune all previews")));

  if (srcs.length) await loadPeaks();
  await loadSweep();
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try { await loadSweep(); } catch {}
  }, 2000);
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
        // Update the watched job in place; the table keeps every other row.
        const i = jobsCache.findIndex(j => j.id === msg.job.id);
        if (i >= 0) jobsCache[i] = msg.job;
        else jobsCache.unshift(msg.job);
        renderJobs(jobsCache.slice(0, 10), false);
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

  let jobsCache = Array.isArray(p.jobs) ? p.jobs : [];

  // Design doc §1.4: never expose the absolute max_epochs ceiling. A tier
  // with no checkpoint asks for epochs (submitted as max_epochs); a trained
  // tier asks for "N more" (submitted as add_epochs) so a resume can never
  // set the ceiling below the restored epoch counter and exit instantly.
  const tier = cfg.tier || "medium";
  const trained = p.tiers_trained.includes(tier);
  const epochs = el("input", { type: "number", min: "1",
                               value: trained ? "1000" : "4000",
                               style: "width:6em" });
  const skipVal = el("input", { type: "checkbox" });
  async function doTrain() {
    const n = parseInt(epochs.value, 10);
    if (!n || n < 1) { uploadErr.textContent = "epochs must be a number >= 1"; return; }
    const params = trained ? { add_epochs: n } : { max_epochs: n };
    if (skipVal.checked) params.skip_validate = true;
    await runJob("train", params);
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
    el("p", {}, el("a", { href: `#/prepare/${name}` },
      "prepare tuner (segment preview + promote)")),
    el("div", { class: "row" },
      el("button", { onclick: () => runJob("prepare") }, "run prepare"),
      el("button", { onclick: () => runJob("transcribe") }, "run transcribe"),
      el("label", { class: "inline" }, trained ? "epochs (more)" : "epochs",
        epochs),
      el("label", { class: "inline",
                    title: "train even when validation reports errors "
                         + "(same as the CLI's --skip-validate)" },
        skipVal, "skip validation"),
      el("button", { onclick: doTrain }, trained ? "train N more" : "run train"),
      el("button", { onclick: () => runJob("export") }, "run export")),
    jobsWrap,
    log,
    el("h2", {}, "Danger"),
    el("form", { onsubmit: doDelete, class: "row" },
      el("button", {}, "delete (moves to .trash)"), deleteErr));

  function render() { projectPage(name); }
  renderJobs(jobsCache);
  const running = jobsCache.find(j => j.state === "running");
  if (running) watch(running.id);

  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    if (ws) return; // the websocket already carries live state
    try {
      const jobs = await api(`/projects/${name}/jobs`);
      jobsCache = jobs;
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
    else if (h.startsWith("#/prepare/")) await preparePage(h.slice(10));
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
