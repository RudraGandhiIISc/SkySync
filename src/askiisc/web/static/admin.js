"use strict";
// Ask IISc admin dashboard. Plain JS; every value is escaped. Needs ui.js
// and policy_editor.js.

let tab = "overview";
let me = null;
let ctx = { departments: [], programs: {}, roles: [] };
let people = [];
let uploadEditor = null;
let policyEditor = null;
apiHooks.onUnauthorized = () => { location.href = "/"; };

const ROLE_LABEL = { student: "Student", faculty: "Faculty", staff: "Staff", admin: "Admin" };
const TITLES = {
  overview: "Overview", people: "People", requests: "Request log", documents: "Documents", grafana: "Grafana",
  requirements: "Requirements", models: "Models",
};
const FALLBACK_PROGRAMS = { phd: "PhD", "mtech-research": "M.Tech (Research)", "mtech-cse": "M.Tech CSE", "mtech-cds": "M.Tech (CDS)" };
const fmt = {
  n: (v) => (v === null || v === undefined ? "–" : Number(v).toLocaleString()),
  ms: (v) => (v === null || v === undefined ? "–" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`),
  pct: (v) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(1)}%`),
  time: (t) => (t ? new Date(t * 1000).toLocaleString() : "–"),
  clock: (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
};
function table(rows, cols, empty) {
  if (!rows.length) return `<div class="empty-note">${esc(empty || "Nothing here yet.")}</div>`;
  const cls = (c) => (c[2] ? ` class="${c[2]}"` : "");
  const head = cols.map((c) => `<th${cls(c)}>${esc(c[0].charAt(0).toUpperCase() + c[0].slice(1))}</th>`).join("");
  const body = rows.map((r) => `<tr>${cols.map((c) => `<td${cls(c)}>${c[1](r)}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}
function windowS() { return Number($("window").value); }
function origin(r) { return r.synthetic ? '<span class="pill locust">Locust</span>' : '<span class="pill">real</span>'; }
const DECISIONS = { answered: "Answered", abstained: "No answer found", rejected: "Turned away (429)", denied: "Denied", error: "Error" };
function decision(d) {
  const cls = d === "answered" ? "good" : d === "error" || d === "denied" ? "bad" : d === "rejected" ? "warn" : "";
  return `<span class="pill ${cls}">${esc(DECISIONS[d] || d)}</span>`;
}
// Short, readable times: "Today, 14:05" or "24 Oct, 14:05".
function when(t) {
  if (!t) return "–";
  const d = new Date(t * 1000); const today = new Date().toDateString() === d.toDateString();
  const hm = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return today ? `Today, ${hm}` : `${d.toLocaleDateString([], { day: "numeric", month: "short" })}, ${hm}`;
}
function setMsg(id, ok, text) { const el = $(id); el.className = ok ? "ok" : "error"; el.textContent = text; }
function originParam() { return $("origin").value; }

// Fixed pages of 20, 50 or 100 rows with previous / next arrows. `load` is
// called on every change; it calls `show(total)` before reading `slice`
// (rows in memory) or `offset` (rows fetched a page at a time).
function pager(id, load) {
  const el = $(id); const st = { page: 0, size: 20 };
  el.className = "pager";
  el.innerHTML = `<label>Rows <select aria-label="Rows per page">${[20, 50, 100].map((n) => `<option>${n}</option>`).join("")}</select></label>
    <button type="button" class="icon" data-dir="-1" title="Previous page" aria-label="Previous page">&#8249;</button>
    <span class="info"></span>
    <button type="button" class="icon" data-dir="1" title="Next page" aria-label="Next page">&#8250;</button>`;
  el.querySelector("select").onchange = (e) => { st.size = Number(e.target.value); st.page = 0; load(); };
  el.querySelectorAll("[data-dir]").forEach((b) => { b.onclick = () => { st.page += Number(b.dataset.dir); load(); }; });
  st.reset = () => { st.page = 0; };
  st.offset = () => st.page * st.size;
  st.slice = (rows) => rows.slice(st.offset(), st.offset() + st.size);
  st.show = (total) => {
    const pages = Math.max(1, Math.ceil(total / st.size));
    st.page = Math.min(Math.max(0, st.page), pages - 1);
    const from = total ? st.offset() + 1 : 0; const to = Math.min(total, st.offset() + st.size);
    el.querySelector(".info").textContent = `${fmt.n(from)}–${fmt.n(to)} of ${fmt.n(total)}`;
    el.querySelector('[data-dir="-1"]').disabled = st.page === 0;
    el.querySelector('[data-dir="1"]').disabled = st.page >= pages - 1;
  };
  return st;
}

// ---------- navigation ----------
function show(t) {
  tab = t;
  document.querySelectorAll("#nav [data-tab]").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  document.querySelectorAll("[data-panel]").forEach((s) => s.classList.toggle("hidden", s.dataset.panel !== t));
  $("title").textContent = TITLES[t];
  $("window-box").classList.toggle("hidden", t !== "overview");
  $("origin-box").classList.toggle("hidden", t !== "overview");
  $("g-open").classList.toggle("hidden", t !== "grafana" || !$("g-open").href);
  $("nav").classList.remove("open");
  try { history.replaceState(null, "", `#${t}`); } catch (_) { /* ignore */ }
  refresh(true);
}
$("nav").addEventListener("click", (e) => { const b = e.target.closest("[data-tab]"); if (b) show(b.dataset.tab); });
window.addEventListener("hashchange", () => { const t = location.hash.slice(1); if (TITLES[t] && t !== tab) show(t); });
$("nav-toggle").onclick = () => $("nav").classList.toggle("open");
$("window").onchange = () => refresh(true);
$("origin").onchange = () => { peoplePage.reset(); refresh(true); };
$("theme").onclick = () => toggleTheme();

// ---------- charts (inline SVG; dataviz rules: 2px lines, hairline solid
// grid, clean ticks, legend with line keys, crosshair tooltip) ----------
function niceTicks(max, count) {
  if (!(max > 0)) return [0, 1];
  const raw = max / count; const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = []; for (let v = 0; v <= max + step * 0.5 && ticks.length <= count + 1; v += step) ticks.push(+v.toFixed(10));
  if (ticks[ticks.length - 1] < max) ticks.push(+(ticks[ticks.length - 1] + step).toFixed(10));
  return ticks;
}
function svgEl(tag, attrs, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
}
// series: [{name, get(p) -> number|null, color}]; opts: {unit(v), target, targetLabel}
function lineChart(el, legendEl, points, series, opts) {
  el.innerHTML = ""; legendEl.innerHTML = "";
  if (!points.length) { el.innerHTML = '<div class="empty-note">No requests in this window.</div>'; return; }
  const W = Math.max(320, el.clientWidth || 600), H = 230, L = 52, R = 56, T = 10, B = 24;
  const xs = points.map((p) => p.time); const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let ymax = opts.target || 0;
  for (const s of series) for (const p of points) ymax = Math.max(ymax, s.get(p) || 0);
  const ticks = niceTicks(ymax || 1, 4); const top = ticks[ticks.length - 1];
  const X = (t) => L + ((t - x0) / Math.max(1, x1 - x0)) * (W - L - R);
  const Y = (v) => T + (1 - v / top) * (H - T - B);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.title });
  for (const v of ticks) {
    svg.appendChild(svgEl("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "gridline" }));
    svg.appendChild(svgEl("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end", class: "axis" }, opts.unit(v)));
  }
  const nx = Math.min(6, points.length);
  for (let i = 0; i < nx; i++) {
    const t = x0 + ((x1 - x0) * i) / Math.max(1, nx - 1);
    svg.appendChild(svgEl("text", { x: X(t), y: H - 6, "text-anchor": i === 0 ? "start" : i === nx - 1 ? "end" : "middle", class: "axis" }, fmt.clock(t)));
  }
  if (opts.target) {
    svg.appendChild(svgEl("line", { x1: L, x2: W - R, y1: Y(opts.target), y2: Y(opts.target), class: "target" }));
    svg.appendChild(svgEl("text", { x: W - R + 6, y: Y(opts.target) + 4, class: "target-label" }, opts.targetLabel));
  }
  const ends = [];
  for (const s of series) {
    const pts = points.filter((p) => s.get(p) !== null && s.get(p) !== undefined);
    if (!pts.length) continue;
    svg.appendChild(svgEl("polyline", { fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round",
      points: pts.map((p) => `${X(p.time).toFixed(1)},${Y(s.get(p)).toFixed(1)}`).join(" ") }));
    const last = pts[pts.length - 1];
    svg.appendChild(svgEl("circle", { cx: X(last.time), cy: Y(s.get(last)), r: 4, fill: s.color, stroke: "var(--panel)", "stroke-width": 2 }));
    ends.push({ y: Y(s.get(last)), x: X(last.time), name: s.name });
  }
  // Direct end labels only where they do not collide; else legend + tooltip.
  if (series.length <= 4 && !opts.target) {
    for (const e of ends) {
      if (ends.some((o) => o !== e && Math.abs(o.y - e.y) < 14)) continue;
      svg.appendChild(svgEl("text", { x: e.x + 8, y: e.y + 4, class: "endlabel" }, e.name));
    }
  }
  const cross = svgEl("line", { y1: T, y2: H - B, class: "cross", visibility: "hidden" });
  svg.appendChild(cross);
  el.appendChild(svg);
  const tip = document.createElement("div"); tip.className = "tooltip hidden"; el.appendChild(tip);
  svg.addEventListener("pointermove", (e) => {
    const box = svg.getBoundingClientRect(); const x = ((e.clientX - box.left) / box.width) * W;
    let best = points[0];
    for (const p of points) if (Math.abs(X(p.time) - x) < Math.abs(X(best.time) - x)) best = p;
    cross.setAttribute("x1", X(best.time)); cross.setAttribute("x2", X(best.time)); cross.setAttribute("visibility", "visible");
    tip.textContent = "";
    const when = document.createElement("div"); when.className = "when"; when.textContent = new Date(best.time * 1000).toLocaleTimeString(); tip.appendChild(when);
    for (const s of series) {
      const row = document.createElement("div"); row.className = "row";
      const key = document.createElement("i"); key.style.background = s.color;
      const val = document.createElement("b"); const v = s.get(best); val.textContent = v === null || v === undefined ? "–" : opts.unit(v);
      const name = document.createElement("span"); name.textContent = s.name;
      row.append(key, val, name); tip.appendChild(row);
    }
    tip.classList.remove("hidden");
    const px = (X(best.time) / W) * box.width;
    tip.style.left = `${Math.min(px + 12, box.width - 160)}px`; tip.style.top = "8px";
  });
  svg.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.classList.add("hidden"); });
  for (const s of series) {
    const item = document.createElement("span"); const key = document.createElement("i"); key.style.background = s.color;
    item.append(key, document.createTextNode(s.name)); legendEl.appendChild(item);
  }
}

// ---------- overview ----------
const P95_TARGET_MS = 3500;
async function overview() {
  const w = windowS(); const users = originParam();
  const [o, s] = await Promise.all([
    api("GET", `/api/admin/overview?window_s=${w}&origin=${users}`),
    api("GET", `/api/admin/series?window_s=${w}&step_s=${Math.max(10, w / 60)}&origin=${users}`),
  ]);
  if (!o.ok) return;
  const d = o.data; const r = d.requests; const u = d.users;
  const p95 = r.p95_ms;
  const over = p95 !== null && p95 !== undefined && p95 > P95_TARGET_MS;
  const share = p95 ? Math.min(100, (p95 / P95_TARGET_MS) * 100) : 0;
  $("hero").innerHTML = `<div class="label">p95 latency, answered requests</div>
    <div class="value">${p95 ? (p95 / 1000).toFixed(2) : "–"}<small>s</small></div>
    <div class="meter ${over ? "over" : ""}" role="img" aria-label="${Math.round(share)}% of the 3.5 second target"><i style="width:${share}%"></i></div>
    <div class="foot">${p95 ? (over ? `<span class="pill bad">${icon("alert")} Over the 3.5 s target</span>` : `<span class="pill good">${icon("check")} Within the 3.5 s target</span>`) : '<span class="pill">No answers yet</span>'}
      <span>p50 ${fmt.ms(r.p50_ms)}, p99 ${fmt.ms(r.p99_ms)}</span></div>`;
  const stat = (label, value, sub) => `<div class="stat"><div class="label">${esc(label)}</div><div class="value">${value}</div><div class="sub">${sub || "&nbsp;"}</div></div>`;
  $("stats").innerHTML = [
    stat("Requests", fmt.n(r.requests), `${fmt.n(r.real)} real, ${fmt.n(r.locust)} Locust`),
    stat("Request rate", `${(r.rate_per_s_last_minute || 0).toFixed(2)}/s`, "in the last minute"),
    stat("People active", fmt.n(r.users_active), `${fmt.n(u.live_sessions)} signed in now`),
    stat("Accounts", fmt.n(u.accounts), `${fmt.n(u.disabled)} disabled, ${fmt.n(u.locust)} Locust`),
    stat("First token, p95", fmt.ms(r.ttft_p95_ms), "target 0.8 s"),
    stat("Errors", fmt.n(r.errors), `${fmt.n(r.rejected_429)} turned away (429)`),
    stat("Cache hit rate", fmt.pct(r.cache_hit_rate), `${fmt.n(r.output_tokens)} tokens written`),
    stat("Feedback", `${fmt.n(d.ratings.up)} up`, `${fmt.n(d.ratings.down)} down`),
  ].join("");
  const pts = s.ok ? s.data.points : []; const step = s.ok ? s.data.step_s : 60;
  const perS = (v) => (v < 1 ? v.toFixed(2) : v.toFixed(1));
  // The lines follow the Users filter: one origin, or both and their total.
  const rateLines = {
    real: [{ name: "Real users", get: (p) => p.real / step, color: "var(--series-1)" }],
    locust: [{ name: "Locust", get: (p) => p.locust / step, color: "var(--series-2)" }],
    "": [
      { name: "Real + Locust", get: (p) => (p.real + p.locust) / step, color: "var(--series-3)" },
      { name: "Real users", get: (p) => p.real / step, color: "var(--series-1)" },
      { name: "Locust", get: (p) => p.locust / step, color: "var(--series-2)" },
    ],
  }[users] || [];
  lineChart($("chart-rate"), $("legend-rate"), pts, [
    ...rateLines,
    { name: "Errors and 429s", get: (p) => (p.errors + p.rejected) / step, color: "var(--critical)" },
  ], { unit: perS, title: "Requests per second" });
  $("rate-desc").textContent = { real: "Real users only", locust: "Locust load-test users only", "": "Real users, Locust load-test users and both together" }[users] + "; errors and rejections in red.";
  lineChart($("chart-latency"), $("legend-latency"), pts, [
    { name: "p50", get: (p) => p.p50_ms, color: "var(--series-3)" },
    { name: "p95", get: (p) => p.p95_ms, color: "var(--series-1)" },
    { name: "p99", get: (p) => p.p99_ms, color: "var(--series-2)" },
  ], { unit: (v) => fmt.ms(v), target: P95_TARGET_MS, targetLabel: "3.5 s", title: "Latency" });
  const outcomes = Object.entries(r.by_decision).map(([k, v]) => ({ k, v }));
  const sources = Object.entries(r.by_source).map(([k, v]) => ({ k, v }));
  $("outcomes").innerHTML = `<div class="stack"><div class="table-wrap">${table(outcomes, [["Outcome", (x) => decision(x.k)], ["Requests", (x) => fmt.n(x.v), "num"]])}</div>` +
    `<div class="table-wrap">${table(sources, [["Entry point", (x) => esc(SOURCE_NAMES[x.k] || x.k)], ["Requests", (x) => fmt.n(x.v), "num"]])}</div></div>`;
  const sys = d.system || {};
  renderSystem(sys);
  linksHtml(sys.links || {});
  const busiest = await api("GET", `/api/admin/users?window_s=${w}&origin=${users}`);
  if (busiest.ok) {
    const top = busiest.data.users.filter((x) => x.requests).sort((a, b) => b.requests - a.requests).slice(0, 10);
    $("top-users").innerHTML = table(top, [
      ["Person", (x) => esc(x.user_id)], ["Department", (x) => esc(x.department.toUpperCase())], ["Role", (x) => esc(ROLE_LABEL[x.role] || x.role)],
      ["Origin", origin], ["Requests", (x) => fmt.n(x.requests), "num"], ["Errors", (x) => fmt.n(x.errors), "num"],
      ["Turned away", (x) => fmt.n(x.rejected_429), "num"], ["Last request", (x) => when(x.last_request)],
    ]);
  }
}
// The serving job as it is right now: model, workers, vLLM queue and every
// GPU on the node, ours or another job's.
const FULL = 0.9;
// Red marks nearly full memory; a busy GPU at 100% is normal.
function meter(frac, other, warnFull) {
  const pct = Math.max(0, Math.min(1, frac || 0));
  const cls = warnFull && pct >= FULL ? "full" : other ? "other" : "";
  return `<div class="meter ${cls}"><i style="width:${(pct * 100).toFixed(1)}%"></i></div>`;
}
function since(t) {
  const s = Math.max(0, Date.now() / 1000 - t); const d = Math.floor(s / 86400); const h = Math.floor((s % 86400) / 3600); const m = Math.floor((s % 3600) / 60);
  return d ? `${d} d ${h} h` : h ? `${h} h ${m} min` : `${m} min`;
}
function renderSystem(sys) {
  const workers = sys.workers || []; const ready = workers.filter((w) => w.state === "ready");
  const state = ready.length ? ["good", "check", "Serving"] : workers.length ? ["warn", "", "Starting"] : ["", "", "Scaled to zero"];
  $("serve-state").innerHTML = `<span class="pill ${state[0]}">${state[1] ? icon(state[1]) : ""}${state[2]}</span>`;
  const onGpus = [...new Set(workers.map((w) => w.gpu))].sort();
  const workerText = workers.length
    ? `${ready.length} running${workers.length > ready.length ? `, ${workers.length - ready.length} starting` : ""} on GPU ${onGpus.join(", ")}`
    : "None (scaled to zero)";
  const q = sys.queue || {};
  const queue = [`${fmt.n(q.waiting || 0)} waiting`, q.kv_cache != null ? `KV cache ${Math.round(q.kv_cache * 100)}%` : "",
    q.tokens_per_s != null ? `${fmt.n(Math.round(q.tokens_per_s))} tok/s` : ""].filter(Boolean).join(" · ");
  const gib = (mib) => (mib / 1024).toFixed(1);
  const gpus = sys.gpus || [];
  const cards = gpus.map((g) => {
    const mem = g.memory_total_mib ? g.memory_used_mib / g.memory_total_mib : 0;
    const busy = !g.ours && (mem > 0.05 || g.utilization > 5);
    const owner = g.ours ? '<span class="ours">Ask IISc worker</span>' : `<span class="muted">${busy ? "Other job on node" : "Free"}</span>`;
    return `<div class="gpu"><div class="gpu-head"><strong>GPU ${esc(g.index)}</strong>${owner}</div>
      <div class="gpu-row"><span>Util</span>${meter(g.utilization / 100, !g.ours, false)}<span class="num">${esc(g.utilization)}%</span></div>
      <div class="gpu-row"><span>Memory</span>${meter(mem, !g.ours, true)}<span class="num ${mem >= FULL ? "full" : ""}">${gib(g.memory_used_mib)} / ${g.memory_total_mib ? gib(g.memory_total_mib) : "?"} GiB</span></div></div>`;
  }).join("");
  const others = gpus.filter((g) => !g.ours && (g.memory_total_mib ? g.memory_used_mib / g.memory_total_mib > 0.05 : g.utilization > 5)).map((g) => g.index);
  const note = others.length ? `<p class="muted small">GPU${others.length > 1 ? "s" : ""} ${others.join(", ")} ${others.length > 1 ? "are" : "is"} in use by another job on this node, not by Ask IISc.</p>` : "";
  const fact = (label, value) => `<div><div class="label">${label}</div><div class="value">${value}</div></div>`;
  $("system").innerHTML = `<div class="facts-row">
      ${fact("Model", `<strong>${esc(sys.model || "–")}</strong>${sys.vllm_version ? `<br><span class="muted">vLLM ${esc(sys.vllm_version)}</span>` : ""}`)}
      ${fact("Workers", esc(workerText))}
      ${fact("Queue", esc(ready.length ? queue : "–"))}
      ${fact("Index", `${fmt.n(sys.chunks)} passages · ${fmt.n(sys.documents)} docs<br><a href="#documents">Open documents →</a>`)}
      ${fact("Running since", `${fmt.time(sys.started)}${sys.started ? `<br><span class="muted">up ${since(sys.started)}</span>` : ""}`)}
      ${fact("Monitoring", '<span id="links"></span>')}</div>
    <div class="gpu-list">${cards || '<span class="muted">No GPU readings.</span>'}</div>${note}`;
}
const SOURCE_NAMES = { chat: "Saved chat", temporary: "Temporary chat", api: "API" };
// Grafana's address as this browser reaches it (a tunnel may map another
// local port); remembered per browser.
let grafanaBase = "";
function grafanaUrl() {
  try { return localStorage.getItem("askiisc-grafana") || grafanaBase; } catch (_) { return grafanaBase; }
}
function logsLink(r) {
  const base = grafanaUrl(); if (!base) return "";
  const q = new URLSearchParams({ "var-request_id": r.request_id, from: String(Math.round((r.time - 300) * 1000)), to: String(Math.round((r.time + 300) * 1000)) });
  return `<a href="${esc(base)}/d/askiisc-requests/ask-iisc-requests?${q}" target="_blank" rel="noopener" title="Request ${esc(r.request_id)} in Grafana">logs ↗</a>`;
}
function linksHtml(links) {
  if (links.grafana) grafanaBase = links.grafana;
  const el = $("links"); if (!el) return;  // shown in the Serving job panel
  const names = { grafana: "Grafana", prometheus: "Prometheus", locust: "Locust" };
  el.innerHTML = Object.entries(links).map(([k, v]) => `<a href="${esc(v)}" target="_blank" rel="noopener">${esc(names[k] || k)}</a>`).join(" · ") ||
    '<span class="muted">Not running</span>';
}

// ---------- people ----------
async function loadContext() {
  const res = await api("GET", "/api/admin/people");
  if (res.ok) {
    people = res.data.people;
    ctx = { departments: res.data.departments, programs: res.data.programs, roles: res.data.roles,
      deptPrograms: res.data.department_programs, kinds: res.data.kinds,
      manages: res.data.manages, institute: res.data.institute_admin };
  } else {
    people = [];
    ctx = { departments: [], programs: FALLBACK_PROGRAMS, deptPrograms: {}, kinds: {}, roles: ["student", "faculty", "staff", "admin"], manages: [], institute: false, unavailable: errText(res) };
  }
  $("dept-list").innerHTML = ctx.departments.map((d) => `<option value="${esc(d)}">`).join("");
}
function fillSelect(id, first, entries) {
  const el = $(id); const cur = el.value;
  el.innerHTML = `<option value="">${esc(first)}</option>` + entries.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("");
  el.value = cur;
}
async function peoplePanel() {
  await loadContext();
  if (ctx.unavailable) {
    $("people-table").innerHTML = `<div class="empty-note">${esc(ctx.unavailable)}</div>`;
    $("add-open").disabled = true; $("bulk-open").disabled = true; return;
  }
  fillSelect("p-dept", "All departments", ["institute", ...ctx.departments].map((d) => [d, d.toUpperCase()]));
  fillSelect("p-role", "All roles", Object.entries(ROLE_LABEL));
  fillSelect("p-prog", "All programmes", Object.entries(ctx.programs));
  const years = [...new Set(people.map((p) => p.joined).filter(Boolean))].sort((a, b) => b - a);
  fillSelect("p-year", "All years", years.map((y) => [String(y), String(y)]));
  $("people-desc").textContent = ctx.institute
    ? "Everyone in the IISc directory. You manage department admins and their people; institute admins are added from the server terminal."
    : `Everyone in the IISc directory. You manage the students, faculty and staff of ${String(ctx.manages).toUpperCase()}.`;
  renderPeople();
}
const peoplePage = pager("people-pager", () => renderPeople());
// ---- People: tabs, one column per field, a row menu and bulk actions ----
let peopleView = "real";
const picked = new Set();
// What the Status column says, from the account and the allowlist.
function personStatus(p) {
  if (p.access === "revoked") return "suspended";
  if (p.access === "not allowed") return "not allowed";
  if (p.account === "disabled") return "disabled";
  if (p.account === "not registered") return "not registered";
  return "active";
}
const STATUS_LOOK = { active: ["good", "Active"], "not registered": ["warn", "Not registered"], suspended: ["bad", "Suspended"],
  "not allowed": ["bad", "Not allowed"], disabled: ["bad", "Disabled"] };
// Real people who cannot sign in yet, and test ids without an account.
function needsAttention(p) {
  const st = personStatus(p);
  return st === "not registered" || st === "not allowed" || (!p.synthetic && p.access === "not registered");
}
const VIEWS_OF = { all: () => true, real: (p) => !p.synthetic, locust: (p) => p.synthetic, attention: needsAttention };
function ago(t) {
  if (!t) return "Never";
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 30 * 86400) return `${Math.floor(s / 86400)} d ago`;
  return new Date(t * 1000).toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
}
function roleLook(p) {
  if (p.role === "admin") return p.department === "institute" ? ["inst", "Institute admin"] : ["admin", "Admin"];
  return [p.role, ROLE_LABEL[p.role] || p.role];
}
function programmeOf(p) {
  if (p.program) return ctx.programs[p.program] || p.program;
  const kinds = ctx.kinds[p.role] || {};
  return p.kind && p.kind !== Object.keys(kinds)[0] ? kinds[p.kind] : "";
}
function signInOf(p) {
  if (p.synthetic) return '<span class="muted">Test</span>';
  if (p.access === "active") return "Password";
  if (p.access === "not registered") return '<span class="warn-text">Not set up</span>';
  return '<span class="muted">–</span>';
}
function renderPeople() {
  const q = $("p-q").value.trim().toLowerCase();
  const f = { department: $("p-dept").value, role: $("p-role").value, program: $("p-prog").value,
    joined: $("p-year").value && Number($("p-year").value) };
  const status = $("p-acct").value;
  const filtered = people.filter((p) =>
    (!q || p.user_id.includes(q) || p.groups.some((g) => g.includes(q))) &&
    (!status || personStatus(p) === status) &&
    Object.entries(f).every(([k, v]) => !v || p[k] === v));
  document.querySelectorAll("#people-tabs [data-view]").forEach((b) => {
    b.setAttribute("aria-selected", String(b.dataset.view === peopleView));
    b.querySelector(".count").textContent = filtered.filter(VIEWS_OF[b.dataset.view]).length;
  });
  const rows = filtered.filter(VIEWS_OF[peopleView]);
  peoplePage.show(rows.length);
  const page = peoplePage.slice(rows);
  $("people-count").textContent = `Showing ${fmt.n(page.length)} of ${fmt.n(rows.length)} people`;
  const managed = page.filter((p) => p.can_manage);
  const allPicked = managed.length && managed.every((p) => picked.has(p.user_id));
  const head = `<tr><th class="check"><input type="checkbox" id="pick-all" aria-label="Select all on this page" ${allPicked ? "checked" : ""} ${managed.length ? "" : "disabled"}></th>
    <th>Person</th><th>Dept</th><th>Role</th><th>Programme</th><th class="num">Year</th><th>Groups</th><th>Status</th><th>Sign-in</th><th class="wrap-head">Last sign-in</th><th></th></tr>`;
  const body = page.map((p) => {
    const [rc, rl] = roleLook(p); const [sc, sl] = STATUS_LOOK[personStatus(p)];
    const groups = p.groups.length
      ? `<span class="chip">${esc(p.groups[0])}</span>${p.groups.length > 1 ? ` <span class="muted" title="${esc(p.groups.join(", "))}">+${p.groups.length - 1}</span>` : ""}`
      : '<span class="muted">–</span>';
    const mine = p.user_id === me.user_id;
    return `<tr class="${mine ? "me" : ""}">
      <td class="check">${p.can_manage ? `<input type="checkbox" data-pick="${esc(p.user_id)}" aria-label="Select ${esc(p.user_id)}" ${picked.has(p.user_id) ? "checked" : ""}>` : ""}</td>
      <td><div class="who-cell"><span class="avatar role-${esc(p.role)}">${esc(initials(p.user_id))}</span><strong class="clip" title="${esc(p.user_id)}">${esc(p.user_id)}</strong>${mine ? '<span class="pill accent">You</span>' : ""}${p.synthetic ? '<span class="pill locust" title="Locust test account">Locust</span>' : ""}</div></td>
      <td class="dept" title="${esc(p.department.toUpperCase())}">${esc(p.department === "institute" ? "INST" : p.department.toUpperCase())}</td>
      <td><span class="role-pill r-${rc}">${esc(rl)}</span></td>
      <td class="clip" title="${esc(programmeOf(p))}">${esc(programmeOf(p)) || '<span class="muted">–</span>'}</td>
      <td class="num">${p.joined ? esc(p.joined) : '<span class="muted">–</span>'}</td>
      <td>${groups}</td>
      <td><span class="status ${sc}">${esc(sl)}</span></td>
      <td>${signInOf(p)}</td>
      <td class="muted" title="${esc(when(p.last_login))}">${esc(ago(p.last_login))}</td>
      <td class="num">${p.can_manage ? `<button type="button" class="icon row-menu" data-menu="${esc(p.user_id)}" aria-label="Actions for ${esc(p.user_id)}" aria-haspopup="menu">&#8943;</button>` : ""}</td></tr>`;
  }).join("");
  $("people-table").innerHTML = rows.length
    ? `<table class="people-table"><thead>${head}</thead><tbody>${body}</tbody></table>`
    : '<div class="empty-note">Nobody matches these filters.</div>';
  showBulk();
}
function showBulk() {
  const n = [...picked].filter((u) => people.some((p) => p.user_id === u)).length;
  $("people-bulk").classList.toggle("hidden", !n);
  $("bulk-count").textContent = `${n} selected`;
}
["p-q", "p-dept", "p-role", "p-prog", "p-year", "p-acct"].forEach((id) => { $(id).oninput = () => { peoplePage.reset(); renderPeople(); }; });
$("people-tabs").onclick = (e) => {
  const b = e.target.closest("[data-view]"); if (!b) return;
  peopleView = b.dataset.view; peoplePage.reset(); renderPeople();
};
// The row menu: one small popup at a time, closed by Escape or a click outside.
let openMenu = null;
function closeMenu() { if (openMenu) { openMenu.remove(); openMenu = null; } }
document.addEventListener("click", (e) => { if (openMenu && !e.target.closest(".menu-pop, .row-menu")) closeMenu(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });
function rowMenu(btn) {
  closeMenu();
  const p = people.find((x) => x.user_id === btn.dataset.menu); if (!p) return;
  const off = p.access === "revoked" || p.access === "not allowed";
  const item = (act, ic, label, cls) => `<button type="button" role="menuitem" class="${cls || ""}" data-act="${act}" data-user="${esc(p.user_id)}">${icon(ic)}${label}</button>`;
  const pop = document.createElement("div");
  pop.className = "menu-pop"; pop.setAttribute("role", "menu");
  pop.innerHTML = item("edit", "edit", "Edit") + (off ? item("restore", "check", "Restore access") : item("revoke", "lock", "Suspend access")) + item("remove", "trash", "Remove", "danger");
  document.body.appendChild(pop);
  const r = btn.getBoundingClientRect();
  pop.style.top = `${r.bottom + window.scrollY + 4}px`;
  pop.style.left = `${Math.max(8, r.right + window.scrollX - pop.offsetWidth)}px`;
  pop.onclick = (e) => { const b = e.target.closest("[data-act]"); if (b) { closeMenu(); personAction(b.dataset.act, b.dataset.user); } };
  pop.querySelector("button").focus();
  openMenu = pop;
}
async function personAction(act, user) {
  if (act === "edit") { openPerson(people.find((p) => p.user_id === user)); return; }
  if (act === "revoke" || act === "restore") {
    // Suspending signs them out and keeps their account and chats.
    if (act === "revoke" && !confirm(`Suspend ${user}'s access? They are signed out and cannot sign in until restored; their chats are kept.`)) return;
    const res = await api("POST", `/api/admin/people/${encodeURIComponent(user)}/${act}`);
    if (!res.ok) alert(errText(res));
  } else if (act === "remove") {
    if (!confirm(`Remove ${user} from the directory? Their account and all their chats are deleted permanently. To keep their chats, use Suspend access instead.`)) return;
    const res = await api("DELETE", `/api/admin/people/${encodeURIComponent(user)}`);
    if (!res.ok) alert(errText(res));
  }
  peoplePanel();
}
$("people-table").onclick = (e) => {
  const menu = e.target.closest("[data-menu]"); if (menu) { e.stopPropagation(); rowMenu(menu); return; }
};
$("people-table").onchange = (e) => {
  const box = e.target;
  if (box.id === "pick-all") {
    $("people-table").querySelectorAll("[data-pick]").forEach((b) => { b.checked = box.checked; picked[box.checked ? "add" : "delete"](b.dataset.pick); });
  } else if (box.dataset.pick) {
    picked[box.checked ? "add" : "delete"](box.dataset.pick);
  }
  showBulk();
};
$("people-bulk").onclick = async (e) => {
  const b = e.target.closest("[data-bulk]"); if (!b) return;
  if (b.dataset.bulk === "clear") { picked.clear(); renderPeople(); return; }
  const users = [...picked]; const act = b.dataset.bulk;
  if (act === "revoke" && !confirm(`Suspend access for ${users.length} people? They are signed out; their chats are kept.`)) return;
  const failed = [];
  for (const user of users) {
    const res = await api("POST", `/api/admin/people/${encodeURIComponent(user)}/${act}`);
    if (!res.ok) failed.push(`${user}: ${errText(res)}`);
  }
  if (failed.length) alert(failed.join("\n"));
  picked.clear(); peoplePanel();
};

// The groups the directory adds (access/directory.py _derived), shown so
// admins add only extras.
const ROLE_STEM = { student: "students", faculty: "faculty", staff: "staff", admin: "admins" };
const RESEARCH = new Set(["phd", "mtech-research"]);
function autoGroups() {
  const dept = $("pf-dept").value; const role = $("pf-role").value; const prog = $("pf-prog").value;
  const kind = $("pf-kind").value; const year = $("pf-year").value;
  const groups = dept ? [`${dept}-${ROLE_STEM[role]}`] : [];
  if (role === "student" && prog) {
    groups.push(prog, `${dept}-${prog}`, `${dept}-${RESEARCH.has(prog) ? "research" : "course"}-students`);
    if (year) groups.push(`${dept}-${prog}-${year}`);
  }
  const plain = Object.keys(ctx.kinds[role] || {})[0];
  if (kind && kind !== plain) groups.push(`${dept}-${kind}-${ROLE_STEM[role]}`);
  $("pf-auto").textContent = groups.length ? `Added automatically: ${groups.join(", ")}.` : "";
}
// Students choose a programme their department offers and must give a
// joining year; faculty and staff choose a type, the year is optional.
function roleFields() {
  const dept = $("pf-dept").value; const role = $("pf-role").value;
  const student = role === "student"; const kinds = ctx.kinds[role] || {};
  $("pf-prog-field").classList.toggle("hidden", !student);
  $("pf-kind-field").classList.toggle("hidden", !Object.keys(kinds).length);
  const offered = (ctx.deptPrograms || {})[dept] || Object.keys(ctx.programs);
  const prog = $("pf-prog").value;
  $("pf-prog").innerHTML = offered.map((k) => `<option value="${esc(k)}">${esc(ctx.programs[k] || k)}</option>`).join("");
  if (offered.includes(prog)) $("pf-prog").value = prog;
  const kind = $("pf-kind").value;
  $("pf-kind").innerHTML = Object.entries(kinds).map(([k, l]) => `<option value="${esc(k)}">${esc(l)}</option>`).join("");
  if (kind in kinds) $("pf-kind").value = kind;
  $("pf-year").required = student;
  $("pf-year-label").textContent = student ? "Joining year" : "Joining year (optional)";
  autoGroups();
}
["pf-dept", "pf-role"].forEach((id) => { $(id).onchange = roleFields; });
["pf-prog", "pf-kind"].forEach((id) => { $(id).onchange = autoGroups; });
$("pf-year").oninput = autoGroups;

let editing = null;
function openPerson(person) {
  editing = person || null;
  $("person-title").textContent = person ? `Edit ${person.user_id}` : "Add person";
  $("pf-id").value = person ? person.user_id : ""; $("pf-id").disabled = !!person;
  const own = Array.isArray(ctx.manages) ? ctx.manages[0] : "";
  const depts = ctx.institute ? ctx.departments : [own];
  $("pf-dept").innerHTML = depts.map((d) => `<option value="${esc(d)}">${esc(d.toUpperCase())}</option>`).join("");
  $("pf-dept").value = person ? person.department : own || depts[0] || "";
  $("pf-dept").disabled = !ctx.institute;
  const roles = ctx.institute ? ["student", "faculty", "staff", "admin"] : ["student", "faculty", "staff"];
  $("pf-role").innerHTML = roles.map((r) => `<option value="${r}">${ROLE_LABEL[r]}${r === "admin" ? " (department admin)" : ""}</option>`).join("");
  $("pf-role").value = person ? person.role : "student";
  $("pf-prog").innerHTML = ""; $("pf-kind").innerHTML = "";
  roleFields();
  if (person && person.program) $("pf-prog").value = person.program;
  if (person && person.kind) $("pf-kind").value = person.kind;
  $("pf-year").value = person && person.joined ? person.joined : "";
  $("pf-groups").value = person ? person.groups.join(", ") : "";
  $("pf-msg").textContent = "";
  autoGroups();
  $("dlg-person").showModal();
}
$("add-open").onclick = () => openPerson(null);
$("pf-cancel").onclick = () => $("dlg-person").close();
$("person-form").onsubmit = async (e) => {
  e.preventDefault();
  const body = {
    user_id: $("pf-id").value.trim(), department: $("pf-dept").value.trim(), role: $("pf-role").value,
    groups: $("pf-groups").value.split(",").map((x) => x.trim()).filter(Boolean),
    program: $("pf-role").value === "student" ? $("pf-prog").value : "",
    kind: $("pf-kind-field").classList.contains("hidden") ? "" : $("pf-kind").value,
    joined: $("pf-year").value ? Number($("pf-year").value) : null,
  };
  const res = editing
    ? await api("PUT", `/api/admin/people/${encodeURIComponent(editing.user_id)}`, body)
    : await api("POST", "/api/admin/people", body);
  if (!res.ok) { setMsg("pf-msg", false, errText(res)); return; }
  $("dlg-person").close(); peoplePanel();
};

$("bulk-open").onclick = () => {
  $("b-text").value = ""; $("b-file").value = ""; $("b-preview").innerHTML = ""; $("b-msg").textContent = "";
  $("b-apply").disabled = true; $("dlg-bulk").showModal();
};
$("b-close").onclick = () => $("dlg-bulk").close();
$("b-file").onchange = async () => { const f = $("b-file").files[0]; if (f) { $("b-text").value = await f.text(); $("b-apply").disabled = true; } };
$("b-text").oninput = () => { $("b-apply").disabled = true; };
async function bulk(apply) {
  const res = await api("POST", "/api/admin/people/bulk", { csv: $("b-text").value, apply });
  if (!res.ok) { setMsg("b-msg", false, errText(res)); $("b-apply").disabled = true; $("b-preview").innerHTML = ""; return; }
  const d = res.data;
  $("b-preview").innerHTML = table(d.people, [
    ["user id", (p) => esc(p.user_id)], ["department", (p) => esc(p.department)], ["role", (p) => esc(p.role)],
    ["programme, type, year", (p) => esc(detail(p))], ["groups", (p) => esc(p.groups.join(", "))],
  ]);
  if (d.applied) {
    setMsg("b-msg", true, `Added ${d.count} people. They can now create their account on the sign-in page.`);
    $("b-apply").disabled = true; peoplePanel();
  }
  else { setMsg("b-msg", true, `${d.count} rows are valid. Check them, then add.`); $("b-apply").disabled = false; $("b-apply").textContent = `Add ${d.count} people`; }
}
$("b-check").onclick = () => bulk(false);
$("b-apply").onclick = () => bulk(true);

// ---------- request log ----------
function filterQuery() {
  const p = new URLSearchParams();
  const add = (k, v) => { if (v) p.set(k, v); };
  add("user_id", $("f-user").value.trim()); add("department", $("f-dept").value.trim());
  add("synthetic", $("f-origin").value); add("source", $("f-source").value);
  add("decision", $("f-decision").value); add("status", $("f-status").value.trim());
  return p;
}
const reqPage = pager("req-pager", () => requests());
async function requests() {
  if (!grafanaBase) { const o = await api("GET", "/api/admin/overview?window_s=60"); if (o.ok) linksHtml((o.data.system || {}).links || {}); }
  if (!$("grafana-base").value) $("grafana-base").value = grafanaUrl();
  const p = filterQuery(); p.set("limit", String(reqPage.size)); p.set("offset", String(reqPage.offset()));
  $("csv").href = `/api/admin/requests.csv?${filterQuery()}`;
  const res = await api("GET", `/api/admin/requests?${p}`);
  if (!res.ok) return;
  const rows = res.data.rows;
  reqPage.show(res.data.total);
  $("req-table").innerHTML = table(rows, [
    ["time", (r) => esc(new Date(r.time * 1000).toLocaleTimeString())], ["user", (r) => `<div class="req-user">${esc(r.user_id)}<span>${esc([r.department, r.role].filter(Boolean).join(" · "))}</span></div>`],
    ["origin", origin],
    ["entry", (r) => esc(r.source)], ["outcome", (r) => decision(r.decision)], ["HTTP", (r) => esc(r.status)],
    ["latency", (r) => fmt.ms(r.latency_ms)], ["first token", (r) => fmt.ms(r.ttft_ms)],
    ["tokens in/out", (r) => `${fmt.n(r.prompt_tokens)} / ${fmt.n(r.completion_tokens)}`],
    ["cache", (r) => flag(r.cache_hit, "Answered from the cache", "Not from the cache")],
    ["cold start", (r) => flag(r.cold_hold, "Waited for a GPU worker to start", "No cold start")],
    ["citations", (r) => fmt.n(r.citations)],
    ["logs", (r) => logsLink(r)],
  ], "No requests match.");
}
// A yes/no column: a green check for yes, a grey cross for no.
function flag(on, yes, no) {
  return on
    ? `<span class="flag on" title="${yes}" aria-label="${yes}">${icon("check")}</span>`
    : `<span class="flag off" title="${no}" aria-label="${no}">${icon("x")}</span>`;
}
// Filters apply as they change; typed ones after a short pause.
let reqTimer = null;
function reqFilter() { clearTimeout(reqTimer); reqTimer = null; reqPage.reset(); requests(); }
["f-origin", "f-source", "f-decision"].forEach((id) => { $(id).onchange = reqFilter; });
["f-user", "f-dept", "f-status"].forEach((id) => {
  $(id).oninput = () => { clearTimeout(reqTimer); reqTimer = setTimeout(reqFilter, 350); };
  $(id).onkeydown = (e) => { if (e.key === "Enter") reqFilter(); };
});
$("grafana-base").onchange = () => {
  try { localStorage.setItem("askiisc-grafana", $("grafana-base").value.trim().replace(/\/$/, "")); } catch (_) { /* private mode */ }
  requests();
};

// ---------- documents ----------
// Find a document on the left; the right side shows who can read it, its
// versions and new versions. New documents come from the upload dialog.
let selectedDoc = null;
const OWNER_CHOICES = () => [...ctx.departments, "institute"];
async function documentsPanel() {
  if (!ctx.departments.length) await loadContext();
  if (!policyEditor) policyEditor = PolicyEditor($("p-policy"), ctx, { owners: OWNER_CHOICES() }, (p) => { showJson(p); preview(p); });
  await findDocs();
}
let findTimer = null;
$("d-q").oninput = () => { clearTimeout(findTimer); findTimer = setTimeout(findDocs, 250); };
$("d-site").onchange = () => findDocs();
$("d-empty-upload").onclick = () => $("u-open").click();
// "Admissions – Department of Aerospace Engineering": the page, then the
// organisation, which the site already shows.
function shortTitle(t) {
  const parts = String(t || "").split(/\s+[–|—]\s+|\s+-\s+/);
  return parts[0].length >= 3 ? parts[0] : String(t || "");
}
const KIND_NAMES = { pdf: "PDF", html: "Web page", htm: "Web page", txt: "Text" };
async function findDocs() {
  const q = $("d-q").value.trim(); const site = $("d-site").value;
  const res = await api("GET", `/api/admin/documents?q=${encodeURIComponent(q)}&limit=40&site=${encodeURIComponent(site)}`);
  const docs = res.ok ? res.data.documents : [];
  if (res.ok) {
    // An older server sends only the documents; count what came back.
    const sites = res.data.sites || []; const total = res.data.total ?? docs.length;
    res.data.sites = sites; res.data.total = total;
    const sel = $("d-site"); const cur = sel.value;
    if (sel.options.length <= 1 && sites.length) {
      sel.innerHTML = '<option value="">All sites</option>' + res.data.sites.map(([s, n]) => `<option value="${esc(s)}">${esc(s)} (${fmt.n(n)})</option>`).join("");
      sel.value = cur;
    }
    const all = res.data.sites.reduce((t, [, n]) => t + n, 0);
    $("d-site").classList.toggle("hidden", !sites.length);
    $("d-count").textContent = !sites.length ? `First ${fmt.n(docs.length)} documents · search to find others`
      : res.data.total > docs.length
      ? `Showing ${fmt.n(docs.length)} of ${fmt.n(res.data.total)} matches · refine the search to narrow it`
      : `${fmt.n(res.data.total)} ${res.data.total === 1 ? "match" : "matches"}${q || site ? "" : ` · ${fmt.n(all)} documents in all`}`;
    $("d-empty-text").textContent = `${all ? `${fmt.n(all)} documents are indexed. ` : ""}Pick one on the left to see who can read it, its versions, and to publish a new version.`;
  }
  $("d-list").innerHTML = docs.length
    ? docs.map((d) => {
      const kind = KIND_NAMES[d.kind] || (d.kind ? d.kind.toUpperCase() : "");
      const meta = [d.site, kind, d.edition].filter(Boolean).map((x) => `<span>${esc(x)}</span>`).join("");
      return `<button type="button" role="option" class="doc-item ${d.doc_id === selectedDoc ? "active" : ""}" data-doc="${esc(d.doc_id)}" aria-selected="${d.doc_id === selectedDoc}" title="${esc(d.title || d.doc_id)}\n${esc(d.doc_id)}">
        <span class="doc-kind k-${esc(d.kind || "doc")}" aria-hidden="true">${esc((kind || "DOC").slice(0, 3).toUpperCase())}</span>
        <span class="doc-text"><span class="t">${esc(shortTitle(d.title) || d.doc_id)}</span><span class="m">${meta || `<span>${esc(d.doc_id)}</span>`}</span></span></button>`;
    }).join("")
    : `<div class="empty-note">${res.ok ? "No document matches." : esc(errText(res))}</div>`;
}
$("d-list").onclick = (e) => { const b = e.target.closest("[data-doc]"); if (b) selectDoc(b.dataset.doc); };
function docPane(name) {
  document.querySelectorAll("#d-tabs [data-dpane]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.dpane === name)));
  document.querySelectorAll("#d-body [data-pane]").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== name));
  if (name === "versions") versionHistory();
}
$("d-tabs").onclick = (e) => { const b = e.target.closest("[data-dpane]"); if (b) docPane(b.dataset.dpane); };
async function selectDoc(doc) {
  selectedDoc = doc;
  document.querySelectorAll("#d-list [data-doc]").forEach((b) => { const on = b.dataset.doc === doc; b.classList.toggle("active", on); b.setAttribute("aria-selected", String(on)); });
  $("d-empty").classList.add("hidden"); $("d-body").classList.remove("hidden");
  const item = document.querySelector(`#d-list [data-doc="${CSS.escape(doc)}"]`);
  $("d-title").textContent = item ? item.title.split("\n")[0] : doc;
  ["p-msg", "v-msg"].forEach((id) => { $(id).textContent = ""; });
  const res = await api("GET", `/api/admin/policies/${encodeURIComponent(doc)}`);
  const mine = res.ok;
  $("d-tabs").classList.toggle("hidden", !mine);
  docPane("access");
  document.querySelectorAll("#d-body [data-pane]").forEach((p) => p.classList.toggle("hidden", !mine || p.dataset.pane !== "access"));
  const found = (await api("GET", `/api/admin/documents?q=${encodeURIComponent(doc)}&limit=10`));
  const info = (found.ok && found.data.documents.find((d) => d.doc_id === doc)) || {};
  $("d-link").classList.toggle("hidden", !info.url); if (info.url) $("d-link").href = info.url;
  if (!mine) {
    $("d-meta").textContent = `${doc} · ${res.status === 403 ? "owned by another department: you can see it in the chat, not change it" : errText(res)}`;
    return;
  }
  const owners = [res.data.owner, ...(res.data.co_owners || [])].map((d) => String(d).toUpperCase()).join(", ");
  $("d-meta").textContent = `${doc} · owned by ${owners} · ${res.data.visibility === "public" ? "public" : "restricted"}`;
  policyEditor.set(res.data);
}
function closeDoc() {
  selectedDoc = null;
  $("d-body").classList.add("hidden"); $("d-empty").classList.remove("hidden");
  document.querySelectorAll("#d-list [data-doc]").forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-selected", "false"); });
}
$("d-close").onclick = closeDoc;
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && tab === "documents" && selectedDoc && !document.querySelector("dialog[open]") && !openMenu) closeDoc();
});
const preview = previewer(() => policyEditor);
function showJson(p) { $("p-json").textContent = JSON.stringify(p, null, 2); }
$("p-save").onclick = async () => {
  const res = await api("PUT", `/api/admin/policies/${encodeURIComponent(selectedDoc)}`, policyEditor.get());
  setMsg("p-msg", res.ok, res.ok ? `Saved (policy version ${res.data.policy_version}). Applies within 60 s.` : errText(res));
  if (res.ok) selectDoc(selectedDoc);
};
$("p-revoke").onclick = async () => {
  if (!selectedDoc || !confirm(`Remove every access rule of ${selectedDoc}? Nobody will be able to read it.`)) return;
  const res = await api("DELETE", `/api/admin/policies/${encodeURIComponent(selectedDoc)}`);
  setMsg("p-msg", res.ok, res.ok ? `Revoked (policy version ${res.data.policy_version}).` : errText(res));
};
async function versionHistory() {
  const res = await api("GET", `/api/admin/documents/${encodeURIComponent(selectedDoc)}/versions`);
  if (!res.ok) { $("h-table").innerHTML = `<div class="empty-note">${esc(errText(res))}</div>`; return; }
  $("h-table").innerHTML = table(res.data.versions.slice().reverse(), [
    ["version", (v) => `v${esc(v.version)}`], ["valid from", (v) => fmt.time(v.valid_from)],
    ["valid to", (v) => (v.valid_to ? fmt.time(v.valid_to) : '<span class="pill good">current</span>')],
    ["chunks", (v) => fmt.n(v.chunks)],
  ]);
}
function fileBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}
const published = (d) => `Published version ${d.version} (${fmt.n(d.chunks)} chunks), searchable from ${fmt.time(d.visible_from)}.`;
$("v-send").onclick = async () => {
  const file = $("v-file").files[0];
  if (!file) { setMsg("v-msg", false, "Choose a file."); return; }
  if (file.size > 20 * 1024 * 1024) { setMsg("v-msg", false, "Files are limited to 20 MB."); return; }
  setMsg("v-msg", true, "Uploading and indexing…");
  const res = await api("POST", `/api/admin/documents/${encodeURIComponent(selectedDoc)}/version`, { filename: file.name, content_b64: await fileBase64(file) });
  setMsg("v-msg", res.ok, res.ok ? published(res.data) : errText(res));
};
$("r-save").onclick = async () => {
  const text = $("r-text").value;
  if (!text.trim()) { setMsg("v-msg", false, "Enter the new text."); return; }
  const res = await api("POST", `/api/admin/documents/${encodeURIComponent(selectedDoc)}/revise`, { text });
  setMsg("v-msg", res.ok, res.ok ? published(res.data) : errText(res));
};

// The upload dialog: a new document only. The id is checked as it is typed,
// and the server refuses an id in use anyway.
let idTaken = false; let idTimer = null;
const DOC_ID_RE = /^[a-z0-9][a-z0-9._-]{2,99}$/;
$("u-open").onclick = async () => {
  if (!uploadEditor) uploadEditor = PolicyEditor($("u-policy"), ctx, { simple: true, owners: OWNER_CHOICES() }, previewer(() => uploadEditor));
  const own = ctx.institute ? "institute" : (Array.isArray(ctx.manages) ? ctx.manages[0] : "");
  uploadEditor.set({ visibility: "restricted", owner: own });
  $("u-doc").value = ""; $("u-file").value = ""; $("u-msg").textContent = "";
  idTaken = false; showIdHint(""); $("dlg-upload").showModal();
};
$("u-cancel").onclick = () => $("dlg-upload").close();
function showIdHint(state, title) {
  const el = $("u-doc-hint");
  el.className = state === "taken" || state === "bad" ? "hint error" : state === "free" ? "hint ok" : "hint";
  el.textContent = {
    taken: `This id is already used${title ? ` by "${title}"` : ""}. Choose another one, or open that document to publish a new version.`,
    bad: 'Use 3 to 100 lower-case letters, digits, "-", "." or "_", starting with a letter or digit.',
    free: "This id is free.",
  }[state] || 'Lower-case letters, digits, "-", "." and "_". It must not be in use.';
  $("u-send").disabled = state === "taken" || state === "bad";
}
$("u-doc").oninput = () => {
  clearTimeout(idTimer); idTaken = false;
  const doc = $("u-doc").value.trim();
  if (!doc) { showIdHint(""); return; }
  if (!DOC_ID_RE.test(doc)) { showIdHint("bad"); return; }
  idTimer = setTimeout(async () => {
    const res = await api("GET", `/api/admin/documents/check?doc_id=${encodeURIComponent(doc)}`);
    if (!res.ok || $("u-doc").value.trim() !== doc) return;
    idTaken = res.data.exists; showIdHint(idTaken ? "taken" : "free", res.data.title);
  }, 300);
};
$("u-send").onclick = async () => {
  const file = $("u-file").files[0]; const doc = $("u-doc").value.trim();
  if (!doc || !file) { setMsg("u-msg", false, "Choose a document id and a file."); return; }
  if (idTaken) return;
  if (file.size > 20 * 1024 * 1024) { setMsg("u-msg", false, "Files are limited to 20 MB."); return; }
  const p = uploadEditor.get();
  if (!p.owner) { setMsg("u-msg", false, "Tick at least one owning department."); return; }
  setMsg("u-msg", true, "Uploading and indexing…");
  const body = {
    doc_id: doc, filename: file.name, content_b64: await fileBase64(file),
    owners: [p.owner, ...(p.co_owners || [])], visibility: p.visibility,
    groups: p.groups, departments: p.departments, roles: p.roles,
  };
  const res = await api("POST", "/api/admin/documents/upload", body);
  if (!res.ok) { if (res.status === 409) showIdHint("taken"); setMsg("u-msg", false, errText(res)); return; }
  $("dlg-upload").close();
  $("d-q").value = doc; await findDocs(); await selectDoc(doc);
  setMsg("p-msg", true, `Published ${res.data.doc_id}: ${published(res.data)} Owners: ${res.data.owners.map((d) => d.toUpperCase()).join(", ")}.`);
};

// ---------- requirements and models ----------
const VERDICTS = { pass: ["good", "check", "Pass"], fail: ["bad", "alert", "Fail"], pending: ["", "", "Pending"] };
function verdict(v) {
  const [cls, ic, label] = VERDICTS[v] || ["", "", v];
  return `<span class="pill ${cls}">${ic ? icon(ic) : ""}${esc(label)}</span>`;
}
// The scorecard: a summary, a filter, and one row per requirement. A row
// shows the measured values one by one ("label: value; ..." from the
// server), and where they come from in small print.
let reqRows = []; let reqView = "all";
async function requirements() {
  const res = await api("GET", "/api/admin/results");
  if (!res.ok) { $("req-list").innerHTML = `<div class="empty-note">${esc(errText(res))}</div>`; return; }
  reqRows = res.data.requirements;
  renderRequirements();
}
function metricsOf(value) {
  return String(value || "").split(/;\s*/).filter(Boolean).map((part) => {
    const i = part.lastIndexOf(":");
    return i > 0 ? { label: part.slice(0, i).trim(), value: part.slice(i + 1).trim() } : { label: "", value: part.trim() };
  });
}
function renderRequirements() {
  const rows = reqRows; const count = (v) => rows.filter((x) => x.verdict === v).length;
  const pass = count("pass"); const fail = count("fail"); const pending = count("pending");
  const share = rows.length ? Math.round((pass / rows.length) * 100) : 0;
  $("req-stats").innerHTML = `
    <div class="req-stat"><div class="label">Requirements met</div><div class="value">${pass}<small> of ${rows.length}</small></div>
      <div class="meter" role="img" aria-label="${share}% met"><i style="width:${share}%"></i></div></div>
    <div class="req-stat good"><div class="label">${icon("check")} Passing</div><div class="value">${pass}</div></div>
    <div class="req-stat bad"><div class="label">${icon("alert")} Failing</div><div class="value">${fail}</div></div>
    <div class="req-stat"><div class="label">Pending</div><div class="value">${pending}</div><div class="sub muted small">not measured yet</div></div>`;
  document.querySelectorAll("#req-tabs [data-rview]").forEach((b) => {
    const v = b.dataset.rview; b.setAttribute("aria-selected", String(v === reqView));
    b.querySelector(".count").textContent = v === "all" ? rows.length : count(v);
  });
  const shown = rows.filter((x) => reqView === "all" || x.verdict === reqView);
  const group = (title, list) => !list.length ? "" : `<div class="req-group"><h3>${title} <span class="muted">${list.length}</span></h3>${list.map(reqRow).join("")}</div>`;
  $("req-list").innerHTML = shown.length
    ? group("Functional requirements", shown.filter((x) => x.id.startsWith("FR"))) + group("Non-functional requirements", shown.filter((x) => !x.id.startsWith("FR")))
    : '<div class="empty-note">No requirement in this group.</div>';
}
function reqRow(x) {
  const metrics = metricsOf(x.value).map((m) =>
    m.label ? `<div class="req-metric"><span class="v">${esc(m.value)}</span><span class="l">${esc(m.label)}</span></div>`
      : `<div class="req-metric plain">${esc(m.value)}</div>`).join("");
  const sources = String(x.source || "").split(/,\s*/).filter(Boolean);
  return `<article class="req-row v-${esc(x.verdict)}">
    <div class="req-id">${esc(x.id)}</div>
    <div class="req-what"><div class="name">${esc(x.name)}</div><div class="target"><span class="muted">Target</span> ${esc(x.target)}</div></div>
    <div class="req-values">${metrics || '<span class="muted">Not measured yet</span>'}
      ${sources.length ? `<div class="req-source" title="${esc(sources.join("\n"))}">${icon("doc")}<span>${esc(sources.join(" · "))}</span></div>` : ""}</div>
    <div class="req-verdict">${verdict(x.verdict)}<div class="muted small">${x.date ? esc(x.date) : "–"}</div></div>
  </article>`;
}
$("req-tabs").onclick = (e) => { const b = e.target.closest("[data-rview]"); if (b) { reqView = b.dataset.rview; renderRequirements(); } };
const SHARE = new Set(["accuracy", "citation_correctness"]);
function metric(field, v) {
  if (v === null || v === undefined) return "–";
  if (SHARE.has(field)) return `${(v * 100).toFixed(1)}%`;
  return Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 });
}
async function models() {
  const res = await api("GET", "/api/admin/models");
  if (!res.ok) return;
  const d = res.data; const fields = Object.keys(d.labels);
  $("models-active").innerHTML = d.active ? `Serving now: <strong>${esc(d.active)}</strong>.` : "The serving model is unknown.";
  $("models-switch").textContent = d.switch;
  $("models-table").innerHTML = table(d.models, [
    ["model", (m) => `<strong>${esc(m.key)}</strong>${m.active ? ' <span class="pill good">Serving</span>' : ""}${m.gated ? ' <span class="muted small">licence needed</span>' : ""}`],
    ["size", (m) => esc(m.size_class)],
    ["status", (m) => (m.status === "measured" ? '<span class="pill accent">Measured</span>' : '<span class="pill">Pending</span>')],
    ...fields.map((f) => [d.labels[f], (m) => {
      const x = m.metrics[f];
      const mark = x.met === true ? `<span class="pill good">${icon("check")}${esc(metric(f, x.value))}</span>`
        : x.met === false ? `<span class="pill bad">${icon("alert")}${esc(metric(f, x.value))}</span>` : esc(metric(f, x.value));
      const target = x.target === null ? "" : `<div class="muted small">target ${esc(metric(f, x.target))}</div>`;
      return mark + target;
    }]),
  ]);
}

// ---------- grafana ----------
// The Grafana dashboard in a frame (kiosk mode, the page's theme). The
// address is the one this browser reaches Grafana at (see grafanaUrl).
async function grafanaPanel() {
  if (!grafanaBase) { const o = await api("GET", "/api/admin/overview?window_s=60"); if (o.ok) linksHtml((o.data.system || {}).links || {}); }
  const base = grafanaUrl();
  $("g-none").classList.toggle("hidden", !!base);
  $("g-frame").classList.toggle("hidden", !base);
  $("g-open").classList.toggle("hidden", !base || tab !== "grafana");
  if (!base) return;
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const url = `${base}/d/askiisc-main/ask-iisc?orgId=1&kiosk&refresh=30s&theme=${dark ? "dark" : "light"}`;
  $("g-open").href = `${base}/d/askiisc-main/ask-iisc`;
  if ($("g-frame").dataset.src !== url) { $("g-frame").dataset.src = url; $("g-frame").src = url; }
}

// ---------- refresh ----------
async function refresh(force) {
  if (!force && !$("auto").checked) return;
  if (tab === "overview") await overview();
  else if (tab === "people" && force) await peoplePanel();
  else if (tab === "requests" && force) { reqPage.reset(); await requests(); }
  else if (tab === "documents" && force) await documentsPanel();
  else if (tab === "grafana" && force) await grafanaPanel();
  else if (tab === "requirements" && force) await requirements();
  else if (tab === "models" && force) await models();
}

async function start() {
  fillIcons();
  const res = await api("GET", "/api/me");
  if (!res.ok || !res.data.is_admin) { $("denied").classList.remove("hidden"); return; }
  me = res.data;
  $("shell").classList.remove("hidden");
  $("who").innerHTML = `<strong>${esc(me.user_id)}</strong>${me.institute_admin ? "Institute admin" : `${esc(me.department.toUpperCase())} admin`}`;
  const first = (location.hash || "").slice(1);
  show(TITLES[first] ? first : "overview");
  setInterval(() => refresh(false), 10000);
}
start();
