"use strict";
// Shared by every Ask IISc page: escaping, API calls, icons, theme.
// Plain JS, no build step, no external files.

const $ = (id) => document.getElementById(id);
const HDR = { "Content-Type": "application/json", "X-AskIISc": "1" };

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

// One JSON call; `onUnauthorized` decides what a 401 does on each page.
const apiHooks = { onUnauthorized: null };
async function api(method, path, body) {
  const r = await fetch(path, {
    method, headers: HDR, credentials: "same-origin",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data = null;
  try { data = await r.json(); } catch (_) { /* empty body */ }
  if (r.status === 401 && apiHooks.onUnauthorized) apiHooks.onUnauthorized(path);
  return { ok: r.ok, status: r.status, data: data || {} };
}
function errText(res) {
  const d = res.data || {};
  if (typeof d.error === "string") return d.error;
  if (typeof d.detail === "string") return d.detail;
  if (Array.isArray(d.detail) && d.detail.length) return d.detail.map((x) => x.msg || String(x)).join("; ");
  return `error ${res.status}`;
}

// Icons: 24x24 stroke paths (drawn for this app).
const ICONS = {
  plus: "M12 5v14M5 12h14",
  x: "M7 7l10 10M17 7L7 17",
  edit: "M4 20h4L19 9l-4-4L4 16v4zM14 6l4 4",
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
  search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM20 20l-4-4",
  send: "M12 19V5M5 12l7-7 7 7",
  stop: "M7 7h10v10H7z",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  up: "M7 11v9H4v-9h3zM7 11l4-8a2 2 0 0 1 2 2v4h6a2 2 0 0 1 2 2l-2 7a2 2 0 0 1-2 2H7",
  down: "M7 13V4H4v9h3zM7 13l4 8a2 2 0 0 0 2-2v-4h6a2 2 0 0 0 2-2l-2-7a2 2 0 0 0-2-2H7",
  redo: "M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7",
  download: "M12 4v11M7 10l5 5 5-5M5 20h14",
  sidebar: "M4 5h16v14H4zM9 5v14",
  menu: "M4 7h16M4 12h16M4 17h16",
  ghost: "M6 20V10a6 6 0 0 1 12 0v10l-2-2-2 2-2-2-2 2-2-2-2 2zM10 11h.01M14 11h.01",
  settings: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z",
  moon: "M21 13A9 9 0 1 1 11 3a7 7 0 0 0 10 10z",
  logout: "M15 4h4v16h-4M10 8l-4 4 4 4M6 12h11",
  shield: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z",
  doc: "M7 3h7l5 5v13H7zM14 3v5h5",
  users: "M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM2 21a7 7 0 0 1 14 0M17 11a3 3 0 1 0 0-6M22 21a6 6 0 0 0-5-6",
  chart: "M4 20V10M10 20V4M16 20v-7M22 20H2",
  list: "M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01",
  lock: "M6 11h12v10H6zM8 11V7a4 4 0 0 1 8 0v4",
  bolt: "M13 2L4 14h7l-1 8 9-12h-7l1-8z",
  check: "M5 12l5 5L20 7",
  cpu: "M7 7h10v10H7zM10 2v3M14 2v3M10 19v3M14 19v3M2 10h3M2 14h3M19 10h3M19 14h3",
  log: "M5 4h14v16H5zM9 8h6M9 12h6M9 16h4",
  chat: "M4 5h16v11H8l-4 4z",
  alert: "M12 3l10 18H2L12 3zM12 10v4M12 17h.01",
  upload: "M12 20V9M7 14l5-5 5 5M5 4h14",
};
function icon(name, cls) {
  return `<svg class="i ${cls || ""}" viewBox="0 0 24 24" aria-hidden="true"><path d="${ICONS[name] || ""}"/></svg>`;
}
function fillIcons(root) {
  (root || document).querySelectorAll("[data-icon]").forEach((el) => {
    el.insertAdjacentHTML("afterbegin", icon(el.dataset.icon));
    el.removeAttribute("data-icon");
  });
}

// Theme: follows the system until the user picks one (kept per browser).
function setTheme(t) {
  if (t) document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem("askiisc-theme", t || ""); } catch (_) { /* private mode */ }
}
function toggleTheme() {
  const cur = document.documentElement.dataset.theme;
  const dark = cur === "dark" || (!cur && matchMedia("(prefers-color-scheme: dark)").matches);
  setTheme(dark ? "light" : "dark");
}
try { const t = localStorage.getItem("askiisc-theme"); if (t) setTheme(t); } catch (_) { /* ignore */ }

function initials(userId) {
  const name = String(userId || "?").split("@")[0].replace(/[^a-zA-Z0-9]+/g, " ").trim();
  const parts = name.split(" ").filter(Boolean);
  return ((parts[0] || "?")[0] + (parts[1] ? parts[1][0] : "")).toUpperCase();
}
function displayName(userId) {
  const id = String(userId || "");
  const name = id.includes("@") ? id.split("@")[0].split(/[._-]+/)[0] : "";
  if (name.length < 2) return "";
  return name.charAt(0).toUpperCase() + name.slice(1);
}
