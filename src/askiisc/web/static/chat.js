"use strict";
// Ask IISc chat client. Plain JS, no build step. All text from the server
// is escaped before any formatting is applied. Needs ui.js.

const state = {
  me: null, chats: [], chatId: null, temporary: false, tempHistory: [],
  messages: [], controller: null, renaming: null, setToken: null,
};
apiHooks.onUnauthorized = (path) => { if (!path.startsWith("/api/auth/")) showAuth(); };

// ---------- markdown (on escaped text) ----------
function inline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
}
function markdown(text) {
  const lines = esc(text).split("\n");
  const out = []; let list = null; let para = []; let code = null;
  const flush = () => { if (para.length) { out.push(`<p>${inline(para.join("<br>"))}</p>`); para = []; } };
  const close = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of lines) {
    if (code !== null) {
      if (/^\s*```/.test(raw)) { out.push(`<pre><code>${code.join("\n")}</code></pre>`); code = null; } else code.push(raw);
      continue;
    }
    const line = raw.trimEnd();
    if (/^\s*```/.test(line)) { flush(); close(); code = []; continue; }
    const head = line.match(/^(#{1,4})\s+(.*)$/);
    const ul = line.match(/^\s*[-*•]\s+(.*)$/); const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (head) { flush(); close(); out.push(`<h${head[1].length < 3 ? 3 : 4}>${inline(head[2])}</h${head[1].length < 3 ? 3 : 4}>`); }
    else if (ul || ol) {
      flush(); const kind = ul ? "ul" : "ol";
      if (list !== kind) { close(); out.push(`<${kind}>`); list = kind; }
      out.push(`<li>${inline((ul || ol)[1])}</li>`);
    } else if (!line.trim()) { flush(); close(); }
    else { close(); para.push(line); }
  }
  if (code !== null) out.push(`<pre><code>${code.join("\n")}</code></pre>`);
  flush(); close();
  return out.join("");
}
// Replace [chunk:a, chunk:b] marks with numbered links to the sources.
function withCitations(text, citations) {
  const number = new Map();
  (citations || []).forEach((c, i) => number.set(c.chunk_id, i + 1));
  const marked = text.replace(/\s*\[(chunk:[^\]]+)\]/g, (_, inner) => {
    const ids = inner.split(/[,;]\s*/).map((x) => x.trim().replace(/^chunk:/, ""));
    const nums = [...new Set(ids.map((id) => number.get(id)).filter(Boolean))];
    return nums.map((n) => `\u0000${n}\u0001`).join("");
  });
  return markdown(marked).replace(/\u0000(\d+)\u0001/g,
    (_, n) => `<sup><a href="#" data-src="${n}">${n}</a></sup>`);
}
function sourcesHtml(citations) {
  if (!citations || !citations.length) return "";
  const cards = citations.map((c, i) => {
    const title = esc(c.title || c.doc_id);
    const href = c.link || c.url;
    const t = href ? `<a href="${esc(href)}" target="_blank" rel="noopener noreferrer" title="Open the cited passage">${title}</a>` : title;
    const where = c.kind === "pdf" ? `PDF, page ${esc(c.page)}` : c.kind === "html" ? "Web page" : `Page ${esc(c.page)}`;
    const site = c.site ? `${esc(c.site)}, ` : "";
    const stale = c.is_stale ? ', <span class="stale">superseded</span>' : "";
    // Yearly documents name their edition; others their version.
    const edition = c.edition ? `edition ${esc(c.edition)}` : `version ${esc(c.version)}`;
    const archived = c.archived ? ", archived copy" : "";
    return `<div class="source" data-n="${i + 1}"><span class="num">${i + 1}</span><div class="t">${t}</div>
      <div class="m">${site}${where}, ${edition}${archived}${stale}</div></div>`;
  }).join("");
  return `<div class="sources-label">Sources</div><div class="sources">${cards}</div>`;
}

// ---------- auth ----------
// Sign in with e-mail and password. A reset link opens the page at
// /#set-password=<token>, a registration link at /#confirm=<token>; the
// token is read once, taken out of the address bar (and history), and
// never stored.
const VIEWS = ["login-form", "register-form", "forgot-form", "set-form"];
function showView(id) { VIEWS.forEach((v) => $(v).classList.toggle("hidden", v !== id)); }
function showAuth(view) {
  $("app").classList.add("hidden"); $("auth").classList.remove("hidden");
  showView(view || (state.setToken ? "set-form" : "login-form"));
}
document.querySelectorAll("[data-to]").forEach((b) => { b.onclick = () => showView(`${b.dataset.to}-form`); });
$("to-forgot").onclick = async () => {
  showView("forgot-form"); $("forgot-msg").textContent = "";
  $("forgot-user").value = $("login-user").value;
  const opts = await api("GET", "/api/auth/options");
  const on = opts.ok && opts.data.reset_by_email;
  $("forgot-field").classList.toggle("hidden", !on); $("forgot-send").classList.toggle("hidden", !on);
  $("forgot-sub").textContent = on
    ? "Enter your IISc e-mail. We send it a link to choose a new password; it works once, for 10 minutes."
    : "Password reset by e-mail is not set up yet. Ask your department admin for a link.";
};
$("to-register").onclick = () => {
  showView("register-form"); $("reg-msg").className = "error"; $("reg-msg").textContent = "";
  $("reg-user").value = $("login-user").value; $("reg-send").disabled = false;
};
$("register-form").onsubmit = async (e) => {
  e.preventDefault(); const el = $("reg-msg"); el.className = "error"; el.textContent = "";
  const password = $("reg-password").value;
  if (password !== $("reg-again").value) { el.textContent = "The two passwords do not match."; return; }
  if (password.length < 8) { el.textContent = "The password needs at least 8 characters (at most 256)."; return; }
  $("reg-send").disabled = true;
  const res = await api("POST", "/api/auth/register",
    { user_id: $("reg-user").value, password, confirm: $("reg-again").value });
  $("reg-send").disabled = false;
  if (!res.ok) { el.textContent = errText(res); return; }
  $("reg-password").value = ""; $("reg-again").value = "";
  el.className = "ok"; el.textContent = res.data.message;
};
async function confirmAccount(token) {
  showAuth("login-form"); const el = $("login-error");
  const res = await api("POST", "/api/auth/register/confirm", { token });
  if (!res.ok) { el.className = "error"; el.textContent = errText(res); return; }
  $("login-user").value = res.data.user_id || "";
  el.className = "ok"; el.textContent = "Your account is ready. Sign in with your password.";
}
document.querySelectorAll("[data-show]").forEach((b) => {
  b.onclick = () => {
    const input = $(b.dataset.show); const show = input.type === "password";
    input.type = show ? "text" : "password"; b.textContent = show ? "Hide" : "Show";
  };
});
$("login-form").onsubmit = async (e) => {
  e.preventDefault(); $("login-error").className = "error"; $("login-error").textContent = "";
  const res = await api("POST", "/api/auth/login",
    { user_id: $("login-user").value, password: $("current-password").value });
  $("current-password").value = "";
  if (!res.ok) { $("login-error").textContent = errText(res); return; }
  start();
};
$("forgot-form").onsubmit = async (e) => {
  e.preventDefault(); const el = $("forgot-msg"); el.className = "error";
  const res = await api("POST", "/api/auth/forgot", { user_id: $("forgot-user").value });
  if (!res.ok) { el.textContent = errText(res); return; }
  el.className = "ok"; el.textContent = res.data.message;
};
$("set-form").onsubmit = async (e) => {
  e.preventDefault(); const el = $("set-msg"); el.className = "error"; el.textContent = "";
  if ($("set-password").value !== $("set-again").value) { el.textContent = "The two passwords differ."; return; }
  const res = await api("POST", "/api/auth/set-password", { token: state.setToken, password: $("set-password").value });
  if (!res.ok) { el.textContent = errText(res); return; }
  state.setToken = null; $("set-password").value = ""; $("set-again").value = "";
  showView("login-form"); $("login-user").value = res.data.user_id || "";
  $("login-error").className = "ok"; $("login-error").textContent = "Password set. Sign in with it now.";
};
$("logout").onclick = async () => { await api("POST", "/api/auth/logout"); location.reload(); };

// ---------- sidebar ----------
function dayGroup(t) {
  const d = new Date(t * 1000); const now = new Date();
  const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((day(now) - day(d)) / 86400000);
  if (diff <= 0) return "Today";
  if (diff === 1) return "Yesterday";
  if (diff < 7) return "Previous 7 days";
  if (diff < 30) return "Previous 30 days";
  return d.toLocaleDateString([], { month: "long", year: "numeric" });
}
async function loadChats() {
  const q = encodeURIComponent($("search").value || "");
  const res = await api("GET", `/api/chats?q=${q}`);
  state.chats = res.ok ? res.data.chats : [];
  renderChatList();
}
function renderChatList() {
  const nav = $("chat-list"); nav.innerHTML = "";
  if (!state.chats.length) {
    nav.innerHTML = `<div class="empty-list">${$("search").value ? "No chats match." : "Your chats will appear here."}</div>`;
    return;
  }
  let group = null;
  for (const c of state.chats) {
    const g = dayGroup(c.updated);
    if (g !== group) { group = g; nav.insertAdjacentHTML("beforeend", `<div class="chat-group">${esc(g)}</div>`); }
    const row = document.createElement("div");
    row.className = "chat-item" + (c.chat_id === state.chatId ? " active" : "");
    row.innerHTML = `<span class="title">${esc(c.title)}</span><span class="actions">
      <button class="icon" data-act="rename" title="Rename">${icon("edit")}</button>
      <button class="icon" data-act="delete" title="Delete">${icon("trash")}</button></span>`;
    row.onclick = (e) => {
      const btn = e.target.closest("[data-act]");
      if (btn && btn.dataset.act === "rename") { e.stopPropagation(); rename(c); }
      else if (btn && btn.dataset.act === "delete") { e.stopPropagation(); removeChat(c); }
      else openChat(c.chat_id);
    };
    nav.appendChild(row);
  }
}
function closeSideOnMobile() { $("side").classList.remove("open"); }
$("menu").onclick = (e) => { e.stopPropagation(); $("side").classList.toggle("open"); };
// On phones the open sidebar covers part of the chat: a tap on the chat closes it.
document.querySelector(".main").addEventListener("click", () => closeSideOnMobile());
$("hide-side").onclick = () => {
  if (matchMedia("(max-width: 760px)").matches) closeSideOnMobile();
  else $("app").classList.add("collapsed");
};
$("show-side").onclick = () => $("app").classList.remove("collapsed");
$("me-btn").onclick = (e) => { e.stopPropagation(); $("me-menu").classList.toggle("hidden"); };
document.addEventListener("click", (e) => { if (!e.target.closest("#me-menu")) $("me-menu").classList.add("hidden"); });
$("theme").onclick = () => toggleTheme();
let searchTimer = null;
$("search").oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadChats, 250); };

// ---------- chats ----------
async function openChat(chatId) {
  stopStream(); state.temporary = false;
  const res = await api("GET", `/api/chats/${chatId}`);
  if (!res.ok) { alert(errText(res)); return; }
  state.chatId = chatId; state.messages = res.data.messages;
  $("chat-name").textContent = res.data.title;
  renderThread(); renderChatList(); closeSideOnMobile();
}
function newChat(temporary) {
  stopStream();
  state.chatId = null; state.temporary = temporary; state.tempHistory = []; state.messages = [];
  $("chat-name").textContent = temporary ? "Temporary chat" : "New chat";
  renderThread(); renderChatList(); $("question").focus(); closeSideOnMobile();
}
// Save (button or Enter) renames; Cancel, Esc or a click outside do not.
function rename(chat) {
  state.renaming = chat; $("rename-input").value = chat.title;
  $("dlg-rename").showModal(); $("rename-input").select();
}
$("dlg-rename").addEventListener("close", () => { state.renaming = null; });
$("rename-cancel").onclick = () => $("dlg-rename").close();
$("rename-form").onsubmit = async (e) => {
  e.preventDefault();
  const chat = state.renaming; const title = $("rename-input").value.trim();
  if (!chat || !title) return;
  const res = await api("PATCH", `/api/chats/${chat.chat_id}`, { title });
  if (!res.ok) { alert(errText(res)); return; }
  $("dlg-rename").close();
  if (chat.chat_id === state.chatId) $("chat-name").textContent = title;
  loadChats();
};
async function removeChat(chat) {
  if (!confirm(`Delete "${chat.title}"? This cannot be undone.`)) return;
  const res = await api("DELETE", `/api/chats/${chat.chat_id}`);
  if (!res.ok) { alert(errText(res)); return; }
  if (chat.chat_id === state.chatId) newChat(false);
  loadChats();
}
$("new-chat").onclick = () => newChat(false);
$("temp-chat").onclick = () => newChat(true);
$("export").onclick = () => {
  if (!state.chatId) { alert(state.temporary ? "Temporary chats are not saved." : "Send a message first."); return; }
  location.href = `/api/chats/${state.chatId}/export`;
};

// ---------- thread ----------
const SUGGESTIONS = [
  ["M.Tech credit requirements", "What are the minimum credits for an M.Tech degree?"],
  ["Applying for a PhD", "How do I apply for a PhD at CDS?"],
  ["Attendance rules", "What is the attendance rule for courses?"],
  ["Hostel fees", "What are the hostel fees and how are they paid?"],
];
function hour() { const h = new Date().getHours(); return h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening"; }
function placeComposer(centered) {
  const form = $("ask-form"); const composer = $("composer");
  if (centered && form.parentElement !== $("composer-slot")) { $("composer-slot").appendChild(form); composer.classList.add("hidden"); }
  if (!centered && form.parentElement !== composer) { composer.prepend(form); composer.classList.remove("hidden"); }
}
function renderThread() {
  const t = $("thread"); t.innerHTML = "";
  $("temp-pill").classList.toggle("hidden", !state.temporary);
  const empty = !state.messages.length;
  $("welcome").classList.toggle("hidden", !empty);
  placeComposer(empty);
  if (empty) {
    const who = displayName(state.me && state.me.user_id);
    $("greeting").textContent = state.temporary ? "Temporary chat" : who ? `${hour()}, ${who}` : hour();
    $("welcome-sub").textContent = state.temporary
      ? "Nothing here is saved, and it disappears when you leave."
      : "Ask about academic rules, courses, hostels, admissions or your department.";
    $("suggest").innerHTML = SUGGESTIONS.map(([title, q]) =>
      `<button type="button" data-q="${esc(q)}"><strong>${esc(title)}</strong><span>${esc(q)}</span></button>`).join("");
    $("suggest").querySelectorAll("[data-q]").forEach((b) => { b.onclick = () => { $("question").value = b.dataset.q; send(); }; });
    return;
  }
  state.messages.forEach((m, i) => t.appendChild(messageEl(m, i === state.messages.length - 1)));
  scrollDown();
}
function messageEl(m, last) {
  const el = document.createElement("div");
  el.className = `msg ${m.role}`;
  if (m.role === "user") {
    el.innerHTML = `<div class="bubble">${markdown(m.content)}</div>`;
    return el;
  }
  el.innerHTML = '<span class="mark" aria-hidden="true">A</span><div class="body"></div>';
  const body = el.querySelector(".body");
  const answer = document.createElement("div"); answer.className = "answer";
  answer.innerHTML = withCitations(m.content, m.meta.citations);
  body.appendChild(answer);
  if (m.meta.stale) body.insertAdjacentHTML("beforeend", `<div class="stale-note">${icon("alert")} This answer cites a superseded version of a document.</div>`);
  body.insertAdjacentHTML("beforeend", sourcesHtml(m.meta.citations));
  body.addEventListener("click", (e) => {
    const n = e.target.dataset && e.target.dataset.src;
    if (!n) return;
    e.preventDefault();
    const card = body.querySelector(`.source[data-n="${n}"]`);
    if (card) { card.scrollIntoView({ block: "nearest", inline: "nearest" }); card.classList.add("flash"); setTimeout(() => card.classList.remove("flash"), 900); }
  });
  if (!m.meta.hidden) body.appendChild(toolsEl(m, last));
  return el;
}
function toolsEl(m, last) {
  const tools = document.createElement("div"); tools.className = "msg-tools";
  tools.appendChild(button("copy", "Copy", (b) => {
    navigator.clipboard.writeText(m.content.replace(/\s*\[chunk:[^\]]+\]/g, ""));
    b.innerHTML = icon("check"); setTimeout(() => { b.innerHTML = icon("copy"); }, 1200);
  }));
  if (m.message_id) {
    tools.appendChild(button("up", "Helpful", () => rate(m, 1), m.rating === 1));
    tools.appendChild(button("down", "Not helpful", () => rate(m, -1), m.rating === -1));
    if (last && !state.temporary) tools.appendChild(button("redo", "Regenerate", regenerate));
  }
  const tags = [];
  if (m.meta.abstained) tags.push("Not found in the documents");
  if (m.meta.cache_hit) tags.push("Answered from cache");
  const tag = document.createElement("span"); tag.className = "tag"; tag.textContent = tags.join(", ");
  tools.appendChild(tag);
  return tools;
}
function button(name, title, fn, on) {
  const b = document.createElement("button"); b.type = "button";
  b.className = "icon" + (on ? " on" : ""); b.innerHTML = icon(name); b.title = title;
  b.onclick = () => fn(b);
  return b;
}
async function rate(m, value) {
  const next = m.rating === value ? 0 : value;
  const res = await api("POST", `/api/messages/${m.message_id}/rating`, { rating: next });
  if (res.ok) { m.rating = next; renderThread(); }
}
function scrollDown() { const s = $("scroller"); s.scrollTop = s.scrollHeight; }
$("scroller").addEventListener("scroll", () => {
  document.querySelector(".topbar").classList.toggle("scrolled", $("scroller").scrollTop > 4);
});

// ---------- sending ----------
async function ensureChat() {
  if (state.chatId || state.temporary) return true;
  const res = await api("POST", "/api/chats", {});
  if (!res.ok) { alert(errText(res)); return false; }
  state.chatId = res.data.chat_id; return true;
}
async function send() {
  const q = $("question").value.trim();
  if (!q || state.controller) return;
  if (!(await ensureChat())) return;
  $("question").value = ""; autosize();
  state.messages.push({ role: "user", content: q, meta: {} });
  renderThread();
  let path, body;
  if (state.temporary) {
    path = "/api/temporary"; body = { question: q, history: state.tempHistory.slice(-6) };
  } else {
    path = `/api/chats/${state.chatId}/messages`; body = { question: q };
  }
  await stream(path, body, q);
}
async function regenerate() {
  if (!state.chatId || state.controller) return;
  state.messages = state.messages.slice(0, -1); renderThread();
  await stream(`/api/chats/${state.chatId}/regenerate`, undefined, null);
}
function liveStatus(text) {
  return `<div class="status"><span class="dots"><i></i><i></i><i></i></span>${esc(text)}</div>`;
}
async function stream(path, body, question) {
  const live = document.createElement("div"); live.className = "msg assistant";
  live.innerHTML = `<span class="mark" aria-hidden="true">A</span><div class="body"><div class="answer">${liveStatus("Searching the documents…")}</div></div>`;
  $("thread").appendChild(live); scrollDown();
  const out = live.querySelector(".answer");
  state.controller = new AbortController(); busy(true);
  let text = ""; let answer = null; let saved = null; let failure = null;
  try {
    const r = await fetch(path, {
      method: "POST", headers: HDR, credentials: "same-origin", signal: state.controller.signal,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!r.ok) {
      let data = {}; try { data = await r.json(); } catch (_) { /* not json */ }
      if (r.status === 429) failure = `The service is busy. Please retry in ${r.headers.get("Retry-After") || "a few"} s.`;
      else if (r.status === 401) { showAuth(); return; }
      else failure = data.error || data.detail || `error ${r.status}`;
    } else {
      const reader = r.body.getReader(); const decoder = new TextDecoder(); let buf = "";
      for (;;) {
        const { value, done } = await reader.read(); if (done) break;
        buf += decoder.decode(value, { stream: true });
        let cut;
        while ((cut = buf.indexOf("\n\n")) >= 0) {
          const block = buf.slice(0, cut); buf = buf.slice(cut + 2);
          const ev = (block.match(/^event: (.*)$/m) || [])[1];
          const raw = (block.match(/^data: (.*)$/m) || [])[1];
          if (!ev || raw === undefined) continue;
          const data = JSON.parse(raw);
          if (ev === "warming") out.innerHTML = liveStatus(`A GPU worker is starting (up to ${Math.round(data.max_wait_s)} s). Your question is queued…`);
          else if (ev === "token") { text += data.text; out.innerHTML = withCitations(text, []); scrollDown(); }
          else if (ev === "answer") answer = data;
          else if (ev === "saved") saved = data;
          else if (ev === "error") failure = `Something went wrong (${data.error}). Please try again.`;
        }
      }
    }
  } catch (e) {
    failure = e.name === "AbortError" ? "Stopped." : "Connection lost. Please try again.";
  } finally { state.controller = null; busy(false); }
  if (answer) {
    const msg = { role: "assistant", content: answer.text, meta: answer, message_id: saved && saved.answer_id, rating: 0 };
    if (saved) {
      const lastUser = [...state.messages].reverse().find((m) => m.role === "user");
      if (lastUser) lastUser.message_id = saved.question_id;
      $("chat-name").textContent = saved.title;
    }
    state.messages.push(msg);
    if (state.temporary && question) {
      state.tempHistory.push({ role: "user", content: question }, { role: "assistant", content: answer.text });
    }
    renderThread();
  } else {
    out.innerHTML = `<div class="status">${esc(failure || "No answer.")}</div>`;
  }
  if (!state.temporary) loadChats();
}
function stopStream() { if (state.controller) state.controller.abort(); }
function busy(on) {
  $("send").classList.toggle("hidden", on); $("stop").classList.toggle("hidden", !on);
}
$("stop").onclick = stopStream;
$("ask-form").onsubmit = (e) => { e.preventDefault(); send(); };
function autosize() { const q = $("question"); q.style.height = "auto"; q.style.height = Math.min(q.scrollHeight, 220) + "px"; }
$("question").addEventListener("input", autosize);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (e.isComposing || e.keyCode === 229) return; // IME still composing
    send();
  }
});

// ---------- settings ----------
$("settings").onclick = () => {
  $("pw-error").textContent = ""; $("del-error").textContent = "";
  $("pw-old").value = ""; $("pw-new").value = ""; $("del-pass").value = "";
  const me = state.me;
  $("account-info").innerHTML = `<strong>${esc(me.user_id)}</strong><br><span class="muted">${esc(describe(me))}</span>`;
  $("dlg-settings").showModal();
};
$("settings-close").onclick = () => $("dlg-settings").close();
$("pw-form").onsubmit = async (e) => {
  e.preventDefault();
  const res = await api("POST", "/api/me/password", { old: $("pw-old").value, new: $("pw-new").value });
  if (!res.ok) { $("pw-error").textContent = errText(res); return; }
  alert("Password changed. Sign in again with the new password."); location.reload();
};
$("delete-all").onclick = async () => {
  if (!confirm("Delete all your chats? This cannot be undone.")) return;
  const res = await api("DELETE", "/api/chats");
  if (!res.ok) { alert(errText(res)); return; }
  newChat(false); loadChats(); $("dlg-settings").close();
};
$("del-form").onsubmit = async (e) => {
  e.preventDefault();
  if (!confirm("Delete your account and every chat?")) return;
  const res = await api("DELETE", "/api/me", { password: $("del-pass").value });
  if (!res.ok) { $("del-error").textContent = errText(res); return; }
  location.reload();
};

// ---------- start ----------
const ROLE_NAMES = { student: "Student", faculty: "Faculty", staff: "Staff", admin: "Admin" };
function describe(me) {
  const role = ROLE_NAMES[me.role] || me.role;
  return `${role}, ${me.department.toUpperCase()}${me.program_name ? `, ${me.program_name}` : ""}`;
}
async function start() {
  const res = await api("GET", "/api/me");
  if (!res.ok) { showAuth(); return; }
  state.me = res.data;
  $("auth").classList.add("hidden"); $("app").classList.remove("hidden");
  $("avatar").textContent = initials(state.me.user_id);
  $("who-name").textContent = state.me.user_id;
  $("who-meta").textContent = describe(state.me);
  $("admin-link").classList.toggle("hidden", !state.me.is_admin);
  newChat(false); loadChats();
}
const setMatch = location.hash.match(/^#set-password=([\w-]+)$/);
const confirmMatch = location.hash.match(/^#confirm=([\w-]+)$/);
if (setMatch || confirmMatch) history.replaceState(null, "", "/");
if (setMatch) { state.setToken = setMatch[1]; showAuth("set-form"); }
else if (confirmMatch) confirmAccount(confirmMatch[1]);
else start();
fillIcons();
