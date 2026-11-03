"use strict";
// A form for one document's access policy (DocPolicy), used by the
// Documents tab (the selected document, and the upload dialog). Needs ui.js.
//
// A person may read a restricted document if ANY rule matches: their
// department, their role, one of their groups, or their user id. Every
// person also has derived groups: "<dept>-students|faculty|staff|admins",
// their programme ("phd") and "<dept>-<programme>" ("csa-phd"), so a
// combination such as "CSA PhD students" is one group.

const ROLE_STEMS = { student: "students", faculty: "faculty", staff: "staff", admin: "admins" };

function chipInput(root, placeholder, onChange) {
  const values = new Set();
  root.className = "chips";
  const input = document.createElement("input"); input.placeholder = placeholder;
  const render = () => {
    root.querySelectorAll(".chip").forEach((c) => c.remove());
    for (const v of values) {
      const chip = document.createElement("span"); chip.className = "chip";
      chip.innerHTML = `${esc(v)}<button type="button" title="Remove">×</button>`;
      chip.querySelector("button").onclick = () => { values.delete(v); render(); onChange(); };
      root.insertBefore(chip, input);
    }
  };
  const add = () => {
    for (const v of input.value.split(",").map((x) => x.trim().toLowerCase()).filter(Boolean)) values.add(v);
    input.value = ""; render(); onChange();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === ",") { e.preventDefault(); add(); }
    else if (e.key === "Backspace" && !input.value && values.size) { values.delete([...values].pop()); render(); onChange(); }
  });
  input.addEventListener("blur", () => { if (input.value.trim()) add(); });
  root.appendChild(input);
  return {
    get: () => [...values],
    set: (list) => { values.clear(); (list || []).forEach((v) => values.add(v)); render(); },
    add: (v) => { values.add(v); render(); onChange(); },
  };
}

function checkList(root, options, onChange) {
  root.className = "checks";
  root.innerHTML = Object.entries(options).map(([key, label]) =>
    `<label><input type="checkbox" value="${esc(key)}"> ${esc(label)}</label>`).join("");
  root.querySelectorAll("input").forEach((i) => { i.onchange = onChange; });
  return {
    get: () => [...root.querySelectorAll("input:checked")].map((i) => i.value),
    set: (list) => { const s = new Set(list || []); root.querySelectorAll("input").forEach((i) => { i.checked = s.has(i.value); }); },
  };
}

function toLocal(t) {
  if (!t) return "";
  const d = new Date(t * 1000); d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 16);
}
function fromLocal(v) { return v ? new Date(v).getTime() / 1000 : null; }

// ctx: {departments: [...], programs: {key: label}}; opts.simple hides
// named users and dates (the upload form only takes groups, departments
// and roles); opts.owners lists the departments that may own documents and
// shows them as checkboxes (the owner and co-owners, whose admins manage
// it). onChange(policy) is called on every edit.
function PolicyEditor(root, ctx, opts, onChange) {
  const simple = !!(opts && opts.simple);
  const name = `vis-${Math.random().toString(36).slice(2)}`;
  const ownerOptions = (opts && opts.owners) || null;
  root.innerHTML = `
    ${ownerOptions ? '<div class="field"><div class="label">Owning departments <span class="hint">their admins manage the document</span></div><div data-owners></div></div>' : ""}
    <div class="field"><div class="seg">
      <label><input type="radio" name="${name}" value="restricted" checked> Restricted</label>
      <label><input type="radio" name="${name}" value="public"> Public (every IISc user)</label></div></div>
    <div data-restricted>
      <p class="hint" style="margin:0 0 12px">A person can read the document if <strong>any</strong> rule below matches. To combine two conditions (for example CSA <em>and</em> PhD), add a combined group.</p>
      <div class="policy-grid">
        <div class="field"><div class="label">Departments (everyone in them)</div><div data-depts></div></div>
        <div class="field"><div class="label">Roles (in any department)</div><div data-roles></div></div>
        <div class="field"><div class="label">Programmes (in any department)</div><div data-progs></div></div>
        <div class="field"><div class="label">Combined groups</div>
          <div class="toolbar" style="margin-bottom:6px"><select data-cdept></select><select data-cwhat></select><button type="button" data-cadd>Add</button></div>
          <div data-combined></div></div>
        <div class="field"><div class="label">Other groups</div><div data-groups></div></div>
        <div class="field" data-full><div class="label">Also allow these people</div><div data-allow></div></div>
      </div>
    </div>
    <div class="policy-grid" data-full>
      <div class="field"><div class="label">Never allow these people (even if public)</div><div data-deny></div></div>
      <div class="field"><div class="label">Readable between</div>
        <div class="toolbar" style="margin:0"><input type="datetime-local" data-from><input type="datetime-local" data-until></div>
        <div class="hint">Leave empty for no limit.</div></div>
    </div>
    <div class="preview" data-preview>Who can read it: …</div>`;
  const q = (sel) => root.querySelector(sel);
  const programs = ctx.programs || {};
  let owner = null; let coOwners = []; let docId = "";
  const changed = () => { q("[data-restricted]").classList.toggle("hidden", vis() === "public"); if (onChange) onChange(get()); };
  const depts = checkList(q("[data-depts]"), Object.fromEntries((ctx.departments || []).map((d) => [d, d.toUpperCase()])), changed);
  const roles = checkList(q("[data-roles]"), { student: "Students", faculty: "Faculty", staff: "Staff", admin: "Admins" }, changed);
  const progs = checkList(q("[data-progs]"), programs, changed);
  const combined = chipInput(q("[data-combined]"), "e.g. csa-phd", changed);
  const groups = chipInput(q("[data-groups]"), "type a group and press Enter", changed);
  const allow = chipInput(q("[data-allow]"), "user ids", changed);
  const deny = chipInput(q("[data-deny]"), "user ids", changed);
  const owners = ownerOptions
    ? checkList(q("[data-owners]"), Object.fromEntries(ownerOptions.map((d) => [d, d.toUpperCase()])), changed)
    : null;
  q("[data-cdept]").innerHTML = (ctx.departments || []).map((d) => `<option value="${esc(d)}">${esc(d.toUpperCase())}</option>`).join("");
  q("[data-cwhat]").innerHTML =
    Object.entries(ROLE_STEMS).map(([r, s]) => `<option value="${esc(s)}">${esc(r)}s</option>`).join("") +
    Object.entries(programs).map(([k, label]) => `<option value="${esc(k)}">${esc(label)}</option>`).join("");
  q("[data-cadd]").onclick = () => { const d = q("[data-cdept]").value; if (d) combined.add(`${d}-${q("[data-cwhat]").value}`); };
  root.querySelectorAll(`input[name="${name}"]`).forEach((i) => { i.onchange = changed; });
  ["[data-from]", "[data-until]"].forEach((s) => { q(s).onchange = changed; });
  if (simple) root.querySelectorAll("[data-full]").forEach((el) => el.classList.add("hidden"));
  const vis = () => root.querySelector(`input[name="${name}"]:checked`).value;
  const isCombined = (g) => (ctx.departments || []).some((d) => g.startsWith(`${d}-`));

  function get() {
    const p = {
      visibility: vis(),
      groups: [...new Set([...progs.get(), ...combined.get(), ...groups.get()])],
      departments: depts.get(),
      roles: roles.get(),
    };
    if (!simple) {
      p.allow_users = allow.get(); p.deny_users = deny.get();
      p.valid_from = fromLocal(q("[data-from]").value); p.valid_until = fromLocal(q("[data-until]").value);
    }
    if (owners) {
      // The main owner stays the main owner while it is still ticked.
      const picked = owners.get();
      const main = picked.includes(owner) ? owner : picked[0];
      if (main) { p.owner = main; p.co_owners = picked.filter((d) => d !== main); }
    } else if (owner) { p.owner = owner; p.co_owners = coOwners; }
    return p;
  }
  function set(policy) {
    const p = policy || {};
    owner = p.owner || null; coOwners = p.co_owners || []; docId = p.doc_id || "";
    if (owners) owners.set(owner ? [owner, ...coOwners] : []);
    root.querySelector(`input[name="${name}"][value="${p.visibility === "public" ? "public" : "restricted"}"]`).checked = true;
    const gs = p.groups || [];
    progs.set(gs.filter((g) => g in programs));
    combined.set(gs.filter((g) => !(g in programs) && isCombined(g)));
    groups.set(gs.filter((g) => !(g in programs) && !isCombined(g)));
    depts.set(p.departments || []); roles.set(p.roles || []);
    allow.set(p.allow_users || []); deny.set(p.deny_users || []);
    q("[data-from]").value = toLocal(p.valid_from); q("[data-until]").value = toLocal(p.valid_until);
    changed();
  }
  function showPreview(result) {
    const el = q("[data-preview]");
    if (!result) { el.textContent = "Who can read it: –"; return; }
    const parts = Object.entries(result.by_department).map(([d, n]) => `${esc(d.toUpperCase())} ${n}`).join(", ");
    el.innerHTML = `${icon("users")}<span><strong>${result.matches}</strong> of ${result.of} people in the directory can read it now${parts ? ` <span class="muted">(${parts})</span>` : ""}</span>`;
  }
  return { get, set, showPreview, docId: () => docId };
}

// Debounced live count from /api/admin/policies/preview.
function previewer(editor) {
  let timer = null;
  return (policy) => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const res = await api("POST", "/api/admin/policies/preview", { doc_id: "preview", ...policy, owner: policy.owner || "institute" });
      editor().showPreview(res.ok ? res.data : null);
    }, 300);
  };
}
