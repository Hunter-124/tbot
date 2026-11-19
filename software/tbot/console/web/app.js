/* TBOT Console — single-page app. No frameworks, no CDNs: works with no internet. */
"use strict";

// ============================================================================================ basics
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const isNum = (v) => typeof v === "number" && isFinite(v);
const fx = (v, d = 1, unit = "") => (isNum(v) ? v.toFixed(d) + unit : "—");

async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body !== undefined ? "POST" : "GET"), cache: "no-store", headers: {} };
  if (opts.body !== undefined) { init.body = JSON.stringify(opts.body); init.headers["Content-Type"] = "application/json"; }
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), opts.timeout || 30000);
  init.signal = ctl.signal;
  let r;
  try { r = await fetch("/api/" + path, init); }
  catch (e) { throw new Error(e.name === "AbortError" ? "the gun did not answer in time" : "no connection to the gun"); }
  finally { clearTimeout(timer); }
  let data = null;
  const text = await r.text();
  try { data = text ? JSON.parse(text) : null; } catch (e) { data = { error: text }; }
  if (!r.ok) throw new Error((data && data.error) || `HTTP ${r.status}`);
  return data;
}

const fmt = {
  bytes(n) {
    if (!isNum(n)) return "—";
    const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(n < 10 && i ? 1 : 0) + " " + u[i];
  },
  dur(s) {
    if (!isNum(s)) return "—";
    s = Math.floor(s);
    const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
    if (d) return `${d}d ${h}h`; if (h) return `${h}h ${m}m`; if (m) return `${m}m ${s % 60}s`; return `${s}s`;
  },
  clock(epoch) { return new Date(epoch * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }); },
  date(epoch) { return new Date(epoch * 1000).toLocaleDateString([], { weekday: "short", month: "short", day: "numeric", year: "numeric" }); },
  datetime(epoch) { return new Date(epoch * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); },
  ago(epoch) {
    const s = Date.now() / 1000 - epoch;
    if (s < 60) return "just now"; if (s < 3600) return Math.floor(s / 60) + " min ago";
    if (s < 86400) return Math.floor(s / 3600) + " h ago"; return Math.floor(s / 86400) + " days ago";
  },
  vt(t) { if (!isNum(t)) return "0:00.00"; const m = Math.floor(t / 60); return `${m}:${(t - m * 60).toFixed(2).padStart(5, "0")}`; },
};

// ============================================================================================ icons
const P = {
  overview: '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>',
  live: '<path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/>',
  sessions: '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="M10 9l5 3-5 3V9z"/>',
  calib: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="M12 1v4M12 19v4M1 12h4M19 12h4"/>',
  camera: '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
  laser: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/>',
  trigger: '<path d="M4 11h9a5 5 0 0 1 5 5v5"/><path d="M4 11V6a2 2 0 0 1 2-2h12"/><path d="M10 11c0 3 1 5 3 6"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
  diag: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
  system: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
  play: '<polygon points="6 3 20 12 6 21 6 3"/>',
  pause: '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>',
  stop: '<rect x="5" y="5" width="14" height="14" rx="2"/>',
  prev: '<polyline points="15 18 9 12 15 6"/>',
  next: '<polyline points="9 18 15 12 9 6"/>',
  skipb: '<polygon points="19 20 9 12 19 4 19 20"/><line x1="5" y1="19" x2="5" y2="5"/>',
  skipf: '<polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19"/>',
  star: '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
  trash: '<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
  target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
  check: '<polyline points="20 6 9 17 4 12"/>',
  x: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
  alert: '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
  info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
  power: '<path d="M18.36 6.64a9 9 0 1 1-12.73 0"/><line x1="12" y1="2" x2="12" y2="12"/>',
  refresh: '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
  clock: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
  thermo: '<path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/>',
  disk: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
  cpu: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>',
  plug: '<path d="M9 2v6M15 2v6M6 8h12v4a6 6 0 0 1-12 0V8zM12 18v4"/>',
  film: '<rect x="2" y="2" width="20" height="20" rx="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/>',
  sun: '<circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>',
  moon: '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
  search: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
  undo: '<path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-15-6.7L3 13"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>',
  gauge: '<path d="M12 14l4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
  imu: '<path d="M12 2l3 7h7l-5.5 4.5L18 21l-6-4-6 4 1.5-7.5L2 9h7z"/>',
  timer: '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l2 2M9 2h6"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>',
  chevron: '<polyline points="6 9 12 15 18 9"/>',
};
const icon = (name, extra = "") => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ${extra}>${P[name] || ""}</svg>`;

// ============================================================================================ toast / modal
function toast(msg, kind = "info", opts = {}) {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  const ic = { ok: "check", err: "alert", warn: "alert", info: "info" }[kind] || "info";
  el.innerHTML = `${icon(ic)}<div style="flex:1">${opts.html ? msg : esc(msg)}${opts.action ? `<div style="margin-top:8px"><button class="btn sm">${esc(opts.action.label)}</button></div>` : ""}</div><span class="x">${icon("x", 'width="14" height="14"')}</span>`;
  $(".x", el).onclick = () => el.remove();
  if (opts.action) $(".btn", el).onclick = () => { el.remove(); opts.action.run(); };
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), opts.ms || (kind === "err" ? 9000 : 5000));
}

function modal({ title, body = "", actions = [], wide = false, onOpen }) {
  return new Promise((resolve) => {
    const root = $("#modal-root");
    const bd = document.createElement("div");
    bd.className = "backdrop";
    bd.innerHTML = `<div class="modal ${wide ? "wide" : ""}" role="dialog"><h3>${title}</h3><div class="mbody">${body}</div><div class="actions"></div></div>`;
    const close = (v) => { bd.remove(); document.removeEventListener("keydown", onKey); resolve(v); };
    const onKey = (e) => { if (e.key === "Escape") close(null); };
    document.addEventListener("keydown", onKey);
    bd.addEventListener("mousedown", (e) => { if (e.target === bd) close(null); });
    const acts = $(".actions", bd);
    for (const a of actions) {
      const b = document.createElement("button");
      b.className = "btn " + (a.cls || "");
      b.textContent = a.label;
      b.onclick = async () => {
        if (a.check && !a.check(bd)) return;
        close(a.value !== undefined ? a.value : a.label);
      };
      acts.appendChild(b);
    }
    root.appendChild(bd);
    if (onOpen) onOpen(bd, close);
    const first = $("input", bd); if (first) setTimeout(() => first.focus(), 30);
  });
}

async function confirmBox(title, text, okLabel = "Continue", cls = "primary") {
  const v = await modal({ title, body: `<p>${text}</p>`, actions: [{ label: "Cancel", value: false }, { label: okLabel, cls, value: true }] });
  return v === true;
}

// Going live is one click (the owner's choice, 2026-11-18): no typed confirmation. The red LIVE banner on every
// page, the mode pill and the arming button (the trigger only moves while it is held) stay.
async function confirmLive(what) {
  toast(`${what}: the servo pulls the trigger while the arming button is held.`, "warn", { ms: 6000 });
  return true;
}

// ============================================================================================ state
const S = {
  status: null, connected: false, failures: 0, everConnected: false, cal: null,
  listeners: new Set(),
  onStatus(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },
};

async function pollStatus() {
  try {
    const st = await api("status", { timeout: 6000 });
    S.status = st; S.failures = 0;
    if (!S.connected) { S.connected = true; onConnected(); }
    renderChrome();
    autoSyncClock();
    S.listeners.forEach((fn) => { try { fn(st); } catch (e) { console.error(e); } });
  } catch (e) {
    S.failures++;
    if (S.failures >= 2 && S.connected) { S.connected = false; renderChrome(); }
    if (!S.connected) showConnect();
  }
}

let connectTimer = null;
async function showConnect() {
  const el = $("#connect");
  el.classList.remove("hidden");
  if (el.dataset.ready) return updateProbe();
  el.dataset.ready = "1";
  el.innerHTML = `<div class="connect-box">
      <div class="radar"><span></span><span></span><span></span>${icon("plug")}</div>
      <div><h1>${S.everConnected ? "Connection to the gun lost" : "Connect your TBOT"}</h1></div>
      <p>The console finds the gun by itself as soon as it answers. No internet is needed.</p>
      <div class="checklist"><ul>
        <li><span class="n">1</span><div><b>Plug the Ethernet cable</b> from the gun's Raspberry Pi straight into this computer (no router needed).</div></li>
        <li><span class="n">2</span><div><b>Power the gun on</b> (battery lever closed). It takes about 50 seconds to boot.</div></li>
        <li><span class="n">3</span><div>Windows gives the cable a <span class="mono">169.254.x.x</span> address by itself; that can take up to a minute after plugging in.</div></li>
      </ul></div>
      <div class="btn-row" style="justify-content:center"><div class="spinner"></div><span class="probe" id="probe">Searching…</span></div>
    </div>`;
  updateProbe();
}
async function updateProbe() {
  const p = $("#probe");
  if (!p) return;
  try {
    const r = await fetch("/local/info", { cache: "no-store" });
    if (r.ok) {
      const info = await r.json();
      p.textContent = info.message || `Trying ${info.candidates.join(", ")}`;
      return;
    }
  } catch (e) { /* served by the gun itself: no launcher */ }
  p.textContent = `Trying ${location.host}… (attempt ${S.failures})`;
}
function onConnected() {
  S.everConnected = true;
  const el = $("#connect");
  el.classList.add("hidden");
  delete el.dataset.ready;
  router();
}

let lastClockTry = 0;
async function autoSyncClock() {
  // the Pi has no clock battery and no internet at the range: keep it on this computer's time
  let on = true;
  try { on = localStorage.getItem("tbot.autoclock") !== "0"; } catch (e) {}
  const st = S.status;
  if (!on || !st || st.dev || Date.now() - lastClockTry < 60000) return;
  const off = st.time - Date.now() / 1000;
  if (Math.abs(off) <= 5) return;
  lastClockTry = Date.now();
  try {
    await api("system", { body: { action: "set_time", epoch_ms: Date.now() } });
    toast(`Gun clock set from this computer (it was ${fmt.dur(Math.abs(off))} ${off < 0 ? "behind" : "ahead"}).`, "ok");
  } catch (e) { /* retried on a later poll */ }
}

// ============================================================================================ chrome
const NAV = [
  { id: "overview", label: "Overview", icon: "overview" },
  { id: "live", label: "Live view", icon: "live" },
  { id: "sessions", label: "Sessions", icon: "sessions" },
  { section: "Setup" },
  { id: "calibration", label: "Calibration", icon: "calib" },
  { id: "camera", label: "Camera & focus", icon: "camera", sub: true },
  { id: "boresight", label: "Boresight", icon: "laser", sub: true },
  { id: "trigger", label: "Trigger servo", icon: "trigger", sub: true },
  { id: "settings", label: "Settings", icon: "settings" },
  { section: "Device" },
  { id: "diagnostics", label: "Diagnostics", icon: "diag" },
  { id: "system", label: "System", icon: "system" },
];

function renderNav(active) {
  $("#nav").innerHTML = NAV.map((n) => n.section
    ? `<div class="nav-label">${esc(n.section)}</div>`
    : `<a href="#/${n.id}" class="${n.id === active ? "active" : ""}" ${n.sub ? 'style="padding-left:22px"' : ""}>${icon(n.icon)}<span>${esc(n.label)}</span></a>`).join("");
}

function modeBadge(p) {
  if (!p) return `<span class="mode-pill off"><span class="dot"></span>OFFLINE</span>`;
  if (!p.active) return `<span class="mode-pill off"><span class="dot"></span>STOPPED</span>`;
  const label = { dry: "DRY RUN", test: "TEST", live: "LIVE" }[p.mode] || p.mode.toUpperCase();
  return `<span class="mode-pill ${esc(p.mode)}"><span class="dot ${p.mode === "live" ? "pulse" : ""}"></span>${label}</span>`;
}

function tempBadge(t) {
  if (!isNum(t)) return "";
  const cls = t >= 80 ? "danger" : t >= 72 ? "warn" : "ok";
  return `<span class="badge ${cls}" title="CPU temperature">${icon("thermo", 'width="12" height="12"')}${t.toFixed(0)} °C</span>`;
}

function throttleIssue(h) {
  const t = h && h.throttled;
  if (!t) return null;
  if (t.under_voltage_now) return { cls: "danger", text: "Under-voltage now" };
  if (t.throttled_now || t.arm_frequency_capped_now) return { cls: "warn", text: "CPU throttled" };
  if (t.under_voltage_occurred) return { cls: "warn", text: "Under-voltage since boot" };
  return null;
}

function renderChrome() {
  const st = S.status;
  const p = st && S.connected ? st.pipeline : null;
  $("#top-status").innerHTML = S.connected && st
    ? `${st.helper && st.helper.alive ? `<span class="badge accent">${icon(st.helper.kind === "servo" ? "trigger" : "camera", 'width="12" height="12"')}${st.helper.kind === "servo" ? "Servo tool" : "Camera tool"}</span>` : ""}
       ${(st.jobs || []).length ? `<span class="badge info"><span class="spinner" style="width:10px;height:10px"></span>${esc(st.jobs[0].title)}</span>` : ""}
       ${tempBadge(st.health && st.health.cpu_temp_c)}${modeBadge(p)}`
    : modeBadge(null);
  const issue = st && throttleIssue(st.health);
  $("#device-card").innerHTML = S.connected && st
    ? `<div class="btn-row" style="gap:8px"><span class="dot" style="color:var(--ok)"></span><b>${esc(st.hostname)}</b>${st.dev ? '<span class="badge warn">DEV</span>' : ""}</div>
       <div class="row"><span class="muted">Pipeline</span><span>${p && p.active ? esc(p.mode) : "stopped"}</span></div>
       <div class="row"><span class="muted">Power-up</span><span>${p ? (p.boot_enabled ? esc(p.boot_mode) : "off") : "—"}</span></div>
       <div class="row"><span class="muted">Disk free</span><span>${fmt.bytes(st.health.disk_free)}</span></div>
       ${issue ? `<span class="badge ${issue.cls}">${esc(issue.text)}</span>` : ""}`
    : `<div class="btn-row" style="gap:8px"><span class="dot" style="color:var(--danger)"></span><b>Not connected</b></div><div class="small muted">Waiting for the gun…</div>`;
  const lb = $("#live-banner");
  if (p && p.active && p.mode === "live") {
    if (!lb.dataset.on) {
      lb.dataset.on = "1";
      lb.innerHTML = `<div class="live-banner">${icon("alert", 'width="14" height="14"')} LIVE — THE SERVO PULLS THE TRIGGER WHILE THE ARMING BUTTON IS HELD <button class="btn" id="go-dry">Switch to dry run</button></div>`;
      $("#go-dry").onclick = () => pipelineAction("start", "dry");
    }
  } else if (lb.dataset.on) { delete lb.dataset.on; lb.innerHTML = ""; }
}

function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("tbot.theme", t); } catch (e) {}
  $("#theme-btn").innerHTML = icon(t === "light" ? "moon" : "sun");
}
$("#theme-btn").onclick = () => setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");

async function pipelineAction(action, mode) {
  try {
    if (action === "start" && mode === "live" && !(await confirmLive("Start LIVE"))) return;
    toast(action === "stop" ? "Stopping the pipeline…" : `Starting ${mode === "dry" ? "dry run" : mode}…`, "info", { ms: 3000 });
    await api("pipeline", { body: { action, mode, confirm: mode === "live" ? "LIVE" : undefined }, timeout: 90000 });
    toast(action === "stop" ? "Pipeline stopped. The camera and servo are free." : `Pipeline running in ${mode === "dry" ? "dry run" : mode}.`, "ok");
  } catch (e) { toast(e.message, "err"); }
  pollStatus();
}

// ============================================================================================ router
const PAGES = {};
let cleanup = null;
let currentPage = null;

function router() {
  const hash = location.hash.replace(/^#\/?/, "") || "overview";
  const [name, ...rest] = hash.split("/");
  const page = PAGES[name] || PAGES.overview;
  const id = PAGES[name] ? name : "overview";
  if (cleanup) { try { cleanup(); } catch (e) { console.error(e); } cleanup = null; }
  const prev = currentPage;
  currentPage = id;
  if (prev && TOOL_PAGES.includes(prev) && !TOOL_PAGES.includes(id)) releaseTool();
  renderNav(id);
  $("#title").textContent = page.title;
  const el = $("#page");
  el.innerHTML = "";
  el.className = "page";
  el.style.animation = "none"; void el.offsetWidth; el.style.animation = "";
  if (!S.connected) return;
  const r = page.render(el, rest.map(decodeURIComponent));
  cleanup = typeof r === "function" ? r : null;
}
window.addEventListener("hashchange", router);

// ============================================================================================ tool helper (camera / servo)
const TOOL_PAGES = ["camera", "boresight", "trigger"];
async function releaseTool() {
  const st = S.status;
  if (st && st.helper && st.helper.alive) {
    try { await api("tool/stop", { body: {} }); } catch (e) {}
    toast("Calibration tool closed. The pipeline is still stopped.", "info", {
      action: { label: "Start dry run", run: () => pipelineAction("start", "dry") }, ms: 9000,
    });
    pollStatus();
  }
}

/** Start (or reuse) a helper. Asks before stopping a running pipeline. */
async function ensureTool(kind, opts = {}, force = false) {
  const cur = await api("tool/state");
  const same = cur && cur.alive && cur.kind === kind && !force;
  if (same) return cur;
  const p = S.status && S.status.pipeline;
  let stop = false;
  if (p && p.active) {
    const ok = await confirmBox(`Stop the pipeline?`,
      `The ${p.mode === "live" ? "<b>LIVE</b> " : ""}pipeline is using the camera and the trigger servo. It has to stop while you ${kind === "servo" ? "set up the trigger" : "use the camera tool"}. Leave this page when you are done and start it again from Overview or System.`,
      "Stop pipeline & continue");
    if (!ok) return null;
    stop = true;
  }
  return api("tool/start", { body: { kind, stop_pipeline: stop, ...opts }, timeout: 60000 });
}

/** Long-poll helper events while `alive()` is true. */
function toolEvents(onEvent, alive) {
  let since = 0, stopped = false;
  (async () => {
    while (!stopped && alive()) {
      try {
        const r = await api(`tool/events?since=${since}&wait=4`, { timeout: 10000 });
        since = r.seq;
        for (const ev of r.events) onEvent(ev);
        if (!r.alive) { onEvent({ type: "exit" }); break; }
      } catch (e) {
        if (stopped) break;
        await sleep(1000);
      }
    }
  })();
  return () => { stopped = true; };
}

const sendTool = (obj) => api("tool/command", { body: obj }).catch((e) => toast(e.message, "err"));

// ============================================================================================ shared bits
function streamViewer(id, { hud = "", overlay = "" } = {}) {
  return `<div class="viewer" id="${id}">
    <img alt="" class="hidden">
    <div class="placeholder">${icon("camera")}<div>Connecting to the camera…</div></div>
    <svg class="overlay" preserveAspectRatio="xMidYMid meet">${overlay}</svg>
    <div class="hud">${hud}</div></div>`;
}
function attachStream(viewer) {
  const img = $("img", viewer), ph = $(".placeholder", viewer);
  let stopped = false, retry = null;
  const start = () => {
    if (stopped) return;
    img.src = `/api/stream.mjpg?t=${Date.now()}`;
  };
  // multipart MJPEG does not reliably fire `load`: watch for the first decoded frame instead
  const shown = setInterval(() => {
    if (img.naturalWidth > 0 && img.classList.contains("hidden")) { img.classList.remove("hidden"); ph.classList.add("hidden"); }
  }, 250);
  img.onerror = () => {
    img.classList.add("hidden"); ph.classList.remove("hidden");
    $("div", ph).textContent = "Waiting for the camera…";
    clearTimeout(retry); retry = setTimeout(start, 2000);
  };
  start();
  // a stream that ended (pipeline restarted) keeps its last frame: reconnect when the status says it restarted
  const off = S.onStatus((st) => {
    const key = `${st.pipeline.active}|${st.pipeline.mode}|${st.helper ? st.helper.kind + st.helper.alive : ""}`;
    if (viewer.dataset.key && viewer.dataset.key !== key) start();
    viewer.dataset.key = key;
  });
  return () => { stopped = true; off(); clearInterval(shown); clearTimeout(retry); img.onerror = null; img.removeAttribute("src"); img.src = ""; };
}
function svgPoint(svg, evt) {
  const pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}
function calAge(entry) {
  if (!entry) return `<span class="badge danger">Not calibrated</span>`;
  const days = (Date.now() / 1000 - entry.mtime) / 86400;
  return `<span class="badge ${days > 60 ? "warn" : "ok"}">${icon("check", 'width="11" height="11"')}${fmt.datetime(entry.mtime)}</span>`;
}
/** command -> sear break (ms) from calib/latency.json: actuator.servo.cmd_to_break_ms.{mean} */
function triggerDelay(lat) {
  const v = lat && lat.actuator && lat.actuator.servo && lat.actuator.servo.cmd_to_break_ms;
  if (isNum(v)) return v;
  if (v && isNum(v.mean)) return v.mean;
  return lat ? (lat.servo_cmd_to_break_ms ?? null) : null;
}
async function loadCal() { S.cal = await api("calibration"); return S.cal; }
function lensToMuzzle(cal) {
  const m = cal && cal.mount;
  return m && isNum(m.muzzle_x_mm) && isNum(m.lens_tip_x_mm) ? (m.muzzle_x_mm - m.lens_tip_x_mm) / 1000 : 0.2728;
}
function colorizeTerm(lines) {
  return lines.map((l) => {
    let s = esc(l);
    s = s.replace(/\bPASS\b/g, '<span class="pass">PASS</span>').replace(/\bFAIL(ED)?\b/g, '<span class="fail">FAIL$1</span>')
      .replace(/\bWARN(ING)?\b/g, '<span class="warnl">WARN$1</span>');
    return s;
  }).join("\n");
}
async function watchJob(job, onUpdate) {
  let j = job;
  while (j.running) {
    await sleep(700);
    try { j = await api(`jobs/${j.id}`); } catch (e) { await sleep(1500); continue; }
    if (onUpdate) onUpdate(j);
  }
  if (onUpdate) onUpdate(j);
  return j;
}
function stopFor(st) {
  // what the pipeline is doing, as one short sentence for empty states
  const p = st.pipeline;
  return p.active ? `Running in ${p.mode}` : "Stopped";
}

// ============================================================================================ OVERVIEW
PAGES.overview = {
  title: "Overview",
  render(el) {
    el.innerHTML = `
      <div class="grid" style="grid-template-columns:minmax(0,1.6fr) minmax(0,1fr)">
        <div class="card flush" id="ov-hero"></div>
        <div class="card" id="ov-pipe"></div>
      </div>
      <div class="grid g4" id="ov-kpis"></div>
      <div class="grid g2">
        <div class="card" id="ov-cal"><div class="card-h"><h2>Calibration</h2><div class="right"><a class="btn sm" href="#/calibration">Open</a></div></div><div class="muted">Loading…</div></div>
        <div class="card" id="ov-sess"><div class="card-h"><h2>Recent presses</h2><div class="right"><a class="btn sm" href="#/sessions">All sessions</a></div></div><div class="muted">Loading…</div></div>
      </div>`;
    let streamStop = null, streamOn = null;
    const hero = $("#ov-hero");
    const drawHero = (st) => {
      const on = st.pipeline.active;
      if (on === streamOn) return;
      streamOn = on;
      if (streamStop) { streamStop(); streamStop = null; }
      if (on) {
        hero.innerHTML = streamViewer("ov-view", { hud: `<span class="badge">PIPELINE VIEW</span>` });
        $(".viewer", hero).style.cssText = "border:0;border-radius:0;aspect-ratio:16/9";
        streamStop = attachStream($("#ov-view"));
      } else {
        hero.innerHTML = `<div class="empty" style="aspect-ratio:16/9">${icon("live")}<h3>The pipeline is stopped</h3>
          <p>Start a dry run to see the camera with the detector overlay. Dry run never moves the trigger.</p>
          <div class="btn-row" style="justify-content:center;margin-top:6px"><button class="btn primary" data-start="dry">${icon("play")}Start dry run</button></div></div>`;
        $("[data-start]", hero).onclick = () => pipelineAction("start", "dry");
      }
    };
    const drawPipe = (st) => {
      const p = st.pipeline, t = st.pipeline_telemetry;
      $("#ov-pipe").innerHTML = `
        <div class="card-h"><h2>Pipeline</h2><div class="right">${modeBadge(p)}</div></div>
        ${t ? `<div class="armed-tile ${t.armed ? "on" : ""}" style="margin-bottom:14px">${t.armed ? "ARMED" : "SAFE"}<small>${t.armed ? "arming button held" : "arming button released"}</small></div>
          <dl class="kv">
            <dt>Frame rate</dt><dd>${fx(t.fps, 1, " fps")}</dd>
            <dt>Exposure</dt><dd>${fx(t.exposure_us, 0, " µs")} × ${fx(t.gain, 1)}</dd>
            <dt>Targets</dt><dd>${t.tracks} tracked · ${t.n_detections} seen</dd>
            <dt>Decision</dt><dd>${esc(t.decision.reason || "—")}</dd>
            <dt>Shots this run</dt><dd>${t.shots ?? 0}</dd>
            <dt>IMU</dt><dd>${t.imu_ready ? '<span class="badge ok">ready</span>' : '<span class="badge warn">settling</span>'}</dd>
          </dl>` : p.active ? `<div class="empty" style="padding:24px"><div class="spinner"></div><p>Starting up (about 15 s)…</p></div>`
          : `<p class="muted" style="margin:0 0 14px">Stopped. The camera and the trigger servo are free for calibration.</p>`}
        <hr class="sep" style="margin:14px 0">
        <div class="btn-row">
          ${p.active ? `<button class="btn" data-a="stop">${icon("stop")}Stop</button>` : ""}
          ${!p.active || p.mode !== "dry" ? `<button class="btn ${p.active ? "" : "primary"}" data-a="dry">${icon("play")}Dry run</button>` : ""}
          ${!p.active || p.mode !== "test" ? `<button class="btn" data-a="test">Test mode</button>` : ""}
          ${!p.active || p.mode !== "live" ? `<button class="btn outline-danger" data-a="live">Go live…</button>` : ""}
        </div>`;
      $$("[data-a]", $("#ov-pipe")).forEach((b) => b.onclick = () => b.dataset.a === "stop" ? pipelineAction("stop") : pipelineAction("start", b.dataset.a));
    };
    const drawKpis = (st) => {
      const h = st.health;
      const used = h.disk_total ? 1 - h.disk_free / h.disk_total : null;
      const mem = h.mem_total ? 1 - h.mem_available / h.mem_total : null;
      const issue = throttleIssue(h);
      const t = h.cpu_temp_c;
      $("#ov-kpis").innerHTML = `
        <div class="card stat"><span class="label">${icon("thermo", 'width="14" height="14"')}CPU temperature</span><span class="value">${fx(t, 0)}<small>°C</small></span>
          <div class="meter ${t >= 80 ? "danger" : t >= 72 ? "warn" : "ok"}"><span style="width:${isNum(t) ? clamp((t - 30) / 55 * 100, 0, 100) : 0}%"></span></div>
          <span class="hint">${t >= 80 ? "Throttling: let it cool" : t >= 72 ? "Warm" : "Normal"}</span></div>
        <div class="card stat"><span class="label">${icon("disk", 'width="14" height="14"')}Storage free</span><span class="value">${fmt.bytes(h.disk_free)}</span>
          <div class="meter ${used > .9 ? "danger" : used > .75 ? "warn" : ""}"><span style="width:${(used || 0) * 100}%"></span></div>
          <span class="hint">${st.sessions.count} recorded press${st.sessions.count === 1 ? "" : "es"}</span></div>
        <div class="card stat"><span class="label">${icon("cpu", 'width="14" height="14"')}Memory in use</span><span class="value">${mem === null ? "—" : (mem * 100).toFixed(0)}<small>%</small></span>
          <div class="meter ${mem > .9 ? "danger" : mem > .75 ? "warn" : ""}"><span style="width:${(mem || 0) * 100}%"></span></div>
          <span class="hint">${h.load ? "load " + h.load.map((x) => x.toFixed(1)).join(" / ") : "&nbsp;"}</span></div>
        <div class="card stat"><span class="label">${icon("power", 'width="14" height="14"')}Power</span><span class="value" style="font-size:18px;padding:4px 0 6px">${issue ? `<span class="badge ${issue.cls}">${esc(issue.text)}</span>` : h.throttled ? '<span class="badge ok">Supply OK</span>' : '<span class="badge">Not measured</span>'}</span>
          <span class="hint">Up ${fmt.dur(h.uptime_s)}${isNum(h.arm_freq_mhz) ? ` · ${h.arm_freq_mhz.toFixed(0)} MHz` : ""}</span></div>`;
    };
    const tick = (st) => { drawHero(st); drawPipe(st); drawKpis(st); };
    tick(S.status);
    const off = S.onStatus(tick);
    loadCal().then((cal) => {
      const b = cal.boresight && cal.boresight.data, i = cal.intrinsics && cal.intrinsics.data, lat = cal.latency && cal.latency.data;
      const fxv = i && i.K ? i.K[0][0] : null;
      const delay = triggerDelay(lat);
      $("#ov-cal").innerHTML = `<div class="card-h"><h2>Calibration</h2><div class="right"><a class="btn sm" href="#/calibration">Open</a></div></div>
        <table class="table">
          <tr><td>Lens (focal length)</td><td class="num">${fxv ? fxv.toFixed(0) + " px" : "—"}${i ? ` @ ${i.width}×${i.height}` : ""}</td><td style="text-align:right">${calAge(cal.intrinsics)}</td></tr>
          <tr><td>Boresight</td><td class="num">${b && b.rpy_deg ? `pitch ${b.rpy_deg[1].toFixed(2)}° · yaw ${b.rpy_deg[2].toFixed(2)}°` : "—"}</td><td style="text-align:right">${calAge(cal.boresight)}</td></tr>
          <tr><td>Trigger servo</td><td class="num">${cal.servo.rest_us ? `rest ${cal.servo.rest_us} · fire ${cal.servo.fire_us} µs` : "—"}</td><td style="text-align:right">${cal.servo.max_us ? '<span class="badge ok">set</span>' : '<span class="badge danger">not set</span>'}</td></tr>
          <tr><td>Trigger delay</td><td class="num">${isNum(delay) ? delay.toFixed(1) + " ms" : "—"}</td><td style="text-align:right">${calAge(cal.latency)}</td></tr>
          <tr><td>IMU</td><td class="num">${cal.imu ? "bias & axes" : "—"}</td><td style="text-align:right">${calAge(cal.imu)}</td></tr>
        </table>`;
    }).catch(() => {});
    api("sessions").then((r) => {
      const list = r.sessions.slice(0, 6);
      $("#ov-sess").innerHTML = `<div class="card-h"><h2>Recent presses</h2><div class="right"><a class="btn sm" href="#/sessions">All sessions</a></div></div>` +
        (list.length ? `<table class="table">${list.map((s) => `<tr style="cursor:pointer" onclick="location.hash='#/sessions/${encodeURIComponent(s.name)}'">
          <td>${fmt.datetime(s.time)}</td><td class="num muted">${fx(s.span_s, 1, " s")}</td>
          <td>${sessionTags(s)}</td><td style="text-align:right">${s.clip ? icon("play", 'width="14" height="14"') : ""}</td></tr>`).join("")}</table>`
          : `<div class="empty" style="padding:24px">${icon("film")}<p>No presses recorded yet. Every held press of the arming button is saved here.</p></div>`);
    }).catch(() => {});
    return () => { off(); if (streamStop) streamStop(); };
  },
};

function sessionTags(s) {
  const t = [];
  if (s.outcome === "hit") t.push(`<span class="badge ok">HIT</span>`);
  else if (s.outcome === "miss") t.push(`<span class="badge danger">MISS</span>`);
  if (s.shots) t.push(`<span class="badge ${s.live ? "accent" : "info"}">${s.shots} shot${s.shots > 1 ? "s" : ""}${s.live ? "" : " · dry"}</span>`);
  if (s.predicted_hits && s.outcome !== "hit") t.push(`<span class="badge ok" title="tbot score says it would have hit">scored hit</span>`);
  if (s.starred) t.push(`<span style="color:var(--warn)">${icon("star", 'width="13" height="13" fill="currentColor"')}</span>`);
  return t.join(" ");
}

// ============================================================================================ LIVE VIEW
PAGES.live = {
  title: "Live view",
  render(el) {
    el.innerHTML = `<div class="split"><div id="lv-main"></div><div class="grid" id="lv-side" style="gap:16px"></div></div>`;
    let stopStream = null, on = null, timer = null;
    const main = $("#lv-main"), side = $("#lv-side");
    const draw = (st) => {
      const active = st.pipeline.active;
      if (active !== on) {
        on = active;
        if (stopStream) { stopStream(); stopStream = null; }
        if (active) {
          main.innerHTML = `${streamViewer("lv-view", { hud: `<span class="badge" id="lv-mode"></span>` })}
            <div class="btn-row" style="margin-top:12px"><a class="btn" href="/api/snapshot.jpg" download="tbot-snapshot.jpg">${icon("image")}Snapshot</a>
            <span class="muted small">The pipeline draws its own overlay: detections, tracks, the lead point and the decision.</span></div>`;
          stopStream = attachStream($("#lv-view"));
        } else {
          main.innerHTML = `<div class="card"><div class="empty">${icon("live")}<h3>No live picture</h3><p>The pipeline is stopped. Start it to watch what the gun sees.
            ${st.helper && st.helper.alive ? "A calibration tool is holding the camera." : ""}</p>
            <div class="btn-row" style="justify-content:center"><button class="btn primary" id="lv-start">${icon("play")}Start dry run</button></div></div></div>`;
          $("#lv-start").onclick = () => pipelineAction("start", "dry");
        }
      }
      const m = $("#lv-mode"); if (m) { m.textContent = { dry: "DRY RUN", test: "TEST", live: "LIVE" }[st.pipeline.mode]; m.className = "badge " + (st.pipeline.mode === "live" ? "live" : ""); }
    };
    const drawTele = async () => {
      if (!S.status || !S.status.pipeline.active) { side.innerHTML = ""; return; }
      let t;
      try { t = await api("telemetry", { timeout: 3000 }); } catch (e) { return; }
      if (!t || !t.stage_ms) { side.innerHTML = `<div class="card"><div class="empty" style="padding:20px"><div class="spinner"></div><p>Waiting for the first frames…</p></div></div>`; return; }
      const d = t.decision || {};
      const ae = t.ae || {};
      side.innerHTML = `
        <div class="armed-tile ${t.armed ? "on" : ""}">${t.armed ? "ARMED" : "SAFE"}<small>${t.armed ? "arming button held" : "button released — nothing can fire"}</small></div>
        <div class="card"><div class="card-h"><h3>Camera</h3><div class="right"><span class="badge ${ae.state === "ok" ? "ok" : "warn"}">AE ${esc(ae.state || "—")}</span></div></div>
          <div class="stats-row">
            <div class="stat"><span class="label">Frame rate</span><span class="value">${fx(t.fps, 0)}<small>fps</small></span></div>
            <div class="stat"><span class="label">Latency</span><span class="value">${fx(t.age_ms, 0)}<small>ms</small></span></div>
            <div class="stat"><span class="label">Exposure</span><span class="value">${fx(t.exposure_us, 0)}<small>µs</small></span></div>
            <div class="stat"><span class="label">Gain</span><span class="value">${fx(t.gain, 1)}<small>×</small></span></div>
          </div>
          <dl class="kv" style="margin-top:12px"><dt>Sky level</dt><dd>${fx(ae.level_dn, 0, " DN")}</dd><dt>Clipped</dt><dd>${fx(ae.clip_pct, 1, " %")}</dd>
          <dt>Frames lost / restarts</dt><dd>${t.lost ?? 0} / ${t.cam_restarts ?? 0}</dd><dt>Processing</dt><dd>${fx(t.stage_ms.total, 1, " ms")} / frame</dd></dl></div>
        <div class="card"><div class="card-h"><h3>Targets & decision</h3></div>
          <div class="stats-row">
            <div class="stat"><span class="label">Detections</span><span class="value">${t.n_detections}</span></div>
            <div class="stat"><span class="label">Tracks</span><span class="value">${(t.tracks || []).length}</span></div>
            <div class="stat"><span class="label">P(hit)</span><span class="value">${isNum(d.p_hit) && d.p_hit > 0 ? (d.p_hit * 100).toFixed(0) : "—"}<small>%</small></span></div>
          </div>
          <div class="callout ${d.fire ? "danger" : ""}" style="margin-top:12px">${icon(d.fire ? "target" : "info")}<div><b>${esc(d.reason || "—")}</b>${isNum(d.range_m) && d.range_m > 0 ? `<div class="small muted">range ${d.range_m.toFixed(1)} m · lead miss ${fx(d.miss_deg, 2)}° vs pattern ${fx(d.pattern_deg, 2)}°</div>` : ""}</div></div>
          <dl class="kv" style="margin-top:12px"><dt>Shots this run</dt><dd>${t.shots}</dd><dt>IMU</dt><dd>${t.imu_ready ? "ready" : "settling"}${isNum(t.imu_age_ms) ? ` · ${t.imu_age_ms} ms` : ""}</dd>
          <dt>Noise</dt><dd>${fx(t.detector && t.detector.noise_dn, 2, " DN")}</dd></dl></div>`;
    };
    draw(S.status);
    const off = S.onStatus(draw);
    drawTele(); timer = setInterval(drawTele, 500);
    return () => { off(); clearInterval(timer); if (stopStream) stopStream(); };
  },
};

// ============================================================================================ SESSIONS
PAGES.sessions = {
  title: "Sessions",
  render(el, params) {
    let list = [], filter = "all", query = "", current = params[0] || null, detail = null, view = "list";
    let saved = new Set(), archive = null, sig = "";
    try { filter = localStorage.getItem("tbot.sfilter") || "all"; view = localStorage.getItem("tbot.sview") || "list"; } catch (e) {}
    el.innerHTML = `<div class="btn-row" style="justify-content:space-between">
        <div class="chips" id="s-chips"></div>
        <div class="btn-row"><div class="input-group" style="width:240px"><input class="input" id="s-q" placeholder="Search date or note"><span class="addon">${icon("search", 'width="14" height="14"')}</span></div>
          <div class="seg" id="s-view"><button data-v="list" title="List and player">${icon("sessions", 'width="15" height="15"')}</button><button data-v="grid" title="Grid of previews">${icon("overview", 'width="15" height="15"')}</button></div></div>
      </div>
      <div id="s-body"></div>
      <div class="card" style="padding:10px 16px"><div class="btn-row small muted" id="s-foot"></div></div>`;
    const filters = {
      all: { label: "All", f: () => true },
      shots: { label: "Shots", f: (s) => s.shots > 0 },
      hits: { label: "Hits", f: (s) => s.outcome === "hit" || (s.predicted_hits > 0 && s.outcome !== "miss") },
      starred: { label: "Starred", f: (s) => s.starred },
      unreviewed: { label: "To review", f: (s) => s.shots > 0 && !s.outcome },
    };
    const visible = () => list.filter(filters[filter].f).filter((s) => !query || (s.name + " " + (s.note || "") + " " + fmt.datetime(s.time)).toLowerCase().includes(query));
    const thumbUrl = (s) => `/api/sessions/${encodeURIComponent(s.name)}/thumb.jpg?v=${s.thumb_v}`;
    const clipUrl = (s) => `/api/sessions/${encodeURIComponent(s.name)}/clip.mp4`;
    const reasonLabel = { shot: "SHOT", "best P(hit)": "BEST AIM", "target tracked": "TRACKED", detection: "SEEN" };
    const thumb = (s, big) => `<div class="thumb ${big ? "big" : ""}" ${s.clip === "clip.mp4" ? `data-clip="${clipUrl(s)}" data-t="${isNum(s.key_t) ? s.key_t : 0}"` : ""}>
        ${s.thumb ? `<img loading="lazy" src="${thumbUrl(s)}" alt="">` : `<div class="noprev">${icon("film")}<span>${s.renderable ? "preview coming" : "no preview"}</span></div>`}
        ${s.key_reason && reasonLabel[s.key_reason] ? `<span class="kr ${s.key_reason === "shot" ? "shot" : ""}">${reasonLabel[s.key_reason]}</span>` : ""}
        <span class="dur">${fx(s.span_s, 1, " s")}</span>
        ${s.clip ? `<span class="playi">${icon("play")}</span>` : ""}
      </div>`;
    const meta = (s) => `${isNum(s.max_p_hit) ? `<span>best P(hit) ${(s.max_p_hit * 100).toFixed(0)}%</span>` : ""}${s.clip ? "" : "<span>clip pending</span>"}${saved.has(s.name) ? `<span title="Copied to ${esc(archive ? archive.dir : "this PC")}">${icon("download", 'width="11" height="11"')} on this PC</span>` : ""}${s.note ? `<span>· ${esc(s.note.slice(0, 40))}</span>` : ""}`;
    const timeOf = (s) => new Date(s.time * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    let hoverVid = null;
    const wireHover = (root) => {
      $$(".thumb[data-clip]", root).forEach((t) => {
        t.addEventListener("mouseenter", () => {
          const v = document.createElement("video");
          const start = Math.max(0, +t.dataset.t - 1.0);
          v.muted = true; v.playsInline = true; v.preload = "auto"; v.src = t.dataset.clip + "#t=" + start.toFixed(2);
          v.addEventListener("timeupdate", () => { if (v.currentTime > start + 2.2) v.currentTime = start; });
          v.addEventListener("loadeddata", () => { v.classList.add("on"); v.play().catch(() => {}); });
          t.appendChild(v); hoverVid = v;
        });
        t.addEventListener("mouseleave", () => { $$("video", t).forEach((v) => { v.pause(); v.removeAttribute("src"); v.load(); v.remove(); }); hoverVid = null; });
      });
    };
    const drawList = () => {
      $("#s-chips").innerHTML = Object.entries(filters).map(([k, v]) => `<button class="chip ${k === filter ? "on" : ""}" data-f="${k}">${v.label} <span class="n">${list.filter(v.f).length}</span></button>`).join("");
      $$("[data-f]", $("#s-chips")).forEach((b) => b.onclick = () => { filter = b.dataset.f; try { localStorage.setItem("tbot.sfilter", filter); } catch (e) {} drawList(); });
      $$("#s-view button").forEach((b) => b.classList.toggle("on", b.dataset.v === view));
      const v = visible();
      const empty = `<div class="empty">${icon("film")}<h3>${list.length ? "Nothing matches" : "No sessions yet"}</h3><p>${list.length ? "Try another filter." : "Every held press of the arming button is recorded and shows up here."}</p></div>`;
      let day = null, html = "";
      if (view === "grid") {
        for (const s of v) {
          const d = fmt.date(s.time);
          if (d !== day) { if (day) html += `</div>`; day = d; html += `<div class="s-day g">${esc(d)}</div><div class="g-grid">`; }
          html += `<div class="g-card" data-n="${esc(s.name)}">${thumb(s, true)}<div class="g-info"><div class="btn-row" style="justify-content:space-between"><b class="num">${timeOf(s)}</b><span class="tags">${sessionTags(s)}</span></div><div class="m">${meta(s)}</div></div></div>`;
        }
        if (day) html += `</div>`;
        $("#s-body").innerHTML = html ? `<div class="g-wrap">${html}</div>` : `<div class="card">${empty}</div>`;
        $$(".g-card", $("#s-body")).forEach((c) => c.onclick = () => { view = "list"; try { localStorage.setItem("tbot.sview", view); } catch (e) {} current = c.dataset.n; layout(); });
        wireHover($("#s-body"));
      } else {
        for (const s of v) {
          const d = fmt.date(s.time);
          if (d !== day) { day = d; html += `<div class="s-day">${esc(d)}</div>`; }
          html += `<div class="s-item ${s.name === current ? "active" : ""}" data-n="${esc(s.name)}">${thumb(s, false)}
            <div class="info"><div class="btn-row" style="justify-content:space-between;gap:6px"><span class="t">${timeOf(s)}</span><span class="tags">${sessionTags(s)}</span></div>
            <div class="m">${meta(s)}</div></div></div>`;
        }
        const box = $("#s-list");
        if (box) {
          box.innerHTML = html || empty;
          $$(".s-item", box).forEach((it) => it.onclick = () => select(it.dataset.n));
          wireHover(box);
        }
      }
      drawFoot();
    };
    const layout = () => {
      if (player) { player(); player = null; }
      $("#s-body").innerHTML = view === "grid" ? "" : `<div class="sessions"><div class="card flush"><div class="s-list" id="s-list"></div></div><div id="s-detail"></div></div>`;
      drawList();
      if (view === "list") {
        const v = visible();
        const pick = current && list.some((s) => s.name === current) ? current : (v[0] && v[0].name);
        if (pick) select(pick, true);
      }
    };
    $$("#s-view button").forEach((b) => b.onclick = () => { view = b.dataset.v; try { localStorage.setItem("tbot.sview", view); } catch (e) {} layout(); });
    const drawFoot = () => {
      const a = (S.status && S.status.autoclips) || null;
      const missing = list.filter((s) => !s.clip && s.renderable).length;
      let clipTxt = "";
      if (a && a.enabled) {
        if (a.state === "rendering") clipTxt = `<span class="spinner" style="width:12px;height:12px"></span> Making clips${a.pending_clips > 1 ? ` (${a.pending_clips} to go)` : ""}`;
        else if (a.state === "previews") clipTxt = `<span class="spinner" style="width:12px;height:12px"></span> Making previews`;
        else if (a.state === "paused") clipTxt = `${icon("pause", 'width="12" height="12"')} Clips wait: ${esc(a.reason)} (${a.pending_clips} to make)`;
        else clipTxt = missing ? `${missing} clip${missing > 1 ? "s" : ""} queued` : `${icon("check", 'width="12" height="12"')} Every press has its clip`;
      } else clipTxt = missing ? `${missing} press${missing > 1 ? "es" : ""} without a clip` : "";
      $("#s-foot").innerHTML = `<span>${list.length} presses</span><span>·</span><span class="btn-row" style="gap:6px">${clipTxt}</span>
        <label class="btn-row" style="gap:8px;margin-left:auto" title="Make a clip and a preview of every press on the gun while the arming button is up (stops the moment it is pressed)">Auto clips <label class="switch"><input type="checkbox" id="s-auto" ${a && a.enabled ? "checked" : ""}><span></span></label></label>
        ${!(a && a.enabled) && missing ? `<button class="btn sm ghost" id="s-render">${icon("film")}Render now</button>` : ""}
        ${archive ? `<span>·</span><button class="btn sm ghost" id="s-arch" title="${esc(archive.state)}">${icon("download", 'width="13" height="13"')}${saved.size} saved to ${esc(archive.dir.split(/[\\/]/).slice(-2).join("\\"))}</button>` : ""}`;
      const rb = $("#s-render"); if (rb) rb.onclick = renderMissing;
      const au = $("#s-auto"); if (au) au.onchange = async (e) => { try { await api("autoclips", { body: { enabled: e.target.checked } }); pollStatus(); } catch (er) { toast(er.message, "err"); } };
      const ar = $("#s-arch"); if (ar) ar.onclick = () => fetch("/local/open", { method: "POST" });
    };
    const renderMissing = async () => {
      try {
        const j = await api("jobs", { body: { kind: "clip_missing" } });
        toast("Rendering clips in the background (about 15 s each on the Pi).", "info");
        await watchJob(j); toast("Clips rendered.", "ok"); refresh();
      } catch (e) { toast(e.message, "err"); }
    };
    const loadArchive = async () => {
      try { const r = await fetch("/local/archive", { cache: "no-store" }); if (r.ok) { archive = await r.json(); saved = new Set(archive.saved); } } catch (e) {}
    };
    const refresh = async (quiet) => {
      let fresh;
      try { fresh = (await api("sessions")).sessions; } catch (e) { if (!quiet) toast(e.message, "err"); return; }
      await loadArchive();
      const nsig = JSON.stringify(fresh.map((s) => [s.name, s.thumb_v, s.clip, s.outcome, s.starred, s.note])) + [...saved].join();
      if (quiet && (nsig === sig || hoverVid)) { drawFoot(); return; }
      sig = nsig; list = fresh;
      if (!quiet) return layout();
      drawList();
      if (detail) { const s = list.find((x) => x.name === detail.name); if (s && s.clip && !detail.clip) select(detail.name, true); }
    };
    let player = null;
    const select = async (name, replace) => {
      current = name;
      const h = `#/sessions/${encodeURIComponent(name)}`;
      if (location.hash !== h) history[replace ? "replaceState" : "pushState"](null, "", h);
      $$(".s-item").forEach((it) => it.classList.toggle("active", it.dataset.n === name));
      const active = $(".s-item.active"); if (active) active.scrollIntoView({ block: "nearest" });
      try { detail = await api(`sessions/${encodeURIComponent(name)}`); } catch (e) { toast(e.message, "err"); return; }
      if (player) player();
      player = drawDetail(detail);
    };
    const step = (dir) => {
      const v = visible(); const i = v.findIndex((s) => s.name === current);
      const n = v[clamp(i + dir, 0, v.length - 1)]; if (n && n.name !== current) select(n.name);
    };
    const setOutcome = async (o) => {
      if (!detail) return;
      const val = detail.outcome === o ? null : o;
      try {
        await api(`sessions/${encodeURIComponent(detail.name)}/notes`, { body: { outcome: val } });
        detail.outcome = val; const s = list.find((x) => x.name === detail.name); if (s) s.outcome = val;
        drawList(); drawOutcome();
      } catch (e) { toast(e.message, "err"); }
    };
    const drawOutcome = () => {
      const o = $("#s-outcome"); if (!o || !detail) return;
      $$("button", o).forEach((b) => { b.className = "btn sm" + (detail.outcome === b.dataset.o ? (b.dataset.o === "hit" ? " on-hit" : b.dataset.o === "miss" ? " on-miss" : " on-none") : ""); });
      const st = $("#s-star"); if (st) { st.style.color = detail.starred ? "var(--warn)" : ""; st.innerHTML = icon("star", detail.starred ? 'fill="currentColor"' : ""); }
    };

    const drawDetail = (d) => {
      const box = $("#s-detail");
      const sum = d.summary || {};
      const slow = isNum(d.fps) && d.clip_fps ? d.fps / d.clip_fps : null;
      const fileUrl = (f, dl) => `/api/sessions/${encodeURIComponent(d.name)}/${f}${dl ? "?download=1" : ""}`;
      box.innerHTML = `<div class="grid" style="gap:16px">
        <div class="player">
          <div class="viewer">${d.clip && d.clip.endsWith(".mp4") ? `<video id="s-video" src="${fileUrl(d.clip)}" ${d.thumb ? `poster="${fileUrl("thumb.jpg")}?v=${d.thumb_v}"` : ""} preload="auto" playsinline muted></video>`
            : `<div class="placeholder">${icon("film")}<div>${d.clip ? "This clip is an AVI: download it to watch." : "No clip for this press yet (live-mode presses are rendered later)."}</div>
               ${!d.clip && d.renderable ? `<button class="btn primary" id="s-mk">${icon("film")}Render clip</button>` : ""}</div>`}
</div>
          <div class="controls">
            <div class="timeline" id="s-tl"><div class="rail"></div><div class="fill"></div>${(d.shot_list || []).map((s) => `<div class="mark" data-t="${s.clip_t}" data-label="SHOT ${s.n}"></div>`).join("")}<div class="knob"></div></div>
            <div class="ctl-row">
              <button class="btn icon" id="s-play" title="Play / pause (Space)">${icon("play")}</button>
              <button class="btn icon sm" id="s-fb" title="Previous frame (←)">${icon("prev")}</button>
              <button class="btn icon sm" id="s-ff" title="Next frame (→)">${icon("next")}</button>
              <span class="time" id="s-time">0:00.00 / 0:00.00</span>
              <div class="seg" id="s-rate">${[0.25, 0.5, 1, 2].map((r) => `<button data-r="${r}" class="${r === 1 ? "on" : ""}">${r}×</button>`).join("")}</div>
              ${d.shot_list && d.shot_list.length ? `<button class="btn sm primary" id="s-jump" title="Jump to the shot (S)">${icon("target")}Jump to shot</button>` : ""}
              <span style="margin-left:auto" class="btn-row">${d.live ? '<span class="badge danger">LIVE</span>' : '<span class="badge info">DRY RUN</span>'}${slow ? `<span class="badge" title="recorded at ${d.fps.toFixed(0)} fps, played at ${d.clip_fps} fps">${slow.toFixed(1)}× slow</span>` : ""}</span>
              <label class="btn-row small muted" style="gap:6px"><input type="checkbox" id="s-loop"> loop shot</label>
            </div>
          </div>
        </div>
        <div class="card">
          <div class="card-h"><div><h2>${fmt.datetime(d.time)}</h2><div class="sub mono">${esc(d.name)}</div></div>
            <div class="right"><button class="btn sm ghost icon" id="s-star" title="Star">${icon("star")}</button>
              <div class="outcome" id="s-outcome"><button data-o="hit" class="btn sm" title="H">Hit</button><button data-o="miss" class="btn sm" title="M">Miss</button><button data-o="no_shot" class="btn sm">No shot</button></div></div></div>
          <div class="stats-row">
            <div class="stat"><span class="label">Length</span><span class="value">${fx(d.span_s, 1)}<small>s</small></span></div>
            <div class="stat"><span class="label">Frames</span><span class="value">${d.frames ?? "—"}<small>${isNum(d.fps) ? "@ " + d.fps.toFixed(0) + " fps" : ""}</small></span></div>
            <div class="stat"><span class="label">Shots</span><span class="value">${d.shots}</span></div>
            <div class="stat"><span class="label">Best P(hit)</span><span class="value">${isNum(d.max_p_hit) ? (d.max_p_hit * 100).toFixed(0) : "—"}<small>%</small></span></div>
            <div class="stat"><span class="label">Tracked</span><span class="value">${d.tracked ?? "—"}<small>frames</small></span></div>
          </div>
          <div class="field" style="margin-top:14px"><textarea class="input" id="s-note" rows="2" placeholder="Notes (station, wind, what happened)…">${esc(d.note || "")}</textarea></div>
        </div>
        ${(d.shot_list || []).length ? `<div class="card"><div class="card-h"><h2>Shots</h2><div class="right">
            <button class="btn sm" id="s-score">${icon("target")}${d.scored ? "Re-score" : "Score shots"}</button></div></div>
            <div class="grid" style="gap:10px">${d.shot_list.map((s) => `<div class="shot-card">
              ${s.image ? `<img src="${fileUrl(s.image)}" data-big="${fileUrl(s.image)}" alt="shot ${s.n}">` : `<div class="viewer" style="aspect-ratio:16/10;width:160px">${icon("image", 'width="24" height="24" style="color:#666"')}</div>`}
              <div><div class="btn-row"><b>Shot ${s.n}</b>${s.verdict ? `<span class="badge ${s.hit ? "ok" : "danger"}">${esc(s.verdict)}</span>` : '<span class="badge">not scored</span>'}
                <button class="btn sm ghost" data-jump="${s.clip_t}">${icon("play")}${fmt.vt(s.clip_t)}</button></div>
                <dl class="kv" style="margin-top:8px;max-width:420px"><dt>P(hit) at fire</dt><dd>${isNum(s.p_hit) ? (s.p_hit * 100).toFixed(0) + " %" : "—"}</dd>
                <dt>Range</dt><dd>${fx(s.range_m, 1, " m")}</dd>
                <dt>Predicted miss</dt><dd>${fx(s.miss_deg, 2, "°")}</dd>
                ${isNum(s.score_miss_deg) ? `<dt>Scored miss / pattern</dt><dd>${s.score_miss_deg.toFixed(2)}° / ${fx(s.pattern_deg, 2, "°")}</dd>` : ""}</dl></div></div>`).join("")}</div>
            <p class="small muted" style="margin:10px 0 0">“Score” replays the frames after the shot and checks where the clay really was when the pattern arrived. Mark what actually happened with Hit / Miss above.</p></div>` : ""}
        <div class="card"><div class="card-h"><h3>Files</h3><div class="right muted small">${fmt.bytes(d.size_bytes)} on the gun</div></div>
          <div class="btn-row">${d.clip ? `<a class="btn sm" href="${fileUrl(d.clip, 1)}">${icon("download")}Clip</a>` : ""}
            ${d.files.filter((f) => /\.(json|jsonl|csv)$/.test(f) && f !== "console.json").map((f) => `<a class="btn sm ghost" href="${fileUrl(f, 1)}">${icon("file")}${esc(f)}</a>`).join("")}
            ${saved.has(d.name) ? `<button class="btn sm ghost" id="s-open" title="${esc(archive.dir)}">${icon("check")}Saved on this PC · open folder</button>` : ""}
            <button class="btn sm outline-danger" id="s-del" style="margin-left:auto">${icon("trash")}Delete</button></div>
          <p class="small muted" style="margin:10px 0 0">Raw frames (frames.bin) stay on the gun; copy the whole folder with scp for replays.</p></div>
        <div class="small muted">Keys: <span class="kbd">Space</span> play · <span class="kbd">←</span><span class="kbd">→</span> frame · <span class="kbd">Shift</span>+arrows 1 s · <span class="kbd">S</span> shot · <span class="kbd">↑</span><span class="kbd">↓</span> session · <span class="kbd">H</span>/<span class="kbd">M</span> hit/miss</div>
      </div>`;
      drawOutcome();
      $$("button", $("#s-outcome")).forEach((b) => b.onclick = () => setOutcome(b.dataset.o));
      $("#s-star").onclick = async () => {
        const v = !detail.starred;
        try { await api(`sessions/${encodeURIComponent(d.name)}/notes`, { body: { starred: v } }); detail.starred = v; const s = list.find((x) => x.name === d.name); if (s) s.starred = v; drawList(); drawOutcome(); } catch (e) { toast(e.message, "err"); }
      };
      let noteT = null;
      $("#s-note").oninput = (e) => {
        clearTimeout(noteT);
        noteT = setTimeout(async () => { try { await api(`sessions/${encodeURIComponent(d.name)}/notes`, { body: { note: e.target.value } }); const s = list.find((x) => x.name === d.name); if (s) s.note = e.target.value; } catch (er) { toast(er.message, "err"); } }, 600);
      };
      const op = $("#s-open"); if (op) op.onclick = () => fetch(`/local/open?name=${encodeURIComponent(d.name)}`, { method: "POST" });
      $("#s-del").onclick = async () => {
        if (!(await confirmBox("Delete this press?", `<b>${esc(d.name)}</b> and all its frames (${fmt.bytes(d.size_bytes)}) are removed from the gun. This cannot be undone.`, "Delete", "danger"))) return;
        try { await api(`sessions/${encodeURIComponent(d.name)}`, { method: "DELETE" }); toast("Deleted.", "ok"); step(1); current = null; refresh(); } catch (e) { toast(e.message, "err"); }
      };
      const sc = $("#s-score");
      if (sc) sc.onclick = async () => {
        sc.disabled = true; sc.innerHTML = `<span class="spinner"></span>Scoring…`;
        try { const j = await api(`sessions/${encodeURIComponent(d.name)}/score`, { body: {} }); const r = await watchJob(j);
          if (r.code !== 0) toast("Scoring failed: " + (r.output.slice(-1)[0] || ""), "err"); else toast("Scored.", "ok");
          refresh(); select(d.name, true);
        } catch (e) { toast(e.message, "err"); sc.disabled = false; }
      };
      const mk = $("#s-mk");
      if (mk) mk.onclick = async () => {
        mk.disabled = true; mk.innerHTML = `<span class="spinner"></span>Rendering…`;
        try { const j = await api(`sessions/${encodeURIComponent(d.name)}/clip`, { body: {} }); const r = await watchJob(j);
          if (r.code !== 0) toast("Render failed: " + (r.output.slice(-1)[0] || ""), "err"); refresh(); select(d.name, true);
        } catch (e) { toast(e.message, "err"); }
      };
      $$("img[data-big]", box).forEach((im) => im.onclick = () => modal({ title: "Arrival frame", body: `<img src="${im.dataset.big}">`, wide: true, actions: [{ label: "Close" }] }));
      $$("[data-jump]", box).forEach((b) => b.onclick = () => { const v = $("#s-video"); if (v) { v.currentTime = Math.max(0, +b.dataset.jump - 0.4); v.play(); } });

      const v = $("#s-video");
      if (!v) return () => {};
      const tl = $("#s-tl"), fill = $(".fill", tl), knob = $(".knob", tl);
      const frame = 1 / (d.clip_fps || 30);
      const firstShot = d.shot_list && d.shot_list.length ? d.shot_list[0].clip_t : null;
      const upd = () => {
        const dur = v.duration || 0, t = v.currentTime || 0, pct = dur ? (t / dur) * 100 : 0;
        fill.style.width = pct + "%"; knob.style.left = pct + "%";
        $("#s-time").textContent = `${fmt.vt(t)} / ${fmt.vt(dur)}`;
        $("#s-play").innerHTML = icon(v.paused ? "play" : "pause");
        if ($("#s-loop").checked && firstShot !== null && !v.paused && t > firstShot + 1.2) v.currentTime = Math.max(0, firstShot - 1.0);
      };
      v.addEventListener("timeupdate", upd); v.addEventListener("play", upd); v.addEventListener("pause", upd);
      v.addEventListener("loadedmetadata", () => {
        $$(".mark", tl).forEach((m) => m.style.left = (v.duration ? (+m.dataset.t / v.duration) * 100 : 0) + "%");
        upd();
        if (firstShot !== null) v.currentTime = Math.max(0, firstShot - 1.0);
        v.play().catch(() => {});
      });
      let raf = 0; const loop = () => { upd(); raf = requestAnimationFrame(loop); }; raf = requestAnimationFrame(loop);
      const seek = (e) => { const r = tl.getBoundingClientRect(); v.currentTime = clamp((e.clientX - r.left) / r.width, 0, 1) * (v.duration || 0); };
      tl.onmousedown = (e) => { seek(e); const mv = (ev) => seek(ev); const up = () => { document.removeEventListener("mousemove", mv); document.removeEventListener("mouseup", up); }; document.addEventListener("mousemove", mv); document.addEventListener("mouseup", up); };
      $("#s-play").onclick = () => (v.paused ? v.play() : v.pause());
      v.onclick = () => (v.paused ? v.play() : v.pause());
      $("#s-fb").onclick = () => { v.pause(); v.currentTime = Math.max(0, v.currentTime - frame); };
      $("#s-ff").onclick = () => { v.pause(); v.currentTime = Math.min(v.duration || 0, v.currentTime + frame); };
      let rate = 1; try { rate = +localStorage.getItem("tbot.rate") || 1; } catch (e) {}
      const setRate = (r) => { rate = r; v.playbackRate = r; $$("button", $("#s-rate")).forEach((b) => b.classList.toggle("on", +b.dataset.r === r)); try { localStorage.setItem("tbot.rate", r); } catch (e) {} };
      setRate(rate);
      $$("button", $("#s-rate")).forEach((b) => b.onclick = () => setRate(+b.dataset.r));
      const jb = $("#s-jump"); if (jb) jb.onclick = () => { v.currentTime = Math.max(0, firstShot - 1.0); v.play(); };
      return () => { cancelAnimationFrame(raf); v.pause(); v.removeAttribute("src"); v.load(); };
    };

    const onKey = (e) => {
      if (e.target.matches("input, textarea, select")) return;
      const v = $("#s-video");
      const frame = 1 / ((detail && detail.clip_fps) || 30);
      if (e.key === " " && v) { e.preventDefault(); v.paused ? v.play() : v.pause(); }
      else if (e.key === "ArrowLeft" && v) { e.preventDefault(); v.pause(); v.currentTime = Math.max(0, v.currentTime - (e.shiftKey ? 1 : frame)); }
      else if (e.key === "ArrowRight" && v) { e.preventDefault(); v.pause(); v.currentTime += e.shiftKey ? 1 : frame; }
      else if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); step(-1); }
      else if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); step(1); }
      else if ((e.key === "s" || e.key === "S") && v && detail.shot_list.length) { v.currentTime = Math.max(0, detail.shot_list[0].clip_t - 1.0); v.play(); }
      else if (e.key === "h" || e.key === "H") setOutcome("hit");
      else if (e.key === "m" || e.key === "M") setOutcome("miss");
    };
    document.addEventListener("keydown", onKey);
    $("#s-q").oninput = (e) => { query = e.target.value.trim().toLowerCase(); drawList(); };
    refresh();
    const tick = setInterval(() => refresh(true), 10000);  // new presses, fresh previews and clips appear by themselves
    const offStatus = S.onStatus(() => { if ($("#s-foot")) drawFoot(); });
    return () => { clearInterval(tick); offStatus(); document.removeEventListener("keydown", onKey); if (player) player(); };
  },
};

// ============================================================================================ CALIBRATION HUB
PAGES.calibration = {
  title: "Calibration",
  render(el) {
    el.innerHTML = `<div class="callout info">${icon("info")}<div>Calibrate in this order after any change to the lens, the mount or the trigger linkage:
      <b>focus</b> → <b>boresight</b> (laser in the bore) → <b>trigger servo</b>. Each saved result keeps a dated backup on the gun.</div></div>
      <div class="grid g3" id="cal-cards"><div class="card"><div class="spinner"></div></div></div>`;
    loadCal().then((cal) => {
      const b = cal.boresight && cal.boresight.data, i = cal.intrinsics && cal.intrinsics.data, lat = cal.latency && cal.latency.data, imu = cal.imu && cal.imu.data;
      const delay = triggerDelay(lat);
      const card = (ic, title, text, rows, status, actions) => `<div class="card cal-card"><div class="top"><div class="ic">${icon(ic)}</div><div><h3>${title}</h3>${status}</div></div>
        <p>${text}</p>${rows ? `<dl class="kv">${rows}</dl>` : ""}<div class="foot">${actions}</div></div>`;
      $("#cal-cards").innerHTML = [
        card("camera", "Camera & focus", "Live picture with a sharpness meter and exposure numbers. Use it after touching the lens.",
          i ? `<dt>Focal length</dt><dd>${i.K ? i.K[0][0].toFixed(0) : "—"} px @ ${i.width}×${i.height}</dd><dt>Model</dt><dd>${esc(i.model || "—")}</dd>` : "",
          calAge(cal.intrinsics), `<a class="btn primary" href="#/camera">${icon("camera")}Open camera</a>`),
        card("laser", "Boresight", "Where the barrel points in the picture. Bore laser in the barrel, a wall at a measured distance, one slow twist.",
          b && b.rpy_deg ? `<dt>Pitch</dt><dd>${b.rpy_deg[1].toFixed(3)}°</dd><dt>Yaw</dt><dd>${b.rpy_deg[2].toFixed(3)}°</dd><dt>Roll</dt><dd>${b.rpy_deg[0].toFixed(3)}°</dd>` : "",
          calAge(cal.boresight), `<a class="btn primary" href="#/boresight">${icon("laser")}Calibrate / check</a>`),
        card("trigger", "Trigger servo", "Rest and fire pulse widths of the MG996R. Gun unloaded; moves only while you hold the arming button.",
          cal.servo.rest_us ? `<dt>Rest</dt><dd>${cal.servo.rest_us} µs</dd><dt>Fire</dt><dd>${cal.servo.fire_us} µs</dd><dt>Travel</dt><dd>${cal.servo.min_us}–${cal.servo.max_us} µs</dd><dt>Hold</dt><dd>${cal.servo.hold_ms} ms</dd><dt>Break</dt><dd>${cal.servo.break_us ? `${cal.servo.break_us} µs` : "not marked"}</dd><dt>Pre-travel</dt><dd>${cal.servo.break_us && cal.servo.preload_frac ? `${Math.round(cal.servo.preload_frac * 100)} %` : "off"}</dd>` : "",
          cal.servo.max_us ? '<span class="badge ok">Set</span>' : '<span class="badge danger">Not set</span>', `<a class="btn primary" href="#/trigger">${icon("trigger")}Set up servo</a>`),
        card("timer", "Trigger delay", "Command to sear break: the solver leads the clay by this. Re-measure after any change to the servo or linkage.",
          isNum(delay) ? `<dt>Command → break</dt><dd>${delay.toFixed(1)} ms</dd>` : "",
          calAge(cal.latency), `<span class="small muted">Measure on the Pi: <span class="mono">tbot latency-mic</span> (microphone) or <span class="mono">tbot latency-recoil</span> from live shots.</span>`),
        card("imu", "IMU", "Gyro bias and the axis mapping of the motion sensor. The gyro re-zeroes by itself whenever the gun rests.",
          imu ? Object.entries(imu).filter(([k, v]) => isNum(v)).slice(0, 3).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v.toFixed(3)}</dd>`).join("") : "",
          calAge(cal.imu), `<span class="small muted">On the Pi: <span class="mono">tbot calib-imu bias</span> / <span class="mono">axes</span>.</span>`),
        card("file", "History", "Every boresight saved from this console is backed up first.",
          cal.history.length ? cal.history.slice(0, 4).map((h) => `<dt class="mono small">${esc(h)}</dt><dd></dd>`).join("") : `<dt class="muted">No backups yet</dt><dd></dd>`,
          "", `<a class="btn" href="#/diagnostics">${icon("diag")}Run doctor</a>`),
      ].join("");
    }).catch((e) => toast(e.message, "err"));
  },
};

// ============================================================================================ CAMERA & FOCUS
const CAM_DEFAULTS = { width: 1280, height: 800, fps: 30, exposure_us: 0, gain: 1 };
function camOpts() { try { return { ...CAM_DEFAULTS, ...JSON.parse(localStorage.getItem("tbot.cam") || "{}") }; } catch (e) { return { ...CAM_DEFAULTS }; } }
function saveCamOpts(o) { try { localStorage.setItem("tbot.cam", JSON.stringify(o)); } catch (e) {} }

function cameraControls(opts) {
  return `<div class="card"><div class="card-h"><h3>Exposure</h3></div>
    <div class="grid" style="gap:12px">
      <div class="seg" id="c-ae"><button data-v="auto" class="${opts.exposure_us ? "" : "on"}">Auto</button><button data-v="manual" class="${opts.exposure_us ? "on" : ""}">Manual</button></div>
      <div id="c-man" class="${opts.exposure_us ? "" : "hidden"} grid" style="gap:10px">
        <div class="field"><label>Exposure</label><div class="input-group"><input class="input mono" id="c-exp" type="number" min="0.01" max="100" step="0.5" value="${(opts.exposure_us || 5000) / 1000}"><span class="addon">ms</span></div></div>
        <div class="field"><label>Gain</label><div class="input-group"><input class="input mono" id="c-gain" type="number" min="1" max="15.9" step="0.5" value="${opts.gain || 1}"><span class="addon">×</span></div></div>
      </div>
      <div class="field"><label>Resolution</label><select class="input" id="c-res"><option value="1280x800">1280 × 800 (full, for calibration)</option><option value="640x400">640 × 400 (as the pipeline runs)</option></select></div>
      <button class="btn" id="c-apply">${icon("refresh")}Apply (restarts the camera)</button>
      <p class="small muted" style="margin:0">Auto here allows long exposures for a still scene; the pipeline's own auto-exposure is not changed. Laser dots indoors: manual 5 ms, gain 2.</p>
    </div></div>`;
}
function wireCameraControls(root, onApply) {
  const o = camOpts();
  $("#c-res", root).value = `${o.width}x${o.height}`;
  $$("#c-ae button", root).forEach((b) => b.onclick = () => {
    $$("#c-ae button", root).forEach((x) => x.classList.toggle("on", x === b));
    $("#c-man", root).classList.toggle("hidden", b.dataset.v === "auto");
  });
  $("#c-apply", root).onclick = () => {
    const manual = $("#c-ae .on", root).dataset.v === "manual";
    const [w, h] = $("#c-res", root).value.split("x").map(Number);
    const n = { ...o, width: w, height: h, exposure_us: manual ? Math.round(+$("#c-exp", root).value * 1000) : 0, gain: +$("#c-gain", root).value || 1 };
    saveCamOpts(n); onApply(n);
  };
}

PAGES.camera = {
  title: "Camera & focus",
  render(el) {
    el.innerHTML = `<div class="split">
      <div class="grid" style="gap:12px">
        ${streamViewer("c-view", { hud: `<span class="badge" id="c-hud">starting…</span>` })}
        <div class="btn-row"><button class="btn" id="c-still">${icon("image")}Save still on the gun</button><button class="btn ghost" id="c-peak">${icon("undo")}Reset peak</button>
          <span class="small muted" style="margin-left:auto">Box = the centre region the sharpness number is measured in.</span></div>
      </div>
      <div class="grid" style="gap:16px">
        <div class="card"><div class="card-h"><h3>Sharpness</h3><div class="right"><span class="badge" id="c-pct">—</span></div></div>
          <div class="bigread" id="c-sharp">—</div>
          <div class="meter ok" style="margin:12px 0 6px"><span id="c-bar" style="width:0"></span></div>
          <div class="small muted">Turn the lens slowly; stop where the bar peaks (100 % = best seen since reset).</div>
          <svg class="spark" id="c-spark" viewBox="0 0 200 56" preserveAspectRatio="none"></svg>
          <div class="quad" id="c-quad"></div></div>
        <div class="card"><div class="card-h"><h3>Picture</h3></div>
          <div class="stats-row"><div class="stat"><span class="label">Mean</span><span class="value" id="c-mean">—</span></div>
            <div class="stat"><span class="label">Clipped</span><span class="value" id="c-sat">—</span></div>
            <div class="stat"><span class="label">FPS</span><span class="value" id="c-fps">—</span></div></div>
          <svg id="c-hist" viewBox="0 0 64 40" preserveAspectRatio="none" style="width:100%;height:60px;margin-top:10px;display:block"></svg>
          <div class="small muted" id="c-exp-now"></div></div>
        ${cameraControls(camOpts())}
      </div></div>`;
    let alive = true, stopEv = null, stopStream = null;
    const hist = [];
    const start = async (force) => {
      if (stopEv) stopEv(); if (stopStream) stopStream();
      try {
        const h = await ensureTool("camera", camOpts(), force);
        if (!h) { $("#c-hud").textContent = "camera not started"; $(".placeholder div", $("#c-view")).textContent = "The pipeline is still running. Stop it to use the camera tool."; return; }
      } catch (e) { toast(e.message, "err"); return; }
      pollStatus();
      stopStream = attachStream($("#c-view"));
      stopEv = toolEvents(onEv, () => alive);
    };
    const onEv = (ev) => {
      if (ev.type === "fatal") toast(ev.message, "err");
      if (ev.type === "ready") {
        const svg = $("#c-view svg.overlay");
        svg.setAttribute("viewBox", `0 0 ${ev.width} ${ev.height}`);
        const rw = ev.width * 0.5, rh = ev.height * 0.5;
        svg.innerHTML = `<rect x="${(ev.width - rw) / 2}" y="${(ev.height - rh) / 2}" width="${rw}" height="${rh}" fill="none" stroke="rgba(255,122,26,.8)" stroke-width="${ev.width / 500}" stroke-dasharray="${ev.width / 100}"/>
          <line x1="${ev.width / 2}" y1="0" x2="${ev.width / 2}" y2="${ev.height}" stroke="rgba(255,255,255,.18)" stroke-width="${ev.width / 900}"/><line x1="0" y1="${ev.height / 2}" x2="${ev.width}" y2="${ev.height / 2}" stroke="rgba(255,255,255,.18)" stroke-width="${ev.width / 900}"/>`;
      }
      if (ev.type === "still") toast("Saved " + ev.path, "ok");
      if (ev.type !== "stats") return;
      $("#c-hud").textContent = `${ev.width}×${ev.height} · ${fx(ev.exposure_us / 1000, 2)} ms × ${fx(ev.gain, 1)}`;
      $("#c-sharp").innerHTML = `${fx(ev.sharpness, 0)}<small> DN²</small>`;
      $("#c-pct").textContent = isNum(ev.peak_pct) ? ev.peak_pct.toFixed(0) + " % of peak" : "—";
      $("#c-pct").className = "badge " + (ev.peak_pct >= 97 ? "ok" : ev.peak_pct >= 85 ? "warn" : "danger");
      $("#c-bar").style.width = (ev.peak_pct || 0) + "%";
      hist.push(ev.sharpness || 0); if (hist.length > 120) hist.shift();
      const mx = Math.max(...hist, 1);
      $("#c-spark").innerHTML = `<polyline fill="none" stroke="var(--accent)" stroke-width="1.5" vector-effect="non-scaling-stroke" points="${hist.map((v, i) => `${(i / 119) * 200},${54 - (v / mx) * 50}`).join(" ")}"/>`;
      const q = ev.quadrants || {};
      $("#c-quad").innerHTML = [["top_left", "Top left"], ["top_right", "Top right"], ["bottom_left", "Bottom left"], ["bottom_right", "Bottom right"]].map(([k, l]) => `<div><b>${fx(q[k], 0)}</b><span class="muted">${l}</span></div>`).join("");
      $("#c-mean").innerHTML = `${fx(ev.mean_dn, 0)}<small>DN</small>`;
      $("#c-sat").innerHTML = `${fx(ev.saturated_pct, 1)}<small>%</small>`;
      $("#c-sat").style.color = ev.saturated_pct > 1 ? "var(--danger)" : "";
      $("#c-fps").textContent = fx(ev.fps, 0);
      const hm = Math.max(...ev.hist, 1);
      $("#c-hist").innerHTML = ev.hist.map((v, i) => `<rect x="${i}" y="${40 - (v / hm) * 40}" width="0.85" height="${(v / hm) * 40}" fill="${i >= 63 ? "var(--danger)" : "var(--text-2)"}"/>`).join("");
      $("#c-exp-now").textContent = `Exposure ${fx(ev.exposure_us, 0)} µs, gain ${fx(ev.gain, 2)}×`;
    };
    $("#c-still").onclick = () => sendTool({ cmd: "still" });
    $("#c-peak").onclick = () => { hist.length = 0; sendTool({ cmd: "reset_peak" }); };
    wireCameraControls(el, () => start(true));
    start(false);
    return () => { alive = false; if (stopEv) stopEv(); if (stopStream) stopStream(); };
  },
};

// ============================================================================================ BORESIGHT
PAGES.boresight = {
  title: "Boresight",
  render(el) {
    let units = "ft";
    try { units = localStorage.getItem("tbot.units") || "ft"; } catch (e) {}
    const W = { step: 1, dist: null, window: null, size: null, intr: null, ring: null, dots: [], mode: "check", capturing: false, stats: null };
    el.innerHTML = `<div class="steps" id="b-steps"></div>
      <div class="split">
        <div class="grid" style="gap:12px">
          ${streamViewer("b-view", { hud: `<span class="badge" id="b-hud">camera starting…</span>` })}
          <div class="small muted" id="b-hint">Click on the laser dot to tell the tool where to look.</div>
        </div>
        <div class="grid" style="gap:16px" id="b-side"></div>
      </div>`;
    let alive = true, stopEv = null, stopStream = null;
    let cal = S.cal;
    const off = lensToMuzzle(cal);
    const muzzleRange = () => (W.dist === null ? null : (units === "ft" ? W.dist * 0.3048 : W.dist) - lensToMuzzle(cal));
    const toIntr = (x, y) => { const s = W.intr && W.size ? W.intr[0] / W.size[0] : 1; return [x * s, y * s]; };
    const drawSteps = () => {
      const names = ["Set up", "Find the dot", "Twist", "Result"];
      $("#b-steps").innerHTML = names.map((n, i) => `<div class="step ${i + 1 < W.step ? "done" : i + 1 === W.step ? "cur" : ""}"><span class="n">${i + 1 < W.step ? "✓" : i + 1}</span>${n}</div>`).join("");
    };
    const overlay = () => {
      const svg = $("#b-view svg.overlay"); if (!W.size) return;
      const [w, h] = W.size, sw = w / 400;
      let s = "";
      if (W.stats && W.stats.bore_px) {
        const [u, v] = W.stats.bore_px;
        s += `<g stroke="#5aa9ff" stroke-width="${sw}" fill="none"><circle cx="${u}" cy="${v}" r="${w / 60}"/><path d="M${u - w / 30} ${v}h${w / 15}M${u} ${v - w / 30}v${w / 15}"/></g>
          <text x="${u + w / 50}" y="${v - w / 50}" fill="#5aa9ff" font-size="${w / 70}" font-family="sans-serif">stored zero @ ${muzzleRange() ? muzzleRange().toFixed(2) : "?"} m</text>`;
      }
      if (W.window) {
        const [cx, cy, r] = W.window;
        s += `<rect x="${cx - r}" y="${cy - r}" width="${2 * r}" height="${2 * r}" fill="none" stroke="rgba(255,122,26,.75)" stroke-width="${sw}" stroke-dasharray="${sw * 5}"/>`;
      }
      for (const d of W.dots) s += `<circle cx="${d[0]}" cy="${d[1]}" r="${sw * 2}" fill="${d[2] === false ? "#ff4d5a" : "#2fd39d"}"/>`;
      if (W.ring) s += `<g stroke="#ff7a1a" stroke-width="${sw * 1.4}" fill="none"><circle cx="${W.ring.cx}" cy="${W.ring.cy}" r="${W.ring.r}"/><path d="M${W.ring.cx - w / 40} ${W.ring.cy}h${w / 20}M${W.ring.cx} ${W.ring.cy - w / 40}v${w / 20}"/></g>`;
      svg.innerHTML = s;
    };
    const side = () => {
      const box = $("#b-side");
      drawSteps();
      if (W.step === 1) {
        box.innerHTML = `<div class="card"><div class="card-h"><h3>1 · Set up</h3></div>
          <ol style="margin:0 0 14px;padding-left:18px;color:var(--text-2);display:flex;flex-direction:column;gap:6px">
            <li><b>Unload the gun.</b> Put the bore laser in the barrel.</li>
            <li>Aim at a flat wall or a sheet of paper at a measured distance: 20–50 ft works. Further is more accurate, but the dot gets fainter.</li>
            <li>Measure from the <b>front of the camera lens</b> to the wall.</li></ol>
          <div class="field"><label>Distance from lens front to the target</label>
            <div class="btn-row" style="flex-wrap:nowrap"><input class="input mono" id="b-dist" type="number" step="0.1" min="2" style="flex:1" value="${W.dist ?? ""}" placeholder="e.g. 35">
            <div class="seg" id="b-units"><button data-u="ft" class="${units === "ft" ? "on" : ""}">ft</button><button data-u="m" class="${units === "m" ? "on" : ""}">m</button></div></div>
            <span class="small muted" id="b-conv">The solver uses the range from the muzzle (${(off * 1000).toFixed(0)} mm ahead of the lens).</span></div>
          <div class="field" style="margin-top:12px"><label>What do you want to do?</label>
            <div class="seg" id="b-mode"><button data-m="check" class="${W.mode === "check" ? "on" : ""}">Check the zero</button><button data-m="solve" class="${W.mode === "solve" ? "on" : ""}">Set a new zero</button></div>
            <span class="small muted">Check changes nothing; it tells you how far off the stored zero is (pass ≤ 0.15°). Set a new zero after the camera or lens moved.</span></div>
          <div class="btn-row" style="margin-top:16px"><button class="btn primary" id="b-next">Next${icon("next")}</button></div></div>
          ${cameraControls(camOpts())}`;
        const conv = () => { const v = parseFloat($("#b-dist").value); W.dist = isNum(v) ? v : null; const m = muzzleRange();
          $("#b-conv").textContent = m ? `= ${m.toFixed(3)} m from the muzzle (the solver's range).` : "The solver uses the range from the muzzle."; if (m) sendTool({ cmd: "bore", range_m: m }); };
        $("#b-dist").oninput = conv; conv();
        $$("#b-units button").forEach((b) => b.onclick = () => { units = b.dataset.u; try { localStorage.setItem("tbot.units", units); } catch (e) {} side(); });
        $$("#b-mode button").forEach((b) => b.onclick = () => { W.mode = b.dataset.m; side(); });
        $("#b-next").onclick = () => { if (!muzzleRange() || muzzleRange() < 1) { $("#b-dist").classList.add("err"); toast("Enter the distance to the target.", "warn"); return; } W.step = 2; side(); };
        wireCameraControls(box, () => startCam(true));
      } else if (W.step === 2) {
        box.innerHTML = `<div class="card"><div class="card-h"><h3>2 · Find the dot</h3></div>
          <p class="muted" style="margin:0 0 12px">Click on the laser dot in the picture. A search box appears around it; only that area is searched, so lamps and reflections elsewhere do not matter.</p>
          <div class="callout ${W.probe ? (W.probe.ok ? "ok" : "warn") : ""}" id="b-probe">${icon(W.probe && W.probe.ok ? "check" : "info")}<div>${W.probe ? esc(W.probe.text) : "No dot selected yet."}</div></div>
          <div class="field" style="margin-top:12px"><label>Search box size</label><input type="range" id="b-r" min="40" max="300" value="${W.window ? W.window[2] : 140}"></div>
          <div class="btn-row" style="margin-top:14px"><button class="btn" id="b-back">${icon("prev")}Back</button><button class="btn primary" id="b-next" ${W.window ? "" : "disabled"}>Next${icon("next")}</button></div></div>
          <div class="card"><div class="card-h"><h3>No laser handy?</h3></div><p class="small muted" style="margin:0 0 10px">If you can see exactly where the bore points (for example a mark you sighted through the barrel), click it, then use it directly.</p>
            <button class="btn sm" id="b-manual" ${W.window ? "" : "disabled"}>Use the clicked point as the bore</button></div>`;
        $("#b-back").onclick = () => { W.step = 1; side(); };
        $("#b-next").onclick = () => { W.step = 3; W.dots = []; W.ring = null; side(); overlay(); };
        $("#b-r").oninput = (e) => { if (W.window) { W.window[2] = +e.target.value; overlay(); } };
        $("#b-manual").onclick = () => { W.ring = { cx: W.window[0], cy: W.window[1], r: 0, manual: true, grade: "manual", sectors: 0, on_ring: 0, clean: 0, frames: 0, sd_px: [NaN, NaN], scatter_px: NaN }; W.step = 4; side(); overlay(); };
      } else if (W.step === 3) {
        box.innerHTML = `<div class="card"><div class="card-h"><h3>3 · Twist the laser</h3></div>
          <p class="muted" style="margin:0 0 12px">A bore laser is never exactly on the bore axis, so turning it in the bore makes the dot draw a circle. The centre of that circle is where the barrel points.</p>
          <ol style="margin:0 0 14px;padding-left:18px;color:var(--text-2);display:flex;flex-direction:column;gap:4px">
            <li>Press <b>Start</b>, then turn the laser slowly with your fingertips, one full turn or more.</li>
            <li>Do not push it sideways. Moving the gun a little does not matter: the camera moves with it.</li></ol>
          <div class="field"><label>Capture length</label><div class="seg" id="b-sec">${[6, 10, 15].map((s) => `<button data-s="${s}" class="${(W.seconds || 10) === s ? "on" : ""}">${s} s</button>`).join("")}</div></div>
          <div style="margin:16px 0 4px" class="btn-row"><div class="bigread" id="b-count">${W.capturing ? "" : "Ready"}</div><span class="muted" id="b-n"></span></div>
          <div class="meter"><span id="b-prog" style="width:0"></span></div>
          <div class="btn-row" style="margin-top:16px"><button class="btn" id="b-back" ${W.capturing ? "disabled" : ""}>${icon("prev")}Back</button>
            <button class="btn ${W.capturing ? "danger" : "primary"}" id="b-go">${W.capturing ? icon("stop") + "Cancel" : icon("play") + "Start"}</button></div>
          ${W.error ? `<div class="callout warn" style="margin-top:14px">${icon("alert")}<div>${esc(W.error)}</div></div>` : ""}</div>`;
        $$("#b-sec button").forEach((b) => b.onclick = () => { W.seconds = +b.dataset.s; side(); });
        $("#b-back").onclick = () => { W.step = 2; side(); };
        $("#b-go").onclick = () => {
          if (W.capturing) { sendTool({ cmd: "cancel_capture" }); W.capturing = false; side(); return; }
          W.dots = []; W.ring = null; W.error = null; W.capturing = true; W.capStart = Date.now();
          sendTool({ cmd: "capture", seconds: W.seconds || 10, window: W.window, range_m: muzzleRange() });
          side();
        };
      } else {
        const r = W.ring, m = muzzleRange();
        const [iu, iv] = toIntr(r.cx, r.cy);
        let diff = "";
        if (W.stats && W.stats.bore_px && cal && cal.intrinsics) {
          const fxp = cal.intrinsics.data.K[0][0] * (W.size[0] / W.intr[0]);
          const du = r.cx - W.stats.bore_px[0], dv = r.cy - W.stats.bore_px[1];
          const deg = Math.atan(Math.hypot(du, dv) / fxp) * 180 / Math.PI;
          diff = `<dt>Off the stored zero</dt><dd><b style="color:${deg <= 0.15 ? "var(--ok)" : "var(--warn)"}">${deg.toFixed(3)}°</b> (${(Math.tan(deg * Math.PI / 180) * 20 * 100).toFixed(1)} cm at 20 m)</dd>`;
        }
        const gcls = { excellent: "ok", good: "ok", fair: "warn", poor: "danger", manual: "info" }[r.grade];
        box.innerHTML = `<div class="card"><div class="card-h"><h3>4 · Result</h3><div class="right"><span class="badge ${gcls}">${esc(r.grade.toUpperCase())}</span></div></div>
          <dl class="kv"><dt>Bore pixel</dt><dd class="mono">${iu.toFixed(2)}, ${iv.toFixed(2)}${W.intr && W.intr[0] !== W.size[0] ? ` <span class="muted">(${W.intr[0]}×${W.intr[1]})</span>` : ""}</dd>
            ${r.manual ? "" : `<dt>Ring radius</dt><dd>${r.r.toFixed(1)} px</dd><dt>Centre uncertainty</dt><dd>± ${Math.max(...r.sd_px).toFixed(2)} px</dd>
            <dt>Scatter</dt><dd>${r.scatter_px.toFixed(1)} px</dd><dt>Coverage</dt><dd>${r.sectors}/12 sectors · ${r.on_ring} of ${r.frames} frames</dd>`}
            <dt>Range (muzzle)</dt><dd>${m.toFixed(3)} m</dd>${diff}</dl>
          ${r.grade === "poor" ? `<div class="callout warn" style="margin-top:12px">${icon("alert")}<div>The ring is not well covered. Twist again more slowly, a full turn, keeping the dot visible.</div></div>` : ""}
          <div class="btn-row" style="margin-top:16px"><button class="btn" id="b-again">${icon("refresh")}Twist again</button>
            <button class="btn ${W.mode === "check" ? "primary" : ""}" id="b-check">${icon("check")}Check zero</button>
            <button class="btn ${W.mode === "solve" ? "primary" : ""}" id="b-solve">${icon("target")}Save as new zero</button></div></div>
          <div class="card ${W.job ? "" : "hidden"}" id="b-jobcard"><div class="card-h"><h3 id="b-jobt">Solver</h3><div class="right" id="b-jobs"></div></div><pre class="term" id="b-out"></pre></div>`;
        $("#b-again").onclick = () => { W.step = 3; W.dots = []; W.ring = null; side(); overlay(); };
        const run = async (check) => {
          if (!check && !(await confirmBox("Save a new zero?", `The boresight is solved from bore pixel <b>${iu.toFixed(2)}, ${iv.toFixed(2)}</b> at <b>${m.toFixed(3)} m</b> and written to <span class="mono">calib/boresight.json</span>. The current file is backed up first. Restart the pipeline afterwards to use it.`, "Save new zero"))) return;
          $("#b-jobcard").classList.remove("hidden");
          $("#b-jobt").textContent = check ? "Checking the stored zero" : "Solving the new zero";
          $("#b-jobs").innerHTML = `<span class="spinner"></span>`;
          try {
            const j = await api("boresight", { body: { u: iu, v: iv, range_m: m, check } });
            const res = await watchJob(j, (jj) => { $("#b-out").innerHTML = colorizeTerm(jj.output); });
            const pass = res.output.some((l) => /^PASS/.test(l)), fail = res.output.some((l) => /^FAIL/.test(l));
            $("#b-jobs").innerHTML = check ? `<span class="badge ${pass ? "ok" : "danger"}">${pass ? "PASS" : fail ? "FAIL" : "ERROR"}</span>`
              : `<span class="badge ${res.code === 0 ? "ok" : "danger"}">${res.code === 0 ? "SAVED" : "ERROR"}</span>`;
            if (!check && res.code === 0) {
              await loadCal(); cal = S.cal;
              toast("New zero saved. Restart the pipeline to use it.", "ok", { action: { label: "Start dry run", run: () => pipelineAction("start", "dry") }, ms: 10000 });
            }
          } catch (e) { $("#b-jobs").innerHTML = ""; toast(e.message, "err"); }
        };
        $("#b-check").onclick = () => run(true);
        $("#b-solve").onclick = () => run(false);
      }
    };
    const onEv = (ev) => {
      if (ev.type === "fatal") toast(ev.message, "err");
      if (ev.type === "ready") {
        W.size = [ev.width, ev.height]; W.intr = ev.intrinsics_size;
        $("#b-view svg.overlay").setAttribute("viewBox", `0 0 ${ev.width} ${ev.height}`);
        const m = muzzleRange(); if (m) sendTool({ cmd: "bore", range_m: m });
        overlay();
      }
      if (ev.type === "stats") {
        W.stats = ev;
        if (!W.size) { W.size = [ev.width, ev.height]; $("#b-view svg.overlay").setAttribute("viewBox", `0 0 ${ev.width} ${ev.height}`); }
        $("#b-hud").textContent = `${ev.width}×${ev.height} · ${fx(ev.exposure_us / 1000, 2)} ms × ${fx(ev.gain, 1)} · ${fx(ev.fps, 0)} fps${ev.saturated_pct > 0.5 ? " · CLIPPING" : ""}`;
        overlay();
      }
      if (ev.type === "dot_probe") {
        W.probe = ev.dot ? { ok: ev.dot[2] > 8 && ev.dot[3] < 0.5 * ev.dot[2], text: ev.dot[2] > 8 && ev.dot[3] < 0.5 * ev.dot[2] ? `Dot found: ${ev.dot[2].toFixed(0)}× above the noise, nothing else close in strength.` : `Weak or ambiguous (signal ${ev.dot[2].toFixed(0)}×, second peak ${ev.dot[3].toFixed(0)}×): shrink the box, dim the room or lower the exposure.` } : { ok: false, text: "Nothing found in the box." };
        if (ev.dot && W.window && W.probe.ok) { W.window[0] = ev.dot[0]; W.window[1] = ev.dot[1]; }
        if (W.step === 2) side(); overlay();
      }
      if (ev.type === "capture_started") W.capturing = true;
      if (ev.type === "dot" && W.capturing) {
        W.dots.push([ev.x, ev.y]);
        const c = $("#b-count"); if (c) c.innerHTML = `${ev.left_s.toFixed(1)}<small> s</small>`;
        const n = $("#b-n"); if (n) n.textContent = `${ev.n} dots`;
        const p = $("#b-prog"); if (p) p.style.width = (100 * (1 - ev.left_s / (W.seconds || 10))) + "%";
        overlay();
      }
      if (ev.type === "ring") { W.capturing = false; W.ring = ev; W.dots = ev.points; W.step = 4; side(); overlay(); }
      if (ev.type === "ring_error") { W.capturing = false; W.error = ev.message; side(); }
      if (ev.type === "capture_cancelled") { W.capturing = false; }
    };
    const svg = $("#b-view svg.overlay");
    svg.classList.add("interactive");
    svg.addEventListener("click", (e) => {
      if (!W.size || W.step > 2 || W.capturing) return;
      const p = svgPoint(svg, e);
      W.window = [p.x, p.y, W.window ? W.window[2] : 140];
      if (W.step === 1 && muzzleRange()) W.step = 2;
      sendTool({ cmd: "find_dot", window: W.window });
      side(); overlay();
    });
    const startCam = async (force) => {
      if (stopEv) stopEv(); if (stopStream) stopStream();
      try {
        const h = await ensureTool("camera", camOpts(), force);
        if (!h) { $("#b-hud").textContent = "camera not started"; return; }
      } catch (e) { toast(e.message, "err"); return; }
      W.size = null;
      pollStatus();
      stopStream = attachStream($("#b-view"));
      stopEv = toolEvents(onEv, () => alive);
    };
    (cal ? Promise.resolve(cal) : loadCal()).then((c) => { cal = c; side(); }).catch(() => side());
    startCam(false);
    return () => { alive = false; if (stopEv) stopEv(); if (stopStream) stopStream(); };
  },
};

// ============================================================================================ TRIGGER SERVO
PAGES.trigger = {
  title: "Trigger servo",
  render(el) {
    el.innerHTML = `<div class="callout danger">${icon("shield")}<div><b>Gun UNLOADED</b> — check the chamber and the magazine now. Nothing here needs a round, ever.
      The servo moves only while you <b>hold the arming button</b>; letting go returns it to rest at once. Fingers out of the trigger guard while it moves.</div></div>
      <div class="split"><div class="grid" style="gap:16px" id="t-main"><div class="card"><div class="empty"><div class="spinner"></div><p>Opening the servo…</p></div></div></div>
      <div class="grid" style="gap:16px" id="t-side"></div></div>`;
    let alive = true, stopEv = null, st = null, logLines = [];
    const draw = () => {
      if (!st) return;
      const lo = st.min_us || 1000, hi = st.max_us || 2000;
      const span = Math.max(100, (st.limits_set ? hi - lo : 600));
      const a = st.limits_set ? lo - span * 0.15 : Math.min(st.rest_us, st.fire_us, st.current_us || 1500) - 150;
      const b = st.limits_set ? hi + span * 0.15 : Math.max(st.rest_us, st.fire_us, st.current_us || 1500) + 150;
      const pos = (v) => clamp((v - a) / (b - a) * 100, 0, 100);
      const cur = st.current_us;
      $("#t-main").innerHTML = `
        <div class="grid g2" style="gap:16px">
          <div class="armed-tile ${st.armed ? "on" : ""}" style="padding:22px">${st.armed ? "BUTTON HELD" : "BUTTON RELEASED"}<small>${st.armed ? "moves are allowed" : "hold the arming button to move the servo"}</small></div>
          <div class="card stat" style="justify-content:center"><span class="label">Last pulse</span><span class="value" style="font-size:34px">${cur ? cur.toFixed(0) : "—"}<small>µs</small></span>
            <span class="hint">${st.driving ? "driving" : "limp"} · each move is a ${st.burst_ms.toFixed(0)} ms burst${st.sim ? " · SIMULATED" : ""}</span></div>
        </div>
        <div class="card"><div class="card-h"><h3>Lever travel</h3><div class="right">${st.limits_set ? `<span class="badge ok">limits ${lo.toFixed(0)}–${hi.toFixed(0)} µs</span>` : `<span class="badge warn">limits not set: moves creep ${st.max_jump_us.toFixed(0)} µs at a time</span>`}</div></div>
          <div class="travel" style="margin-bottom:${st.break_us || st.preload_us ? 16 : 0}px">
            <div class="rail"></div>
            ${st.limits_set ? `<div class="range" style="left:${pos(lo)}%;width:${pos(hi) - pos(lo)}%"></div>` : ""}
            <div class="tick rest" style="left:${pos(st.rest_us)}%"><span>REST ${st.rest_us.toFixed(0)}</span></div>
            <div class="tick fire" style="left:${pos(st.fire_us)}%"><span>FIRE ${st.fire_us.toFixed(0)}</span></div>
            ${st.break_us ? `<div class="tick brk" style="left:${pos(st.break_us)}%"><span>BREAK ${st.break_us.toFixed(0)}</span></div>` : ""}
            ${st.preload_us ? `<div class="tick pre" style="left:${pos(st.preload_us)}%"><span>PRE ${st.preload_us.toFixed(0)}</span></div>` : ""}
            ${cur ? `<div class="cur" style="left:${pos(cur)}%" data-v="${cur.toFixed(0)}"></div>` : ""}
          </div>
          <div class="jog" style="margin-top:22px">${[-50, -10, -5, 5, 10, 50].map((d) => `<button class="btn" data-d="${d}" ${st.armed ? "" : "disabled"}>${d > 0 ? "+" : "−"}${Math.abs(d)}</button>`).join("")}</div>
          <div class="btn-row" style="margin-top:12px"><div class="input-group" style="width:200px"><input class="input mono" id="t-go" type="number" step="5" placeholder="${cur ? cur.toFixed(0) : 1500}"><span class="addon">µs</span></div>
            <button class="btn" id="t-gob" ${st.armed ? "" : "disabled"}>Go to pulse</button>
            ${st.sim ? `<button class="btn ghost" id="t-sim" style="margin-left:auto">${st.armed ? "Release" : "Hold"} simulated button</button>` : ""}</div></div>
        <div class="card"><div class="card-h"><h3>Mark positions</h3><div class="sub">at the last pulse</div></div>
          <div class="btn-row">
            <button class="btn" data-l="lo" ${cur ? "" : "disabled"}>Mark low end</button><button class="btn" data-l="hi" ${cur ? "" : "disabled"}>Mark high end</button>
            <button class="btn" data-l="rest" ${cur ? "" : "disabled"} style="border-color:var(--ok)">Set as REST</button>
            <button class="btn" data-l="fire" ${cur ? "" : "disabled"} style="border-color:var(--danger)">Set as FIRE</button>
            <button class="btn" data-l="break" ${cur ? "" : "disabled"} style="border-color:var(--warn)">Set as BREAK</button></div>
          <p class="small muted" style="margin:10px 0 0"><b>Rest</b>: the lever just off the trigger. <b>Break</b>: exactly where the sear breaks (creep toward it in 5 µs steps). <b>Fire</b>: a little past the break. Low/high ends: as far as the lever may ever go.</p></div>
        <div class="card"><div class="card-h"><h3>Log</h3></div><pre class="term" id="t-log" style="max-height:220px"></pre></div>`;
      $("#t-side").innerHTML = `
        <div class="card"><div class="card-h"><h3>Test & save</h3></div>
          <button class="btn lg" id="t-test" style="width:100%" ${st.armed && st.limits_set ? "" : "disabled"}>${icon("target")}Test one fire cycle</button>
          <p class="small muted">${st.preload_us ? "Pre-travel, then the" : "The"} fire pulse for the hold time, back to rest, then limp — exactly what a shot does. Recock between tests.</p>
          <div class="field"><label>Hold at fire</label><div class="input-group"><input class="input mono" id="t-hold" type="number" min="50" max="1500" step="10" value="${st.hold_ms}"><span class="addon">ms</span></div></div>
          <div class="field"><label>Pre-travel while the button is held</label>
            <div style="display:flex;gap:10px;align-items:center"><input id="t-pre-r" type="range" min="0" max="90" step="5" value="${Math.round(st.preload_frac * 100)}" style="flex:1">
              <div class="input-group" style="width:110px"><input class="input mono" id="t-pre" type="number" min="0" max="90" step="5" value="${Math.round(st.preload_frac * 100)}"><span class="addon">%</span></div></div>
            <p class="small muted" style="margin:6px 0 0">${st.preload_us
              ? `Holding the button parks the lever at <b class="mono">${st.preload_us.toFixed(0)} µs</b>, ${Math.round(st.preload_frac * 100)} % of the way from rest to the break, so a shot only travels the rest of the way. Released = rest, then limp; after ${st.preload_max_s.toFixed(0)} s held it goes limp anyway.`
              : st.preload_frac > 0 ? `<span style="color:var(--warn)">Inactive: mark <b>BREAK</b> (between rest and fire, inside the travel limits) to turn it on.</span>` : "Off: every shot travels the whole way from rest."}</p>
            ${st.preload_us ? `<button class="btn" id="t-pre-go" style="margin-top:8px" ${st.armed ? "" : "disabled"}>Go to pre-travel (${st.preload_us.toFixed(0)} µs)</button>` : ""}</div>
          <hr class="sep" style="margin:14px 0">
          <dl class="kv"><dt>Rest</dt><dd>${st.rest_us.toFixed(0)} µs</dd><dt>Break</dt><dd>${st.break_us ? `${st.break_us.toFixed(0)} µs` : "not marked"}</dd><dt>Fire</dt><dd>${st.fire_us.toFixed(0)} µs</dd><dt>Pre-travel</dt><dd>${st.preload_us ? `${st.preload_us.toFixed(0)} µs (${Math.round(st.preload_frac * 100)} %)` : "off"}</dd><dt>Travel</dt><dd>${st.limits_set ? `${lo.toFixed(0)}–${hi.toFixed(0)} µs` : "not set"}</dd></dl>
          <button class="btn primary" id="t-save" style="width:100%;margin-top:14px" ${st.limits_set ? "" : "disabled"}>${icon("check")}Save to the gun</button>
          <p class="small muted" style="margin:8px 0 0">Writes <span class="mono">config/local.toml</span> (comments kept, old file backed up). The pipeline uses it from its next start.</p></div>
        <div class="card"><div class="card-h"><h3>After a change</h3></div><p class="small muted" style="margin:0">Re-measure the trigger delay (command → break) on the Pi with <span class="mono">tbot latency-mic</span>: the solver leads the clay by it. Changing the pre-travel changes that delay too.</p></div>`;
      $("#t-log").textContent = logLines.slice(-80).join("\n");
      $("#t-log").scrollTop = 1e9;
      $$("[data-d]").forEach((b) => b.onclick = () => sendTool({ line: (+b.dataset.d > 0 ? "+" : "") + b.dataset.d }));
      $$("[data-l]").forEach((b) => b.onclick = () => sendTool({ line: b.dataset.l }));
      $("#t-gob").onclick = () => { const v = parseFloat($("#t-go").value); if (isNum(v)) sendTool({ line: String(v) }); };
      $("#t-go").onkeydown = (e) => { if (e.key === "Enter") $("#t-gob").click(); };
      $("#t-test").onclick = () => sendTool({ line: "test" });
      const setPre = (pct) => { const f = clamp(pct, 0, 90) / 100; if (isNum(f)) sendTool({ line: `preload ${f}` }); };
      $("#t-pre-r").oninput = () => { $("#t-pre").value = $("#t-pre-r").value; };
      $("#t-pre-r").onchange = () => setPre(parseFloat($("#t-pre-r").value));
      $("#t-pre").onchange = () => setPre(parseFloat($("#t-pre").value));
      const preGo = $("#t-pre-go"); if (preGo) preGo.onclick = () => sendTool({ line: st.preload_us.toFixed(0) });
      const sim = $("#t-sim"); if (sim) sim.onclick = () => sendTool({ cmd: "sim_button", held: !st.armed });
      $("#t-save").onclick = async () => {
        try {
          const r = await api("servo/save", { body: { rest_us: st.rest_us, fire_us: st.fire_us, min_us: st.min_us, max_us: st.max_us, hold_ms: parseFloat($("#t-hold").value), break_us: st.break_us, preload_frac: st.preload_frac } });
          toast(r.changed ? "Servo settings saved." : "Nothing changed.", "ok");
        } catch (e) { toast(e.message, "err"); }
      };
    };
    const onEv = (ev) => {
      if (ev.type === "fatal") { toast(ev.message, "err"); $("#t-main").innerHTML = `<div class="card"><div class="empty">${icon("alert")}<h3>Servo not available</h3><p>${esc(ev.message)}</p></div></div>`; }
      if (ev.type === "log") { logLines.push(...ev.text.split("\n")); const l = $("#t-log"); if (l) { l.textContent = logLines.slice(-80).join("\n"); l.scrollTop = 1e9; } }
      if (ev.type === "state") { st = ev; draw(); }
      if (ev.type === "exit" && alive) toast("The servo tool stopped.", "warn");
    };
    (async () => {
      try {
        const h = await ensureTool("servo", {});
        if (!h) { $("#t-main").innerHTML = `<div class="card"><div class="empty">${icon("trigger")}<h3>Pipeline still running</h3><p>Stop it to set up the servo.</p></div></div>`; return; }
      } catch (e) { toast(e.message, "err"); return; }
      pollStatus();
      stopEv = toolEvents(onEv, () => alive);
    })();
    return () => { alive = false; if (stopEv) stopEv(); };
  },
};

// ============================================================================================ SETTINGS
PAGES.settings = {
  title: "Settings",
  render(el) {
    let schema = null, pending = {}, resets = new Set(), q = "", modOnly = false;
    el.innerHTML = `<div class="settings"><nav class="toc" id="st-toc"></nav><div class="grid" style="gap:20px" id="st-body"><div class="card"><div class="spinner"></div></div></div></div><div id="st-bar"></div>`;
    const keyIndex = () => Object.fromEntries(schema.sections.flatMap((s) => s.keys.map((k) => [k.key, k])));
    const ctl = (k) => {
      const v = k.key in pending ? pending[k.key] : resets.has(k.key) ? k.default : k.value;
      const id = `f-${k.key.replace(/\./g, "_")}`;
      if (k.type === "bool") return `<label class="switch"><input type="checkbox" id="${id}" data-k="${k.key}" ${v ? "checked" : ""}><span></span></label>`;
      if (k.choices) return `<select class="input" id="${id}" data-k="${k.key}">${k.choices.map((c) => `<option ${c === v ? "selected" : ""}>${esc(c)}</option>`).join("")}${k.choices.includes(v) ? "" : `<option selected>${esc(v)}</option>`}</select>`;
      if (k.type === "int" || k.type === "float") {
        const inp = `<input class="input mono" id="${id}" data-k="${k.key}" type="number" step="any" value="${esc(v)}">`;
        return k.unit ? `<div class="input-group">${inp}<span class="addon">${esc(k.unit)}</span></div>` : inp;
      }
      if (k.type === "list") return `<input class="input mono" id="${id}" data-k="${k.key}" value="${esc(JSON.stringify(v))}">`;
      return `<input class="input" id="${id}" data-k="${k.key}" value="${esc(v)}">`;
    };
    const row = (k) => {
      const dirty = k.key in pending || resets.has(k.key);
      return `<div class="set-row ${dirty ? "dirty" : ""}" data-row="${k.key}">
        <div><div class="name">${esc(k.name.replace(/_/g, " "))}${k.overridden && !resets.has(k.key) ? '<span class="badge accent">changed</span>' : ""}${k.inference ? '<span class="badge" title="An estimate in default.toml, not yet measured">estimate</span>' : ""}</div>
          <div class="key">${esc(k.key)} · default ${esc(JSON.stringify(k.default))}</div>${k.doc ? `<div class="doc">${esc(k.doc)}</div>` : ""}</div>
        <div class="ctl">${ctl(k)}${(k.overridden && !resets.has(k.key)) ? `<button class="btn sm ghost icon" data-reset="${k.key}" title="Back to default">${icon("undo")}</button>` : ""}</div></div>`;
    };
    const match = (k) => (!q || (k.key + " " + k.doc).toLowerCase().includes(q)) && (!modOnly || k.overridden || k.key in pending);
    const draw = () => {
      const idx = keyIndex();
      const common = schema.common.map((k) => idx[k]).filter(Boolean).filter(match);
      const secs = schema.sections.map((s) => ({ ...s, keys: s.keys.filter(match) })).filter((s) => s.keys.length);
      $("#st-toc").innerHTML = `<div class="field" style="margin-bottom:10px"><input class="input" id="st-q" placeholder="Search settings" value="${esc(q)}"></div>
        <label class="btn-row small" style="padding:4px 10px 10px;gap:8px"><label class="switch"><input type="checkbox" id="st-mod" ${modOnly ? "checked" : ""}><span></span></label>Changed only</label>
        ${common.length ? `<a href="#" data-to="common">Common</a>` : ""}
        ${secs.map((s) => { const n = s.keys.filter((k) => k.overridden).length; return `<a href="#" data-to="${s.name}">${esc(s.title)}${n ? `<span class="badge accent">${n}</span>` : ""}</a>`; }).join("")}
        <a href="#" data-to="raw" style="margin-top:8px">local.toml</a>`;
      $("#st-body").innerHTML = `
        <div class="callout info">${icon("info")}<div>Changes are written to <span class="mono">config/local.toml</span> on the gun (its comments are kept, the old file is backed up). The pipeline reads it when it starts.</div></div>
        ${(() => { const rk = idx["record.mode"]; if (!rk) return ""; const rv = "record.mode" in pending ? pending["record.mode"] : resets.has("record.mode") ? rk.default : rk.value; const on = rv !== "off";
          return `<div class="card" id="sec-rec"><div class="card-h"><div><h2>Press recording</h2><div class="sub">Save video, IMU and telemetry of every held press to the gun's SD card (record.mode).</div></div>
            <div class="right"><label class="switch" title="Press recording on / off"><input type="checkbox" id="rec-sw" ${on ? "checked" : ""}><span></span></label></div></div>
            <p class="small muted" style="margin:0"><b>${on ? "ON" : "OFF"}</b>${rv === "continuous" ? " (continuous: records the whole run)" : ""} — ${on ? "each press is saved when you let go of the arming button." : "nothing is saved; the Sessions page gets no new presses."} Takes effect when the pipeline restarts: save, then press Restart in the message that appears.</p></div>`; })()}
        ${common.length ? `<div class="card set-sec" id="sec-common"><div class="card-h"><div><h2>Common</h2><div class="sub">The settings you are most likely to change.</div></div></div>${common.map(row).join("")}</div>` : ""}
        ${secs.map((s) => `<div class="card set-sec" id="sec-${s.name}"><div class="card-h"><div><h2>${esc(s.title)}</h2><div class="sub">${esc(s.description || s.note)}</div></div><div class="right mono small muted">[${esc(s.name)}]</div></div>${s.keys.map(row).join("")}</div>`).join("")}
        <div class="card set-sec" id="sec-raw"><div class="card-h"><h2>local.toml</h2><div class="sub mono">${esc(schema.local_path)}</div></div><pre class="term">${esc(schema.local_text || "(empty: everything is at its default)")}</pre></div>`;
      $$("[data-to]", $("#st-toc")).forEach((a) => a.onclick = (e) => { e.preventDefault(); const t = $("#sec-" + a.dataset.to); if (t) t.scrollIntoView({ behavior: "smooth" }); });
      $("#st-q").oninput = (e) => { q = e.target.value.toLowerCase(); const pos = e.target.selectionStart; draw(); const n = $("#st-q"); n.focus(); n.setSelectionRange(pos, pos); };
      $("#st-mod").onchange = (e) => { modOnly = e.target.checked; draw(); };
      $$("[data-k]", $("#st-body")).forEach((inp) => inp.addEventListener(inp.type === "checkbox" || inp.tagName === "SELECT" ? "change" : "input", () => {
        const k = idx[inp.dataset.k];
        let v = inp.type === "checkbox" ? inp.checked : inp.value;
        if (k.type === "int" || k.type === "float") v = inp.value === "" ? null : Number(inp.value);
        if (k.type === "list") { try { v = JSON.parse(inp.value); inp.classList.remove("err"); } catch (e) { inp.classList.add("err"); return; } }
        resets.delete(k.key);
        if (v === null || (typeof v === "number" && !isFinite(v))) { inp.classList.add("err"); return; }
        inp.classList.remove("err");
        if (JSON.stringify(v) === JSON.stringify(k.value)) delete pending[k.key]; else pending[k.key] = v;
        $(`[data-row="${k.key}"]`).classList.toggle("dirty", k.key in pending);
        bar();
      }));
      const rsw = $("#rec-sw"); if (rsw) rsw.onchange = () => { const k = idx["record.mode"]; const v = rsw.checked ? (k.default === "off" ? "armed" : k.default) : "off";
        resets.delete(k.key); if (v === k.value) delete pending[k.key]; else pending[k.key] = v; draw(); bar(); };
      $$("[data-reset]", $("#st-body")).forEach((b) => b.onclick = () => { resets.add(b.dataset.reset); delete pending[b.dataset.reset]; draw(); bar(); });
    };
    const bar = () => {
      const n = Object.keys(pending).length + resets.size;
      $("#st-bar").innerHTML = n ? `<div class="savebar"><b>${n} unsaved change${n > 1 ? "s" : ""}</b><span class="muted small">${[...Object.keys(pending), ...resets].slice(0, 4).join(", ")}${n > 4 ? "…" : ""}</span>
        <div style="margin-left:auto" class="btn-row"><button class="btn ghost" id="st-discard">Discard</button><button class="btn primary" id="st-save">${icon("check")}Save to the gun</button></div></div>` : "";
      if (!n) return;
      $("#st-discard").onclick = () => { pending = {}; resets.clear(); draw(); bar(); };
      $("#st-save").onclick = async () => {
        const b = $("#st-save"); b.disabled = true; b.innerHTML = `<span class="spinner"></span>Checking…`;
        try {
          const r = await api("config", { body: { set: pending, reset: [...resets] }, timeout: 90000 });
          pending = {}; resets.clear();
          if (r.restart_needed) {
            const mode = S.status.pipeline.mode;
            toast("Saved. The running pipeline still uses the old values.", "ok", { action: { label: `Restart ${mode === "dry" ? "dry run" : mode}`, run: () => pipelineAction("start", mode) }, ms: 12000 });
          } else toast(r.changed ? "Saved." : "Nothing changed.", "ok");
          await load();
        } catch (e) { toast(e.message, "err"); b.disabled = false; b.innerHTML = `${icon("check")}Save to the gun`; }
      };
    };
    const load = async () => { schema = await api("config"); draw(); bar(); };
    load().catch((e) => toast(e.message, "err"));
    const guard = (e) => { if (Object.keys(pending).length) { e.preventDefault(); e.returnValue = ""; } };
    window.addEventListener("beforeunload", guard);
    return () => window.removeEventListener("beforeunload", guard);
  },
};

// ============================================================================================ DIAGNOSTICS
PAGES.diagnostics = {
  title: "Diagnostics",
  render(el) {
    el.innerHTML = `<div class="grid g3">
        ${[["doctor", "diag", "Configuration check", "Config, calibration files and layout consistency (tbot doctor). Safe any time."],
           ["hwcheck", "cpu", "Hardware check", "Camera fps, IMU, arming pin, PWM, power and temperature. Needs the pipeline stopped. Never moves the trigger."],
           ["latency_report", "timer", "Latency budget", "Every delay between the photon and the shot, measured vs default, and the lead it gives."]]
          .map(([k, ic, t, d]) => `<div class="card cal-card"><div class="top"><div class="ic">${icon(ic)}</div><h3>${t}</h3></div><p>${d}</p><div class="foot"><button class="btn primary" data-job="${k}">${icon("play")}Run</button></div></div>`).join("")}
      </div>
      <div class="card hidden" id="d-job"><div class="card-h"><h2 id="d-title"></h2><div class="right" id="d-state"></div></div><pre class="term" id="d-out"></pre></div>
      <div class="card"><div class="card-h"><h2>Pipeline log</h2><div class="sub" id="d-logf"></div><div class="right"><label class="btn-row small muted" style="gap:6px"><input type="checkbox" id="d-follow" checked> follow</label><button class="btn sm" id="d-reload">${icon("refresh")}Refresh</button></div></div>
        <pre class="term" id="d-log" style="max-height:440px">Loading…</pre></div>`;
    $$("[data-job]", el).forEach((b) => b.onclick = async () => {
      if (b.dataset.job === "hwcheck" && S.status.pipeline.active && !(await confirmBox("Stop the pipeline?", "The hardware check needs the camera.", "Stop & run"))) return;
      try {
        if (b.dataset.job === "hwcheck" && S.status.pipeline.active) await api("pipeline", { body: { action: "stop" }, timeout: 60000 });
        const j = await api("jobs", { body: { kind: b.dataset.job } });
        $("#d-job").classList.remove("hidden"); $("#d-title").textContent = j.title;
        $("#d-state").innerHTML = `<span class="spinner"></span>`;
        $("#d-job").scrollIntoView({ behavior: "smooth" });
        const r = await watchJob(j, (jj) => { $("#d-out").innerHTML = colorizeTerm(jj.output); });
        $("#d-state").innerHTML = `<span class="badge ${r.code === 0 ? "ok" : "warn"}">exit ${r.code}</span>`;
      } catch (e) { toast(e.message, "err"); }
    });
    let timer = null;
    const loadLog = async () => {
      try {
        const r = await api("logs?n=400");
        $("#d-logf").textContent = r.file;
        const box = $("#d-log");
        const atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
        box.innerHTML = r.lines.length ? colorizeTerm(r.lines) : '<span class="dim">(empty)</span>';
        if (atEnd || $("#d-follow").checked) box.scrollTop = 1e9;
      } catch (e) { /* offline */ }
    };
    $("#d-reload").onclick = loadLog;
    loadLog(); timer = setInterval(() => { if ($("#d-follow").checked) loadLog(); }, 3000);
    return () => clearInterval(timer);
  },
};

// ============================================================================================ SYSTEM
PAGES.system = {
  title: "System",
  render(el) {
    el.innerHTML = `<div class="grid g2">
      <div class="card" id="sy-pipe"></div>
      <div class="card" id="sy-boot"></div>
      <div class="card" id="sy-clock"></div>
      <div class="card" id="sy-power"></div>
      <div class="card" id="sy-about"></div>
      <div class="card" id="sy-app"></div></div>`;
    let selMode = null;
    const draw = (st) => {
      const p = st.pipeline;
      if (!selMode) selMode = p.mode;
      $("#sy-pipe").innerHTML = `<div class="card-h"><h2>Pipeline</h2><div class="right">${modeBadge(p)}</div></div>
        <div class="seg" id="sy-mode">${["dry", "test", "live"].map((m) => `<button data-m="${m}" class="${m === selMode ? "on " + m : ""}">${{ dry: "Dry run", test: "Test", live: "Live" }[m]}</button>`).join("")}</div>
        <p class="small muted" style="margin:10px 0 14px">${{ dry: "Decisions are logged and recorded; the servo is never driven.", test: "Dry run that 'fires' on paper at anything moving it tracks (birds, a thrown clay). Score the presses afterwards.", live: "The servo pulls the trigger when the arming button is held and the solver fires." }[selMode]}</p>
        <div class="btn-row"><button class="btn ${selMode === "live" ? "danger" : "primary"}" id="sy-start">${icon("play")}${p.active ? "Restart" : "Start"} in ${selMode === "dry" ? "dry run" : selMode}</button>
          ${p.active ? `<button class="btn" id="sy-stop">${icon("stop")}Stop</button>` : ""}</div>`;
      $$("#sy-mode button").forEach((b) => b.onclick = () => { selMode = b.dataset.m; draw(S.status); });
      $("#sy-start").onclick = () => pipelineAction("start", selMode);
      const sb = $("#sy-stop"); if (sb) sb.onclick = () => pipelineAction("stop");
      $("#sy-boot").innerHTML = `<div class="card-h"><h2>At power-up</h2><div class="right">${p.boot_enabled ? `<span class="mode-pill ${p.boot_mode}">${p.boot_mode.toUpperCase()}</span>` : '<span class="mode-pill off">OFF</span>'}</div></div>
        <div class="btn-row" style="justify-content:space-between"><span>Start the pipeline when the gun powers on</span><label class="switch"><input type="checkbox" id="sy-ben" ${p.boot_enabled ? "checked" : ""}><span></span></label></div>
        <div class="field" style="margin-top:12px"><label>Comes up in</label><div class="seg" id="sy-bmode">${["dry", "test", "live"].map((m) => `<button data-m="${m}" class="${m === p.boot_mode ? "on " + m : ""}">${{ dry: "Dry run", test: "Test", live: "Live (field mode)" }[m]}</button>`).join("")}</div></div>
        ${p.boot_mode === "live" ? `<div class="callout danger" style="margin-top:12px">${icon("alert")}<div><b>Field mode:</b> the gun comes up LIVE after every power cycle, with nobody typing anything. Switch back to dry run when the range day is over.</div></div>` : `<p class="small muted" style="margin:12px 0 0">Field mode (live at power-up) is for a range day without a laptop.</p>`}`;
      $("#sy-ben").onchange = (e) => bootAct(e.target.checked ? "on" : "off");
      $$("#sy-bmode button").forEach((b) => b.onclick = () => bootAct(b.dataset.m));
      const off = st.time - Date.now() / 1000;
      let auto = true; try { auto = localStorage.getItem("tbot.autoclock") !== "0"; } catch (e) {}
      $("#sy-clock").innerHTML = `<div class="card-h"><h2>Clock</h2><div class="right">${Math.abs(off) < 3 ? '<span class="badge ok">in sync</span>' : `<span class="badge warn">${Math.abs(off).toFixed(0)} s ${off < 0 ? "behind" : "ahead"}</span>`}</div></div>
        <p class="small muted" style="margin:0 0 12px">The Pi has no clock battery and no internet at the range, so it can be wrong after a power-up. Session names and dates come from it.</p>
        <dl class="kv"><dt>Gun</dt><dd>${new Date(st.time * 1000).toLocaleString()}</dd><dt>This computer</dt><dd>${new Date().toLocaleString()}</dd></dl>
        <div class="btn-row" style="margin-top:14px;justify-content:space-between"><button class="btn" id="sy-sync">${icon("clock")}Set from this computer</button>
          <label class="btn-row small" style="gap:8px">Sync on connect <label class="switch"><input type="checkbox" id="sy-auto" ${auto ? "checked" : ""}><span></span></label></label></div>`;
      $("#sy-sync").onclick = async () => { try { await api("system", { body: { action: "set_time", epoch_ms: Date.now() } }); toast("Clock set.", "ok"); pollStatus(); } catch (e) { toast(e.message, "err"); } };
      $("#sy-auto").onchange = (e) => { try { localStorage.setItem("tbot.autoclock", e.target.checked ? "1" : "0"); } catch (er) {} };
      $("#sy-power").innerHTML = `<div class="card-h"><h2>Power</h2></div>
        <p class="small muted" style="margin:0 0 14px">Shut down before opening the battery lever: cutting power while the gun writes a recording can damage the SD card. Wait for the green LED to stop blinking.</p>
        <div class="btn-row"><button class="btn" id="sy-reboot">${icon("refresh")}Restart</button><button class="btn outline-danger" id="sy-off">${icon("power")}Shut down</button></div>`;
      $("#sy-reboot").onclick = () => power("reboot");
      $("#sy-off").onclick = () => power("poweroff");
      const h = st.health;
      $("#sy-about").innerHTML = `<div class="card-h"><h2>Device</h2></div><dl class="kv">
        <dt>Hostname</dt><dd>${esc(st.hostname)}</dd><dt>Address</dt><dd class="mono" id="sy-addr">${esc(S.gunAddr || location.host)}</dd><dt>Software</dt><dd>tbot ${esc(st.version)} · API ${st.api}</dd>
        <dt>Uptime</dt><dd>${fmt.dur(h.uptime_s)}</dd><dt>Storage</dt><dd>${fmt.bytes(h.disk_free)} free of ${fmt.bytes(h.disk_total)}</dd>
        <dt>Memory</dt><dd>${fmt.bytes(h.mem_available)} free of ${fmt.bytes(h.mem_total)}</dd><dt>CPU</dt><dd>${fx(h.cpu_temp_c, 1, " °C")}${isNum(h.arm_freq_mhz) ? ` · ${h.arm_freq_mhz.toFixed(0)} MHz` : ""}</dd>
        <dt>Power supply</dt><dd>${(throttleIssue(h) || { text: h.throttled ? "OK" : "—" }).text}</dd>${st.dev ? "<dt>Mode</dt><dd><span class='badge warn'>DEV: laptop stand-in</span></dd>" : ""}</dl>`;
      $("#sy-app").innerHTML = `<div class="card-h"><h2>Appearance</h2></div>
        <div class="btn-row" style="justify-content:space-between"><span>Theme</span><div class="seg" id="sy-theme"><button data-t="dark" class="${document.documentElement.dataset.theme !== "light" ? "on" : ""}">Dark</button><button data-t="light" class="${document.documentElement.dataset.theme === "light" ? "on" : ""}">Light (sunlight)</button></div></div>
        <p class="small muted" style="margin:14px 0 0">TBOT is open source. The console runs on the gun itself (<span class="mono">tbot console</span>) and needs no internet.</p>`;
      $$("#sy-theme button").forEach((b) => b.onclick = () => { setTheme(b.dataset.t); draw(S.status); });
    };
    const bootAct = async (m) => {
      try {
        if (m === "live" && !(await confirmLive("Field mode: LIVE at every power-up"))) { draw(S.status); return; }
        await api("boot", { body: { mode: m, confirm: m === "live" ? "LIVE" : undefined } });
        toast(m === "on" ? "Starts at power-up." : m === "off" ? "Will not start at power-up." : `Power-up mode: ${m}.`, "ok");
      } catch (e) { toast(e.message, "err"); }
      pollStatus();
    };
    const power = async (a) => {
      const ok = await confirmBox(a === "reboot" ? "Restart the gun?" : "Shut down the gun?", a === "reboot" ? "The console reconnects by itself in about a minute." : "Wait about 20 seconds before opening the battery lever. To start it again, cycle the power.", a === "reboot" ? "Restart" : "Shut down", a === "reboot" ? "primary" : "danger");
      if (!ok) return;
      try { await api("system", { body: { action: a } }); toast(a === "reboot" ? "Restarting…" : "Shutting down…", "info"); } catch (e) { toast(e.message, "err"); }
    };
    draw(S.status);
    fetch("/local/info").then((r) => (r.ok ? r.json() : null)).then((i) => {
      if (i && i.host) { S.gunAddr = i.host; const a = $("#sy-addr"); if (a) a.textContent = i.host; }
    }).catch(() => {});
    return S.onStatus(draw);
  },
};

// ============================================================================================ start
setTheme(document.documentElement.dataset.theme || "dark");
renderNav("overview");
pollStatus();
setInterval(pollStatus, 2000);
