"""Self-contained monitor target page for the HITL bench."""

from __future__ import annotations

PAGE_HTML: str = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TBOT HITL target</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#090c10;color:#edf1f5;
 font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
canvas{display:block;width:100vw;height:100vh}
:fullscreen{cursor:none}
#panel{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
 padding:24px;background:rgba(9,12,16,.88);overflow:auto}
#panel[hidden],#panel.compact[hidden]{display:none}
.card{width:min(760px,100%);max-height:100%;overflow:auto;background:#171d25;border:1px solid #303b49;
 border-radius:16px;padding:32px;box-shadow:0 18px 60px #0008}
.eyebrow{color:#9aaabc;text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:700}
h1{font-size:28px;letter-spacing:-.03em;margin:8px 0}
.intro,.hint{color:#aab6c4;line-height:1.5;margin:0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#aab6c4;margin:26px 0 12px}
ol{margin:0;padding-left:24px;color:#e2e8ef;line-height:1.9}
button{border:0;border-radius:9px;background:#dce8f3;color:#101820;font:inherit;font-weight:700;
 padding:14px 28px;min-width:190px;cursor:pointer;margin:24px 0 10px}
button:hover{background:white} button:focus-visible{outline:3px solid #80b8ed;outline-offset:3px}
kbd{font:inherit;color:#edf1f5;background:#29333f;border:1px solid #465363;border-radius:4px;padding:1px 5px}
.health{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.metric{background:#202832;border:1px solid #303b49;border-radius:9px;padding:12px;min-width:0}
.metric-title{font-size:12px;font-weight:700;color:#c0ccd8;display:flex;align-items:center;gap:7px}
.dot{width:8px;height:8px;border-radius:50%;background:#d2a84d;flex:none}
.metric[data-state="good"] .dot{background:#63c997}
.metric[data-state="bad"] .dot{background:#e17b7b}
.metric p{font-size:12px;color:#aab6c4;line-height:1.5;margin:7px 0 0;overflow-wrap:anywhere}
#completion{color:#8ddbb1;font-weight:700;margin:16px 0 0}
#completion[hidden]{display:none}
#panel.compact{inset:12px auto auto 12px;display:block;padding:0;background:none;overflow:visible;
 pointer-events:none}
#panel.compact .card{width:auto;max-width:380px;padding:10px 14px;border-radius:8px;background:#171d25e8;
 box-shadow:none}
#panel.compact .eyebrow,#panel.compact .intro,#panel.compact .setup,#panel.compact .actions,
#panel.compact h2,#panel.compact .health{display:none}
#panel.compact h1{font-size:13px;margin:0}
#panel.compact #run-message{font-size:12px;color:#c0ccd8;margin:4px 0 0}
#panel.compact #completion{font-size:12px;margin:4px 0 0}
@media(max-width:600px){.card{padding:22px}.health{grid-template-columns:1fr} h1{font-size:23px}}
</style></head><body>
<canvas id="canvas"></canvas>
<div id="panel"><main class="card">
  <div class="eyebrow">Hardware-in-the-loop bench</div>
  <h1>TBOT · HITL target</h1>
  <p class="intro">This screen presents timed targets to the rig's camera.</p>
  <p id="run-message" hidden></p>
  <p id="completion" hidden>Run finished. You can return to the rig for the results.</p>
  <section class="setup"><h2>Before you start</h2><ol>
    <li>Set the monitor to full brightness.</li>
    <li>Place the camera 1–2 m from the screen.</li>
    <li>Point the bore at the screen.</li>
    <li>Keep this tab in front throughout the run.</li>
  </ol></section>
  <div class="actions"><button id="start" type="button">Start fullscreen</button>
    <p class="hint"><kbd>F</kbd> fullscreen &nbsp;·&nbsp; <kbd>H</kbd> toggle overlay</p></div>
  <h2>Live health</h2>
  <div class="health" aria-live="polite">
    <div class="metric" id="sync-health"><div class="metric-title"><span class="dot"></span>Clock sync</div><p id="sync-info">Waiting for sync</p></div>
    <div class="metric" id="display-health"><div class="metric-title"><span class="dot"></span>Display</div><p id="display-info">Measuring refresh</p></div>
    <div class="metric" id="rig-health"><div class="metric-title"><span class="dot"></span>Rig</div><p id="rig-info">Waiting for rig</p></div>
  </div>
</main></div>
<script>
'use strict';
const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
const panel = document.getElementById('panel');
let program = {v:0,bg:0,status:false,message:'',items:[]};
let showOverlay = true, wakeLock = null, samples = [], model = null;
let frames = [], frameIndex = 0, posting = false, syncing = false;
let lastFrame = null, intervals = [], lateFrames = 0, maxInterval = null;
let rigState = null, lastStatePoll = null, statePolling = false;
const median = a => { if (!a.length) return null; const s = [...a].sort((x,y)=>x-y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m-1]+s[m])/2; };
const fmt = (x, n=2) => x == null || !Number.isFinite(x) ? '—' : x.toFixed(n);
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(innerWidth*dpr)), h = Math.max(1, Math.round(innerHeight*dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
}
window.addEventListener('resize', resize);
function fitClock() {
  if (!samples.length) { model = null; return; }
  const sorted = [...samples].sort((a,b)=>a.rtt-b.rtt);
  const chosen = samples.length < 6 ? sorted.slice(0,1) : sorted.slice(0,Math.max(2,Math.ceil(sorted.length*0.3)));
  const ref = chosen.reduce((s,x)=>s+x.local,0)/chosen.length;
  const a = chosen.reduce((s,x)=>s+x.offset,0)/chosen.length;
  let xx=0, xy=0;
  for (const x of chosen) { const d=x.local-ref; xx+=d*d; xy+=d*(x.offset-a); }
  model = {ref, a, b:xx ? xy/xx : 0};
}
function toRig(local) { return local + (model ? model.a + model.b*(local-model.ref) : 0); }
async function sync() {
  if (syncing) return;
  syncing = true;
  try {
    const t0 = performance.now();
    const response = await fetch('/hitl/sync', {cache:'no-store', keepalive:true});
    if (!response.ok) throw new Error('sync failed');
    const data = await response.json(), t1 = performance.now();
    const local = (t0+t1)/2;
    samples.push({local, offset:data.t-local, rtt:t1-t0});
    samples = samples.filter(x=>x.local >= t1-15000);
    fitClock();
    for (const frame of frames) {
      if (frame.unsynced) { frame.t=toRig(frame.t); delete frame.unsynced; }
    }
  } catch (_) {} finally { syncing = false; }
}
let burst = 0;
const burstTimer = setInterval(() => { sync(); if (++burst >= 20) { clearInterval(burstTimer); setInterval(sync,200); } },30);
sync();
let polling = false;
async function pollScene() {
  if (polling) return;
  polling = true;
  try { const r=await fetch('/hitl/scene',{cache:'no-store'}); if (r.ok) { program=await r.json(); updatePanel(); } } catch (_) {} finally { polling=false; }
}
setInterval(pollScene,100); pollScene();
async function pollState() {
  if (statePolling) return;
  statePolling=true;
  try {
    const r=await fetch('/hitl/state',{cache:'no-store'});
    if (r.ok) { rigState=await r.json(); lastStatePoll=performance.now(); }
  } catch (_) {} finally { statePolling=false; updatePanel(); }
}
setInterval(pollState,1000); pollState();
function status() {
  const now=performance.now(), live=samples.filter(x=>x.local >= now-15000);
  const rtts=live.map(x=>x.rtt), p50=median(intervals);
  return {w:canvas.width,h:canvas.height,dpr:window.devicePixelRatio || 1,
    fullscreen:!!document.fullscreenElement,visible:document.visibilityState==='visible',
    synced:!!model && live.length > 0,n_sync:live.length,rtt_min_ms:rtts.length ? Math.min(...rtts) : null,
    rtt_p50_ms:median(rtts),offset_ms:model ? model.a+model.b*(now-model.ref) : null,
    drift_ppm:model ? model.b*1e6 : null,refresh_hz:p50 ? 1000/p50 : null,
    frame_ms_p50:p50,frame_ms_max:maxInterval,
    late_frames:lateFrames,ua:navigator.userAgent,page_ms:now};
}
function updatePanel() {
  const s=status(), fullscreen=!!document.fullscreenElement;
  const finished=program.status && /\bfinished\b/i.test(program.message || '');
  panel.classList.toggle('compact',fullscreen);
  panel.hidden=fullscreen && (!program.status || (!showOverlay && !finished));
  const message=document.getElementById('run-message');
  message.hidden=!fullscreen || finished;
  message.textContent=program.message || 'Waiting for the rig';
  document.getElementById('completion').hidden=!finished;
  document.getElementById('sync-health').dataset.state=s.synced ? 'good' : 'bad';
  document.getElementById('sync-info').textContent=
    `${s.synced ? 'Synced' : 'Not synced'} · RTT min ${fmt(s.rtt_min_ms)} ms · drift ${fmt(s.drift_ppm)} ppm`;
  document.getElementById('display-health').dataset.state=s.refresh_hz == null ? 'waiting' : (s.late_frames ? 'waiting' : 'good');
  document.getElementById('display-info').textContent=
    `Refresh ${fmt(s.refresh_hz,1)} Hz · late frames ${s.late_frames}`;
  const age=lastStatePoll == null ? null : (performance.now()-lastStatePoll)/1000;
  const connected=age != null && age < 3;
  document.getElementById('rig-health').dataset.state=connected ? 'good' : 'bad';
  document.getElementById('rig-info').textContent=
    `${connected ? 'Connected' : 'Disconnected'} · last poll ${age == null ? '—' : fmt(age,1)+' s ago'} · test ${rigState?.test || '—'} · frames logged ${rigState?.frames_logged ?? '—'}`;
}
setInterval(updatePanel,500);
async function postLog() {
  if (posting || !model) return;
  posting=true;
  const batch=frames.slice(), last=batch.length ? batch[batch.length-1].i : -1;
  try {
    const r=await fetch('/hitl/log',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({frames:batch,status:status()}),cache:'no-store'});
    if (r.ok && last >= 0) frames=frames.filter(f=>f.i>last);
  } catch (_) {} finally { posting=false; }
}
setInterval(postLog,250);
function gray(level) { return `rgb(${level},${level},${level})`; }
function draw(timestamp) {
  requestAnimationFrame(draw);
  const ids=[];
  const t=toRig(timestamp), index=frameIndex++;
  try {
    resize();
    if (lastFrame !== null) {
      const dt=timestamp-lastFrame, med=median(intervals);
      if (med !== null && dt>1.5*med) lateFrames++;
      if (dt>0) { maxInterval=Math.max(maxInterval || 0,dt); intervals.push(dt); if (intervals.length>600) intervals.shift(); }
    }
    lastFrame=timestamp;
    ctx.fillStyle=gray(program.bg); ctx.fillRect(0,0,canvas.width,canvas.height);
    for (const item of program.items) {
      if (t<item.t0 || (item.t1 !== null && t>=item.t1)) continue;
      ctx.fillStyle=gray(item.level);
      if (item.kind==='disc') {
        const dt=t-item.t0, x=item.x0+item.vx*dt+0.5*item.ax*dt*dt, y=item.y0+item.vy*dt+0.5*item.ay*dt*dt;
        ctx.beginPath(); ctx.arc(x,y,item.r,0,2*Math.PI); ctx.fill();
      } else if (item.kind==='rect') ctx.fillRect(item.x,item.y,item.w,item.h);
      else continue;
      ids.push(item.id);
    }
  } catch (_) {} finally {
    frames.push(model ? {i:index,t,ids} : {i:index,t,ids,unsynced:true});
    if (frames.length>5000) frames.splice(0,frames.length-5000);
  }
}
requestAnimationFrame(draw);
async function start() {
  try { if (!document.fullscreenElement) await document.documentElement.requestFullscreen(); } catch (_) {}
  try { if ('wakeLock' in navigator) wakeLock=await navigator.wakeLock.request('screen'); } catch (_) {}
}
document.getElementById('start').addEventListener('click',start);
document.addEventListener('fullscreenchange',()=>{ updatePanel(); resize(); });
document.addEventListener('keydown',async e=>{
  if (e.key.toLowerCase()==='h') { showOverlay=!showOverlay; updatePanel(); }
  if (e.key.toLowerCase()==='f') {
    if (document.fullscreenElement) { try { await document.exitFullscreen(); } catch (_) {} }
    else await start();
  }
});
resize();
</script></body></html>'''
