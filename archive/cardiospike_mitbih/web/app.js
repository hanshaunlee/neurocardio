/* CardioSpike front-end.
   All numbers / rasters / probability traces come from web/data/*.json
   produced by src/evaluate.py running on the trained snntorch model. */

const CLASS_COLORS = { N: "#6df0a8", S: "#5ad7ff", V: "#ff5b6e", F: "#ffb454", Q: "#c9a4ff" };
const T_INPUT = 260;     // matches preprocess.WIN_LEN
const T_L3 = 32;         // matches model.py
const T_DENSE = 12;

let SUMMARY = null, STRIP = null, INF = null;
let activeClass = "N";

function fmt(x, n=3)  { return (typeof x === "number") ? x.toFixed(n) : "—"; }
function pct(x, n=1) { return (typeof x === "number") ? (x * 100).toFixed(n) + "%" : "—"; }

/* ────── data loading ─────────────────────────────────────────────── */
async function loadAll() {
  const [s, st, inf] = await Promise.all([
    fetch("data/summary.json").then(r => r.json()),
    fetch("data/strip.json").then(r => r.json()),
    fetch("data/inference.json").then(r => r.json()),
  ]);
  SUMMARY = s; STRIP = st; INF = inf;
}

function paintTopStats() {
  document.getElementById("statAcc").textContent = pct(SUMMARY.accuracy, 2);
  document.getElementById("statF1").textContent  = fmt(SUMMARY.macro_f1, 3);
  document.getElementById("statEnergy").textContent = fmt(SUMMARY.energy_ratio_ann_over_snn, 2) + "×";
  document.getElementById("statParams").textContent = SUMMARY.params.toLocaleString();
  document.getElementById("statBeats").textContent = SUMMARY.n_test_beats.toLocaleString();
}

/* ────── canvas helpers ───────────────────────────────────────────── */
function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || canvas.parentElement.clientWidth;
  const cssH = parseInt(canvas.getAttribute("height"), 10);
  canvas.style.height = cssH + "px";
  canvas.width  = Math.floor(cssW * ratio);
  canvas.height = Math.floor(cssH * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  return { ctx, w: cssW, h: cssH };
}

function clearLane(ctx, w, h) {
  ctx.fillStyle = "rgba(0,0,0,0.0)";
  ctx.clearRect(0, 0, w, h);
}

function drawGrid(ctx, w, h, step=20) {
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < w; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
  for (let y = 0; y < h; y += step) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
  ctx.stroke();
}

/* ────── per-class anatomy renderer ──────────────────────────────── */
function paintClassTabs() {
  const wrap = document.getElementById("classTabs");
  wrap.innerHTML = "";
  INF.classes.forEach(c => {
    const btn = document.createElement("button");
    btn.className = (c === activeClass) ? "active" : "";
    const sup = SUMMARY.per_class_support[c] ?? 0;
    const f1 = SUMMARY.per_class_f1[c] ?? 0;
    btn.innerHTML = `<span style="color:${CLASS_COLORS[c]}">●</span>
                     <span>${c} · ${INF.class_names[c]}</span>
                     <span class="chip">n=${sup.toLocaleString()} · F1=${f1.toFixed(2)}</span>`;
    btn.onclick = () => { activeClass = c; paintClassTabs(); paintAnatomy(); };
    wrap.appendChild(btn);
  });
}

function pickExample(cls) {
  const list = INF.examples.filter(e => e.true === cls);
  if (list.length === 0) return null;
  return list[0];
}

function drawEcg(canvas, ecg, color = "#ffffff") {
  const { ctx, w, h } = setupCanvas(canvas);
  drawGrid(ctx, w, h, 20);
  const lo = Math.min(...ecg), hi = Math.max(...ecg);
  const pad = (hi - lo) * 0.1 + 0.001;
  const ymin = lo - pad, ymax = hi + pad;
  ctx.strokeStyle = color; ctx.lineWidth = 1.6;
  ctx.beginPath();
  ecg.forEach((v, i) => {
    const x = i / (ecg.length - 1) * w;
    const y = h - (v - ymin) / (ymax - ymin) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawInputSpikes(canvas, on, off, T = T_INPUT) {
  const { ctx, w, h } = setupCanvas(canvas);
  drawGrid(ctx, w, h, 20);
  const cy_on = h * 0.35, cy_off = h * 0.7;
  ctx.fillStyle = "#ff5f7a";
  on.forEach(t => {
    const x = t / (T - 1) * w;
    ctx.beginPath(); ctx.arc(x, cy_on, 2.4, 0, Math.PI * 2); ctx.fill();
    ctx.fillRect(x - 0.6, cy_on - 12, 1.2, 24);
  });
  ctx.fillStyle = "#5ad7ff";
  off.forEach(t => {
    const x = t / (T - 1) * w;
    ctx.beginPath(); ctx.arc(x, cy_off, 2.4, 0, Math.PI * 2); ctx.fill();
    ctx.fillRect(x - 0.6, cy_off - 12, 1.2, 24);
  });
}

function drawRaster(canvas, events, nChannels, T) {
  const { ctx, w, h } = setupCanvas(canvas);
  drawGrid(ctx, w, h, 20);
  const cellW = w / T;
  const cellH = h / nChannels;
  ctx.fillStyle = "#ffd166";
  events.forEach(([ch, t]) => {
    const x = t * cellW;
    const y = ch * cellH;
    ctx.fillRect(x, y, Math.max(cellW - 0.5, 1.2), Math.max(cellH - 0.5, 1.2));
  });
}

function drawVout(canvas, vout) {
  // vout: (5, Td)
  const { ctx, w, h } = setupCanvas(canvas);
  drawGrid(ctx, w, h, 20);
  const Td = vout[0].length;
  const flat = vout.flat();
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const pad = (hi - lo) * 0.1 + 0.001;
  const ymin = lo - pad, ymax = hi + pad;
  ctx.lineWidth = 2;
  INF.classes.forEach((c, i) => {
    ctx.strokeStyle = CLASS_COLORS[c];
    ctx.beginPath();
    vout[i].forEach((v, t) => {
      const x = t / (Td - 1) * w;
      const y = h - (v - ymin) / (ymax - ymin) * h;
      if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // label at end
    const lastY = h - (vout[i][Td-1] - ymin) / (ymax - ymin) * h;
    ctx.fillStyle = CLASS_COLORS[c];
    ctx.fillText(c, w - 12, lastY + 4);
  });
}

function paintAnatomy() {
  const ex = pickExample(activeClass);
  if (!ex) return;
  drawEcg(document.getElementById("anEcg"), ex.ecg, CLASS_COLORS[ex.true]);
  drawInputSpikes(document.getElementById("anIn"),
                  ex.input_spikes.on, ex.input_spikes.off, ex.ecg.length);
  // For raster_l3, events are [channel, t-step]. Channels=64, T=32.
  drawRaster(document.getElementById("anL3"), ex.raster_l3, 64, T_L3);
  drawVout(document.getElementById("anVout"), ex.vout);

  const correct = ex.true === ex.pred;
  const probs = INF.classes.map(c => [c, ex.probs[c]]);
  probs.sort((a, b) => b[1] - a[1]);
  const top = probs[0];
  const second = probs[1];
  const el = document.getElementById("anatDecision");
  el.innerHTML = `
    <span class="badge ${correct ? 'ok' : 'bad'}">${correct ? '✓ correct' : '✗ wrong'}</span>
    <span class="badge dim">record ${ex.record_id}</span>
    <span>True <strong style="color:${CLASS_COLORS[ex.true]}">${ex.true}</strong> ·
         Predicted <strong style="color:${CLASS_COLORS[ex.pred]}">${ex.pred}</strong>
         (p=${(ex.probs[ex.pred]*100).toFixed(1)}%) ·
         runner-up <strong style="color:${CLASS_COLORS[second[0]]}">${second[0]}</strong>
         (p=${(second[1]*100).toFixed(1)}%)</span>
  `;
}

/* ────── live strip player ───────────────────────────────────────── */
const live = {
  playing: true,
  beatIdx: 0,
  tIdx: 0,
  speed: 1.2,
  ecgBuf: [],            // ring buffer of recent samples (value, drawn-at-x)
  inputEvents: [],       // [{x, polarity}]
  l3Events: [],          // [{ch, x, age}]
  l4Events: [],
  probs: { N:0, S:0, V:0, F:0, Q:0 },
  predLabel: "N",
  lastBeatLabel: "N",
};

function tickLive(dt) {
  if (!live.playing || !STRIP) return;
  const beats = STRIP.beats;
  if (beats.length === 0) return;

  // Each beat is 260 samples; we advance live.tIdx by speed*dt*samplesPerSecond
  const samplesPerSecond = 360 * live.speed; // ECG native fs
  live.tIdx += dt * samplesPerSecond;

  while (live.tIdx >= T_INPUT) {
    live.tIdx -= T_INPUT;
    live.beatIdx = (live.beatIdx + 1) % beats.length;
    // commit prediction of the just-finished beat
    const finishedIdx = (live.beatIdx - 1 + beats.length) % beats.length;
    const b = beats[finishedIdx];
    live.probs = b.probs;
    live.predLabel = b.pred;
    live.lastBeatLabel = b.true;
  }

  // We don't need to maintain ECG / event buffers manually — we redraw the
  // strip in place each frame from the current beat + a small window of the
  // previous beat for context.
}

function drawLive() {
  if (!STRIP) return;
  const beats = STRIP.beats;
  const ecgCanvas = document.getElementById("ecgCanvas");
  const inSpkCanvas = document.getElementById("inputSpikeCanvas");
  const l3Canvas = document.getElementById("l3Canvas");
  const l4Canvas = document.getElementById("l4Canvas");

  // We render a 2.5-beat window: prev beat (faded) + current beat (full)
  const wBeats = 2.5;
  const total = Math.round(T_INPUT * wBeats);

  // Build a synthetic timeline by stitching beats around current position
  const cur = live.beatIdx;
  const prev = (cur - 1 + beats.length) % beats.length;
  const next = (cur + 1) % beats.length;
  const stitched = [].concat(
    beats[prev].ecg.slice(Math.floor(T_INPUT * 0.5)),  // last half of prev
    beats[cur].ecg,
    beats[next].ecg.slice(0, Math.floor(T_INPUT * 0.5))
  );
  // current cursor position (samples since start of stitched window)
  const cursorSample = Math.floor(T_INPUT * 0.5) + Math.floor(live.tIdx);

  // --- ECG lane ---
  {
    const { ctx, w, h } = setupCanvas(ecgCanvas);
    drawGrid(ctx, w, h, 20);
    const lo = Math.min(...stitched), hi = Math.max(...stitched);
    const pad = (hi - lo) * 0.1 + 0.001;
    const ymin = lo - pad, ymax = hi + pad;
    ctx.lineWidth = 1.8;
    // draw past portion in bright color, future in dim
    ctx.strokeStyle = "#5ad7ff";
    ctx.beginPath();
    for (let i = 0; i <= cursorSample; i++) {
      const x = i / (stitched.length - 1) * w;
      const y = h - (stitched[i] - ymin) / (ymax - ymin) * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.strokeStyle = "rgba(90,215,255,0.18)";
    ctx.beginPath();
    for (let i = cursorSample; i < stitched.length; i++) {
      const x = i / (stitched.length - 1) * w;
      const y = h - (stitched[i] - ymin) / (ymax - ymin) * h;
      if (i === cursorSample) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // cursor line
    const cx = cursorSample / (stitched.length - 1) * w;
    ctx.strokeStyle = "rgba(255,77,109,0.9)";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
    // beat boundary markers
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    [Math.floor(T_INPUT * 0.5), Math.floor(T_INPUT * 1.5)].forEach(bx => {
      const x = bx / (stitched.length - 1) * w;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    });
  }

  // --- input spike lane: stitch ON/OFF events from 3 beats and shift ---
  {
    const { ctx, w, h } = setupCanvas(inSpkCanvas);
    drawGrid(ctx, w, h, 20);
    const off0 = 0;
    const off1 = Math.floor(T_INPUT * 0.5);
    const off2 = Math.floor(T_INPUT * 1.5);
    const drawEvents = (arr, baseOff, fromIdx, color, alphaFn) => {
      ctx.fillStyle = color;
      arr.forEach(t => {
        // include only events that have happened (t + baseOff <= cursorSample)
        const stitchedT = baseOff + t - fromIdx;
        if (stitchedT < 0 || stitchedT > stitched.length - 1) return;
        const isPast = stitchedT <= cursorSample;
        const x = stitchedT / (stitched.length - 1) * w;
        const y = color === "#ff5f7a" ? h * 0.35 : h * 0.70;
        ctx.globalAlpha = isPast ? 1 : 0.25;
        ctx.fillRect(x - 0.7, y - 12, 1.4, 24);
        ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI * 2); ctx.fill();
      });
      ctx.globalAlpha = 1;
    };
    drawEvents(beats[prev].input_spikes.on,  off0, Math.floor(T_INPUT * 0.5), "#ff5f7a");
    drawEvents(beats[prev].input_spikes.off, off0, Math.floor(T_INPUT * 0.5), "#5ad7ff");
    drawEvents(beats[cur ].input_spikes.on,  off1, 0,                          "#ff5f7a");
    drawEvents(beats[cur ].input_spikes.off, off1, 0,                          "#5ad7ff");
    drawEvents(beats[next].input_spikes.on,  off2, 0,                          "#ff5f7a");
    drawEvents(beats[next].input_spikes.off, off2, 0,                          "#5ad7ff");
    // cursor
    const cx = cursorSample / (stitched.length - 1) * w;
    ctx.strokeStyle = "rgba(255,77,109,0.7)";
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
  }

  // --- layer-3 raster (64 channels × 32 t-steps), stitched across beats ---
  drawStitchedRaster(l3Canvas, beats, prev, cur, next, "raster_l3", 64, T_L3, cursorSample, stitched.length);
  // --- layer-4 raster (128 dense neurons × T_dense), stitched across beats ---
  drawStitchedRaster(l4Canvas, beats, prev, cur, next, "raster_l4", 128, T_DENSE, cursorSample, stitched.length);

  // --- prediction bars ---
  updatePredBars(live.probs, live.predLabel);
}

function drawStitchedRaster(canvas, beats, prev, cur, next, key, nCh, T_layer, cursorSample, totalSamples) {
  const { ctx, w, h } = setupCanvas(canvas);
  drawGrid(ctx, w, h, 20);
  // The raster's "time axis" is the layer's own T (compressed). We stretch
  // it to span the same ECG sample range as that beat.
  const drawBeat = (b, beatStartSample, beatEndSample, alpha) => {
    const events = b[key];
    if (!events) return;
    ctx.globalAlpha = alpha;
    events.forEach(([ch, t]) => {
      const sampleX = beatStartSample + (t + 0.5) / T_layer * (beatEndSample - beatStartSample);
      const x = sampleX / (totalSamples - 1) * w;
      const y = ch / nCh * h;
      const cellH = Math.max(h / nCh - 0.4, 1.1);
      ctx.fillStyle = sampleX <= cursorSample ? "#ffd166" : "rgba(255,209,102,0.25)";
      ctx.fillRect(x - 1.4, y, 2.8, cellH);
    });
    ctx.globalAlpha = 1;
  };
  const half = Math.floor(T_INPUT * 0.5);
  drawBeat(beats[prev], 0,              half,            0.55);
  drawBeat(beats[cur ], half,           half + T_INPUT,  1.0);
  drawBeat(beats[next], half + T_INPUT, totalSamples,    0.55);
  // cursor
  const cx = cursorSample / (totalSamples - 1) * w;
  ctx.strokeStyle = "rgba(255,77,109,0.85)";
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
}

function updatePredBars(probs, winner) {
  const el = document.getElementById("predBars");
  if (!INF) return;
  const truth = live.lastBeatLabel;
  const correct = truth === winner;
  const header = `<div class="pred-header">
    <span class="pred-tag" style="color:${CLASS_COLORS[truth] || '#888'}">truth: ${truth}</span>
    <span class="pred-tag" style="color:${CLASS_COLORS[winner] || '#888'}">model: ${winner}</span>
    <span class="pred-status ${correct ? 'ok' : 'bad'}">${correct ? '✓' : '✗'}</span>
  </div>`;
  const html = INF.classes.map(c => {
    const p = probs[c] || 0;
    const cls = (c === winner) ? "pred-row winner" : "pred-row";
    return `<div class="${cls}">
      <div class="pred-name" style="color:${CLASS_COLORS[c]}">${c}</div>
      <div class="pred-track"><div class="pred-fill" style="width:${(p*100).toFixed(1)}%"></div></div>
      <div class="pred-pct">${(p*100).toFixed(1)}%</div>
    </div>`;
  }).join("");
  el.innerHTML = header + html;
}

/* ────── confusion matrix ────────────────────────────────────────── */
function paintConfusion() {
  const cm = SUMMARY.confusion_matrix;
  const classes = INF.classes;
  const wrap = document.getElementById("cmatrix");
  const rowsTotals = cm.map(r => r.reduce((a,b) => a+b, 0));
  wrap.innerHTML = "";
  // header
  wrap.innerHTML += `<div class="cm-cell cm-hdr"></div>`;
  classes.forEach(c => {
    wrap.innerHTML += `<div class="cm-cell cm-hdr" style="color:${CLASS_COLORS[c]}">pred ${c}</div>`;
  });
  wrap.innerHTML += `<div class="cm-cell cm-hdr">recall</div>`;
  // rows
  cm.forEach((row, i) => {
    const total = rowsTotals[i];
    wrap.innerHTML += `<div class="cm-cell cm-hdr" style="color:${CLASS_COLORS[classes[i]]}">true ${classes[i]}</div>`;
    row.forEach((v, j) => {
      const isDiag = i === j;
      const cls = "cm-cell " + (isDiag ? "cm-diag" : (v === 0 ? "cm-off cm-zero" : "cm-off"));
      const pctV = total ? (v / total * 100).toFixed(1) + "%" : "—";
      wrap.innerHTML += `<div class="${cls}">${v.toLocaleString()}<div class="cm-pct">${pctV}</div></div>`;
    });
    const recall = total ? cm[i][i] / total : 0;
    wrap.innerHTML += `<div class="cm-cell cm-recall">${(recall*100).toFixed(1)}%</div>`;
  });
}

/* ────── energy chart ────────────────────────────────────────────── */
async function paintEnergy() {
  const sp = await fetch("data/sparsity.json").then(r => r.json()).catch(() => null);
  // Try web-relative first; fall back to summary-derived approximate
  const wrap = document.getElementById("energyChart");
  let layers, ratio;
  if (sp) {
    layers = sp.per_layer;
    ratio = sp.energy_ratio_ann_over_snn;
  } else {
    document.getElementById("energyTotal").innerHTML =
      `SNN cost is <strong>${SUMMARY.energy_ratio_ann_over_snn.toFixed(2)}×</strong> cheaper per beat than the dense baseline (mean hidden spike rate ${(SUMMARY.mean_hidden_spike_rate*100).toFixed(1)}%).`;
    return;
  }
  const maxAnn = Math.max(...Object.values(layers).map(l => l.dense_ops));
  let html = `<div class="energy-row">
    <div class="energy-name"></div>
    <div class="energy-name" style="text-align:center">dense baseline (ANN)</div>
    <div class="energy-name" style="text-align:center">spiking, measured</div>
    <div class="energy-name" style="text-align:right">SNN / ANN</div></div>`;
  Object.entries(layers).forEach(([name, l]) => {
    const annW = (l.dense_ops / maxAnn) * 100;
    const snnW = (l.snn_ops  / maxAnn) * 100;
    const r = (l.snn_ops / Math.max(l.dense_ops, 1)) * 100;
    html += `<div class="energy-row">
      <div class="energy-name">${name}</div>
      <div class="energy-bar ann"><div class="fill" style="width:${annW}%"></div></div>
      <div class="energy-bar snn"><div class="fill" style="width:${snnW}%"></div></div>
      <div class="energy-pct">${r.toFixed(1)}%</div>
    </div>`;
  });
  wrap.innerHTML = html;
  document.getElementById("energyTotal").innerHTML =
    `Per-beat SNN cost is <strong>${ratio.toFixed(2)}×</strong> below the dense baseline. ` +
    `This is the speedup an event-driven neuromorphic chip (Loihi, SpiNNaker) would see — ` +
    `silicon counts MAC operations on actual spikes, not on zeros.`;
}

/* ────── main ────────────────────────────────────────────────────── */
async function main() {
  await loadAll();
  paintTopStats();
  paintClassTabs();
  paintAnatomy();
  paintConfusion();
  paintEnergy();

  document.getElementById("playBtn").addEventListener("click", e => {
    live.playing = !live.playing;
    e.target.textContent = live.playing ? "⏸ Pause" : "▶ Play";
  });
  document.getElementById("speedSlider").addEventListener("input", e => {
    live.speed = parseFloat(e.target.value);
  });

  let lastTs = performance.now();
  function frame(ts) {
    const dt = Math.min(0.1, (ts - lastTs) / 1000);
    lastTs = ts;
    tickLive(dt);
    drawLive();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  window.addEventListener("resize", () => { paintAnatomy(); });
}

main().catch(err => {
  document.body.innerHTML = `<pre style="color:#ff5b6e;padding:24px">${err.stack || err}</pre>`;
});
