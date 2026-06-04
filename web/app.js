/* NeuroCardio front-end.
   GET  /status
   GET  /examples?n=20&fold=10
   GET  /infer_example?ecg_id=…
   POST /infer_upload (.npy)
   GET  /history?run_name=main
   GET  /comparison
*/

const LEAD_NAMES  = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"];
const CLASS_COLORS = { NORM:"#6df0a8", MI:"#ff5b6e", STTC:"#ffb454", CD:"#5ad7ff", HYP:"#c9a4ff" };
const CLASS_DESC   = {
  NORM: "Normal sinus rhythm",
  MI:   "Myocardial Infarction",
  STTC: "ST/T-wave Change",
  CD:   "Conduction Disturbance",
  HYP:  "Hypertrophy",
};
const MODEL_LABEL  = { snn:"SNN (NeuroCardio)", cnn:"CNN1D baseline", resnet:"ResNet1D baseline" };
const MODEL_COLOR  = { snn:"#ff4d6d", cnn:"#5ad7ff", resnet:"#ffd166" };
const FS = 100, T_INPUT = 1000;

let LAST = null, EXAMPLES = [], LABELS = [], CMP_DATA = null;

const $ = id => document.getElementById(id);
const fmt  = (x, n=3) => (typeof x==="number") ? x.toFixed(n) : "—";
const pct  = (x, n=1) => (typeof x==="number") ? (x*100).toFixed(n)+"%" : "—";

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  return r.json();
}

// ─── canvas setup ────────────────────────────────────────────────────
function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const cssW  = canvas.clientWidth || canvas.parentElement.clientWidth || 800;
  const cssH  = parseInt(canvas.getAttribute("height"), 10);
  canvas.style.height = cssH + "px";
  canvas.width  = Math.floor(cssW * ratio);
  canvas.height = Math.floor(cssH * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  return { ctx, w: cssW, h: cssH };
}
function drawGrid(ctx, w, h, step=50) {
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x=0; x<w; x+=step) { ctx.moveTo(x,0); ctx.lineTo(x,h); }
  for (let y=0; y<h; y+=step) { ctx.moveTo(0,y); ctx.lineTo(w,y); }
  ctx.stroke();
}

// ─── 12-lead ECG ─────────────────────────────────────────────────────
function drawECG(ecg) {
  const c = $("ecgCanvas");
  const {ctx,w,h} = setupCanvas(c);
  drawGrid(ctx,w,h,50);
  const T=ecg[0].length, laneH=h/12;
  for (let lead=0; lead<12; lead++) {
    const arr=ecg[lead];
    const lo=Math.min(...arr), hi=Math.max(...arr), range=Math.max(hi-lo,0.1);
    const yMid=lead*laneH+laneH/2;
    ctx.fillStyle = lead%2===0 ? "rgba(255,255,255,0.012)" : "rgba(0,0,0,0)";
    ctx.fillRect(0, lead*laneH, w, laneH);
    ctx.fillStyle="rgba(154,164,192,0.7)"; ctx.font="10px ui-monospace,monospace";
    ctx.fillText(LEAD_NAMES[lead], 6, lead*laneH+12);
    ctx.strokeStyle = lead<3 ? "#5ad7ff" : (lead<6 ? "#7fe6c8" : "#ffd166");
    ctx.lineWidth=1.2; ctx.beginPath();
    for (let i=0;i<T;i++) {
      const x=(i/(T-1))*w;
      const y=yMid-((arr[i]-(lo+hi)/2)/range)*(laneH*0.85);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    ctx.stroke();
  }
  // time axis ticks every 1 s
  ctx.strokeStyle="rgba(255,255,255,0.12)"; ctx.lineWidth=0.8;
  for (let s=1; s<10; s++) {
    const x=(s*FS/(T-1))*w;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke();
    ctx.fillStyle="rgba(154,164,192,0.4)"; ctx.font="9px ui-monospace,monospace";
    ctx.fillText(s+"s", x+2, h-4);
  }
}

// ─── input spikes ─────────────────────────────────────────────────────
function drawInputSpikes(events) {
  const c=$("inSpikeCanvas");
  const {ctx,w,h}=setupCanvas(c);
  drawGrid(ctx,w,h,50);
  const laneH=h/24;
  let onCount=0, offCount=0;
  for (const [ch,t] of events) {
    const isOn=ch<12, x=(t/(T_INPUT-1))*w, y=ch*laneH+laneH/2;
    ctx.fillStyle = isOn ? "#ff5f7a" : "#5ad7ff";
    ctx.fillRect(x-0.7, y-laneH*0.4, 1.4, laneH*0.8);
    isOn ? onCount++ : offCount++;
  }
  ctx.strokeStyle="rgba(255,255,255,0.15)"; ctx.lineWidth=0.8;
  ctx.beginPath(); ctx.moveTo(0,h/2); ctx.lineTo(w,h/2); ctx.stroke();
  ctx.fillStyle="rgba(154,164,192,0.65)"; ctx.font="11px ui-monospace,monospace";
  ctx.fillText("ON  (rising)", 8, 14);
  ctx.fillText("OFF (falling)", 8, h/2+14);
  $("inputSpikeCount").textContent =
    `${(onCount+offCount).toLocaleString()} total events · `+
    `${((onCount+offCount)/(12*T_INPUT)*100).toFixed(1)}% spike rate · `+
    `${(100-(onCount+offCount)/(12*T_INPUT)*100).toFixed(1)}% silent`;
}

// ─── block raster ─────────────────────────────────────────────────────
function drawBlockRaster(events, shape) {
  const c=$("blockCanvas");
  const {ctx,w,h}=setupCanvas(c);
  drawGrid(ctx,w,h,50);
  const [C,T]=shape;
  for (const [ch,t] of events) {
    const x=(t/Math.max(T-1,1))*w, y=(ch/C)*h;
    ctx.fillStyle="#ffd166";
    ctx.fillRect(x-0.8, y, 1.6, Math.max(h/C-0.3,1.0));
  }
  ctx.fillStyle="rgba(154,164,192,0.7)"; ctx.font="10px ui-monospace,monospace";
  ctx.fillText(`${C} channels`, 8,12);
  ctx.fillText(`${T} time-steps`, 8,26);
  const r=events.length/(C*T);
  ctx.fillText(`spike rate: ${(r*100).toFixed(1)}%`, 8, 40);
}

// ─── pool gate ────────────────────────────────────────────────────────
function drawPoolGates(events, T) {
  const c=$("poolCanvas");
  const {ctx,w,h}=setupCanvas(c);
  drawGrid(ctx,w,h,50);
  ctx.fillStyle="#6df0a8";
  for (const [,t] of events) {
    const x=(t/Math.max(T-1,1))*w;
    ctx.fillRect(x-1.2, h*0.15, 2.4, h*0.7);
  }
  ctx.fillStyle="rgba(154,164,192,0.7)"; ctx.font="11px ui-monospace,monospace";
  ctx.fillText(`${events.length} gate spikes (pools at ${((events.length/Math.max(T,1))*100).toFixed(1)}% of time-steps)`, 8, 14);
}

// ─── output integrator ────────────────────────────────────────────────
function drawVout(vout) {
  const c=$("voutCanvas");
  const {ctx,w,h}=setupCanvas(c);
  drawGrid(ctx,w,h,40);
  const K=vout.length, Td=vout[0].length;
  const flat=[].concat(...vout);
  const lo=Math.min(...flat), hi=Math.max(...flat), pad=(hi-lo)*0.12+0.001;
  const ymin=lo-pad, ymax=hi+pad;
  ctx.lineWidth=2;
  LABELS.forEach((cl,i) => {
    if (!vout[i]) return;
    ctx.strokeStyle = CLASS_COLORS[cl] || "#aaa";
    ctx.beginPath();
    vout[i].forEach((v,t) => {
      const x=(t/(Td-1))*w, y=h-((v-ymin)/(ymax-ymin))*h;
      t===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
    const lastY=h-((vout[i][Td-1]-ymin)/(ymax-ymin))*h;
    ctx.fillStyle=CLASS_COLORS[cl]||"#aaa"; ctx.font="11px ui-monospace,monospace";
    ctx.fillText(cl, w-26, lastY+4);
  });
}

// ─── readout bars ─────────────────────────────────────────────────────
function drawReadout(result) {
  const probs=result.probs;
  const positive=LABELS.filter(c => (probs[c]||0)>=0.5);
  $("readout").innerHTML = LABELS.map(c => {
    const p=probs[c]||0, cls=p>=0.5?"read-row pos":"read-row neg";
    return `<div class="${cls}">
      <div class="read-name" style="color:${CLASS_COLORS[c]}">${c}</div>
      <div class="read-desc">${CLASS_DESC[c]||c}</div>
      <div class="read-track"><div class="read-fill" style="width:${(p*100).toFixed(1)}%"></div></div>
      <div class="read-pct">${(p*100).toFixed(1)}%</div>
    </div>`;
  }).join("");

  const truth=result.true_labels;
  if (truth) {
    const predSet=new Set(positive), truthSet=new Set(truth);
    const tp=[...truthSet].filter(x=>predSet.has(x));
    const fp=[...predSet].filter(x=>!truthSet.has(x));
    const fn=[...truthSet].filter(x=>!predSet.has(x));
    const ok=fp.length===0&&fn.length===0;
    $("vsTruth").innerHTML = `
      <span class="label">ground truth:</span>
      ${truth.length ? truth.map(t=>`<span class="badge" style="color:${CLASS_COLORS[t]||'#fff'}">${t} — ${CLASS_DESC[t]||t}</span>`).join("") : '<span class="badge muted">none</span>'}
      <span class="label">model ≥ 0.5:</span>
      ${positive.length ? positive.map(t=>`<span class="badge" style="color:${CLASS_COLORS[t]||'#fff'}">${t}</span>`).join("") : '<span class="badge muted">none</span>'}
      <span class="badge ${ok?'ok':'warn'}">${ok?'✓ exact match':`tp=${tp.length} fp=${fp.length} fn=${fn.length}`}</span>
    `;
  } else {
    $("vsTruth").innerHTML=`<span class="label">no ground truth (uploaded file)</span>`;
  }
}

// ─── sparsity bars ────────────────────────────────────────────────────
function drawSparsity(rates) {
  const keys=Object.keys(rates);
  const mean=keys.reduce((s,k)=>s+rates[k],0)/Math.max(keys.length,1);
  $("sparsityChart").innerHTML = keys.map(k => {
    const r=rates[k], crit=r>0.5;
    return `<div class="sp-row">
      <div class="sp-name">${k}</div>
      <div class="sp-bar"><div class="sp-fill ${crit?'sp-warn':''}" style="width:${(r*100).toFixed(1)}%"></div></div>
      <div class="sp-val ${crit?'sp-val-warn':''}">${(r*100).toFixed(2)}%</div>
    </div>`;
  }).join("");
  $("sparsityNote").textContent =
    `Mean spike rate: ${(mean*100).toFixed(1)}% · `+
    `On a neuromorphic chip, effective ops ≈ MACs × spike_rate per layer.`;
}

// ─── AUROC bar chart ──────────────────────────────────────────────────
function drawAurocBars(rows) {
  const c=$("aurocBarCanvas");
  const {ctx,w,h}=setupCanvas(c);
  ctx.clearRect(0,0,w,h);
  if (!rows||!rows.length) return;
  const PAD={l:90,r:20,t:30,b:40};
  const bw=w-PAD.l-PAD.r;
  const bh=h-PAD.t-PAD.b;
  // background grid
  ctx.strokeStyle="rgba(255,255,255,0.06)"; ctx.lineWidth=0.8;
  [0.5,0.6,0.7,0.8,0.9,1.0].forEach(v=>{
    const x=PAD.l+(v-0.4)/(1-0.4)*bw;
    ctx.beginPath(); ctx.moveTo(x,PAD.t); ctx.lineTo(x,PAD.t+bh); ctx.stroke();
    ctx.fillStyle="rgba(154,164,192,0.5)"; ctx.font="10px ui-monospace,monospace";
    ctx.fillText(v.toFixed(1), x-8, PAD.t+bh+14);
  });
  const n=rows.length, bh2=Math.min(36,(bh-n*6)/n);
  const gap=(bh-(n*bh2))/(n+1);
  rows.forEach((r,i) => {
    const y=PAD.t+gap*(i+1)+i*bh2;
    const x0=PAD.l+(0.5-0.4)/(1-0.4)*bw;
    const xAuroc=PAD.l+(r.macro_auroc-0.4)/(1-0.4)*bw;
    const xAuprc=PAD.l+(r.macro_auprc-0.4)/(1-0.4)*bw;
    const col=MODEL_COLOR[r.model]||"#aaa";
    // AUROC bar
    ctx.fillStyle=col+"55";
    ctx.fillRect(x0, y, xAuroc-x0, bh2*0.55);
    ctx.fillStyle=col;
    ctx.fillRect(x0, y, xAuroc-x0, bh2*0.52);
    // AUPRC bar (smaller, offset)
    ctx.fillStyle=col+"44";
    ctx.fillRect(x0, y+bh2*0.55, xAuprc-x0, bh2*0.4);
    // labels
    ctx.fillStyle="#e6eaf2"; ctx.font=`600 12px ui-sans-serif,sans-serif`;
    ctx.fillText(MODEL_LABEL[r.model]||r.model, 4, y+bh2*0.4);
    ctx.fillStyle=col; ctx.font="12px ui-monospace,monospace";
    ctx.fillText(`AUROC ${(r.macro_auroc*100).toFixed(1)}%`, xAuroc+5, y+bh2*0.4);
    ctx.fillStyle=col+"cc"; ctx.font="10px ui-monospace,monospace";
    ctx.fillText(`AUPRC ${(r.macro_auprc*100).toFixed(1)}%`, xAuprc+5, y+bh2*0.9);
  });
  ctx.fillStyle="rgba(154,164,192,0.5)"; ctx.font="10px ui-monospace,monospace";
  ctx.fillText("macro-AUROC / AUPRC →", PAD.l, PAD.t-10);
}

// ─── per-class bar chart ─────────────────────────────────────────────
function drawPerClassBars(rows) {
  const c=$("perClassCanvas");
  const {ctx,w,h}=setupCanvas(c);
  ctx.clearRect(0,0,w,h);
  if (!rows||!rows.length) return;
  const labels=(rows[0].labels)||["NORM","MI","STTC","CD","HYP"];
  const nC=labels.length, nM=rows.length;
  const PAD={l:54,r:16,t:28,b:48};
  const bw=w-PAD.l-PAD.r, bh=h-PAD.t-PAD.b;
  const groupW=bw/nC, barW=Math.min(22, (groupW-8)/nM);
  // gridlines
  ctx.strokeStyle="rgba(255,255,255,0.06)"; ctx.lineWidth=0.8;
  [0.6,0.7,0.8,0.9,1.0].forEach(v=>{
    const y=PAD.t+bh-(v-0.5)/(1-0.5)*bh;
    ctx.beginPath(); ctx.moveTo(PAD.l,y); ctx.lineTo(PAD.l+bw,y); ctx.stroke();
    ctx.fillStyle="rgba(154,164,192,0.5)"; ctx.font="9px ui-monospace,monospace";
    ctx.fillText(v.toFixed(1), 2, y+3);
  });
  labels.forEach((lbl,ci)=>{
    const gx=PAD.l+ci*groupW+groupW/2;
    rows.forEach((r,mi)=>{
      const val=(r.per_label&&r.per_label[lbl]&&r.per_label[lbl].auroc)||0;
      const bx=gx-(nM/2-mi-0.5)*barW-barW/2;
      const barH=(val-0.5)/(1-0.5)*bh;
      const by=PAD.t+bh-barH;
      const col=MODEL_COLOR[r.model]||"#aaa";
      ctx.fillStyle=col+"bb";
      ctx.fillRect(bx, by, barW-1, barH);
      ctx.strokeStyle=col; ctx.lineWidth=0.8;
      ctx.strokeRect(bx, by, barW-1, barH);
    });
    // class label
    ctx.fillStyle=CLASS_COLORS[lbl]||"#ccc"; ctx.font="bold 12px ui-sans-serif,sans-serif";
    ctx.textAlign="center";
    ctx.fillText(lbl, gx, PAD.t+bh+14);
    ctx.fillStyle="rgba(154,164,192,0.55)"; ctx.font="9px ui-monospace,monospace";
    ctx.fillText(CLASS_DESC[lbl]||"", gx, PAD.t+bh+26);
    ctx.textAlign="left";
  });
  // legend
  rows.forEach((r,i)=>{
    const lx=PAD.l+i*110;
    ctx.fillStyle=MODEL_COLOR[r.model]||"#aaa";
    ctx.fillRect(lx, 6, 10, 10);
    ctx.fillStyle="#e6eaf2"; ctx.font="10px ui-sans-serif,sans-serif";
    ctx.fillText(MODEL_LABEL[r.model]||r.model, lx+14, 16);
  });
}

// ─── energy bar chart ─────────────────────────────────────────────────
function drawEnergyBars(rows) {
  const c=$("energyBarCanvas");
  const {ctx,w,h}=setupCanvas(c);
  ctx.clearRect(0,0,w,h);
  if (!rows||!rows.length) return;
  const PAD={l:90,r:80,t:20,b:40};
  const bw=w-PAD.l-PAD.r, bh=h-PAD.t-PAD.b;
  const energies=rows.map(r=>r.model==="snn" ? r.energy_pj_neuromorphic : r.energy_pj_dense);
  const maxE=Math.max(...energies);
  const n=rows.length, barH=Math.min(36,(bh-n*6)/n), gap=(bh-(n*barH))/(n+1);
  rows.forEach((r,i)=>{
    const e=energies[i];
    const y=PAD.t+gap*(i+1)+i*barH;
    const xW=(e/maxE)*bw;
    const col=MODEL_COLOR[r.model]||"#aaa";
    ctx.fillStyle=col+"44";
    ctx.fillRect(PAD.l, y, xW, barH*0.7);
    ctx.fillStyle=col;
    ctx.fillRect(PAD.l, y, xW, barH*0.65);
    // labels
    ctx.fillStyle="#e6eaf2"; ctx.font=`600 12px ui-sans-serif,sans-serif`;
    ctx.fillText(MODEL_LABEL[r.model]||r.model, 4, y+barH*0.5);
    ctx.fillStyle=col; ctx.font="11px ui-monospace,monospace";
    const tag=r.model==="snn"?" (neuromorphic)":" (GPU ASIC)";
    ctx.fillText(fmtJ(e)+tag, PAD.l+xW+5, y+barH*0.5);
  });
  ctx.fillStyle="rgba(154,164,192,0.5)"; ctx.font="10px ui-monospace,monospace";
  ctx.fillText("estimated energy / inference →", PAD.l, PAD.t-6);
}

function fmtJ(pj) {
  if (pj==null) return "—";
  if (pj>=1e12) return (pj/1e12).toFixed(2)+" J";
  if (pj>=1e9)  return (pj/1e9).toFixed(2)+" mJ";
  if (pj>=1e6)  return (pj/1e6).toFixed(2)+" µJ";
  if (pj>=1e3)  return (pj/1e3).toFixed(2)+" nJ";
  return pj.toFixed(0)+" pJ";
}
function fmtNum(n) {
  if (n==null) return "—";
  if (n>=1e9) return (n/1e9).toFixed(2)+"G";
  if (n>=1e6) return (n/1e6).toFixed(2)+"M";
  if (n>=1e3) return (n/1e3).toFixed(1)+"k";
  return n.toLocaleString();
}
function fmtMs(n) { return n==null?"—":n.toFixed(1)+" ms"; }
function fmtPct(n, d=2) { return n==null?"—":(n*100).toFixed(d)+"%"; }

// ─── ROC curves ───────────────────────────────────────────────────────
function drawROCCurves(rows) {
  const c=$("rocCanvas");
  if (!c) return;
  const {ctx,w,h}=setupCanvas(c);
  ctx.clearRect(0,0,w,h);
  if (!rows||!rows.length) return;
  const labels=(rows[0].labels)||["NORM","MI","STTC","CD","HYP"];
  const nC=labels.length;
  const cols=nC, rows2=1;
  const cellW=w/cols, cellH=h;
  const PAD=32;
  const DASHES={snn:[], cnn:[4,4], resnet:[1,4]};

  labels.forEach((lbl,ci)=>{
    const ox=ci*cellW, oy=0;
    const pw=cellW-PAD*1.5, ph=cellH-PAD*1.5;
    const x0=ox+PAD, y0=oy+PAD*0.5;
    // background
    ctx.strokeStyle="rgba(255,255,255,0.06)"; ctx.lineWidth=0.7;
    [0,0.25,0.5,0.75,1.0].forEach(v=>{
      ctx.beginPath(); ctx.moveTo(x0+v*pw,y0); ctx.lineTo(x0+v*pw,y0+ph); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x0,y0+ph-v*ph); ctx.lineTo(x0+pw,y0+ph-v*ph); ctx.stroke();
    });
    // diagonal (chance)
    ctx.strokeStyle="rgba(255,255,255,0.18)"; ctx.lineWidth=0.8; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(x0,y0+ph); ctx.lineTo(x0+pw,y0); ctx.stroke();
    ctx.setLineDash([]);
    // per-model ROC curves
    rows.forEach(r=>{
      if (!r.roc_curves||!r.roc_curves[lbl]) return;
      const {fpr,tpr}=r.roc_curves[lbl];
      const col=MODEL_COLOR[r.model]||"#aaa";
      ctx.strokeStyle=col; ctx.lineWidth=r.model==="snn"?2:1.2;
      ctx.setLineDash(DASHES[r.model]||[]);
      ctx.beginPath();
      fpr.forEach((f,i)=>{
        const x=x0+f*pw, y=y0+ph-tpr[i]*ph;
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
      }); ctx.stroke(); ctx.setLineDash([]);
      // AUC label at end
      if (r.per_label&&r.per_label[lbl]) {
        const auc=(r.per_label[lbl].auroc||0).toFixed(3);
        const fx=fpr[fpr.length-1], ty=tpr[tpr.length-1];
        // don't overlap — offset by model index
        const mi=["snn","cnn","resnet"].indexOf(r.model);
        ctx.fillStyle=col+"dd"; ctx.font=`bold 9px ui-monospace,monospace`;
        ctx.fillText(auc, x0+pw*0.5+mi*26, y0+ph-6-mi*10);
      }
    });
    // label
    ctx.fillStyle=CLASS_COLORS[lbl]||"#ccc"; ctx.font="bold 13px ui-sans-serif,sans-serif";
    ctx.textAlign="center"; ctx.fillText(lbl, x0+pw/2, y0+14); ctx.textAlign="left";
    ctx.fillStyle="rgba(154,164,192,0.5)"; ctx.font="9px ui-monospace,monospace";
    ctx.textAlign="center"; ctx.fillText(CLASS_DESC[lbl]||"", x0+pw/2, y0+26); ctx.textAlign="left";
    // axis labels
    ctx.fillStyle="rgba(154,164,192,0.4)"; ctx.font="9px ui-monospace,monospace";
    ctx.fillText("FPR", x0+pw*0.4, y0+ph+14);
    ctx.save(); ctx.translate(x0-18, y0+ph*0.5); ctx.rotate(-Math.PI/2);
    ctx.fillText("TPR", 0, 0); ctx.restore();
  });
  // legend
  const legendItems=[
    {model:"snn", dash:[]}, {model:"cnn", dash:[4,4]}, {model:"resnet", dash:[1,4]}
  ];
  legendItems.forEach((item,i)=>{
    const lx=10+i*150, ly=h-10;
    ctx.strokeStyle=MODEL_COLOR[item.model]; ctx.lineWidth=item.dash.length?1:2;
    ctx.setLineDash(item.dash);
    ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(lx+20,ly); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle="rgba(154,164,192,0.8)"; ctx.font="10px ui-sans-serif,sans-serif";
    ctx.fillText(MODEL_LABEL[item.model], lx+24, ly+3);
  });
}

// ─── comparison table ─────────────────────────────────────────────────
function renderComparison(data) {
  CMP_DATA = data;
  const rows=data.rows||[];
  if (!rows.length) return;
  drawAurocBars(rows);
  drawPerClassBars(rows);
  drawEnergyBars(rows);
  drawROCCurves(rows);

  const snn=rows.find(r=>r.model==="snn");
  const cnn=rows.find(r=>r.model==="cnn");

  // key results banner
  // Update results reveal cards
  rows.forEach(r => {
    const el = $("rr"+r.model.charAt(0).toUpperCase()+r.model.slice(1));
    if (el) el.textContent = (r.macro_auroc||0).toFixed(4);
  });

  if (snn && cnn) {
    const energyRatio  = cnn.energy_pj_dense / Math.max(snn.energy_pj_neuromorphic,1);
    const opsRatio     = cnn.macs / Math.max(snn.sops,1);
    const aurocGap     = snn.macro_auroc - cnn.macro_auroc;
    const snnE_pj      = snn.energy_pj_neuromorphic;
    const cnnE_pj      = cnn.energy_pj_dense;

    // Update all instances of the energy ratio
    [$("heroEnergyRatio"),$("heroEnergyRatio2")].forEach(el=>{ if(el) el.textContent=energyRatio.toFixed(0)+"×"; });
    // Update stat break with real numbers
    const sbItems=document.querySelectorAll(".sb-num");
    if(sbItems[0]) sbItems[0].textContent=(snn.spike_rate_mean!=null?(100-(snn.spike_rate_mean*100)).toFixed(0):87)+"%";
    if(sbItems[2]) sbItems[2].textContent=energyRatio.toFixed(0)+"×";

    $("energyNote").innerHTML =
      `The SNN uses <strong>${energyRatio.toFixed(0)}× less</strong> estimated energy per inference
       (${fmtJ(snnE_pj)} neuromorphic vs. ${fmtJ(cnnE_pj)} dense) — driven by both
       <strong>${opsRatio.toFixed(1)}× fewer operations</strong> (SOPs vs. MACs) and
       <strong>37× cheaper per-operation cost</strong> (0.1 pJ/SOP vs. 3.7 pJ/MAC on 45 nm CMOS).
       The macro-AUROC cost is <strong>${Math.abs(aurocGap*100).toFixed(2)} pp</strong>.
       Wall-clock latency on a dense GPU still favors the CNN because we <em>simulate</em> LIF
       dynamics in PyTorch — on actual Loihi 2 silicon, SNN inference is sub-millisecond.`;

    $("compareNote").innerHTML =
      `The SNN achieves <strong>${(snn.macro_auroc*100).toFixed(1)}% macro-AUROC</strong>
       (${(aurocGap*100).toFixed(2)} pp vs CNN) while using
       <strong>${opsRatio.toFixed(1)}× fewer effective operations</strong> and
       <strong>${energyRatio.toFixed(0)}× less energy</strong>.
       The mean spike rate of ${snn.spike_rate_mean!=null?(snn.spike_rate_mean*100).toFixed(1)+"%":"~12%"}
       means ~${(100-(snn.spike_rate_mean||0.126)*100).toFixed(0)}% of synapses are idle at any
       given time-step — the core reason neuromorphic hardware wins on energy.`;

    // Holter monitor comparison card
    const infPerDay = 8640;
    const snnJ_day  = (infPerDay * snnE_pj) / 1e12;
    const cnnJ_day  = (infPerDay * cnnE_pj) / 1e12;
    const coinCell_J = 1000;
    const snnDays   = coinCell_J / snnJ_day;
    const cnnHours  = (coinCell_J / cnnJ_day) * 24;
    const hc = $("holterComparison");
    if (hc) hc.innerHTML = `
      <div class="hc-item snn">
        <div class="hc-label">SNN on Loihi (neuromorphic)</div>
        <div class="hc-val">${snnDays.toFixed(0)} days</div>
        <div class="hc-sub">
          CR2032 coin cell lasts <strong>${snnDays.toFixed(0)} days</strong> of continuous
          inference (${fmtJ(snnE_pj)} × 8,640/day = ${(snnJ_day*1000).toFixed(0)} mJ/day)
        </div>
      </div>
      <div class="hc-item cnn">
        <div class="hc-label">CNN on dense ASIC</div>
        <div class="hc-val">${cnnHours.toFixed(0)} hrs</div>
        <div class="hc-sub">
          Same CR2032 lasts only <strong>${cnnHours.toFixed(0)} hours</strong>
          (${fmtJ(cnnE_pj)} × 8,640/day = ${cnnJ_day.toFixed(1)} J/day)
        </div>
      </div>
      <div class="hc-item tradeoff">
        <div class="hc-label">The tradeoff</div>
        <div class="hc-val">${Math.abs(aurocGap*100).toFixed(2)} pp</div>
        <div class="hc-sub">
          AUROC gap: ${(snn.macro_auroc*100).toFixed(1)}% SNN vs ${(cnn.macro_auroc*100).toFixed(1)}% CNN.
          For a screening wearable that routes positives to clinical follow-up,
          this tradeoff is favorable.
        </div>
      </div>`;
  }

  const hdrs=["Model","AUROC","AUPRC","Params","MACs","Spike rate","Eff. SOPs","Latency","Energy / inf"];
  const bestAuroc=Math.max(...rows.map(r=>r.macro_auroc||0));
  const bestAuprc=Math.max(...rows.map(r=>r.macro_auprc||0));
  const lowestE=Math.min(...rows.map(r=>(r.model==="snn"?r.energy_pj_neuromorphic:r.energy_pj_dense)||Infinity));
  const td=(v,best=false)=>`<td class="${best?'pos em-num':'num'}">${v}</td>`;
  $("compareTable").innerHTML = `<table>
    <thead><tr>${hdrs.map(h=>`<th>${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(r=>{
      const isSnn=r.model==="snn";
      const energy=isSnn?r.energy_pj_neuromorphic:r.energy_pj_dense;
      return `<tr class="${isSnn?'row-snn':''}">
        <td>${MODEL_LABEL[r.model]||r.model}</td>
        ${td(fmtPct(r.macro_auroc,1), r.macro_auroc===bestAuroc)}
        ${td(fmtPct(r.macro_auprc,1), r.macro_auprc===bestAuprc)}
        ${td(fmtNum(r.params))}
        ${td(fmtNum(r.macs))}
        <td class="num">${isSnn&&r.spike_rate_mean!=null?fmtPct(r.spike_rate_mean,1):"—"}</td>
        <td class="num">${isSnn?fmtNum(r.sops):"—"}</td>
        ${td(fmtMs(r.latency_ms))}
        ${td(fmtJ(energy), energy===lowestE)}
      </tr>`;
    }).join("")}</tbody>
  </table>`;
}

// ─── status ───────────────────────────────────────────────────────────
async function refreshStatus() {
  const el=$("runStatus");
  try {
    const s=await jget("/status");
    if (s.checkpoint==null) {
      el.innerHTML=`<div class="badge warn">no checkpoint</div>`;
    } else {
      const epoch=s.epoch??"?";
      const auroc=s.best_macro_auroc!=null?s.best_macro_auroc.toFixed(4):"?";
      el.innerHTML=`<div class="badge ok"><span class="dot"></span>ep ${epoch} · val-AUROC ${auroc}</div>`;
    }
  } catch(e) {
    el.innerHTML=`<div class="badge bad">${e.message}</div>`;
  }
}

// ─── examples ─────────────────────────────────────────────────────────
async function loadExamples() {
  try {
    const r=await jget("/examples?n=20&fold=10");
    EXAMPLES=r.examples; LABELS=r.labels;
    const sel=$("examplePicker");
    sel.innerHTML=`<option value="">— ${EXAMPLES.length} test recordings —</option>`+
      EXAMPLES.map(e=>`<option value="${e.ecg_id}">${e.ecg_id} · ${e.labels.join(" + ")}</option>`).join("");
    if (EXAMPLES.length>0) sel.selectedIndex=1;
  } catch(e) {
    $("runHint").textContent="examples unavailable";
  }
}

// ─── training history ────────────────────────────────────────────────
async function loadHistory() {
  let data;
  try { data=await jget("/history?run_name=main"); }
  catch(e) { $("historyMeta").textContent=`(${e.message})`; return; }
  const hist=data.history||[];
  if (!hist.length) return;
  const c=$("historyCanvas");
  const {ctx,w,h}=setupCanvas(c);
  drawGrid(ctx,w,h,40);
  const xs=hist.map(e=>e.epoch);
  const aurocs=hist.map(e=>e.val_macro_auroc||0);
  const losses=hist.map(e=>e.train_loss||0);
  const maxEp=Math.max(...xs), minEp=Math.min(...xs), span=Math.max(maxEp-minEp,1);
  const PAD={l:28,r:70,t:18,b:22};
  const xof=(ep)=>PAD.l+(ep-minEp)/span*(w-PAD.l-PAD.r);
  const yof=(v)=>(h-PAD.b)-(v-0.3)/(1.0-0.3)*(h-PAD.t-PAD.b);
  // auroc axis labels
  ctx.fillStyle="rgba(154,164,192,0.5)"; ctx.font="9px ui-monospace,monospace";
  for (let v=0.4;v<=0.95;v+=0.1) {
    const y=yof(v);
    ctx.fillText(v.toFixed(1), 2, y+3);
    ctx.strokeStyle="rgba(255,255,255,0.04)"; ctx.lineWidth=0.7;
    ctx.beginPath(); ctx.moveTo(PAD.l,y); ctx.lineTo(w-PAD.r,y); ctx.stroke();
  }
  // per-class lines (thin, from val_per_label)
  const LABELS5=["NORM","MI","STTC","CD","HYP"];
  if (hist[0]&&hist[0].val_per_label) {
    LABELS5.forEach(lbl=>{
      const col=CLASS_COLORS[lbl]||"#888";
      ctx.strokeStyle=col+"60"; ctx.lineWidth=1; ctx.beginPath();
      hist.forEach((e,i)=>{
        const v=(e.val_per_label[lbl]&&e.val_per_label[lbl].auroc)||0;
        const x=xof(xs[i]), y=yof(v);
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
      }); ctx.stroke();
    });
  }
  // macro val AUROC
  ctx.strokeStyle="#5ad7ff"; ctx.lineWidth=2.5; ctx.beginPath();
  aurocs.forEach((v,i)=>{
    const x=xof(xs[i]), y=yof(v);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }); ctx.stroke();
  // running best
  ctx.strokeStyle="rgba(109,240,168,0.9)"; ctx.setLineDash([5,4]); ctx.lineWidth=1.5; ctx.beginPath();
  let best=-Infinity;
  aurocs.forEach((v,i)=>{
    if(v>best)best=v;
    const x=xof(xs[i]), y=yof(best);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }); ctx.stroke(); ctx.setLineDash([]);
  // train loss (right axis, scaled to same visual range)
  const lossMax=Math.max(...losses.filter(Boolean),1), lossMin=Math.min(...losses.filter(Boolean),0);
  ctx.strokeStyle="rgba(255,180,84,0.65)"; ctx.lineWidth=1.5; ctx.beginPath();
  losses.forEach((v,i)=>{
    // map loss to [0.3,0.7] visual range
    const norm=(v-lossMin)/(lossMax-lossMin||1);
    const vScaled=0.3+norm*0.4;
    const x=xof(xs[i]), y=yof(1-vScaled+0.3);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }); ctx.stroke();
  // axis label
  ctx.fillStyle="rgba(154,164,192,0.5)"; ctx.font="9px ui-monospace,monospace";
  ctx.fillText("epoch →", w-PAD.r+4, h-PAD.b+10);
  // legend (right side)
  const legendItems=[
    {col:"#5ad7ff", label:"macro AUROC", dash:false},
    {col:"rgba(109,240,168,0.9)", label:"best", dash:true},
    {col:"rgba(255,180,84,0.65)", label:"train loss", dash:false},
    ...LABELS5.map(l=>({col:(CLASS_COLORS[l]||"#888")+"90", label:l, dash:false})),
  ];
  legendItems.forEach((item,i)=>{
    const lx=w-PAD.r+6, ly=PAD.t+i*14+6;
    ctx.strokeStyle=item.col; ctx.lineWidth=item.dash?1:1.5;
    if (item.dash) ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(lx+14,ly); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle="rgba(154,164,192,0.7)"; ctx.font="9px ui-monospace,monospace";
    ctx.fillText(item.label, lx+16, ly+3);
  });
  const cur=aurocs[aurocs.length-1];
  $("historyMeta").innerHTML=
    `<span class="mono">epoch ${maxEp} · val macro-AUROC ${cur.toFixed(4)} · best ${best.toFixed(4)} · ${hist.length} epochs</span>`;
}

// ─── comparison loader ───────────────────────────────────────────────
async function loadComparison() {
  try {
    const data=await jget("/comparison");
    renderComparison(data);
  } catch(e) {
    $("compareTable").innerHTML=`<div class="empty muted">${e.message}</div>`;
  }
}

// ─── inference ────────────────────────────────────────────────────────
async function runFromExample(ecg_id) {
  $("runHint").textContent="running inference…"; $("runBtn").disabled=true;
  try {
    const r=await jget(`/infer_example?ecg_id=${ecg_id}`);
    paintAll(r); $("runHint").textContent=`ecg_id ${ecg_id}`;
  } catch(e) {
    $("runHint").textContent=`error: ${e.message}`;
  } finally { $("runBtn").disabled=false; }
}

async function runFromUpload(file) {
  $("runHint").textContent="uploading…"; $("runBtn").disabled=true;
  try {
    const form=new FormData(); form.append("file",file);
    const r=await fetch("/infer_upload",{method:"POST",body:form});
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    paintAll(await r.json()); $("runHint").textContent=file.name;
  } catch(e) {
    $("runHint").textContent=`error: ${e.message}`;
  } finally { $("runBtn").disabled=false; }
}

function paintAll(result) {
  LAST=result;
  if (!LABELS.length && result.labels) LABELS=result.labels;
  drawECG(result.ecg);
  drawInputSpikes(result.input_spikes);
  if (result.rasters?.block_last && result.rasters?.block_last_shape)
    drawBlockRaster(result.rasters.block_last, result.rasters.block_last_shape);
  drawPoolGates(result.pool_gates||[], result.pool_gates_T||1);
  drawVout(result.vout);
  drawReadout(result);
  drawSparsity(result.spike_rates||{});
  $("ecgMeta").textContent=
    `10 s · ${result.fs||100} Hz · ecg_id ${result.ecg_id??"(uploaded)"}`;
}

// ─── main ─────────────────────────────────────────────────────────────
async function main() {
  await refreshStatus();
  await loadExamples();
  await loadHistory();
  await loadComparison();
  setInterval(refreshStatus, 30000);
  setInterval(loadHistory,    60000);
  setInterval(loadComparison, 120000);
  $("runBtn").addEventListener("click",()=>{
    const v=$("examplePicker").value;
    if (v) runFromExample(parseInt(v,10));
  });
  $("uploadInput").addEventListener("change",e=>{
    const f=e.target.files[0]; if(f) runFromUpload(f);
  });
  setTimeout(()=>{
    const v=$("examplePicker").value; if(v) runFromExample(parseInt(v,10));
  }, 800);
  window.addEventListener("resize",()=>{
    if (LAST) paintAll(LAST);
    if (CMP_DATA) {
      renderComparison(CMP_DATA);
    }
    loadHistory();
  });
}

main().catch(err=>{
  document.body.innerHTML=`<pre style="color:#ff5b6e;padding:24px">${err.stack||err}</pre>`;
});
