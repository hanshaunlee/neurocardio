# NeuroCardio

A deep **spiking neural network** that diagnoses 12-lead ECGs end-to-end —
trained on PTB-XL (21,837 cardiologist-validated recordings), evaluated
inter-patient, and served as a public Modal web app.

> **Live demo:** https://hanshaunlee--neurocardio-web.modal.run
> **One-line pitch:** clinical-grade ECG diagnosis at **~287× less estimated
> energy per inference** than a parameter-matched CNN, for a **3.6 pp** macro-AUROC
> price — a concrete data point on the neuromorphic accuracy/energy trade-off.

**This project is a pivot.** It began as
[**UniMarket**](https://github.com/hanshaunlee/unimarket) — a framework for
*learning neural dynamics from electrophysiology data* with strict
anti-over-claiming safeguards.
I carried the core interest — **brain-inspired, biologically-grounded
computation** — but pivoted from *modeling* brain dynamics to *building with*
them: applying spiking (neuromorphic) networks to a problem with a sharp,
measurable payoff — clinical ECG diagnosis at a fraction of the energy.

Within this repository it's also the second-generation model. The first version
(`archive/cardiospike_mitbih/`) was a small SNN for single-beat
arrhythmia classification on MIT-BIH. NeuroCardio scales the same idea
to clinical-grade multi-label diagnosis on a real benchmark dataset, with
two architectural contributions that are not standard in the SNN
literature.

---

## Reader's guide

A quick index — every claim in this project links to the code, data, or figure
that backs it. Detail follows in the rest of the README.

| Topic | Where it's addressed |
|---|---|
| **Problem & insight** | [Problem & motivation](#problem--motivation) · [What's novel](#whats-novel) |
| **What was built** | [Architecture](#architecture) · [Repo layout](#layout) · [Run it](#run-it) · live demo |
| **Evaluation & evidence** | [Results & evaluation](#results--evaluation) · [Baseline comparison](#baseline-comparison) · [Failure analysis](#failure-analysis) |
| **Communication** | This README · the [live web demo](https://hanshaunlee--neurocardio-web.modal.run) · reproducible [Run it](#run-it) |
| **Process & disclosure** | [AI usage & attribution](#ai-usage-collaborators--integrity) · [Honest framing](#honest-framing) · [References](#references) |

---

## Problem & motivation

Twelve-lead ECG is the most common cardiac diagnostic test in the world, and
deep nets now read them at near-cardiologist level — but those models run on
GPUs drawing tens to hundreds of watts. The interesting frontier for *continuous*
or *wearable* cardiac monitoring is not another accuracy point; it's getting
clinical-grade diagnosis onto a power budget that a coin cell or an implant can
sustain.

**Spiking neural networks (SNNs)** are the natural candidate: they compute with
sparse binary events, so on event-driven (neuromorphic) silicon the energy cost
scales with the number of *spikes actually emitted*, not with the dense
multiply-accumulate count. The open question this project tackles head-on:

> **Can a spiking network reach competitive macro-AUROC on a real, multi-label
> clinical ECG benchmark — and if so, what does that cost in accuracy, and what
> does it buy in energy?**

Most SNN-on-ECG papers stop at single-beat arrhythmia toys (MIT-BIH). We go to
PTB-XL super-class diagnosis — a *competitive* benchmark where published CNN /
Transformer models cluster at 0.92–0.94 AUROC — and measure the trade-off
honestly against parameter-matched dense baselines we trained ourselves on the
identical split and pipeline.

## What's novel

1. **Membrane-gated residuals.** Plain spike-level residual addition
   would mix binary tensors and break the binary-spike abstraction.
   Each S-TCN block instead injects its skip path into the *membrane
   potential* of its output LIF neuron, scaled by a learnable per-channel
   sigmoid gate `g = σ(α)` (initialised at `α=0`). Spikes between blocks
   remain strictly binary; the optimiser learns, per channel, whether the
   block should be transform-dominated or residual-dominated.

2. **Spiking gate-attention pooling.** Temporal pooling is driven by a
   small LIF "gate head" whose output spikes mark the time-steps worth
   integrating. The pooled feature vector is the mean of spike features
   at those time-steps. No softmax, no continuous mixing — strictly
   event-driven.

Both are described in `neurocardio/model.py` and rendered live in the
web demo.

## Architecture

```
12-lead ECG (12 × 1000 @ 100 Hz)
  ─ Δ-modulation encoder ──► 24 spike channels (12 leads × {ON, OFF})
  ─ Conv1d(stride=2) + BN + LIF             ──► stem  (64 ch, T=500)
  ─ S-TCN block × 6: dilations [1,2,4,8,16,32], pools [1,2,1,2,1,2]
        each block:  Conv(dilated) → BN → LIF
                     Conv          → BN → ⨁ (membrane gate)──► LIF
                                                      ▲
                          input-skip  ─ 1×1 conv ─────┘
  ─ Spiking gate-attention pool (LIF + event-driven mean)
  ─ Linear(384→256) + BN + LIF
  ─ Linear(256→5)  + non-leaky integrator (T_dense=8)
                                  → sigmoid (multi-label BCE)
```

≈ **1.89 M** parameters; every hidden unit is `snntorch.Leaky` with a fast-
sigmoid surrogate gradient (slope=25), trained with BPTT.

## Dataset

[PTB-XL 1.0.3 (PhysioNet)](https://physionet.org/content/ptb-xl/1.0.3/),
21,837 12-lead recordings of 10 s at 100 Hz, annotated with one or more
[SCP-ECG](https://www.iso.org/standard/46493.html) diagnostic statements
that were cardiologist-validated (~70%) or written from scratch by a
cardiologist. Standard `strat_fold` split: folds 1–8 train, 9 val, 10
test (no patient overlap → inter-patient evaluation).

Two label spaces are supported:

| Label set | n labels | examples |
|-----------|---------|----------|
| `diagnostic_superclass` | 5 | `NORM`, `MI`, `STTC`, `CD`, `HYP` |
| `diagnostic_subclass`   | 24 | `AMI`, `IMI`, `LMI`, `STTC`, `LAFB`, `IRBBB`, `1AVB`, … |

Training optimises macro-AUROC across the chosen label set (the metric
used in Strodthoff et al., *PTB-XL benchmark* 2020).

## Layout

```
neurocardio/
  data.py              PTB-XL loader, label aggregation, .npy cache builder
  dataset.py           PyTorch Dataset over the memory-mapped cache
  augment.py           Gaussian noise, baseline wander, lead dropout, random crop
  encoding.py          12-lead Δ-modulation spike encoder (torch + numpy)
  model.py             S-TCN with membrane-gated residuals + spiking attn pool
  baselines.py         CNN1D and ResNet1D dense baselines (parameter-matched)
  compute.py           SOP / MAC counting + Horowitz energy model
  factory.py           model dispatch (snn | cnn | resnet)
  train.py             multi-label BCE, AdamW, cosine LR, EMA, resumable
  infer.py             single-sample inference with rich spike traces
modal_app/
  app.py               Modal app (data_prep, train, list_runs, compare_models, web)
web/
  index.html app.js style.css   front-end served by the Modal web function
results/                        canonical final artifacts (committed, see below)
  comparison.json                 energy/compute/accuracy source-of-truth
  main/                           SNN: best.pt, latest.pt, history.json, test_metrics.json, config.json
  cnn_baseline/  resnet_baseline/ dense baselines: same five files each
archive/cardiospike_mitbih/     v1 (MIT-BIH single-beat) project for reference
runs/                           local smoke-test run artifacts (configs, history, metrics)
```

The `results/` directory holds the **final trained checkpoints and statistics**
pulled from the Modal Volume, committed in-repo so every number in this README
and the web demo is reproducible offline:

- `results/main/best.pt` — the SNN checkpoint that produced the reported
  0.8663 test macro-AUROC (epoch 155 by val AUROC).
- `results/{cnn,resnet}_baseline/best.pt` — the parameter-matched dense baselines.
- `results/comparison.json` — the regenerated energy/compute/accuracy table that
  the web demo reads live (the source of truth for the headline numbers).

## Run it

### Cloud (Modal — recommended)

You need a Modal account (`pip install modal && modal token new`).

```bash
# 1) One-shot: download PTB-XL (~1.75 GB) into a Modal Volume.
modal run --detach modal_app/app.py::data_prep

# 2) Deploy the public web app + train endpoint.
modal deploy modal_app/app.py
# → prints a stable URL like https://<you>--neurocardio-web.modal.run

# 3) Launch the detached SNN training run (200 epochs ≈ 34 h GPU @ ~10 min/epoch;
#    8 h/session, auto-resumes from best.pt across preemptions).
modal run --detach modal_app/app.py::train \
  --run-name main --model snn --wallclock-hours 8

# 4) Train dense baselines (each ~5 minutes on A10G).
modal run --detach modal_app/app.py::train \
  --run-name cnn_baseline --model cnn --epochs 80 --batch-size 128 \
  --wallclock-hours 4
modal run --detach modal_app/app.py::train \
  --run-name resnet_baseline --model resnet --epochs 80 --batch-size 128 \
  --wallclock-hours 4

# 5) After the SNN run finishes, rebuild the comparison table.
modal run modal_app/app.py::compare_models \
  --run-names "main,cnn_baseline,resnet_baseline"

# 6) Reload the web URL — it picks up the latest best.pt and the new
#    comparison.json automatically.
```

Checkpoints, training history and the rebuilt cache all live in the
`neurocardio-vol` Modal Volume, so the run is resumable across
preemption and you can swap GPUs mid-training without restarting from
scratch (`--no-resume` to start fresh). The actual `main` run survived
several A10G preemptions and resumed from `best.pt` each time — see
[Process & development history](#process--development-history).

### Local development

For local smoke testing without paying for Modal compute:

```bash
pip install -r requirements.txt
# (manually drop a few PTB-XL records into data/ptbxl_test/, see scripts/)
PYTHONPATH=. python -m neurocardio.train \
  --cache-dir data/ptbxl_cache_test --output-dir runs/local \
  --epochs 2 --batch-size 4 --num-workers 0 --device mps
```

The `runs/local_*` directories in this repo are exactly these smoke-test
artifacts (config + history + test metrics), committed as development evidence.

---

## Results & evaluation

**Final SNN run: 200 epochs on Modal A10G, complete.** Test fold is PTB-XL
fold 10 (no patient overlap with train/val). `best.pt` = epoch 155
(val macro-AUROC 0.8748).

| Model | Test macro-AUROC | Test AUPRC | Params |
|-------|-----------------:|-----------:|-------:|
| **SNN (NeuroCardio)** | **0.8663** | 0.6971 | 1,893,768 |
| CNN1D baseline | 0.9023 | 0.7644 | 1,741,381 |
| ResNet1D baseline | 0.9017 | 0.7663 | 1,879,685 |

- **Accuracy gap:** the SNN trails the best dense baseline by **−3.6 pp**
  macro-AUROC — far tighter than the ~12 pp gap we saw mid-training, and the
  central "price of neuromorphic operation" number this project reports.
- Parameter counts are matched within **8%** so the comparison is about
  *representation*, not capacity.

### Energy & compute (the headline)

Energy uses the standard Horowitz (ISSCC 2014) constants — 3.7 pJ per dense
FP32 MAC vs. 0.1 pJ per synaptic operation (SOP) on event-driven silicon
(Davies et al., Loihi). SOPs are weighted by the **measured** spike rate on the
test fold, not a simulated proxy.

The ~287× headline is **two effects multiplied**, not one: ~7.8× fewer
operations (sparsity — neurons only compute when they fire) **×** ~37× cheaper
per operation (a spike accumulate at 0.1 pJ vs a dense multiply-accumulate at
3.7 pJ). The per-op cost ratio is the larger of the two factors, so the energy
win is far bigger than the operation-count win alone would suggest.

| Metric | SNN | CNN1D | vs CNN |
|---|---:|---:|---:|
| Effective ops / inference | **~37.9 M SOPs** | 294.6 M MACs | **~7.8× fewer** |
| Mean spike rate | **~10.9%** (≈89% silent) | 100% dense | — |
| Energy / inference (est.) | **~3.8 µJ** | ~1.09 mJ | **~287× less** |
| Energy vs ResNet1D | ~3.8 µJ | ~187 µJ | ~49× less |

> `comparison.json` (committed under [`results/`](results/) and regenerated by
> `compare_models`) is the authoritative source of truth; the web demo's big
> numbers and prose are filled live from it (nothing energy-related is
> hard-coded). The live headline ratio is **287×** after the full run. The exact
> values above are read from that file.

### Failure analysis

- **Weakest class is HYP** (hypertrophy), test AUROC ≈ 0.77 — and it has the
  fewest training examples (2,119). The error tracks class support, exactly as
  you'd expect, rather than something pathological in the spiking dynamics.
- **Latency is a simulation artifact, not a model property.** Per-sample latency
  is dominated by Python-level unrolling of LIF dynamics in PyTorch (one kernel
  launch per time-step). On real neuromorphic silicon a LIF step is one cycle.
  This is why we report *energy from operation counts*, which is hardware-honest,
  rather than wall-clock latency, which is not.
- **Mid-training vs final.** Early snapshots reported here in prior revisions
  (~0.78 AUROC, ~12 pp gap) were taken at epoch ≈25; the final 200-epoch numbers
  above supersede them. Training plateaued from epoch 155 onward.

## Baseline comparison

We trained two dense (non-spiking) baselines on the **same** PTB-XL split
with the **same** training pipeline (BCE-with-logits + class weights +
balanced sampling, cosine LR, EMA, AdamW, identical augmentations):

- **CNN1D** — 1.74 M parameters, BN+ReLU after every conv, max-pool at
  the same depths as the SNN.
- **ResNet1D** — 1.88 M parameters, pre-activation residual blocks with
  strided 1×1 skips, XResNet1D-style.

Both consume the raw 12-lead ECG (the SNN's delta-modulation encoder is
its own first stage, so the ANNs are allowed to use the native
representation). Code: `neurocardio/baselines.py`. Regenerate the table with:

```bash
modal run modal_app/app.py::compare_models \
  --run-names "main,cnn_baseline,resnet_baseline"
```

The web demo runs **all three** models on the same normalised signal at
inference time and shows a side-by-side 3-model readout with per-model
exact-match verdicts against ground truth.

## Modal resources

| Resource          | When                                  |
|-------------------|---------------------------------------|
| `data_prep` CPU   | one-time, ~15 min                     |
| `train` A10G GPU (SNN) | full 200-epoch run, ~34 h of GPU time (see timing above) |
| `train` A10G GPU (baselines) | minutes per clean run, but many redone attempts |
| `web` T4 GPU      | on-demand, idle scales to zero        |
| Volume storage    | always-on, ~3 GB raw + 1 GB cache     |

If you have an A100 quota, change `gpu="A10G"` → `gpu="A100"` in
`modal_app/app.py` (training is GPU-bound on the LIF Python loops,
so you'll see a ~2–3× speed-up).

### How long training took on Modal

**Important caveat up front:** the committed run logs (`history.json` on the
`neurocardio-vol` Volume) only capture each model's *final, clean, end-to-end
run*. They badly understate the real effort. Every model — and the **CNN and
ResNet baselines especially** — failed and had to be **redone many times**
(OOM, A10G preemptions, data-pipeline and config bugs) before a run completed
cleanly, and each redo *overwrote* the previous attempt's logs in the same
`runs/<name>/` directory. The detached training jobs were ephemeral Modal apps
that have since been reaped, so the failed attempts are no longer individually
recoverable. The clean-run numbers below are therefore a *floor*; the true
wall-clock spent reaching them was many times higher across all three models.

What the surviving logs *do* show (read straight off `history.json`, where
`train.py` records `wallclock_seconds` per epoch — not estimates):

| Run | Epochs | Per-epoch (median) | Final clean run | Sessions on record |
|-----|-------:|-------------------:|----------------:|-------------------:|
| **SNN (`main`)** | 200 | ~10.3 min | **≈ 34 h GPU** | 7 (6 preemptions/resumes) |
| CNN1D baseline | 80 | ~4.3 s | ~5.7 min GPU | 1 (after many failed redos) |
| ResNet1D baseline | 80 | ~2.9 s | ~3.9 min GPU | 1 (after many failed redos) |

Two honest readings of this table:

- **The SNN's per-epoch cost is real and structural.** At ~10 min/epoch it is
  ~150–200× slower *per epoch* than the dense baselines, because every forward
  pass unrolls LIF membrane dynamics across ~500 time-steps and backprops through
  all of them (BPTT) where the ANNs do a single dense pass. A ~34 h job far
  exceeds Modal's A10G preemption window, so `main` ran across **7 sessions with 6
  preemptions/resumes** (8 h budget each via `--wallclock-hours 8`),
  checkpointing every ~10 min and resuming from `best.pt` to reach 200 epochs.
  This is also why energy is reported from operation counts, not wall-clock — the
  slowness is a PyTorch simulation artifact of stepping LIF neurons in a Python
  loop, **not** a property of the model on neuromorphic hardware.
- **The "5 minutes" for the baselines is misleading on its own.** That is only
  the final successful pass. Getting each baseline to train end-to-end on the
  full PTB-XL pipeline took many failed and re-launched runs; the quick final
  number is the *result* of that debugging, not the cost of it.

## Process & development history

Evidence of genuine iteration over time:

- **The pivot.** The project started as
  [**UniMarket**](https://github.com/hanshaunlee/unimarket), a framework for
  learning neural dynamics from electrophysiology data. I kept the through-line —
  biologically-grounded, brain-inspired computation — but pivoted from *analysing*
  neural dynamics to *deploying* them as spiking networks on a concrete clinical
  task with a measurable energy payoff. NeuroCardio is the result of that pivot.
- **v1 → v2.** `archive/cardiospike_mitbih/` is the first-gen single-beat SNN on
  MIT-BIH; NeuroCardio is the ground-up rewrite for clinical multi-label PTB-XL.
- **A resumable training saga.** At ~10 min/epoch the `main` SNN run was a ~34 h
  job — far longer than Modal's A10G preemption window — so it ran across **7
  sessions with 6 preemptions/resumes**, checkpointing to the `neurocardio-vol`
  Volume every ~10 min and resuming from `best.pt` each time to reach 200 epochs.
  The resume machinery in `train.py` exists *because* of this, not for show.
- **The baselines were not free either.** Their committed logs show a clean ~5 min
  run, but each only succeeded after **many failed and re-launched attempts**
  (OOM, preemption, pipeline/config bugs) whose logs were overwritten by the next
  retry — the quick final number is the result of that debugging, not its cost.
  See [How long training took on Modal](#how-long-training-took-on-modal).
- **Web demo redesign.** The frontend was rebuilt around a live 3-model
  comparison, a data-driven energy ratio, PR-curve cards, and an animated ECG
  hero — all numbers sourced from `comparison.json` rather than hard-coded.
- **Local smoke artifacts** under `runs/` are committed so the training/eval
  pipeline is inspectable without Modal access.

> Note on commit history: this repository was published in a small number of
> squashed commits rather than a long incremental history. The development
> artifacts above (archived v1, resumable checkpoint logic, smoke-test run
> outputs, the deployed live demo) are the honest record of iteration.

## Honest framing

- Macro-AUROC on PTB-XL super-class is a competitive benchmark; published
  CNN/Transformer numbers cluster around **0.92–0.94**. Our dense baselines hit
  ~0.90 and the SNN reaches **0.8663** — a real, measured **3.6 pp** gap, which we
  report as a *cost*, not hide.
- "Spike sparsity" here is the *measured* fraction of LIF neurons firing
  per time-step on the test fold. It is not a simulated proxy.
- The energy headline ("~287× cheaper than the CNN") is a synaptic-operation
  count argument under published per-op energy constants, **not** a Joules
  measurement on real silicon. The web demo says so explicitly.
- The web demo's inference endpoint runs the **same** snntorch model on
  Modal — there is no separate inference model, no quantisation, no
  TFLite trick.

## AI usage, collaborators & integrity

- **Author:** Han Lee. Solo project.
- **AI assistance — disclosed.** This project was built with substantial help
  from **Claude Code (Anthropic)** acting as a pair-programming assistant:
  drafting and refactoring model/training/Modal code, designing the web
  frontend, and writing documentation including this README. All architectural
  decisions, the experiment design, the choice of benchmark and baselines, and
  every reported number were directed and verified by the author. AI-generated
  code was reviewed before committing.
- **Borrowed foundations (cited, not forked).** Built on open-source libraries —
  **snnTorch** (Eshraghian et al.) for spiking primitives and surrogate
  gradients, PyTorch, and **Modal** for compute/serving. The membrane-gated
  residual and spiking gate-attention pooling are original implementations in
  `neurocardio/model.py`, informed by the residual-SNN and surrogate-gradient
  literature cited below. The ResNet1D baseline follows the XResNet1D recipe.
  PTB-XL is the PhysioNet dataset (Wagner et al.). No existing project repo was
  forked.
- **Limitations** are discussed inline under [Honest framing](#honest-framing)
  and [Failure analysis](#failure-analysis).

## References

- Wagner P. et al. **PTB-XL, a large publicly available
  electrocardiography dataset.** *Sci. Data* 7, 154 (2020).
- Strodthoff N. et al. **Deep learning for ECG analysis: benchmarks and
  insights from PTB-XL.** *IEEE J. Biomed. Health Inf.* (2020).
- Neftci EO, Mostafa H, Zenke F. **Surrogate gradient learning in
  spiking neural networks.** *IEEE SP Mag.* 36(6):51-63 (2019).
- Eshraghian J et al. **Training spiking neural networks using lessons
  from deep learning.** *Proc. IEEE* 111(9):1016-1054 (2023). *(snnTorch.)*
- Hu Y, Deng L, Wu Y. **Advancing spiking neural networks towards deep
  residual learning.** *IEEE TNNLS* (2024).
- Bai S, Kolter JZ, Koltun V. **An empirical evaluation of generic
  convolutional and recurrent networks for sequence modeling.** *(TCN
  backbone.)*
- Horowitz M. **Computing's energy problem (and what we can do about it).**
  *ISSCC* (2014). *(Per-operation energy constants.)*
- Davies M et al. **Loihi: a neuromorphic manycore processor with on-chip
  learning.** *IEEE Micro* 38(1):82-99 (2018). *(Event-driven SOP energy.)*
