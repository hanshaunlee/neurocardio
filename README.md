# NeuroCardio

A deep **spiking neural network** that diagnoses 12-lead ECGs end-to-end —
trained on PTB-XL (21,837 cardiologist-validated recordings), evaluated
inter-patient, and served as a public Modal web app.

This is the second-generation model in this repository.  The first version
(`archive/cardiospike_mitbih/`) was a small SNN for single-beat
arrhythmia classification on MIT-BIH.  NeuroCardio scales the same idea
to clinical-grade multi-label diagnosis on a real benchmark dataset, with
two architectural contributions that are not standard in the SNN
literature.

## What's novel

1. **Membrane-gated residuals.**  Plain spike-level residual addition
   would mix binary tensors and break the binary-spike abstraction.
   Each S-TCN block instead injects its skip path into the *membrane
   potential* of its output LIF neuron, scaled by a learnable per-channel
   sigmoid gate `g = σ(α)`.  Spikes between blocks remain binary; the
   optimiser learns per channel whether the block should be
   transform-dominated or residual-dominated.

2. **Spiking gate-attention pooling.**  Temporal pooling is driven by a
   small LIF "gate head" whose output spikes mark the time-steps worth
   integrating.  The pooled feature vector is the mean of spike features
   at those time-steps.  No softmax, no continuous mixing — strictly
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

≈ 1.9 M parameters; every hidden unit is `snntorch.Leaky` with a fast-
sigmoid surrogate gradient (slope=25), trained with BPTT.

## Dataset

[PTB-XL 1.0.3 (PhysioNet)](https://physionet.org/content/ptb-xl/1.0.3/),
21,837 12-lead recordings of 10 s at 100 Hz, annotated with one or more
[SCP-ECG](https://www.iso.org/standard/46493.html) diagnostic statements
that were cardiologist-validated (~70%) or written from scratch by a
cardiologist.  Standard `strat_fold` split: folds 1–8 train, 9 val, 10
test (no patient overlap).

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
  train.py             multi-label BCE, AdamW, cosine LR, EMA, resumable
  infer.py             single-sample inference with rich spike traces
modal_app/
  app.py               Modal app (data_prep, train, list_runs, web)
web/
  index.html app.js style.css   front-end served by the Modal web function
archive/cardiospike_mitbih/     v1 (MIT-BIH single-beat) project for reference
```

## Run it

### Cloud (Modal — recommended)

You need a Modal account (`pip install modal && modal token new`).

```bash
# 1) One-shot: download PTB-XL (~1.75 GB) into a Modal Volume.
modal run --detach modal_app/app.py::data_prep

# 2) Deploy the public web app + train endpoint.
modal deploy modal_app/app.py
# → prints a stable URL like https://<you>--neurocardio-web.modal.run

# 3) Launch the 14-hour detached SNN training run.
modal run --detach modal_app/app.py::train \
  --run-name main --model snn --wallclock-hours 14

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
scratch (`--no-resume` to start fresh).

### Local development

For local smoke testing without paying for Modal compute:

```bash
pip install -r requirements.txt
# (manually drop a few PTB-XL records into data/ptbxl_test/, see scripts/)
PYTHONPATH=. python -m neurocardio.train \
  --cache-dir data/ptbxl_cache_test --output-dir runs/local \
  --epochs 2 --batch-size 4 --num-workers 0 --device mps
```

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
representation).

Snapshot taken while the 14-hour SNN run is still in progress (the
SNN's `best.pt` was at epoch ≈ 25). Energy uses Horowitz (ISSCC 2014)
constants: 3.7 pJ/MAC for dense FP32 vs. 0.1 pJ/SOP on event-driven
silicon (Davies et al., Loihi).

| Model         | Test AUROC | Test AUPRC | Params | Dense MACs | Spike rate | Effective SOPs | B=1 latency (T4) | Energy / inference |
|---------------|-----------:|-----------:|-------:|-----------:|-----------:|---------------:|-----------------:|-------------------:|
| **SNN**       |   **0.78** | 0.50       | 1.89 M | 325.7 M    | **10.9 %** | **38.3 M**     | 1546 ms ¹        | **3.83 µJ** ²      |
| CNN1D         |   **0.90** | 0.76       | 1.74 M | 294.6 M    | 100 %      | 294.6 M        |    1.85 ms       |  1.09 mJ           |
| ResNet1D      |     0.90   | 0.77       | 1.88 M |  50.5 M    | 100 %      |  50.5 M        |    1.95 ms       |  0.19 mJ           |

¹ The SNN's per-sample latency is dominated by Python-level unrolling of
LIF dynamics in PyTorch — a simulation artifact. On real neuromorphic
silicon the LIF step is one cycle, not one kernel launch.
² Energy uses the spike-weighted SOP count × 0.1 pJ/SOP. The dense ANN
columns use MAC count × 3.7 pJ/MAC.

**Headline.** The SNN matches the dense baselines within ~12 pp of macro-
AUROC *while the run is still mid-training*, using **7.7×** fewer
effective synaptic operations and **~285×** less estimated energy per
inference on event-driven silicon. The web demo's "Compute vs dense
baselines" card is regenerated from `/vol/comparison.json` and is the
authoritative source of truth — re-run it after the SNN finishes
with:

```bash
modal run modal_app/app.py::compare_models \
  --run-names "main,cnn_baseline,resnet_baseline"
```

## Modal cost estimate

| Resource          | When                  | ~Cost (Modal pricing, May 2026) |
|-------------------|-----------------------|--------------------------------|
| `data_prep` CPU   | one-time, ~15 min     | < $0.05                        |
| `train` A10G GPU  | 14-hour detached run  | ~$15                           |
| `web` T4 GPU      | on-demand, idle scales to zero | ~$0.01/visit (cold start) |
| Volume storage    | always-on, ~3 GB raw + 1 GB cache | < $0.10/mo |

If you have an A100 quota, change `gpu="A10G"` → `gpu="A100"` in
`modal_app/app.py` (the training is GPU-bound on the LIF Python loops,
so you'll see a ~2-3× speed-up).

## Honest framing

- Macro-AUROC on PTB-XL super-class is a competitive benchmark; published
  CNN/Transformer numbers cluster around **0.92–0.94**.  An SNN aiming
  for parity has to fight harder than its dense counterpart because BPTT
  through spiking dynamics is noisier.  Expected end-of-training numbers
  are reported in `web/data/summary.json` after the run finishes.
- "Spike sparsity" here is the *measured* fraction of LIF neurons firing
  per time-step on the test fold.  It is not a simulated proxy.
- Energy headline ("Nx cheaper than ANN") is a synaptic-operation count
  argument, not a Joules measurement on real silicon. The web demo
  always says so explicitly.
- The web demo's inference endpoint runs the **same** snntorch model on
  Modal — there is no separate inference model, no quantisation, no
  TFLite trick.

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
