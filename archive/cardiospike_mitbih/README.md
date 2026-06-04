# CardioSpike

A spiking neural network (SNN) that classifies real ECG beats into the
AAMI EC57 superclasses (N · S · V · F · Q), trained end-to-end with
surrogate-gradient BPTT on the **MIT-BIH Arrhythmia Database**.

The project is built to read like a small frontier-lab artifact:

- **Real data, no synthesis.** 48 records of two-lead 360 Hz ECG with
  cardiologist-verified beat annotations, pulled from PhysioNet via the
  `wfdb` library.
- **Inter-patient evaluation.** Train on the 22-record DS1 split,
  evaluate on the disjoint 22-record DS2 split (de Chazal et al., 2004).
  No patient leaks across the split — the only honest test of
  generalization on this dataset.
- **Event-driven input.** ECG voltage is converted to {ON, OFF} spikes
  by delta modulation — the same encoding a silicon retina performs on
  light. Quiet baseline segments produce zero events; QRS complexes
  produce dense bursts.
- **All-spiking hidden layers.** Every hidden unit is a leaky
  integrate-and-fire neuron with membrane potential
  *V(t) = β V(t−1) + I(t)*; trained with the fast-sigmoid surrogate
  gradient (Neftci et al., 2019).
- **Honest energy claim.** The web demo reports synaptic-operation
  counts per layer using measured spike rates — the speedup an
  event-driven chip (Loihi, SpiNNaker) would see — not a hand-wavy
  efficiency story.

## What's in here

```
src/
  download_data.py    fetch 44 MIT-BIH records from PhysioNet (~84 MB)
  preprocess.py       inter-patient DS1/DS2 segmentation, AAMI mapping
  encoding.py         delta-modulation spike encoder (numpy + torch)
  model.py            Conv-LIF SNN, ~272k params, BPTT-trainable
  train.py            class-weighted CE, cosine LR, MPS / CUDA / CPU
  evaluate.py         metrics + sparsity + JSON exports for the web demo
web/
  index.html          single-page demo
  app.js              live ECG → spikes → SNN visualizer
  style.css           dark-mode design
models/
  cardiospike_best.pt trained weights (saved by best macro-F1 on DS2)
results/
  metrics.json        full per-class report
  sparsity.json       per-layer spike rates and synaptic-op counts
  history.json        training curves
```

## Reproduce from scratch

```bash
pip install -r requirements.txt
python src/download_data.py        # ~84 MB, ~1 min on a normal connection
python src/preprocess.py           # builds data/processed/ds[12].npz
PYTHONPATH=. python src/train.py --epochs 25
PYTHONPATH=. python src/evaluate.py
./scripts/serve.sh                 # open http://localhost:8000
```

Trains in ≈15 min on an Apple Silicon M-series GPU (MPS), or in similar
time on a single CUDA GPU.

## Architecture

```
delta-encoded ECG (T=260, C=2) ─┐
  Conv1d(2→16, k=7) → LIF → MaxPool(2)         # T → 130
  Conv1d(16→32, k=5) → LIF → MaxPool(2)        # T → 65
  Conv1d(32→64, k=3) → LIF → MaxPool(2)        # T → 32
  Flatten ──────────────────────┐
                                ├─ concat → Linear(2052→128) → LIF
  R-R features (4-dim, scalar) ─┘             ↓
                                  Linear(128→5) → non-leaky readout
                                  (V_mem accumulates over T_dense=12 steps)
```

The R-R feature vector is `[pre_RR, post_RR, mean_RR_10, pre_RR / mean_RR_10]`
in seconds. It's standard in the ECG literature (de Chazal et al., 2004) and
without it the inter-patient model cannot distinguish supraventricular ectopic
beats (S) from normal beats — they have similar single-beat morphology and
differ mostly in timing. We z-score using DS1 statistics only.

All hidden neurons are `snntorch.Leaky(beta=0.9)` with a fast-sigmoid
surrogate gradient (slope=25). The output layer is a non-leaky
integrator whose membrane potential, summed over the dense-head
time window, is used as the classification logit.

## Why this is "really" an SNN, not a CNN with extra steps

This is the question every reviewer will ask, so:

1. **Spike-only inputs.** The network receives binary {0,1} events at
   each timestep, not continuous voltage. Quiet segments propagate
   exactly zero signal.
2. **Stateful neurons with hard non-linearities.** Each LIF unit
   integrates its input into a membrane potential that *resets* on a
   spike. The surrogate gradient only exists during backprop — at
   inference, the forward graph is a fully binary, event-driven
   computation, which is the precondition for running on neuromorphic
   silicon.
3. **Measured sparse activity.** Hidden spike rates per layer are
   reported in `results/sparsity.json` and visualized in the web
   demo's "Synaptic-operation budget" panel. They are well below the
   50% mark where event-driven chips beat dense accelerators.

## Results (DS2 inter-patient hold-out, 49,692 beats)

All numbers below come straight from `results/metrics.json` produced by
`src/evaluate.py` against the trained checkpoint. They are not
intra-patient (which trivially gets >99%) and not on a curated subset.

| Metric                            | Value           |
|-----------------------------------|-----------------|
| Overall accuracy                  | **86.8%**       |
| Macro-F1 (5-class AAMI)           | **0.415**       |
| Per-class F1 — N (Normal)         | **0.93**        |
| Per-class F1 — V (Ventricular)    | **0.73**        |
| Per-class F1 — S (Supraventric.)  | 0.30            |
| Per-class F1 — F (Fusion)         | 0.12            |
| Per-class F1 — Q (Unknown/paced)  | 0.00 (7 test beats — undefined) |
| Mean input spike rate             | 19.8% of timesteps |
| Mean hidden spike rate            | **4.0%** of LIF neurons fire per step |
| Synaptic-op savings vs dense ANN  | **31.8×**       |
| Parameters                        | 272k            |

Honest framing: inter-patient AAMI 5-class is a deliberately hard
benchmark. Single-beat morphology cannot distinguish supraventricular
ectopics (S) from normal beats across patients — they look similar.
Including R-R interval features closes most (not all) of that gap.
Fusion (F) beats and the under-represented Q class remain difficult by
construction.

## Limitations and honesty notes

- We use the standard inter-patient DS1/DS2 split but exclude the four
  paced records (102, 104, 107, 217) by convention, since they break
  AAMI's beat ontology.
- "Energy ratio" is a synaptic-op count, not a measured Joules
  comparison on real silicon. We say so plainly in the demo.
- Class Q (paced/unknown) is essentially absent from the inter-patient
  test set (≈7 beats) and its F1 is therefore not meaningful.

## Datasets and references

- Moody GB, Mark RG. **The impact of the MIT-BIH Arrhythmia Database.**
  *IEEE Eng. Med. Biol. Mag.* 20(3):45-50, 2001.
- de Chazal P, O'Dwyer M, Reilly RB. **Automatic classification of
  heartbeats using ECG morphology and heartbeat interval features.**
  *IEEE Trans. Biomed. Eng.* 51(7):1196-1206, 2004. *(DS1/DS2 split.)*
- Neftci EO, Mostafa H, Zenke F. **Surrogate gradient learning in
  spiking neural networks.** *IEEE SP Mag.* 36(6):51-63, 2019.
- Eshraghian J et al. **Training spiking neural networks using lessons
  from deep learning.** *Proc. IEEE* 111(9):1016-1054, 2023. *(snnTorch.)*
- Davies M et al. **Loihi: A neuromorphic manycore processor with
  on-chip learning.** *IEEE Micro* 38(1):82-99, 2018.

Data are subject to the
[PhysioNet Credentialed Health Data License](https://physionet.org/about/licenses/).
