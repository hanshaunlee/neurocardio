"""
Evaluate the trained CardioSpikeSNN on DS2 and export everything the web demo
and the README need:

  results/metrics.json       — accuracy, macro-F1, per-class P/R/F1, AAMI-style CM
  results/sparsity.json      — average spike rate per LIF layer + synaptic-op estimate
  web/data/inference.json    — per-class example beats with spike traces & probs
  web/data/summary.json      — top-line numbers for the dashboard header
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, precision_recall_fscore_support)

from src.model import CardioSpikeSNN
from src.encoding import delta_encode_batch, delta_encode_numpy

HERE = os.path.dirname(os.path.dirname(__file__))
PROC = os.path.join(HERE, "data", "processed")
MODEL_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cardiospike_best.pt")
RR_STATS = os.path.join(MODEL_DIR, "rr_stats.npz")
RESULTS_DIR = os.path.join(HERE, "results")
WEB_DATA = os.path.join(HERE, "web", "data")
os.makedirs(WEB_DATA, exist_ok=True)

CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_NAMES = {
    "N": "Normal beat",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion beat",
    "Q": "Unknown / paced",
}


def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def load_test():
    d = np.load(os.path.join(PROC, "ds2.npz"))
    rr_stats = np.load(RR_STATS)
    R = (d["R"] - rr_stats["mean"]) / rr_stats["std"]
    return d["X"], d["y"], d["rec"], R.astype(np.float32)


def synaptic_op_estimate(model: CardioSpikeSNN, spike_rates: dict, T_input: int):
    """
    A standard neuromorphic figure of merit: count synaptic operations driven
    by each spike. For a Conv1d layer with C_in input channels, C_out output
    channels, kernel size k applied at each of T' timesteps, with input spike
    rate r ∈ [0,1], the expected number of MAC-equivalent synaptic ops is

        ops_SNN(layer) ≈ r · C_in · C_out · k · T'

    The dense ANN equivalent (no sparsity, runs once) is

        ops_ANN(layer) =     C_in · C_out · k · T'

    so the SNN cost is reduced by exactly the input spike rate. We sum across
    the convolutional stack as a comparable lower-bound estimate. This is
    consistent with reporting in Roy et al. (Nature 2019) and Davies et al.
    (Loihi papers).
    """
    # Layer shapes (must match model.py)
    layers = [
        ("conv1", 2,  16, 7, 130, spike_rates.get("input", 1.0)),
        ("conv2", 16, 32, 5, 65,  spike_rates["s1"]),
        ("conv3", 32, 64, 3, 32,  spike_rates["s2"]),
        ("fc1",   2048, 128, 1, model.T_dense, spike_rates["s3"]),
        ("fc2",   128, 5, 1, model.T_dense, spike_rates["s4"]),
    ]
    ops_snn = 0.0
    ops_ann = 0.0
    per_layer = {}
    for name, cin, cout, k, T, r in layers:
        dense = cin * cout * k * T
        sparse = r * dense
        per_layer[name] = {"dense_ops": dense, "snn_ops": sparse,
                            "input_spike_rate": r}
        ops_snn += sparse
        ops_ann += dense
    return {"per_layer": per_layer, "total_snn_ops": ops_snn,
            "total_ann_ops": ops_ann,
            "energy_ratio_ann_over_snn": ops_ann / max(ops_snn, 1.0)}


def main(theta=0.15):
    dev = device()
    ckpt = torch.load(MODEL_PATH, map_location=dev, weights_only=False)
    model = CardioSpikeSNN(T_dense=ckpt["config"]["T_dense"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    Xnp, ynp, recnp, Rnp = load_test()
    X = torch.from_numpy(Xnp).float()
    y = torch.from_numpy(ynp).long()
    R = torch.from_numpy(Rnp).float()

    # Full-test eval
    preds, probs = [], []
    spike_means = {"input": 0.0, "s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0}
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = X[i:i+512].to(dev)
            rb = R[i:i+512].to(dev)
            sp = delta_encode_batch(xb, theta=theta)
            logits, tr = model(sp, rr=rb, return_traces=True)
            preds.append(logits.argmax(1).cpu())
            probs.append(F.softmax(logits, dim=1).cpu())
            spike_means["input"] += float(sp.mean().item())
            for k in ("s1", "s2", "s3", "s4"):
                spike_means[k] += float(tr[k].float().mean().item())
            n_batches += 1
    for k in spike_means:
        spike_means[k] /= n_batches

    preds = torch.cat(preds).numpy()
    probs = torch.cat(probs).numpy()

    acc = float(accuracy_score(ynp, preds))
    macro_f1 = float(f1_score(ynp, preds, average="macro", zero_division=0))
    p, r, f, sup = precision_recall_fscore_support(ynp, preds, labels=list(range(5)),
                                                    zero_division=0)
    cm = confusion_matrix(ynp, preds, labels=list(range(5))).tolist()
    report = classification_report(ynp, preds, labels=list(range(5)),
                                   target_names=CLASSES, zero_division=0,
                                   output_dict=True)

    metrics = {
        "overall_accuracy": acc,
        "macro_f1": macro_f1,
        "n_test_beats": int(len(ynp)),
        "classes": CLASSES,
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(sup[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": cm,
        "sklearn_report": report,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "sklearn_report"},
                     indent=2))

    sparsity = synaptic_op_estimate(model, spike_means, T_input=260)
    sparsity_full = {"spike_rates": spike_means, **sparsity}
    with open(os.path.join(RESULTS_DIR, "sparsity.json"), "w") as fp:
        json.dump(sparsity_full, fp, indent=2)
    with open(os.path.join(WEB_DATA, "sparsity.json"), "w") as fp:
        json.dump(sparsity_full, fp, indent=2)
    print("Mean spike rates:", spike_means)
    print(f"Synaptic-op energy ratio (ANN/SNN): "
          f"{sparsity['energy_ratio_ann_over_snn']:.2f}×")

    # ---- Web demo data --------------------------------------------------
    rng = np.random.default_rng(0)
    examples = []
    for cls_idx, cls in enumerate(CLASSES):
        idxs = np.where(ynp == cls_idx)[0]
        if len(idxs) == 0:
            continue
        correct_idxs = idxs[preds[idxs] == cls_idx]
        chosen = rng.choice(correct_idxs if len(correct_idxs) > 0 else idxs,
                            size=min(6, len(correct_idxs) if len(correct_idxs) > 0
                                     else len(idxs)),
                            replace=False)
        for k in chosen:
            beat = Xnp[k].astype(float)
            spk = delta_encode_numpy(beat, theta=theta)  # (T, 2)
            x_in = torch.from_numpy(beat).float().unsqueeze(0).to(dev)
            r_in = R[k:k+1].to(dev)
            sp = delta_encode_batch(x_in, theta=theta)
            with torch.no_grad():
                logits, tr = model(sp, rr=r_in, return_traces=True)
                prob = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
            # Compress conv-layer spike rasters to integer events for the JS
            def raster(t):
                # t: (1, C, T') -> list of [neuron_idx, time_step]
                a = t[0].cpu().numpy()
                C, Tp = a.shape
                events = np.argwhere(a > 0).astype(int)  # (N, 2): [channel, time]
                return events.tolist()
            examples.append({
                "true": cls,
                "true_name": CLASS_NAMES[cls],
                "pred": CLASSES[int(preds[k])],
                "pred_name": CLASS_NAMES[CLASSES[int(preds[k])]],
                "probs": dict(zip(CLASSES, prob)),
                "record_id": int(recnp[k]),
                "ecg": beat.tolist(),
                "input_spikes": {
                    "on":  np.where(spk[:, 0] > 0)[0].tolist(),
                    "off": np.where(spk[:, 1] > 0)[0].tolist(),
                },
                "raster_l1": raster(tr["s1"]),  # (channel, time)
                "raster_l2": raster(tr["s2"]),
                "raster_l3": raster(tr["s3"]),
                "raster_l4": raster(tr["s4"]),  # (channel, dense-step)
                "vout":   tr["vout"][0].cpu().numpy().tolist(),  # (5, Td)
            })
    with open(os.path.join(WEB_DATA, "inference.json"), "w") as fp:
        json.dump({"examples": examples, "classes": CLASSES,
                   "class_names": CLASS_NAMES}, fp)
    print(f"Wrote {len(examples)} demo examples → web/data/inference.json")

    # ---- Strip ECG: 10s window from DS2 for the streaming demo -------
    # We pick a record with rich pathology (record 200 has many V beats).
    # To keep the demo interesting, pick a contiguous window that contains
    # at least one V or S beat — slide along record 200 until we find one.
    rec200 = np.where(recnp == 200)[0]
    window = 40
    chosen_start = rec200[0]
    for i in range(len(rec200) - window):
        seg = ynp[rec200[i:i+window]]
        # require ≥2 V beats AND ≥1 S beat ideally; relax progressively
        n_v = int((seg == CLASSES.index("V")).sum())
        n_s = int((seg == CLASSES.index("S")).sum())
        if n_v >= 2 and n_s >= 1:
            chosen_start = i
            break
    else:
        for i in range(len(rec200) - window):
            seg = ynp[rec200[i:i+window]]
            if int((seg == CLASSES.index("V")).sum()) >= 3:
                chosen_start = i; break
    strip_idx = rec200[chosen_start:chosen_start + window]
    strip = []
    for k in strip_idx:
        beat = Xnp[k].astype(float)
        spk = delta_encode_numpy(beat, theta=theta)
        sp = delta_encode_batch(torch.from_numpy(beat).float().unsqueeze(0).to(dev),
                                theta=theta)
        r_in = R[k:k+1].to(dev)
        with torch.no_grad():
            logits, tr = model(sp, rr=r_in, return_traces=True)
            prob = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
        strip.append({
            "true": CLASSES[int(ynp[k])],
            "pred": CLASSES[int(preds[k])],
            "probs": dict(zip(CLASSES, prob)),
            "ecg": beat.tolist(),
            "input_spikes": {
                "on":  np.where(spk[:, 0] > 0)[0].tolist(),
                "off": np.where(spk[:, 1] > 0)[0].tolist(),
            },
            "raster_l3": np.argwhere(tr["s3"][0].cpu().numpy() > 0).astype(int).tolist(),
            "raster_l4": np.argwhere(tr["s4"][0].cpu().numpy() > 0).astype(int).tolist(),
        })
    with open(os.path.join(WEB_DATA, "strip.json"), "w") as fp:
        json.dump({"record": 200, "beats": strip, "classes": CLASSES,
                   "class_names": CLASS_NAMES}, fp)
    print(f"Wrote strip (record 200, {len(strip)} beats) → web/data/strip.json")

    # ---- Headline summary --------------------------------------------------
    summary = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n_test_beats": int(len(ynp)),
        "n_test_records": int(len(np.unique(recnp))),
        "params": sum(p.numel() for p in model.parameters()),
        "mean_input_spike_rate": spike_means["input"],
        "mean_hidden_spike_rate": float(np.mean([spike_means[k] for k in ("s1","s2","s3","s4")])),
        "energy_ratio_ann_over_snn": sparsity["energy_ratio_ann_over_snn"],
        "per_class_f1": {c: float(f[i]) for i, c in enumerate(CLASSES)},
        "per_class_support": {c: int(sup[i]) for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm,
    }
    with open(os.path.join(WEB_DATA, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.15)
    args = ap.parse_args()
    main(theta=args.theta)
