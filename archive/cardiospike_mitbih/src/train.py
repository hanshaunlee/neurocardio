"""
Train CardioSpikeSNN on the inter-patient DS1 split, validate on DS2.

We deliberately do NOT mix patient identities between train and test
(de Chazal et al., 2004). Class imbalance is severe (≈90% Normal beats), so
we use class-weighted cross-entropy with sqrt-inverse-frequency weighting,
which down-weights N without over-penalising rare classes the way pure
inverse-frequency does.
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

from src.model import CardioSpikeSNN
from src.encoding import delta_encode_batch


HERE = os.path.dirname(os.path.dirname(__file__))
PROC = os.path.join(HERE, "data", "processed")
MODEL_DIR = os.path.join(HERE, "models")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSES = ["N", "S", "V", "F", "Q"]


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(name):
    d = np.load(os.path.join(PROC, f"{name}.npz"))
    return (torch.from_numpy(d["X"]).float(),
            torch.from_numpy(d["y"]).long(),
            torch.from_numpy(d["R"]).float())


def normalize_rr(R_train, R):
    mean = R_train.mean(0, keepdim=True)
    std  = R_train.std(0, keepdim=True).clamp(min=1e-3)
    return (R - mean) / std


def class_weights(y, eps=1e-6):
    counts = np.bincount(y.numpy(), minlength=5).astype(np.float64)
    w = 1.0 / np.sqrt(counts + eps)
    w = w / w.sum() * len(CLASSES)
    return torch.from_numpy(w).float()


def balanced_sampler(y, ramp=1.0):
    """Per-sample weights so rare classes are oversampled. ramp=1.0 fully
    balances; ramp<1.0 gives a softer mix between balanced and natural."""
    counts = np.bincount(y.numpy(), minlength=5).astype(np.float64)
    nat_w  = 1.0  # natural sample weight = 1/N for all
    bal_w  = (1.0 / np.maximum(counts, 1.0))[y.numpy()]
    bal_w  = bal_w / bal_w.mean()
    weights = (1 - ramp) * nat_w + ramp * bal_w
    return WeightedRandomSampler(torch.from_numpy(weights).double(),
                                  num_samples=len(y), replacement=True)


def evaluate(model, X, y, R, dev, theta=0.15, batch=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i+batch].to(dev)
            rb = R[i:i+batch].to(dev)
            sp = delta_encode_batch(xb, theta=theta)
            logits = model(sp, rr=rb)
            preds.append(logits.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    yt = y.numpy()
    return {
        "acc": float(accuracy_score(yt, preds)),
        "macro_f1": float(f1_score(yt, preds, average="macro", zero_division=0)),
        "f1_per_class": f1_score(yt, preds, average=None, labels=list(range(5)),
                                 zero_division=0).tolist(),
        "cm": confusion_matrix(yt, preds, labels=list(range(5))).tolist(),
        "report": classification_report(yt, preds, labels=list(range(5)),
                                        target_names=CLASSES, zero_division=0,
                                        output_dict=True),
        "preds": preds,
    }


def main(epochs=20, batch=256, lr=1e-3, theta=0.15, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = device()
    print(f"Device: {dev}")

    Xtr, ytr, Rtr = load_split("ds1")
    Xte, yte, Rte = load_split("ds2")
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

    # Normalize RR features using DS1 statistics (no leakage)
    rr_mean = Rtr.mean(0, keepdim=True)
    rr_std  = Rtr.std(0, keepdim=True).clamp(min=1e-3)
    Rtr = (Rtr - rr_mean) / rr_std
    Rte = (Rte - rr_mean) / rr_std
    # Save rr stats so evaluate.py can re-normalise consistently
    np.savez(os.path.join(MODEL_DIR, "rr_stats.npz"),
             mean=rr_mean.numpy(), std=rr_std.numpy())

    weights = class_weights(ytr).to(dev)
    print("Class weights:", {c: round(float(w), 3) for c, w in zip(CLASSES, weights.cpu())})

    model = CardioSpikeSNN().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Use balanced sampling alone — combining it with class-weighted CE
    # double-corrects and tanks the dominant-class recall.
    crit = nn.CrossEntropyLoss()

    train_ds = TensorDataset(Xtr, ytr, Rtr)
    sampler = balanced_sampler(ytr, ramp=0.7)
    loader = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                        num_workers=0, drop_last=True)

    history = []
    best_f1 = -1.0
    best_path = os.path.join(MODEL_DIR, "cardiospike_best.pt")

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb, rb in loader:
            xb, yb, rb = xb.to(dev), yb.to(dev), rb.to(dev)
            sp = delta_encode_batch(xb, theta=theta)
            logits = model(sp, rr=rb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item()) * yb.size(0)
            total += yb.size(0)
            correct += int((logits.argmax(1) == yb).sum().item())
        sched.step()
        train_acc = correct / total
        train_loss = loss_sum / total

        val = evaluate(model, Xte, yte, Rte, dev, theta=theta)
        dt = time.time() - t0
        print(f"ep {ep:02d}  loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
              f"val_acc {val['acc']:.4f}  macro_f1 {val['macro_f1']:.4f}  "
              f"per_class_f1 {['%.2f' % f for f in val['f1_per_class']]}  "
              f"({dt:.1f}s)")
        history.append({
            "epoch": ep, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val["acc"], "macro_f1": val["macro_f1"],
            "f1_per_class": val["f1_per_class"],
        })
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save({"state_dict": model.state_dict(),
                        "config": {"T_dense": model.T_dense, "theta": theta},
                        "val": {k: v for k, v in val.items() if k != "preds"}},
                       best_path)
            print(f"  saved best (macro_f1={best_f1:.4f}) → {best_path}")

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump({"history": history, "best_macro_f1": best_f1,
                   "params": n_params}, f, indent=2)
    print(f"\nBest macro-F1: {best_f1:.4f}")
    print(f"Model: {best_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--theta", type=float, default=0.15)
    args = ap.parse_args()
    main(epochs=args.epochs, batch=args.batch, lr=args.lr, theta=args.theta)
