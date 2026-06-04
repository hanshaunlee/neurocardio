"""
Preprocess MIT-BIH Arrhythmia data into per-beat segments using the
inter-patient DS1/DS2 split (de Chazal et al., 2004) and AAMI EC57
5-class superclass labels (N, S, V, F, Q).

We keep the inter-patient split because it is the only honest test of
generalization in this dataset — intra-patient splits leak morphology and
inflate accuracy.
"""
import os
import numpy as np
import wfdb
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mitdb")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Standard inter-patient split from de Chazal et al. 2004
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
       "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
       "233", "234"]

# AAMI EC57 superclass mapping
AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

FS = 360  # MIT-BIH sampling rate (Hz)
WIN_BEFORE = 90   # 250 ms before R-peak
WIN_AFTER = 170   # 472 ms after R-peak  (total 260 samples ≈ 722 ms)
WIN_LEN = WIN_BEFORE + WIN_AFTER


def load_record(rec_id):
    rec = wfdb.rdrecord(os.path.join(DATA_DIR, rec_id))
    ann = wfdb.rdann(os.path.join(DATA_DIR, rec_id), "atr")
    # MLII is channel 0 on most records; fall back to channel 0 if absent
    sig_names = [s.upper() for s in rec.sig_name]
    ch = sig_names.index("MLII") if "MLII" in sig_names else 0
    signal = rec.p_signal[:, ch].astype(np.float32)
    return signal, ann


def normalize(sig):
    # z-score per record (median/MAD is robust to spikes)
    med = np.median(sig)
    mad = np.median(np.abs(sig - med)) + 1e-6
    return (sig - med) / (1.4826 * mad)


def segment_record(rec_id):
    sig, ann = load_record(rec_id)
    sig = normalize(sig)

    # Precompute R-peak indices of *all* AAMI-mappable beats for RR features.
    keep = [(s, sy) for s, sy in zip(ann.sample, ann.symbol) if sy in AAMI]
    samps = [s for s, _ in keep]

    beats, labels, rrs = [], [], []
    rolling = []
    for i, (samp, sym) in enumerate(keep):
        if samp - WIN_BEFORE < 0 or samp + WIN_AFTER > len(sig):
            continue
        beat = sig[samp - WIN_BEFORE: samp + WIN_AFTER]
        # R-R features in seconds
        pre_rr  = (samp - samps[i - 1]) / FS if i > 0                else 0.8
        post_rr = (samps[i + 1] - samp) / FS if i + 1 < len(samps)   else 0.8
        rolling.append(pre_rr if i > 0 else 0.8)
        if len(rolling) > 10: rolling.pop(0)
        avg_rr = float(np.mean(rolling))
        local_ratio = pre_rr / max(avg_rr, 1e-6)  # premature beats < 1, escape > 1
        beats.append(beat)
        labels.append(CLASS_IDX[AAMI[sym]])
        rrs.append([pre_rr, post_rr, avg_rr, local_ratio])
    return (np.asarray(beats, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
            np.asarray(rrs, dtype=np.float32))


def build_split(records):
    Xs, ys, ids, Rs = [], [], [], []
    for rec_id in records:
        X, y, R = segment_record(rec_id)
        Xs.append(X)
        ys.append(y)
        Rs.append(R)
        ids.append(np.full(len(y), int(rec_id), dtype=np.int32))
        print(f"  {rec_id}: {len(y):5d} beats  {dict(Counter([CLASSES[i] for i in y]))}")
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(ids), np.concatenate(Rs, axis=0))


def main():
    print(f"DS1 (train, {len(DS1)} records)")
    Xtr, ytr, idtr, Rtr = build_split(DS1)
    print(f"DS2 (test, {len(DS2)} records)")
    Xte, yte, idte, Rte = build_split(DS2)

    print("\nTrain class counts:", dict(zip(CLASSES, np.bincount(ytr, minlength=5).tolist())))
    print("Test class counts :", dict(zip(CLASSES, np.bincount(yte, minlength=5).tolist())))

    np.savez_compressed(os.path.join(OUT_DIR, "ds1.npz"), X=Xtr, y=ytr, rec=idtr, R=Rtr)
    np.savez_compressed(os.path.join(OUT_DIR, "ds2.npz"), X=Xte, y=yte, rec=idte, R=Rte)
    print(f"\nSaved to {OUT_DIR}")
    print(f"Train shape: {Xtr.shape}, RR shape: {Rtr.shape}")
    print(f"Test  shape: {Xte.shape}, RR shape: {Rte.shape}")


if __name__ == "__main__":
    main()
