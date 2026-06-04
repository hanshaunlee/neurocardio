"""
Modal app for NeuroCardio.

One app, several functions:

  data_prep              one-shot download + cache of PTB-XL (~1.8 GB) into
                          the persistent Volume
  train                  long-running training entrypoint with wall-clock budget;
                          designed to be launched detached with
                            modal run --detach modal_app/app.py::train --wallclock-hours 14
  fetch_run              copy a finished `runs/<name>/` from the Volume to local
  web                    public ASGI web app (FastAPI) serving the demo UI and
                          a JSON inference endpoint backed by the latest
                          best.pt checkpoint in the Volume
"""
import io
import json
import os
import shutil
from pathlib import Path
from typing import Optional, List

import modal


# ---------------------------------------------------------------------------
# Image and resources
# ---------------------------------------------------------------------------
APP_NAME = "neurocardio"
VOLUME_NAME = "neurocardio-vol"
RUN_DIR_DEFAULT = "main"
LABEL_SET = "diagnostic_superclass"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip", "awscli")
    .pip_install(
        "torch==2.4.1",
        "snntorch==0.9.4",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "scikit-learn==1.5.1",
        "wfdb==4.1.2",
        "tqdm==4.66.5",
        "fastapi==0.115.0",
        "python-multipart==0.0.10",
        "pydantic==2.9.2",
    )
    # the neurocardio package lives next to modal_app/
    .add_local_python_source("neurocardio")
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOL_MOUNT = "/vol"

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Data preparation (one-shot)
# ---------------------------------------------------------------------------
# PhysioNet's PTB-XL is mirrored on the AWS Open Data registry, which gives
# us ~100× the throughput of physionet.org when downloading from a cloud GPU.
PTBXL_S3_PREFIX = "s3://physionet-open/ptb-xl/1.0.3/"


@app.function(
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60,  # up to 1 hour for the download + cache build
    cpu=4.0,
    memory=8192,
)
def data_prep(force: bool = False, label_set: str = LABEL_SET):
    """Sync PTB-XL 1.0.3 from the AWS Open Data mirror into /vol/ptbxl/ and
    build the training cache. Uses `aws s3 sync --no-sign-request` for
    parallelized, resumable transfer (~10× faster than serial HTTP from
    physionet.org)."""
    import subprocess as _sp

    root = Path(VOL_MOUNT)
    raw = root / "ptbxl"
    cache = root / "ptbxl_cache"
    if cache.exists() and not force and (cache / "labels.csv").exists():
        print(f"cache already exists at {cache} — skipping (use force=True to rebuild)",
              flush=True)
        return

    raw.mkdir(parents=True, exist_ok=True)

    # We download into /tmp/ptbxl_dl (fast local disk) and only move
    # finished files into the Volume to avoid burning Volume IOPS on every
    # ~24 KB file.
    if not (raw / "ptbxl_database.csv").exists():
        tmp = Path("/tmp/ptbxl_dl")
        tmp.mkdir(parents=True, exist_ok=True)
        print(f"syncing {PTBXL_S3_PREFIX} → {tmp} (no AWS credentials needed)",
              flush=True)
        # --only-show-errors keeps stdout quiet for the per-file noise; sync
        # is already parallel by default (10 concurrent transfers).
        # Skip records500/ (8 GB of 500 Hz signals we don't use), boost
        # concurrency well above the aws-cli default of 10 for many small
        # files, and turn down per-file chatter so the log is readable.
        _sp.run(["aws", "configure", "set",
                  "default.s3.max_concurrent_requests", "100"], check=True)
        _sp.run(["aws", "configure", "set",
                  "default.s3.max_queue_size", "10000"], check=True)
        r = _sp.run([
            "aws", "s3", "sync", PTBXL_S3_PREFIX, str(tmp),
            "--no-sign-request",
            "--only-show-errors",
            "--exclude", "records500/*",
        ], check=False)
        if r.returncode != 0:
            raise RuntimeError(f"aws s3 sync failed (rc={r.returncode})")
        # Sanity check
        n = sum(1 for _ in tmp.rglob("*.dat"))
        print(f"downloaded {n} .dat files; moving into Volume", flush=True)
        if n < 20000:
            raise RuntimeError(f"too few files downloaded ({n}); aborting")
        # Move into volume (rename is fast, same fs)
        import shutil as _sh
        for entry in tmp.iterdir():
            target = raw / entry.name
            if target.exists():
                if target.is_dir():
                    _sh.rmtree(target)
                else:
                    target.unlink()
            _sh.move(str(entry), str(target))
        tmp.rmdir() if not any(tmp.iterdir()) else None
        vol.commit()
        print("raw PTB-XL committed to Volume.", flush=True)

    print(f"building cache (label_set={label_set}) at {cache}", flush=True)
    from neurocardio.data import PTBXLConfig, build_cache
    cfg = PTBXLConfig(root=raw, label_set=label_set, sampling_rate=100)
    build_cache(cfg, cache, verbose=True)
    vol.commit()
    print("data_prep done.", flush=True)


# ---------------------------------------------------------------------------
# Training (long-running)
# ---------------------------------------------------------------------------
@app.function(
    volumes={VOL_MOUNT: vol},
    gpu="A10G",                 # change to "A100" or "H100" when budget allows
    cpu=8.0,
    memory=32 * 1024,
    timeout=60 * 60 * 16,       # 16 hours hard ceiling for the 14h budget
)
def train(
    run_name: str = RUN_DIR_DEFAULT,
    epochs: int = 200,
    batch_size: int = 96,
    lr: float = 1e-3,
    wallclock_hours: float = 14.0,
    label_set: str = LABEL_SET,
    num_workers: int = 4,
    max_steps_per_epoch: int = 0,
    resume: bool = True,
    model: str = "snn",
):
    """Run training on a Modal GPU instance against the cached dataset."""
    cache = Path(VOL_MOUNT) / "ptbxl_cache"
    output = Path(VOL_MOUNT) / "runs" / run_name
    output.mkdir(parents=True, exist_ok=True)
    print(f"cache:  {cache}")
    print(f"output: {output}")

    from neurocardio.train import TrainConfig, main as train_main
    cfg = TrainConfig(
        cache_dir=str(cache),
        output_dir=str(output),
        label_set=label_set,
        model_name=model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        wallclock_hours=wallclock_hours,
        max_steps_per_epoch=max_steps_per_epoch,
        resume=resume,
        device="cuda",
    )

    # Commit volume periodically so progress survives preemption
    import threading
    import time
    stop_evt = threading.Event()

    def _periodic_commit():
        while not stop_evt.wait(timeout=600):  # every 10 min
            try:
                vol.commit()
                print("[vol] committed")
            except Exception as e:
                print(f"[vol] commit failed: {e}")
    t = threading.Thread(target=_periodic_commit, daemon=True)
    t.start()

    try:
        train_main(cfg)
    finally:
        stop_evt.set()
        vol.commit()
        print("training entrypoint exiting; volume committed.")


# ---------------------------------------------------------------------------
# Fetch a finished run back to local disk (for archiving / inspection)
# ---------------------------------------------------------------------------
@app.function(volumes={VOL_MOUNT: vol}, timeout=600)
def list_runs():
    runs_root = Path(VOL_MOUNT) / "runs"
    if not runs_root.exists():
        return []
    out = []
    for d in sorted(runs_root.iterdir()):
        files = list(d.iterdir()) if d.is_dir() else []
        out.append({
            "name": d.name,
            "n_files": len(files),
            "files": [f.name for f in files],
        })
    return out


@app.function(volumes={VOL_MOUNT: vol}, gpu="T4", timeout=60 * 60, cpu=4.0,
              memory=16 * 1024)
def compare_models(run_names: str = ""):
    """Load best.pt from each run, compute test-set AUROC/AUPRC + analytical
    MACs + measured spike rates + batch-1 latency. Writes a
    /vol/comparison.json that the web app reads."""
    import json, time, numpy as np, torch
    from torch.utils.data import DataLoader
    from neurocardio.dataset import PTBXLCache
    from neurocardio.factory import make_model
    from neurocardio.train import multilabel_metrics
    from neurocardio.compute import (cost_cnn1d, cost_snn, time_inference,
                                      calibrate_spike_rates,
                                      ENERGY_PJ_PER_MAC_FP32,
                                      ENERGY_PJ_PER_SOP)

    cache = Path(VOL_MOUNT) / "ptbxl_cache"
    runs_root = Path(VOL_MOUNT) / "runs"
    if not run_names:
        names = sorted([d.name for d in runs_root.iterdir()
                        if (d / "best.pt").exists()])
    else:
        names = [n.strip() for n in run_names.split(",") if n.strip()]
    if not names:
        raise RuntimeError("no runs with best.pt found")
    print(f"comparing: {names}", flush=True)
    run_names = names

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = PTBXLCache(cache, fold_subset=[10])
    loader = DataLoader(test, batch_size=64, num_workers=4, pin_memory=True)
    val = PTBXLCache(cache, fold_subset=[9])
    val_loader = DataLoader(val, batch_size=32, num_workers=2, pin_memory=True)

    rows = []
    for run_name in run_names:
        ckpt_path = runs_root / run_name / "best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {})
        labels = ckpt.get("label_columns") or ["NORM","MI","STTC","CD","HYP"]
        model_name = cfg.get("model_name") or "snn"
        print(f"\n=== {run_name}  ({model_name}) ===", flush=True)

        model = make_model(model_name, num_classes=len(labels))
        state = ckpt.get("ema") or ckpt.get("state_dict")
        model.load_state_dict(state)
        model.to(dev).eval()

        # Inference on test fold (chunked, full pass)
        all_y, all_p = [], []
        t0 = time.time()
        with torch.no_grad():
            for i, (x, y, _) in enumerate(loader):
                x = x.to(dev); y = y.to(dev)
                logits = model(x)
                all_y.append(y.cpu().numpy())
                all_p.append(torch.sigmoid(logits).cpu().numpy())
        infer_dt_total = time.time() - t0
        y_true = np.concatenate(all_y); y_score = np.concatenate(all_p)
        metrics = multilabel_metrics(y_true, y_score, labels)
        print(f"  test macro-AUROC={metrics['macro_auroc']:.4f}  "
              f"macro-AUPRC={metrics['macro_auprc']:.4f}", flush=True)

        # Spike-rate calibration (SNN only)
        spike_rates = {}
        if model_name == "snn":
            spike_rates = calibrate_spike_rates(model, val_loader, dev,
                                                  n_batches=8)
            print(f"  spike rates: " + ", ".join(
                f"{k}={v:.3f}" for k, v in spike_rates.items()), flush=True)
            cost = cost_snn(model, spike_rates, T_in=1000)
        else:
            cost = cost_cnn1d(model, T_in=1000)

        # Batch-1 latency
        lat = time_inference(model, dev, T_in=1000, n_warmup=5, n_iter=20)
        cost.latency_ms = lat
        cost.latency_device = "cuda" if dev.type == "cuda" else dev.type
        print(f"  latency: {lat:.2f} ms/inf (B=1, {dev.type})", flush=True)
        print(f"  params={cost.total_params:,}  MACs={cost.total_macs:,}  SOPs={cost.total_sops:,}",
              flush=True)

        # ROC curves (downsampled to ≤200 pts per class for JSON size)
        from sklearn.metrics import roc_curve, precision_recall_curve
        roc_curves, pr_curves = {}, {}
        for j, lbl in enumerate(labels):
            col = y_true[:, j].astype(int)
            if col.sum() == 0 or col.sum() == len(col):
                continue
            fpr, tpr, _ = roc_curve(col, y_score[:, j])
            prec, rec, _ = precision_recall_curve(col, y_score[:, j])
            # Downsample to ≤200 evenly-spaced points
            def _ds(arr, n=200):
                if len(arr) <= n: return arr.tolist()
                idx = np.round(np.linspace(0, len(arr)-1, n)).astype(int)
                return arr[idx].round(4).tolist()
            roc_curves[lbl] = {"fpr": _ds(fpr), "tpr": _ds(tpr)}
            pr_curves[lbl]  = {"precision": _ds(prec), "recall": _ds(rec)}
        print(f"  ROC + PR curves computed for {len(roc_curves)} classes", flush=True)

        rows.append({
            "run_name": run_name,
            "model": model_name,
            "params": cost.total_params,
            "macs":   cost.total_macs,
            "sops":   cost.total_sops,
            "spike_rate_mean": cost.spike_rate_mean,
            "spike_rates": spike_rates,
            "latency_ms": cost.latency_ms,
            "latency_device": cost.latency_device,
            "energy_pj_dense":        cost.energy_pj_dense,
            "energy_pj_neuromorphic": cost.energy_pj_neuromorphic,
            "macro_auroc":   metrics["macro_auroc"],
            "macro_auprc":   metrics["macro_auprc"],
            "per_label":     metrics["per_label"],
            "labels":        labels,
            "roc_curves":    roc_curves,
            "pr_curves":     pr_curves,
            "test_inference_seconds": infer_dt_total,
        })

    out = {
        "comparison_generated_at": time.time(),
        "rows": rows,
        "constants": {
            "ENERGY_PJ_PER_MAC_FP32": ENERGY_PJ_PER_MAC_FP32,
            "ENERGY_PJ_PER_SOP":      ENERGY_PJ_PER_SOP,
        },
    }
    out_path = Path(VOL_MOUNT) / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print(f"\n→ wrote {out_path}", flush=True)
    return rows


@app.function(volumes={VOL_MOUNT: vol}, gpu="T4", timeout=60 * 30, cpu=4.0,
              memory=8192)
def evaluate_test(run_name: str = RUN_DIR_DEFAULT):
    """Run the best checkpoint on the fold-10 test set and persist a
    per-recording prediction matrix + macro/per-label AUROC + AUPRC."""
    import json, numpy as np, torch
    from torch.utils.data import DataLoader
    from neurocardio.dataset import PTBXLCache
    from neurocardio.model import NeuroCardio
    from neurocardio.train import multilabel_metrics

    out_dir = Path(VOL_MOUNT) / "runs" / run_name
    cache = Path(VOL_MOUNT) / "ptbxl_cache"
    best = out_dir / "best.pt"
    if not best.exists():
        raise RuntimeError(f"no best.pt at {best}")
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    labels = ckpt.get("label_columns") or ["NORM","MI","STTC","CD","HYP"]
    model = NeuroCardio(num_classes=len(labels))
    state = ckpt.get("ema") or ckpt.get("state_dict")
    model.load_state_dict(state)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    test = PTBXLCache(cache, fold_subset=[10])
    loader = DataLoader(test, batch_size=64, num_workers=4, pin_memory=True)
    all_y, all_p = [], []
    print(f"evaluating {len(test)} test recordings", flush=True)
    with torch.no_grad():
        for i, (x, y, _) in enumerate(loader):
            x = x.to(dev); y = y.to(dev)
            logits = model(x)
            all_y.append(y.cpu().numpy())
            all_p.append(torch.sigmoid(logits).cpu().numpy())
            if (i+1) % 20 == 0:
                print(f"  {(i+1) * loader.batch_size} done", flush=True)
    y_true = np.concatenate(all_y); y_score = np.concatenate(all_p)
    m = multilabel_metrics(y_true, y_score, labels)
    print(json.dumps({"macro_auroc": m["macro_auroc"],
                       "macro_auprc": m["macro_auprc"]}, indent=2), flush=True)
    out_path = out_dir / "test_predictions.json"
    with open(out_path, "w") as f:
        json.dump({"labels": labels, "metrics": m,
                   "ecg_ids": test.meta["ecg_id"].astype(int).tolist(),
                   "true": y_true.astype(int).tolist(),
                   "scores": y_score.astype(float).round(4).tolist()},
                  f)
    vol.commit()
    print(f"wrote {out_path}", flush=True)
    return m


# ---------------------------------------------------------------------------
# Web app (FastAPI ASGI)
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).parent.parent / "web"

web_image = image.add_local_dir(str(WEB_DIR), remote_path="/web")


@app.function(
    image=web_image,
    volumes={VOL_MOUNT: vol},
    gpu="T4",                   # cheap GPU for inference
    cpu=2.0,
    memory=8192,
    min_containers=0,
    scaledown_window=60 * 5,
    max_containers=1,
)
@modal.asgi_app()
def web():
    """Public web demo. Serves the static frontend and a /infer endpoint
    that runs the latest best checkpoint on a posted ECG."""
    from fastapi import FastAPI, HTTPException, UploadFile, File, Query
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from neurocardio.infer import CardioInferencer
    from neurocardio.factory import make_model
    from neurocardio.data import (
        PTBXLConfig, load_metadata, read_signal, normalize)
    import numpy as np

    api = FastAPI(title="NeuroCardio")

    # -------- lazy model load (warm on first request) --------
    _state: dict = {"infer": None, "ckpt_mtime": 0.0, "ckpt_path": None}

    # Baseline models (CNN1D / ResNet1D) for live side-by-side comparison.
    # Each entry: key -> {"model", "labels", "mtime", "dev", "name"}.
    _baselines: dict = {}
    # Which run directory holds each baseline checkpoint.
    BASELINE_RUNS = {"cnn": "cnn_baseline", "resnet": "resnet_baseline"}

    def _get_baseline(key: str):
        """Lazily load (and hot-reload) a baseline checkpoint by key."""
        import torch
        run_name = BASELINE_RUNS.get(key)
        if not run_name:
            return None
        try: vol.reload()
        except Exception: pass
        ckpt_path = Path(VOL_MOUNT) / "runs" / run_name / "best.pt"
        if not ckpt_path.exists():
            return None
        m = ckpt_path.stat().st_mtime
        cached = _baselines.get(key)
        if cached is None or cached["mtime"] < m:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            labels = ckpt.get("label_columns") or ["NORM", "MI", "STTC", "CD", "HYP"]
            cfg = ckpt.get("cfg", {})
            name = cfg.get("model_name") or key
            model = make_model(name, num_classes=len(labels))
            state = ckpt.get("ema") or ckpt.get("state_dict")
            model.load_state_dict(state)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(dev).eval()
            _baselines[key] = {"model": model, "labels": labels,
                               "mtime": m, "dev": dev, "name": name}
            print(f"loaded baseline {key} ({name}) from {ckpt_path}")
        return _baselines[key]

    def _baseline_probs(sig: "np.ndarray") -> dict:
        """Run every available baseline on a (12, T) normalized signal and
        return {key: {class: prob}}. Same array the SNN sees → fair compare."""
        import torch
        out: dict = {}
        for key in BASELINE_RUNS:
            try:
                b = _get_baseline(key)
                if b is None:
                    continue
                x = torch.from_numpy(np.ascontiguousarray(sig)).float().unsqueeze(0).to(b["dev"])
                with torch.no_grad():
                    logits = b["model"](x)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                out[key] = {c: float(probs[i]) for i, c in enumerate(b["labels"])}
            except Exception as e:
                # A baseline must never break the core SNN inference path.
                print(f"baseline {key} failed: {e}")
        return out

    def _resolve_ckpt():
        # Make sure we see writes from other Modal containers (the training
        # function commits to the volume every ~10 min). reload() is cheap
        # when there is nothing new.
        try: vol.reload()
        except Exception: pass
        runs = Path(VOL_MOUNT) / "runs"
        if not runs.exists(): return None
        candidates = sorted(runs.glob("*/best.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def _get_infer():
        p = _resolve_ckpt()
        if p is None:
            return None
        m = p.stat().st_mtime
        if _state["infer"] is None or _state["ckpt_mtime"] < m:
            print(f"loading checkpoint {p} (mtime {m})")
            _state["infer"] = CardioInferencer(p, label_set=LABEL_SET)
            _state["ckpt_mtime"] = m
            _state["ckpt_path"] = str(p)
        return _state["infer"]

    # -------- routes --------
    def _asset_version() -> str:
        """Short content hash of the CSS+JS so the asset URLs change only when
        the files do. Defeats browser heuristic caching (StaticFiles serves a
        1970 last-modified, which makes browsers cache for years)."""
        import hashlib
        h = hashlib.md5()
        try:
            for f in ("style.css", "app.js"):
                h.update(Path(f"/web/{f}").read_bytes())
            return h.hexdigest()[:8]
        except Exception:
            return "dev"

    @api.get("/")
    def index():
        from fastapi.responses import HTMLResponse
        html = Path("/web/index.html").read_text()
        ver = _asset_version()
        html = html.replace("/static/style.css", f"/static/style.css?v={ver}")
        html = html.replace("/static/app.js", f"/static/app.js?v={ver}")
        # Always revalidate the HTML so a new deploy's asset hashes are picked up.
        return HTMLResponse(html, headers={"Cache-Control": "no-cache, must-revalidate"})

    api.mount("/static", StaticFiles(directory="/web"), name="static")

    @api.get("/warm")
    def warm():
        """Eagerly load the model so the first /infer_* call is hot."""
        m = _get_infer()
        return {"loaded": m is not None}

    @api.get("/history")
    def history(run_name: str = RUN_DIR_DEFAULT):
        """Return the JSON training history for a given run_name."""
        try: vol.reload()
        except Exception: pass
        p = Path(VOL_MOUNT) / "runs" / run_name / "history.json"
        if not p.exists():
            raise HTTPException(404, f"no history.json for run {run_name}")
        with open(p) as f:
            return JSONResponse(json.load(f))

    @api.get("/comparison")
    def comparison():
        try: vol.reload()
        except Exception: pass
        p = Path(VOL_MOUNT) / "comparison.json"
        if not p.exists():
            raise HTTPException(404, "no comparison.json yet — "
                                       "run `compare_models` first")
        with open(p) as f:
            return JSONResponse(json.load(f))

    @api.get("/status")
    def status():
        p = _resolve_ckpt()
        info = {
            "checkpoint": str(p) if p else None,
            "checkpoint_mtime": p.stat().st_mtime if p else None,
        }
        if p:
            try:
                import torch
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                info["epoch"] = ckpt.get("epoch")
                info["best_macro_auroc"] = ckpt.get("best_macro_auroc")
                hist = ckpt.get("history", [])
                if hist:
                    info["latest"] = hist[-1]
            except Exception as e:
                info["error"] = str(e)
        # list available examples
        cache = Path(VOL_MOUNT) / "ptbxl_cache"
        info["cache_ready"] = (cache / "labels.csv").exists()
        return info

    @api.get("/examples")
    def examples(n: int = 12, fold: int = 10, label_set: str = LABEL_SET):
        """Return a small list of test-set ECG IDs with metadata to populate
        the demo dropdown."""
        cache = Path(VOL_MOUNT) / "ptbxl_cache"
        if not (cache / "labels.csv").exists():
            raise HTTPException(404, "cache not built")
        import pandas as pd
        meta = pd.read_csv(cache / "labels.csv")
        with open(cache / "label_columns.txt") as f:
            labels = [l.strip() for l in f if l.strip()]
        test = meta[meta.strat_fold == fold]
        # Take a diverse sample: a few of each label combination
        out = []
        for label in labels:
            sub = test[test[label] == 1].head(max(1, n // len(labels)))
            for _, row in sub.iterrows():
                out.append({
                    "ecg_id": int(row["ecg_id"]),
                    "labels": [c for c in labels if int(row[c]) == 1],
                })
        # de-dup by ecg_id
        seen = set(); deduped = []
        for o in out:
            if o["ecg_id"] in seen: continue
            seen.add(o["ecg_id"]); deduped.append(o)
        return {"labels": labels, "examples": deduped[:n]}

    @api.get("/infer_example")
    def infer_example(ecg_id: int):
        """Run the model on a single PTB-XL recording (by ecg_id)."""
        infer = _get_infer()
        if infer is None:
            raise HTTPException(503, "no checkpoint available yet")
        cache = Path(VOL_MOUNT) / "ptbxl_cache"
        signals = np.load(cache / "signals_100.npy", mmap_mode="r")
        import pandas as pd
        meta = pd.read_csv(cache / "labels.csv")
        idx_arr = meta.index[meta["ecg_id"] == ecg_id].tolist()
        if not idx_arr:
            raise HTTPException(404, f"ecg_id {ecg_id} not in cache")
        idx = idx_arr[0]
        sig = np.asarray(signals[idx])  # (12, T)
        with open(cache / "label_columns.txt") as f:
            label_cols = [l.strip() for l in f if l.strip()]
        true_labels = [c for c in label_cols if int(meta.iloc[idx][c]) == 1]
        sig = sig.astype(np.float32)
        result = infer.infer(sig)
        # Side-by-side: same recording through the trained baselines.
        result["compare_probs"] = {"snn": result["probs"], **_baseline_probs(sig)}
        result["ecg_id"] = int(ecg_id)
        result["true_labels"] = true_labels
        result["ecg"] = sig.tolist()
        result["fs"] = 100
        return JSONResponse(result)

    @api.post("/infer_upload")
    async def infer_upload(file: UploadFile = File(...)):
        """Run the model on an uploaded 12-lead ECG (numpy .npy of shape (12,T))."""
        infer = _get_infer()
        if infer is None:
            raise HTTPException(503, "no checkpoint available yet")
        data = await file.read()
        try:
            sig = np.load(io.BytesIO(data))
        except Exception as e:
            raise HTTPException(400, f"invalid .npy file: {e}")
        if sig.ndim != 2 or sig.shape[0] != 12:
            raise HTTPException(400, f"expected shape (12, T), got {sig.shape}")
        sig = sig[:, :1000].astype(np.float32)
        if sig.shape[1] < 1000:
            sig = np.pad(sig, ((0, 0), (0, 1000 - sig.shape[1])))
        sig = (sig - np.median(sig, axis=1, keepdims=True)) / \
              (1.4826 * np.median(np.abs(sig - np.median(sig, axis=1, keepdims=True)),
                                  axis=1, keepdims=True) + 1e-6)
        result = infer.infer(sig)
        result["compare_probs"] = {"snn": result["probs"], **_baseline_probs(sig)}
        result["ecg"] = sig.tolist()
        result["fs"] = 100
        return JSONResponse(result)

    return api


# ---------------------------------------------------------------------------
# Local entrypoint (for inspection from the host)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def status():
    print("Available functions: data_prep, train, list_runs, web")
    print("Volume:", VOLUME_NAME)
