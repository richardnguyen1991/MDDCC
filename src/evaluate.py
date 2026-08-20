"""Danh gia cuoi sau epoch 100 - muc 5, 6, 7.G.

Chay nhu mot BUOC RIENG BIET sau khi train xong (muc 4.8, 7.E5): neu buoc nay
loi thi checkpoint van con nguyen tren S3 va chay lai duoc bang make_report.py.
Buoc nay phai IDEMPOTENT (muc 7.E6): seed co dinh, giu nguyen thu tu test,
khong lay mau ngau nhien cho metric chinh.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, log_loss, matthews_corrcoef,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

LOG = logging.getLogger(__name__)

# Table 9 cua bai bao tren CIC-DDoS2019 (binary view)
PAPER_TABLE9 = {"Accuracy": 0.9963, "Precision": 0.9798, "Recall": 0.9871,
                "F1": 0.9834, "FPR": 0.0818}
# Table 10: so sanh voi cac phuong phap khac (accuracy tren CIC-DDoS2019)
PAPER_TABLE10 = {"MDDCC (bai bao)": 0.9963}


# ------------------------------------------------------------------ du doan
def predict_in_chunks(model, dataset, indices, *, batch_size: int,
                      num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Tra ve (y_true int16, y_prob float32) theo DUNG thu tu indices.

    Khong shuffle -> chay lai cho ket qua giong het (muc 7.E6).
    """
    import torch

    model.eval()
    n = int(indices.size)
    y_true = np.empty(n, dtype=np.int16)
    y_prob = np.empty((n, num_classes), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = indices[start:start + batch_size]
            xb, yb = dataset.batch(rows)
            p = model(xb).numpy()
            y_true[start:start + rows.size] = yb.numpy().astype(np.int16)
            y_prob[start:start + rows.size] = p
    return y_true, y_prob


# ------------------------------------------------------------------ metric
def fpr_per_class(cm: np.ndarray) -> np.ndarray:
    """FPR one-vs-rest tung lop: FP / (FP + TN) - muc 7.G4."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(np.where(fp + tn > 0, fp / (fp + tn), 0.0))


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> dict:
    """Muc 7.G2: bao cao du macro-OVR, weighted-OVR va micro.

    Muc 7.G3: bo qua lop khong co mau duong trong test thay vi nem loi/tra NaN.
    """
    present = [c for c in range(num_classes) if (y_true == c).sum() > 0]
    out = {"macro-OVR": float("nan"), "weighted-OVR": float("nan"),
           "micro": float("nan"), "classes_used": len(present),
           "classes_skipped": num_classes - len(present)}
    if len(present) < 2:
        return out

    prob = y_prob[:, present]
    prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
    remap = np.searchsorted(present, y_true)
    try:
        out["macro-OVR"] = float(roc_auc_score(remap, prob, multi_class="ovr",
                                               average="macro", labels=range(len(present))))
        out["weighted-OVR"] = float(roc_auc_score(remap, prob, multi_class="ovr",
                                                  average="weighted",
                                                  labels=range(len(present))))
    except ValueError as exc:
        LOG.warning("roc_auc_score OVR that bai: %s", exc)

    onehot = np.zeros_like(prob)
    onehot[np.arange(remap.size), remap] = 1.0
    try:
        out["micro"] = float(roc_auc_score(onehot.ravel(), prob.ravel()))
    except ValueError as exc:
        LOG.warning("roc_auc_score micro that bai: %s", exc)
    return out


def compute_curves(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str],
                   *, max_points: int = 2000) -> tuple[dict, dict]:
    """ROC va PR one-vs-rest. Giam so diem de CSV/hinh khong phinh vo ich."""
    roc, pr = {}, {}
    n_classes = len(classes)
    onehot = np.zeros((y_true.size, n_classes), dtype=np.int8)
    onehot[np.arange(y_true.size), y_true] = 1

    def thin(*arrays):
        m = arrays[0].size
        if m <= max_points:
            return arrays
        idx = np.unique(np.linspace(0, m - 1, max_points).astype(int))
        return tuple(a[idx] for a in arrays)

    all_fpr = []
    for c, name in enumerate(classes):
        pos = onehot[:, c]
        if pos.sum() == 0 or pos.sum() == pos.size:
            continue                                   # muc 7.G3
        f, t, th = roc_curve(pos, y_prob[:, c])
        auc = float(roc_auc_score(pos, y_prob[:, c]))
        f2, t2, th2 = thin(f, t, np.append(th, th[-1] if th.size else 0)[:f.size])
        roc[name] = {"fpr": f2, "tpr": t2, "threshold": th2, "auc": auc}
        all_fpr.append((f, t))

        p, r, thp = precision_recall_curve(pos, y_prob[:, c])
        ap = float(average_precision_score(pos, y_prob[:, c]))   # muc 7.G1
        r2, p2, th3 = thin(r, p, np.append(thp, 1.0)[:r.size])
        pr[name] = {"recall": r2, "precision": p2, "threshold": th3, "ap": ap}
        del f, t, th, p, r, thp                        # muc 7.B1

    # micro
    f, t, th = roc_curve(onehot.ravel(), y_prob.ravel())
    f2, t2, th2 = thin(f, t, np.append(th, 0)[:f.size])
    roc["micro"] = {"fpr": f2, "tpr": t2, "threshold": th2,
                    "auc": float(roc_auc_score(onehot.ravel(), y_prob.ravel()))}
    p, r, thp = precision_recall_curve(onehot.ravel(), y_prob.ravel())
    r2, p2, th3 = thin(r, p, np.append(thp, 1.0)[:r.size])
    pr["micro"] = {"recall": r2, "precision": p2, "threshold": th3,
                   "ap": float(average_precision_score(onehot.ravel(), y_prob.ravel()))}

    # macro ROC: noi suy tren luoi FPR chung
    if all_fpr:
        grid = np.linspace(0, 1, 1000)
        mean_tpr = np.mean([np.interp(grid, f, t) for f, t in all_fpr], axis=0)
        roc["macro"] = {"fpr": grid, "tpr": mean_tpr,
                        "threshold": np.zeros_like(grid),
                        "auc": float(np.trapezoid(mean_tpr, grid))}
    del onehot
    return roc, pr


@dataclass
class EvalResult:
    y_true: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray
    cm: np.ndarray
    summary: dict
    per_class: list[dict]
    report_text: str
    binary: dict


def evaluate_full(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str],
                  benign_index: int) -> EvalResult:
    """Metric da lop + binary view. Tinh tren TOAN BO test (muc 7.B4)."""
    n_classes = len(classes)
    labels = list(range(n_classes))                    # muc 7.D9
    y_pred = y_prob.argmax(axis=1).astype(np.int16)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    fpr = fpr_per_class(cm)
    auc = safe_roc_auc(y_true, y_prob, n_classes)

    per_class = []
    for c, name in enumerate(classes):
        pos = (y_true == c).astype(np.int8)
        roc_auc = pr_auc = float("nan")
        if 0 < pos.sum() < pos.size:
            roc_auc = float(roc_auc_score(pos, y_prob[:, c]))
            pr_auc = float(average_precision_score(pos, y_prob[:, c]))
        per_class.append({
            "class": name, "support": int(support[c]),
            "precision": float(prec[c]), "recall": float(rec[c]),
            "f1": float(f1[c]), "fpr": float(fpr[c]),
            "roc_auc": roc_auc, "pr_auc": pr_auc,
        })

    present = support > 0
    smallest = min((p for p in per_class if p["support"] > 0),
                   key=lambda p: p["support"], default=None)

    # ---- binary view: BENIGN vs ATTACK (muc 3.E), gop tu du doan da lop
    bt = (y_true != benign_index).astype(np.int8)
    bp = (y_pred != benign_index).astype(np.int8)
    bcm = confusion_matrix(bt, bp, labels=[0, 1])
    tn, fp, fn, tp = bcm.ravel()
    bprec = tp / (tp + fp) if tp + fp else 0.0
    brec = tp / (tp + fn) if tp + fn else 0.0
    binary = {
        "Accuracy": float((bt == bp).mean()),
        "Precision": float(bprec),
        "Recall": float(brec),
        "F1": float(2 * bprec * brec / (bprec + brec)) if bprec + brec else 0.0,
        "FPR": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    try:
        ll = float(log_loss(y_true, y_prob, labels=labels))
    except ValueError as exc:
        LOG.warning("log_loss that bai: %s", exc)
        ll = float("nan")

    summary = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "MacroPrecision": float(prec[present].mean()),
        "MacroRecall": float(rec[present].mean()),
        "MacroF1": float(f1[present].mean()),
        "WeightedF1": float(np.average(f1[present], weights=support[present])),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "MinorityClassF1": float(smallest["f1"]) if smallest else float("nan"),
        "MinorityClassName": smallest["class"] if smallest else "",
        "MinorityClassSupport": int(smallest["support"]) if smallest else 0,
        "MacroFPR": float(fpr[present].mean()),
        "BinaryFPR": binary["FPR"],
        "LogLoss": ll,
        "AUC-ROC_macro-OVR": auc["macro-OVR"],
        "AUC-ROC_weighted-OVR": auc["weighted-OVR"],
        "AUC-ROC_micro": auc["micro"],
        "PR-AUC_macro": float(np.nanmean([p["pr_auc"] for p in per_class])),
        "num_classes": n_classes,
        "n_test_samples": int(y_true.size),
    }

    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=classes, digits=4,
                                   zero_division=0)
    return EvalResult(y_true, y_prob, y_pred, cm, summary, per_class, report, binary)


def paper_comparison_rows(binary: dict) -> list[dict]:
    """paper_comparison.csv - doi chieu binary view voi Table 9 + Table 10."""
    rows = []
    for metric, paper in PAPER_TABLE9.items():
        ours = float(binary[metric])
        note = {
            "Accuracy": "khac tap dac trung (81 vs 48) va thiet bi (CPU vs RTX 3090)",
            "Precision": "khac ty le chia va khong xu ly mat can bang",
            "Recall": "bai bao chu yeu nhi phan; ta gop tu du doan 18 lop",
            "F1": "he qua cua Precision/Recall",
            "FPR": "bai bao ghi nhan 8.18% do mat can bang; ta cung khong xu ly",
        }[metric]
        rows.append({"metric": metric, "ours_binary": round(ours, 6),
                     "paper_table9": paper, "delta": round(ours - paper, 6),
                     "note": note})
    return rows


# ------------------------------------------------------- do hieu nang (muc 6)
def benchmark_inference(model, scaler, geom, sample_raw: np.ndarray, *,
                        batch_sizes=(4096, 1), warmup: int = 50,
                        iters: int = 500) -> dict:
    """Tach rieng t_scale / t_swt / t_forward - diem khac biet cot loi cua MDDCC.

    Chi phi wavelet PHAI duoc boc tach, khong duoc giau trong tong thoi gian.
    """
    import torch

    from .wavelet import transform_batch

    model.eval()
    out = {"warmup_iters": warmup, "measure_iters": iters,
           "torch_num_threads": torch.get_num_threads(),
           "n_parameters": model.n_parameters(),
           "model_size_mb": round(model.size_mb(), 4),
           "dropout_disabled": not model.training,
           "batches": {}}

    for bs in batch_sizes:
        raw = np.ascontiguousarray(sample_raw[:bs] if sample_raw.shape[0] >= bs
                                   else np.resize(sample_raw, (bs, sample_raw.shape[1])))
        t_scale, t_swt, t_fwd = [], [], []

        def one_pass(record: bool):
            t0 = time.perf_counter()
            scaled = scaler.transform(raw)
            t1 = time.perf_counter()
            img = transform_batch(scaled.astype(np.float64), geom)
            t2 = time.perf_counter()
            with torch.no_grad():
                model(torch.from_numpy(img))
            t3 = time.perf_counter()
            if record:
                t_scale.append(t1 - t0); t_swt.append(t2 - t1); t_fwd.append(t3 - t2)

        for _ in range(warmup):
            one_pass(False)
        n = max(1, iters if bs > 1 else min(iters, 200))
        for _ in range(n):
            one_pass(True)

        def stats(v):
            a = np.array(v) * 1000.0        # ms
            return {"p50_ms": round(float(np.percentile(a, 50)), 4),
                    "p95_ms": round(float(np.percentile(a, 95)), 4)}

        total = np.array(t_scale) + np.array(t_swt) + np.array(t_fwd)
        out["batches"][str(bs)] = {
            "batch_size": bs,
            "t_scale": stats(t_scale), "t_swt": stats(t_swt),
            "t_forward": stats(t_fwd),
            "t_total": {"p50_ms": round(float(np.percentile(total * 1000, 50)), 4),
                        "p95_ms": round(float(np.percentile(total * 1000, 95)), 4)},
            "throughput_samples_per_s": round(bs / float(np.median(total)), 2),
            "swt_share_percent": round(
                100 * float(np.median(t_swt)) / float(np.median(total)), 2),
        }
    return out


# ================================================ buoc danh gia cuoi (muc 4.8)
def run_evaluation(cfg: dict, *, local_store_root=None, run_id: str | None = None,
                   skip_explain: bool = False, skip_benchmark: bool = False) -> int:
    """Chay SAU khi train xong 100 epoch. Idempotent (muc 7.E6).

    Buoc nay tach roi train: neu no loi thi checkpoint van nguyen tren S3 va
    chay lai duoc bang chinh lenh nay hoac bang make_report.py.
    """
    import io
    from pathlib import Path

    import torch

    from . import data as D
    from . import explain as X
    from .checkpoint import RunRegistry
    from .model import build_model
    from .s3io import SafeWriter, store_from_env

    torch.set_num_threads(int(cfg["train"].get("torch_num_threads", 4)))
    store = store_from_env(cfg, local_root=local_store_root)
    from .config import run_id_key

    run_id = run_id or RunRegistry(store, key=run_id_key(cfg)).get()
    if not run_id:
        raise RuntimeError("Khong tim thay current_run_id.json - chua co run nao")
    writer = SafeWriter(store)
    layout = cfg.get("s3", {}).get("layout", {})
    L = {k: layout.get(k, k) for k in
         ("metrics", "raw", "explainability", "config", "figures", "checkpoints")}

    def key(folder, name):
        return f"{run_id}/{L[folder]}/{name}"

    # ---- kiem tra da xong 100 epoch chua (muc 4.8)
    state = store.get_json_or_none(key("checkpoints", "training_state.json")) or {}
    if not state.get("is_complete"):
        raise RuntimeError(
            "current_epoch={} / {} - chua xong, chua duoc chay danh gia cuoi "
            "(muc 4.8)".format(state.get("current_epoch"), state.get("total_epochs")))

    # ---- du lieu + model
    work = Path(cfg["data"]["cache_dir"]).parent / "mddcc_work"
    prep = D.prepare_dataset(cfg, work / "artifacts")
    geom, schema, labels, splits = prep.geom, prep.schema, prep.labels, prep.splits
    classes = labels.classes
    nc = labels.num_classes

    model = build_model(cfg, side=geom.side, num_classes=nc)
    final_name = cfg["checkpoint"].get("final_name", "final_model_epoch_100.pt")
    local_ckpt = work / final_name
    store.get_file(key("checkpoints", final_name), local_ckpt)
    ckpt = torch.load(local_ckpt, map_location="cpu", weights_only=False)
    if ckpt.get("feature_schema_hash") not in (None, schema.hash):
        raise RuntimeError("feature_schema_hash lech giua checkpoint va du lieu hien tai")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    LOG.info("Nap %s (epoch=%s)", final_name, ckpt.get("epoch"))

    # ---- du doan tren TOAN BO test, giu nguyen thu tu
    test_ds = D.MDDCCDataset(prep.cache_path, splits.test, labels.codes, geom)
    bs = int(cfg["train"]["batch_size"])
    y_true, y_prob = predict_in_chunks(model, test_ds, splits.test,
                                       batch_size=bs, num_classes=nc)
    LOG.info("Du doan xong %d mau test", y_true.size)

    for name, arr in (("y_true.npy", y_true), ("y_prob.npy", y_prob)):
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        writer.put_bytes(buf.getvalue(), key("raw", name))

    # ---- metric
    res = evaluate_full(y_true, y_prob, classes, labels.benign_index)
    writer.put_json({"summary": res.summary, "binary": res.binary,
                     "per_class": res.per_class,
                     "confusion_matrix": res.cm.tolist(),
                     "classes": classes},
                    key("metrics", "test_metrics.json"))
    writer.put_bytes(res.report_text.encode("utf-8"),
                     key("metrics", "classification_report.txt"))
    writer.put_json(paper_comparison_rows(res.binary),
                    key("metrics", "paper_comparison.json"))
    LOG.info("Accuracy %.6f | Macro-F1 %.6f | MCC %.6f | BinaryFPR %.6f",
             res.summary["Accuracy"], res.summary["MacroF1"],
             res.summary["MCC"], res.summary["BinaryFPR"])

    # ---- do hieu nang trien khai (muc 6)
    summary = dict(res.summary)
    raw_cache = np.load(prep.cache_path, mmap_mode="r")
    if not skip_benchmark:
        sample = np.asarray(raw_cache[splits.test[:bs]], dtype=np.float64)
        b = cfg.get("evaluate", {}).get("benchmark", {})
        bench = benchmark_inference(
            model, prep.scaler, geom, sample,
            batch_sizes=tuple(b.get("batch_sizes", (4096, 1))),
            warmup=int(b.get("warmup_iters", 50)),
            iters=int(b.get("measure_iters", 500)))
        writer.put_json(bench, key("metrics", "inference_benchmark.json"))
        big = bench["batches"].get(str(bs)) or next(iter(bench["batches"].values()))
        summary.update({
            "inference_p50_ms": big["t_total"]["p50_ms"],
            "inference_p95_ms": big["t_total"]["p95_ms"],
            "throughput_samples_per_s": big["throughput_samples_per_s"],
            "t_swt_share_percent": big["swt_share_percent"],
            "n_parameters": bench["n_parameters"],
            "model_size_mb": bench["model_size_mb"],
        })

    history = store.get_json_or_none(key("metrics", "history.json")) or []
    summary.update({
        "final_epoch": state.get("current_epoch"),
        "n_sessions": len({r.get("session_id") for r in history}),
        "total_train_seconds": round(sum(r.get("epoch_seconds", 0) for r in history), 2),
        "peak_rss_mb": max((r.get("peak_rss_mb", 0) for r in history), default=0),
        "cache_build_seconds": state.get("cache_build_seconds", 0),
    })
    writer.put_json(summary, key("metrics", "summary_metrics.json"))

    # ---- giai thich (muc 7.J) - chi sau khi da co final model
    if not skip_explain:
        ecfg = cfg.get("explain", {})
        sample_idx = X.stratified_sample(
            labels.codes, splits.test, int(ecfg.get("sample_max_rows", 50000)),
            seed=cfg["experiment"]["seed"])
        raw = np.array(raw_cache[sample_idx], dtype=np.float64)
        ys = labels.codes[sample_idx].astype(np.int64)
        LOG.info("explain_sample: %d dong", raw.shape[0])
        writer.put_json({"n_rows": int(raw.shape[0]),
                         "source": "test split, phan tang theo ty le lop tu nhien",
                         "seed": cfg["experiment"]["seed"],
                         "causality_note": X.CAUSALITY_NOTE},
                        key("explainability", "explain_sample_manifest.json"))

        ab = X.branch_ablation(model, raw, ys, geom,
                               subbands=list(geom.subband_order),
                               num_classes=nc, batch_size=bs)
        writer.put_json(ab, key("explainability", "branch_ablation.json"))

        energy = X.subband_energy_by_class(raw, ys, geom, nc, batch_size=bs)
        writer.put_json({"energy": energy.tolist(),
                         "subbands": list(geom.subband_order), "classes": classes},
                        key("explainability", "wavelet_subband_energy.json"))

        p = ecfg.get("permutation", {})
        perm = X.permutation_importance(
            model, raw, ys, geom, schema.feature_columns, num_classes=nc,
            n_repeats=int(p.get("n_repeats", 5)),
            seed=int(p.get("seed", 42)), batch_size=bs)
        writer.put_json(perm, key("explainability", "permutation_importance.json"))

        s = ecfg.get("shap", {})
        shap_rows, shap_meta = X.shap_importance(
            model, raw, geom, schema.feature_columns,
            max_samples=int(s.get("max_samples", 2000)),
            background=int(s.get("background_samples", 200)),
            seed=cfg["experiment"]["seed"])
        writer.put_json(shap_rows, key("explainability", "shap_feature_importance.json"))
        writer.put_json(shap_meta, key("explainability", "shap_meta.json"))
        writer.put_json(X.importance_comparison(perm, shap_rows),
                        key("explainability", "feature_importance_comparison.json"))

    LOG.info("Danh gia cuoi HOAN TAT cho run_id=%s", run_id)
    return 0


def main(argv=None) -> int:
    import argparse
    import sys
    from pathlib import Path

    import yaml

    ap = argparse.ArgumentParser(description="Buoc danh gia cuoi MDDCC")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--local-store", type=Path, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--skip-explain", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    from .config import load_config

    cfg = load_config(args.config, args.variant)
    if args.input_dir:
        cfg["data"]["kaggle_input_dir"] = str(args.input_dir)
    if args.cache_dir:
        cfg["data"]["cache_dir"] = str(args.cache_dir)
    return run_evaluation(cfg, local_store_root=args.local_store, run_id=args.run_id,
                          skip_explain=args.skip_explain,
                          skip_benchmark=args.skip_benchmark)


if __name__ == "__main__":
    raise SystemExit(main())
