"""Do muc do quan trong cua dac trung - muc 7.J.

CHI chay sau khi da co final_model_epoch_100.pt (muc 7.J1). Khong dung ket qua
o day de chon feature roi huan luyen lai trong cung thi nghiem; chi lam dau vao
cho thi nghiem giam chieu sau nay.

CANH BAO dien giai (muc 7.J9): SHAP va permutation the hien dong gop du doan,
KHONG chung minh quan he nhan qua. Cac dac trung tuong quan co the chia se
importance.
"""
from __future__ import annotations

import logging

import numpy as np

from .wavelet import transform_batch

LOG = logging.getLogger(__name__)

CAUSALITY_NOTE = (
    "SHAP va permutation importance the hien dong gop vao du doan cua mo hinh, "
    "KHONG chung minh quan he nhan qua. Cac dac trung tuong quan manh co the "
    "chia se importance, lam ca hai cung thap mot cach gia tao."
)


# --------------------------------------------------------------- lay mau
def stratified_sample(y: np.ndarray, indices: np.ndarray, max_rows: int,
                      seed: int = 42) -> np.ndarray:
    """Lay mau phan tang theo DUNG ty le lop tu nhien (muc 7.J5)."""
    indices = np.asarray(indices)
    if indices.size <= max_rows:
        return indices
    rng = np.random.default_rng(seed)
    labels = y[indices]
    out = []
    for c in np.unique(labels):
        pool = indices[labels == c]
        take = max(1, int(round(max_rows * pool.size / indices.size)))
        take = min(take, pool.size)
        out.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(out))


def macro_f1_from_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                              num_classes: int) -> float:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    idx = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    cm += np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = cm.sum(axis=1) > 0
    return float(f1[present].mean()) if present.any() else 0.0


def _predict(model, raw: np.ndarray, geom, batch_size: int = 4096) -> np.ndarray:
    """raw [N, F] da scale -> nhan du doan. SWT tinh lai moi lan (bat buoc)."""
    import torch

    model.eval()
    out = np.empty(raw.shape[0], dtype=np.int16)
    with torch.no_grad():
        for s in range(0, raw.shape[0], batch_size):
            chunk = raw[s:s + batch_size].astype(np.float64)
            img = transform_batch(chunk, geom)
            out[s:s + chunk.shape[0]] = model(
                torch.from_numpy(img)).argmax(dim=1).numpy().astype(np.int16)
    return out


# ------------------------------------------- J2 permutation importance (C12)
def permutation_importance(model, raw: np.ndarray, y: np.ndarray, geom,
                           feature_names: list[str], *, num_classes: int,
                           n_repeats: int = 5, seed: int = 42,
                           batch_size: int = 4096) -> list[dict]:
    """Hoan vi tung cot TREN KHONG GIAN FEATURE GOC (truoc SWT), roi tinh lai SWT.

    Hoan vi truc tiep tren subband da bien doi se cho ket qua vo nghia, vi mot
    cot goc anh huong den ca 4 subband qua bo loc wavelet.
    """
    if raw.shape[1] != len(feature_names):
        raise ValueError(
            f"raw co {raw.shape[1]} cot nhung feature_schema co {len(feature_names)} "
            "- muc 7.J6 doi hoi khop tuyet doi")

    base = macro_f1_from_predictions(y, _predict(model, raw, geom, batch_size),
                                     num_classes)
    LOG.info("permutation: Macro-F1 goc = %.6f tren %d mau", base, raw.shape[0])

    rows = []
    for j, name in enumerate(feature_names):
        rng = np.random.default_rng([seed, j])
        drops = []
        original = raw[:, j].copy()
        for _ in range(n_repeats):
            raw[:, j] = rng.permutation(original)
            score = macro_f1_from_predictions(
                y, _predict(model, raw, geom, batch_size), num_classes)
            drops.append(base - score)
        raw[:, j] = original                       # tra lai nguyen trang
        rows.append({
            "feature": name,
            "mean_decrease": float(np.mean(drops)),   # muc 7.J2: giu ca gia tri am
            "std_decrease": float(np.std(drops)),
            "n_repeats": n_repeats,
            "baseline_macro_f1": float(base),
        })
        if (j + 1) % 10 == 0:
            LOG.info("  permutation %d/%d cot", j + 1, len(feature_names))

    order = np.argsort([-r["mean_decrease"] for r in rows])
    for rank, i in enumerate(order, start=1):
        rows[i]["rank"] = rank
    return rows


# ----------------------------------------------------------- J3 SHAP (C13)
def shap_importance(model, raw: np.ndarray, geom, feature_names: list[str], *,
                    max_samples: int = 2000, background: int = 200,
                    seed: int = 42, chunk: int = 200) -> tuple[list[dict], dict]:
    """mean(|SHAP|) toan cuc bang GradientExplainer, quy ve FEATURE GOC.

    Moi subband co cung do dai chuoi goc, nen dong gop cua vi tri i duoc cong
    |SHAP| qua ca 4 subband roi BO cac vi tri padding (muc 7.J3).
    """
    import torch

    try:
        import shap
    except ImportError:
        LOG.warning("Khong co thu vien shap -> bo qua C13")
        return [], {"skipped": True, "reason": "shap chua duoc cai"}

    rng = np.random.default_rng(seed)
    n = min(max_samples, raw.shape[0])
    bg_n = min(background, max(1, n // 4))
    idx = rng.choice(raw.shape[0], size=n, replace=False)
    bg_idx = rng.choice(raw.shape[0], size=bg_n, replace=False)

    model.eval()
    bg = torch.from_numpy(transform_batch(raw[bg_idx].astype(np.float64), geom))
    explainer = shap.GradientExplainer(model, bg)

    total = np.zeros(geom.side * geom.side, dtype=np.float64)
    used = 0
    for s in range(0, n, chunk):
        part = raw[idx[s:s + chunk]].astype(np.float64)
        x = torch.from_numpy(transform_batch(part, geom))
        try:
            vals = explainer.shap_values(x)
        except Exception as exc:                    # noqa: BLE001
            LOG.warning("SHAP that bai o chunk %d: %s -> dung lai", s, exc)
            break
        arr = np.stack(vals, axis=-1) if isinstance(vals, list) else np.asarray(vals)
        # [B, 4, S, S, (class)] -> cong |.| qua class, qua 4 subband, qua batch
        a = np.abs(arr)
        while a.ndim > 4:
            a = a.sum(axis=-1)
        total += a.sum(axis=0).sum(axis=0).reshape(-1)   # muc 7.J5: cong don, khong giu tensor
        used += part.shape[0]
        del x, vals, arr, a

    if used == 0:
        return [], {"skipped": True, "reason": "khong chunk nao thanh cong"}

    total /= used
    valid = total[:geom.n_features]                # bo vi tri padding
    s = valid.sum()
    rows = [{"feature": name, "mean_abs_shap": float(valid[i]),
             "shap_percent": float(100 * valid[i] / s) if s else 0.0}
            for i, name in enumerate(feature_names)]
    for rank, i in enumerate(np.argsort([-r["mean_abs_shap"] for r in rows]), start=1):
        rows[i]["rank_shap"] = rank
    meta = {"skipped": False, "n_samples_used": int(used),
            "n_background": int(bg_n), "requested_samples": int(max_samples)}
    LOG.info("SHAP: dung %d mau, background %d", used, bg_n)
    return rows, meta


# ------------------------------------------------- J4 branch ablation (C11)
def branch_ablation(model, raw: np.ndarray, y: np.ndarray, geom, *,
                    subbands: list[str], num_classes: int,
                    batch_size: int = 4096) -> list[dict]:
    """Zero-out lan luot tung nhanh, do muc giam Macro-F1. Khong huan luyen lai."""
    import torch

    model.eval()

    def score(mask):
        preds = np.empty(raw.shape[0], dtype=np.int16)
        with torch.no_grad():
            for s in range(0, raw.shape[0], batch_size):
                img = transform_batch(raw[s:s + batch_size].astype(np.float64), geom)
                p = model.forward_with_branch_mask(torch.from_numpy(img), mask)
                preds[s:s + img.shape[0]] = p.argmax(dim=1).numpy().astype(np.int16)
        return macro_f1_from_predictions(y, preds, num_classes)

    full = score([True] * len(subbands))
    rows = []
    for i, name in enumerate(subbands):
        mask = [True] * len(subbands)
        mask[i] = False
        s = score(mask)
        rows.append({"branch": name, "macro_f1_full": float(full),
                     "macro_f1_ablated": float(s),
                     "macro_f1_drop": float(full - s),
                     "drop_percent": float(100 * (full - s) / full) if full else 0.0})
        LOG.info("  ablation %s: %.6f -> %.6f", name, full, s)
    return rows


# --------------------------------------------- C10 nang luong subband theo lop
def subband_energy_by_class(raw: np.ndarray, y: np.ndarray, geom,
                            num_classes: int, *, batch_size: int = 4096) -> np.ndarray:
    """Tra ve [num_classes, 4]: nang luong trung binh tung subband cho tung lop."""
    from .wavelet import subband_energy

    acc = np.zeros((num_classes, geom.n_subbands), dtype=np.float64)
    cnt = np.zeros(num_classes, dtype=np.int64)
    for s in range(0, raw.shape[0], batch_size):
        img = transform_batch(raw[s:s + batch_size].astype(np.float64), geom)
        e = subband_energy(img)
        yy = y[s:s + img.shape[0]]
        for c in np.unique(yy):
            m = yy == c
            acc[c] += e[m].sum(axis=0)
            cnt[c] += int(m.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(acc / np.maximum(cnt, 1)[:, None])


# ------------------------------------------------------ J8 bang doi chieu
def importance_comparison(perm_rows: list[dict], shap_rows: list[dict]) -> list[dict]:
    """feature_importance_comparison.csv - KHONG gop hai thuoc do thanh mot diem."""
    shap_by = {r["feature"]: r for r in shap_rows}
    perm_top10 = {r["feature"] for r in sorted(
        perm_rows, key=lambda r: -r["mean_decrease"])[:10]}
    shap_top10 = {r["feature"] for r in sorted(
        shap_rows, key=lambda r: -r["mean_abs_shap"])[:10]} if shap_rows else set()

    rows = []
    for r in perm_rows:
        sh = shap_by.get(r["feature"], {})
        rows.append({
            "feature": r["feature"],
            "mean_decrease": r["mean_decrease"],
            "rank_permutation": r.get("rank"),
            "mean_abs_shap": sh.get("mean_abs_shap"),
            "rank_shap": sh.get("rank_shap"),
            "top10_consensus": r["feature"] in perm_top10 and r["feature"] in shap_top10,
        })
    return sorted(rows, key=lambda r: r["rank_permutation"] or 1e9)
