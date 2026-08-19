"""TOAN BO ham ve bieu do - muc 7.

train.py KHONG duoc import module nay o vong lap chinh (muc 7.A1). Moi hinh xuat
DONG THOI 3 file cung ten: .png (300 dpi), .pdf (vector), .csv (du lieu dung nhu
da ve) - muc 7.C, de sau nay gop nhieu mo hinh len chung mot hinh ma khong phai
chay lai thi nghiem.
"""
from __future__ import annotations

import csv
import gc
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # BAT BUOC truoc moi import pyplot (muc 7.A3)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

LOG = logging.getLogger(__name__)

# --------------------------------------------------------- bang mau (muc 7.D1)
# KHONG hardcode "b-"/"r-". Khi so duong > 6, moi duong khac nhau DONG THOI o
# mau + linestyle + marker de con doc duoc khi in den trang.
# tab20 xen ke cap dam/nhat cung tong mau -> hai duong lien tiep gan giong nhau
# khi in den trang. Lay 10 mau dam truoc, roi moi den 10 mau nhat.
_TAB20 = plt.get_cmap("tab20").colors
PALETTE = tuple(_TAB20[0::2]) + tuple(_TAB20[1::2])
LINESTYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

FONT = {"axis_label": 10, "title": 11, "tick": 10, "legend": 9}
GRID = {"linestyle": "--", "alpha": 0.4}
LEGEND_OUTSIDE_ABOVE = 6
CM_ANNOT_SMALL_ABOVE = 10
CM_NO_ANNOT_ABOVE = 15
DPI = 300


def style_for(i: int) -> dict:
    return {"color": PALETTE[i % len(PALETTE)],
            "linestyle": LINESTYLES[(i // len(PALETTE)) % len(LINESTYLES)
                                    if len(PALETTE) else 0],
            "marker": MARKERS[i % len(MARKERS)]}


def make_title(base: str, run_id: str, final_epoch: int, extra: str = "") -> str:
    """Muc 7.D7: title chua MDDCC + run_id + final_epoch."""
    t = f"MDDCC | {base} | {run_id} | final_epoch={final_epoch}"
    return f"{t}\n{extra}" if extra else t


def apply_axes(ax, *, xlabel="", ylabel="", title="", rotate_x=False):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT["axis_label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT["axis_label"])
    if title:
        ax.set_title(title, fontsize=FONT["title"])
    ax.tick_params(labelsize=FONT["tick"])
    ax.grid(True, **GRID)
    if rotate_x:
        for lb in ax.get_xticklabels():
            lb.set_rotation(45)
            lb.set_ha("right")


def legend_for(ax, n_items: int, **kw):
    """Muc 7.D2: >6 lop thi dua legend ra ngoai vung ve."""
    if n_items > LEGEND_OUTSIDE_ABOVE:
        return ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                         fontsize=FONT["legend"], **kw)
    return ax.legend(fontsize=FONT["legend"], **kw)


def save_figure(fig, out_dir: Path, name: str) -> list[Path]:
    """Luu .png (300 dpi) + .pdf (vector). Luon dong figure (muc 7.B2)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)               # KHONG dung plt.close() trong
    gc.collect()                 # muc 7.B1
    return paths


def save_csv(rows: list[dict], out_dir: Path, name: str,
             fieldnames: list[str] | None = None) -> Path:
    """Du lieu tho di kem moi hinh - muc 7.F."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.csv"
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    fields = fieldnames or list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    return p


def r6(x) -> float:
    """Muc 5: moi so thuc trong CSV lam tron 6 chu so."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return round(v, 6) if np.isfinite(v) else ""


# ============================================================ C1 learning curves
def plot_learning_curves(history: list[dict], out_dir: Path, run_id: str,
                         final_epoch: int) -> list[Path]:
    """1 hang 4 cot: MSE loss, Accuracy, Macro-F1, chan doan grad_norm + sigma(w)."""
    if not history:
        return []
    ep = [r["epoch"] for r in history]

    def col(k):
        return [r.get(k, float("nan")) for r in history]

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6))
    panels = [
        (axes[0], "MSE loss", [("train_mse_loss", "train"), ("val_mse_loss", "val")]),
        (axes[1], "Accuracy", [("train_accuracy", "train"), ("val_accuracy", "val")]),
        (axes[2], "Macro-F1", [("train_macro_f1", "train"), ("val_macro_f1", "val")]),
    ]
    for ax, ylabel, series in panels:
        # train lien net, val net dut - phan biet duoc ca khi in den trang
        for i, (key, lbl) in enumerate(series):
            ax.plot(ep, col(key), label=lbl, color=PALETTE[i % len(PALETTE)],
                    linestyle=("-", "--")[i % 2], linewidth=1.6)
        apply_axes(ax, xlabel="Epoch", ylabel=ylabel, title=ylabel)
        ax.legend(fontsize=FONT["legend"])

    # Panel (d): bang chung mo hinh co thuc su hoc voi MSE + SGD
    ax = axes[3]
    st = style_for(0)
    ax.plot(ep, col("grad_norm_mean"), color=st["color"], linestyle="-",
            label="grad_norm", linewidth=1.6)
    apply_axes(ax, xlabel="Epoch", ylabel="grad_norm", title="Chan doan huan luyen")
    ax2 = ax.twinx()
    st = style_for(1)
    ax2.plot(ep, col("train_std_reg"), color=st["color"], linestyle="--",
             label="sigma(w)", linewidth=1.6)
    ax2.set_ylabel("std_reg  sigma(w)", fontsize=FONT["axis_label"])
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=FONT["legend"])

    # Vach Resume tai moi cho doi session (muc 7.E3)
    boundaries, prev = [], None
    for r in history:
        sid = r.get("session_id")
        if prev is not None and sid != prev:
            boundaries.append(r["epoch"])
        prev = sid
    for ax_ in axes:
        for i, b in enumerate(boundaries):
            ax_.axvline(b, color="grey", linestyle=":", linewidth=1.1,
                        label="Resume" if i == 0 else None)
    if boundaries:
        axes[0].legend(fontsize=FONT["legend"])

    fig.suptitle(make_title("learning_curves", run_id, final_epoch),
                 fontsize=FONT["title"])
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "learning_curves")
    save_csv([{k: r6(v) if isinstance(v, (int, float)) else v
               for k, v in r.items()} for r in history],
             out_dir, "learning_curves")
    return paths


# ================================================================ C2 lr schedule
def plot_lr_schedule(history: list[dict], out_dir: Path, run_id: str,
                     final_epoch: int) -> list[Path]:
    if not history:
        return []
    ep = [r["epoch"] for r in history]
    lr = [r.get("learning_rate", float("nan")) for r in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    st = style_for(0)
    ax.plot(ep, lr, color=st["color"], linestyle="-", linewidth=1.8)
    apply_axes(ax, xlabel="Epoch", ylabel="Learning rate",
               title=make_title("lr_schedule", run_id, final_epoch))
    ax.set_ylim(0, max(lr) * 1.3 if max(lr) > 0 else 1)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "lr_schedule")
    save_csv([{"epoch": e, "learning_rate": r6(v)} for e, v in zip(ep, lr)],
             out_dir, "lr_schedule")
    return paths


# =========================================================== C3/C4 confusion mtx
def _cm_annot_kw(n: int) -> dict | None:
    if n > CM_NO_ANNOT_ABOVE:
        return None                       # muc 7.D3: tat annot, chi dung colorbar
    return {"size": 7} if n > CM_ANNOT_SMALL_ABOVE else {"size": 9}


def _draw_cm(cm: np.ndarray, classes: list[str], *, title: str, fmt: str,
             log_scale: bool, cbar_label: str):
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.62), max(6, n * 0.55)))
    norm = LogNorm(vmin=max(cm[cm > 0].min(), 1), vmax=cm.max()) \
        if log_scale and (cm > 0).any() else None
    im = ax.imshow(cm, cmap="viridis", norm=norm,
                   aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label=cbar_label)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    apply_axes(ax, xlabel="Du doan", ylabel="That", title=title, rotate_x=True)
    ax.grid(False)

    kw = _cm_annot_kw(n)
    if kw:
        thr = cm.max() / 2 if cm.max() else 0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                        color="white" if cm[i, j] < thr else "black", **kw)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_dir: Path,
                          run_id: str, final_epoch: int) -> list[Path]:
    """C3: chuan hoa theo HANG -> doc ra recall tung lop."""
    cm = np.asarray(cm, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = cm / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)            # muc 7.D10
    fig = _draw_cm(norm, classes,
                   title=make_title("confusion_matrix (chuan hoa theo hang)",
                                    run_id, final_epoch),
                   fmt=".2f", log_scale=False, cbar_label="Ty le trong hang (recall)")
    paths = save_figure(fig, out_dir, "confusion_matrix")
    save_csv([{"class": c, **{cj: r6(norm[i, j]) for j, cj in enumerate(classes)}}
              for i, c in enumerate(classes)],
             out_dir, "confusion_matrix_normalized", ["class"] + classes)
    return paths


def plot_confusion_matrix_raw(cm: np.ndarray, classes: list[str], out_dir: Path,
                              run_id: str, final_epoch: int) -> list[Path]:
    """C4: so dem tho, thang mau log."""
    cm = np.asarray(cm, dtype=np.int64)
    fig = _draw_cm(cm.astype(float), classes,
                   title=make_title("confusion_matrix_raw (thang log)",
                                    run_id, final_epoch),
                   fmt=".0f", log_scale=True, cbar_label="So mau (log)")
    paths = save_figure(fig, out_dir, "confusion_matrix_raw")
    save_csv([{"class": c, **{cj: int(cm[i, j]) for j, cj in enumerate(classes)}}
              for i, c in enumerate(classes)],
             out_dir, "confusion_matrix", ["class"] + classes)
    return paths


# ================================================================== C5 ROC / C6 PR
def plot_roc_curves(curves: dict, classes: list[str], out_dir: Path, run_id: str,
                    final_epoch: int, n_samples: int | None = None) -> list[Path]:
    """One-vs-rest tung lop + micro + macro + duong cheo Random Guess."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    rows = []
    for i, name in enumerate([*classes, "micro", "macro"]):
        c = curves.get(name)
        if c is None:
            continue
        st = style_for(i)
        ax.plot(c["fpr"], c["tpr"], color=st["color"], linestyle=st["linestyle"],
                linewidth=1.4, label=f"{name} (AUC={c['auc']:.4f})")
        for f, t, th in zip(c["fpr"], c["tpr"], c.get("threshold", [])):
            rows.append({"class": name, "fpr": r6(f), "tpr": r6(t), "threshold": r6(th)})
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.0,
            label="Random Guess")
    extra = f"n_samples_ve={n_samples:,}" if n_samples else ""
    apply_axes(ax, xlabel="False Positive Rate", ylabel="True Positive Rate",
               title=make_title("roc_curves", run_id, final_epoch, extra))
    legend_for(ax, len(classes) + 3)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "roc_curves")
    save_csv(rows, out_dir, "roc_curves", ["class", "fpr", "tpr", "threshold"])
    return paths


def plot_pr_curves(curves: dict, classes: list[str], out_dir: Path, run_id: str,
                   final_epoch: int, n_samples: int | None = None) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    rows = []
    for i, name in enumerate([*classes, "micro"]):
        c = curves.get(name)
        if c is None:
            continue
        st = style_for(i)
        ax.plot(c["recall"], c["precision"], color=st["color"],
                linestyle=st["linestyle"], linewidth=1.4,
                label=f"{name} (AP={c['ap']:.4f})")
        thr = list(c.get("threshold", []))
        for k, (rc, pr) in enumerate(zip(c["recall"], c["precision"])):
            rows.append({"class": name, "recall": r6(rc), "precision": r6(pr),
                         "threshold": r6(thr[k]) if k < len(thr) else ""})
    extra = f"n_samples_ve={n_samples:,}" if n_samples else ""
    apply_axes(ax, xlabel="Recall", ylabel="Precision",
               title=make_title("pr_curves", run_id, final_epoch, extra))
    legend_for(ax, len(classes) + 2)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "pr_curves")
    save_csv(rows, out_dir, "pr_curves", ["class", "recall", "precision", "threshold"])
    return paths


# ========================================================== C7 per-class metrics
def plot_per_class_metrics(rows: list[dict], out_dir: Path, run_id: str,
                           final_epoch: int) -> list[Path]:
    """Cot ngang F1/Precision/Recall/FPR, sap xep theo support tang dan."""
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r.get("support", 0))
    names = [f"{r['class']}  (n={int(r.get('support', 0)):,})" for r in rows]
    metrics = ["f1", "precision", "recall", "fpr"]
    y = np.arange(len(rows))
    h = 0.2

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * len(rows))))
    for i, m in enumerate(metrics):
        st = style_for(i)
        ax.barh(y + (i - 1.5) * h, [float(r.get(m, 0) or 0) for r in rows],
                height=h, color=st["color"], label=m)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    apply_axes(ax, xlabel="Gia tri", ylabel="Lop (support tang dan)",
               title=make_title("per_class_metrics", run_id, final_epoch))
    ax.legend(fontsize=FONT["legend"])
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "per_class_metrics")
    return paths


# ========================================================== C8 class distribution
def plot_class_distribution(counts: dict[str, dict[str, int]], classes: list[str],
                            out_dir: Path, run_id: str, final_epoch: int) -> list[Path]:
    """Phan bo lop o train/val/test, thang log."""
    splits = list(counts.keys())
    x = np.arange(len(classes))
    w = 0.8 / max(len(splits), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.6), 5))
    for i, sp in enumerate(splits):
        st = style_for(i)
        ax.bar(x + (i - (len(splits) - 1) / 2) * w,
               [max(counts[sp].get(c, 0), 0) for c in classes],
               width=w, color=st["color"], label=sp)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    apply_axes(ax, xlabel="Lop", ylabel="So mau (log)",
               title=make_title("class_distribution", run_id, final_epoch),
               rotate_x=True)
    ax.legend(fontsize=FONT["legend"])
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "class_distribution")
    save_csv([{"class": c, **{sp: counts[sp].get(c, 0) for sp in splits}}
              for c in classes], out_dir, "class_distribution", ["class"] + splits)
    return paths


# ================================================================= C9 epoch time
def plot_epoch_time(history: list[dict], out_dir: Path, run_id: str,
                    final_epoch: int) -> list[Path]:
    """Thoi gian moi epoch, to mau theo session_id - de uoc luong so session con lai."""
    if not history:
        return []
    sessions = []
    for r in history:
        if r.get("session_id") not in sessions:
            sessions.append(r.get("session_id"))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, sid in enumerate(sessions):
        pts = [(r["epoch"], r.get("epoch_seconds", 0)) for r in history
               if r.get("session_id") == sid]
        st = style_for(i)
        ax.bar([p[0] for p in pts], [p[1] for p in pts], color=st["color"],
               label=f"session {i + 1}", width=0.85)
    apply_axes(ax, xlabel="Epoch", ylabel="Giay",
               title=make_title("epoch_time", run_id, final_epoch,
                                f"{len(sessions)} session"))
    legend_for(ax, len(sessions))
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "epoch_time")
    save_csv([{"epoch": r["epoch"], "session_id": r.get("session_id"),
               "epoch_seconds": r6(r.get("epoch_seconds")),
               "samples_per_second": r6(r.get("samples_per_second"))}
              for r in history], out_dir, "epoch_time")
    return paths


# ====================================================== C10 wavelet subband energy
def plot_wavelet_subband_energy(energy: np.ndarray, classes: list[str],
                                subbands: list[str], out_dir: Path, run_id: str,
                                final_epoch: int) -> list[Path]:
    """Nang luong trung binh tung subband theo lop (chuan hoa theo hang).

    Hinh dac thu cua MDDCC: minh chung "tan so cao <-> nhieu loan ngan han,
    tan so thap <-> xu huong dai han".
    """
    e = np.asarray(energy, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.nan_to_num(e / e.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(max(7, len(subbands) * 1.6),
                                    max(5, len(classes) * 0.45)))
    im = ax.imshow(e, cmap="viridis", aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Ty le nang luong trong lop")
    ax.set_xticks(range(len(subbands))); ax.set_xticklabels(subbands)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(subbands)):
            ax.text(j, i, f"{e[i, j]:.3f}", ha="center", va="center",
                    color="white" if e[i, j] < e.max() / 2 else "black", size=8)
    apply_axes(ax, xlabel="Subband", ylabel="Lop",
               title=make_title("wavelet_subband_energy", run_id, final_epoch))
    ax.grid(False)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "wavelet_subband_energy")
    save_csv([{"class": c, **{s: r6(e[i, j]) for j, s in enumerate(subbands)}}
              for i, c in enumerate(classes)],
             out_dir, "wavelet_subband_energy", ["class"] + subbands)
    return paths


# ============================================================= C11 branch ablation
def plot_branch_ablation(rows: list[dict], out_dir: Path, run_id: str,
                         final_epoch: int) -> list[Path]:
    """Macro-F1 giam bao nhieu khi zero-out tung nhanh CNN."""
    if not rows:
        return []
    names = [r["branch"] for r in rows]
    drop = [float(r.get("macro_f1_drop", 0)) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (n, d) in enumerate(zip(names, drop)):
        ax.bar(i, d, color=style_for(i)["color"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    apply_axes(ax, xlabel="Nhanh bi zero-out", ylabel="Muc giam Macro-F1",
               title=make_title("branch_ablation", run_id, final_epoch))
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "branch_ablation")
    save_csv([{k: r6(v) if isinstance(v, float) else v for k, v in r.items()}
              for r in rows], out_dir, "branch_ablation")
    return paths


# ================================================ C12/C13 feature importance
def _plot_importance(rows: list[dict], value_key: str, err_key: str | None,
                     out_dir: Path, name: str, run_id: str, final_epoch: int,
                     top_k: int = 30) -> list[Path]:
    if not rows:
        return []
    top = sorted(rows, key=lambda r: float(r.get(value_key, 0)), reverse=True)[:top_k]
    top = list(reversed(top))                 # cot ngang giam dan tu tren xuong
    y = np.arange(len(top))
    err = [float(r.get(err_key, 0) or 0) for r in top] if err_key else None

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.32 * len(top))))
    ax.barh(y, [float(r[value_key]) for r in top], xerr=err,
            color=[style_for(i)["color"] for i in range(len(top))],
            error_kw={"elinewidth": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels([r["feature"] for r in top], fontsize=8)
    apply_axes(ax, xlabel=value_key, ylabel="Dac trung",
               title=make_title(name, run_id, final_epoch, f"top {len(top)}"))
    fig.tight_layout()
    return save_figure(fig, out_dir, name)


def plot_permutation_importance(rows, out_dir, run_id, final_epoch, top_k=30):
    paths = _plot_importance(rows, "mean_decrease", "std_decrease", out_dir,
                             "permutation_importance", run_id, final_epoch, top_k)
    save_csv([{k: r6(v) if isinstance(v, float) else v for k, v in r.items()}
              for r in rows], out_dir, "permutation_importance")
    return paths


def plot_shap_importance(rows, out_dir, run_id, final_epoch, top_k=30,
                         n_samples=None):
    paths = _plot_importance(rows, "mean_abs_shap", None, out_dir,
                             "shap_feature_importance", run_id, final_epoch, top_k)
    save_csv([{k: r6(v) if isinstance(v, float) else v for k, v in r.items()}
              for r in rows], out_dir, "shap_feature_importance")
    return paths


# ============================================================ C14 paper comparison
def plot_paper_comparison(rows: list[dict], out_dir: Path, run_id: str,
                          final_epoch: int) -> list[Path]:
    """Cot nhom so Accuracy/Precision/Recall/F1/FPR cua ta voi Table 9 / Table 10."""
    if not rows:
        return []
    metrics = [r["metric"] for r in rows]
    series = [k for k in rows[0] if k not in ("metric", "delta", "note")]
    x = np.arange(len(metrics))
    w = 0.8 / max(len(series), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.5), 5))
    for i, s in enumerate(series):
        vals = [float(r.get(s) or 0) for r in rows]
        st = style_for(i)
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, width=w,
                      color=st["color"], label=s)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    apply_axes(ax, xlabel="Chi so", ylabel="Gia tri",
               title=make_title("paper_comparison (binary view)", run_id, final_epoch),
               rotate_x=True)
    ax.legend(fontsize=FONT["legend"])
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "paper_comparison")
    save_csv([{k: r6(v) if isinstance(v, float) else v for k, v in r.items()}
              for r in rows], out_dir, "paper_comparison")
    return paths


# ------------------------------------------------------------------ an toan
def safe(fn, *args, **kwargs):
    """Muc 7.A4: loi ve chi ghi WARNING, TUYET DOI khong lam hong train/upload."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                      # noqa: BLE001
        LOG.warning("Ve hinh %s that bai: %s", getattr(fn, "__name__", fn), exc)
        return []
