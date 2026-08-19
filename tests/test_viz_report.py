"""Kiem chung ve hinh va make_report - muc 7.A, 7.C, 7.D, 7.I."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import viz

REPO = Path(__file__).resolve().parents[1]
RUN_ID = "mddcc_20260819-0000"
CLASSES = ["BENIGN", "Syn", "TFTP", "UDPLag", "WebDDoS"]


def fake_history(n=6, sessions=("s1", "s1", "s1", "s2", "s2", "s3")):
    return [{"epoch": i + 1, "session_id": sessions[i],
             "learning_rate": 0.01,
             "train_mse_loss": 0.9 - i * 0.05, "val_mse_loss": 0.92 - i * 0.05,
             "train_std_reg": 1.06 - i * 0.002, "train_total_loss": 1.9 - i * 0.05,
             "train_accuracy": 0.3 + i * 0.05, "val_accuracy": 0.28 + i * 0.05,
             "train_macro_f1": 0.1 + i * 0.03, "val_macro_f1": 0.09 + i * 0.03,
             "grad_norm_mean": 0.8 - i * 0.05, "epoch_seconds": 100 + i,
             "samples_per_second": 2800.0, "peak_rss_mb": 1200.0,
             "is_final_epoch": i == n - 1}
            for i in range(n)]


def assert_triplet(out_dir: Path, name: str, *, csv: bool = True):
    """Muc 7.C: moi hinh xuat DONG THOI png + pdf (+ csv)."""
    assert (out_dir / f"{name}.png").exists(), f"thieu {name}.png"
    assert (out_dir / f"{name}.pdf").exists(), f"thieu {name}.pdf"
    if csv:
        assert (out_dir / f"{name}.csv").exists(), f"thieu {name}.csv"
    assert (out_dir / f"{name}.png").stat().st_size > 1000


# ------------------------------------------------------- rang buoc kien truc
def test_backend_is_agg():
    """Muc 7.A3: bat buoc matplotlib.use('Agg')."""
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"


def code_only(path: Path) -> str:
    """Bo comment va chuoi/docstring - chi con MA THUC SU chay.

    Can thiet vi cac module co viet "khong hardcode b-/r-" hay "KHONG chua
    matplotlib" ngay trong comment; tim chuoi tho se bat nham chinh loi nhac.
    """
    import io
    import tokenize

    out = []
    with path.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


def test_no_plt_show_anywhere_in_src():
    """Muc 7.I1: khong duoc goi plt.show() o bat cu dau."""
    for p in list((REPO / "src").glob("*.py")) + [REPO / "make_report.py"]:
        # tokenize tach "plt.show()" thanh "plt . show ( )"
        assert "plt . show" not in code_only(p), f"{p.name} co plt.show()"


def test_train_does_not_import_matplotlib():
    """Muc 7.A1: train.py KHONG duoc chua code matplotlib."""
    code = code_only(REPO / "src" / "train.py")
    for bad in ("matplotlib", "pyplot", "plt .", "viz"):
        assert bad not in code, f"train.py co tham chieu {bad!r}"


def test_figures_are_closed_not_leaked(tmp_path):
    """Muc 7.B2: plt.close(fig) sau moi hinh, khong giu figure toan cuc."""
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    # Ghi vao tmp_path, KHONG vao "." - neu khong se rai hinh vao goc repo
    viz.plot_lr_schedule(fake_history(), tmp_path, RUN_ID, 6)
    assert len(plt.get_fignums()) == before, "figure bi ro ri"


def test_palette_has_no_hardcoded_shorthand():
    """Muc 7.D1: khong hardcode kieu 'b-' / 'r-'."""
    # Bo comment truoc khi tim: viz.py co ghi loi nhac "khong hardcode b-/r-"
    src = (REPO / "src" / "viz.py").read_text(encoding="utf-8")
    code = "\n".join(line.split("#")[0] for line in src.splitlines())
    for bad in ('"b-"', "'b-'", '"r-"', "'r-'", '"g-"', "'g-'"):
        assert bad not in code


def test_title_contains_run_id_and_final_epoch():
    """Muc 7.D7."""
    t = viz.make_title("roc_curves", RUN_ID, 100)
    assert "MDDCC" in t and RUN_ID in t and "final_epoch=100" in t


def test_legend_moves_outside_when_many_classes():
    """Muc 7.D2."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="x")
    lg = viz.legend_for(ax, 12)
    assert lg._loc in (2, "upper left") or lg.get_bbox_to_anchor() is not None
    plt.close(fig)


def test_csv_rounds_to_six_digits():
    """Muc 5: moi so thuc trong CSV lam tron 6 chu so."""
    assert viz.r6(0.123456789) == 0.123457
    assert viz.r6(float("nan")) == ""
    assert viz.r6("khong phai so") == "khong phai so"


def test_safe_wrapper_swallows_plotting_errors():
    """Muc 7.A4: loi ve chi ghi WARNING, khong lam hong train/upload."""
    def boom():
        raise RuntimeError("hong")
    assert viz.safe(boom) == []


# --------------------------------------------------------------- tung hinh
def test_c1_learning_curves(tmp_path):
    viz.plot_learning_curves(fake_history(), tmp_path, RUN_ID, 6)
    assert_triplet(tmp_path, "learning_curves")


def test_c2_lr_schedule(tmp_path):
    viz.plot_lr_schedule(fake_history(), tmp_path, RUN_ID, 6)
    assert_triplet(tmp_path, "lr_schedule")
    rows = (tmp_path / "lr_schedule.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "epoch,learning_rate"
    assert rows[1].endswith("0.01")


def test_c3_c4_confusion_matrix(tmp_path):
    cm = np.array([[50, 2, 1, 0, 0], [3, 40, 5, 1, 0], [0, 4, 60, 2, 0],
                   [1, 0, 3, 20, 1], [0, 1, 0, 2, 5]])
    viz.plot_confusion_matrix(cm, CLASSES, tmp_path, RUN_ID, 6)
    viz.plot_confusion_matrix_raw(cm, CLASSES, tmp_path, RUN_ID, 6)
    assert_triplet(tmp_path, "confusion_matrix", csv=False)
    assert_triplet(tmp_path, "confusion_matrix_raw", csv=False)
    assert (tmp_path / "confusion_matrix.csv").exists()            # muc 7.F3
    assert (tmp_path / "confusion_matrix_normalized.csv").exists()  # muc 7.F4


def test_normalized_confusion_matrix_rows_sum_to_one(tmp_path):
    cm = np.array([[8, 2], [1, 9]])
    viz.plot_confusion_matrix(cm, ["a", "b"], tmp_path, RUN_ID, 6)
    txt = (tmp_path / "confusion_matrix_normalized.csv").read_text(encoding="utf-8")
    rows = [l.split(",") for l in txt.strip().splitlines()[1:]]
    for r in rows:
        assert sum(float(v) for v in r[1:]) == pytest.approx(1.0, abs=1e-6)


def test_confusion_matrix_handles_empty_row(tmp_path):
    """Muc 7.D10: sau khi chuan hoa phai np.nan_to_num."""
    cm = np.array([[5, 1], [0, 0]])          # lop 1 khong co mau that
    viz.plot_confusion_matrix(cm, ["a", "b"], tmp_path, RUN_ID, 6)
    txt = (tmp_path / "confusion_matrix_normalized.csv").read_text(encoding="utf-8")
    assert "nan" not in txt.lower()


def test_c5_c6_roc_pr(tmp_path):
    from src.evaluate import compute_curves
    from tests.test_evaluate import fake_predictions

    y, prob = fake_predictions(k=5)
    roc, pr = compute_curves(y, prob, CLASSES)
    viz.plot_roc_curves(roc, CLASSES, tmp_path, RUN_ID, 6, n_samples=y.size)
    viz.plot_pr_curves(pr, CLASSES, tmp_path, RUN_ID, 6, n_samples=y.size)
    assert_triplet(tmp_path, "roc_curves")
    assert_triplet(tmp_path, "pr_curves")

    head = (tmp_path / "roc_curves.csv").read_text(encoding="utf-8").splitlines()[0]
    assert head == "class,fpr,tpr,threshold"          # muc 7.F1
    head = (tmp_path / "pr_curves.csv").read_text(encoding="utf-8").splitlines()[0]
    assert head == "class,recall,precision,threshold"  # muc 7.F2


def test_c7_per_class_sorted_by_support(tmp_path):
    rows = [{"class": c, "support": s, "precision": 0.5, "recall": 0.5,
             "f1": 0.5, "fpr": 0.1}
            for c, s in zip(CLASSES, [1000, 50, 300, 20, 5])]
    viz.plot_per_class_metrics(rows, tmp_path, RUN_ID, 6)
    assert_triplet(tmp_path, "per_class_metrics", csv=False)


def test_c8_class_distribution(tmp_path):
    counts = {sp: {c: 100 * (i + 1) for i, c in enumerate(CLASSES)}
              for sp in ("train", "val", "test")}
    viz.plot_class_distribution(counts, CLASSES, tmp_path, RUN_ID, 6)
    assert_triplet(tmp_path, "class_distribution")


def test_c9_epoch_time_colours_by_session(tmp_path):
    viz.plot_epoch_time(fake_history(), tmp_path, RUN_ID, 6)
    assert_triplet(tmp_path, "epoch_time")
    txt = (tmp_path / "epoch_time.csv").read_text(encoding="utf-8")
    assert "session_id" in txt and "s3" in txt


def test_c10_subband_energy(tmp_path):
    e = np.abs(np.random.default_rng(0).random((5, 4)))
    viz.plot_wavelet_subband_energy(e, CLASSES, ["cD1", "cD2", "cD3", "cA3"],
                                    tmp_path, RUN_ID, 6)
    assert_triplet(tmp_path, "wavelet_subband_energy")


def test_c11_branch_ablation(tmp_path):
    rows = [{"branch": b, "macro_f1_full": 0.5, "macro_f1_ablated": 0.5 - i * 0.05,
             "macro_f1_drop": i * 0.05, "drop_percent": i * 10.0}
            for i, b in enumerate(["cD1", "cD2", "cD3", "cA3"])]
    viz.plot_branch_ablation(rows, tmp_path, RUN_ID, 6)
    assert_triplet(tmp_path, "branch_ablation")


def test_c12_c13_importance_keeps_zero_rows(tmp_path):
    """Muc 7.J7: CSV giu ca dac trung co importance bang 0."""
    perm = [{"feature": f"f{i}", "mean_decrease": max(0.0, 0.5 - i * 0.05),
             "std_decrease": 0.01, "rank": i + 1} for i in range(40)]
    shap = [{"feature": f"f{i}", "mean_abs_shap": max(0.0, 0.4 - i * 0.04),
             "shap_percent": 1.0, "rank_shap": i + 1} for i in range(40)]
    viz.plot_permutation_importance(perm, tmp_path, RUN_ID, 6)
    viz.plot_shap_importance(shap, tmp_path, RUN_ID, 6)
    assert_triplet(tmp_path, "permutation_importance")
    assert_triplet(tmp_path, "shap_feature_importance")
    lines = (tmp_path / "permutation_importance.csv").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 41, "CSV phai giu du 40 dac trung ke ca importance 0"


def test_c14_paper_comparison(tmp_path):
    rows = [{"metric": m, "ours_binary": 0.9, "paper_table9": v}
            for m, v in [("Accuracy", 0.9963), ("Precision", 0.9798),
                         ("Recall", 0.9871), ("F1", 0.9834), ("FPR", 0.0818)]]
    viz.plot_paper_comparison(rows, tmp_path, RUN_ID, 6)
    assert_triplet(tmp_path, "paper_comparison")


# --------------------------------------------------------------- make_report
def test_make_report_regenerates_from_artifacts_only(tmp_path):
    """Muc 7.A2 + tieu chi 11.A.7: sinh lai hinh CHI tu artifact, khong train lai."""
    import make_report
    from src.s3io import LocalStore
    from tests.test_evaluate import fake_predictions

    store_root = tmp_path / "store"
    store = LocalStore(store_root)
    rid = RUN_ID

    y, prob = fake_predictions(n=400, k=5)
    import io
    for name, arr in (("y_true.npy", y.astype(np.int16)),
                      ("y_prob.npy", prob.astype(np.float32))):
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        store.put_bytes(buf.getvalue(), f"{rid}/raw/{name}")

    store.put_json(fake_history(), f"{rid}/metrics/history.json")
    store.put_json({"classes": CLASSES, "num_classes": 5, "benign_index": 0},
                   f"{rid}/config/label_mapping.json")
    store.put_json({"epochs": 6, "full_config": {}}, f"{rid}/config/run_config.json")
    store.put_json({"per_split_class_counts":
                    {sp: {c: 100 for c in CLASSES} for sp in ("train", "val", "test")}},
                   f"{rid}/config/sample_manifest.json")
    store.put_json({"binary": {"Accuracy": 0.9, "Precision": 0.9, "Recall": 0.9,
                               "F1": 0.9, "FPR": 0.1}},
                   f"{rid}/metrics/test_metrics.json")

    art = make_report.RunArtifacts(store, rid)
    made = make_report.build_report(art, tmp_path / "report")

    for fig in ("C1 learning_curves", "C2 lr_schedule", "C3 confusion_matrix",
                "C4 confusion_matrix_raw", "C5 roc_curves", "C6 pr_curves",
                "C7 per_class_metrics", "C8 class_distribution",
                "C9 epoch_time", "C14 paper_comparison"):
        assert fig in made, f"make_report thieu {fig}"

    out = tmp_path / "report"
    assert len(list(out.glob("*.png"))) >= 10
    assert len(list(out.glob("*.pdf"))) >= 10
    assert (out / "history.csv").exists()          # muc 7.F6


def test_make_report_parses_s3_uri():
    import make_report

    art = make_report.RunArtifacts.from_arg("s3://buck/pre/fix/mddcc_1", None)
    assert art.run_id == "mddcc_1"
    assert art.store.bucket == "buck" and art.store.prefix == "pre/fix"
