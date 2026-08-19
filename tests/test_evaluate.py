"""Kiem chung metric va giai thich - muc 5, 6, 7.G, 7.J."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef)

from src import evaluate as E
from src import explain as X
from src import model as M
from src.wavelet import compute_geometry
from tests.test_model import CFG


def fake_predictions(n=800, k=5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n).astype(np.int16)
    prob = rng.random((n, k)).astype(np.float32)
    prob[np.arange(n), y] += 1.5           # co tin hieu that de AUC > 0.5
    prob /= prob.sum(axis=1, keepdims=True)
    return y, prob


# --------------------------------------------------------------- metric co ban
def test_summary_matches_sklearn():
    y, prob = fake_predictions()
    res = E.evaluate_full(y, prob, [f"c{i}" for i in range(5)], benign_index=0)
    pred = prob.argmax(1)

    assert res.summary["Accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert res.summary["BalancedAccuracy"] == pytest.approx(
        balanced_accuracy_score(y, pred))
    assert res.summary["MacroF1"] == pytest.approx(
        f1_score(y, pred, average="macro", labels=range(5), zero_division=0))
    assert res.summary["MCC"] == pytest.approx(matthews_corrcoef(y, pred))
    assert np.array_equal(res.cm, confusion_matrix(y, pred, labels=range(5)))


def test_summary_has_every_required_metric():
    """Muc 5: bat buoc giu rieng Balanced Accuracy, MCC, F1 lop hiem, FPR."""
    y, prob = fake_predictions()
    s = E.evaluate_full(y, prob, [f"c{i}" for i in range(5)], 0).summary
    for key in ("Accuracy", "BalancedAccuracy", "MacroPrecision", "MacroRecall",
                "MacroF1", "WeightedF1", "MCC", "MinorityClassF1", "MacroFPR",
                "BinaryFPR", "LogLoss", "AUC-ROC_macro-OVR",
                "AUC-ROC_weighted-OVR", "AUC-ROC_micro", "PR-AUC_macro"):
        assert key in s, f"muc 5 doi hoi {key}"


def test_fpr_formula_one_vs_rest():
    """FPR = FP / (FP + TN) tinh one-vs-rest cho tung lop - muc 7.G4."""
    cm = np.array([[8, 2, 0], [1, 7, 2], [0, 3, 7]])
    fpr = E.fpr_per_class(cm)
    # lop 0: FP = 1, TN = 7+2+3+7 = 19 -> 1/20
    assert fpr[0] == pytest.approx(1 / 20)
    # lop 1: FP = 2+3 = 5, TN = 8+0+0+7 = 15 -> 5/20
    assert fpr[1] == pytest.approx(5 / 20)


def test_fpr_of_perfect_prediction_is_zero():
    assert np.allclose(E.fpr_per_class(np.diag([5, 5, 5])), 0.0)


def test_per_class_metrics_columns():
    """Muc 7.F5: class, support, precision, recall, f1, fpr, roc_auc, pr_auc."""
    y, prob = fake_predictions()
    rows = E.evaluate_full(y, prob, [f"c{i}" for i in range(5)], 0).per_class
    assert len(rows) == 5
    for r in rows:
        assert set(r) == {"class", "support", "precision", "recall", "f1",
                          "fpr", "roc_auc", "pr_auc"}


# ------------------------------------------------------------------- AUC (7.G)
def test_auc_reports_three_variants():
    y, prob = fake_predictions()
    auc = E.safe_roc_auc(y, prob, 5)
    for k in ("macro-OVR", "weighted-OVR", "micro"):
        assert 0.5 < auc[k] <= 1.0, f"{k} khong hop le: {auc[k]}"


def test_auc_skips_classes_absent_from_test():
    """Muc 7.G3: lop khong co mau duong thi bo qua, khong nem loi, khong NaN."""
    y, prob = fake_predictions(n=400, k=5, seed=1)
    y[y == 4] = 3                              # lop 4 bien mat khoi test
    prob = np.concatenate([prob, np.zeros((400, 0), dtype=np.float32)], axis=1)
    auc = E.safe_roc_auc(y, prob, 5)
    assert auc["classes_skipped"] == 1
    assert np.isfinite(auc["macro-OVR"])


def test_evaluate_does_not_crash_on_degenerate_model():
    """Mo hinh du doan mot lop duy nhat - phai ra so, khong duoc nem loi."""
    n, k = 300, 4
    y = np.random.default_rng(2).integers(0, k, n).astype(np.int16)
    prob = np.zeros((n, k), dtype=np.float32)
    prob[:, 0] = 1.0
    res = E.evaluate_full(y, prob, [f"c{i}" for i in range(k)], benign_index=0)
    assert res.summary["MCC"] == pytest.approx(0.0)
    assert np.isfinite(res.summary["MacroF1"])


# --------------------------------------------------------------- binary view
def test_binary_view_collapses_multiclass_predictions():
    """Muc 3.E: BENIGN vs ATTACK gop tu du doan da lop."""
    classes = ["BENIGN", "Syn", "TFTP"]
    y = np.array([0, 0, 1, 2, 1], dtype=np.int16)
    prob = np.eye(3, dtype=np.float32)[[0, 1, 1, 2, 0]]   # 1 FP, 1 FN
    b = E.evaluate_full(y, prob, classes, benign_index=0).binary
    assert b["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert b["FPR"] == pytest.approx(0.5)
    assert b["Recall"] == pytest.approx(2 / 3)


def test_paper_comparison_has_delta_and_note():
    """Muc 5: doi chieu Table 9, co cot delta va note giai thich sai khac."""
    binary = {"Accuracy": 0.95, "Precision": 0.9, "Recall": 0.92,
              "F1": 0.91, "FPR": 0.10}
    rows = E.paper_comparison_rows(binary)
    assert {r["metric"] for r in rows} == set(E.PAPER_TABLE9)
    for r in rows:
        assert r["delta"] == pytest.approx(r["ours_binary"] - r["paper_table9"])
        assert r["note"]
    fpr = next(r for r in rows if r["metric"] == "FPR")
    assert fpr["paper_table9"] == 0.0818


# ---------------------------------------------------------------- ROC/PR curves
def test_curves_contain_micro_and_macro():
    y, prob = fake_predictions()
    roc, pr = E.compute_curves(y, prob, [f"c{i}" for i in range(5)])
    assert "micro" in roc and "macro" in roc and "micro" in pr
    for name, c in roc.items():
        assert c["fpr"].size == c["tpr"].size
        assert 0.0 <= c["auc"] <= 1.0


def test_pr_auc_uses_average_precision_not_trapezoid():
    """Muc 7.G1: PR-AUC phai dung average_precision_score."""
    from sklearn.metrics import average_precision_score

    y, prob = fake_predictions()
    _, pr = E.compute_curves(y, prob, [f"c{i}" for i in range(5)])
    expected = average_precision_score((y == 0).astype(int), prob[:, 0])
    assert pr["c0"]["ap"] == pytest.approx(expected)


# ------------------------------------------------------------ explain (7.J)
def geom_and_model(n_features=24, num_classes=4):
    g = compute_geometry(n_features, level=3)
    m = M.build_model(CFG, side=g.side, num_classes=num_classes)
    m.eval()
    return g, m


def test_stratified_sample_keeps_class_proportions():
    rng = np.random.default_rng(0)
    y = rng.choice(4, 10000, p=[.6, .25, .1, .05]).astype(np.int16)
    idx = X.stratified_sample(y, np.arange(10000), 1000, seed=42)
    assert 950 <= idx.size <= 1050
    for c in range(4):
        assert abs((y[idx] == c).mean() - (y == c).mean()) < 0.03


def test_stratified_sample_returns_all_when_small():
    y = np.zeros(50, dtype=np.int16)
    assert X.stratified_sample(y, np.arange(50), 1000).size == 50


def test_macro_f1_helper_matches_sklearn():
    rng = np.random.default_rng(3)
    yt = rng.integers(0, 4, 500)
    yp = rng.integers(0, 4, 500)
    assert X.macro_f1_from_predictions(yt, yp, 4) == pytest.approx(
        f1_score(yt, yp, average="macro", labels=range(4), zero_division=0))


def test_permutation_permutes_raw_features_not_subbands():
    """Muc 7.J2: hoan vi tren khong gian GOC va tra lai nguyen trang sau moi cot."""
    g, m = geom_and_model()
    rng = np.random.default_rng(0)
    raw = rng.random((120, 24))
    before = raw.copy()
    y = rng.integers(0, 4, 120)

    rows = X.permutation_importance(m, raw, y, g, [f"f{i}" for i in range(24)],
                                    num_classes=4, n_repeats=2, batch_size=64)
    assert np.allclose(raw, before), "phai tra lai du lieu goc sau khi hoan vi"
    assert len(rows) == 24
    assert all("rank" in r and "std_decrease" in r for r in rows)
    assert {r["rank"] for r in rows} == set(range(1, 25))


def test_permutation_rejects_feature_count_mismatch():
    g, m = geom_and_model()
    raw = np.random.default_rng(0).random((20, 24))
    with pytest.raises(ValueError, match="feature_schema"):
        X.permutation_importance(m, raw, np.zeros(20, dtype=int), g,
                                 ["a", "b"], num_classes=4, n_repeats=1)


def test_branch_ablation_reports_all_four_branches():
    g, m = geom_and_model()
    rng = np.random.default_rng(1)
    rows = X.branch_ablation(m, rng.random((100, 24)), rng.integers(0, 4, 100), g,
                             subbands=["cD1", "cD2", "cD3", "cA3"],
                             num_classes=4, batch_size=64)
    assert [r["branch"] for r in rows] == ["cD1", "cD2", "cD3", "cA3"]
    for r in rows:
        assert r["macro_f1_drop"] == pytest.approx(
            r["macro_f1_full"] - r["macro_f1_ablated"])


def test_subband_energy_shape_per_class():
    g, m = geom_and_model()
    rng = np.random.default_rng(2)
    e = X.subband_energy_by_class(rng.random((200, 24)),
                                  rng.integers(0, 4, 200), g, 4, batch_size=64)
    assert e.shape == (4, 4)
    assert (e >= 0).all()


def test_importance_comparison_flags_consensus():
    """Muc 7.J8: co rank cua ca hai thuoc do, KHONG gop thanh mot diem."""
    perm = [{"feature": f"f{i}", "mean_decrease": 1.0 - i * 0.05, "rank": i + 1}
            for i in range(20)]
    shap = [{"feature": f"f{i}", "mean_abs_shap": 1.0 - i * 0.04, "rank_shap": i + 1}
            for i in range(20)]
    rows = X.importance_comparison(perm, shap)
    assert set(rows[0]) == {"feature", "mean_decrease", "rank_permutation",
                            "mean_abs_shap", "rank_shap", "top10_consensus"}
    assert rows[0]["top10_consensus"] is True
    assert rows[-1]["top10_consensus"] is False


def test_causality_note_is_explicit():
    """Muc 7.J9: phai neu ro khong chung minh nhan qua."""
    assert "KHONG chung minh quan he nhan qua" in X.CAUSALITY_NOTE


# ------------------------------------------------------------ benchmark (muc 6)
def test_benchmark_separates_scale_swt_and_forward():
    from src.data import ScalerStats

    g, m = geom_and_model()
    sc = ScalerStats([f"f{i}" for i in range(24)], np.zeros(24), np.ones(24),
                     np.zeros(24), np.zeros(24), np.zeros(24, dtype=np.int64),
                     np.zeros(24, dtype=np.int64), 100)
    raw = np.random.default_rng(0).random((64, 24))
    b = E.benchmark_inference(m, sc, g, raw, batch_sizes=(64, 1),
                              warmup=2, iters=5)

    assert b["dropout_disabled"] is True, "muc 6: phai chay o model.eval()"
    for bs in ("64", "1"):
        e = b["batches"][bs]
        for part in ("t_scale", "t_swt", "t_forward", "t_total"):
            assert e[part]["p50_ms"] >= 0 and e[part]["p95_ms"] >= e[part]["p50_ms"]
        assert e["throughput_samples_per_s"] > 0
        assert 0 <= e["swt_share_percent"] <= 100


# --------------------------------------------------------- SHAP that (7.J3)
def test_shap_maps_subbands_back_to_raw_features():
    """Chay voi thu vien shap THAT: 4 subband -> feature goc, bo padding."""
    shap = pytest.importorskip("shap")

    torch.manual_seed(0)
    g, m = geom_and_model(n_features=24, num_classes=4)
    raw = np.random.default_rng(0).random((240, 24))
    names = [f"f{i}" for i in range(24)]

    rows, meta = X.shap_importance(m, raw, g, names, max_samples=100,
                                   background=25, chunk=40)
    assert meta["skipped"] is False
    assert meta["n_samples_used"] == 100 and meta["n_background"] == 25
    assert len(rows) == len(names), "phai quy ve DUNG so feature goc, khong phai S*S"
    assert {r["rank_shap"] for r in rows} == set(range(1, len(names) + 1))
    assert sum(r["shap_percent"] for r in rows) == pytest.approx(100.0, abs=1e-6)
    assert all(r["mean_abs_shap"] >= 0 for r in rows)


def test_shap_downscales_when_asked_for_more_than_available():
    """Muc 7.J5: tu giam sample size va ghi ro so mau THUC TE da dung."""
    pytest.importorskip("shap")

    g, m = geom_and_model(n_features=24, num_classes=4)
    raw = np.random.default_rng(1).random((60, 24))
    rows, meta = X.shap_importance(m, raw, g, [f"f{i}" for i in range(24)],
                                   max_samples=5000, background=200, chunk=30)
    assert meta["n_samples_used"] == 60, "phai bao so mau THUC TE"
    assert meta["requested_samples"] == 5000
    assert len(rows) == 24


def test_shap_missing_library_is_not_fatal(monkeypatch):
    """Thieu shap thi bo qua C13 kem canh bao, khong lam hong buoc danh gia."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "shap":
            raise ImportError("gia lap thieu shap")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    g, m = geom_and_model()
    rows, meta = X.shap_importance(m, np.zeros((10, 24)), g,
                                   [f"f{i}" for i in range(24)])
    assert rows == [] and meta["skipped"] is True
    assert "shap" in meta["reason"]
