"""Kiem chung cong thuc (1)(2)(3) + quy tac 3-sigma tren DU LIEU TONG HOP - muc 3.G.

Khong dung CIC-DDoS2019: dataset nay khong co so lieu cong switch SDN.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import stage1_switch_stats as S1


# ------------------------------------------------------------- cong thuc
def test_formula_1_r_pi():
    assert S1.r_pi(np.array([100.0]), np.array([10.0]))[0] == pytest.approx(10.0)


def test_formula_2_r_fi():
    assert S1.r_fi(np.array([80.0]), np.array([100.0]))[0] == pytest.approx(0.8)


def test_formula_3_delta_np_is_absolute():
    assert S1.delta_n_p(np.array([500.0]), np.array([420.0]))[0] == pytest.approx(80.0)
    assert S1.delta_n_p(np.array([420.0]), np.array([500.0]))[0] == pytest.approx(80.0)


def test_zero_denominator_does_not_raise_or_false_alarm():
    """N_Pi = 0 -> +inf, khong bao gio duoi nguong -> khong bao dong gia."""
    assert np.isinf(S1.r_pi(np.array([100.0]), np.array([0.0]))[0])
    assert np.isinf(S1.r_fi(np.array([100.0]), np.array([0.0]))[0])


def test_compute_features_requires_all_five_counters():
    with pytest.raises(KeyError, match="N_PO"):
        S1.compute_features({"N_FI": np.array([1.0]), "N_FO": np.array([1.0]),
                             "N_Pi": np.array([1.0]), "N_PI": np.array([1.0])})


# --------------------------------------------------------------- nguong
def synth(n, rng, *, attack=False):
    """Sinh thong ke cong switch tong hop.

    Khi bi tan cong: N_Pi tang vot (nhieu PacketIn) -> R_Pi giam;
    ty le flow duoc chuyen tiep giam -> R_FI giam;
    switch tran buffer -> chenh lech packet vao/ra tang -> dN_P tang.
    """
    n_fi = rng.normal(1000, 50, n)
    if attack:
        n_pi = rng.normal(500, 30, n)     # binh thuong ~50
        n_fo = n_fi * rng.normal(0.30, 0.02, n)
        n_p_in = rng.normal(20000, 500, n)
        n_p_out = n_p_in - rng.normal(5000, 200, n)
    else:
        n_pi = rng.normal(50, 5, n)
        n_fo = n_fi * rng.normal(0.98, 0.005, n)
        n_p_in = rng.normal(20000, 500, n)
        n_p_out = n_p_in - rng.normal(50, 10, n)
    return {"N_FI": n_fi, "N_FO": n_fo, "N_Pi": n_pi,
            "N_PI": n_p_in, "N_PO": n_p_out}


def test_thresholds_follow_three_sigma_direction():
    """R_Pi/R_FI dung mu - 3sigma; dN_P dung mu + 3sigma."""
    th = S1.SigmaThresholds(r_pi_mean=20.0, r_pi_std=2.0,
                            r_fi_mean=0.98, r_fi_std=0.01,
                            dnp_mean=50.0, dnp_std=10.0)
    assert th.r_pi_threshold == pytest.approx(20.0 - 3 * 2.0)
    assert th.r_fi_threshold == pytest.approx(0.98 - 3 * 0.01)
    assert th.dnp_threshold == pytest.approx(50.0 + 3 * 10.0)


def test_fit_thresholds_on_normal_traffic_only():
    rng = np.random.default_rng(0)
    feats = S1.compute_features(synth(S1.BASELINE_SAMPLE_SETS, rng))
    th = S1.fit_thresholds(feats)
    assert th.n_baseline_sets == S1.BASELINE_SAMPLE_SETS
    assert th.k == 3.0
    assert th.r_pi_mean == pytest.approx(20.0, rel=0.1)


def test_strict_mode_enforces_10000_sets():
    rng = np.random.default_rng(1)
    feats = S1.compute_features(synth(100, rng))
    S1.fit_thresholds(feats)                       # khong strict -> chap nhan
    with pytest.raises(ValueError, match="10000"):
        S1.fit_thresholds(feats, strict=True)


def test_detects_attack_and_keeps_false_positive_rate_low():
    rng = np.random.default_rng(2)
    th = S1.fit_thresholds(S1.compute_features(synth(S1.BASELINE_SAMPLE_SETS, rng)))

    normal = S1.detect_from_stats(synth(5000, rng), th)
    attack = S1.detect_from_stats(synth(5000, rng, attack=True), th)

    assert attack.mean() > 0.99, "phai bat duoc hau het mau tan cong"
    assert normal.mean() < 0.01, "3-sigma phai giu bao dong gia rat thap"


def test_detection_requires_all_three_features():
    """Chi mot chi so vuot nguong thi KHONG duoc ket luan tan cong (phep AND)."""
    th = S1.SigmaThresholds(r_pi_mean=20.0, r_pi_std=1.0,
                            r_fi_mean=0.9, r_fi_std=0.01,
                            dnp_mean=50.0, dnp_std=5.0)
    only_r_pi = {"R_Pi": np.array([1.0]), "R_FI": np.array([0.95]),
                 "dN_P": np.array([50.0])}
    only_dnp = {"R_Pi": np.array([20.0]), "R_FI": np.array([0.95]),
                "dN_P": np.array([999.0])}
    all_three = {"R_Pi": np.array([1.0]), "R_FI": np.array([0.5]),
                 "dN_P": np.array([999.0])}

    assert not S1.detect(only_r_pi, th)[0]
    assert not S1.detect(only_dnp, th)[0]
    assert S1.detect(all_three, th)[0]


def test_explain_reports_each_component():
    rng = np.random.default_rng(3)
    th = S1.fit_thresholds(S1.compute_features(synth(2000, rng)))
    info = S1.explain(S1.compute_features(synth(500, rng, attack=True)), th)
    assert set(info) >= {"R_Pi_below", "R_FI_below", "dN_P_above", "all_three"}
    assert info["all_three"] <= min(info["R_Pi_below"], info["R_FI_below"],
                                    info["dN_P_above"]), "AND khong the lon hon tung thanh phan"


def test_scope_guard_blocks_running_on_cicddos2019():
    with pytest.raises(NotImplementedError, match="KHONG duoc danh gia"):
        S1.assert_not_applicable_to_cicddos2019()
    assert "Mininet/POX" in S1.SCOPE_NOTE
