"""Khoa cac sieu tham so da chot trong configs/mddcc.yaml - tieu chi 11.A.9.

Muc dich: mot lan sua nham file config se lam hong ca run 39 session ma khong ai
biet cho den khi doc lai run_config.json vai ngay sau. Test nay bat ngay.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CFG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "configs" / "mddcc.yaml")
    .read_text(encoding="utf-8"))


# ------------------------------------------------- sieu tham so bat buoc (2.D)
def test_exactly_100_epochs_no_early_stopping():
    assert CFG["train"]["epochs"] == 100
    assert CFG["train"]["early_stopping"] is False


def test_batch_size_and_learning_rate():
    assert CFG["train"]["batch_size"] == 4096
    # Da doi ve 0.01 ngay 2026-08-19 theo dung bai bao (truoc do la 0.001).
    assert CFG["optim"]["learning_rate"] == 0.01


def test_learning_rate_is_constant_no_scheduler():
    assert CFG["optim"]["scheduler"] == "none"
    assert CFG["optim"]["warmup"] == "none"


def test_optimizer_is_sgd_without_momentum_or_weight_decay():
    o = CFG["optim"]
    assert o["name"] == "sgd"
    assert o["momentum"] == 0.0 and o["nesterov"] is False
    assert o["weight_decay"] == 0.0, "da co sigma(w), khong chong them L2"


def test_loss_is_mse_with_std_regularizer():
    assert CFG["loss"]["name"] == "mse"
    assert CFG["loss"]["std_regularizer"]["enabled"] is True
    assert CFG["loss"]["std_regularizer"]["include_bias"] is False


def test_device_is_cpu():
    assert CFG["train"]["device"] == "cpu"


def test_no_feature_selection_no_imbalance_handling():
    assert CFG["data"]["feature_selection"] == "none"
    assert CFG["train"]["imbalance_handling"] == "none"


# ------------------------------------------------------------- wavelet (2.A)
def test_wavelet_is_db4_level3_swt():
    w = CFG["wavelet"]
    assert w["name"] == "db4" and w["level"] == 3
    assert w["transform"] == "swt", "wavedec co downsampling, sai voi bai bao"
    assert w["subband_order"] == ["cD1", "cD2", "cD3", "cA3"]


def test_geometry_guard_is_enabled():
    assert CFG["wavelet"]["reshape"]["min_final_map"] >= 2
    assert CFG["model"]["pool_ceil_mode"] is True


def test_compose_is_sum_per_equation_10():
    assert CFG["model"]["compose"] == "sum"


def test_branch_specs_match_table3():
    specs = CFG["model"]["branches"]
    assert [s["conv_out"] for s in specs] == [32, 64, 32]
    assert [s["dropout"] for s in specs] == [0.2, 0.3, 0.2]
    assert all(s["kernel"] == 3 and s["padding"] == 1 and s["pool"] == 2 for s in specs)
    assert CFG["model"]["num_branches"] == 4


# -------------------------------------------------------------- split (3.D)
def test_split_ratios():
    s = CFG["split"]
    assert s["test_size"] == 0.30
    assert s["val_size_within_trainval"] == 0.15
    assert s["stratify"] is True and s["assert_no_overlap"] is True


# ------------------------------------------------------------- sai khac (11.B)
def test_learning_rate_no_longer_listed_as_deviation():
    """Da khop bai bao thi khong duoc con nam trong bang sai khac."""
    assert "learning_rate" not in CFG["deviations_from_paper"]


def test_remaining_deviations_are_documented():
    dev = CFG["deviations_from_paper"]
    for key in ("batch_size", "stopping", "feature_set", "hardware", "datasets",
                "split", "task", "stage1", "regularization", "pool_ceil_mode",
                "label_merge"):
        assert key in dev, f"muc 11.B doi hoi ghi sai khac {key}"
        assert "ours" in dev[key], f"{key} thieu mo ta 'ours'"


def test_label_merge_is_declared():
    assert CFG["data"]["label"]["merge_map"] == {"UDP-lag": "UDPLag"}


# ----------------------------------------------------------------- session
def test_time_guard_fits_kaggle_12h_limit():
    s = CFG["session"]
    assert s["time_limit_seconds"] <= 43200, "Kaggle cat sau 12h"
    assert s["exit_guard_seconds"] >= 600, "can du thoi gian de luu va upload"
