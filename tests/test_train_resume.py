"""Chay that vong lap huan luyen, ngat giua epoch, resume - tieu chi 11.A.1.

Dung dataset tong hop nho + LocalStore de chay duoc trong CI, nhung di qua DUNG
duong ma run that di: prepare_dataset -> model -> train_one_epoch -> checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src import checkpoint as C
from src import data as D
from src import train as T
from src.s3io import LocalStore

REPO = Path(__file__).resolve().parents[1]


def build_cfg(tmp_path, *, epochs=2, batch_size=64, rows=1500):
    cfg = yaml.safe_load((REPO / "configs" / "mddcc.yaml").read_text(encoding="utf-8"))
    cfg["data"]["kaggle_input_dir"] = str(tmp_path / "data")
    cfg["data"]["cache_dir"] = str(tmp_path / "cache")
    cfg["train"]["epochs"] = epochs
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["torch_num_threads"] = 1
    cfg["checkpoint"]["interval_steps"] = 3
    cfg["experiment"]["deterministic_algorithms"] = False   # nhanh hon trong test
    return cfg


def make_dataset(tmp_path, rows=1500, n_files=2):
    """Dataset tong hop du de co 18+ cot va nhieu lop."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(0)
    out = tmp_path / "data"
    out.mkdir(parents=True, exist_ok=True)
    classes = ["BENIGN", "Syn", "TFTP", "UDP-lag"]   # merge_map -> UDPLag
    for f in range(n_files):
        cols = {f" Feat{i}": pa.array(rng.normal(i, 1 + i, rows)) for i in range(24)}
        cols["Flow ID"] = pa.array(["x"] * rows)
        cols[" Label"] = pa.array(rng.choice(classes, rows, p=[.25, .35, .3, .1]))
        pq.write_table(pa.table(cols), out / f"p{f}.parquet", row_group_size=311)
    return out


class TripAfterNSteps(C.TimeGuard):
    """Gia lap Kaggle cat session: bao should_stop sau N lan hoi."""

    trip_at = 5

    def __init__(self, cfg, start=None):
        super().__init__(cfg, start=start)
        self.calls = 0

    def should_stop(self) -> bool:
        self.calls += 1
        return self.calls >= self.trip_at


# ------------------------------------------------------------- chay tron ven
def test_full_run_two_epochs(tmp_path):
    make_dataset(tmp_path)
    cfg = build_cfg(tmp_path, epochs=2)
    store_root = tmp_path / "store"

    rc = T.run(cfg, config_path=REPO / "configs" / "mddcc.yaml",
               local_store_root=store_root)
    assert rc == 0

    store = LocalStore(store_root)
    run_id = C.RunRegistry(store).get()
    state = store.get_json(f"{run_id}/checkpoints/training_state.json")
    assert state["current_epoch"] == 2
    assert state["status"] == C.STATUS_COMPLETED
    assert state["is_complete"] is True

    hist = store.get_json(f"{run_id}/metrics/history.json")
    assert [r["epoch"] for r in hist] == [1, 2]
    assert hist[-1]["is_final_epoch"] is True
    assert store.exists(f"{run_id}/checkpoints/final_model_epoch_100.pt")

    # gop nhan phai di qua duoc duong that
    lm = store.get_json(f"{run_id}/config/label_mapping.json")
    assert lm["num_classes"] == 4 and "UDP-lag" not in lm["classes"]
    assert lm["label_merge"]["applied"] is True


def test_history_record_has_all_required_fields(tmp_path):
    make_dataset(tmp_path)
    cfg = build_cfg(tmp_path, epochs=1)
    store_root = tmp_path / "store"
    T.run(cfg, config_path=REPO / "configs" / "mddcc.yaml", local_store_root=store_root)

    store = LocalStore(store_root)
    run_id = C.RunRegistry(store).get()
    rec = store.get_json(f"{run_id}/metrics/history.json")[0]
    for key in ("epoch", "session_id", "timestamp_start", "timestamp_end",
                "learning_rate", "train_mse_loss", "val_mse_loss", "train_std_reg",
                "train_total_loss", "train_accuracy", "val_accuracy",
                "train_macro_f1", "val_macro_f1", "grad_norm_mean",
                "epoch_seconds", "samples_per_second", "peak_rss_mb",
                "is_final_epoch"):
        assert key in rec, f"muc 7.E1 doi hoi khoa {key}"


def test_run_config_records_acceptance_keys(tmp_path):
    make_dataset(tmp_path)
    cfg = build_cfg(tmp_path, epochs=1)
    store_root = tmp_path / "store"
    T.run(cfg, config_path=REPO / "configs" / "mddcc.yaml", local_store_root=store_root)

    store = LocalStore(store_root)
    run_id = C.RunRegistry(store).get()
    rc = store.get_json(f"{run_id}/config/run_config.json")
    assert rc["experiment_role"] == "paper_reproduction_mddcc"
    assert rc["early_stopping"] is False
    assert rc["imbalance_handling"] == "none"
    assert rc["feature_selection"] == "none"
    assert rc["use_all_features"] is True
    assert rc["device"] == "cpu"
    assert rc["wavelet"] == "db4" and rc["wavelet_level"] == 3 and rc["swt"] is True
    assert "deviations_from_paper" in rc
    assert set(rc["versions"]) >= {"torch", "numpy", "pywt", "sklearn"}


# ------------------------------------------------------ ngat giua epoch (4.4)
def test_interrupt_mid_epoch_then_resume_exactly(tmp_path, monkeypatch):
    make_dataset(tmp_path)
    cfg = build_cfg(tmp_path, epochs=2)
    store_root = tmp_path / "store"
    cfg_path = REPO / "configs" / "mddcc.yaml"

    # --- lan 1: bi cat giua epoch 1
    TripAfterNSteps.trip_at = 4
    monkeypatch.setattr(T, "TimeGuard", TripAfterNSteps)
    assert T.run(cfg, config_path=cfg_path, local_store_root=store_root) == 0

    store = LocalStore(store_root)
    run_id = C.RunRegistry(store).get()
    st1 = store.get_json(f"{run_id}/checkpoints/training_state.json")
    assert st1["status"] == C.STATUS_INTERRUPTED
    assert st1["exit_reason"] == "time_guard"
    assert st1["current_epoch"] == 0, "epoch 1 chua xong"
    assert st1["steps_done_in_epoch"] > 0, "phai nho da chay bao nhieu step"
    steps_before = st1["steps_done_in_epoch"]

    hist1 = store.get_json_or_none(f"{run_id}/metrics/history.json") or []
    assert hist1 == [], "chua xong epoch nao thi history phai rong"

    # --- lan 2: chay lai binh thuong, phai tiep tu dung cho
    monkeypatch.undo()
    assert T.run(cfg, config_path=cfg_path, local_store_root=store_root) == 0

    st2 = store.get_json(f"{run_id}/checkpoints/training_state.json")
    assert st2["run_id"] == run_id, "muc 4.5: run_id KHONG duoc doi khi resume"
    assert st2["session_id"] != st1["session_id"], "session_id phai moi"
    assert st2["current_epoch"] == 2 and st2["status"] == C.STATUS_COMPLETED
    assert st2["restart_count"] >= 1

    hist2 = store.get_json(f"{run_id}/metrics/history.json")
    assert [r["epoch"] for r in hist2] == [1, 2], "khong duoc thieu hay lap epoch"
    assert hist2[0]["train_metrics_partial"] is True
    assert hist2[0]["resumed_after_batches"] == steps_before
    assert hist2[1]["train_metrics_partial"] is False


def test_resume_covers_every_sample_exactly_once(tmp_path):
    """Ghep phan truoc va sau khi ngat phai phu dung mot lan toan bo tap train."""
    sampler = D.BatchSampler(1000, 128, seed=42)
    full = np.concatenate(list(sampler.batches(epoch=1)))

    cut = 3
    before = np.concatenate(list(sampler.batches(epoch=1))[:cut])
    after = np.concatenate(list(sampler.batches(epoch=1, skip=cut)))
    rejoined = np.concatenate([before, after])

    assert np.array_equal(rejoined, full)
    assert np.array_equal(np.sort(rejoined), np.arange(1000))
    assert len(set(before.tolist()) & set(after.tolist())) == 0, "khong duoc lap batch"


def test_completed_run_does_not_train_again(tmp_path):
    """Muc 4.8: current_epoch = total -> khong train them."""
    make_dataset(tmp_path)
    cfg = build_cfg(tmp_path, epochs=1)
    store_root = tmp_path / "store"
    cfg_path = REPO / "configs" / "mddcc.yaml"

    T.run(cfg, config_path=cfg_path, local_store_root=store_root)
    store = LocalStore(store_root)
    run_id = C.RunRegistry(store).get()
    hist_before = store.get_json(f"{run_id}/metrics/history.json")

    assert T.run(cfg, config_path=cfg_path, local_store_root=store_root) == 0
    assert store.get_json(f"{run_id}/metrics/history.json") == hist_before


def test_hash_mismatch_between_sessions_fails_fast(tmp_path):
    """Doi tap cot giua hai session -> phai dung, khong train lai tu dau."""
    make_dataset(tmp_path)
    cfg = build_cfg(tmp_path, epochs=1)
    store_root = tmp_path / "store"
    cfg_path = REPO / "configs" / "mddcc.yaml"
    T.run(cfg, config_path=cfg_path, local_store_root=store_root)

    # bo bot mot cot -> feature_schema_hash doi
    cfg2 = build_cfg(tmp_path, epochs=2)
    cfg2["data"]["identifier_columns"] = cfg2["data"]["identifier_columns"] + [" Feat3"]
    with pytest.raises(RuntimeError, match="khong khop cau hinh"):
        T.run(cfg2, config_path=cfg_path, local_store_root=store_root)


# ------------------------------------------------------------------ metrics
def test_confusion_accumulator_matches_sklearn():
    from sklearn.metrics import confusion_matrix, f1_score

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 500)
    y_pred = rng.integers(0, 5, 500)

    acc = T.ConfusionAccumulator(5)
    for i in range(0, 500, 64):                      # cong don theo chunk
        acc.update(y_true[i:i + 64], y_pred[i:i + 64])

    assert np.array_equal(acc.cm, confusion_matrix(y_true, y_pred, labels=range(5)))
    assert acc.macro_f1() == pytest.approx(
        f1_score(y_true, y_pred, average="macro", labels=range(5), zero_division=0))
    assert acc.accuracy == pytest.approx((y_true == y_pred).mean())


def test_macro_f1_skips_absent_classes():
    """Muc 7.G3: lop khong co mau that thi bo qua, khong keo Macro-F1 xuong 0."""
    acc = T.ConfusionAccumulator(4)
    acc.update(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    assert acc.macro_f1(present_only=True) == pytest.approx(1.0)
    assert acc.macro_f1(present_only=False) == pytest.approx(0.5)
