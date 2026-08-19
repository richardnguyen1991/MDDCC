"""Kiem chung checkpoint, upload an toan va resume - muc 4."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src import checkpoint as C
from src import model as M
from src.s3io import LocalStore, SafeWriter, sha256_bytes
from tests.test_model import CFG as MODEL_CFG

CFG = {
    **MODEL_CFG,
    "experiment": {"seed": 42, "run_id_prefix": "mddcc"},
    "checkpoint": {"interval_steps": 5, "keep_last_n": 3,
                   "permanent_every_n_epochs": 10,
                   "final_name": "final_model_epoch_100.pt",
                   "on_hash_mismatch": "fail_fast"},
    "session": {"time_limit_seconds": 40800, "exit_guard_seconds": 1200,
                "exit_reason_on_guard": "time_guard"},
    "s3": {"layout": {"checkpoints": "checkpoints", "metrics": "metrics"},
           "safe_upload": {"tmp_prefix": "_tmp"}},
}
HASHES = {"params_hash": "p" * 8, "feature_schema_hash": "f" * 8,
          "scaler_hash": "s" * 8}


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "store")


@pytest.fixture
def mgr(store, tmp_path):
    return C.CheckpointManager(store, CFG, "mddcc_20260818-1200", tmp_path / "local")


def make_model():
    m = M.build_model(CFG, side=10, num_classes=18)
    return m, M.build_optimizer(CFG, m)


# ------------------------------------------------------- upload an toan (4.2)
def test_safe_writer_leaves_no_temp_key(store):
    w = SafeWriter(store)
    w.put_json({"a": 1}, "x/y.json")
    keys = store.list_keys("")
    assert "x/y.json" in keys
    assert not [k for k in keys if k.startswith("_tmp")], "phai xoa key tam"


def test_safe_writer_verifies_checksum(store, monkeypatch):
    """Neu ban tam bi hong thi key chinh KHONG duoc dong toi."""
    w = SafeWriter(store)
    w.put_json({"good": True}, "k.json")
    before = store.get_bytes("k.json")

    monkeypatch.setattr(store, "get_bytes",
                        lambda key: b"hong" if key.startswith("_tmp") else before)
    with pytest.raises(RuntimeError, match="Upload hong"):
        w.put_json({"good": False}, "k.json")

    monkeypatch.undo()
    assert store.get_bytes("k.json") == before, "key chinh phai con nguyen ven"


def test_local_store_roundtrip_and_listing(store, tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    store.put_file(p, "a/b/f.bin")
    assert store.exists("a/b/f.bin")
    assert store.get_bytes("a/b/f.bin") == b"hello"
    assert "a/b/f.bin" in store.list_keys("a/")
    store.delete("a/b/f.bin")
    assert not store.exists("a/b/f.bin")


# ------------------------------------------------------- training state (4.9)
def test_training_state_roundtrip_and_required_keys():
    s = C.TrainingState(run_id="r", session_id="s", current_epoch=7,
                        total_epochs=100, global_step=1234)
    d = s.to_dict()
    for key in ("run_id", "session_id", "current_epoch", "total_epochs",
                "global_step", "status", "exit_reason", "updated_at_utc",
                "restart_count"):
        assert key in d, f"muc 4.9 doi hoi khoa {key}"
    assert C.TrainingState.from_dict(d).current_epoch == 7


def test_next_epoch_and_completion():
    s = C.TrainingState(run_id="r", session_id="s", current_epoch=99, total_epochs=100)
    assert s.next_epoch == 100 and not s.is_complete
    s.current_epoch = 100
    assert s.is_complete and s.to_dict()["is_complete"]


# ------------------------------------------------------------- history (7.E)
def test_history_is_append_only_and_rejects_duplicate_epoch():
    h = C.History()
    h.append({"epoch": 1, "session_id": "a"})
    h.append({"epoch": 2, "session_id": "a"})
    with pytest.raises(RuntimeError, match="da co trong history"):
        h.append({"epoch": 2, "session_id": "b"})


def test_history_detects_gap():
    h = C.History([{"epoch": 1}, {"epoch": 3}])
    with pytest.raises(RuntimeError, match="khong lien tuc"):
        h.validate_continuous()


def test_history_session_boundaries_for_resume_lines():
    h = C.History([{"epoch": 1, "session_id": "s1"}, {"epoch": 2, "session_id": "s1"},
                   {"epoch": 3, "session_id": "s2"}, {"epoch": 4, "session_id": "s2"},
                   {"epoch": 5, "session_id": "s3"}])
    assert h.session_boundaries() == [3, 5], "vach Resume o epoch doi session"


def test_history_survives_reload_from_store(mgr):
    h = C.History()
    for e in (1, 2, 3):
        h.append({"epoch": e, "session_id": "s1"})
    mgr.save_history(h)

    reloaded = mgr.load_history()
    assert [r["epoch"] for r in reloaded.records] == [1, 2, 3]
    reloaded.append({"epoch": 4, "session_id": "s2"})
    assert reloaded.last_epoch == 4


# ------------------------------------------------------------ save / load
def test_save_writes_checkpoint_and_state(mgr, store):
    m, opt = make_model()
    st = C.TrainingState(run_id=mgr.run_id, session_id="s1", current_epoch=1,
                         total_epochs=100, global_step=10)
    mgr.save(model=m, optimizer=opt, state=st, hashes=HASHES)

    assert store.exists(mgr.last_key)
    assert store.get_json(mgr.state_key)["current_epoch"] == 1


def test_permanent_snapshot_every_ten_epochs(mgr, store):
    m, opt = make_model()
    for e in (9, 10, 11, 20):
        st = C.TrainingState(run_id=mgr.run_id, session_id="s", current_epoch=e,
                             steps_done_in_epoch=0)
        mgr.save(model=m, optimizer=opt, state=st, hashes=HASHES)
    snaps = [k for k in store.list_keys(f"{mgr.run_id}/checkpoints/epoch_")]
    assert any("epoch_010.pt" in k for k in snaps)
    assert any("epoch_020.pt" in k for k in snaps)
    assert not any("epoch_009.pt" in k for k in snaps)


def test_final_model_saved_at_last_epoch(mgr, store):
    m, opt = make_model()
    st = C.TrainingState(run_id=mgr.run_id, session_id="s", current_epoch=100,
                         total_epochs=100, status=C.STATUS_COMPLETED)
    mgr.save(model=m, optimizer=opt, state=st, hashes=HASHES, is_final=True)
    assert store.exists(mgr.key("checkpoints", "final_model_epoch_100.pt"))


def test_load_restores_weights_exactly(mgr):
    m, opt = make_model()
    with torch.no_grad():
        m.fc.weight.fill_(0.1234)
    st = C.TrainingState(run_id=mgr.run_id, session_id="s", current_epoch=3,
                         global_step=30)
    mgr.save(model=m, optimizer=opt, state=st, hashes=HASHES)

    m2, opt2 = make_model()
    ckpt = mgr.load_checkpoint(model=m2, optimizer=opt2, expected_hashes=HASHES)
    assert ckpt["epoch"] == 3 and ckpt["global_step"] == 30
    assert torch.allclose(m2.fc.weight, m.fc.weight)


def test_load_without_checkpoint_returns_none(mgr):
    m, opt = make_model()
    assert mgr.load_checkpoint(model=m, optimizer=opt, expected_hashes=HASHES) is None


def test_hash_mismatch_fails_fast(mgr):
    """Muc 4.3: lech hash -> dung, TUYET DOI khong am tham train lai tu dau."""
    m, opt = make_model()
    st = C.TrainingState(run_id=mgr.run_id, session_id="s", current_epoch=1)
    mgr.save(model=m, optimizer=opt, state=st, hashes=HASHES)

    m2, opt2 = make_model()
    bad = {**HASHES, "feature_schema_hash": "DOI-ROI"}
    with pytest.raises(RuntimeError, match="khong khop cau hinh"):
        mgr.load_checkpoint(model=m2, optimizer=opt2, expected_hashes=bad)


def test_rng_state_is_restored(mgr):
    m, opt = make_model()
    torch.manual_seed(7)
    np.random.seed(7)
    st = C.TrainingState(run_id=mgr.run_id, session_id="s", current_epoch=1)
    mgr.save(model=m, optimizer=opt, state=st, hashes=HASHES)

    expected_torch = torch.randn(3)
    expected_np = np.random.rand(3)

    m2, opt2 = make_model()
    mgr.load_checkpoint(model=m2, optimizer=opt2, expected_hashes=HASHES)
    assert torch.allclose(torch.randn(3), expected_torch)
    assert np.allclose(np.random.rand(3), expected_np)


# --------------------------------------------------------- run registry (4.5)
def test_run_id_is_stable_across_sessions(store):
    r = C.RunRegistry(store)
    a, new_a = r.get_or_create("mddcc")
    b, new_b = r.get_or_create("mddcc")
    assert a == b, "muc 4.5: run_id giu nguyen tu epoch 1 den 100"
    assert new_a is True and new_b is False
    assert a.startswith("mddcc_")


def test_new_run_id_after_registry_cleared(store):
    r = C.RunRegistry(store)
    a, _ = r.get_or_create("mddcc")
    store.delete(C.RunRegistry.KEY)
    assert r.get() is None
    C.RunRegistry(store).get_or_create("mddcc")   # tao lai duoc


# ------------------------------------------------------------ time guard (4.7)
def test_time_guard_triggers_near_limit():
    import time
    g = C.TimeGuard(CFG, start=time.time() - (40800 - 1000))
    assert g.should_stop(), "con 1000s < guard 1200s -> phai dung"
    assert g.reason == "time_guard"


def test_time_guard_quiet_at_start():
    import time
    assert not C.TimeGuard(CFG, start=time.time()).should_stop()
