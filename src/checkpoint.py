"""Luu/nap checkpoint + training_state + history, dong bo S3 - muc 4.

Kaggle CHAC CHAN cat session giua chung (toi da 12 gio). Voi phuong an C thi
100 epoch can ~39 session, nen resume phai dung tuyet doi: khong train lai epoch
da xong, khong bo sot batch, khong am tham bat dau lai tu dau.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .s3io import ObjectStore, SafeWriter

LOG = logging.getLogger(__name__)

STATUS_RUNNING = "running"
STATUS_INTERRUPTED = "interrupted"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(prefix: str = "mddcc") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"


def new_session_id() -> str:
    """Moi lan khoi dong mot id rieng - muc 4.6.

    Them hau to ngau nhien vi dau thoi gian den giay VAN co the trung khi hai
    session khoi dong sat nhau; trung session_id se lam vach 'Resume' tren hinh
    C1 bi ve sai.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{os.urandom(2).hex()}"


# --------------------------------------------------------------- RNG state
def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng(state: dict) -> None:
    if not state:
        return
    if state.get("python") is not None:
        py = state["python"]
        random.setstate((py[0], tuple(py[1]), py[2]) if isinstance(py, list) else py)
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        t = state["torch"]
        torch.set_rng_state(t if isinstance(t, torch.Tensor)
                            else torch.tensor(t, dtype=torch.uint8))


# ----------------------------------------------------------- training state
@dataclass
class TrainingState:
    """Nguon su that cho GitHub Actions - muc 4.9."""

    run_id: str
    session_id: str
    current_epoch: int = 0            # epoch da HOAN THANH
    total_epochs: int = 100
    global_step: int = 0
    steps_done_in_epoch: int = 0
    status: str = STATUS_RUNNING
    exit_reason: str = ""
    updated_at_utc: str = field(default_factory=utc_now)
    restart_count: int = 0
    cache_build_seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["updated_at_utc"] = utc_now()
        d["is_complete"] = self.current_epoch >= self.total_epochs
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingState":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def is_complete(self) -> bool:
        return self.current_epoch >= self.total_epochs

    @property
    def next_epoch(self) -> int:
        """Epoch se chay tiep theo (1-based). Muc 4.4."""
        return self.current_epoch + 1


# ------------------------------------------------------------------ history
class History:
    """history.json APPEND-ONLY - muc 7.E1, E2. Khong bao gio ghi de khi resume."""

    def __init__(self, records: list[dict] | None = None):
        self.records: list[dict] = list(records or [])

    def append(self, record: dict) -> None:
        epoch = record["epoch"]
        existing = {r["epoch"] for r in self.records}
        if epoch in existing:
            raise RuntimeError(
                f"Epoch {epoch} da co trong history - dau hieu train lai epoch da xong. "
                "Muc 4.4 cam dieu nay."
            )
        self.records.append(record)
        self.records.sort(key=lambda r: r["epoch"])

    def validate_continuous(self) -> None:
        """Epoch phai lien tuc 1..n, khong thieu khong lap - muc 7.E2."""
        epochs = [r["epoch"] for r in self.records]
        if not epochs:
            return
        expected = list(range(1, len(epochs) + 1))
        if epochs != expected:
            missing = sorted(set(expected) - set(epochs))
            dup = sorted({e for e in epochs if epochs.count(e) > 1})
            raise RuntimeError(
                f"history.json khong lien tuc: co {len(epochs)} ban ghi, "
                f"thieu {missing}, lap {dup}"
            )

    @property
    def last_epoch(self) -> int:
        return max((r["epoch"] for r in self.records), default=0)

    def session_boundaries(self) -> list[int]:
        """Epoch bat dau moi session - de ve axvline 'Resume' (muc 7.E3)."""
        out, prev = [], None
        for r in self.records:
            sid = r.get("session_id")
            if prev is not None and sid != prev:
                out.append(r["epoch"])
            prev = sid
        return out

    def to_csv_rows(self) -> tuple[list[str], list[list]]:
        keys: list[str] = []
        for r in self.records:
            for k in r:
                if k not in keys:
                    keys.append(k)
        return keys, [[r.get(k, "") for k in keys] for r in self.records]


# --------------------------------------------------------------- manager
class CheckpointManager:
    """Luu/nap checkpoint va dong bo len S3 an toan."""

    def __init__(self, store: ObjectStore, cfg: dict, run_id: str,
                 local_dir: Path):
        self.store = store
        self.cfg = cfg
        self.run_id = run_id
        self.local = Path(local_dir)
        self.local.mkdir(parents=True, exist_ok=True)

        c = cfg["checkpoint"]
        self.keep_last_n = int(c.get("keep_last_n", 3))
        self.permanent_every = int(c.get("permanent_every_n_epochs", 10))
        self.final_name = c.get("final_name", "final_model_epoch_100.pt")
        self.on_mismatch = c.get("on_hash_mismatch", "fail_fast")

        layout = cfg.get("s3", {}).get("layout", {})
        self.dir_ckpt = layout.get("checkpoints", "checkpoints")
        self.dir_metrics = layout.get("metrics", "metrics")
        self.writer = SafeWriter(
            store, cfg.get("s3", {}).get("safe_upload", {}).get("tmp_prefix", "_tmp"))

    # ----------------------------------------------------------------- keys
    def key(self, folder: str, name: str) -> str:
        return f"{self.run_id}/{folder}/{name}"

    @property
    def state_key(self) -> str:
        return self.key(self.dir_ckpt, "training_state.json")

    @property
    def last_key(self) -> str:
        return self.key(self.dir_ckpt, "last_checkpoint.pt")

    @property
    def history_key(self) -> str:
        return self.key(self.dir_metrics, "history.json")

    # ---------------------------------------------------------------- luu
    def save(self, *, model, optimizer, state: TrainingState, hashes: dict,
             is_final: bool = False) -> dict:
        """Luu checkpoint len local roi day len S3 bang SafeWriter."""
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": state.current_epoch,
            "global_step": state.global_step,
            "steps_done_in_epoch": state.steps_done_in_epoch,
            "rng_state": capture_rng(),
            "run_id": state.run_id,
            "session_id": state.session_id,
            **hashes,
        }
        tmp = self.local / "last_checkpoint.pt"
        torch.save(payload, tmp)

        info = self.writer.put_file(tmp, self.last_key)
        self.writer.put_json(state.to_dict(), self.state_key)

        # Ban vinh vien moi permanent_every epoch (muc 4.2)
        if state.steps_done_in_epoch == 0 and state.current_epoch > 0 \
                and state.current_epoch % self.permanent_every == 0:
            self.writer.put_file(
                tmp, self.key(self.dir_ckpt, f"epoch_{state.current_epoch:03d}.pt"))
            self._prune_epoch_snapshots(state.current_epoch)

        if is_final:
            self.writer.put_file(tmp, self.key(self.dir_ckpt, self.final_name))
            LOG.info("  da luu %s", self.final_name)
        return info

    def _prune_epoch_snapshots(self, current: int) -> None:
        """Giu keep_last_n ban gan nhat; ban o boi so cua 10 luu vinh vien."""
        prefix = f"{self.run_id}/{self.dir_ckpt}/epoch_"
        keys = [k for k in self.store.list_keys(prefix) if k.endswith(".pt")]
        snaps = sorted(keys)
        # Tat ca deu la boi so cua permanent_every -> giu lai het theo muc 4.2.
        # Ham nay chi don ban thua neu permanent_every bi doi nho hon.
        if self.permanent_every >= 10:
            return
        for k in snaps[:-self.keep_last_n]:
            self.store.delete(k)

    # ---------------------------------------------------------------- nap
    def load_state(self) -> TrainingState | None:
        d = self.store.get_json_or_none(self.state_key)
        return TrainingState.from_dict(d) if d else None

    def load_history(self) -> History:
        d = self.store.get_json_or_none(self.history_key)
        h = History(d or [])
        h.validate_continuous()
        return h

    def save_history(self, history: History) -> None:
        self.writer.put_json(history.records, self.history_key)

    def load_checkpoint(self, *, model, optimizer, expected_hashes: dict) -> dict | None:
        """Nap checkpoint moi nhat va kiem tra hash - muc 4.3."""
        if not self.store.exists(self.last_key):
            LOG.info("Khong co checkpoint tren store -> bat dau tu epoch 1")
            return None

        local = self.local / "resume_checkpoint.pt"
        self.store.get_file(self.last_key, local)
        ckpt = torch.load(local, map_location="cpu", weights_only=False)

        self._verify_hashes(ckpt, expected_hashes)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        restore_rng(ckpt.get("rng_state", {}))
        LOG.info("Da nap checkpoint: epoch=%s global_step=%s steps_in_epoch=%s",
                 ckpt.get("epoch"), ckpt.get("global_step"),
                 ckpt.get("steps_done_in_epoch"))
        return ckpt

    def _verify_hashes(self, ckpt: dict, expected: dict) -> None:
        bad = {k: (ckpt.get(k), v) for k, v in expected.items() if ckpt.get(k) != v}
        if not bad:
            return
        lines = "\n".join(
            f"    {k}: checkpoint={got!s:.16}... hien tai={want!s:.16}..."
            for k, (got, want) in bad.items())
        msg = ("Checkpoint khong khop cau hinh hien tai (muc 4.3):\n" + lines +
               "\n  Nguyen nhan thuong gap: doi tap cot, doi scaler, doi kien truc.\n"
               "  KHONG duoc am tham train lai tu dau - hoac khoi phuc dung cau hinh,\n"
               "  hoac xoa current_run_id.json de bat dau mot run_id MOI.")
        if self.on_mismatch == "fail_fast":
            raise RuntimeError(msg)
        LOG.warning(msg)


# ------------------------------------------------------------- run_id chung
class RunRegistry:
    """current_run_id.json - GitHub Actions va session sau doc de biet run nao.

    Muc 4.5: run_id giu nguyen tu epoch 1 den 100. Notebook chay lai lan thu N
    chi tiep tuc, khong tao run_id moi, tru khi file nay bi xoa.
    """

    KEY = "current_run_id.json"

    def __init__(self, store: ObjectStore, key: str | None = None):
        # Moi bien the mot khoa RIENG. Neu hai run song song dung chung khoa nay
        # thi chung se cuop run_id cua nhau va ghi de checkpoint len nhau.
        self.store = store
        self.key = key or self.KEY
        self.writer = SafeWriter(store)

    def get_or_create(self, prefix: str = "mddcc") -> tuple[str, bool]:
        d = self.store.get_json_or_none(self.key)
        if d and d.get("run_id"):
            return d["run_id"], False
        run_id = new_run_id(prefix)
        self.writer.put_json(
            {"run_id": run_id, "created_at_utc": utc_now()}, self.key)
        LOG.info("Tao run_id moi: %s", run_id)
        return run_id, True

    def get(self) -> str | None:
        d = self.store.get_json_or_none(self.key)
        return d.get("run_id") if d else None


# ---------------------------------------------------------------- time guard
class TimeGuard:
    """Thoat chu dong truoc khi Kaggle cat session - muc 4.7."""

    def __init__(self, cfg: dict, start: float | None = None):
        s = cfg.get("session", {})
        self.limit = float(s.get("time_limit_seconds", 40800))
        self.guard = float(s.get("exit_guard_seconds", 1200))
        self.reason = s.get("exit_reason_on_guard", "time_guard")
        self.start = start if start is not None else time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def remaining(self) -> float:
        return self.limit - self.elapsed

    def should_stop(self) -> bool:
        return self.remaining <= self.guard

    def summary(self) -> dict:
        return {"elapsed_seconds": round(self.elapsed, 1),
                "remaining_seconds": round(self.remaining, 1),
                "limit_seconds": self.limit,
                "guard_seconds": self.guard}
