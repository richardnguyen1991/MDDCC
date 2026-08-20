"""Vong lap huan luyen MDDCC - muc 2.D, 4, 7.E.

KHONG chua matplotlib (muc 7.A1). Ve hinh do make_report.py / viz.py lo.

Chay:
    python -m src.train --config configs/mddcc.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .checkpoint import (STATUS_COMPLETED, STATUS_INTERRUPTED, STATUS_RUNNING,
                         CheckpointManager, History, RunRegistry, TimeGuard,
                         TrainingState, new_session_id, utc_now)
from .model import MDDCCLoss, build_model, build_optimizer, grad_norm
from .s3io import SafeWriter, store_from_env
from .wavelet import geometry_hash

LOG = logging.getLogger("mddcc.train")


# ------------------------------------------------------------------ tien ich
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def peak_rss_mb() -> float:
    """Peak RSS - de chung minh khong tran RAM tren Kaggle (muc 11.A.4)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:                       # Windows
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e6
        except Exception:
            return float("nan")


def assert_cpu_only(cfg: dict) -> torch.device:
    """Muc 2.D: CPU bat buoc, cam moi loi goi .cuda()."""
    dev = cfg["train"].get("device", "cpu")
    if dev != "cpu":
        raise RuntimeError(f"train.device={dev!r} - muc 2.D bat buoc 'cpu'.")
    threads = int(cfg["train"].get("torch_num_threads", 4))
    torch.set_num_threads(threads)
    LOG.info("Device=cpu, torch threads=%d", threads)
    return torch.device("cpu")


# --------------------------------------------------------------- danh gia
@dataclass
class EpochMetrics:
    loss: float
    mse: float
    std_reg: float
    accuracy: float
    macro_f1: float

    def as_dict(self, prefix: str) -> dict:
        return {f"{prefix}_total_loss": round(self.loss, 8),
                f"{prefix}_mse_loss": round(self.mse, 8),
                f"{prefix}_std_reg": round(self.std_reg, 8),
                f"{prefix}_accuracy": round(self.accuracy, 8),
                f"{prefix}_macro_f1": round(self.macro_f1, 8)}


class ConfusionAccumulator:
    """Cong don confusion matrix theo chunk - muc 7.B5, khong giu ca y_pred."""

    def __init__(self, num_classes: int):
        self.n = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        idx = y_true.astype(np.int64) * self.n + y_pred.astype(np.int64)
        self.cm += np.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    @property
    def accuracy(self) -> float:
        total = self.cm.sum()
        return float(np.trace(self.cm) / total) if total else 0.0

    def per_class_f1(self) -> np.ndarray:
        tp = np.diag(self.cm).astype(np.float64)
        fp = self.cm.sum(axis=0) - tp
        fn = self.cm.sum(axis=1) - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        return f1

    def macro_f1(self, present_only: bool = True) -> float:
        """Macro-F1. present_only bo qua lop khong co mau that (muc 7.G3)."""
        f1 = self.per_class_f1()
        if present_only:
            support = self.cm.sum(axis=1)
            f1 = f1[support > 0]
        return float(f1.mean()) if f1.size else 0.0


@torch.no_grad()
def evaluate(model, dataset, indices, *, batch_size: int, loss_fn,
             num_classes: int) -> EpochMetrics:
    """Danh gia mot split. model.eval() -> dropout tat (muc 6)."""
    was_training = model.training
    model.eval()
    cm = ConfusionAccumulator(num_classes)
    tot_loss = tot_mse = tot_reg = 0.0
    n_batches = 0

    for start in range(0, indices.size, batch_size):
        rows = indices[start:start + batch_size]
        xb, yb = dataset.batch(rows)
        probs = model(xb)
        parts = loss_fn(probs, yb, model)
        tot_loss += float(parts.total.detach())
        tot_mse += float(parts.mse.detach())
        tot_reg += float(parts.std_reg.detach())
        n_batches += 1
        cm.update(yb.numpy(), probs.argmax(dim=1).numpy())

    if was_training:
        model.train()
    d = max(n_batches, 1)
    return EpochMetrics(tot_loss / d, tot_mse / d, tot_reg / d,
                        cm.accuracy, cm.macro_f1())


# ------------------------------------------------------------------- epoch
def train_one_epoch(*, model, optimizer, loss_fn, dataset, sampler, epoch: int,
                    num_classes: int, state: TrainingState, ckpt_mgr, hashes: dict,
                    guard: TimeGuard, checkpoint_interval_steps: int,
                    skip_batches: int = 0) -> tuple[EpochMetrics, float, bool]:
    """Chay mot epoch. Tra ve (metrics, grad_norm_mean, bi_ngat_giua_chung)."""
    model.train()
    cm = ConfusionAccumulator(num_classes)
    tot_loss = tot_mse = tot_reg = tot_gn = 0.0
    n_batches = 0
    interrupted = False

    if skip_batches:
        LOG.info("  resume giua epoch: bo qua %d batch dau", skip_batches)

    for local_step, rows in enumerate(sampler.batches(epoch, skip=skip_batches),
                                      start=skip_batches):
        xb, yb = dataset.batch(rows)

        optimizer.zero_grad(set_to_none=True)
        probs = model(xb)
        parts = loss_fn(probs, yb, model)
        parts.total.backward()
        gn = grad_norm(model)
        optimizer.step()

        tot_loss += float(parts.total.detach())
        tot_mse += float(parts.mse.detach())
        tot_reg += float(parts.std_reg.detach())
        tot_gn += gn
        n_batches += 1
        cm.update(yb.numpy(), probs.detach().argmax(dim=1).numpy())

        state.global_step += 1
        state.steps_done_in_epoch = local_step + 1

        # Checkpoint theo step - muc 4.1
        if checkpoint_interval_steps and \
                state.steps_done_in_epoch % checkpoint_interval_steps == 0:
            state.status = STATUS_RUNNING
            ckpt_mgr.save(model=model, optimizer=optimizer, state=state,
                          hashes=hashes)
            LOG.info("  step %d/%d | loss %.6f (mse %.6f + reg %.6f) | grad %.4f",
                     state.steps_done_in_epoch, sampler.n_batches(),
                     float(parts.total.detach()), float(parts.mse.detach()),
                     float(parts.std_reg.detach()), gn)

        # Thoat chu dong truoc khi Kaggle cat - muc 4.7
        if guard.should_stop():
            LOG.warning("  time_guard: con %.0f s, luu va thoat giua epoch %d",
                        guard.remaining, epoch)
            state.status = STATUS_INTERRUPTED
            state.exit_reason = guard.reason
            ckpt_mgr.save(model=model, optimizer=optimizer, state=state,
                          hashes=hashes)
            interrupted = True
            break

    d = max(n_batches, 1)
    metrics = EpochMetrics(tot_loss / d, tot_mse / d, tot_reg / d,
                           cm.accuracy, cm.macro_f1())
    return metrics, tot_gn / d, interrupted


# -------------------------------------------------------------------- main
def run(cfg: dict, *, config_path: Path, local_store_root: Path | None = None,
        max_epochs_override: int | None = None) -> int:
    t_session = time.time()
    session_id = new_session_id()

    set_seed(cfg["experiment"]["seed"],
             deterministic=cfg["experiment"].get("deterministic_algorithms", True))
    assert_cpu_only(cfg)

    store = store_from_env(cfg, local_root=local_store_root)
    from .config import run_id_key, variant_of

    variant = variant_of(cfg)
    registry = RunRegistry(store, key=run_id_key(cfg))
    run_id, is_new = registry.get_or_create(cfg["experiment"].get("run_id_prefix", "mddcc"))
    LOG.info("bien the=%s | run_id=%s (moi=%s) session_id=%s",
             variant, run_id, is_new, session_id)

    # ---------------------------------------------------------------- du lieu
    work = Path(cfg["data"]["cache_dir"]).parent / "mddcc_work"
    prepared = D.prepare_dataset(cfg, work / "artifacts")
    geom, schema, labels, splits = (prepared.geom, prepared.schema,
                                    prepared.labels, prepared.splits)
    num_classes = labels.num_classes
    LOG.info("F=%d -> F_swt=%d -> S=%d | anh %s | %d lop | cache %.1fs",
             geom.n_features, geom.n_padded, geom.side, geom.image_shape,
             num_classes, prepared.cache_build_seconds)

    # ----------------------------------------------------------------- model
    model = build_model(cfg, side=geom.side, num_classes=num_classes)
    optimizer = build_optimizer(cfg, model)
    loss_fn = MDDCCLoss(cfg, num_classes)
    LOG.info("Model: %s -> flatten %d | %d tham so | %.3f MB",
             model.feature_map_shape, model.flatten_dim,
             model.n_parameters(), model.size_mb())

    hashes = {
        "params_hash": model.params_hash(),
        "feature_schema_hash": schema.hash,
        "scaler_hash": prepared.scaler.hash,
        "wavelet_geometry_hash": geometry_hash(geom, schema.feature_columns),
    }

    ckpt_mgr = CheckpointManager(store, cfg, run_id, work / "checkpoints")
    writer = SafeWriter(store)
    total_epochs = max_epochs_override or int(cfg["train"]["epochs"])

    # -------------------------------------------------------------- resume
    ckpt = ckpt_mgr.load_checkpoint(model=model, optimizer=optimizer,
                                    expected_hashes=hashes)
    state = ckpt_mgr.load_state() or TrainingState(
        run_id=run_id, session_id=session_id, total_epochs=total_epochs)
    state.session_id = session_id
    state.total_epochs = total_epochs
    state.cache_build_seconds = prepared.cache_build_seconds
    if ckpt is not None:
        state.restart_count += 1
    skip_batches = int(state.steps_done_in_epoch or 0)

    history = ckpt_mgr.load_history()
    if history.last_epoch != state.current_epoch:
        raise RuntimeError(
            f"history.json den epoch {history.last_epoch} nhung training_state bao "
            f"{state.current_epoch} - khong dong bo, dung lai de kiem tra thu cong.")

    if state.is_complete:
        LOG.info("current_epoch=%d >= %d -> DA XONG, khong train them (muc 4.8)",
                 state.current_epoch, total_epochs)
        return 0

    # ---------------------------------------------------------- ghi cau hinh
    _upload_run_config(writer, cfg, run_id, config_path, model, geom, schema,
                       prepared, hashes, total_epochs, splits)
    _upload_data_artifacts(writer, prepared, run_id, cfg)

    # ------------------------------------------------------------ dataloader
    tcfg = cfg["train"]
    batch_size = int(tcfg["batch_size"])
    dl = tcfg.get("dataloader", {})
    train_ds = D.MDDCCDataset(prepared.cache_path, splits.train, labels.codes, geom)
    val_ds = D.MDDCCDataset(prepared.cache_path, splits.val, labels.codes, geom)
    sampler = D.BatchSampler(splits.train.size, batch_size,
                             seed=cfg["experiment"]["seed"],
                             shuffle=dl.get("shuffle_train", True),
                             drop_last=dl.get("drop_last", False))
    guard = TimeGuard(cfg, start=t_session)
    interval = int(cfg["checkpoint"].get("interval_steps", 200))

    LOG.info("Train %d mau (%d step/epoch) | Val %d mau | epoch %d -> %d",
             splits.train.size, sampler.n_batches(), splits.val.size,
             state.next_epoch, total_epochs)

    # ------------------------------------------------------------ vong lap
    reported_first = False
    for epoch in range(state.next_epoch, total_epochs + 1):
        t0 = time.time()
        ts_start = utc_now()
        resumed_mid_epoch = skip_batches > 0

        metrics, gn_mean, interrupted = train_one_epoch(
            model=model, optimizer=optimizer, loss_fn=loss_fn, dataset=train_ds,
            sampler=sampler, epoch=epoch, num_classes=num_classes, state=state,
            ckpt_mgr=ckpt_mgr, hashes=hashes, guard=guard,
            checkpoint_interval_steps=interval, skip_batches=skip_batches)
        resumed_batches, skip_batches = skip_batches, 0

        if interrupted:
            ckpt_mgr.save_history(history)
            LOG.warning("Thoat theo time_guard giua epoch %d, exit 0", epoch)
            return 0

        val = evaluate(model, val_ds, splits.val, batch_size=batch_size,
                       loss_fn=loss_fn, num_classes=num_classes)

        # Epoch da HOAN THANH -> cap nhat state truoc khi luu
        state.current_epoch = epoch
        state.steps_done_in_epoch = 0
        state.status = (STATUS_COMPLETED if epoch >= total_epochs else STATUS_RUNNING)
        state.exit_reason = ""

        secs = time.time() - t0
        record = {
            "epoch": epoch,
            "session_id": session_id,
            "timestamp_start": ts_start,
            "timestamp_end": utc_now(),
            "learning_rate": float(cfg["optim"]["learning_rate"]),
            **metrics.as_dict("train"),
            **val.as_dict("val"),
            "grad_norm_mean": round(gn_mean, 8),
            "epoch_seconds": round(secs, 2),
            "samples_per_second": round(splits.train.size / max(secs, 1e-9), 2),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "global_step": state.global_step,
            "is_final_epoch": epoch >= total_epochs,
            # Neu epoch nay bi resume giua chung thi metric train chi tinh tren
            # phan batch chay trong session nay, khong phai ca epoch. Ghi ro de
            # khong doc nham learning curve.
            "train_metrics_partial": resumed_mid_epoch,
            "resumed_after_batches": resumed_batches if resumed_mid_epoch else 0,
        }
        history.append(record)
        history.validate_continuous()

        ckpt_mgr.save(model=model, optimizer=optimizer, state=state, hashes=hashes,
                      is_final=epoch >= total_epochs)
        ckpt_mgr.save_history(history)

        LOG.info("epoch %d/%d | train loss %.6f (mse %.6f + reg %.6f) acc %.4f f1 %.4f "
                 "| val loss %.6f acc %.4f f1 %.4f | grad %.4f | %.1fs",
                 epoch, total_epochs, metrics.loss, metrics.mse, metrics.std_reg,
                 metrics.accuracy, metrics.macro_f1, val.loss, val.accuracy,
                 val.macro_f1, gn_mean, secs)

        if not reported_first:
            _report_first_epoch(secs, sampler.n_batches(), total_epochs,
                                prepared.cache_build_seconds, cfg, record)
            reported_first = True

        if guard.should_stop() and epoch < total_epochs:
            state.status = STATUS_INTERRUPTED
            state.exit_reason = guard.reason
            ckpt_mgr.save(model=model, optimizer=optimizer, state=state, hashes=hashes)
            LOG.warning("time_guard sau epoch %d -> thoat 0 de session sau tiep tuc",
                        epoch)
            return 0

    LOG.info("HOAN THANH %d epoch. run_id=%s", total_epochs, run_id)
    return 0


def _report_first_epoch(secs: float, steps: int, total_epochs: int,
                        cache_seconds: float, cfg: dict, record: dict) -> None:
    """Muc 2.D: in ETA va so session can thiet ngay sau epoch dau tien."""
    session_limit = float(cfg.get("session", {}).get("time_limit_seconds", 40800))
    total = secs * total_epochs
    LOG.info("=" * 68)
    LOG.info("BAO CAO SAU EPOCH DAU TIEN")
    LOG.info("  thoi gian 1 epoch      : %.1f s (%.2f h)", secs, secs / 3600)
    LOG.info("  thoi gian 1 step       : %.3f s (%d step)", secs / max(steps, 1), steps)
    LOG.info("  uoc tinh %3d epoch     : %.1f h (%.1f ngay)",
             total_epochs, total / 3600, total / 86400)
    LOG.info("  so session Kaggle can  : %d", int(np.ceil(total / session_limit)))
    LOG.info("  cache_build_seconds    : %.1f s (moi session phai dung lai)",
             cache_seconds)
    LOG.info("  peak RSS               : %.0f MB", record["peak_rss_mb"])
    LOG.info("=" * 68)


def _upload_run_config(writer, cfg, run_id, config_path, model, geom, schema,
                       prepared, hashes, total_epochs, splits) -> None:
    """run_config.json - muc 2.D. Ghi du cac khoa nghiem thu doi hoi."""
    import platform
    import sklearn
    import pywt

    layout = cfg.get("s3", {}).get("layout", {})
    cdir = layout.get("config", "config")

    run_config = {
        "run_id": run_id,
        "variant": cfg["experiment"].get("variant", "full"),
        "experiment_role": cfg["experiment"]["role"],
        "created_at_utc": utc_now(),
        "config_file": str(config_path),
        "epochs": total_epochs,
        "early_stopping": False,
        "imbalance_handling": "none",
        "feature_selection": "none",
        "use_all_features": True,
        "batch_size": int(cfg["train"]["batch_size"]),
        "learning_rate": float(cfg["optim"]["learning_rate"]),
        "optimizer": cfg["optim"]["name"],
        "loss": cfg["loss"]["name"],
        "mse_reduction": cfg["loss"].get("mse_reduction", "mean_elements"),
        "lambda_std": float(cfg["loss"]["std_regularizer"]["lambda_std"]),
        "wavelet": cfg["wavelet"]["name"],
        "wavelet_level": int(cfg["wavelet"]["level"]),
        "swt": True,
        "device": "cpu",
        "seed": cfg["experiment"]["seed"],
        "deterministic_algorithms": cfg["experiment"].get("deterministic_algorithms", True),
        "torch_num_threads": torch.get_num_threads(),
        "split_sizes": splits.sizes(),
        "num_classes": prepared.labels.num_classes,
        "model": model.spec(),
        "wavelet_geometry": geom.to_dict(),
        "hashes": hashes,
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pywt": pywt.__version__,
            "sklearn": sklearn.__version__,
            "python": platform.python_version(),
        },
        "platform": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "system": platform.system(),
            "cpu_count": os.cpu_count(),
        },
        "deviations_from_paper": cfg.get("deviations_from_paper", {}),
        "full_config": cfg,
    }
    writer.put_json(run_config, f"{run_id}/{cdir}/run_config.json")
    writer.put_json(model.spec(), f"{run_id}/{cdir}/model_params.json")


def _upload_data_artifacts(writer, prepared, run_id, cfg) -> None:
    layout = cfg.get("s3", {}).get("layout", {})
    cdir = layout.get("config", "config")
    cache_dir = layout.get("cache", "cache")
    for name, payload in prepared.artifacts.items():
        folder = cache_dir if name == "cache_manifest.json" else cdir
        writer.put_json(payload, f"{run_id}/{folder}/{name}")


def main(argv=None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description="Huan luyen MDDCC")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--variant", default=None,
                    help="Ten overlay trong configs/variants/ (vd capped10m)")
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--local-store", type=Path, default=None,
                    help="Dung LocalStore thay S3 (chi de chay thu)")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Ghi de so epoch - CHI dung khi chay thu, khong dung cho run chinh")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    setup_logging(args.log_level)
    from .config import load_config

    cfg = load_config(args.config, args.variant)
    if args.input_dir:
        cfg["data"]["kaggle_input_dir"] = str(args.input_dir)
    if args.cache_dir:
        cfg["data"]["cache_dir"] = str(args.cache_dir)

    try:
        return run(cfg, config_path=args.config, local_store_root=args.local_store,
                   max_epochs_override=args.max_epochs)
    except KeyboardInterrupt:
        LOG.warning("Bi ngat boi nguoi dung - checkpoint gan nhat van con tren store")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
