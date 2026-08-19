#!/usr/bin/env python3
"""Sinh lai TOAN BO hinh + CSV tu artifact da luu - muc 7.A2.

Chay doc lap, KHONG can train lai:
    python make_report.py --run-dir s3://bucket/prefix/mddcc_20260819-0251
    python make_report.py --run-dir ./_localstore/mddcc_20260819-0251
    python make_report.py --run-id mddcc_20260819-0251        # doc S3 tu env

Ly do ton tai: Kaggle cat session bat ky luc nao, va khi can sua nhan/mau hinh
cho luan van thi khong duoc phep huan luyen lai 39 session.

Buoc danh gia cuoi (--evaluate) can them checkpoint + du lieu goc; neu chi co
artifact metric thi van ve duoc C1-C9 va C14.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import viz                                    # noqa: E402
from src.s3io import LocalStore, S3Store               # noqa: E402

LOG = logging.getLogger("mddcc.report")


# ------------------------------------------------------------------ nguon
class RunArtifacts:
    """Doc artifact tu S3 hoac thu muc cuc bo, cung mot giao dien."""

    def __init__(self, store, run_id: str):
        self.store = store
        self.run_id = run_id

    @classmethod
    def from_arg(cls, run_dir: str | None, run_id: str | None) -> "RunArtifacts":
        if run_dir and run_dir.startswith("s3://"):
            rest = run_dir[5:].rstrip("/")
            bucket, _, key = rest.partition("/")
            prefix, _, rid = key.rpartition("/")
            return cls(S3Store(bucket, prefix), rid)
        if run_dir:
            p = Path(run_dir).resolve()
            return cls(LocalStore(p.parent), p.name)
        if run_id:
            import os
            bucket = os.environ.get("S3_BUCKET", "").strip()
            if not bucket:
                raise SystemExit("Can --run-dir hoac bien moi truong S3_BUCKET")
            return cls(S3Store(bucket, os.environ.get("S3_PREFIX", "")), run_id)
        raise SystemExit("Phai truyen --run-dir hoac --run-id")

    def key(self, *parts: str) -> str:
        return "/".join([self.run_id, *parts])

    def json(self, *parts: str):
        return self.store.get_json(self.key(*parts))

    def json_or_none(self, *parts: str):
        try:
            return self.store.get_json_or_none(self.key(*parts))
        except Exception:                              # noqa: BLE001
            return None

    def npy(self, *parts: str) -> np.ndarray | None:
        try:
            raw = self.store.get_bytes(self.key(*parts))
        except Exception:                              # noqa: BLE001
            return None
        return np.load(io.BytesIO(raw), allow_pickle=False)


# ------------------------------------------------------------------ sinh hinh
def build_report(art: RunArtifacts, out_dir: Path, *, upload: bool = False) -> dict:
    from src.s3io import SafeWriter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made: dict[str, list[str]] = {}

    label_map = art.json_or_none("config", "label_mapping.json") or {}
    classes = label_map.get("classes", [])
    benign_index = label_map.get("benign_index", 0)
    run_config = art.json_or_none("config", "run_config.json") or {}
    final_epoch = run_config.get("epochs", 100)
    history = art.json_or_none("metrics", "history.json") or []

    def record(name, paths):
        if paths:
            made[name] = [str(p) for p in paths]
            LOG.info("  [ok] %s", name)
        else:
            LOG.warning("  [bo qua] %s - thieu du lieu dau vao", name)

    # ---- C1, C2, C9: chi can history.json
    record("C1 learning_curves", viz.safe(
        viz.plot_learning_curves, history, out_dir, art.run_id, final_epoch))
    record("C2 lr_schedule", viz.safe(
        viz.plot_lr_schedule, history, out_dir, art.run_id, final_epoch))
    record("C9 epoch_time", viz.safe(
        viz.plot_epoch_time, history, out_dir, art.run_id, final_epoch))

    # ---- C8: phan bo lop tu sample_manifest.json
    manifest = art.json_or_none("config", "sample_manifest.json") or {}
    counts = manifest.get("per_split_class_counts")
    if counts and classes:
        record("C8 class_distribution", viz.safe(
            viz.plot_class_distribution, counts, classes, out_dir,
            art.run_id, final_epoch))

    # ---- C3, C4, C5, C6, C7: can y_true/y_prob hoac confusion matrix da luu
    y_true = art.npy("raw", "y_true.npy")
    y_prob = art.npy("raw", "y_prob.npy")
    test_metrics = art.json_or_none("metrics", "test_metrics.json")

    if y_true is not None and y_prob is not None and classes:
        from src.evaluate import compute_curves, evaluate_full

        res = evaluate_full(y_true, y_prob, classes, benign_index)
        record("C3 confusion_matrix", viz.safe(
            viz.plot_confusion_matrix, res.cm, classes, out_dir,
            art.run_id, final_epoch))
        record("C4 confusion_matrix_raw", viz.safe(
            viz.plot_confusion_matrix_raw, res.cm, classes, out_dir,
            art.run_id, final_epoch))
        record("C7 per_class_metrics", viz.safe(
            viz.plot_per_class_metrics, res.per_class, out_dir,
            art.run_id, final_epoch))

        # muc 7.B4: > 2 trieu mau thi lay mau phan tang CHI de ve ROC/PR
        n = y_true.size
        cap = int(run_config.get("full_config", {})
                  .get("evaluate", {}).get("roc_pr_max_samples", 2_000_000))
        idx = np.arange(n)
        if n > cap:
            from src.explain import stratified_sample
            idx = stratified_sample(y_true, idx, cap, seed=42)
            LOG.info("  ROC/PR ve tren %d/%d mau (metric van tinh tren toan bo)",
                     idx.size, n)
        roc, pr = compute_curves(y_true[idx], y_prob[idx], classes)
        record("C5 roc_curves", viz.safe(
            viz.plot_roc_curves, roc, classes, out_dir, art.run_id,
            final_epoch, idx.size))
        record("C6 pr_curves", viz.safe(
            viz.plot_pr_curves, pr, classes, out_dir, art.run_id,
            final_epoch, idx.size))
        del roc, pr
    else:
        LOG.warning("  khong co raw/y_true.npy + y_prob.npy -> bo qua C3-C7")

    # ---- C10, C11, C12, C13: tu artifact explainability da luu
    for name, fn, key in [
        ("C11 branch_ablation", viz.plot_branch_ablation, "branch_ablation.json"),
        ("C12 permutation_importance", viz.plot_permutation_importance,
         "permutation_importance.json"),
        ("C13 shap_feature_importance", viz.plot_shap_importance,
         "shap_feature_importance.json"),
    ]:
        rows = art.json_or_none("explainability", key)
        if rows:
            record(name, viz.safe(fn, rows, out_dir, art.run_id, final_epoch))

    energy = art.json_or_none("explainability", "wavelet_subband_energy.json")
    if energy and classes:
        record("C10 wavelet_subband_energy", viz.safe(
            viz.plot_wavelet_subband_energy, np.array(energy["energy"]), classes,
            energy.get("subbands", ["cD1", "cD2", "cD3", "cA3"]),
            out_dir, art.run_id, final_epoch))

    # ---- C14: doi chieu bai bao
    if test_metrics and "binary" in test_metrics:
        from src.evaluate import paper_comparison_rows
        rows = paper_comparison_rows(test_metrics["binary"])
        plot_rows = [{"metric": r["metric"], "ours_binary": r["ours_binary"],
                      "paper_table9": r["paper_table9"]} for r in rows]
        record("C14 paper_comparison", viz.safe(
            viz.plot_paper_comparison, plot_rows, out_dir, art.run_id, final_epoch))
        viz.save_csv(rows, out_dir, "paper_comparison")

    # ---- history.csv (muc 7.F6)
    if history:
        keys: list[str] = []
        for r in history:
            for k in r:
                if k not in keys:
                    keys.append(k)
        viz.save_csv([{k: r.get(k, "") for k in keys} for r in history],
                     out_dir, "history", keys)

    if upload:
        writer = SafeWriter(art.store)
        for p in sorted(out_dir.iterdir()):
            if p.suffix in (".png", ".pdf"):
                writer.put_file(p, art.key("figures", p.name))
            elif p.suffix == ".csv":
                writer.put_file(p, art.key("metrics", p.name))
        LOG.info("Da upload toan bo hinh + CSV len store")

    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sinh lai bao cao MDDCC tu artifact")
    ap.add_argument("--run-dir", default=None,
                    help="s3://bucket/prefix/<run_id> hoac duong dan cuc bo")
    ap.add_argument("--run-id", default=None, help="doc S3 tu bien moi truong")
    ap.add_argument("--out", type=Path, default=Path("report"))
    ap.add_argument("--upload", action="store_true",
                    help="Day hinh + CSV nguoc len store (muc 7.I3)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-7s %(message)s", stream=sys.stdout)

    art = RunArtifacts.from_arg(args.run_dir, args.run_id)
    LOG.info("run_id = %s -> %s", art.run_id, args.out)
    made = build_report(art, args.out, upload=args.upload)

    LOG.info("=" * 60)
    LOG.info("Da sinh %d nhom hinh:", len(made))
    for name in made:
        LOG.info("  - %s", name)
    if not made:
        LOG.error("KHONG sinh duoc hinh nao - kiem tra run_id va noi dung store")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
