#!/usr/bin/env python3
"""Dien tap toan bo chuoi truoc khi dot 39 session Kaggle - muc 11.A.1.

    python scripts/rehearse_resume.py --work-dir /tmp/mddcc_rehearsal

Chay tren du lieu tong hop nho nhung di DUNG duong ma run that di:
  1. Train, bi ngat giua epoch 1 boi time_guard (gia lap Kaggle cat session)
  2. Chay lai -> phai tiep tu dung step ke tiep, khong lap khong bo sot
  3. Chay den het 2 epoch
  4. Danh gia cuoi + sinh bao cao
  5. Kiem tra 12 tieu chi nghiem thu

Muc dich: phat hien loi trong chuoi resume TRUOC khi cam ket 18 ngay compute.
Neu buoc nao that bai thi script thoat != 0.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

N_ROWS = 4000
N_FEATURES = 81
CLASSES = ["BENIGN", "Syn", "TFTP", "UDP-lag", "WebDDoS"]


def make_dataset(out: Path) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(0)
    out.mkdir(parents=True, exist_ok=True)
    for f in range(2):
        y = rng.choice(len(CLASSES), N_ROWS, p=[.15, .3, .3, .2, .05])
        cols = {}
        for i in range(N_FEATURES):
            v = rng.normal(0, 1, N_ROWS)
            if i < 25:
                v += y * 2.0            # cho vai cot mang tin hieu that
            cols[f" Feat{i}"] = pa.array(v)
        cols[" Label"] = pa.array([CLASSES[c] for c in y])
        pq.write_table(pa.table(cols), out / f"p{f}.parquet", row_group_size=1000)


def write_config(dst: Path, data_dir: Path, cache_dir: Path, *,
                 time_limit: int | None) -> Path:
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "mddcc.yaml").read_text(encoding="utf-8"))
    cfg["data"]["kaggle_input_dir"] = str(data_dir)
    cfg["data"]["cache_dir"] = str(cache_dir)
    cfg["train"]["epochs"] = 2
    cfg["train"]["batch_size"] = 256          # de co nhieu step trong mot epoch
    cfg["train"]["torch_num_threads"] = 4
    cfg["checkpoint"]["interval_steps"] = 2
    cfg["explain"]["sample_max_rows"] = 400
    cfg["explain"]["permutation"]["n_repeats"] = 2
    cfg["explain"]["shap"]["max_samples"] = 80
    cfg["explain"]["shap"]["background_samples"] = 16
    cfg["evaluate"]["benchmark"]["warmup_iters"] = 2
    cfg["evaluate"]["benchmark"]["measure_iters"] = 5
    if time_limit is not None:
        # Gia lap Kaggle cat session bang DUNG co che thuc te (time_guard), chi
        # thu nho ca hai nguong. should_stop() khi remaining <= guard, tuc la
        # elapsed >= time_limit - guard. Voi 4 - 1 = 3 giay: du de dung cache va
        # chay vai step roi moi bi cat -> ngat GIUA epoch.
        #
        # Ban truoc dat time_limit = 1200 + 8 va giu guard = 1200 mac dinh, nen
        # can elapsed >= 8s; ca run chi mat 4s nen cu ngat khong bao gio xay ra.
        cfg["session"]["time_limit_seconds"] = time_limit
        cfg["session"]["exit_guard_seconds"] = 1
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return dst


def sh(args: list[str], *, tag: str) -> int:
    print(f"\n{'=' * 74}\n>>> {tag}\n{'=' * 74}")
    p = subprocess.run(args, cwd=REPO)
    print(f"<<< {tag}: exit {p.returncode}")
    return p.returncode


def read_state(store_root: Path) -> tuple[str, dict, list]:
    rid = json.loads((store_root / "current_run_id.json").read_text())["run_id"]
    sp = store_root / rid / "checkpoints" / "training_state.json"
    hp = store_root / rid / "metrics" / "history.json"
    state = json.loads(sp.read_text()) if sp.exists() else {}
    hist = json.loads(hp.read_text()) if hp.exists() else []
    return rid, state, hist


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dien tap ngat va resume")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--keep", action="store_true", help="Giu lai thu muc lam viec")
    args = ap.parse_args(argv)

    work = args.work_dir.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    data, cache, store = work / "data", work / "cache", work / "store"

    print(f"Thu muc lam viec: {work}")
    make_dataset(data)
    print(f"Da tao {2 * N_ROWS} hang x {N_FEATURES} cot, {len(CLASSES)} lop")

    failures: list[str] = []

    # --- Buoc 1: bi ngat giua epoch
    cfg_cut = write_config(work / "cfg_cut.yaml", data, cache, time_limit=4)
    if sh([sys.executable, "-m", "src.train", "--config", str(cfg_cut),
           "--local-store", str(store)], tag="Buoc 1: train, bi ngat giua epoch"):
        failures.append("buoc 1 thoat khac 0")

    rid, st1, h1 = read_state(store)
    print(f"\n  run_id            = {rid}")
    print(f"  status            = {st1.get('status')} / {st1.get('exit_reason')}")
    print(f"  current_epoch     = {st1.get('current_epoch')}")
    print(f"  steps_in_epoch    = {st1.get('steps_done_in_epoch')}")
    print(f"  epoch trong hist  = {len(h1)}")

    steps_cut = st1.get("steps_done_in_epoch") or 0
    session1 = st1.get("session_id")
    # Epoch DANG chay khi bi cat (1-based). Khong gia dinh la epoch 1: tren may
    # nhanh, epoch 1 co the xong truoc khi guard kich hoat.
    cut_epoch = (st1.get("current_epoch") or 0) + 1
    print(f"  -> bi cat giua epoch {cut_epoch} sau {steps_cut} step")

    if st1.get("status") != "interrupted":
        failures.append(
            f"mong doi status=interrupted, nhan {st1.get('status')!r}. Neu la "
            "'completed' thi ca run da xong truoc khi time_guard kich hoat - "
            "giam session.time_limit_seconds trong write_config()")
    if st1.get("exit_reason") != "time_guard":
        failures.append(f"mong doi exit_reason=time_guard, nhan {st1.get('exit_reason')!r}")
    if steps_cut <= 0:
        failures.append("khong ghi lai steps_done_in_epoch -> khong resume giua epoch duoc")
    if len(h1) != (st1.get("current_epoch") or 0):
        failures.append(f"history co {len(h1)} ban ghi nhung current_epoch="
                        f"{st1.get('current_epoch')} - phai khop nhau")

    # --- Buoc 2+3: resume va chay den het
    cfg = write_config(work / "cfg.yaml", data, cache, time_limit=None)
    if sh([sys.executable, "-m", "src.train", "--config", str(cfg),
           "--local-store", str(store)], tag="Buoc 2-3: resume va chay den het"):
        failures.append("buoc 2-3 thoat khac 0")

    rid2, st2, h2 = read_state(store)
    print(f"\n  run_id            = {rid2}  (phai giong buoc 1)")
    print(f"  session_id        = {st2.get('session_id')}  (phai khac buoc 1)")
    print(f"  current_epoch     = {st2.get('current_epoch')} / {st2.get('total_epochs')}")
    print(f"  status            = {st2.get('status')}")
    print(f"  restart_count     = {st2.get('restart_count')}")
    print(f"  epoch trong hist  = {[r['epoch'] for r in h2]}")

    if rid2 != rid:
        failures.append(f"run_id doi tu {rid} thanh {rid2} - vi pham muc 4.5")
    if st2.get("session_id") == session1:
        failures.append("session_id khong doi giua hai lan chay")
    if [r["epoch"] for r in h2] != [1, 2]:
        failures.append(f"history phai la [1, 2], nhan {[r['epoch'] for r in h2]}")

    by_epoch = {r["epoch"]: r for r in h2}
    cut_rec = by_epoch.get(cut_epoch)
    if cut_rec is None:
        failures.append(f"khong tim thay ban ghi cho epoch {cut_epoch} da bi cat")
    else:
        if not cut_rec.get("train_metrics_partial"):
            failures.append(f"epoch {cut_epoch} bi ngat giua chung nhung khong "
                            "duoc danh dau train_metrics_partial")
        if cut_rec.get("resumed_after_batches") != steps_cut:
            failures.append(
                f"epoch {cut_epoch}: resumed_after_batches="
                f"{cut_rec.get('resumed_after_batches')} khong khop "
                f"steps_done_in_epoch={steps_cut} luc bi cat")
    for r in h2:
        if r["epoch"] != cut_epoch and r.get("train_metrics_partial"):
            failures.append(f"epoch {r['epoch']} khong bi ngat nen khong duoc "
                            "danh dau partial")
    if st2.get("status") != "completed":
        failures.append(f"mong doi status=completed, nhan {st2.get('status')}")

    # --- Buoc 4: danh gia cuoi + bao cao
    if sh([sys.executable, "-m", "src.evaluate", "--config", str(cfg),
           "--local-store", str(store)], tag="Buoc 4a: danh gia cuoi"):
        failures.append("danh gia cuoi thoat khac 0")
    if sh([sys.executable, "make_report.py", "--run-dir", str(store / rid),
           "--out", str(work / "report"), "--upload"], tag="Buoc 4b: sinh bao cao"):
        failures.append("make_report thoat khac 0")

    figs = sorted((work / "report").glob("*.png"))
    print(f"\n  hinh PNG sinh ra  = {len(figs)}")
    if len(figs) < 14:
        failures.append(f"chi sinh {len(figs)}/14 hinh")

    # --- Buoc 5: kiem tra tieu chi nghiem thu
    rc = sh([sys.executable, "scripts/verify_acceptance.py",
             "--run-dir", str(store / rid), "--with-data", "--config", str(cfg)],
            tag="Buoc 5: kiem tra 12 tieu chi nghiem thu")

    print(f"\n{'=' * 74}")
    if failures:
        print(f"DIEN TAP THAT BAI - {len(failures)} van de:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("DIEN TAP DAT: chuoi ngat -> resume -> hoan thanh -> danh gia -> bao cao")
        print("hoat dong dung. Tieu chi 9 se bao FAIL vi dien tap dung epochs=2,")
        print("batch_size=256; run that lay 100/4096 tu configs/mddcc.yaml.")
    print(f"{'=' * 74}")

    if not args.keep:
        print(f"\nXoa {work} (dung --keep de giu lai)")
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
