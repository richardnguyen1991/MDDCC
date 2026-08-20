#!/usr/bin/env python3
"""Quyet dinh co khoi dong lai kernel Kaggle hay khong - muc 8.B.

Tach logic ra Python thay vi nhoi vao bash de test duoc. GitHub Actions chi goi
script nay; workflow KHONG train tren runner cua GitHub.

Quy tac (muc 8.B):
  * current_epoch >= total hoac status == completed  -> DONE, exit 0
  * kernel dang running/queued                       -> khong dung vao, exit 0
  * kernel complete/error ma current_epoch < total   -> push de mo session moi
  * 3 lan push lien tiep ma current_epoch KHONG tang -> dung, ghi status=failed,
    mo GitHub Issue (chong vong lap vo han)
  * vuot max_restarts                                -> dung
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

DEFAULT_MAX_RESTARTS = 60          # 39 session du kien + bien an toan
STALL_LIMIT = 3                    # 3 lan push khong tien -> dung

RUNNING_STATES = {"running", "queued"}
FINISHED_STATES = {"complete", "cancelacknowledged", "error", "cancelrequested"}

ACTION_DONE = "done"
ACTION_WAIT = "wait"
ACTION_PUSH = "push"
ACTION_ABORT = "abort"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Decision:
    action: str
    reason: str
    current_epoch: int = 0
    total_epochs: int = 100
    kernel_status: str = ""
    restart_count: int = 0

    def to_dict(self) -> dict:
        return {**asdict(self), "decided_at_utc": utc_now()}


def decide(*, state: dict | None, kernel_status: str, restart_history: list[dict],
           max_restarts: int = DEFAULT_MAX_RESTARTS,
           stall_limit: int = STALL_LIMIT) -> Decision:
    """Ham THUAN: khong doc/ghi gi, chi quyet dinh. Day la phan duoc test."""
    status_norm = (kernel_status or "").strip().lower()

    # Chua co run nao -> phai push de tao run dau tien
    if state is None:
        if status_norm in RUNNING_STATES:
            return Decision(ACTION_WAIT, "chua co training_state nhung kernel dang chay",
                            kernel_status=status_norm)
        return Decision(ACTION_PUSH, "chua co training_state -> khoi dong run dau tien",
                        kernel_status=status_norm)

    epoch = int(state.get("current_epoch", 0))
    total = int(state.get("total_epochs", 100))
    restarts = len(restart_history)

    # 1. Da xong -> khong push nua (muc 8.B.3)
    if epoch >= total or state.get("status") == "completed":
        return Decision(ACTION_DONE, f"da xong {epoch}/{total} epoch",
                        epoch, total, status_norm, restarts)

    # 2. Dang chay -> khong dung vao (muc 8.B.4)
    if status_norm in RUNNING_STATES:
        return Decision(ACTION_WAIT, f"kernel dang {status_norm}, epoch {epoch}/{total}",
                        epoch, total, status_norm, restarts)

    # 3. Chong vong lap vo han: 3 lan push lien tiep ma epoch khong tang
    recent = restart_history[-stall_limit:]
    if len(recent) >= stall_limit and all(
            int(r.get("current_epoch_before", -1)) == epoch for r in recent):
        return Decision(
            ACTION_ABORT,
            f"{stall_limit} lan push lien tiep ma current_epoch dung yen o {epoch} "
            "-> dung tu dong khoi dong lai, can nguoi kiem tra log Kaggle",
            epoch, total, status_norm, restarts)

    # 4. Vuot tran so lan restart
    if restarts >= max_restarts:
        return Decision(ACTION_ABORT,
                        f"da restart {restarts} lan >= max_restarts={max_restarts}",
                        epoch, total, status_norm, restarts)

    # 5. Kernel da ket thuc ma chua du epoch -> push session moi
    if status_norm in FINISHED_STATES or status_norm == "":
        return Decision(ACTION_PUSH,
                        f"kernel {status_norm or 'khong ro'}, epoch {epoch}/{total} "
                        "-> khoi dong session moi", epoch, total, status_norm, restarts)

    return Decision(ACTION_WAIT, f"trang thai kernel la khong ro: {status_norm!r}",
                    epoch, total, status_norm, restarts)


def parse_kernel_status(raw: str) -> str:
    """Doc dau ra cua `kaggle kernels status`.

    Dinh dang co the la JSON hoac cau van kieu
    'has status "complete"'. Xu ly ca hai.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return str(obj.get("status", "")).strip().lower()
    except json.JSONDecodeError:
        pass
    import re
    m = re.search(r'status\s+"?([A-Za-z]+)"?', raw)
    return m.group(1).strip().lower() if m else raw.lower()


# ------------------------------------------------------------------- vao ra
def open_store(local_root: str | None = None):
    """S3 khi co S3_BUCKET, nguoc lai LocalStore de dry-run duoc ma khong can AWS.

    Dung chung store_from_env voi phan con lai cua pipeline nen hanh vi giong het.
    """
    from pathlib import Path
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.s3io import store_from_env

    cfg = {"s3": {"bucket_env": "S3_BUCKET", "prefix_env": "S3_PREFIX"}}
    if not os.environ.get("S3_BUCKET", "").strip() and not local_root:
        raise SystemExit(
            "Thieu bien moi truong S3_BUCKET. Neu chi muon thu logic quyet dinh, "
            "chay voi --local-store <thu-muc> --dry-run.")
    return store_from_env(cfg, local_root=local_root)


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dieu phoi kernel Kaggle cho MDDCC")
    ap.add_argument("--kernel", default=os.environ.get("KAGGLE_KERNEL", ""))
    ap.add_argument("--kernel-dir", default="kernel")
    ap.add_argument("--variant", default="full",
                    help="full | ten overlay trong configs/variants/")
    ap.add_argument("--max-restarts", type=int,
                    default=int(os.environ.get("MAX_RESTARTS", DEFAULT_MAX_RESTARTS)))
    ap.add_argument("--dry-run", action="store_true",
                    help="Quyet dinh nhung khong push, khong ghi S3")
    ap.add_argument("--local-store", default=None,
                    help="Dung thu muc cuc bo thay S3 - chi de thu logic quyet dinh")
    ap.add_argument("--kernel-status", default=None,
                    help="Ep trang thai kernel thay vi goi Kaggle API - chi de thu")
    args = ap.parse_args(argv)

    if not args.kernel:
        raise SystemExit("Thieu KAGGLE_KERNEL (dang <username>/<slug>)")

    store = open_store(args.local_store)

    # Moi bien the co khoa current_run_id RIENG. Neu doc nham khoa thi
    # orchestrator se tuong bien the nay dang o epoch cua bien the kia.
    from pathlib import Path as _P
    import sys as _s
    _s.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from src.config import load_config, run_id_key

    cfg = load_config(_P(__file__).resolve().parents[1] / "configs" / "mddcc.yaml",
                      None if args.variant == "full" else args.variant)
    key = run_id_key(cfg)

    run_id = None
    d = store.get_json_or_none(key)
    if d:
        run_id = d.get("run_id")

    state = None
    history: list[dict] = []
    if run_id:
        state = store.get_json_or_none(f"{run_id}/checkpoints/training_state.json")
        history = store.get_json_or_none(f"{run_id}/orchestration/restart_history.json") or []

    if args.kernel_status is not None:
        rc, kernel_status = 0, parse_kernel_status(args.kernel_status)
    else:
        rc, out = run(["kaggle", "kernels", "status", args.kernel])
        kernel_status = parse_kernel_status(out)
    print(f"bien the      = {args.variant}  (khoa {key})")
    print(f"run_id        = {run_id}")
    print(f"kernel status = {kernel_status!r} (exit {rc})")
    if state:
        print(f"epoch         = {state.get('current_epoch')}/{state.get('total_epochs')}"
              f"  status={state.get('status')}  exit_reason={state.get('exit_reason')}")
    print(f"restarts      = {len(history)} / {args.max_restarts}")

    decision = decide(state=state, kernel_status=kernel_status,
                      restart_history=history, max_restarts=args.max_restarts)
    print(f"\n==> {decision.action.upper()}: {decision.reason}")

    if args.dry_run:
        return 0

    if decision.action == ACTION_DONE:
        return 0
    if decision.action == ACTION_WAIT:
        return 0

    if decision.action == ACTION_ABORT:
        if run_id and state:
            store.put_json({**state, "status": "failed",
                            "exit_reason": decision.reason,
                            "updated_at_utc": utc_now()},
                           f"{run_id}/checkpoints/training_state.json")
        # Bao hieu cho workflow mo GitHub Issue
        gh_out = os.environ.get("GITHUB_OUTPUT")
        if gh_out:
            with open(gh_out, "a", encoding="utf-8") as fh:
                fh.write("abort=true\n")
                fh.write(f"abort_reason={decision.reason}\n")
        return 0

    # ---- ACTION_PUSH
    rc, out = run(["kaggle", "kernels", "push", "-p", args.kernel_dir])
    print(out)
    entry = {
        "pushed_at_utc": utc_now(),
        "current_epoch_before": decision.current_epoch,
        "total_epochs": decision.total_epochs,
        "kernel_status_before": kernel_status,
        "kernel": args.kernel,
        "push_exit_code": rc,
        "run_url": f"https://www.kaggle.com/code/{args.kernel}",
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
    }
    if run_id:
        history.append(entry)
        store.put_json(history, f"{run_id}/orchestration/restart_history.json")
        store.put_json(entry,
                       f"{run_id}/orchestration/restart_"
                       f"{entry['pushed_at_utc'].replace(':', '-')}.json")
    if rc != 0:
        print("CANH BAO: kaggle kernels push tra ve loi", rc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
