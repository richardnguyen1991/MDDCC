#!/usr/bin/env python3
"""Sinh kernel/kaggle_notebook.ipynb - muc 8.D.

Viet notebook bang Python roi xuat ra .ipynb, thay vi sua JSON bang tay: JSON cua
notebook rat de hong va khong review duoc trong diff.

    python scripts/build_notebook.py

Notebook phai:
  * fail-fast ngay dong dau neu khong thay dataset (muc 8.C)
  * doc secret tu kaggle_secrets -> bien moi truong, KHONG hardcode
  * in run_id / session_id / current_epoch o dau VA cuoi log
  * idempotent: lan chay thu N chi tiep tuc, khong tao run_id moi
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/richardnguyen1991/MDDCC.git"
WORK = "/kaggle/working/mddcc"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(text: str) -> None:
    CELLS.append(("code", text))


# ============================================================== noi dung
md(f"""# MDDCC - Huan luyen tren Kaggle CPU

Tai hien Wang K. et al., *Scientific Reports* 14:16421 (2024),
DOI 10.1038/s41598-024-66907-z.

Notebook nay **idempotent**: lan chay thu N chi tiep tuc tu checkpoint tren S3,
khong tao `run_id` moi (tru khi xoa `current_run_id.json`).

Repo: {REPO_URL}

**Kaggle KHONG luu secret nao.** Toan bo credential nam tren GitHub Secrets.
Moi lan `kaggle kernels push`, GitHub Actions goi `sts:GetSessionToken` bang khoa
dai han de lay mot bo credential TAM THOI roi tiem vao notebook nay. Token tu het
han sau vai gio; khoa dai han khong bao gio roi khoi GitHub.

=> Dung khoi dong notebook nay bang tay tren UI khi chua tiem credential - no se
fail-fast o muc 4b. Hay chay workflow **run-kaggle** tren GitHub.
""")

md("## 1. Fail-fast: dataset phai duoc mount")

code('''# Muc 8.C: neu kernel-metadata.json thieu "dataset_sources" thi session do
# GitHub Actions khoi dong se KHONG co dataset, notebook chet ngay o buoc doc du
# lieu va vong lap restart quay vo ich. Kiem tra ngay dong dau.
import os
import sys
from pathlib import Path

ROOT = Path("/kaggle/input")
print("Co trong /kaggle/input:", sorted(p.name for p in ROOT.iterdir())
      if ROOT.exists() else "KHONG TON TAI")

parquets = sorted(ROOT.glob("**/*.parquet")) if ROOT.exists() else []
if not parquets:
    raise SystemExit(
        "FAIL-FAST: khong thay file .parquet nao duoi /kaggle/input.\\n"
        "  - Neu chay tay: Add Input -> Datasets -> cicddos2019-parquet\\n"
        "  - Neu do GitHub Actions push: kiem tra kernel/kernel-metadata.json\\n"
        "    co khai bao \\"dataset_sources\\" khong (muc 8.C)."
    )

# Thu muc goc chung cua cac file parquet
DATA_DIR = parquets[0].parent
while not all(str(f).startswith(str(DATA_DIR) + os.sep) for f in parquets):
    DATA_DIR = DATA_DIR.parent
print(f"\\nDATA_DIR = {DATA_DIR}")
print(f"So file parquet = {len(parquets)}")''')

md("## 2. Lay ma nguon")

code(f'''# Roi khoi thu muc truoc khi xoa - neu dang dung trong do thi shell mat CWD
os.chdir("/kaggle/working")
WORK = "{WORK}"

if Path(WORK).exists():
    import shutil
    shutil.rmtree(WORK, ignore_errors=True)

rc = os.system(f"git clone -q --depth 1 {REPO_URL} {{WORK}}")
if rc != 0 or not Path(WORK, "src", "train.py").exists():
    raise SystemExit(f"FAIL-FAST: git clone that bai (rc={{rc}}). Kiem tra "
                     "Settings -> Internet da bat chua.")
os.chdir(WORK)
sys.path.insert(0, WORK)
print("Commit:", os.popen("git rev-parse --short HEAD").read().strip())''')

md("## 3. Cai phu thuoc")

code('''# Kaggle da co torch/numpy/sklearn/pyarrow. Chi cai nhung goi con thieu,
# phien ban ghim trong requirements.txt (muc 8.D).
os.system("pip install -q -r requirements.txt")

import numpy, sklearn, pyarrow, pywt, torch
print("torch  ", torch.__version__)
print("numpy  ", numpy.__version__)
print("sklearn", sklearn.__version__)
print("pyarrow", pyarrow.__version__)
print("pywt   ", pywt.__version__)
try:
    import shap
    print("shap   ", shap.__version__)
except ImportError:
    print("shap    KHONG CO -> se bo qua hinh C13")''')

md("""## 4. Nap credential do GitHub Actions tiem vao

Kaggle **khong luu secret nao**. Moi lan push, GitHub Actions goi
`sts:GetSessionToken` bang khoa dai han (chi nam tren GitHub) de lay mot bo
credential TAM THOI, roi tiem vao chinh o duoi day. Token tu het han, khoa dai
han khong bao gio roi khoi GitHub.
""")

code('''# Chuoi nay do scripts/prepare_kernel_push.py thay the luc push.
# Ban trong repo LUON rong - co test khang dinh dieu do.
CREDENTIALS_B64 = "__MDDCC_CREDENTIALS_B64__"
VARIANT = "__MDDCC_VARIANT__"

INJECTED = {}
if CREDENTIALS_B64 and not CREDENTIALS_B64.startswith("__MDDCC"):
    import base64
    import datetime as _dt
    import json

    INJECTED = json.loads(base64.b64decode(CREDENTIALS_B64).decode("utf-8"))
    expires = INJECTED.pop("expiration", None)
    for k, v in INJECTED.items():
        os.environ[k] = v
    print(f"Da nap {len(INJECTED)} bien tu credential tam thoi do GitHub Actions tiem")
    if expires:
        left = (_dt.datetime.fromisoformat(expires)
                - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
        print(f"  het han luc {expires} (con {left / 3600:.1f} gio)")
        if left < 3600:
            print("  CANH BAO: token con duoi 1 gio, session nay co the "
                  "khong upload duoc checkpoint cuoi")
        if left <= 0:
            raise SystemExit(
                "FAIL-FAST: credential da het han. Chay lai workflow run-kaggle "
                "de push phien ban moi voi token moi.")
else:
    print("Khong co credential duoc tiem -> se doc tu bien moi truong / "
          "kaggle_secrets (che do chay tay)")''')

md("""## 4b. Doi chieu credential

Kiem tra du 5 bien can thiet. Thu tu uu tien: credential do Actions tiem vao
(o tren) -> bien moi truong -> `kaggle_secrets` (neu ai do van muon luu tren
Kaggle). Khoa dai han TUYET DOI khong nam trong notebook hay git (muc 8.A).
""")

code('''REQUIRED = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION", "S3_BUCKET", "S3_PREFIX"]

try:
    from kaggle_secrets import UserSecretsClient
    client = UserSecretsClient()

    def read(name):
        try:
            return client.get_secret(name)
        except Exception:
            return os.environ.get(name, "")
except ImportError:
    print("Khong o Kaggle -> doc tu bien moi truong")

    def read(name):
        return os.environ.get(name, "")

missing = []
for name in REQUIRED:
    value = (read(name) or "").strip()
    if value:
        os.environ[name] = value
    else:
        missing.append(name)

if missing:
    raise SystemExit(
        "FAIL-FAST: thieu " + ", ".join(missing) + ".\\n"
        "  Cach dung chinh: chay workflow run-kaggle tren GitHub - no goi\\n"
        "  sts:GetSessionToken va tiem credential tam thoi vao notebook.\\n"
        "  Kaggle KHONG luu secret nao, dung tim trong Add-ons -> Secrets.\\n"
        "  Neu chay tay: python scripts/prepare_kernel_push.py --out-dir <dir>"
    )

# Chi in do dai, TUYET DOI khong in gia tri
for name in REQUIRED:
    v = os.environ[name]
    shown = v if name in ("AWS_DEFAULT_REGION", "S3_BUCKET", "S3_PREFIX") else f"<{len(v)} ky tu>"
    print(f"  {name:<24} {shown}")''')

md("## 5. Trang thai TRUOC khi chay")

code('''import json
from src.s3io import store_from_env
from src.checkpoint import RunRegistry
from src.config import load_config, run_id_key
import yaml

CFG_PATH = "configs/mddcc.yaml"
_VARIANT = VARIANT if VARIANT and not VARIANT.startswith("__MDDCC") else None
cfg = load_config(Path(CFG_PATH), _VARIANT)

# Duong dan mount that co the khac config (Kaggle dat ten theo slug) -> ghi de
cfg_dir = Path(cfg["data"]["kaggle_input_dir"])
if not cfg_dir.exists():
    print(f"CHU Y: {cfg_dir} khong ton tai, dung DATA_DIR do duoc: {DATA_DIR}")
    cfg["data"]["kaggle_input_dir"] = str(DATA_DIR)
    Path("configs/mddcc.runtime.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    CFG_PATH = "configs/mddcc.runtime.yaml"

store = store_from_env(cfg)
REGISTRY = RunRegistry(store, key=run_id_key(cfg))
run_id = REGISTRY.get()
print(f"run_id truoc khi chay = {run_id}")
if run_id:
    st = store.get_json_or_none(f"{run_id}/checkpoints/training_state.json") or {}
    print(f"  current_epoch = {st.get('current_epoch')} / {st.get('total_epochs')}")
    print(f"  status        = {st.get('status')}  exit_reason={st.get('exit_reason')}")
    print(f"  restart_count = {st.get('restart_count')}")
    if st.get("is_complete"):
        print("\\n=> DA XONG 100 EPOCH. Session nay se chuyen sang buoc danh gia cuoi.")
else:
    print("  chua co run nao -> se tao run_id moi")''')

md("""## 6. Huan luyen

Tu nap checkpoint + training state tu S3, tu luu dinh ky len S3, va tu thoat
truoc khi Kaggle cat session (`time_guard`). Thoat code 0 la binh thuong -
GitHub Actions se khoi dong session tiep theo.
""")

code('''VARIANT_ARG = f" --variant {VARIANT}" if VARIANT and not VARIANT.startswith("__MDDCC") else ""
print(f"Bien the: {VARIANT if VARIANT_ARG else 'full'}")
rc = os.system(f"python -m src.train --config {CFG_PATH}{VARIANT_ARG} 2>&1")
print(f"\\ntrain exit code = {rc}")
if rc != 0:
    raise SystemExit(f"src.train that bai (rc={rc}) - xem log phia tren")''')

md("## 7. Danh gia cuoi (chi khi da du 100 epoch)")

code('''# Muc 4.8 + 7.E5: buoc RIENG BIET. Neu buoc nay loi thi checkpoint van nguyen
# tren S3 va chay lai duoc bang make_report.py.
run_id = REGISTRY.get()
st = store.get_json_or_none(f"{run_id}/checkpoints/training_state.json") or {}

if st.get("is_complete"):
    print("Du 100 epoch -> chay danh gia cuoi + sinh bao cao")
    rc_eval = os.system(f"python -m src.evaluate --config {CFG_PATH}{VARIANT_ARG} 2>&1")
    print(f"evaluate exit code = {rc_eval}")
    if rc_eval == 0:
        bucket, prefix = os.environ["S3_BUCKET"], os.environ["S3_PREFIX"]
        rc_rep = os.system(
            f"python make_report.py --run-dir s3://{bucket}/{prefix}/{run_id} "
            f"--out /kaggle/working/report --upload 2>&1")
        print(f"make_report exit code = {rc_rep}")
else:
    print(f"Moi den epoch {st.get('current_epoch')}/{st.get('total_epochs')} "
          "-> chua danh gia. GitHub Actions se khoi dong session tiep theo.")''')

md("## 8. Trang thai SAU khi chay")

code('''run_id = REGISTRY.get()
st = store.get_json_or_none(f"{run_id}/checkpoints/training_state.json") or {}
hist = store.get_json_or_none(f"{run_id}/metrics/history.json") or []

print("=" * 62)
print(f"run_id        = {run_id}")
print(f"session_id    = {st.get('session_id')}")
print(f"current_epoch = {st.get('current_epoch')} / {st.get('total_epochs')}")
print(f"status        = {st.get('status')}")
print(f"exit_reason   = {st.get('exit_reason')}")
print(f"restart_count = {st.get('restart_count')}")
print(f"so epoch trong history = {len(hist)}")
if hist:
    last = hist[-1]
    print(f"epoch cuoi    = {last['epoch']}  "
          f"val_macro_f1={last.get('val_macro_f1')}  "
          f"{last.get('epoch_seconds')}s")
    sessions = len({r.get("session_id") for r in hist})
    done = st.get("current_epoch") or 1
    remain = (st.get("total_epochs", 100) - done)
    avg = sum(r.get("epoch_seconds", 0) for r in hist) / max(len(hist), 1)
    print(f"so session da dung = {sessions}")
    print(f"uoc tinh con lai   = {remain} epoch x {avg:.0f}s = "
          f"{remain * avg / 3600:.1f}h")
print("=" * 62)''')


# ============================================================== xuat file
def build(credentials_b64: str = "", variant: str = "") -> dict:
    cells = []
    for kind, text in CELLS:
        source = text.rstrip("\n").split("\n")
        source = [line + "\n" for line in source[:-1]] + [source[-1]]
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": source})
    for token, value in (("__MDDCC_CREDENTIALS_B64__", credentials_b64),
                         ("__MDDCC_VARIANT__", variant)):
        if value:
            for c in cells:
                c["source"] = [s.replace(token, value) for s in c["source"]]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    out = Path(__file__).resolve().parents[1] / "kernel" / "kaggle_notebook.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print(f"Da ghi {out}  ({len(CELLS)} cell, {n_code} cell code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
