#!/usr/bin/env python3
"""Dung thu muc kernel co credential TAM THOI de push - Kaggle khong luu secret.

Van de: notebook chay tren Kaggle can credential AWS de ghi checkpoint len S3,
nhung GitHub Secrets khong tu co mat trong runtime cua Kaggle. Neu khong muon luu
secret tren Kaggle thi credential phai di theo notebook luc push.

Cach lam:
  1. Dung khoa DAI HAN (chi nam tren GitHub Secrets) goi sts:GetSessionToken.
  2. Ma hoa base64 bo credential TAM THOI vao notebook.
  3. Push tu thu muc TAM, KHONG bao gio ghi vao kernel/ da commit.

Danh gia bao mat trung thuc:
  * Khoa dai han khong bao gio roi khoi GitHub. Day la loi ich chinh.
  * Token tam thoi CO nam trong ma nguon notebook ma Kaggle luu lai (kernel
    private). Dieu nay khong tranh duoc: runtime Kaggle phai co mot credential
    nao do. Bu lai, token tu het han sau vai gio.
  * Kaggle giu lai cac phien ban kernel cu, moi phien ban chua token cua lan do.
    Sau khi het han thi vo hai, nhung nen dung IAM user gioi han trong
    arn:aws:s3:::$S3_BUCKET/$S3_PREFIX/* de thiet hai toi da la co gioi han.
  * Neu can chat hon nua: dung AssumeRole voi inline session policy.

    python scripts/prepare_kernel_push.py --out-dir /tmp/kernel_push
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# 16 gio: du cho session 11h20m cong thoi gian xep hang cua Kaggle.
DEFAULT_DURATION = 57600
PASSTHROUGH = ("AWS_DEFAULT_REGION", "S3_BUCKET", "S3_PREFIX")


def get_temporary_credentials(duration: int) -> dict:
    """Goi sts:GetSessionToken bang khoa dai han trong bien moi truong."""
    import boto3

    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(name, "").strip():
            raise SystemExit(f"Thieu bien moi truong {name}")

    sts = boto3.client("sts", region_name=os.environ.get("AWS_DEFAULT_REGION"))
    ident = sts.get_caller_identity()
    if ":root" in ident.get("Arn", ""):
        raise SystemExit(
            "Dang dung khoa cua ROOT account: sts:GetSessionToken chi cho toi da "
            "3600s va cap quyen toan bo tai khoan. Hay tao IAM user rieng, gioi han "
            "trong arn:aws:s3:::$S3_BUCKET/$S3_PREFIX/*.")

    resp = sts.get_session_token(DurationSeconds=duration)
    c = resp["Credentials"]
    print(f"STS GetSessionToken OK cho {ident['Arn']}")
    print(f"  het han: {c['Expiration'].isoformat()} (~{duration / 3600:.1f} gio)")

    payload = {
        "AWS_ACCESS_KEY_ID": c["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": c["SecretAccessKey"],
        "AWS_SESSION_TOKEN": c["SessionToken"],
        "expiration": c["Expiration"].isoformat(),
    }
    for name in PASSTHROUGH:
        value = os.environ.get(name, "").strip()
        if not value:
            raise SystemExit(f"Thieu bien moi truong {name}")
        payload[name] = value
    return payload


def encode(payload: dict) -> str:
    return base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")


def build_push_dir(out_dir: Path, credentials_b64: str) -> Path:
    """Ghi notebook (da tiem credential) + kernel-metadata.json vao out_dir."""
    import build_notebook

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    nb = build_notebook.build(credentials_b64)
    (out_dir / "kaggle_notebook.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(REPO / "kernel" / "kernel-metadata.json",
                 out_dir / "kernel-metadata.json")

    # Xac minh credential that su da vao notebook
    body = (out_dir / "kaggle_notebook.ipynb").read_text(encoding="utf-8")
    if credentials_b64 and "__MDDCC_CREDENTIALS_B64__" in body:
        raise SystemExit("Tiem credential that bai: placeholder van con trong notebook")
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Dung thu muc kernel co credential tam thoi de push")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--duration", type=int,
                    default=int(os.environ.get("STS_DURATION_SECONDS",
                                               DEFAULT_DURATION)))
    ap.add_argument("--no-credentials", action="store_true",
                    help="Dung thu muc push ma KHONG tiem credential (de kiem tra)")
    args = ap.parse_args(argv)

    b64 = "" if args.no_credentials else encode(
        get_temporary_credentials(args.duration))
    out = build_push_dir(args.out_dir, b64)

    print(f"\nThu muc push: {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size:,} byte)")
    if b64:
        print("\nCHU Y: thu muc nay chua credential tam thoi - dung commit,")
        print("dung luu lai sau khi push xong.")

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            fh.write(f"push_dir={out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
