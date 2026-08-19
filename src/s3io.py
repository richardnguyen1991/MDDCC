"""Doc/ghi artifact len S3 an toan truoc viec Kaggle cat session - muc 4.2, 7.H.

Nguyen tac: KHONG bao gio ghi truc tiep len key chinh. Ghi len key tam, kiem tra
size + sha256, roi moi copy sang key chinh va xoa key tam. Neu session bi cat
giua chung thi key chinh van la ban cu nguyen ven, khong bao gio hong.

LocalStore ton tai de chay thu va chay test ma khong can AWS. No KHONG thay the
S3 trong run that: muc 1 quy dinh S3 la nguon su that duy nhat song sot qua cac
session bi cancel.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

LOG = logging.getLogger(__name__)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ObjectStore(ABC):
    """Giao dien toi thieu ma checkpoint/train can."""

    @abstractmethod
    def put_file(self, local: Path, key: str) -> dict: ...
    @abstractmethod
    def get_file(self, key: str, local: Path) -> Path: ...
    @abstractmethod
    def exists(self, key: str) -> bool: ...
    @abstractmethod
    def delete(self, key: str) -> None: ...
    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]: ...
    @abstractmethod
    def put_bytes(self, data: bytes, key: str) -> dict: ...
    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    # ------------------------------------------------------------- tien ich
    def put_json(self, obj, key: str) -> dict:
        return self.put_bytes(
            json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            key)

    def get_json(self, key: str):
        return json.loads(self.get_bytes(key).decode("utf-8"))

    def get_json_or_none(self, key: str):
        return self.get_json(key) if self.exists(key) else None


class LocalStore(ObjectStore):
    """Luu xuong thu muc cuc bo - dung cho test va chay thu ngoai Kaggle."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        return self.root / key

    def put_file(self, local: Path, key: str) -> dict:
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, dst)
        return {"key": key, "size": dst.stat().st_size, "sha256": sha256_file(dst)}

    def get_file(self, key: str, local: Path) -> Path:
        src = self._p(key)
        if not src.exists():
            raise FileNotFoundError(f"khong co key {key}")
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local)
        return Path(local)

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def delete(self, key: str) -> None:
        self._p(key).unlink(missing_ok=True)

    def list_keys(self, prefix: str) -> list[str]:
        base = self._p(prefix)
        root = base if base.is_dir() else base.parent
        if not root.exists():
            return []
        out = [str(p.relative_to(self.root)).replace(os.sep, "/")
               for p in root.rglob("*") if p.is_file()]
        return sorted(k for k in out if k.startswith(prefix))

    def put_bytes(self, data: bytes, key: str) -> dict:
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return {"key": key, "size": len(data), "sha256": sha256_bytes(data)}

    def get_bytes(self, key: str) -> bytes:
        src = self._p(key)
        if not src.exists():
            raise FileNotFoundError(f"khong co key {key}")
        return src.read_bytes()

    def copy(self, src_key: str, dst_key: str) -> None:
        dst = self._p(dst_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._p(src_key), dst)


class S3Store(ObjectStore):
    """S3 that. Credential doc tu bien moi truong, khong bao gio tu code."""

    def __init__(self, bucket: str, prefix: str = "", client=None):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client or boto3.client("s3")

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_file(self, local: Path, key: str) -> dict:
        local = Path(local)
        self.client.upload_file(str(local), self.bucket, self._k(key))
        return {"key": key, "size": local.stat().st_size, "sha256": sha256_file(local)}

    def get_file(self, key: str, local: Path) -> Path:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._k(key), str(local))
        return Path(local)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._k(key))

    def list_keys(self, prefix: str) -> list[str]:
        out, token = [], None
        full = self._k(prefix)
        while True:
            kw = {"Bucket": self.bucket, "Prefix": full}
            if token:
                kw["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kw)
            for item in resp.get("Contents", []):
                k = item["Key"]
                out.append(k[len(self.prefix) + 1:] if self.prefix else k)
            if not resp.get("IsTruncated"):
                return sorted(out)
            token = resp["NextContinuationToken"]

    def put_bytes(self, data: bytes, key: str) -> dict:
        self.client.put_object(Bucket=self.bucket, Key=self._k(key), Body=data)
        return {"key": key, "size": len(data), "sha256": sha256_bytes(data)}

    def get_bytes(self, key: str) -> bytes:
        resp = self.client.get_object(Bucket=self.bucket, Key=self._k(key))
        return resp["Body"].read()

    def head_size(self, key: str) -> int:
        return int(self.client.head_object(
            Bucket=self.bucket, Key=self._k(key))["ContentLength"])

    def copy(self, src_key: str, dst_key: str) -> None:
        self.client.copy_object(
            Bucket=self.bucket,
            CopySource={"Bucket": self.bucket, "Key": self._k(src_key)},
            Key=self._k(dst_key))


class SafeWriter:
    """Ghi an toan: key tam -> kiem tra size + sha256 -> copy -> xoa key tam.

    Day la co che chong hong file khi Kaggle cat session giua luc upload
    (muc 4.2). Key chinh chi doi khi ban tam da duoc xac minh nguyen ven.
    """

    def __init__(self, store: ObjectStore, tmp_prefix: str = "_tmp"):
        self.store = store
        self.tmp_prefix = tmp_prefix.strip("/")

    def _tmp_key(self, key: str) -> str:
        return f"{self.tmp_prefix}/{key}"

    def put_file(self, local: Path, key: str) -> dict:
        local = Path(local)
        expect_size = local.stat().st_size
        expect_hash = sha256_file(local)
        tmp = self._tmp_key(key)

        self.store.put_file(local, tmp)
        self._verify(tmp, expect_size, expect_hash, key)
        self._promote(tmp, key)
        LOG.info("  upload %s (%.2f MB, sha %s)", key, expect_size / 1e6,
                 expect_hash[:12])
        return {"key": key, "size": expect_size, "sha256": expect_hash}

    def put_bytes(self, data: bytes, key: str) -> dict:
        expect_size, expect_hash = len(data), sha256_bytes(data)
        tmp = self._tmp_key(key)

        self.store.put_bytes(data, tmp)
        self._verify(tmp, expect_size, expect_hash, key)
        self._promote(tmp, key)
        return {"key": key, "size": expect_size, "sha256": expect_hash}

    def put_json(self, obj, key: str) -> dict:
        return self.put_bytes(
            json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            key)

    # ---------------------------------------------------------------- noi bo
    def _verify(self, tmp_key: str, size: int, digest: str, key: str) -> None:
        got = self.store.get_bytes(tmp_key)
        if len(got) != size or sha256_bytes(got) != digest:
            self.store.delete(tmp_key)
            raise RuntimeError(
                f"Upload hong khi ghi {key}: size {len(got)} vs {size}, "
                f"sha256 {sha256_bytes(got)[:12]} vs {digest[:12]}. "
                "Key chinh KHONG bi dong toi."
            )

    def _promote(self, tmp_key: str, key: str) -> None:
        copy = getattr(self.store, "copy", None)
        if copy is not None:
            copy(tmp_key, key)
        else:
            self.store.put_bytes(self.store.get_bytes(tmp_key), key)
        self.store.delete(tmp_key)


def store_from_env(cfg: dict, *, local_root: Path | None = None) -> ObjectStore:
    """Dung S3 khi co du bien moi truong, nguoc lai dung LocalStore.

    Tren Kaggle, credential den tu kaggle_secrets -> bien moi truong (muc 8.A).
    """
    s = cfg.get("s3", {})
    bucket = os.environ.get(s.get("bucket_env", "S3_BUCKET"), "").strip()
    prefix = os.environ.get(s.get("prefix_env", "S3_PREFIX"), "").strip()

    if bucket:
        LOG.info("Dung S3: s3://%s/%s", bucket, prefix)
        return S3Store(bucket, prefix)

    root = Path(local_root or os.environ.get("MDDCC_LOCAL_STORE", "./_localstore"))
    LOG.warning("KHONG co S3_BUCKET - dung LocalStore tai %s. "
                "Chi hop le khi chay thu; run that BAT BUOC dung S3.", root)
    return LocalStore(root)
