"""Doc Parquet, chia tap chong ro ri, Min-Max, cache float32 - muc 3 cua prompt.

Rang buoc thiet ke, bat nguon tu phuong an C (giu tron 70.4M hang):
  * KHONG bao gio nap ca dataset vao RAM. Moi thu di qua row-group + memmap.
  * KHONG cache subband [N,4,S,S] (91.3 GB, vuot dia Kaggle). Chi cache feature
    da scale [N, F] float32 (22.5 GB); SWT tinh on-the-fly trong DataLoader,
    ton ~0.2 h/epoch so voi ~4.1 h compute.
  * /kaggle/input la READ-ONLY tuyet doi. Cache nam o data.cache_dir.

Thu tu bat buoc (muc 3.C.5): split -> fit scaler tren train -> transform -> SWT.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LOG = logging.getLogger(__name__)

# Cot khong bao gio duoc lam feature. So sanh sau khi strip() + lower().
PROVENANCE_COLUMNS = ("__capture_day", "__source_file_id", "__source_row_id")
NUMERIC_ARROW_KINDS = ("int", "uint", "float", "double", "decimal")


# ---------------------------------------------------------------- tien ich
def norm_name(name: str) -> str:
    return name.strip().lower()


def is_numeric(arrow_type: pa.DataType) -> bool:
    return any(k in str(arrow_type) for k in NUMERIC_ARROW_KINDS)


def sha256_of(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


# ------------------------------------------------------------------ schema
@dataclass
class FeatureSchema:
    """Nguon SU THAT DUY NHAT ve ten va thu tu cot - muc 3.B."""

    feature_columns: list[str]
    label_column: str
    dropped_columns: list[dict]
    label_detection: str

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def hash(self) -> str:
        return sha256_of({"features": self.feature_columns, "label": self.label_column})

    def to_dict(self) -> dict:
        return {
            "feature_columns": self.feature_columns,
            "n_features": self.n_features,
            "label_column": self.label_column,
            "label_detection": self.label_detection,
            "dropped_columns": self.dropped_columns,
            "feature_schema_hash": self.hash,
        }

    def assert_matches(self, other_hash: str) -> None:
        if other_hash != self.hash:
            raise RuntimeError(
                "feature_schema_hash lech - fail-fast theo muc 3.B/4.3.\n"
                f"  hien tai: {self.hash}\n  mong doi: {other_hash}\n"
                "Khong duoc am tham train lai tu dau."
            )

    def pruned(self, keep_mask: np.ndarray, extra_dropped: list[dict]) -> "FeatureSchema":
        """Tra ve schema moi sau khi loai cot hang so / trung lap / thieu qua nhieu."""
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.size != self.n_features:
            raise ValueError(f"mask {keep_mask.size} != n_features {self.n_features}")
        kept = [c for c, k in zip(self.feature_columns, keep_mask) if k]
        if not kept:
            raise RuntimeError("Loai het cot dac trung - kiem tra nguong trong config.")
        return FeatureSchema(kept, self.label_column,
                             self.dropped_columns + list(extra_dropped),
                             self.label_detection)


def discover_schema(files: Sequence[Path], cfg: dict) -> FeatureSchema:
    """Xac dinh cot nhan + cot dac trung. Fail-fast neu khong ro rang."""
    dcfg = cfg["data"]
    schema = pq.ParquetFile(files[0]).schema_arrow
    names = [f.name for f in schema]

    # --- nhan: uu tien cot co san (muc 3.A)
    label_col, how = None, "not_found"
    by_norm = {norm_name(n): n for n in names}
    for cand in dcfg["label"]["column_candidates"]:
        if norm_name(cand) in by_norm:
            label_col, how = by_norm[norm_name(cand)], f"column_candidate:{cand}"
            break
    if label_col is None:
        if dcfg["label"].get("fail_if_undetermined", True):
            raise RuntimeError(
                "Khong tim thay cot nhan trong "
                f"{dcfg['label']['column_candidates']}. Cot co san: {names}. "
                "Muc 3.A cam doan ngam."
            )
        raise RuntimeError("infer_from_path_if_missing chua duoc kich hoat cho dataset nay")

    # --- cot bi loai
    drop_ids = set()
    if dcfg.get("drop_identifier_columns", True):
        drop_ids = {norm_name(c) for c in dcfg["identifier_columns"]}
    drop_ids |= {norm_name(c) for c in PROVENANCE_COLUMNS}

    features, dropped = [], []
    for f in schema:
        if f.name == label_col:
            continue
        n = norm_name(f.name)
        if n in drop_ids:
            reason = ("provenance (khong duoc lam feature)"
                      if n in {norm_name(c) for c in PROVENANCE_COLUMNS}
                      else "dinh danh / ro ri (muc 3.B)")
            dropped.append({"name": f.name, "dtype": str(f.type), "reason": reason})
        elif not is_numeric(f.type):
            # Sau khi bo dinh danh, CIC-DDoS2019 khong con cot phi so.
            # Neu con, dung lai de nguoi dung quyet (muc 3.C.3), khong doan ngam.
            raise RuntimeError(
                f"Cot phi so con lai: {f.name!r} ({f.type}). Muc 3.C.3 yeu cau one-hot "
                "hoac loai bo - hay bo sung vao data.identifier_columns roi chay lai."
            )
        else:
            features.append(f.name)

    if not features:
        raise RuntimeError("Khong con cot dac trung nao sau khi loai - kiem tra config.")
    return FeatureSchema(features, label_col, dropped, how)


def assert_schema_consistent(files: Sequence[Path], schema: FeatureSchema) -> None:
    """Moi file phai co du cot dac trung + cot nhan (muc 3.A)."""
    need = set(schema.feature_columns) | {schema.label_column}
    for fp in files:
        have = {f.name for f in pq.ParquetFile(fp).schema_arrow}
        missing = need - have
        if missing:
            raise RuntimeError(f"{fp.name} thieu cot: {sorted(missing)}")


# ------------------------------------------------------------------- nhan
@dataclass
class LabelIndex:
    """label_mapping.json - nguon duy nhat cho ten va thu tu lop (muc 3.E)."""

    classes: list[str]
    codes: np.ndarray            # int16 [N], code theo vi tri trong classes
    benign_class: str = "BENIGN"

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def benign_index(self) -> int:
        return self.classes.index(self.benign_class)

    def counts(self) -> dict[str, int]:
        c = np.bincount(self.codes, minlength=self.num_classes)
        return {name: int(c[i]) for i, name in enumerate(self.classes)}

    def to_dict(self) -> dict:
        return {
            "classes": self.classes,
            "num_classes": self.num_classes,
            "benign_class": self.benign_class,
            "benign_index": self.benign_index,
            "counts": self.counts(),
            "binary_view": {"BENIGN": [self.benign_class],
                            "ATTACK": [c for c in self.classes if c != self.benign_class]},
        }


def scan_labels(files: Sequence[Path], schema: FeatureSchema,
                row_counts: Sequence[int]) -> LabelIndex:
    """Doc RIENG cot nhan cua moi file. 70.4M nhan int16 = 141 MB, chap nhan duoc."""
    total = int(sum(row_counts))
    raw = np.empty(total, dtype=object)
    pos = 0
    for fp, n in zip(files, row_counts):
        col = pq.read_table(fp, columns=[schema.label_column]).column(0)
        vals = col.to_pylist()
        if len(vals) != n:
            raise RuntimeError(f"{fp.name}: doc {len(vals)} nhan nhung metadata bao {n}")
        raw[pos:pos + n] = [str(v).strip() for v in vals]
        pos += n
        LOG.info("  nhan: %s (%d hang)", fp.name, n)

    classes = sorted({str(v) for v in raw})
    lut = {c: i for i, c in enumerate(classes)}
    codes = np.fromiter((lut[v] for v in raw), dtype=np.int16, count=total)
    del raw

    benign = next((c for c in classes if c.upper() == "BENIGN"), None)
    if benign is None:
        raise RuntimeError(f"Khong thay lop BENIGN trong {classes} - can cho binary view (muc 3.E)")
    return LabelIndex(classes, codes, benign)


# ------------------------------------------------------------------ split
@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int
    group_key: str | None = None

    def sizes(self) -> dict[str, int]:
        return {"train": self.train.size, "val": self.val.size, "test": self.test.size}


def make_splits(labels: LabelIndex, cfg: dict,
                groups: np.ndarray | None = None) -> Splits:
    """Stratified split 59.5 / 10.5 / 30 - muc 3.D. Split TRUOC moi bien doi."""
    scfg = cfg["split"]
    rng = np.random.default_rng(scfg["seed"])
    n = labels.codes.size

    if scfg.get("group_aware") and groups is not None:
        return _group_split(labels, groups, scfg, rng)

    test_frac = scfg["test_size"]
    val_frac_within = scfg["val_size_within_trainval"]
    tr, va, te = [], [], []
    for c in range(labels.num_classes):
        idx = np.flatnonzero(labels.codes == c)
        rng.shuffle(idx)
        n_test = int(round(idx.size * test_frac))
        n_val = int(round((idx.size - n_test) * val_frac_within))
        te.append(idx[:n_test])
        va.append(idx[n_test:n_test + n_val])
        tr.append(idx[n_test + n_val:])

    splits = Splits(
        np.sort(np.concatenate(tr)).astype(np.int64),
        np.sort(np.concatenate(va)).astype(np.int64),
        np.sort(np.concatenate(te)).astype(np.int64),
        scfg["seed"],
    )
    if scfg.get("assert_no_overlap", True):
        assert_no_leakage(splits, n)
    return splits


def _group_split(labels: LabelIndex, groups: np.ndarray, scfg: dict,
                 rng: np.random.Generator) -> Splits:
    """Chia theo group (vd __source_file_id) - khong de cung mot flow nam hai split."""
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_test = int(round(uniq.size * scfg["test_size"]))
    n_val = int(round((uniq.size - n_test) * scfg["val_size_within_trainval"]))
    g_test, g_val = set(uniq[:n_test].tolist()), set(uniq[n_test:n_test + n_val].tolist())

    is_test = np.isin(groups, list(g_test))
    is_val = np.isin(groups, list(g_val))
    splits = Splits(
        np.flatnonzero(~is_test & ~is_val).astype(np.int64),
        np.flatnonzero(is_val).astype(np.int64),
        np.flatnonzero(is_test).astype(np.int64),
        scfg["seed"], group_key=scfg.get("group_key"),
    )
    assert_no_leakage(splits, labels.codes.size)
    return splits


def assert_no_leakage(splits: Splits, n_total: int) -> dict:
    """Bang chung chong ro ri - muc 3.D, ghi vao sample_manifest.json."""
    tr, va, te = splits.train, splits.val, splits.test
    pairs = {"train_val": np.intersect1d(tr, va, assume_unique=True),
             "train_test": np.intersect1d(tr, te, assume_unique=True),
             "val_test": np.intersect1d(va, te, assume_unique=True)}
    for name, inter in pairs.items():
        if inter.size:
            raise RuntimeError(
                f"RO RI SPLIT: {inter.size} sample_id nam ca o {name}. "
                f"Vi du: {inter[:5].tolist()}"
            )
    covered = tr.size + va.size + te.size
    if covered != n_total:
        raise RuntimeError(f"Split phu {covered} / {n_total} hang - thieu hoac lap.")
    return {
        "n_total": int(n_total),
        "sizes": splits.sizes(),
        "fractions": {k: v / n_total for k, v in splits.sizes().items()},
        "overlaps": {k: int(v.size) for k, v in pairs.items()},
        "disjoint": True,
        "covers_all_rows": True,
        "seed": splits.seed,
        "group_key": splits.group_key,
        "sample_id_definition": "chi so hang toan cuc theo thu tu file da sort",
    }


# ---------------------------------------------------- thong ke tren train
@dataclass
class ColumnStats:
    """Thong ke tich luy CHI tren tap train - muc 3.C."""

    n_features: int
    count: np.ndarray = field(init=False)   # so gia tri huu han
    total: np.ndarray = field(init=False)
    minimum: np.ndarray = field(init=False)
    maximum: np.ndarray = field(init=False)
    n_nan: np.ndarray = field(init=False)
    n_inf: np.ndarray = field(init=False)
    n_rows: int = 0

    def __post_init__(self):
        f = self.n_features
        self.count = np.zeros(f, dtype=np.int64)
        self.total = np.zeros(f, dtype=np.float64)
        self.minimum = np.full(f, np.inf, dtype=np.float64)
        self.maximum = np.full(f, -np.inf, dtype=np.float64)
        self.n_nan = np.zeros(f, dtype=np.int64)
        self.n_inf = np.zeros(f, dtype=np.int64)

    def update(self, block: np.ndarray) -> None:
        """block: [n, F] float64, da thay +-Inf bang NaN o buoc goi."""
        finite = np.isfinite(block)
        self.n_rows += block.shape[0]
        self.count += finite.sum(axis=0)
        self.total += np.where(finite, block, 0.0).sum(axis=0)
        self.minimum = np.minimum(self.minimum, np.where(finite, block, np.inf).min(axis=0))
        self.maximum = np.maximum(self.maximum, np.where(finite, block, -np.inf).max(axis=0))

    def finalize(self, names: Sequence[str]) -> "ScalerStats":
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self.count > 0, self.total / np.maximum(self.count, 1), 0.0)
        lo = np.where(np.isfinite(self.minimum), self.minimum, 0.0)
        hi = np.where(np.isfinite(self.maximum), self.maximum, 0.0)
        missing_ratio = 1.0 - self.count / max(self.n_rows, 1)
        return ScalerStats(list(names), lo, hi, mean, missing_ratio,
                           self.n_nan.copy(), self.n_inf.copy(), self.n_rows)


@dataclass
class ScalerStats:
    """scaler_stats.json - fit CHI tren train, dung lai cho val/test va inference."""

    columns: list[str]
    minimum: np.ndarray
    maximum: np.ndarray
    mean: np.ndarray
    missing_ratio: np.ndarray
    n_nan: np.ndarray
    n_inf: np.ndarray
    n_train_rows: int

    @property
    def span(self) -> np.ndarray:
        s = self.maximum - self.minimum
        return np.where(s > 0, s, 1.0)   # cot hang so -> map ve 0, khong chia 0

    @property
    def constant_mask(self) -> np.ndarray:
        return (self.maximum - self.minimum) <= 0

    @property
    def hash(self) -> str:
        return sha256_of({
            "columns": self.columns,
            "min": np.round(self.minimum, 9).tolist(),
            "max": np.round(self.maximum, 9).tolist(),
            "mean": np.round(self.mean, 9).tolist(),
        })

    def transform(self, block: np.ndarray) -> np.ndarray:
        """NaN -> mean(train) -> Min-Max [0,1] -> clip. Tra ve float32."""
        out = np.where(np.isfinite(block), block, self.mean[None, :])
        out = (out - self.minimum[None, :]) / self.span[None, :]
        np.clip(out, 0.0, 1.0, out=out)      # val/test co the vuot bien cua train
        return out.astype(np.float32, copy=False)

    def subset(self, keep_mask: np.ndarray) -> "ScalerStats":
        """Cat thong ke theo mask cot. Khong can fit lai: Min-Max doc lap tung cot."""
        m = np.asarray(keep_mask, dtype=bool)
        if m.size != len(self.columns):
            raise ValueError(f"mask {m.size} != so cot {len(self.columns)}")
        return ScalerStats(
            [c for c, k in zip(self.columns, m) if k],
            self.minimum[m], self.maximum[m], self.mean[m],
            self.missing_ratio[m], self.n_nan[m], self.n_inf[m], self.n_train_rows,
        )

    def to_dict(self) -> dict:
        return {
            "fit_on": "train_split_only",
            "n_train_rows": self.n_train_rows,
            "scaler_hash": self.hash,
            "columns": [
                {"name": c, "min": float(self.minimum[i]), "max": float(self.maximum[i]),
                 "mean": float(self.mean[i]), "missing_ratio": float(self.missing_ratio[i]),
                 "n_nan": int(self.n_nan[i]), "n_inf": int(self.n_inf[i]),
                 "is_constant": bool(self.constant_mask[i])}
                for i, c in enumerate(self.columns)
            ],
        }


# ------------------------------------------------------------- doc parquet
def iter_row_groups(fp: Path, columns: Sequence[str]) -> Iterator[np.ndarray]:
    """Yield tung row-group duoi dang [n, F] float64, +-Inf da thanh NaN."""
    pf = pq.ParquetFile(fp)
    for i in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(i, columns=list(columns))
        block = np.column_stack([
            tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            for c in columns
        ])
        block[~np.isfinite(block)] = np.nan    # muc 3.C.2
        yield block


def fit_scaler_on_train(files: Sequence[Path], row_counts: Sequence[int],
                        schema: FeatureSchema, splits: Splits) -> ScalerStats:
    """Pass 1: thong ke min/max/mean CHI tren hang thuoc tap train (muc 3.C.4)."""
    stats = ColumnStats(schema.n_features)
    train_mask_all = np.zeros(int(sum(row_counts)), dtype=bool)
    train_mask_all[splits.train] = True

    offset = 0
    for fp, n_rows in zip(files, row_counts):
        for block in iter_row_groups(fp, schema.feature_columns):
            m = train_mask_all[offset:offset + block.shape[0]]
            if m.any():
                sub = block[m]
                stats.n_nan += np.isnan(sub).sum(axis=0)
                stats.update(sub)
            offset += block.shape[0]
        LOG.info("  scaler pass: %s", fp.name)
    if offset != train_mask_all.size:
        raise RuntimeError(f"Doc {offset} hang, mong doi {train_mask_all.size}")
    return stats.finalize(schema.feature_columns)


# ------------------------------------------------- loai cot (muc 3.B, 3.C.1)
@dataclass
class ColumnSelection:
    """Ket qua loai cot hang so / trung lap / thieu qua nhieu."""

    keep_mask: np.ndarray
    dropped: list[dict]

    @property
    def n_kept(self) -> int:
        return int(self.keep_mask.sum())


def read_train_sample(files: Sequence[Path], row_counts: Sequence[int],
                      schema: FeatureSchema, splits: Splits,
                      max_rows: int = 50_000) -> np.ndarray:
    """Doc mot mau hang thuoc tap train de doi chieu cot trung lap.

    Chi doc du max_rows roi dung - khong quet het 70.4M hang.
    """
    train_mask = np.zeros(int(sum(row_counts)), dtype=bool)
    train_mask[splits.train] = True

    chunks, taken, offset = [], 0, 0
    for fp in files:
        for block in iter_row_groups(fp, schema.feature_columns):
            n = block.shape[0]
            m = train_mask[offset:offset + n]
            offset += n
            if m.any():
                sub = block[m][: max_rows - taken]
                chunks.append(sub)
                taken += sub.shape[0]
                if taken >= max_rows:
                    return np.vstack(chunks)
    return np.vstack(chunks) if chunks else np.empty((0, schema.n_features))


def select_columns(schema: FeatureSchema, scaler: ScalerStats, cfg: dict,
                   sample: np.ndarray | None = None) -> ColumnSelection:
    """Quyet dinh giu / loai tung cot dac trung. Ghi ly do cho tung cot bi loai.

    Thu tu: thieu qua nguong -> hang so tuyet doi -> trung lap.
    Khong dung 'do quan trong' de loai (muc 3.B: feature_selection=none);
    chi loai cot khong mang thong tin hoac lap lai cot khac.
    """
    dcfg, pcfg = cfg["data"], cfg["preprocessing"]
    n = schema.n_features
    keep = np.ones(n, dtype=bool)
    dropped: list[dict] = []

    # 1. Thieu > nguong (muc 3.C.1). Ty le tinh tren tap train.
    thr = pcfg["drop_column_if_missing_ratio_above"]
    for i, name in enumerate(schema.feature_columns):
        if scaler.missing_ratio[i] > thr:
            keep[i] = False
            dropped.append({"name": name, "reason": "missing_ratio_above_threshold",
                            "missing_ratio": float(scaler.missing_ratio[i]),
                            "threshold": thr})

    # 2. Hang so tuyet doi (min == max tren train) - khong mang thong tin.
    if dcfg.get("drop_constant_columns", True):
        for i, name in enumerate(schema.feature_columns):
            if keep[i] and scaler.constant_mask[i]:
                keep[i] = False
                dropped.append({"name": name, "reason": "constant_on_train",
                                "value": float(scaler.minimum[i])})

    # 3. Trung lap. Loc ung vien bang chu ky thong ke roi doi chieu that tren mau.
    if dcfg.get("drop_duplicate_columns", True) and sample is not None and sample.size:
        sig: dict[tuple, list[int]] = {}
        for i in range(n):
            if not keep[i]:
                continue
            key = (round(float(scaler.minimum[i]), 9), round(float(scaler.maximum[i]), 9),
                   round(float(scaler.mean[i]), 9), int(scaler.n_nan[i]))
            sig.setdefault(key, []).append(i)

        for group in sig.values():
            if len(group) < 2:
                continue
            keeper = group[0]
            for j in group[1:]:
                a, b = sample[:, keeper], sample[:, j]
                same = np.array_equal(np.nan_to_num(a, nan=np.inf),
                                      np.nan_to_num(b, nan=np.inf))
                if same:
                    keep[j] = False
                    dropped.append({
                        "name": schema.feature_columns[j],
                        "reason": "duplicate_of",
                        "duplicate_of": schema.feature_columns[keeper],
                        "verified_on_rows": int(sample.shape[0]),
                    })

    return ColumnSelection(keep, dropped)


def build_feature_cache(files: Sequence[Path], row_counts: Sequence[int],
                        schema: FeatureSchema, scaler: ScalerStats,
                        cache_path: Path) -> tuple[np.memmap, float]:
    """Pass 2: impute + Min-Max -> memmap float32 [N, F]. Tra ve (memmap, giay)."""
    total = int(sum(row_counts))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    mm = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32,
                                   shape=(total, schema.n_features))
    offset = 0
    for fp in files:
        for block in iter_row_groups(fp, schema.feature_columns):
            n = block.shape[0]
            mm[offset:offset + n] = scaler.transform(block)
            offset += n
        LOG.info("  cache: %s -> %d/%d hang", fp.name, offset, total)
    mm.flush()
    if offset != total:
        raise RuntimeError(f"Ghi {offset} hang, mong doi {total}")
    return mm, time.perf_counter() - t0


# ------------------------------------------------------------------ manifest
def build_cache_manifest(schema: FeatureSchema, scaler: ScalerStats,
                         geom, row_counts: Sequence[int], files: Sequence[Path],
                         cache_build_seconds: float) -> dict:
    """cache_manifest.json - phien sau xac minh cache dung lai khop hệt (muc 3.A)."""
    from .wavelet import geometry_hash
    return {
        "feature_schema_hash": schema.hash,
        "scaler_hash": scaler.hash,
        "wavelet_geometry_hash": geometry_hash(geom, schema.feature_columns),
        "wavelet_geometry": geom.to_dict(),
        "n_rows": int(sum(row_counts)),
        "n_features": schema.n_features,
        "files": [{"name": f.name, "rows": int(n)} for f, n in zip(files, row_counts)],
        "cache_layout": "feature_cache.npy = float32 [N, F] da Min-Max; "
                        "SWT tinh on-the-fly trong DataLoader (khong cache subband)",
        "cache_build_seconds": round(cache_build_seconds, 2),
    }


def list_parquet_files(input_dir: Path, pattern: str = "**/*.parquet") -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise RuntimeError(f"Khong co file khop {pattern!r} trong {input_dir}")
    return files


def row_counts_of(files: Sequence[Path]) -> list[int]:
    return [pq.ParquetFile(f).metadata.num_rows for f in files]


# --------------------------------------------------------- DataLoader (3.F)
class MDDCCDataset:
    """Doc memmap float32 [N, F], tra ve tensor [4, S, S] + nhan.

    SWT tinh on-the-fly: khong cache subband vi [N,4,9,9] = 91.3 GB voi
    70.4M hang (phuong an C). Chi phi ~0.2 h/epoch so voi ~4.1 h compute.
    """

    def __init__(self, cache_path, indices, codes, geom, *, memmap=None):
        import torch  # noqa: F401  (bao dam torch san sang o worker)

        self.cache_path = Path(cache_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.codes = np.asarray(codes, dtype=np.int64)
        self.geom = geom
        self._mm = memmap          # worker se tu mo lai neu None

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def features(self) -> np.ndarray:
        if self._mm is None:       # mo lazy de an toan voi num_workers > 0
            self._mm = np.load(self.cache_path, mmap_mode="r")
        return self._mm

    def __getitem__(self, i):
        import torch
        from .wavelet import transform_batch

        row = int(self.indices[i])
        x = np.asarray(self.features[row], dtype=np.float64)[None, :]
        img = transform_batch(x, self.geom)[0]          # [4, S, S] float32
        return torch.from_numpy(img), int(self.codes[row])

    def batch(self, rows: np.ndarray):
        """Bien doi ca lo cung luc - nhanh hon nhieu so voi tung dong."""
        import torch
        from .wavelet import transform_batch

        rows = np.asarray(rows, dtype=np.int64)
        x = np.asarray(self.features[rows], dtype=np.float64)
        img = transform_batch(x, self.geom)
        return torch.from_numpy(img), torch.from_numpy(self.codes[rows])


class BatchSampler:
    """Shuffle theo (seed, epoch) de tai lap duoc, ho tro bo qua n batch dau khi resume."""

    def __init__(self, n: int, batch_size: int, *, seed: int, shuffle: bool = True,
                 drop_last: bool = False):
        self.n, self.batch_size = int(n), int(batch_size)
        self.seed, self.shuffle, self.drop_last = seed, shuffle, drop_last

    def n_batches(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return math.ceil(self.n / self.batch_size)

    def permutation(self, epoch: int) -> np.ndarray:
        if not self.shuffle:
            return np.arange(self.n, dtype=np.int64)
        # seed theo (seed, epoch) - muc 3.F, tai lap chinh xac khi resume
        return np.random.default_rng([self.seed, epoch]).permutation(self.n)

    def batches(self, epoch: int, skip: int = 0) -> Iterator[np.ndarray]:
        perm = self.permutation(epoch)
        total = self.n_batches()
        for b in range(skip, total):
            chunk = perm[b * self.batch_size:(b + 1) * self.batch_size]
            if chunk.size:
                yield chunk


# ------------------------------------------------------------- orchestrator
@dataclass
class PreparedData:
    schema: "FeatureSchema"
    labels: "LabelIndex"
    splits: "Splits"
    scaler: "ScalerStats"
    geom: object
    cache_path: Path
    cache_build_seconds: float
    artifacts: dict


def prepare_dataset(cfg: dict, out_dir: Path, *, files=None) -> PreparedData:
    """Chay tron muc 3: discovery -> split -> fit scaler -> transform -> cache.

    Sinh feature_schema.json, label_mapping.json, sample_manifest.json,
    preprocessing.json, scaler_stats.json, data_profile.json, cache_manifest.json.
    """
    from .wavelet import geometry_from_config, geometry_hash

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dcfg = cfg["data"]

    input_dir = Path(dcfg["kaggle_input_dir"])
    files = list(files) if files else list_parquet_files(input_dir, dcfg["parquet_glob"])
    rc = row_counts_of(files)
    LOG.info("Tim thay %d file, %d hang", len(files), sum(rc))

    schema = discover_schema(files, cfg)
    assert_schema_consistent(files, schema)
    labels = scan_labels(files, schema, rc)

    groups = None
    if cfg["split"].get("group_aware"):
        key = cfg["split"].get("group_key")
        if not key:
            raise RuntimeError("split.group_aware=true nhung thieu split.group_key")
        groups = np.concatenate([
            pq.read_table(f, columns=[key]).column(0).to_numpy(zero_copy_only=False)
            for f in files
        ])

    splits = make_splits(labels, cfg, groups=groups)
    manifest = assert_no_leakage(splits, labels.codes.size)

    # Fit scaler tren train, roi moi loai cot - vi tieu chi loai (missing/hang so/
    # trung lap) deu phai tinh CHI tren tap train. Min-Max doc lap tung cot nen
    # cat bot cot khong lam sai thong ke, khong can fit lai.
    scaler_full = fit_scaler_on_train(files, rc, schema, splits)
    sample = read_train_sample(files, rc, schema, splits)
    selection = select_columns(schema, scaler_full, cfg, sample)
    del sample

    n_before = schema.n_features
    schema = schema.pruned(selection.keep_mask, selection.dropped)
    scaler = scaler_full.subset(selection.keep_mask)
    LOG.info("Loai %d cot, giu %d/%d feature", len(selection.dropped),
             schema.n_features, n_before)

    geom = geometry_from_config(schema.n_features, cfg)

    cache_path = Path(dcfg["cache_dir"]) / "feature_cache.npy"
    _, secs = build_feature_cache(files, rc, schema, scaler, cache_path)

    per_split_counts = {
        name: {labels.classes[c]: int((labels.codes[idx] == c).sum())
               for c in range(labels.num_classes)}
        for name, idx in (("train", splits.train), ("val", splits.val), ("test", splits.test))
    }
    artifacts = {
        "feature_schema.json": schema.to_dict(),
        "label_mapping.json": labels.to_dict(),
        "sample_manifest.json": {**manifest, "per_split_class_counts": per_split_counts},
        "scaler_stats.json": scaler.to_dict(),
        "preprocessing.json": {
            "order": cfg["preprocessing"]["order"],
            "inf_to_nan": True,
            "impute_strategy": "mean_train_only",
            "scaler": "minmax_[0,1]_fit_train_only",
            "clip_to_range": True,
            "dropped_columns": schema.dropped_columns,
            "n_features_before_pruning": n_before,
            "n_features_final": schema.n_features,
            "pruning_rules": {
                "missing_ratio_above": cfg["preprocessing"]["drop_column_if_missing_ratio_above"],
                "drop_constant_columns": dcfg.get("drop_constant_columns", True),
                "drop_duplicate_columns": dcfg.get("drop_duplicate_columns", True),
                "duplicate_check": "chu ky thong ke tren train + doi chieu that tren mau <= 50k hang",
            },
            "wavelet": geom.to_dict(),
            "wavelet_geometry_hash": geometry_hash(geom, schema.feature_columns),
            "swt_padding": {"sequence": geom.pad_mode, "image": geom.image_pad_mode,
                            "n_swt_pad": geom.n_padded - geom.n_features,
                            "n_image_pad": geom.n_image_pad},
            "label_detection": schema.label_detection,
            "feature_selection": "none",
        },
        "data_profile.json": {
            "input_dir": str(input_dir),
            "files": [{"name": f.name, "rows": int(n)} for f, n in zip(files, rc)],
            "total_rows": int(sum(rc)),
            "n_features": schema.n_features,
            "num_class": labels.num_classes,
            "class_counts": labels.counts(),
            "split_sizes": splits.sizes(),
            "cache_bytes": int(sum(rc) * schema.n_features * 4),
            "cache_build_seconds": round(secs, 2),
        },
        "cache_manifest.json": build_cache_manifest(schema, scaler, geom, rc, files, secs),
    }
    for name, payload in artifacts.items():
        (out_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        LOG.info("  ghi %s", name)

    return PreparedData(schema, labels, splits, scaler, geom, cache_path, secs, artifacts)


def _main() -> int:
    import argparse
    import yaml

    ap = argparse.ArgumentParser(description="Buoc 2b - dung cache va artifact du lieu")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--input-dir", type=Path, default=None, help="ghi de data.kaggle_input_dir")
    ap.add_argument("--cache-dir", type=Path, default=None, help="ghi de data.cache_dir")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.input_dir:
        cfg["data"]["kaggle_input_dir"] = str(args.input_dir)
    if args.cache_dir:
        cfg["data"]["cache_dir"] = str(args.cache_dir)

    p = prepare_dataset(cfg, args.out_dir)
    print(f"\nOK: {p.labels.num_classes} lop, {p.schema.n_features} feature, "
          f"anh {p.geom.image_shape}, split {p.splits.sizes()}, "
          f"cache_build_seconds={p.cache_build_seconds:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
