"""Kiem chung muc 3: schema, chong ro ri split, scaler fit chi tren train, cache."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import data as D
from src import wavelet as W

N_FEAT = 20
CLASSES = ["BENIGN", "Syn", "TFTP", "WebDDoS"]


def make_parquet(tmp_path, n_files=3, rows=500, seed=0):
    """Dung dataset gia mo phong dung dac diem CIC-FlowMeter + provenance."""
    rng = np.random.default_rng(seed)
    out = tmp_path / "data"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in range(n_files):
        cols = {
            "Unnamed: 0": pa.array(np.arange(rows)),
            " Flow ID": pa.array(["fid"] * rows),
            " Source IP": pa.array(["10.0.0.1"] * rows),
            " Destination IP": pa.array(["10.0.0.2"] * rows),
            " Timestamp": pa.array(["2018-12-01"] * rows),
            "SimillarHTTP": pa.array([""] * rows),
        }
        for i in range(N_FEAT):
            v = rng.normal(i, 1 + i, rows)
            if i == 0:                       # cot co NaN va Inf
                v[rng.integers(0, rows, 20)] = np.nan
                v[rng.integers(0, rows, 5)] = np.inf
            if i == 1:
                v[:] = 7.0                   # cot hang so
            cols[f" Feat{i}"] = pa.array(v)
        cols["__capture_day"] = pa.array(["01-12"] * rows)
        cols["__source_file_id"] = pa.array([f] * rows)
        cols["__source_row_id"] = pa.array(np.arange(rows))
        cols[" Label"] = pa.array(rng.choice(CLASSES, rows, p=[.1, .5, .35, .05]))
        p = out / f"part{f}.parquet"
        pq.write_table(pa.table(cols), p, row_group_size=137)   # nhieu row-group
        paths.append(p)
    return out, paths


@pytest.fixture
def cfg():
    return {
        "data": {
            "label": {"column_candidates": ["Label", " Label", "Class", "label"],
                      "fail_if_undetermined": True},
            "drop_identifier_columns": True,
            "identifier_columns": ["Flow ID", "Source IP", "Destination IP",
                                   "Timestamp", "Unnamed: 0", "SimillarHTTP"],
        },
        "split": {"test_size": 0.30, "val_size_within_trainval": 0.15,
                  "stratify": True, "group_aware": False, "group_key": None,
                  "seed": 42, "assert_no_overlap": True},
        "wavelet": {"name": "db4", "level": 3, "transform": "swt",
                    "pad_mode": "reflect", "reshape": {"min_side": 8, "force_side": None}},
    }


# ------------------------------------------------------------------ schema
def test_label_column_with_leading_space_is_found(tmp_path, cfg):
    _, files = make_parquet(tmp_path)
    s = D.discover_schema(files, cfg)
    assert s.label_column == " Label"          # dac trung CIC-FlowMeter
    assert "column_candidate" in s.label_detection


def test_identifier_and_provenance_columns_are_dropped(tmp_path, cfg):
    _, files = make_parquet(tmp_path)
    s = D.discover_schema(files, cfg)
    dropped = {d["name"] for d in s.dropped_columns}
    assert {" Flow ID", " Source IP", " Destination IP", " Timestamp",
            "Unnamed: 0", "SimillarHTTP"} <= dropped
    assert {"__capture_day", "__source_file_id", "__source_row_id"} <= dropped
    assert s.n_features == N_FEAT
    assert all(c.startswith(" Feat") for c in s.feature_columns)


def test_all_features_kept_no_feature_selection(tmp_path, cfg):
    _, files = make_parquet(tmp_path)
    s = D.discover_schema(files, cfg)
    assert s.n_features == N_FEAT, "muc 3.B: feature_selection=none"


def test_schema_hash_changes_with_order(tmp_path, cfg):
    _, files = make_parquet(tmp_path)
    s = D.discover_schema(files, cfg)
    other = D.FeatureSchema(list(reversed(s.feature_columns)), s.label_column, [], "x")
    assert s.hash != other.hash
    with pytest.raises(RuntimeError, match="fail-fast"):
        s.assert_matches(other.hash)


# ------------------------------------------------------------------ split
def test_split_has_no_leakage_and_covers_all_rows(tmp_path, cfg):
    _, files = make_parquet(tmp_path)
    s = D.discover_schema(files, cfg)
    rc = D.row_counts_of(files)
    labels = D.scan_labels(files, s, rc)
    sp = D.make_splits(labels, cfg)

    man = D.assert_no_leakage(sp, labels.codes.size)
    assert man["disjoint"] and man["covers_all_rows"]
    assert all(v == 0 for v in man["overlaps"].values())


def test_split_ratios_are_close_to_target(tmp_path, cfg):
    _, files = make_parquet(tmp_path, rows=2000)
    s = D.discover_schema(files, cfg)
    labels = D.scan_labels(files, s, D.row_counts_of(files))
    sp = D.make_splits(labels, cfg)
    n = labels.codes.size
    assert abs(sp.test.size / n - 0.30) < 0.01
    assert abs(sp.val.size / n - 0.105) < 0.01
    assert abs(sp.train.size / n - 0.595) < 0.01


def test_split_is_stratified(tmp_path, cfg):
    _, files = make_parquet(tmp_path, rows=2000)
    s = D.discover_schema(files, cfg)
    labels = D.scan_labels(files, s, D.row_counts_of(files))
    sp = D.make_splits(labels, cfg)
    for c in range(labels.num_classes):
        overall = (labels.codes == c).mean()
        in_test = (labels.codes[sp.test] == c).mean()
        assert abs(overall - in_test) < 0.02, f"lop {labels.classes[c]} lech ty le"


def test_split_is_reproducible(tmp_path, cfg):
    _, files = make_parquet(tmp_path)
    s = D.discover_schema(files, cfg)
    labels = D.scan_labels(files, s, D.row_counts_of(files))
    a, b = D.make_splits(labels, cfg), D.make_splits(labels, cfg)
    assert np.array_equal(a.train, b.train) and np.array_equal(a.test, b.test)


def test_leakage_assert_actually_catches_overlap():
    bad = D.Splits(np.array([1, 2, 3]), np.array([3, 4]), np.array([5]), seed=0)
    with pytest.raises(RuntimeError, match="RO RI SPLIT"):
        D.assert_no_leakage(bad, 6)


def test_group_aware_split_keeps_groups_intact(tmp_path, cfg):
    _, files = make_parquet(tmp_path, n_files=10, rows=300)
    cfg["split"]["group_aware"] = True
    cfg["split"]["group_key"] = "__source_file_id"
    s = D.discover_schema(files, cfg)
    rc = D.row_counts_of(files)
    labels = D.scan_labels(files, s, rc)
    groups = np.repeat(np.arange(len(files)), rc)
    sp = D.make_splits(labels, cfg, groups=groups)
    for a, b in [(sp.train, sp.test), (sp.train, sp.val), (sp.val, sp.test)]:
        assert not (set(groups[a]) & set(groups[b])), "mot group nam o hai split"


# ----------------------------------------------------------------- scaler
def test_scaler_is_fit_on_train_only(tmp_path, cfg):
    _, files = make_parquet(tmp_path, rows=1000)
    s = D.discover_schema(files, cfg)
    rc = D.row_counts_of(files)
    labels = D.scan_labels(files, s, rc)
    sp = D.make_splits(labels, cfg)
    sc = D.fit_scaler_on_train(files, rc, s, sp)
    assert sc.n_train_rows == sp.train.size, "chi duoc dung hang thuoc tap train"
    assert len(sc.columns) == s.n_features


def test_inf_counted_as_missing_and_imputed(tmp_path, cfg):
    _, files = make_parquet(tmp_path, rows=1000)
    s = D.discover_schema(files, cfg)
    rc = D.row_counts_of(files)
    labels = D.scan_labels(files, s, rc)
    sp = D.make_splits(labels, cfg)
    sc = D.fit_scaler_on_train(files, rc, s, sp)
    assert sc.missing_ratio[0] > 0, "Feat0 co NaN + Inf"
    assert np.isfinite(sc.minimum[0]) and np.isfinite(sc.maximum[0])


def test_minmax_output_in_range_and_constant_column_safe(tmp_path, cfg):
    _, files = make_parquet(tmp_path, rows=1000)
    s = D.discover_schema(files, cfg)
    rc = D.row_counts_of(files)
    labels = D.scan_labels(files, s, rc)
    sp = D.make_splits(labels, cfg)
    sc = D.fit_scaler_on_train(files, rc, s, sp)

    assert sc.constant_mask[1], "Feat1 la hang so"
    block = next(D.iter_row_groups(files[0], s.feature_columns))
    out = sc.transform(block)
    assert out.dtype == np.float32
    assert np.isfinite(out).all(), "khong duoc con NaN/Inf sau transform"
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_cache_roundtrip_and_manifest(tmp_path, cfg):
    _, files = make_parquet(tmp_path, rows=800)
    s = D.discover_schema(files, cfg)
    rc = D.row_counts_of(files)
    labels = D.scan_labels(files, s, rc)
    sp = D.make_splits(labels, cfg)
    sc = D.fit_scaler_on_train(files, rc, s, sp)

    cache = tmp_path / "cache" / "feature_cache.npy"
    mm, secs = D.build_feature_cache(files, rc, s, sc, cache)
    assert mm.shape == (sum(rc), s.n_features)
    assert mm.dtype == np.float32
    assert secs >= 0

    geom = W.geometry_from_config(s.n_features, cfg)
    man = D.build_cache_manifest(s, sc, geom, rc, files, secs)
    assert man["feature_schema_hash"] == s.hash
    assert man["scaler_hash"] == sc.hash
    assert man["n_rows"] == sum(rc)
    assert "cache_build_seconds" in man


# -------------------------------------------------------------- dataloader
def test_dataset_returns_four_subband_images(tmp_path, cfg):
    import torch

    _, files = make_parquet(tmp_path, rows=400)
    s = D.discover_schema(files, cfg)
    rc = D.row_counts_of(files)
    labels = D.scan_labels(files, s, rc)
    sp = D.make_splits(labels, cfg)
    sc = D.fit_scaler_on_train(files, rc, s, sp)
    cache = tmp_path / "cache" / "feature_cache.npy"
    D.build_feature_cache(files, rc, s, sc, cache)

    geom = W.geometry_from_config(s.n_features, cfg)
    ds = D.MDDCCDataset(cache, sp.train, labels.codes, geom)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert tuple(x.shape) == (4, geom.side, geom.side)
    assert 0 <= y < labels.num_classes

    xb, yb = ds.batch(sp.train[:64])
    assert tuple(xb.shape) == (64, 4, geom.side, geom.side)
    assert xb.dtype == torch.float32 and yb.shape == (64,)


def test_batch_sampler_is_reproducible_and_resumable():
    bs = D.BatchSampler(1000, 128, seed=42)
    assert bs.n_batches() == 8
    a = list(bs.batches(epoch=3))
    b = list(bs.batches(epoch=3))
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "cung epoch -> cung thu tu"
    assert not np.array_equal(a[0], list(bs.batches(epoch=4))[0]), "khac epoch -> khac"

    # resume giua epoch: bo qua 3 batch dau phai khop duoi cua lan chay day du
    resumed = list(bs.batches(epoch=3, skip=3))
    assert len(resumed) == len(a) - 3
    assert all(np.array_equal(x, y) for x, y in zip(resumed, a[3:]))


def test_batch_sampler_covers_every_sample_exactly_once():
    bs = D.BatchSampler(1000, 128, seed=7)
    seen = np.concatenate(list(bs.batches(epoch=1)))
    assert seen.size == 1000
    assert np.array_equal(np.sort(seen), np.arange(1000)), "khong lap, khong bo sot"
