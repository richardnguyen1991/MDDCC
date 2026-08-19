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
            "drop_constant_columns": True,
            "drop_duplicate_columns": True,
        },
        "preprocessing": {
            "drop_column_if_missing_ratio_above": 0.80,
            "order": ["split", "fit_scaler_on_train", "transform", "swt", "cache"],
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


# ------------------------------------------------ loai cot (muc 3.B, 3.C.1)
def _fit(tmp_path, cfg, rows=1000, **kw):
    _, files = make_parquet(tmp_path, rows=rows, **kw)
    s = D.discover_schema(files, cfg)
    rc = D.row_counts_of(files)
    labels = D.scan_labels(files, s, rc)
    sp = D.make_splits(labels, cfg)
    sc = D.fit_scaler_on_train(files, rc, s, sp)
    return files, rc, s, labels, sp, sc


def test_constant_column_is_dropped(tmp_path, cfg):
    files, rc, s, _, sp, sc = _fit(tmp_path, cfg)
    sample = D.read_train_sample(files, rc, s, sp)
    sel = D.select_columns(s, sc, cfg, sample)

    dropped = {d["name"]: d for d in sel.dropped}
    assert " Feat1" in dropped, "Feat1 la hang so, phai bi loai"
    assert dropped[" Feat1"]["reason"] == "constant_on_train"
    assert sel.n_kept == N_FEAT - 1


def test_constant_column_kept_when_disabled(tmp_path, cfg):
    cfg["data"]["drop_constant_columns"] = False
    cfg["data"]["drop_duplicate_columns"] = False
    files, rc, s, _, sp, sc = _fit(tmp_path, cfg)
    sel = D.select_columns(s, sc, cfg, None)
    assert sel.n_kept == N_FEAT, "tat co che thi phai giu nguyen"


def test_high_missing_column_is_dropped(tmp_path, cfg):
    cfg["preprocessing"] = {"drop_column_if_missing_ratio_above": 0.001}
    files, rc, s, _, sp, sc = _fit(tmp_path, cfg)
    sel = D.select_columns(s, sc, cfg, None)
    reasons = {d["name"]: d["reason"] for d in sel.dropped}
    assert reasons.get(" Feat0") == "missing_ratio_above_threshold", \
        "Feat0 co NaN+Inf, vuot nguong 0.1% thi phai bi loai"


def test_duplicate_column_is_detected(tmp_path, cfg):
    """Dung cot lap that: Feat5 duoc sao chep sang FeatDup."""
    rng = np.random.default_rng(9)
    out = tmp_path / "dup"
    out.mkdir(parents=True)
    rows = 800
    base = rng.normal(3, 2, rows)
    cols = {" Label": pa.array(rng.choice(CLASSES, rows, p=[.1, .5, .35, .05]))}
    for i in range(4):
        cols[f" Feat{i}"] = pa.array(rng.normal(i, 1 + i, rows))
    cols[" Feat5"] = pa.array(base)
    cols[" FeatDup"] = pa.array(base.copy())
    pq.write_table(pa.table(cols), out / "p.parquet")

    files = [out / "p.parquet"]
    s = D.discover_schema(files, cfg)
    rc = D.row_counts_of(files)
    labels = D.scan_labels(files, s, rc)
    sp = D.make_splits(labels, cfg)
    sc = D.fit_scaler_on_train(files, rc, s, sp)
    sample = D.read_train_sample(files, rc, s, sp)
    sel = D.select_columns(s, sc, cfg, sample)

    dups = [d for d in sel.dropped if d["reason"] == "duplicate_of"]
    assert len(dups) == 1
    assert {dups[0]["name"], dups[0]["duplicate_of"]} == {" Feat5", " FeatDup"}


def test_read_train_sample_only_returns_train_rows(tmp_path, cfg):
    files, rc, s, _, sp, _ = _fit(tmp_path, cfg)
    sample = D.read_train_sample(files, rc, s, sp, max_rows=100)
    assert sample.shape == (100, s.n_features)
    assert sample.shape[0] <= sp.train.size


def test_pruning_flows_into_schema_scaler_and_geometry(tmp_path, cfg):
    """Sau khi loai cot, schema/scaler/hinh hoc wavelet phai dong bo."""
    files, rc, s, _, sp, sc = _fit(tmp_path, cfg)
    sample = D.read_train_sample(files, rc, s, sp)
    sel = D.select_columns(s, sc, cfg, sample)

    s2, sc2 = s.pruned(sel.keep_mask, sel.dropped), sc.subset(sel.keep_mask)
    assert s2.n_features == sel.n_kept == len(sc2.columns)
    assert s2.feature_columns == sc2.columns, "thu tu cot phai khop tuyet doi"
    assert s2.hash != s.hash, "doi tap cot phai doi feature_schema_hash"

    geom = W.geometry_from_config(s2.n_features, cfg)
    assert geom.n_features == s2.n_features


def test_prepare_dataset_applies_pruning_end_to_end(tmp_path, cfg):
    cfg["data"]["kaggle_input_dir"] = str(tmp_path / "data")
    cfg["data"]["cache_dir"] = str(tmp_path / "cache")
    cfg["data"]["parquet_glob"] = "**/*.parquet"
    cfg["preprocessing"]["order"] = ["split", "fit_scaler_on_train", "transform", "swt", "cache"]
    make_parquet(tmp_path, n_files=2, rows=1000)

    p = D.prepare_dataset(cfg, tmp_path / "artifacts")
    pre = p.artifacts["preprocessing.json"]
    assert pre["n_features_before_pruning"] == N_FEAT
    assert pre["n_features_final"] == p.schema.n_features < N_FEAT
    assert any(d["reason"] == "constant_on_train" for d in pre["dropped_columns"])

    # cache phai co dung so cot sau khi loai
    mm = np.load(p.cache_path, mmap_mode="r")
    assert mm.shape[1] == p.schema.n_features
    assert p.geom.n_features == p.schema.n_features


# --------------------------------------------------- gop nhan dong nghia (3.E)
def _labelled_parquet(tmp_path, labels_seq):
    """Tao parquet voi day nhan cho truoc, du cot de discover_schema chay duoc."""
    out = tmp_path / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows = len(labels_seq)
    rng = np.random.default_rng(0)
    cols = {f" Feat{i}": pa.array(rng.normal(i, 1 + i, rows)) for i in range(4)}
    cols[" Label"] = pa.array(list(labels_seq))
    p = out / "p.parquet"
    pq.write_table(pa.table(cols), p)
    return [p]


def test_merge_map_collapses_synonym_labels(tmp_path, cfg):
    files = _labelled_parquet(
        tmp_path, ["BENIGN"] * 10 + ["UDP-lag"] * 30 + ["UDPLag"] * 5 + ["Syn"] * 20)
    s = D.discover_schema(files, cfg)
    rc = D.row_counts_of(files)

    raw = D.scan_labels(files, s, rc)
    assert raw.num_classes == 4
    assert raw.counts()["UDP-lag"] == 30 and raw.counts()["UDPLag"] == 5

    merged = D.scan_labels(files, s, rc, merge_map={"UDP-lag": "UDPLag"})
    assert merged.num_classes == 3, "gop xong phai con 3 lop"
    assert "UDP-lag" not in merged.classes
    assert merged.counts()["UDPLag"] == 35, "so mau phai cong don"
    assert merged.counts()["BENIGN"] == 10 and merged.counts()["Syn"] == 20


def test_merge_preserves_raw_counts_for_traceability(tmp_path, cfg):
    files = _labelled_parquet(
        tmp_path, ["BENIGN"] * 10 + ["UDP-lag"] * 30 + ["UDPLag"] * 5)
    s = D.discover_schema(files, cfg)
    m = D.scan_labels(files, s, D.row_counts_of(files),
                      merge_map={"UDP-lag": "UDPLag"})

    info = m.to_dict()["label_merge"]
    assert info["applied"] is True
    assert info["map"] == {"UDP-lag": "UDPLag"}
    assert info["merged_into"] == {"UDPLag": ["UDP-lag"]}
    assert info["raw_counts_before_merge"] == {"BENIGN": 10, "UDP-lag": 30, "UDPLag": 5}


def test_no_merge_key_when_map_empty(tmp_path, cfg):
    files = _labelled_parquet(tmp_path, ["BENIGN"] * 5 + ["Syn"] * 5)
    s = D.discover_schema(files, cfg)
    m = D.scan_labels(files, s, D.row_counts_of(files), merge_map={})
    assert "label_merge" not in m.to_dict(), "khong gop thi khong ghi khoa nay"


def test_merge_map_with_unknown_label_fails_fast(tmp_path, cfg):
    files = _labelled_parquet(tmp_path, ["BENIGN"] * 5 + ["Syn"] * 5)
    s = D.discover_schema(files, cfg)
    with pytest.raises(RuntimeError, match="khong ton tai"):
        D.scan_labels(files, s, D.row_counts_of(files),
                      merge_map={"KhongCoLop": "Syn"})


def test_merged_labels_flow_through_prepare_dataset(tmp_path, cfg):
    files = _labelled_parquet(
        tmp_path, (["BENIGN"] * 40 + ["UDP-lag"] * 60 + ["UDPLag"] * 20 + ["Syn"] * 80) * 5)
    cfg["data"]["kaggle_input_dir"] = str(tmp_path / "data")
    cfg["data"]["cache_dir"] = str(tmp_path / "cache")
    cfg["data"]["parquet_glob"] = "**/*.parquet"
    cfg["data"]["label"]["merge_map"] = {"UDP-lag": "UDPLag"}

    p = D.prepare_dataset(cfg, tmp_path / "artifacts")
    assert p.labels.num_classes == 3
    lm = p.artifacts["label_mapping.json"]
    assert lm["num_classes"] == 3
    assert lm["counts"]["UDPLag"] == 400          # (60 + 20) * 5
    assert lm["label_merge"]["applied"] is True
    assert set(lm["binary_view"]["ATTACK"]) == {"UDPLag", "Syn"}
