"""Kiem chung SWT theo muc 2.A: dung 4 subband, moi subband dai bang chuoi goc."""
from __future__ import annotations

import numpy as np
import pytest
import pywt

from src import wavelet as W


def test_geometry_matches_paper_formula():
    g = W.compute_geometry(80, level=3, min_side=8)
    assert g.n_padded == 80          # ceil(80/8)*8
    assert g.side == 9               # max(8, ceil(sqrt(80)))
    assert g.side * g.side >= g.n_padded
    assert g.image_shape == (4, 9, 9)


@pytest.mark.parametrize("n_features", [40, 48, 70, 76, 80, 84, 88])
def test_padding_is_multiple_of_2_pow_level(n_features):
    g = W.compute_geometry(n_features, level=3)
    assert g.n_padded % 8 == 0
    assert g.n_padded >= n_features
    assert g.n_padded - n_features < 8


def test_min_side_floor_is_respected():
    """San min_side=8 duoc ap dung truoc, sau do rang buoc min_final_map nang len 9."""
    g = W.compute_geometry(16, level=3, min_side=8)
    assert g.side_bumped_from == 8   # ceil(sqrt(16))=4 -> san 8
    assert g.side == 9               # 8 se sup ve 1x1 sau 3 lan pool


# ------------------------------------------- rang buoc chong sup do feature map
def test_pooled_side_matches_maxpool_behaviour():
    assert W.pooled_side(9, 3) == 2          # ceil: 9->5->3->2
    assert W.pooled_side(9, 3, ceil_mode=False) == 1   # floor: 9->4->2->1
    assert W.pooled_side(10, 3) == 2
    assert W.pooled_side(8, 3) == 1


def test_side_is_bumped_when_feature_map_would_collapse():
    """F <= 64 cho S = 8, sau 3 lan pool con 1x1 -> chi 32 chieu truoc FC."""
    g = W.compute_geometry(64, level=3)
    assert g.side_bumped_from == 8
    assert g.side == 9
    assert g.final_map_side == 2
    assert g.flatten_dim(32) == 128


def test_side_not_bumped_when_already_safe():
    g = W.compute_geometry(81, level=3)
    assert g.n_padded == 88 and g.side == 10
    assert g.side_bumped_from is None
    assert g.final_map_side == 2 and g.flatten_dim(32) == 128


def test_unsafe_force_side_is_rejected():
    with pytest.raises(ValueError, match="min_final_map"):
        W.compute_geometry(64, level=3, force_side=8)


def test_force_side_too_small_for_swt_is_rejected():
    with pytest.raises(ValueError, match="qua nho"):
        W.compute_geometry(81, level=3, force_side=8)


def test_floor_pooling_needs_a_much_larger_side():
    """Neu tat ceil_mode thi phai nang S len nhieu hon de tranh sup do."""
    g = W.compute_geometry(81, level=3, pool_ceil_mode=False)
    assert g.final_map_side >= 2
    assert g.side >= 16


def test_exactly_four_subbands_of_original_length():
    """Yeu cau cot loi: n+1 = 4 subsequence, KHONG downsampling."""
    g = W.compute_geometry(80, level=3)
    x = np.random.default_rng(0).random((32, 80))
    sub = W.swt_subbands(W.pad_features(x, g), g)

    assert sub.shape[1] == 4, "chi(3) phai co dung n+1 = 4 subsequence"
    assert sub.shape[2] == g.n_padded, "SWT khong duoc downsampling"
    assert sub.shape == (32, 4, 80)


def test_subband_order_is_cD1_cD2_cD3_cA3():
    """So khop truc tiep voi pywt de chac thu tu khong bi dao."""
    g = W.compute_geometry(80, level=3)
    x = np.random.default_rng(1).random((4, 80))
    xp = W.pad_features(x, g)
    got = W.swt_subbands(xp, g)

    ref = pywt.swt(xp, "db4", level=3, axis=1)   # [(cA3,cD3),(cA2,cD2),(cA1,cD1)]
    assert np.allclose(got[:, 0], ref[-1][1])    # cD1
    assert np.allclose(got[:, 1], ref[-2][1])    # cD2
    assert np.allclose(got[:, 2], ref[0][1])     # cD3
    assert np.allclose(got[:, 3], ref[0][0])     # cA3
    assert g.subband_order == ("cD1", "cD2", "cD3", "cA3")


def test_wavedec_is_rejected():
    """pywt.wavedec co downsampling -> phai bi tu choi tu config."""
    cfg = {"wavelet": {"transform": "wavedec", "level": 3, "name": "db4",
                       "pad_mode": "reflect", "reshape": {}}}
    with pytest.raises(ValueError, match="swt"):
        W.geometry_from_config(80, cfg)


def test_transform_batch_shape_and_dtype():
    g = W.compute_geometry(80, level=3)
    x = np.random.default_rng(2).random((16, 80))
    img = W.transform_batch(x, g)
    assert img.shape == (16, 4, 9, 9)
    assert img.dtype == np.float32
    assert np.isfinite(img).all()


def test_transform_is_deterministic():
    g = W.compute_geometry(80, level=3)
    x = np.random.default_rng(3).random((8, 80))
    assert np.array_equal(W.transform_batch(x, g), W.transform_batch(x, g))


def test_image_padding_is_zero_and_counted():
    g = W.compute_geometry(80, level=3)
    assert g.n_image_pad == 81 - 80 == 1
    x = np.random.default_rng(4).random((2, 80))
    img = W.transform_batch(x, g).reshape(2, 4, -1)
    assert np.all(img[:, :, 80:] == 0.0), "o padding phai bang 0 de bo khi quy SHAP"


def test_geometry_hash_is_sensitive_to_feature_order():
    g = W.compute_geometry(3, level=1, min_side=8)
    a = W.geometry_hash(g, ["a", "b", "c"])
    b = W.geometry_hash(g, ["b", "a", "c"])
    assert a != b, "doi thu tu feature phai doi hash de fail-fast khi resume"
    assert a == W.geometry_hash(g, ["a", "b", "c"])


def test_feature_count_mismatch_fails_fast():
    g = W.compute_geometry(80, level=3)
    with pytest.raises(ValueError, match="khong khop"):
        W.pad_features(np.zeros((4, 79)), g)


def test_subband_energy_shape():
    g = W.compute_geometry(80, level=3)
    img = W.transform_batch(np.random.default_rng(5).random((10, 80)), g)
    e = W.subband_energy(img)
    assert e.shape == (10, 4)
    assert (e >= 0).all()
