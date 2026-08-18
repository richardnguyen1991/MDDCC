"""SWT DB4 level 3 doc theo truc feature - muc 2.A cua prompt.

Bai bao: "we don't need wavelet reconstruction, so we don't need to adopt
downsampling" => dung bien doi wavelet dung (pywt.swt), TUYET DOI khong dung
pywt.wavedec. Moi subband co do dai bang chuoi goc.

chi(3) = {xh(1), xh(2), xh(3), xl(3)} = (cD1, cD2, cD3, cA3) -> dung 4 subband.

Module nay khong co tham so hoc duoc va khong phu thuoc torch.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import pywt

SUBBAND_ORDER = ("cD1", "cD2", "cD3", "cA3")


def pooled_side(side: int, n_pools: int, *, ceil_mode: bool = True) -> int:
    """Canh feature map sau n_pools lan MaxPool2d(2x2)."""
    s = int(side)
    for _ in range(n_pools):
        s = math.ceil(s / 2) if ceil_mode else s // 2
    return s


@dataclass(frozen=True)
class WaveletGeometry:
    """Hinh hoc co dinh cua phep bien doi. Bat buoc khop giua train/eval/explain."""

    n_features: int      # F - so cot dac trung goc
    n_padded: int        # F_swt = ceil(F / 2^level) * 2^level
    side: int            # S - canh anh vuong
    level: int
    wavelet: str
    pad_mode: str        # padding chuoi truoc SWT
    image_pad_mode: str  # padding tu n_padded len side*side
    subband_order: tuple[str, ...] = SUBBAND_ORDER
    n_pool_levels: int = 3       # so lan MaxPool trong moi nhanh CNN (Table 3)
    pool_ceil_mode: bool = True
    min_final_map: int = 2       # canh feature map toi thieu truoc lop FC
    side_bumped_from: int | None = None   # S goc theo cong thuc, neu da phai nang

    @property
    def n_subbands(self) -> int:
        return len(self.subband_order)

    @property
    def final_map_side(self) -> int:
        """Canh feature map di vao lop FC."""
        return pooled_side(self.side, self.n_pool_levels,
                           ceil_mode=self.pool_ceil_mode)

    def flatten_dim(self, channels: int = 32) -> int:
        return channels * self.final_map_side ** 2

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (self.n_subbands, self.side, self.side)

    @property
    def n_image_pad(self) -> int:
        """So o padding o duoi cua anh - dung de bo khi quy SHAP ve feature goc."""
        return self.side * self.side - self.n_padded

    def valid_positions(self) -> np.ndarray:
        """Chi so phang [0, side*side) tuong ung vi tri feature that (chua padding SWT)."""
        return np.arange(self.n_features, dtype=np.int64)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["subband_order"] = list(self.subband_order)
        d["image_shape"] = list(self.image_shape)
        d["n_image_pad"] = self.n_image_pad
        d["n_swt_pad"] = self.n_padded - self.n_features
        d["final_map_side"] = self.final_map_side
        d["flatten_dim_at_32ch"] = self.flatten_dim(32)
        return d


def compute_geometry(
    n_features: int,
    *,
    level: int = 3,
    wavelet: str = "db4",
    pad_mode: str = "reflect",
    image_pad_mode: str = "constant",
    min_side: int = 8,
    force_side: int | None = None,
    n_pool_levels: int = 3,
    pool_ceil_mode: bool = True,
    min_final_map: int = 2,
) -> WaveletGeometry:
    """F -> F_swt -> S theo dung cong thuc muc 2.A, kem rang buoc chong sup do.

    Cong thuc S = max(8, ceil(sqrt(F_swt))) mot minh KHONG an toan: voi F <= 64
    thi S = 8 va sau 3 lan MaxPool 2x2 feature map con 1x1, tuc ca 4 nhanh CNN
    bi nen xuong 32 so truoc lop FC. So cot dac trung thay doi theo ket qua loai
    cot (hang so / trung lap / thieu qua nhieu), nen tinh huong nay co the xay ra
    ma khong ai de y. Vi vay S duoc NANG len den khi feature map cuoi >=
    min_final_map, va viec nang duoc ghi lai trong side_bumped_from.
    """
    if n_features < 1:
        raise ValueError(f"n_features phai >= 1, nhan {n_features}")
    if level < 1:
        raise ValueError(f"level phai >= 1, nhan {level}")

    stride = 2 ** level
    n_padded = math.ceil(n_features / stride) * stride

    if force_side is not None:
        side = int(force_side)
        if side * side < n_padded:
            raise ValueError(
                f"force_side={side} qua nho: {side}*{side} < F_swt={n_padded}."
            )
        final = pooled_side(side, n_pool_levels, ceil_mode=pool_ceil_mode)
        if final < min_final_map:
            raise ValueError(
                f"force_side={side} lam feature map cuoi con {final}x{final} "
                f"(< min_final_map={min_final_map}) -> chi con "
                f"{32 * final ** 2} chieu truoc lop FC. Tang force_side."
            )
        bumped_from = None
    else:
        base = max(min_side, math.ceil(math.sqrt(n_padded)))
        side = base
        while pooled_side(side, n_pool_levels, ceil_mode=pool_ceil_mode) < min_final_map:
            side += 1
        bumped_from = base if side != base else None

    if side * side < n_padded:
        raise ValueError(
            f"side={side} qua nho: side*side={side*side} < F_swt={n_padded}. "
            "Tang wavelet.reshape.force_side trong configs/mddcc.yaml."
        )
    return WaveletGeometry(
        n_features=n_features,
        n_padded=n_padded,
        side=side,
        level=level,
        wavelet=wavelet,
        pad_mode=pad_mode,
        image_pad_mode=image_pad_mode,
        n_pool_levels=n_pool_levels,
        pool_ceil_mode=pool_ceil_mode,
        min_final_map=min_final_map,
        side_bumped_from=bumped_from,
    )


def geometry_from_config(n_features: int, cfg: dict) -> WaveletGeometry:
    w = cfg["wavelet"]
    if w["transform"] != "swt":
        raise ValueError(
            f"wavelet.transform={w['transform']!r} - muc 2.A bat buoc 'swt'. "
            "pywt.wavedec co downsampling, khong dung duoc."
        )
    rs = w.get("reshape", {})
    m = cfg.get("model", {})
    return compute_geometry(
        n_features,
        level=w["level"],
        wavelet=w["name"],
        pad_mode=w["pad_mode"],
        image_pad_mode=rs.get("pad_mode", "constant"),
        min_side=rs.get("min_side", 8),
        force_side=rs.get("force_side"),
        n_pool_levels=len(m.get("branches", [None, None, None])),
        pool_ceil_mode=m.get("pool_ceil_mode", True),
        min_final_map=rs.get("min_final_map", 2),
    )


def pad_features(x: np.ndarray, geom: WaveletGeometry) -> np.ndarray:
    """[N, F] -> [N, F_swt] bang padding doi xung (mac dinh reflect)."""
    if x.ndim != 2:
        raise ValueError(f"can mang 2D [N, F], nhan shape {x.shape}")
    if x.shape[1] != geom.n_features:
        raise ValueError(
            f"so cot khong khop: nhan {x.shape[1]}, geometry mong doi {geom.n_features}"
        )
    pad = geom.n_padded - geom.n_features
    if pad == 0:
        return x
    return np.pad(x, ((0, 0), (0, pad)), mode=geom.pad_mode)


def swt_subbands(x_padded: np.ndarray, geom: WaveletGeometry) -> np.ndarray:
    """[N, F_swt] -> [N, 4, F_swt] theo thu tu cD1, cD2, cD3, cA3.

    pywt.swt tra ve [(cA_n, cD_n), ..., (cA_1, cD_1)] voi n = level.
    """
    coeffs = pywt.swt(x_padded, geom.wavelet, level=geom.level, axis=1,
                      trim_approx=False)
    if len(coeffs) != geom.level:
        raise RuntimeError(f"pywt.swt tra ve {len(coeffs)} muc, mong doi {geom.level}")

    # coeffs[-k] tuong ung muc k (k = 1..level)
    details = [coeffs[-k][1] for k in range(1, geom.level + 1)]  # cD1..cD_level
    approx = coeffs[0][0]                                        # cA_level
    stacked = np.stack([*details, approx], axis=1)

    if stacked.shape[1] != geom.n_subbands:
        raise RuntimeError(
            f"tao ra {stacked.shape[1]} subband, muc 2.A yeu cau {geom.n_subbands}"
        )
    if stacked.shape[2] != x_padded.shape[1]:
        raise RuntimeError(
            f"do dai subband {stacked.shape[2]} != do dai chuoi goc {x_padded.shape[1]} "
            "- dau hieu da downsampling, sai voi yeu cau SWT."
        )
    return stacked


def to_images(subbands: np.ndarray, geom: WaveletGeometry) -> np.ndarray:
    """[N, 4, F_swt] -> [N, 4, S, S] float32."""
    n, k, length = subbands.shape
    pad = geom.side * geom.side - length
    if pad < 0:
        raise ValueError(f"F_swt={length} > S*S={geom.side ** 2}")
    if pad:
        subbands = np.pad(subbands, ((0, 0), (0, 0), (0, pad)),
                          mode=geom.image_pad_mode)
    return np.ascontiguousarray(
        subbands.reshape(n, k, geom.side, geom.side), dtype=np.float32
    )


def transform_batch(x: np.ndarray, geom: WaveletGeometry) -> np.ndarray:
    """[N, F] float -> [N, 4, S, S] float32. Day la ham duy nhat train/eval nen goi."""
    return to_images(swt_subbands(pad_features(x, geom), geom), geom)


def geometry_hash(geom: WaveletGeometry, feature_order: Sequence[str]) -> str:
    """Hash cua (feature_order, wavelet, level, padding, side) - muc 2.A."""
    payload = {
        "feature_order": list(feature_order),
        "wavelet": geom.wavelet,
        "level": geom.level,
        "pad_mode": geom.pad_mode,
        "image_pad_mode": geom.image_pad_mode,
        "n_features": geom.n_features,
        "n_padded": geom.n_padded,
        "side": geom.side,
        "subband_order": list(geom.subband_order),
        "n_pool_levels": geom.n_pool_levels,
        "pool_ceil_mode": geom.pool_ceil_mode,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def subband_energy(images: np.ndarray) -> np.ndarray:
    """Nang luong trung binh moi subband: [N, 4, S, S] -> [N, 4]. Phuc vu hinh C10."""
    return (images.astype(np.float64) ** 2).sum(axis=(2, 3))
