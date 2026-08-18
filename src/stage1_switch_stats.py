"""Giai doan 1 cua bai bao: phat hien tho dua tren thong ke cong switch SDN.

NGOAI PHAM VI DANH GIA - muc 3.G cua prompt.
CIC-DDoS2019 la du lieu luong da trich xuat bang CIC-FlowMeter, KHONG chua so
lieu cong switch (N_FI, N_FO, N_Pi, N_PI, N_PO). Vi vay module nay duoc
implement dung cong thuc va co unit test tren du lieu tong hop, nhung KHONG
duoc chay tren CIC-DDoS2019 va KHONG duoc bao cao nhu da tai hien du.
Tai hien day du can moi truong Mininet/POX, nam ngoai pham vi luan van.

Ky hieu (Table 1 cua bai bao):
    N_FI  so luong network flow di VAO switch
    N_FO  so luong network flow di RA khoi switch
    N_Pi  so goi PacketIn switch chuyen len controller
    N_PI  so packet di VAO switch
    N_PO  so packet di RA khoi switch

Cong thuc:
    (1) R_Pi   = N_FI / N_Pi        ty le luong vao / PacketIn chuyen tiep
    (2) R_FI   = N_FO / N_FI        ty le chuyen tiep binh thuong
    (3) dN_P   = |N_PI - N_PO|      chenh lech so packet vao/ra

Quy tac nguong (trich nguyen van bai bao):
    "the threshold refers to the mean and standard deviation of the features
     R_FI, R_Pi and dN_P calculated after sampling 10,000 sets when only normal
     traffic exists in the network. Then, following the 'three-sigma (3sigma)'
     rule, the thresholds for R_FI and R_Pi are set to the mean minus three
     times the standard deviation, while the threshold for dN_P is set to the
     mean plus three times the standard deviation."

    => R_Pi < mu - 3*sigma   VA   R_FI < mu - 3*sigma   VA   dN_P > mu + 3*sigma
    Bai bao yeu cau CA BA vuot nguong ("when all feature values exceed their
    thresholds") moi ket luan co tan cong -> phep AND, khong phai OR.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

BASELINE_SAMPLE_SETS = 10_000   # bai bao: sampling 10,000 sets
SIGMA_MULTIPLIER = 3.0


# ------------------------------------------------------------- cong thuc
def r_pi(n_fi: np.ndarray, n_pi: np.ndarray) -> np.ndarray:
    """Cong thuc (1): R_Pi = N_FI / N_Pi.

    N_Pi = 0 nghia la switch khong hoi controller lan nao - khong co dau hieu
    tan cong theo chi so nay, tra ve +inf (khong bao gio duoi nguong).
    """
    n_fi = np.asarray(n_fi, dtype=np.float64)
    n_pi = np.asarray(n_pi, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(n_pi > 0, n_fi / np.where(n_pi > 0, n_pi, 1.0), np.inf)
    return out


def r_fi(n_fo: np.ndarray, n_fi: np.ndarray) -> np.ndarray:
    """Cong thuc (2): R_FI = N_FO / N_FI."""
    n_fo = np.asarray(n_fo, dtype=np.float64)
    n_fi = np.asarray(n_fi, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(n_fi > 0, n_fo / np.where(n_fi > 0, n_fi, 1.0), np.inf)
    return out


def delta_n_p(n_pi_in: np.ndarray, n_po: np.ndarray) -> np.ndarray:
    """Cong thuc (3): dN_P = |N_PI - N_PO|."""
    return np.abs(np.asarray(n_pi_in, dtype=np.float64)
                  - np.asarray(n_po, dtype=np.float64))


def compute_features(stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """stats can co N_FI, N_FO, N_Pi, N_PI, N_PO -> tra ve R_Pi, R_FI, dN_P."""
    need = {"N_FI", "N_FO", "N_Pi", "N_PI", "N_PO"}
    missing = need - set(stats)
    if missing:
        raise KeyError(f"Thieu thong ke cong switch: {sorted(missing)}")
    return {
        "R_Pi": r_pi(stats["N_FI"], stats["N_Pi"]),
        "R_FI": r_fi(stats["N_FO"], stats["N_FI"]),
        "dN_P": delta_n_p(stats["N_PI"], stats["N_PO"]),
    }


# --------------------------------------------------------------- nguong
@dataclass(frozen=True)
class SigmaThresholds:
    """Nguong 3-sigma hoc tu luu luong BINH THUONG."""

    r_pi_mean: float
    r_pi_std: float
    r_fi_mean: float
    r_fi_std: float
    dnp_mean: float
    dnp_std: float
    k: float = SIGMA_MULTIPLIER
    n_baseline_sets: int = 0

    # R_Pi va R_FI GIAM khi bi tan cong -> nguong duoi
    @property
    def r_pi_threshold(self) -> float:
        return self.r_pi_mean - self.k * self.r_pi_std

    @property
    def r_fi_threshold(self) -> float:
        return self.r_fi_mean - self.k * self.r_fi_std

    # dN_P TANG khi bi tan cong -> nguong tren
    @property
    def dnp_threshold(self) -> float:
        return self.dnp_mean + self.k * self.dnp_std

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({"r_pi_threshold": self.r_pi_threshold,
                  "r_fi_threshold": self.r_fi_threshold,
                  "dnp_threshold": self.dnp_threshold})
        return d


def fit_thresholds(normal_features: dict[str, np.ndarray], *,
                   k: float = SIGMA_MULTIPLIER,
                   require_min_sets: int = BASELINE_SAMPLE_SETS,
                   strict: bool = False) -> SigmaThresholds:
    """Hoc mu va sigma tu CHI luu luong binh thuong (bai bao: 10.000 bo mau)."""
    rp = np.asarray(normal_features["R_Pi"], dtype=np.float64)
    rf = np.asarray(normal_features["R_FI"], dtype=np.float64)
    dp = np.asarray(normal_features["dN_P"], dtype=np.float64)

    n = rp.size
    if not (rf.size == dp.size == n):
        raise ValueError("Ba dac trung phai cung so mau")
    if n < 2:
        raise ValueError("Can it nhat 2 mau de tinh do lech chuan")
    if strict and n < require_min_sets:
        raise ValueError(f"Bai bao dung {require_min_sets} bo mau, chi co {n}")

    finite = np.isfinite(rp) & np.isfinite(rf) & np.isfinite(dp)
    if not finite.any():
        raise ValueError("Khong co mau huu han nao de hoc nguong")
    rp, rf, dp = rp[finite], rf[finite], dp[finite]

    return SigmaThresholds(
        float(rp.mean()), float(rp.std(ddof=0)),
        float(rf.mean()), float(rf.std(ddof=0)),
        float(dp.mean()), float(dp.std(ddof=0)),
        k=k, n_baseline_sets=int(finite.sum()),
    )


def detect(features: dict[str, np.ndarray], th: SigmaThresholds) -> np.ndarray:
    """Tra ve mask bool: True = nghi ngo co DDoS tren switch do.

    Bai bao: "when all feature values exceed their thresholds" -> AND ca ba.
    """
    below_r_pi = np.asarray(features["R_Pi"], dtype=np.float64) < th.r_pi_threshold
    below_r_fi = np.asarray(features["R_FI"], dtype=np.float64) < th.r_fi_threshold
    above_dnp = np.asarray(features["dN_P"], dtype=np.float64) > th.dnp_threshold
    return below_r_pi & below_r_fi & above_dnp


def detect_from_stats(stats: dict[str, np.ndarray], th: SigmaThresholds) -> np.ndarray:
    return detect(compute_features(stats), th)


def explain(features: dict[str, np.ndarray], th: SigmaThresholds) -> dict:
    """Tach rieng dong gop tung chi so - phuc vu chan doan, khong dung de bao cao metric."""
    return {
        "R_Pi_below": (np.asarray(features["R_Pi"]) < th.r_pi_threshold).mean(),
        "R_FI_below": (np.asarray(features["R_FI"]) < th.r_fi_threshold).mean(),
        "dN_P_above": (np.asarray(features["dN_P"]) > th.dnp_threshold).mean(),
        "all_three": detect(features, th).mean(),
        "thresholds": th.to_dict(),
    }


# ------------------------------------------------------------ pham vi
SCOPE_NOTE = (
    "Giai doan 1 (thong ke cong switch) va module giam thieu dua tren do thi "
    "KHONG duoc danh gia trong luan van nay vi CIC-DDoS2019 khong chua so lieu "
    "cong switch SDN va khong co moi truong Mininet/POX. Module nay chi "
    "implement cong thuc (1)(2)(3) + quy tac 3-sigma kem unit test tren du lieu "
    "tong hop."
)


def assert_not_applicable_to_cicddos2019() -> None:
    """Chan moi y dinh chay giai doan 1 tren CIC-DDoS2019."""
    raise NotImplementedError(SCOPE_NOTE)
