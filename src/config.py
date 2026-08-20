"""Nap config goc + overlay cua bien the.

Hai run chay SONG SONG (full va capped10m) dung chung moi thu tru vai khoa. Neu
copy ca file config thanh hai ban thi chac chan se lech nhau sau vai lan sua, nen
bien the chi ghi de dung nhung khoa khac biet.

    cfg = load_config(Path("configs/mddcc.yaml"))                  # run full
    cfg = load_config(Path("configs/mddcc.yaml"), "capped10m")     # bien the
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

VARIANT_DIR = "variants"
DEFAULT_VARIANT = "full"


def deep_merge(base: dict, overlay: dict) -> dict:
    """Ghi de theo tung khoa, dict long nhau thi merge tiep chu khong thay ca cum."""
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def variant_path(config_path: Path, variant: str) -> Path:
    return Path(config_path).parent / VARIANT_DIR / f"{variant}.yaml"


def available_variants(config_path: Path) -> list[str]:
    d = Path(config_path).parent / VARIANT_DIR
    found = sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []
    return [DEFAULT_VARIANT] + found


def load_config(config_path: Path, variant: str | None = None) -> dict:
    """Nap config, ghi de bang overlay cua bien the neu co."""
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg.setdefault("experiment", {}).setdefault("variant", DEFAULT_VARIANT)

    if not variant or variant == DEFAULT_VARIANT:
        return cfg

    p = variant_path(config_path, variant)
    if not p.exists():
        raise SystemExit(
            f"Khong co bien the {variant!r} ({p}). "
            f"Cac bien the hien co: {available_variants(config_path)}")
    merged = deep_merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    merged["experiment"]["variant"] = variant
    return merged


def variant_of(cfg: dict) -> str:
    return (cfg.get("experiment", {}) or {}).get("variant", DEFAULT_VARIANT)


def kernel_slug(cfg: dict, owner: str) -> str:
    """Slug notebook Kaggle cua bien the: <owner>/<slug>."""
    slug = (cfg.get("kaggle", {}) or {}).get("kernel_slug", "mddcc")
    return f"{owner}/{slug}"


def run_id_key(cfg: dict) -> str:
    """Khoa S3 giu run_id hien tai. Moi bien the mot khoa rieng."""
    return (cfg.get("s3", {}) or {}).get("current_run_id_key",
                                         "current_run_id.json")
