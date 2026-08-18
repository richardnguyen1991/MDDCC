#!/usr/bin/env python3
"""Buoc 2a - Discovery Kaggle Dataset cho pipeline MDDCC.

CHI DOC METADATA + cot nhan. Khong nap toan bo file vao RAM.
Khong ghi bat cu thu gi vao /kaggle/input.

Chay tren Kaggle:
    python scripts/discover_dataset.py --config configs/mddcc.yaml
Chay tren thu muc parquet bat ky:
    python scripts/discover_dataset.py --input-dir /duong/dan/parquet

Ket qua: in ra man hinh + ghi data_profile.json (mac dinh /kaggle/working).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

# Thong lượng da do thuc te tren CPU 4 luong (bench.py), dung de uoc luong ETA.
# Doi lai bang --throughput neu do duoc so khac tren Kaggle.
DEFAULT_TRAIN_THROUGHPUT = 2842.0  # samples/s, fwd+bwd+SGD, batch_size=4096
DEFAULT_EVAL_THROUGHPUT = 7825.0   # samples/s, forward + no_grad
DEFAULT_SWT_THROUGHPUT = 96796.0   # rows/s, db4 level 3, axis=1
KAGGLE_SESSION_SECONDS = 40800     # 11h20m

# Cot khong bao gio duoc dung lam feature (dinh danh / ro ri / provenance).
NON_FEATURE_COLUMNS = {
    "flow id", "source ip", "src ip", "destination ip", "dst ip",
    "timestamp", "unnamed: 0", "simillarhttp",
    "__capture_day", "__source_file_id", "__source_row_id",
}
LABEL_CANDIDATES = ["Label", " Label", "Class", "label"]


def norm(name: str) -> str:
    return name.strip().lower()


def load_config(path: Path) -> dict:
    import yaml
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_label_column(columns: list[str]) -> tuple[str | None, str]:
    """Tra ve (ten cot nhan thuc te, cach xac dinh). Fail-fast neu khong thay."""
    by_norm = {norm(c): c for c in columns}
    for cand in LABEL_CANDIDATES:
        if norm(cand) in by_norm:
            return by_norm[norm(cand)], f"column_candidate:{cand}"
    return None, "not_found"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data_profile.json"))
    ap.add_argument("--throughput", type=float, default=DEFAULT_TRAIN_THROUGHPUT,
                    help="mau/s khi train; do lai tren Kaggle roi truyen vao")
    ap.add_argument("--eval-throughput", type=float, default=DEFAULT_EVAL_THROUGHPUT)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--count-labels", action="store_true",
                    help="Doc cot nhan cua MOI file de dem phan bo lop (ton I/O, nen bat).")
    args = ap.parse_args()

    if args.config and args.config.exists():
        cfg = load_config(args.config)
        input_dir = Path(args.input_dir or cfg["data"]["kaggle_input_dir"])
        glob_pat = cfg["data"]["parquet_glob"]
        level = cfg["wavelet"]["level"]
        min_side = cfg["wavelet"]["reshape"]["min_side"]
        force_side = cfg["wavelet"]["reshape"]["force_side"]
        batch_size = cfg["train"]["batch_size"]
        test_frac = cfg["split"]["test_size"]
        val_frac = cfg["split"]["val_size_within_trainval"]
    else:
        input_dir = Path(args.input_dir or "/kaggle/input/cicddos2019-parquet")
        glob_pat, level, min_side, force_side, batch_size = "**/*.parquet", 3, 8, None, 4096
        test_frac, val_frac = 0.30, 0.15

    print("=" * 78)
    print("MDDCC - BUOC 2a: DISCOVERY KAGGLE DATASET")
    print("=" * 78)
    print(f"input_dir = {input_dir}")

    files = sorted(input_dir.glob(glob_pat)) if input_dir.exists() else []

    # Ten thu muc mount tren Kaggle theo SLUG cua dataset, khong theo tieu de
    # hien thi, nen co the khac duong dan trong config. Tu do lai thay vi bat
    # nguoi dung doan - va in ro da chon gi de khong bao gio chon nham am tham.
    if not files:
        print(f"\nKhong thay file parquet tai {input_dir} - dang tu do /kaggle/input ...")
        root = Path("/kaggle/input")
        if not root.exists():
            print(f"FAIL-FAST: khong co {input_dir} va cung khong co {root}.")
            print("Tren Kaggle: kiem tra kernel-metadata.json co 'dataset_sources'.")
            return 2

        entries = sorted(p.name for p in root.iterdir())
        print(f"  Co trong /kaggle/input: {entries}")
        found = sorted(root.glob("**/*.parquet"))
        if not found:
            print("FAIL-FAST: khong co file .parquet nao duoi /kaggle/input.")
            print("Kiem tra dataset da duoc Add Input vao notebook chua.")
            return 2

        base = found[0].parent
        while not all(str(f).startswith(str(base) + os.sep) for f in found):
            if base.parent == base:
                break
            base = base.parent
        input_dir = base
        files = sorted(input_dir.glob(glob_pat))
        print(f"  Tu dong chon input_dir = {input_dir}  ({len(files)} file parquet)")
        print("  Neu sai, chay lai voi --input-dir <duong-dan-dung>.")

    # ---------------------------------------------------------------- files
    print(f"\n[1] FILE PARQUET: {len(files)} file")
    print(f"{'file':<52}{'rows':>14}{'MB':>10}{'rg':>5}")
    print("-" * 81)

    total_rows = 0
    total_bytes = 0
    schemas: dict[str, list[str]] = {}
    per_file = []
    for fp in files:
        pf = pq.ParquetFile(fp)          # metadata only
        md = pf.metadata
        size = fp.stat().st_size
        cols = [f.name for f in pf.schema_arrow]
        key = "|".join(cols)
        schemas.setdefault(key, cols)
        total_rows += md.num_rows
        total_bytes += size
        rel = str(fp.relative_to(input_dir))
        per_file.append({"file": rel, "rows": md.num_rows,
                         "bytes": size, "row_groups": md.num_row_groups,
                         "schema_key": key[:16]})
        print(f"{rel[:51]:<52}{md.num_rows:>14,}{size/1e6:>10,.1f}{md.num_row_groups:>5}")

    print("-" * 81)
    print(f"{'TONG':<52}{total_rows:>14,}{total_bytes/1e6:>10,.1f}")

    if len(schemas) > 1:
        print(f"\nCANH BAO: {len(schemas)} schema khac nhau giua cac file -> can hop nhat truoc khi train.")

    # --------------------------------------------------------------- schema
    columns = next(iter(schemas.values()))
    arrow_schema = pq.ParquetFile(files[0]).schema_arrow
    print(f"\n[2] SCHEMA: {len(columns)} cot")
    label_col, label_how = find_label_column(columns)
    feature_cols, dropped = [], []
    for f in arrow_schema:
        if label_col and f.name == label_col:
            continue
        if norm(f.name) in NON_FEATURE_COLUMNS:
            dropped.append((f.name, str(f.type), "identifier/leakage/provenance"))
        else:
            feature_cols.append((f.name, str(f.type)))

    print(f"\n  Cot nhan  : {label_col!r}  ({label_how})")
    if label_col is None:
        print("  FAIL-FAST: khong xac dinh duoc cot nhan. Xem muc 3.A cua prompt.")
        return 3

    print(f"\n  Cot bi loai ({len(dropped)}):")
    for n, t, why in dropped:
        print(f"    - {n!r:<28} {t:<12} {why}")

    print(f"\n  Cot dac trung ({len(feature_cols)}):")
    for i, (n, t) in enumerate(feature_cols):
        print(f"    {i:>3}. {n!r:<44} {t}")

    non_numeric = [(n, t) for n, t in feature_cols
                   if not any(k in t for k in ("int", "float", "double", "decimal"))]
    if non_numeric:
        print(f"\n  CANH BAO: {len(non_numeric)} cot phi so (can one-hot hoac loai):")
        for n, t in non_numeric:
            print(f"    - {n!r} {t}")

    # ---------------------------------------------------------------- nhan
    label_counts: dict[str, int] = {}
    if args.count_labels:
        print("\n[3] PHAN BO LOP (doc rieng cot nhan)")
        import pyarrow.compute as pc

        counter: Counter[str] = Counter()
        for fp in files:
            # value_counts tren cot da dictionary-encode: khong tao list Python
            # cho 70.4M hang (se ton ~4-5 GB RAM va rat cham).
            pf = pq.ParquetFile(fp)
            for i in range(pf.metadata.num_row_groups):
                col = pf.read_row_group(i, columns=[label_col]).column(0).combine_chunks()
                for pair in pc.value_counts(col):
                    counter[str(pair["values"]).strip()] += pair["counts"].as_py()
            print(f"    ... {fp.name}")
        label_counts = dict(counter.most_common())
        print(f"\n  {'lop':<20}{'so mau':>14}{'ty le %':>10}")
        print("  " + "-" * 44)
        for k, v in label_counts.items():
            print(f"  {k:<20}{v:>14,}{100*v/total_rows:>10.4f}")
        print(f"\n  num_class = {len(label_counts)}")
        rarest = min(label_counts.items(), key=lambda kv: kv[1])
        print(f"  Lop hiem nhat: {rarest[0]} = {rarest[1]:,} "
              f"({100*rarest[1]/total_rows:.5f}%) -> mat can bang 1:{total_rows//max(rarest[1],1):,}")
    else:
        print("\n[3] PHAN BO LOP: bo qua (them --count-labels de dem that)")

    # ------------------------------------------------------ hinh hoc wavelet
    F = len(feature_cols)
    F_swt = math.ceil(F / 2 ** level) * (2 ** level)
    side = force_side or max(min_side, math.ceil(math.sqrt(F_swt)))
    print(f"\n[4] HINH HOC WAVELET (db4, level={level}, SWT)")
    print(f"  F={F} -> F_swt={F_swt} (pad reflect) -> S={side} (anh {side}x{side}={side*side})")
    print(f"  4 subband cD1,cD2,cD3,cA3, moi subband dai {F_swt} = do dai chuoi goc")

    # ---------------------------------------------------------- kha thi CPU
    # Mot epoch chi duyet tap TRAIN (fwd+bwd) roi danh gia tap VAL (forward).
    # Khong duoc lay total_rows lam co so - se uoc luong thua rat nhieu.
    train_rows = total_rows * (1 - test_frac) * (1 - val_frac)
    val_rows = total_rows * (1 - test_frac) * val_frac
    sec_train = train_rows / args.throughput
    sec_val = val_rows / args.eval_throughput
    sec_epoch = sec_train + sec_val
    sec_total = sec_epoch * args.epochs
    cache_gb = total_rows * 4 * side * side * 4 / 1e9
    raw_gb = total_rows * F * 4 / 1e9
    swt_hours = train_rows / DEFAULT_SWT_THROUGHPUT / 3600

    print(f"\n[5] UOC LUONG KHA THI (train {args.throughput:,.0f} mau/s, "
          f"eval {args.eval_throughput:,.0f} mau/s, bs={batch_size})")
    print(f"  Split                   : train {train_rows:,.0f} | val {val_rows:,.0f} "
          f"| test {total_rows*test_frac:,.0f}")
    print(f"  1 epoch                 : {sec_epoch/3600:>10,.2f} h "
          f"(train {sec_train/3600:.2f} h + val {sec_val/3600:.2f} h)")
    print(f"  Step moi epoch          : {math.ceil(train_rows/batch_size):>10,}")
    print(f"  {args.epochs} epoch              : {sec_total/3600:>10,.0f} h = {sec_total/86400:,.0f} ngay")
    print(f"  So session Kaggle 11h20m: {math.ceil(sec_total/KAGGLE_SESSION_SECONDS):>10,}")
    print(f"  Cache SWT float32       : {cache_gb:>10,.1f} GB  <- neu cache subband")
    print(f"  Cache feature da scale  : {raw_gb:>10,.1f} GB  <- phuong an dang dung")
    print(f"  SWT on-the-fly          : {swt_hours:>10,.2f} h/epoch "
          f"({100*swt_hours*3600/sec_epoch:.1f}% chi phi epoch)")
    if cache_gb > 50 or sec_total / KAGGLE_SESSION_SECONDS > 10:
        print("\n  ==> VUOT NGUON LUC KAGGLE. Dung lai, bao cao cho chu nhiem de quyet dinh")
        print("      (muc 3.A: 'khong tu y cat bot du lieu').")

    # ---------------------------------------------------------------- xuat
    profile = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "file_count": len(files),
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "distinct_schemas": len(schemas),
        "files": per_file,
        "label_column": label_col,
        "label_detection": label_how,
        "label_counts": label_counts,
        "num_class": len(label_counts) or None,
        "dropped_columns": [{"name": n, "dtype": t, "reason": w} for n, t, w in dropped],
        "feature_columns": [{"name": n, "dtype": t} for n, t in feature_cols],
        "n_features": F,
        "non_numeric_features": [{"name": n, "dtype": t} for n, t in non_numeric],
        "wavelet_geometry": {"level": level, "F": F, "F_swt": F_swt, "side": side,
                             "image": f"4x{side}x{side}"},
        "feasibility": {
            "train_throughput_samples_per_s": args.throughput,
            "seconds_per_epoch": sec_epoch,
            "seconds_total": sec_total,
            "eval_throughput_samples_per_s": args.eval_throughput,
            "train_rows": int(train_rows),
            "val_rows": int(val_rows),
            "steps_per_epoch": math.ceil(train_rows / batch_size),
            "kaggle_sessions_needed": math.ceil(sec_total / KAGGLE_SESSION_SECONDS),
            "swt_cache_gb": cache_gb,
            "raw_feature_cache_gb": raw_gb,
        },
    }
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[6] Da ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
