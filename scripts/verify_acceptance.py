#!/usr/bin/env python3
"""Kiem tra tu dong 12 tieu chi nghiem thu muc 11.A.

    python scripts/verify_acceptance.py --run-dir s3://bucket/prefix/<run_id>
    python scripts/verify_acceptance.py --run-dir ./_localstore/<run_id>
    python scripts/verify_acceptance.py --run-dir ... --with-data \\
        --config configs/mddcc.yaml     # thêm tiêu chí 6 (tái lập y_prob)

Thoat != 0 neu co tieu chi FAIL. Tieu chi khong du du lieu de ket luan thi bao
SKIP chu khong bao PASS - de khong ai tuong da nghiem thu xong.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Check:
    n: str
    title: str
    status: str = SKIP
    detail: str = ""
    notes: list[str] = field(default_factory=list)

    def ok(self, detail=""):
        self.status, self.detail = PASS, detail
        return self

    def bad(self, detail):
        self.status, self.detail = FAIL, detail
        return self

    def skip(self, detail):
        self.status, self.detail = SKIP, detail
        return self


class Store:
    def __init__(self, run_dir: str):
        if run_dir.startswith("s3://"):
            from src.s3io import S3Store
            rest = run_dir[5:].rstrip("/")
            bucket, _, key = rest.partition("/")
            prefix, _, rid = key.rpartition("/")
            self.store, self.run_id = S3Store(bucket, prefix), rid
        else:
            from src.s3io import LocalStore
            p = Path(run_dir).resolve()
            self.store, self.run_id = LocalStore(p.parent), p.name

    def json(self, *parts):
        try:
            return self.store.get_json(f"{self.run_id}/" + "/".join(parts))
        except Exception:                       # noqa: BLE001
            return None

    def npy(self, *parts):
        try:
            raw = self.store.get_bytes(f"{self.run_id}/" + "/".join(parts))
        except Exception:                       # noqa: BLE001
            return None
        return np.load(io.BytesIO(raw), allow_pickle=False)

    def exists(self, *parts):
        return self.store.exists(f"{self.run_id}/" + "/".join(parts))

    def keys(self, prefix=""):
        return self.store.list_keys(f"{self.run_id}/{prefix}")


# ============================================================== tung tieu chi
def c1_resume_mid_epoch(s: Store) -> Check:
    c = Check("1", "Ngat giua epoch roi resume dung step ke tiep")
    hist = s.json("metrics", "history.json")
    if not hist:
        return c.skip("chua co history.json")
    partial = [r for r in hist if r.get("train_metrics_partial")]
    if not partial:
        return c.skip(f"{len(hist)} epoch, chua epoch nao bi ngat giua chung "
                      "-> chua kiem chung duoc resume giua epoch")
    bad = [r for r in partial if not r.get("resumed_after_batches")]
    if bad:
        return c.bad(f"epoch {[r['epoch'] for r in bad]} bao partial nhung "
                     "thieu resumed_after_batches")
    return c.ok(f"{len(partial)} epoch resume giua chung, vi du epoch "
                f"{partial[0]['epoch']} tiep tu batch {partial[0]['resumed_after_batches']}")


def c2_history_continuous(s: Store) -> Check:
    c = Check("2", "history.json lien tuc 1..N, khong thieu khong lap")
    hist = s.json("metrics", "history.json")
    if not hist:
        return c.skip("chua co history.json")
    epochs = [r["epoch"] for r in hist]
    expected = list(range(1, len(epochs) + 1))
    if epochs != expected:
        missing = sorted(set(expected) - set(epochs))
        dup = sorted({e for e in epochs if epochs.count(e) > 1})
        return c.bad(f"thieu {missing}, lap {dup}")
    sessions = []
    for r in hist:
        if r.get("session_id") not in sessions:
            sessions.append(r.get("session_id"))
    return c.ok(f"epoch 1..{len(epochs)} lien tuc, {len(sessions)} session "
                f"-> {len(sessions) - 1} vach Resume")


def c3_orchestration_log(s: Store) -> Check:
    c = Check("3", "Log restart do GitHub Actions co tren S3")
    hist = s.json("orchestration", "restart_history.json")
    state = s.json("checkpoints", "training_state.json")
    if hist is None:
        return c.skip("chua co orchestration/restart_history.json "
                      "-> chua chay qua GitHub Actions")
    epochs = [r.get("current_epoch_before") for r in hist]
    stalled = len(epochs) >= 3 and len(set(epochs[-3:])) == 1
    done = bool(state and state.get("is_complete"))
    if stalled and not done:
        return c.bad(f"3 lan push cuoi dung yen o epoch {epochs[-1]}")
    return c.ok(f"{len(hist)} lan restart, epoch truoc moi lan: {epochs[:8]}"
                f"{'...' if len(epochs) > 8 else ''}")


def c4_no_ram_overflow(s: Store) -> Check:
    c = Check("4", "Khong tran RAM tren Kaggle CPU")
    prof = s.json("config", "data_profile.json")
    hist = s.json("metrics", "history.json")
    if not prof or not hist:
        return c.skip("thieu data_profile.json hoac history.json")
    peak = max((r.get("peak_rss_mb") or 0) for r in hist)
    cache_gb = prof.get("cache_bytes", 0) / 1e9
    if peak <= 0:
        return c.skip("history khong ghi peak_rss_mb")
    if peak > 28000:
        return c.bad(f"peak RSS {peak:,.0f} MB - sat gioi han 30 GB cua Kaggle")
    return c.ok(f"peak RSS {peak:,.0f} MB; cache tren dia {cache_gb:.1f} GB "
                "(khong nap vao RAM)")


def c5_no_split_leakage(s: Store) -> Check:
    c = Check("5", "Assert chong ro ri split pass")
    man = s.json("config", "sample_manifest.json")
    if not man:
        return c.skip("chua co sample_manifest.json")
    if not man.get("disjoint") or not man.get("covers_all_rows"):
        return c.bad(f"disjoint={man.get('disjoint')} "
                     f"covers_all_rows={man.get('covers_all_rows')}")
    over = man.get("overlaps", {})
    if any(v for v in over.values()):
        return c.bad(f"overlaps={over}")
    fr = man.get("fractions", {})
    return c.ok(f"train/val/test = {fr.get('train', 0):.3f}/{fr.get('val', 0):.3f}/"
                f"{fr.get('test', 0):.3f}, moi cap overlap = 0")


def c6_final_model_reproduces_yprob(s: Store, cfg_path: Path | None) -> Check:
    c = Check("6", "final_model_epoch_100.pt tai lap dung y_prob.npy")
    state = s.json("checkpoints", "training_state.json") or {}
    name = "final_model_epoch_100.pt"
    if not s.exists("checkpoints", name):
        return c.skip("chua co final_model_epoch_100.pt")
    y_prob = s.npy("raw", "y_prob.npy")
    if y_prob is None:
        return c.skip("chua co raw/y_prob.npy")
    if cfg_path is None:
        return c.skip("can --with-data --config de nap lai model va du doan lai")

    import torch
    import yaml

    from src import data as D
    from src.evaluate import predict_in_chunks
    from src.model import build_model

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    prep = D.prepare_dataset(cfg, Path(cfg["data"]["cache_dir"]).parent / "verify")
    model = build_model(cfg, side=prep.geom.side, num_classes=prep.labels.num_classes)

    local = Path(cfg["data"]["cache_dir"]).parent / name
    s.store.get_file(f"{s.run_id}/checkpoints/{name}", local)
    ckpt = torch.load(local, map_location="cpu", weights_only=False)
    if ckpt.get("epoch") != state.get("total_epochs"):
        return c.bad(f"checkpoint o epoch {ckpt.get('epoch')} chu khong phai "
                     f"{state.get('total_epochs')}")
    model.load_state_dict(ckpt["model_state_dict"])

    ds = D.MDDCCDataset(prep.cache_path, prep.splits.test, prep.labels.codes, prep.geom)
    _, again = predict_in_chunks(model, ds, prep.splits.test,
                                batch_size=int(cfg["train"]["batch_size"]),
                                num_classes=prep.labels.num_classes)
    if again.shape != y_prob.shape:
        return c.bad(f"shape lech: {again.shape} vs {y_prob.shape}")
    err = float(np.abs(again - y_prob).max())
    if err > 1e-5:
        return c.bad(f"sai lech toi da {err:.2e} > 1e-5")
    return c.ok(f"epoch {ckpt['epoch']}, sai lech toi da {err:.2e} tren "
                f"{y_prob.shape[0]:,} mau")


def c7_report_regenerates(s: Store) -> Check:
    c = Check("7", "make_report.py sinh lai C1-C14 chi tu artifact")
    need = {"C1": ("metrics", "history.json"),
            "C3": ("raw", "y_true.npy"),
            "C5": ("raw", "y_prob.npy"),
            "C8": ("config", "sample_manifest.json"),
            "C10": ("explainability", "wavelet_subband_energy.json"),
            "C11": ("explainability", "branch_ablation.json"),
            "C12": ("explainability", "permutation_importance.json"),
            "C14": ("metrics", "test_metrics.json")}
    missing = [k for k, parts in need.items() if not s.exists(*parts)]
    if missing:
        return c.skip(f"thieu dau vao cho {missing}")
    figs = [k for k in s.keys("figures/") if k.endswith(".png")]
    detail = f"du dau vao cho toan bo C1-C14; {len(figs)} PNG da tren store"
    return c.ok(detail) if figs else Check(
        "7", c.title, PASS, detail + " (chua upload hinh - chay make_report --upload)")


def c8_architecture(s: Store) -> Check:
    c = Check("8", "Kien truc dung bai bao")
    rc = s.json("config", "run_config.json")
    if not rc:
        return c.skip("chua co run_config.json")
    m, g = rc.get("model", {}), rc.get("wavelet_geometry", {})
    problems = []
    if g.get("subband_order") != ["cD1", "cD2", "cD3", "cA3"]:
        problems.append(f"subband {g.get('subband_order')}")
    if m.get("n_branches") != 4:
        problems.append(f"{m.get('n_branches')} nhanh")
    if [b.get("conv_out") for b in m.get("branch_specs", [])] != [32, 64, 32]:
        problems.append("conv khong phai 32/64/32")
    if [b.get("dropout") for b in m.get("branch_specs", [])] != [0.2, 0.3, 0.2]:
        problems.append("dropout khong phai 0.2/0.3/0.2")
    if m.get("compose") != "sum":
        problems.append(f"compose={m.get('compose')}")
    if m.get("output_activation") != "softmax":
        problems.append("khong softmax")
    if rc.get("loss") != "mse" or rc.get("swt") is not True:
        problems.append(f"loss={rc.get('loss')} swt={rc.get('swt')}")
    if g.get("level") != 3 or g.get("wavelet") != "db4":
        problems.append(f"{g.get('wavelet')} level {g.get('level')}")
    if problems:
        return c.bad("; ".join(problems))
    return c.ok(f"4 nhanh 32/64/32 dropout .2/.3/.2 -> compose sum -> FC "
                f"{m.get('flatten_dim')} -> softmax; MSE + sigma(w); "
                f"db4 level 3 SWT, anh {g.get('image_shape')}")


def c9_run_config_keys(s: Store) -> Check:
    c = Check("9", "run_config.json ghi du khoa nghiem thu")
    rc = s.json("config", "run_config.json")
    if not rc:
        return c.skip("chua co run_config.json")
    want = {"epochs": 100, "early_stopping": False, "batch_size": 4096,
            "learning_rate": 0.01, "device": "cpu", "feature_selection": "none",
            "imbalance_handling": "none", "use_all_features": True,
            "experiment_role": "paper_reproduction_mddcc"}
    wrong = {k: (rc.get(k), v) for k, v in want.items() if rc.get(k) != v}
    if wrong:
        return c.bad("; ".join(f"{k}={g!r} (mong doi {e!r})"
                              for k, (g, e) in wrong.items()))
    if not rc.get("deviations_from_paper"):
        return c.bad("thieu muc deviations_from_paper")
    return c.ok(f"du 9 khoa; {len(rc['deviations_from_paper'])} sai khac duoc ghi")


def c10_paper_comparison(s: Store) -> Check:
    c = Check("10", "paper_comparison doi chieu duoc voi Table 9")
    rows = s.json("metrics", "paper_comparison.json")
    if not rows:
        return c.skip("chua co paper_comparison.json")
    need = {"Accuracy", "Precision", "Recall", "F1", "FPR"}
    got = {r["metric"] for r in rows}
    if not need <= got:
        return c.bad(f"thieu {sorted(need - got)}")
    if any("delta" not in r or "note" not in r for r in rows):
        return c.bad("thieu cot delta hoac note")
    acc = next(r for r in rows if r["metric"] == "Accuracy")
    return c.ok(f"du 5 chi so; Accuracy ta {acc['ours_binary']:.4f} vs bai bao "
                f"{acc['paper_table9']:.4f} (delta {acc['delta']:+.4f})")


def c11_kernel_metadata() -> Check:
    c = Check("11", "kernel-metadata.json khai bao dataset_sources")
    p = REPO / "kernel" / "kernel-metadata.json"
    if not p.exists():
        return c.bad("khong co kernel/kernel-metadata.json")
    m = json.loads(p.read_text(encoding="utf-8"))
    if not m.get("dataset_sources"):
        return c.bad("thieu dataset_sources -> session do Actions push se khong "
                     "co du lieu")
    if m.get("enable_gpu") or m.get("accelerator") != "none":
        return c.bad("phai chay CPU")
    if not m.get("enable_internet"):
        return c.bad("can internet de pip install va upload S3")
    return c.ok(f"dataset_sources={m['dataset_sources']}, CPU, internet on")


def c12_data_profile(s: Store) -> Check:
    c = Check("12", "data_profile.json ghi dung so hang/cot/phan bo lop")
    prof = s.json("config", "data_profile.json")
    lm = s.json("config", "label_mapping.json")
    if not prof or not lm:
        return c.skip("thieu data_profile.json hoac label_mapping.json")
    total = prof.get("total_rows", 0)
    files_sum = sum(f.get("rows", 0) for f in prof.get("files", []))
    if files_sum != total:
        return c.bad(f"tong so hang tung file {files_sum:,} != total_rows {total:,}")
    counts_sum = sum(lm.get("counts", {}).values())
    if counts_sum != total:
        return c.bad(f"tong phan bo lop {counts_sum:,} != total_rows {total:,}")
    splits_sum = sum(prof.get("split_sizes", {}).values())
    if splits_sum != total:
        return c.bad(f"tong split {splits_sum:,} != total_rows {total:,}")
    if prof.get("cache_build_seconds") is None:
        return c.bad("thieu cache_build_seconds")
    return c.ok(f"{total:,} hang, {prof.get('n_features')} feature, "
                f"{lm.get('num_classes')} lop, cache_build "
                f"{prof['cache_build_seconds']:.1f}s")


# ==================================================================== chay
def run_all(run_dir: str, cfg_path: Path | None) -> list[Check]:
    s = Store(run_dir)
    return [
        c1_resume_mid_epoch(s), c2_history_continuous(s), c3_orchestration_log(s),
        c4_no_ram_overflow(s), c5_no_split_leakage(s),
        c6_final_model_reproduces_yprob(s, cfg_path), c7_report_regenerates(s),
        c8_architecture(s), c9_run_config_keys(s), c10_paper_comparison(s),
        c11_kernel_metadata(), c12_data_profile(s),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kiem tra 12 tieu chi nghiem thu 11.A")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--with-data", action="store_true",
                    help="Bat tieu chi 6 (nap lai model va du doan lai)")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = args.config if (args.with_data and args.config) else None
    checks = run_all(args.run_dir, cfg)

    width = 62
    print("=" * (width + 20))
    print("TIEU CHI NGHIEM THU MDDCC (muc 11.A)")
    print("=" * (width + 20))
    for c in checks:
        mark = {PASS: "[PASS]", FAIL: "[FAIL]", SKIP: "[skip]"}[c.status]
        print(f"{mark} {c.n:>2}. {c.title}")
        if c.detail:
            for line in _wrap(c.detail, width):
                print(f"           {line}")
    n_pass = sum(c.status == PASS for c in checks)
    n_fail = sum(c.status == FAIL for c in checks)
    n_skip = sum(c.status == SKIP for c in checks)
    print("-" * (width + 20))
    print(f"PASS {n_pass}  |  FAIL {n_fail}  |  SKIP {n_skip}  /  {len(checks)}")
    if n_skip:
        print("\nSKIP = chua du du lieu de ket luan, KHONG dong nghia da dat.")
    return 1 if n_fail else 0


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
