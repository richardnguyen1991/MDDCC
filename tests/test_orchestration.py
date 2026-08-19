"""Kiem chung dieu phoi Kaggle, notebook va workflow - muc 8, tieu chi 11.A.3, 11.A.11."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import kaggle_orchestrator as K   # noqa: E402


def state(epoch, total=100, status="running", **kw):
    return {"current_epoch": epoch, "total_epochs": total, "status": status, **kw}


def pushes(*epochs):
    return [{"current_epoch_before": e, "pushed_at_utc": f"t{i}"}
            for i, e in enumerate(epochs)]


# ------------------------------------------------------ quyet dinh (muc 8.B)
def test_done_when_all_epochs_finished():
    """Muc 8.B.3: epoch >= 100 -> DONE, khong push nua."""
    d = K.decide(state=state(100), kernel_status="complete", restart_history=[])
    assert d.action == K.ACTION_DONE


def test_done_when_status_completed_even_if_epoch_lower():
    d = K.decide(state=state(97, status="completed"), kernel_status="complete",
                 restart_history=[])
    assert d.action == K.ACTION_DONE


@pytest.mark.parametrize("st", ["running", "queued", "RUNNING", " Queued "])
def test_wait_while_kernel_is_alive(st):
    """Muc 8.B.4: dang chay/xep hang -> khong dung vao."""
    d = K.decide(state=state(12), kernel_status=st, restart_history=[])
    assert d.action == K.ACTION_WAIT


@pytest.mark.parametrize("st", ["complete", "error", "cancelAcknowledged",
                                "cancelRequested"])
def test_push_when_kernel_finished_and_epochs_remain(st):
    d = K.decide(state=state(12), kernel_status=st, restart_history=[])
    assert d.action == K.ACTION_PUSH
    assert d.current_epoch == 12 and d.total_epochs == 100


def test_push_when_no_state_yet():
    """Chua co run nao -> phai push de tao run dau tien."""
    d = K.decide(state=None, kernel_status="complete", restart_history=[])
    assert d.action == K.ACTION_PUSH


def test_wait_when_no_state_but_kernel_running():
    d = K.decide(state=None, kernel_status="running", restart_history=[])
    assert d.action == K.ACTION_WAIT


def test_push_when_kernel_status_unknown():
    """Khong doc duoc trang thai (Kaggle API loi) -> van push, khong treo mai."""
    d = K.decide(state=state(5), kernel_status="", restart_history=[])
    assert d.action == K.ACTION_PUSH


# --------------------------------------------- chong vong lap vo han (muc 8.B)
def test_abort_after_three_pushes_without_progress():
    """3 lan push lien tiep ma current_epoch dung yen -> dung."""
    d = K.decide(state=state(7), kernel_status="error",
                 restart_history=pushes(7, 7, 7))
    assert d.action == K.ACTION_ABORT
    assert "dung yen o 7" in d.reason


def test_no_abort_when_epoch_is_progressing():
    d = K.decide(state=state(9), kernel_status="error",
                 restart_history=pushes(7, 8, 9))
    assert d.action == K.ACTION_PUSH


def test_no_abort_when_only_two_stalled_pushes():
    d = K.decide(state=state(7), kernel_status="complete",
                 restart_history=pushes(5, 7, 7))
    assert d.action == K.ACTION_PUSH


def test_stall_check_uses_only_the_last_three():
    """Da tung dung yen o qua khu nhung nay dang tien -> van push."""
    d = K.decide(state=state(20), kernel_status="complete",
                 restart_history=pushes(7, 7, 7, 15, 18, 20))
    assert d.action == K.ACTION_PUSH


def test_abort_when_max_restarts_exceeded():
    d = K.decide(state=state(50), kernel_status="complete",
                 restart_history=pushes(*range(1, 61)), max_restarts=60)
    assert d.action == K.ACTION_ABORT
    assert "max_restarts=60" in d.reason


def test_default_max_restarts_covers_39_sessions():
    """Ngan sach do duoc la 39 session -> mac dinh phai lon hon, co bien an toan."""
    assert K.DEFAULT_MAX_RESTARTS >= 45


def test_done_wins_over_stall_guard():
    """Da xong roi thi khong duoc bao abort du history dung yen."""
    d = K.decide(state=state(100), kernel_status="error",
                 restart_history=pushes(100, 100, 100))
    assert d.action == K.ACTION_DONE


# ------------------------------------------------------ doc trang thai kernel
@pytest.mark.parametrize("raw,expect", [
    ('{"status": "complete"}', "complete"),
    ('{"status": "running", "failureMessage": null}', "running"),
    ('Kernel is in status "complete"', "complete"),
    ('has status "error"', "error"),
    ("", ""),
])
def test_parse_kernel_status(raw, expect):
    assert K.parse_kernel_status(raw) == expect


# ---------------------------------------------------- kernel-metadata (muc 8.C)
META = json.loads((REPO / "kernel" / "kernel-metadata.json").read_text(encoding="utf-8"))


def test_dataset_sources_is_declared():
    """Tieu chi 11.A.11 - loi hay gap nhat khi tu dong hoa bang kaggle kernels push."""
    assert META["dataset_sources"] == ["dungnguyen28101991/cicddos2019-parquet"]


def test_kernel_runs_on_cpu_with_internet():
    assert META["enable_internet"] is True, "can internet de pip install + upload S3"
    assert META["enable_gpu"] is False
    assert META["enable_tpu"] is False
    assert META["accelerator"] == "none"


def test_kernel_is_private_and_points_at_notebook():
    assert META["is_private"] is True
    assert META["kernel_type"] == "notebook"
    assert META["code_file"] == "kaggle_notebook.ipynb"
    assert (REPO / "kernel" / META["code_file"]).exists()


def test_kernel_id_matches_agreed_slug():
    assert META["id"] == "richardnguyen1991/mddcc"


def test_metadata_has_no_credentials():
    """Muc 8.A: TUYET DOI khong nhet credential vao kernel-metadata.json."""
    raw = (REPO / "kernel" / "kernel-metadata.json").read_text(encoding="utf-8").lower()
    for bad in ("aws_secret", "aws_access", "akia", "password", "token"):
        assert bad not in raw


# ------------------------------------------------------------ notebook (8.D)
NB = json.loads((REPO / "kernel" / "kaggle_notebook.ipynb").read_text(encoding="utf-8"))
CODE_CELLS = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
ALL_CODE = "\n".join(CODE_CELLS)


def test_every_code_cell_compiles():
    for i, src in enumerate(CODE_CELLS):
        try:
            ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(f"cell code {i} loi cu phap: {exc}")


def test_notebook_fails_fast_when_dataset_missing():
    """Muc 8.C: phai kiem tra /kaggle/input ngay dong dau."""
    first = CODE_CELLS[0]
    assert "/kaggle/input" in first
    assert "SystemExit" in first and "FAIL-FAST" in first
    assert "dataset_sources" in first, "thong bao loi phai chi ro nguyen nhan"


def test_notebook_reads_secrets_from_kaggle_not_hardcoded():
    assert "kaggle_secrets" in ALL_CODE and "UserSecretsClient" in ALL_CODE
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                 "AWS_DEFAULT_REGION", "S3_BUCKET", "S3_PREFIX"):
        assert name in ALL_CODE, f"thieu secret {name}"


def test_notebook_never_prints_secret_values():
    """Chi duoc in do dai, khong in gia tri."""
    assert "ky tu>" in ALL_CODE
    assert "print(os.environ[\"AWS_SECRET_ACCESS_KEY\"])" not in ALL_CODE
    assert "print(read(" not in ALL_CODE


def test_notebook_has_no_hardcoded_credentials():
    for bad in ("AKIA", "aws_secret_access_key =", "password="):
        assert bad not in ALL_CODE


def test_notebook_prints_state_before_and_after():
    """Muc 8.D: in run_id / session_id / current_epoch o dau VA cuoi log."""
    assert ALL_CODE.count("run_id") >= 4
    assert "current_epoch" in ALL_CODE and "session_id" in ALL_CODE
    assert "truoc khi chay" in ALL_CODE


def test_notebook_is_idempotent_does_not_force_new_run_id():
    """Lan chay thu N chi tiep tuc, khong tao run_id moi."""
    assert "RunRegistry(store).get()" in ALL_CODE
    assert "new_run_id" not in ALL_CODE, "notebook khong duoc tu tao run_id moi"


def test_notebook_changes_dir_before_removing_it():
    """Xoa thu muc dang dung se lam mat CWD cua shell - loi da gap that."""
    clone_cell = next(c for c in CODE_CELLS if "git clone" in c)
    assert clone_cell.index('os.chdir("/kaggle/working")') < clone_cell.index("rmtree")


def test_notebook_runs_evaluation_only_when_complete():
    """Muc 4.8: danh gia cuoi la buoc rieng, chi chay khi da du 100 epoch."""
    cell = next(c for c in CODE_CELLS if "src.evaluate" in c)
    assert "is_complete" in cell
    assert "make_report.py" in cell


def test_notebook_does_not_train_on_github_runner():
    """Workflow chi goi Kaggle API - khong duoc train tren runner."""
    wf = (REPO / ".github" / "workflows" / "run-kaggle.yml").read_text(encoding="utf-8")
    assert "src.train" not in wf
    assert "src.evaluate" not in wf


# ------------------------------------------------------------ workflow (8.B)
WF = yaml.safe_load(
    (REPO / ".github" / "workflows" / "run-kaggle.yml").read_text(encoding="utf-8"))
WF_TRIGGERS = WF[True]        # YAML doc "on:" thanh True


def test_workflow_has_schedule_and_manual_trigger():
    assert WF_TRIGGERS["schedule"] == [{"cron": "*/30 * * * *"}]
    assert "workflow_dispatch" in WF_TRIGGERS


def test_workflow_concurrency_prevents_overlapping_pushes():
    assert WF["concurrency"]["group"] == "mddcc-kaggle"
    assert WF["concurrency"]["cancel-in-progress"] is False


def test_workflow_can_open_issues():
    assert WF["permissions"]["issues"] == "write"


def test_workflow_checks_secrets_before_doing_anything():
    steps = WF["jobs"]["orchestrate"]["steps"]
    names = [s.get("name", "") for s in steps]
    check = next(i for i, n in enumerate(names) if "secret" in n.lower())
    push = next(i for i, n in enumerate(names) if "Dieu phoi" in n)
    assert check < push, "phai kiem tra secret truoc khi goi Kaggle API"


def test_workflow_handles_both_kaggle_token_formats():
    """Muc 8.A: KAGGLE_API_TOKEN co the la key thuan hoac noi dung kaggle.json."""
    step = next(s for s in WF["jobs"]["orchestrate"]["steps"]
                if "kaggle.json" in s.get("name", ""))
    assert "json.loads" in step["run"]
    assert "KAGGLE_USERNAME" in step["run"]
    assert "chmod 600" in step["run"]


def test_workflow_opens_issue_only_on_abort():
    step = next(s for s in WF["jobs"]["orchestrate"]["steps"]
                if "Issue" in s.get("name", ""))
    assert step["if"] == "steps.orchestrate.outputs.abort == 'true'"


def test_workflow_passes_max_restarts():
    assert WF["jobs"]["orchestrate"]["env"]["MAX_RESTARTS"] == "60"


def test_workflow_does_not_embed_secret_values():
    raw = (REPO / ".github" / "workflows" / "run-kaggle.yml").read_text(encoding="utf-8")
    assert "AKIA" not in raw
    # moi secret phai di qua ${{ secrets.* }}
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET",
                 "KAGGLE_KERNEL"):
        assert f"secrets.{name}" in raw


# ---------------------------------------------------- build_notebook idempotent
def test_build_notebook_is_deterministic(tmp_path, monkeypatch):
    """Chay lai build_notebook.py phai cho file y het - de diff git sach."""
    import importlib

    import build_notebook

    before = (REPO / "kernel" / "kaggle_notebook.ipynb").read_text(encoding="utf-8")
    importlib.reload(build_notebook)
    rebuilt = json.dumps(build_notebook.build(), indent=1, ensure_ascii=False) + "\n"
    assert rebuilt == before, "kernel/kaggle_notebook.ipynb da lech voi script sinh no"
