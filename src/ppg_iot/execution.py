"""Isolated notebook workspaces; historical experiments are never reset."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import uuid

from .provenance import hash_object, now_utc, sha256_file, write_json

NOTEBOOK = "DinhLeHoangDanh_DoAn_PPG_DaLiA.ipynb"
METADATA_FILES = ("pyproject.toml", "uv.lock", ".python-version", "requirements-gpu-base.txt",
                  "requirements-notebook.txt")


def execution_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"


def notebook_signature(path: Path) -> str:
    book = json.loads(path.read_text(encoding="utf-8"))
    return hash_object([{"cell_type": c["cell_type"], "source": c["source"]} for c in book["cells"]])


def _source_files(root: Path, notebook: Path) -> dict[str, str]:
    paths = [notebook, *(root / name for name in METADATA_FILES if (root / name).is_file())]
    for folder in ("src", "scripts", "configs"):
        suffix = "*.toml" if folder == "configs" else "*.py"
        paths.extend(sorted((root / folder).rglob(suffix)))
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def _run_path(source: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name):
        raise ValueError("Execution ID must be a simple directory name")
    base = (source / "results/notebook_runs").resolve()
    target = (base / name).resolve()
    if target.parent != base:
        raise ValueError("Execution workspace must stay inside results/notebook_runs")
    return target


def validate_resume_source(source: Path, workspace: Path, notebook: Path) -> dict:
    """A resumed experiment uses exactly the snapshot it started with."""
    source, workspace, notebook = source.resolve(), workspace.resolve(), notebook.resolve()
    if workspace.parent != (source / "results/notebook_runs").resolve():
        raise ValueError("Resume workspace is outside this project's notebook runs")
    record = json.loads((workspace / "workspace.json").read_text(encoding="utf-8"))
    current = _source_files(source, notebook)
    # Outputs and execution counts change while running; source cells must not.
    notebook_relative = notebook.relative_to(source).as_posix()
    current.pop(notebook_relative)
    recorded = dict(record["source_files"])
    recorded.pop(notebook_relative)
    if current != recorded or notebook_signature(notebook) != record["notebook_source_sha256"]:
        raise ValueError("Resume source/config/environment differs; start a fresh execution")
    for relative, digest in recorded.items():
        if sha256_file(workspace / relative) != digest:
            raise ValueError(f"Resume snapshot changed: {relative}")
    if notebook_signature(workspace / NOTEBOOK) != record["notebook_source_sha256"]:
        raise ValueError("Resume notebook snapshot changed")
    return record


def create_notebook_workspace(source_root: Path, notebook: Path, *, run_id: str | None = None,
                              resume_run: str | None = None) -> Path:
    source, notebook = Path(source_root).resolve(), Path(notebook).resolve()
    if not (source / "pyproject.toml").is_file() or not (source / "configs/default.toml").is_file():
        raise ValueError("Notebook source needs pyproject.toml and configs/default.toml")
    if notebook.parent != source or not notebook.is_file():
        raise ValueError("Notebook must be a file in the source project root")
    if run_id and resume_run:
        raise ValueError("Choose a fresh execution ID or a resume execution, not both")
    if resume_run:
        workspace = _run_path(source, resume_run)
        require_unlocked(workspace)
        validate_resume_source(source, workspace, notebook)
        return workspace
    name = run_id or execution_id()
    workspace = _run_path(source, name)
    workspace.mkdir(parents=True, exist_ok=False)
    paths = _source_files(source, notebook)
    for relative in paths:
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    if notebook.name != NOTEBOOK:
        shutil.copy2(notebook, workspace / NOTEBOOK)
    source_report = source / "results/data_sources.json"
    if source_report.is_file():
        (workspace / "results").mkdir(exist_ok=True)
        shutil.copy2(source_report, workspace / "results/data_sources.json")
    record = {"schema_version": 1, "execution_id": name, "created_at": now_utc(),
              "source_root": str(source), "workspace": str(workspace), "source_files": paths,
              "notebook_source_sha256": notebook_signature(notebook),
              "scope": "Fresh isolated notebook run; no historical artifacts imported except source metadata"}
    write_json(workspace / "workspace.json", record)
    write_json(source / "results/notebook_runs/latest.json",
               {"execution_id": name, "workspace": workspace.relative_to(source).as_posix(),
                "status": "prepared", "created_at": record["created_at"]})
    return workspace


def require_unlocked(root: Path) -> None:
    if (Path(root) / "results/protocol_lock.json").exists():
        raise ValueError("This execution is locked; start a fresh workspace for preprocessing/training/calibration")
