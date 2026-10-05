"""Run the project notebook with per-attempt evidence and isolated artifacts."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="DinhLeHoangDanh_DoAn_PPG_DaLiA.ipynb")
    parser.add_argument("--output", help="Executed notebook; default updates the primary notebook")
    parser.add_argument("--run-id", help="Optional name for a fresh isolated workspace")
    parser.add_argument("--resume-run", help="Resume an unlocked workspace with identical source/config")
    args = parser.parse_args()
    if args.run_id and args.resume_run:
        parser.error("--run-id and --resume-run are mutually exclusive")
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from ppg_iot.execution import execution_id, _run_path, require_unlocked, validate_resume_source
    from ppg_iot.provenance import write_json, now_utc, sha256_file

    source = (ROOT / args.notebook).resolve()
    output = (ROOT / args.output).resolve() if args.output else source
    if source.parent != ROOT or not output.is_relative_to(ROOT):
        parser.error("Use a notebook in the project root and an output inside this project")
    attempt_id = execution_id()
    run_id = args.run_id or attempt_id
    workspace = _run_path(ROOT, args.resume_run or run_id)
    if args.resume_run:
        require_unlocked(workspace)
        validate_resume_source(ROOT, workspace, source)
    elif workspace.exists():
        parser.error("Fresh workspace already exists; use a new ID or explicit --resume-run")
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    evidence = ROOT / "results/notebook_executions" / attempt_id
    kernel_directory = evidence / "kernel/ppg-project"
    kernel_directory.mkdir(parents=True, exist_ok=False)
    (kernel_directory / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "PPG-DaLiA pinned runtime", "language": "python"}), encoding="utf-8")
    manager = KernelManager(kernel_name="ppg-project", ip="127.0.0.1")
    manager.kernel_spec_manager = KernelSpecManager(kernel_dirs=[str(kernel_directory.parent)])
    environment = dict(os.environ, PYTHONUTF8="1", PPG_PROJECT_ROOT=str(ROOT))
    environment.pop("PPG_EXECUTION_ID", None)
    environment.pop("PPG_RESUME_RUN", None)
    environment["PPG_RESUME_RUN" if args.resume_run else "PPG_EXECUTION_ID"] = args.resume_run or run_id
    completed = []
    start = time.perf_counter()
    status = {"passed": False, "status": "running", "attempt_id": attempt_id,
              "workspace": workspace.relative_to(ROOT).as_posix(), "started_at": now_utc(),
              "python": sys.version, "scope": "Actual notebook execution; artifacts isolated per run"}
    write_json(evidence / "status.json", status)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save_notebook(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".ipynb.tmp")
        nbformat.write(notebook, temporary)
        for retry in range(8):
            try:
                temporary.replace(target)
                return
            except PermissionError:
                if retry == 7:
                    raise
                time.sleep(0.15)

    def checkpoint(cell, cell_index, **_):
        completed.append(cell_index)
        save_notebook(output)
        save_notebook(evidence / "executed.ipynb")
        status.update(completed_code_cells=len(completed), last_cell_index=cell_index)
        write_json(evidence / "status.json", status)
        print(f"saved notebook: cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    def finish(passed, error=None):
        save_notebook(output)
        save_notebook(evidence / "executed.ipynb")
        errors = sum(o.output_type == "error" for c in notebook.cells if c.cell_type == "code" for o in c.outputs)
        expected = sum(c.cell_type == "code" for c in notebook.cells)
        passed = passed and not errors and len(completed) == expected
        status.update(passed=passed, status="complete" if passed else "failed", error=error,
                      executed_code_cells=len(completed), error_outputs=errors,
                      expected_code_cells=expected,
                      ended_at=now_utc(), wall_seconds=time.perf_counter() - start,
                      notebook_sha256=sha256_file(output))
        write_json(evidence / "status.json", status)
        if workspace.is_dir():
            save_notebook(workspace / "DinhLeHoangDanh_DoAn_PPG_DaLiA.ipynb")
            write_json(workspace / "results/notebook_execution.json", status)
            write_json(ROOT / "results/notebook_runs/latest.json", {
                "execution_id": workspace.name, "workspace": status["workspace"],
                "status": status["status"], "attempt_id": attempt_id})
        write_json(ROOT / "results/notebook_executions/latest.json", {"status_file": (evidence / "status.json").relative_to(ROOT).as_posix(), **status})
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)

    client = NotebookClient(notebook, km=manager, timeout=3600, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}}, on_cell_executed=checkpoint)
    try:
        client.execute(cwd=str(ROOT), env=environment)
        finish(True)
        return 0
    except BaseException as error:
        finish(False, str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
