"""Regression checks for overwriting locked results and unsafe resumes."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppg_iot.execution import create_notebook_workspace, require_unlocked


class ExecutionTests(unittest.TestCase):
    def source_project(self, root):
        for name, data in {"pyproject.toml": "[project]\nname='fixture'\n", "uv.lock": "version=1\n",
                           "configs/default.toml": "[run]\nprotocol='fixture'\n", "src/ppg_iot/helper.py": "value=1\n",
                           "scripts/run.py": "print('fixture')\n"}.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data, encoding="utf-8")
        notebook = root / "DinhLeHoangDanh_DoAn_PPG_DaLiA.ipynb"
        notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["x=1"], "outputs": [], "execution_count": None}]}))
        results = root / "results"
        results.mkdir()
        (results / "protocol_lock.json").write_text('{"historical":true}')
        (results / "data_sources.json").write_text('{"source_fixture":true}')
        return notebook

    def test_fresh_runs_separate_outputs_and_preserve_historical_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = self.source_project(root)
            before = (root / "results/protocol_lock.json").read_bytes()
            first = create_notebook_workspace(root, notebook, run_id="first")
            second = create_notebook_workspace(root, notebook, run_id="second")
            self.assertNotEqual(first, second)
            self.assertFalse((first / "results/protocol_lock.json").exists())
            self.assertFalse((first / "AGENTS.md").exists())
            self.assertEqual((first / "configs/default.toml").read_bytes(), (root / "configs/default.toml").read_bytes())
            self.assertEqual((root / "results/protocol_lock.json").read_bytes(), before)
            with self.assertRaises(FileExistsError):
                create_notebook_workspace(root, notebook, run_id="first")

    def test_resume_accepts_outputs_but_rejects_changed_helper_or_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = self.source_project(root)
            work = create_notebook_workspace(root, notebook, run_id="resumable")
            book = json.loads(notebook.read_text())
            book["cells"][0].update(execution_count=1, outputs=[{"output_type": "stream", "text": "actual fixture output"}])
            notebook.write_text(json.dumps(book))
            self.assertEqual(create_notebook_workspace(root, notebook, resume_run="resumable"), work)
            (root / "src/ppg_iot/helper.py").write_text("value=2\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                create_notebook_workspace(root, notebook, resume_run="resumable")

    def test_locked_run_rejects_resume_and_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = self.source_project(root)
            work = create_notebook_workspace(root, notebook, run_id="locked")
            (work / "results").mkdir(exist_ok=True)
            (work / "results/protocol_lock.json").write_text('{"fixture_lock":true}')
            with self.assertRaisesRegex(ValueError, "locked"):
                create_notebook_workspace(root, notebook, resume_run="locked")
            with self.assertRaisesRegex(ValueError, "locked"):
                require_unlocked(work)

    def test_path_escape_is_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = self.source_project(root)
            for name in ("..", "../outside", "D:/outside", "a/b", "a\\b"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    create_notebook_workspace(root, notebook, run_id=name)

    def test_resume_rejects_modified_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = self.source_project(root)
            work = create_notebook_workspace(root, notebook, run_id="snapshot")
            (work / "configs/default.toml").write_text("changed=true\n")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                create_notebook_workspace(root, notebook, resume_run="snapshot")


if __name__ == "__main__":
    unittest.main()
