"""A new notebook run must not silently resolve to the historical root run."""
import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.paths import artifact_root


class DemoPathsTests(unittest.TestCase):
    def test_default_selects_pinned_primary_run(self):
        with patch.dict(os.environ, {}, clear=True):
            choice = json.loads((ROOT / "configs/primary_run.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact_root(), ROOT / choice["workspace"])

    def test_missing_pin_does_not_fall_back_to_historical_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("demo.paths.SOURCE_ROOT", Path(directory)), patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(FileNotFoundError, "primary_run"):
                    artifact_root()

    def test_relative_workspace_is_resolved_from_source_root(self):
        with patch.dict(os.environ, {"PPG_RUN_DIR": "results/notebook_runs/fixture"}):
            self.assertEqual(artifact_root(), ROOT / "results/notebook_runs/fixture")

    def test_explicit_absolute_workspace_overrides_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"PPG_RUN_DIR": "results/notebook_runs/obsolete-fixture"}):
                self.assertEqual(artifact_root(directory), Path(directory).resolve())
