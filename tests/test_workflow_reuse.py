"""Catch stale PTQ parent/calibration reuse without any HR experiment."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppg_iot import notebook_workflow as w


class WorkflowReuseTests(unittest.TestCase):
    def fixture(self, root):
        # These are bytes for checksum-contract tests, not neural weights or results.
        (root / "parent.bin").write_bytes(b"parent checksum fixture")
        (root / "converted.bin").write_bytes(b"quantized checksum fixture")
        ctx = {"root": root, "config": {"quantization": {"backend": "onednn"}}, "index": {"preprocess_hash": "fixture-preprocessing"}}
        parent = {"artifact": "parent.bin", "artifact_sha256": w.sha256_file(root / "parent.bin")}
        calibration = {"split": "train", "n_windows": 1000, "window_ids_sha256": "fixture-train-ids", "calibration_sha256": "fixture-calibration"}
        previous = {"passed": True, "ecode": "E4", "seed": 42, "source_artifact": parent["artifact"],
                    "source_sha256": parent["artifact_sha256"], "config_sha256": w.hash_object(ctx["config"]),
                    "preprocess_hash": ctx["index"]["preprocess_hash"], "calibration": copy.deepcopy(calibration),
                    "backend": "onednn", "runtime": {"python": "3.11.fixture", "torch": "2.10.fixture"},
                    "artifact": "converted.bin", "artifact_sha256": w.sha256_file(root / "converted.bin"),
                    "artifact_bytes": (root / "converted.bin").stat().st_size}
        return ctx, parent, calibration, previous

    @patch.object(w, "accelerator_info", return_value={"python": "3.11.fixture", "torch": "2.10.fixture"})
    def test_parent_and_calibration_changes_reject_previously_valid_reuse(self, _runtime):
        with tempfile.TemporaryDirectory() as directory:
            ctx, parent, calibration, previous = self.fixture(Path(directory))
            self.assertTrue(w.ptq_record_matches(ctx, previous, parent, calibration, 42))
            changed = dict(parent, artifact_sha256="different-parent")
            self.assertFalse(w.ptq_record_matches(ctx, previous, changed, calibration, 42))
            self.assertFalse(w.ptq_record_matches(ctx, previous, parent, dict(calibration, window_ids_sha256="different-ids"), 42))
            self.assertFalse(w.ptq_record_matches(ctx, previous, parent, dict(calibration, split="test"), 42))
            (ctx["root"] / "parent.bin").write_bytes(b"tampered parent")
            self.assertFalse(w.ptq_record_matches(ctx, previous, parent, calibration, 42))

    @patch.object(w, "accelerator_info", return_value={"python": "3.11.fixture", "torch": "2.10.fixture"})
    def test_config_runtime_seed_and_artifact_changes_reject_reuse(self, _runtime):
        with tempfile.TemporaryDirectory() as directory:
            ctx, parent, calibration, previous = self.fixture(Path(directory))
            for update in ({"seed": 123}, {"config_sha256": "other-config"}, {"backend": "other"},
                           {"runtime": {"python": "3.11.fixture", "torch": "other-version"}}):
                with self.subTest(update=update):
                    self.assertFalse(w.ptq_record_matches(ctx, dict(previous, **update), parent, calibration, 42))
            legacy = dict(previous)
            legacy.pop("config_sha256")
            self.assertFalse(w.ptq_record_matches(ctx, legacy, parent, calibration, 42))
            (ctx["root"] / "converted.bin").write_bytes(b"tampered converted bytes")
            self.assertFalse(w.ptq_record_matches(ctx, previous, parent, calibration, 42))

    @patch.object(w, "accelerator_info", return_value={"python": "3.11.fixture", "torch": "2.10.fixture"})
    def test_registry_routes_stale_ptq_to_conversion_and_keeps_valid_reuse(self, _runtime):
        with tempfile.TemporaryDirectory() as directory:
            ctx, parent, calibration, previous = self.fixture(Path(directory))
            w.write_json(ctx["root"] / "results/ptq_index.json", {"runs": {"E4_42": previous}})
            training = {"runs": {"E3_42": parent}}
            with patch.object(w, "_ptq_one", side_effect=AssertionError("conversion must not run")):
                w._ptq_run_registry(ctx, training, calibration, None, None, None, (42,))
            training["runs"]["E3_42"] = dict(parent, artifact_sha256="new-parent")
            with patch.object(w, "_ptq_one", return_value={"passed": True, "test_fixture": True}) as convert:
                w._ptq_run_registry(ctx, training, calibration, None, None, None, (42,))
                convert.assert_called_once()

    def test_locked_execution_rejects_ptq_before_any_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "results/protocol_lock.json").write_text("{}")
            with patch.object(w, "_ptq_one") as convert, self.assertRaisesRegex(ValueError, "locked"):
                w._ptq_run_registry({"root": root}, {}, {}, None, None, None, (42,))
            convert.assert_not_called()

    @patch.object(w, "accelerator_info", return_value={"python": "3.11.fixture", "torch": "2.10.fixture"})
    def test_partial_refresh_cannot_mark_stale_three_seed_registry_complete(self, _runtime):
        with tempfile.TemporaryDirectory() as directory:
            ctx, parent, calibration, previous = self.fixture(Path(directory))
            training = {"runs": {f"E3_{s}": dict(parent) for s in (42,123,999)}}
            index = {"runs": {f"E4_{s}": dict(previous, seed=s) for s in (42,123,999)}}
            self.assertTrue(w.ptq_registry_complete(ctx,index,training,calibration))
            index["runs"]["E4_999"]["source_sha256"] = "obsolete-parent"
            self.assertFalse(w.ptq_registry_complete(ctx,index,training,calibration))

    def test_resumed_early_stopped_training_does_not_add_an_epoch(self):
        ctx = {"config": {"training": {"patience": 10}}}
        state = {"stale": 10}
        with patch.object(w, "train_epoch", side_effect=AssertionError("no further training")):
            w._train_fit(ctx,state,None,None,(None,None,None),None)

    @patch.object(w, "accelerator_info", return_value={"python": "3.11.fixture", "torch": "2.10.fixture", "gpu": "fixture-gpu"})
    def test_resume_rejects_changed_runtime_before_loading_model_state(self, _runtime):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"checksum fixture; torch.load is mocked")
            saved = {"runtime": {"python": "3.11.fixture", "torch": "different", "gpu": "fixture-gpu"}}
            with patch.object(w.torch, "load", return_value=saved):
                with self.assertRaisesRegex(ValueError, "runtime changed"):
                    w._train_resume({"root": root}, {}, checkpoint, w.sha256_file(checkpoint))


if __name__ == "__main__":
    unittest.main()
