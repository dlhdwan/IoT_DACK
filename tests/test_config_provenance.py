from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppg_iot.config import ProtocolError, load_config, validate_protocol
from ppg_iot.provenance import (RunRecorder, code_snapshot, hash_object,
                                sha256_file, write_json)


class ConfigTests(unittest.TestCase):
    def test_defaults_match_fixed_model(self):
        config = load_config()
        self.assertEqual(config["model"]["conv_channels"], [32, 64, 64])
        self.assertEqual(config["training"]["max_epochs"], 80)

    def test_protocol_changes_are_rejected(self):
        for section, name, value in (
            ("split", "test", ["S10", "S14", "S15"]),
            ("model", "paddings", [3, 2, 1]), ("project", "target_macro_mae_bpm", 5.0),
            ("signal", "model_hz", 64), ("signal", "normalization_ddof", 1),
            ("quantization", "calibration_split", "test"),
            ("evaluation", "measurement_iterations", 999),
            ("quantization", "calibration_windows", True),
            ("training", "learning_rate", float("nan")),
        ):
            with self.subTest(section=section, name=name):
                config = load_config()
                config[section][name] = value
                with self.assertRaises(ProtocolError):
                    validate_protocol(config)

    def test_model_missing_section_rejected(self):
        config = load_config()
        del config["model"]
        with self.assertRaises(ProtocolError):
            validate_protocol(config)

    def test_repeated_load_is_not_mutable_shared_state(self):
        first = load_config()
        first["project"]["seeds"].append(1)
        self.assertEqual(load_config()["project"]["seeds"], [42, 123, 999])


class ProvenanceTests(unittest.TestCase):
    def test_canonical_hash_and_input_sensitivity(self):
        self.assertEqual(hash_object({"a": 1, "b": 2}), hash_object({"b": 2, "a": 1}))
        config = load_config()
        altered = deepcopy(config)
        altered["training"]["learning_rate"] = 2e-3
        self.assertNotEqual(hash_object(config), hash_object(altered))
        with self.assertRaises(ValueError):
            hash_object({"invalid": float("nan")})

    def test_code_hash_tracks_bytes_without_git(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "src").mkdir()
            source = root / "src" / "science.py"
            source.write_text("value=1\n")
            before = code_snapshot(root)["sha256"]
            source.write_text("value=2\n")
            self.assertNotEqual(before, code_snapshot(root)["sha256"])

    def test_run_records_evidence_error_and_unique_ids(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            input_file = root / "input.txt"
            input_file.write_text("input")
            run = RunRecorder(ROOT, root / "runs", load_config(), ecode="SETUP",
                              kind="smoke", inputs={"source": input_file})
            other = RunRecorder(ROOT, root / "runs", load_config(), ecode="SETUP", kind="smoke")
            self.assertNotEqual(run.record["run_id"], other.record["run_id"])
            with self.assertRaises(ValueError):
                run.finish()
            with self.assertRaises(FileNotFoundError):
                run.finish(evidence={"missing": root / "missing.json"})
            artifact = run.directory / "check.json"
            write_json(artifact, {"passed": True, "kind": "synthetic_fixture"})
            run.finish(evidence={"check": artifact})
            saved = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "complete")
            self.assertEqual(saved["source_config_manifest_stats"]["source"]["sha256"],
                             sha256_file(input_file))
            self.assertEqual(saved["evidence"]["check"]["sha256"], sha256_file(artifact))
            with self.assertRaises(RuntimeError):
                run.finish(error="cannot rewrite history")
            other.finish(error="fixture failure")
            self.assertEqual(other.record["status"], "failed")


if __name__ == "__main__":
    unittest.main()
