"""Actual artifact integration, input isolation and UI interaction checks."""
import copy
from dataclasses import replace
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from demo import runtime
from scripts.demo_pipeline import encode_packet
from demo.paths import artifact_root

SOURCE_ROOT = ROOT
ROOT = artifact_root()


class DemoIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "results/demo/replay_windows.json").is_file():
            raise unittest.SkipTest("Validation integration requires the selected primary run's replay artifacts")
        cls.windows = runtime.load_windows(ROOT)
        cls.models = {code: runtime.load_model(ROOT, code, 42) for code in ("E3", "E4")}
        cls.manifest = pd.read_csv(ROOT / "results/manifests/windows.csv", float_precision="round_trip")
        cls.records = runtime.read_json(ROOT / "results/demo/replay_windows.json")

    def test_fresh_batch_one_matches_notebook_on_every_window(self):
        for code, model in self.models.items():
            actual = [runtime.infer(model, w).predicted_bpm for w in self.windows]
            expected = [record[f"{code}_bpm"] for record in self.records]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)

    def test_stored_predictions_are_not_inputs(self):
        records = copy.deepcopy(self.records)
        for row in records:
            row["E3_bpm"], row["E4_bpm"] = -9999, 9999
        windows = runtime.validate_windows(records, self.manifest, self.windows[0].preprocess_hash)
        for actual, expected in zip(windows, self.windows):
            np.testing.assert_array_equal(actual.features, expected.features)
        model = self.models["E3"]
        self.assertEqual(runtime.infer(model, windows[0]).predicted_bpm,
                         runtime.infer(model, self.windows[0]).predicted_bpm)

    def test_reference_hr_is_not_model_input(self):
        model, window = self.models["E3"], self.windows[0]
        original = runtime.infer(model, window)
        changed = runtime.infer(model, replace(window, hr_ref_bpm=window.hr_ref_bpm + 10))
        self.assertEqual(original.predicted_bpm, changed.predicted_bpm)
        self.assertNotEqual(original.absolute_error_bpm, changed.absolute_error_bpm)

    def test_rejects_changed_reference_and_waveform_shape(self):
        for column, bad_value in [("hr_ref_bpm", 123.0), ("ppg", [0.0] * 255)]:
            records = copy.deepcopy(self.records)
            records[0][column] = bad_value
            with self.assertRaises(runtime.DemoError):
                runtime.validate_windows(records, self.manifest, self.windows[0].preprocess_hash)

    def test_rejects_test_split_and_wrong_preprocessing(self):
        manifest = self.manifest.copy()
        manifest.loc[manifest.window_id == self.windows[0].window_id, "split"] = "test"
        with self.assertRaises(runtime.DemoError):
            runtime.validate_windows(self.records, manifest, self.windows[0].preprocess_hash)
        with self.assertRaises(runtime.DemoError):
            runtime.infer(self.models["E3"], replace(self.windows[0], preprocess_hash="wrong"))

    def test_rejects_wrong_int8_parent_and_bad_checksum(self):
        original = runtime.read_json
        def altered(path):
            packet = original(path)
            if path.name == "ptq_index.json":
                packet["runs"]["E4_42"]["source_sha256"] = "0" * 64
            return packet
        with patch.object(runtime, "read_json", side_effect=altered):
            with self.assertRaisesRegex(runtime.DemoError, "different FP32"):
                runtime.model_entry(ROOT, "E4", 42)
        with patch.object(runtime, "digest", return_value="0" * 64):
            with self.assertRaisesRegex(runtime.DemoError, "checksum"):
                runtime.model_entry(ROOT, "E3", 42)

    def test_new_packet_preserves_checkpoint_identity(self):
        reading = runtime.infer(self.models["E4"], self.windows[0])
        row = pd.Series(dict(reading.as_record(), E4_bpm=reading.predicted_bpm))
        packet = encode_packet(row, "E4")
        fields = packet.strip().split("|")
        self.assertEqual(fields[0], reading.window_id)
        self.assertEqual(fields[1:3], ["validation", "sitting"])
        self.assertEqual(fields[-2:], ["E4", "42"])
        self.assertEqual(fields[3], f"{reading.predicted_bpm:.2f}")

    def test_ui_model_seed_episode_and_controls(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(SOURCE_ROOT / "demo/app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        first = app.session_state["latest_readings"]["E3"]
        app.button(key="next").click().run()
        self.assertEqual(app.session_state["position"], 1)
        self.assertNotEqual(first.window_id, app.session_state["latest_readings"]["E3"].window_id)
        app.selectbox(key="model_code").select("E4").run()
        self.assertEqual(app.session_state["position"], 0)
        app.selectbox(key="seed").select(123).run()
        self.assertEqual(app.session_state["latest_readings"]["E4"].seed, 123)
        app.selectbox(key="activity_id").select(7).run()
        self.assertEqual(app.session_state["latest_readings"]["E4"].activity_name, "walking")
        app.button(key="play").click().run()
        self.assertTrue(app.session_state["playing"])
        self.assertEqual(app.button(key="play").label, "Dừng")
        app.button(key="play").click().run()
        self.assertFalse(app.session_state["playing"])
        self.assertEqual(app.button(key="play").label, "Phát")
        app.button(key="next").click().run()
        app.button(key="reset").click().run()
        self.assertEqual(app.session_state["position"], 0)
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
