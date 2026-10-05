"""Presentation coverage, Test provenance, frozen results and actual UI controls."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from demo import data
from demo.runtime import DemoError, infer, load_model
from demo.results import load_research_results
from scripts.demo_pipeline import encode_packet, make_scenario
from demo.paths import artifact_root

SOURCE_ROOT = ROOT
ROOT = artifact_root()


class LecturerDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "results/demo_final/bundle.json").is_file():
            raise unittest.SkipTest("Test integration requires the selected primary run's full demo bundle")
        cls.windows = data.load_test_windows(ROOT)

    def test_all_test_people_activities_and_windows_kept(self):
        self.assertEqual(len(self.windows), 13007)
        self.assertEqual({w.subject_id for w in self.windows}, {"S13", "S14", "S15"})
        self.assertEqual({w.split for w in self.windows}, {"test"})
        self.assertEqual(len({w.window_id for w in self.windows}), len(self.windows))
        for subject in data.TEST_SUBJECTS:
            self.assertEqual({w.activity_id for w in self.windows if w.subject_id == subject and w.activity_id is not None}, set(range(1, 9)))
        self.assertTrue(any(w.activity_id is None for w in self.windows))
        self.assertTrue(all(w.features.shape == (4, 256) and not w.features.flags.writeable for w in self.windows))

    def test_segments_preserve_all_selected_windows_and_real_time(self):
        for subject in data.TEST_SUBJECTS:
            for activity in [-1, 0, *range(1, 9)]:
                segments = data.replay_segments(self.windows, subject, activity)
                actual = [w.window_id for segment in segments for w in segment]
                expected = [w.window_id for w in self.windows if w.subject_id == subject and
                            (activity == -1 or (w.activity_id or 0) == activity)]
                self.assertEqual(actual, expected)
                for segment in segments:
                    self.assertTrue(all(b.start_s - a.start_s == 2 for a, b in zip(segment, segment[1:])))

    def test_bundle_tampering_is_rejected(self):
        with patch.object(data, "digest", return_value="0" * 64):
            with self.assertRaisesRegex(DemoError, "checksum"):
                data.load_test_windows(ROOT)

    def test_reference_metadata_does_not_enter_forward_and_test_packet_is_preserved(self):
        model, window = load_model(ROOT, "E4", 42), self.windows[0]
        original = infer(model, window)
        changed = infer(model, replace(window, subject_id="display_only", activity_id=8, hr_ref_bpm=window.hr_ref_bpm + 10))
        self.assertEqual(original.predicted_bpm, changed.predicted_bpm)
        packet = encode_packet(pd.Series(dict(original.as_record(), E4_bpm=original.predicted_bpm)), "E4")
        self.assertEqual(packet.split("|")[1], "test")
        self.assertEqual(original.split, "test")

    def test_research_summary_uses_full_locked_test_and_adverse_benchmark(self):
        results = load_research_results(ROOT)
        table = results["summary"].set_index("method")
        self.assertEqual(set(table.index), {"E0", "E1", "E2", "E3", "E4"})
        self.assertTrue((table.n_windows == 13007).all())
        published = pd.read_csv(SOURCE_ROOT / "results/published/three_seed_summary.csv")
        for code in ("E3", "E4"):
            expected = published.loc[(published.experiment == code) & (published.metric == "mae_bpm"), "mean"].iloc[0]
            self.assertAlmostEqual(table.loc[code, "mae_bpm"], expected, places=10)
        saved_pairs = pd.read_csv(SOURCE_ROOT / "results/published/paired_benchmark.csv")
        pd.testing.assert_frame_equal(results["pairs"], saved_pairs)

    def test_ui_selects_test_people_all_activities_position_and_tabs(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(SOURCE_ROOT / "demo/app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["latest_readings"]["E3"].split, "test")
        self.assertEqual([t.label for t in app.tabs], ["Trình diễn Test", "Kết quả đồ án", "Mô hình & phương pháp"])
        for subject in data.TEST_SUBJECTS:
            app.selectbox(key="subject_id").select(subject).run()
            self.assertEqual(app.session_state["latest_readings"]["E3"].subject_id, subject)
            for activity in (1, 2, 7, 0, -1):
                app.selectbox(key="activity_id").select(activity).run()
                self.assertFalse(app.exception)
                if activity != -1:
                    self.assertEqual(app.session_state["latest_readings"]["E3"].activity_id or 0, activity)
        # A prior browser widget label must not become an index in another subject's episode.
        app.session_state["segment_id"] = "previous formatted episode label"
        app.selectbox(key="activity_id").select(2).run()
        segments = data.replay_segments(self.windows, "S15", 2)
        app.selectbox(key="segment_S15_2").select(len(segments) - 1).run()
        self.assertEqual(app.session_state["latest_readings"]["E3"].window_id, segments[-1][0].window_id)
        self.assertFalse(app.exception)
        app.selectbox(key="activity_id").select(7).run()
        app.selectbox(key="seed").select(999).run()
        app.selectbox(key="model_code").select("E4").run()
        app.number_input(key="start_at").set_value(10).run()
        self.assertEqual(app.session_state["position"], 9)
        app.button(key="next").click().run()
        self.assertEqual(app.session_state["position"], 10)
        app.button(key="previous").click().run()
        self.assertEqual(app.session_state["position"], 9)
        app.button(key="reset").click().run()
        self.assertEqual(app.session_state["position"], 0)
        app.button(key="play").click().run()
        self.assertEqual(app.button(key="play").label, "Dừng")
        app.button(key="play").click().run()
        self.assertEqual(app.button(key="play").label, "Phát")
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
