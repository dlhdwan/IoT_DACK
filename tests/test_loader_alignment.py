from dataclasses import replace
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppg_iot.data.loader import decode_subject, load_subject
from ppg_iot.data.alignment import validate_alignment, sample_bounds
from ppg_iot.provenance import sha256_file


def raw_fixture():
    return {"subject": "S1", "signal": {
        "wrist": {"BVP": np.ones((768, 1)), "ACC": np.ones((384, 3))},
        "chest": {"ECG": np.ones((8400, 1))}},
        "label": np.array([51.0, 71.0, 81.0]),
        "activity": np.ones((48, 1))}


def record_fixture():
    return decode_subject(raw_fixture(), "S1", Path("synthetic_fixture.pkl"), "synthetic_fixture")


class LoaderAlignmentTests(unittest.TestCase):
    def test_first_middle_last_inclusive_complete_windows(self):
        record = record_fixture()
        alignment = validate_alignment(record)
        self.assertEqual(alignment["label_count"], 3)
        self.assertEqual(sample_bounds(record, 2, 32), (128, 384))
        self.assertEqual(alignment["checks"][0]["hr_ref_bpm"], 51.0)
        self.assertEqual(alignment["checks"][-1]["end_s"], 12.0)

    def test_offset_missing_label_incomplete_channel_wrong_rate_fail(self):
        record = record_fixture()
        altered = (replace(record, label_start_s=2), replace(record, hr_labels=np.ones(2)),
                   replace(record, acc32=np.ones((383, 3))), replace(record, activity_hz=32),
                   replace(record, ecg_sample_count=8399))
        for value in altered:
            with self.subTest(record=value.subject_id), self.assertRaises(ValueError):
                validate_alignment(value)

    def test_unknown_missing_or_subject_mismatched_schema_fail(self):
        candidates = [{"data": raw_fixture()}, {**raw_fixture(), "subject": "S2"}]
        missing = raw_fixture()
        del missing["label"]
        candidates.append(missing)
        for raw in candidates:
            with self.assertRaises(ValueError):
                decode_subject(raw, "S1", Path("fixture"), "synthetic_fixture")

    def test_hr_is_official_label_independent_of_ppg(self):
        raw = raw_fixture()
        first = decode_subject(raw, "S1", Path("fixture"), "synthetic_fixture")
        raw["signal"]["wrist"]["BVP"] = np.zeros((768, 1))
        other = decode_subject(raw, "S1", Path("fixture"), "synthetic_fixture")
        np.testing.assert_array_equal(first.hr_labels, other.hr_labels)

    def test_source_hash_is_enforced_before_pickle_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "S1/S1.pkl"
            path.parent.mkdir()
            with path.open("wb") as stream:
                pickle.dump(raw_fixture(), stream)
            with self.assertRaises(ValueError):
                load_subject("S1", root, {"sha256": "wrong"})
            record = load_subject("S1", root, {"sha256": sha256_file(path)})
            self.assertEqual(validate_alignment(record)["label_count"], 3)


if __name__ == "__main__":
    unittest.main()
