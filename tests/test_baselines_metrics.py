import math
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppg_iot.baselines import fft_hr, ibi_hr, apply_ibi_fallback
from ppg_iot.metrics import (join_predictions, compute_metrics, train_median_fallback,
                             summarize_seeds, paired_subset_metrics)


def fixture(subjects=("S1", "S2", "S2"), errors=(0.0, 10.0, 10.0)):
    manifest = pd.DataFrame({
        "window_id": [f"w{i}" for i in range(len(subjects))], "subject_id": subjects,
        "split": ["train"] * len(subjects), "hr_ref_bpm": [60.0] * len(subjects),
        "activity_id": [1] * len(subjects), "activity_status": ["assigned"] * len(subjects),
        "eligible": [True] * len(subjects)})
    predictions = manifest[["window_id", "hr_ref_bpm"]].copy()
    predictions["hr_pred_bpm"] = 60.0 + np.asarray(errors)
    predictions["prediction_valid"] = True
    return manifest, predictions


class DSPTests(unittest.TestCase):
    def test_fft_grid_and_ibi_at_actual_rates(self):
        for fs in (32, 64):
            for frequency in (1.0, 2.0):
                with self.subTest(fs=fs, frequency=frequency):
                    x = np.sin(2 * np.pi * frequency * np.arange(fs * 8) / fs)
                    self.assertAlmostEqual(fft_hr(x, fs).hr_pred_bpm, frequency * 60,
                                           delta=fs / 4096 * 60)
                    self.assertAlmostEqual(ibi_hr(x, fs).hr_pred_bpm, frequency * 60, places=8)

    def test_flat_fft_is_finite_flagged_first_grid_bin(self):
        result = fft_hr(np.ones(256), 32)
        self.assertEqual(result.reason, "low_signal")
        self.assertEqual(result.hr_pred_bpm, 36.09375)
        self.assertTrue(result.prediction_valid)

    def test_insufficient_peaks_and_named_fallback(self):
        result = ibi_hr(np.ones(256), 32)
        self.assertFalse(result.prediction_valid)
        self.assertIsNone(result.hr_pred_bpm)
        fallback = apply_ibi_fallback(result, 73.0)
        self.assertTrue(fallback["used_fallback"])
        self.assertEqual(fallback["hr_pred_bpm"], 73.0)

    def test_nonfinite_input_fails_instead_of_dropping(self):
        for fn in (fft_hr, ibi_hr):
            with self.assertRaises(ValueError):
                fn(np.array([1.0, float("nan"), 1.0]), 32)


class MetricTests(unittest.TestCase):
    def test_subject_macro_differs_from_pooled(self):
        m, p = fixture()
        report = compute_metrics(join_predictions(m, p))
        self.assertEqual(report["overall"]["macro_subject"]["mae_bpm"], 5.0)
        self.assertAlmostEqual(report["overall"]["pooled_window"]["mae_bpm"], 20 / 3)
        self.assertEqual(report["overall"]["pooled_window"]["error_gt_10_percent"], 0.0)

    def test_rmse_subject_first_and_strict_large_error(self):
        m, p = fixture(errors=(2.0, 0.0, 4.0))
        report = compute_metrics(join_predictions(m, p))
        self.assertAlmostEqual(report["overall"]["macro_subject"]["rmse_bpm"],
                               (2 + math.sqrt(8)) / 2)
        self.assertAlmostEqual(report["overall"]["pooled_window"]["rmse_bpm"], math.sqrt(20 / 3))
        p.loc[2, "hr_pred_bpm"] = 71.0
        report = compute_metrics(join_predictions(m, p))
        self.assertAlmostEqual(report["overall"]["pooled_window"]["error_gt_10_percent"], 100 / 3)

    def test_absent_activities_and_subjects_are_na_not_zero(self):
        m, p = fixture()
        report = compute_metrics(join_predictions(m, p), expected_subjects=["S1", "S2", "S3"])
        self.assertIsNone(report["by_activity"][1]["macro_subject"]["mae_bpm"])
        self.assertEqual(report["by_activity"][1]["pooled_window"]["n_total"], 0)
        self.assertIsNone(report["by_subject"][2]["mae_bpm"])

    def test_reordered_predictions_use_ids(self):
        m, p = fixture()
        joined = join_predictions(m, p.iloc[::-1])
        self.assertEqual(joined.hr_pred_bpm.tolist(), [60, 70, 70])

    def test_population_reference_and_finite_checks(self):
        m, p = fixture()
        candidates = [p.iloc[:2], pd.concat([p, p.iloc[:1]], ignore_index=True)]
        changed = p.copy()
        changed.loc[0, "hr_ref_bpm"] = 61
        candidates.append(changed)
        nonfinite = p.copy()
        nonfinite.loc[0, "hr_pred_bpm"] = np.inf
        candidates.append(nonfinite)
        for value in candidates:
            with self.subTest(index=len(value)), self.assertRaises(ValueError):
                join_predictions(m, value)

    def test_pure_ibi_coverage_and_paired_subset(self):
        m, p = fixture()
        pure = p.copy()
        pure.loc[0, "prediction_valid"] = False
        pure.loc[0, "hr_pred_bpm"] = np.nan
        with self.assertRaises(ValueError):
            join_predictions(m, pure)
        pure_joined = join_predictions(m, pure, allow_invalid=True)
        report = compute_metrics(pure_joined)
        self.assertAlmostEqual(report["overall"]["pooled_window"]["coverage"], 2 / 3)
        self.assertEqual(report["overall"]["pooled_window"]["n_total"], 3)
        paired = paired_subset_metrics({"FFT": join_predictions(m, p), "IBI_pure": pure_joined})
        self.assertEqual(paired["n_paired"], 2)
        self.assertEqual(paired["methods"]["FFT"]["overall"]["pooled_window"]["mae_bpm"], 10)

    def test_fallback_fit_uses_train_only(self):
        m, _ = fixture()
        m.loc[2, "split"] = "validation"
        m.loc[2, "hr_ref_bpm"] = 1000
        self.assertEqual(train_median_fallback(m)["hr_bpm"], 60.0)

    def test_seed_std_is_sample_std_of_three_scores(self):
        self.assertEqual(summarize_seeds({42: 1.0, 123: 2.0, 999: 3.0})["std"], 1.0)
        with self.assertRaises(ValueError):
            summarize_seeds({42: 1.0})


if __name__ == "__main__":
    unittest.main()
