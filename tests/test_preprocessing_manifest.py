from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppg_iot.config import load_config
from ppg_iot.data.schema import SubjectRecord
from ppg_iot.data.preprocessing import (fit_acc_statistics, filter_continuous, normalize_ppg,
                                        normalize_acc, preflight)
from ppg_iot.data.windows import assign_activity, build_manifest
from ppg_iot.data.dataset import WindowDataset, collate_windows
from ppg_iot.provenance import sha256_file, write_json


def fixture(subject="S1"):
    t = np.arange(768) / 64
    ppg = np.sin(2*np.pi*t) + 0.15*np.sin(2*np.pi*0.1*t)
    acc = np.column_stack([np.arange(384), np.arange(384)**2, np.ones(384)]).astype(float)
    return SubjectRecord(subject, ppg, acc, np.array([60., 61., 62.]),
                         np.ones(48, dtype=int), "synthetic_fixture", Path("fixture"), 8400)


class PreprocessingTests(unittest.TestCase):
    def test_continuous_filter_is_not_per_window_filter(self):
        record = fixture()
        result = filter_continuous(record, load_config())
        self.assertEqual(result.shape, (384,))
        sos = butter(2, [0.5, 4], btype="bandpass", fs=64, output="sos")
        continuous = sosfiltfilt(sos, record.ppg64)[128:640]
        local = sosfiltfilt(sos, record.ppg64[128:640])
        self.assertGreater(np.max(np.abs(continuous-local)), 0.01)

    def test_nonfinite_fails_before_filter(self):
        record = fixture()
        ppg = record.ppg64.copy()
        ppg[300] = np.nan
        with self.assertRaises(ValueError):
            filter_continuous(replace(record, ppg64=ppg), load_config())

    def test_acc_continuous_moments_ignore_validation_and_overlap(self):
        train, validation = fixture(), fixture("S11")
        records = {"S1": train, "S11": validation}
        stats = fit_acc_statistics(records)
        records["S11"] = replace(validation, acc32=validation.acc32*1000)
        self.assertEqual(stats, fit_acc_statistics(records))
        self.assertEqual(stats["n_samples_per_axis"], 384)
        np.testing.assert_allclose(stats["mean"], train.acc32.mean(axis=0), rtol=0, atol=1e-10)
        np.testing.assert_allclose(stats["std"], train.acc32.std(axis=0, ddof=0), rtol=0, atol=1e-10)
        self.assertEqual(normalize_acc(train.acc32, stats).dtype, np.float32)

    def test_activity_duration_transition_unknown_and_flat_ppg(self):
        labels = np.array([1, 2, 2, 2])
        result = assign_activity(labels, 0.2, 1.0)
        self.assertEqual(result["activity_id"], 2)
        self.assertAlmostEqual(result["activity_fraction"], 0.75/0.8)
        self.assertEqual(assign_activity(np.zeros(32, dtype=int), 0, 8)["activity_status"],
                         "mixed_transition")
        self.assertEqual(assign_activity(np.full(32, 9), 0, 8)["activity_status"], "unknown")
        np.testing.assert_array_equal(normalize_ppg(np.ones(256)), np.zeros(256))

    def test_manifest_keeps_invalid_hr_candidates_and_flat_ppg(self):
        record = replace(fixture(), hr_labels=np.array([60., np.nan, 0.]))
        frame = build_manifest(record, np.ones(384), preprocess_hash="fixture",
                               protocol_version="final_32hz_v1")
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.eligible.tolist(), [True, False, False])
        self.assertTrue(frame.ppg_flat.all())
        self.assertEqual(frame.iloc[-1].ppg_end_idx, 384)

    def test_dataset_channel_and_target_contract_and_stale_cache(self):
        record, config = fixture(), load_config()
        ppg = filter_continuous(record, config)
        acc = normalize_acc(record.acc32, fit_acc_statistics({"S1": record}))
        frame = build_manifest(record, ppg, preprocess_hash="fixture",
                               protocol_version="final_32hz_v1")
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); manifest=root/"windows.csv"; index_path=root/"processed_index.json"
            frame.to_csv(manifest,index=False)
            entries={}
            for name, data in (("ppg32",ppg),("acc32_normalized",acc)):
                path=root/f"{name}.npy"; np.save(path,data,allow_pickle=False)
                entries[name]={"file":path.name,"sha256":sha256_file(path)}
            write_json(index_path,{"passed":True,"manifest_sha256":sha256_file(manifest),
                                  "preprocess_hash":"fixture","subjects":{
                                      "S1":{"source_sha256":"synthetic_fixture",**entries}}})
            e2=WindowDataset(manifest,index_path,split="train",input_channels=1)
            e3=WindowDataset(manifest,index_path,split="train",input_channels=4)
            x,y,meta=e3[0]
            self.assertEqual((x.shape,y.shape),(torch.Size([4,256]),torch.Size([1])))
            self.assertEqual(x.dtype,torch.float32)
            self.assertEqual(meta["subject_id"],"S1")
            torch.testing.assert_close(e2[0][0][0],x[0],rtol=0,atol=0)
            batch=collate_windows([e3[2]])
            self.assertEqual((batch[0].shape,batch[1].shape),
                             (torch.Size([1,4,256]),torch.Size([1,1])))
            e2.close()
            e3.close()
            (root/"ppg32.npy").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                WindowDataset(manifest,index_path,split="train",input_channels=4)


if __name__=="__main__":
    unittest.main()
