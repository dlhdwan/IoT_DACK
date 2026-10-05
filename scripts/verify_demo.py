"""Verify new Validation inference, notebook parity and optional real Wokwi LCD."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from demo.runtime import digest, infer, load_model, load_windows, read_json
from demo.paths import artifact_root

SOURCE_ROOT = ROOT
ROOT = artifact_root()


def verify_source_waveforms(windows):
    """Compare exported demo samples to checksummed continuous preprocessed arrays."""
    pointer = read_json(ROOT / "results/manifests/processed_index.json")
    index_path = ROOT / pointer["cache_relative_path"]
    if not index_path.is_file():
        return {"available": False, "scope": "bulk cache is optional for portable demo execution"}
    if digest(index_path) != pointer["index_sha256"]:
        raise ValueError("Processed array index checksum mismatch")
    index = read_json(index_path)
    frame = pd.read_csv(ROOT / "results/manifests/windows.csv", float_precision="round_trip").set_index("window_id")
    arrays, deltas = {}, []
    for window in windows:
        if window.subject_id not in arrays:
            subject = index["subjects"][window.subject_id]
            values = {}
            for field in ("ppg32", "acc32_normalized"):
                path = index_path.parent / subject[field]["file"]
                if digest(path) != subject[field]["sha256"]:
                    raise ValueError("Continuous signal checksum mismatch")
                values[field] = np.load(path, mmap_mode="r")
            arrays[window.subject_id] = values
        row = frame.loc[window.window_id]
        values = arrays[window.subject_id]
        ppg = values["ppg32"][int(row.ppg_start_idx):int(row.ppg_end_idx)]
        normalized = ((ppg - ppg.mean()) / max(ppg.std(ddof=0), 1e-8)).astype(np.float32)
        acc = values["acc32_normalized"][int(row.acc_start_idx):int(row.acc_end_idx)].T
        expected = np.concatenate((normalized[None, :], acc), axis=0)
        delta = float(np.max(np.abs(expected - window.features)))
        if delta > 1e-6:
            raise ValueError("Demo waveform differs from notebook preprocessing")
        deltas.append(delta)
    return {"available": True, "passed": True, "n_windows": len(windows),
            "max_abs_feature_delta": max(deltas), "tolerance": 1e-6, "processed_index_sha256": digest(index_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firmware", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "results/demo_ui/checks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    windows = load_windows(ROOT)
    waveform_check = verify_source_waveforms(windows)
    records = read_json(ROOT / "results/demo/replay_windows.json")
    reference = {r["window_id"]: r for r in records}
    rows, checks, firmware_readings = [], [], []
    for seed in (42, 123, 999):
        for code in ("E3", "E4"):
            model = load_model(ROOT, code, seed)
            readings = [infer(model, window) for window in windows]
            rows.extend(r.as_record() for r in readings)
            entry = {"model": code, "seed": seed, "n_windows": len(readings),
                     "artifact_sha256": model.info.artifact_sha256, "artifact_bytes": model.info.artifact_bytes,
                     "all_finite": all(np.isfinite(r.predicted_bpm) for r in readings)}
            if seed == 42:
                delta = max(abs(r.predicted_bpm - reference[r.window_id][f"{code}_bpm"]) for r in readings)
                entry.update(max_abs_delta_vs_notebook_bpm=delta, tolerance_bpm=1e-4, parity_passed=delta <= 1e-4)
                if code == "E4":
                    firmware_readings = readings
            checks.append(entry)
    predictions = folder / "fresh_predictions.csv"
    pd.DataFrame(rows).to_csv(predictions, index=False)
    suite = unittest.defaultTestLoader.discover(str(SOURCE_ROOT / "tests"), pattern="test_demo.py")
    with (folder / "tests.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    lock = read_json(ROOT / "results/protocol_lock.json")
    mismatches = [name for name, value in lock["files"].items() if digest(ROOT / name) != value]
    old_lock = ROOT / "uv.lock"
    report = {"at": datetime.now(timezone.utc).isoformat(), "scope": "new CPU inference on 24 Validation windows; no training or Test inference",
              "passed": result.wasSuccessful() and all(c["all_finite"] and c.get("parity_passed", True) for c in checks) and mismatches == [],
              "models": checks, "n_fresh_predictions": len(rows), "tests_run": result.testsRun,
              "waveform_source_check": waveform_check,
              "replay_windows_sha256": digest(ROOT / "results/demo/replay_windows.json"),
              "tests_passed": result.wasSuccessful(), "tests_log": (folder / "tests.log").relative_to(ROOT).as_posix(),
              "predictions": predictions.relative_to(ROOT).as_posix(), "predictions_sha256": digest(predictions),
              "protocol_lock_sha256": digest(ROOT / "results/protocol_lock.json"),
              "historical_lock_file_mismatches": mismatches,
              "demo_uv_lock_sha256": digest(ROOT / "uv.lock"), "reference_uv_lock_sha256": digest(old_lock),
              "reference_uv_lock_matches_history": digest(old_lock) == lock["files"]["uv.lock"],
              "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                          "platform": platform.platform(), "backend": torch.backends.quantized.engine, "threads": torch.get_num_threads()}}
    if args.firmware:
        from demo.firmware import run_firmware
        report["firmware"] = run_firmware(firmware_readings, root=ROOT)
        report["passed"] = report["passed"] and report["firmware"]["passed"]
    output = folder / "verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (ROOT / "results/demo_ui/index.json").write_text(json.dumps({"passed": report["passed"], "verification": output.relative_to(ROOT).as_posix()}, indent=2) + "\n", encoding="utf-8")
    print(f"Demo checks: {'PASS' if report['passed'] else 'FAIL'}; {len(rows)} fresh predictions; {result.testsRun} tests")
    print(output.relative_to(ROOT))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
