"""Verify the lecturer Test presentation without training or changing research artifacts."""
import argparse
import os
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
from demo.data import TEST_SUBJECTS, bundle_index, load_test_windows, record_test_use, replay_segments, write_json
from demo.runtime import digest, infer, load_model, read_json
from demo.paths import artifact_root

SOURCE_ROOT = ROOT
ROOT = artifact_root()


def sample_presentation_windows(windows):
    chosen = {}
    for subject in TEST_SUBJECTS:
        for activity in [0, *range(1, 9)]:
            for segment in replay_segments(windows, subject, activity):
                for index in {0, len(segment) // 2, len(segment) - 1}:
                    chosen[segment[index].window_id] = segment[index]
    return sorted(chosen.values(), key=lambda w: (int(w.subject_id[1:]), w.start_s))


def audit_waveforms(windows):
    pointer = read_json(ROOT / "results/manifests/processed_index.json")
    path = ROOT / pointer["cache_relative_path"]
    if not path.exists():
        return {"available": False, "scope": "portable presentation bundle does not require the continuous cache"}
    if digest(path) != bundle_index(ROOT)["processed_index_sha256"]:
        raise ValueError("Continuous source index differs from the frozen presentation export")
    index = read_json(path)
    frame = pd.read_csv(ROOT / "results/manifests/windows.csv", float_precision="round_trip").set_index("window_id")
    signals, delta = {}, 0.0
    for window in windows:
        if window.subject_id not in signals:
            entries = index["subjects"][window.subject_id]
            signals[window.subject_id] = {}
            for name in ("ppg32", "acc32_normalized"):
                source = path.parent / entries[name]["file"]
                if digest(source) != entries[name]["sha256"]:
                    raise ValueError("Continuous source checksum mismatch")
                signals[window.subject_id][name] = np.load(source, mmap_mode="r")
        row, arrays = frame.loc[window.window_id], signals[window.subject_id]
        ppg = arrays["ppg32"][int(row.ppg_start_idx):int(row.ppg_end_idx)]
        ppg = ((ppg - ppg.mean()) / max(ppg.std(ddof=0), 1e-8)).astype(np.float32)
        acc = arrays["acc32_normalized"][int(row.acc_start_idx):int(row.acc_end_idx)].T
        expected = np.concatenate((ppg[None, :], acc), axis=0)
        delta = max(delta, float(np.max(np.abs(window.features - expected))))
    return {"available": True, "passed": delta <= 1e-6, "n_windows": len(windows), "max_abs_delta": delta, "tolerance": 1e-6}


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firmware", action="store_true")
    parser.add_argument("--run-dir", help="Defaults to the pinned primary run")
    args = parser.parse_args()
    ROOT = artifact_root(args.run_dir)
    os.environ["PPG_RUN_DIR"] = str(ROOT)
    folder = ROOT / "results/demo_final/checks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    report = {"passed": False, "at": datetime.now(timezone.utc).isoformat(), "scope": "locked Test presentation parity/UI; not a new independent evaluation"}
    write_json(folder / "verification.json", report)
    windows = load_test_windows(ROOT)
    selected = sample_presentation_windows(windows)
    report["waveforms"] = audit_waveforms(windows)
    final = read_json(ROOT / "results/final_test/index.json")
    record_test_use(ROOT, "verification_start", {"window_ids": [w.window_id for w in selected],
        "sampling_rule": "first/middle/last of every contiguous subject/activity segment; no error-based selection", "seeds": [42, 123, 999]})
    rows, models, firmware_rows = [], [], []
    for seed in (42, 123, 999):
        for code in ("E3", "E4"):
            model = load_model(ROOT, code, seed)
            predictions = pd.read_csv(ROOT / final["methods"][f"{code}_{seed}"]["predictions"], float_precision="round_trip").set_index("window_id")
            readings = [infer(model, window) for window in selected]
            rows.extend(r.as_record() for r in readings)
            maximum = float(max(abs(r.predicted_bpm - predictions.loc[r.window_id, "hr_pred_bpm"]) for r in readings))
            models.append({"ecode": code, "seed": seed, "artifact_sha256": model.info.artifact_sha256,
                           "n_windows": len(readings), "max_abs_delta_vs_locked_test_bpm": maximum,
                           "tolerance_bpm": 1e-4, "passed": maximum <= 1e-4})
            if code == "E4" and seed == 42:
                for subject in TEST_SUBJECTS:
                    for activity in (1, 7):
                        firmware_rows.append(next(r for r in readings if r.subject_id == subject and r.activity_id == activity))
    pd.DataFrame(rows).to_csv(folder / "fresh_predictions.csv", index=False)
    suite = unittest.defaultTestLoader.discover(str(SOURCE_ROOT / "tests"), pattern="test_demo*.py")
    with (folder / "tests.log").open("w", encoding="utf-8") as stream:
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    lock = read_json(ROOT / "results/protocol_lock.json")
    changed = [name for name, expected in lock["files"].items() if digest(ROOT / name) != expected]
    report.update(n_available_windows=len(windows), subjects=list(TEST_SUBJECTS), n_sample_windows=len(selected),
                  n_fresh_predictions=len(rows), models=models, tests_run=tests.testsRun, tests_passed=tests.wasSuccessful(),
                  historical_lock_mismatches=changed, protocol_lock_sha256=digest(ROOT / "results/protocol_lock.json"),
                  bundle_index_sha256=digest(ROOT / "results/demo_final/bundle.json"),
                  runtime={"python": platform.python_version(), "torch": torch.__version__, "backend": torch.backends.quantized.engine, "threads": torch.get_num_threads()},
                  passed=tests.wasSuccessful() and all(m["passed"] for m in models) and report["waveforms"].get("passed", True) and changed == [])
    if args.firmware:
        from demo.firmware import run_firmware
        report["firmware"] = run_firmware(firmware_rows, root=ROOT)
        report["passed"] = report["passed"] and report["firmware"]["passed"]
    report["source_run_id"] = read_json(ROOT / "workspace.json")["execution_id"]
    report["artifact_workspace"] = str(ROOT)
    write_json(folder / "verification.json", report)
    write_json(ROOT / "results/demo_final/verification_index.json", {"passed": report["passed"], "verification": (folder / "verification.json").relative_to(ROOT).as_posix()})
    print(f"Final demo: {'PASS' if report['passed'] else 'FAIL'}; {len(windows)} available Test windows; {len(rows)} new predictions; {tests.testsRun} tests")
    print(folder.relative_to(ROOT))
    print("Parity:", [(m["ecode"], m["seed"], m["max_abs_delta_vs_locked_test_bpm"]) for m in models])
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
