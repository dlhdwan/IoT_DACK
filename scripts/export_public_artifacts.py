"""Export a small, checksummed selection of real final models and results for Git."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT
SEEDS = (42, 123, 999)
FIGURES = (
    "model_block_diagram.png", "final_method_mae.png", "final_activity_mae.png",
    "training_history.png", "errors_bland_altman.png", "timeline_hard.png",
    "failure_largest_E3_error.png", "failure_largest_ACC_minus_PPG_error.png",
    "failure_largest_INT8_minus_FP32_error.png",
)
REPORT_TABLES = ("three_seed_summary.csv", "activities_test.csv")
BENCHMARK_TABLES = ("model_benchmark.csv", "paired_benchmark.csv", "target_status.csv")
DEMO_FILES = ("demo_replay.png", "demo_results.png", "firmware_lcd.png", "demo_verification.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_path(relative):
    path = (SOURCE_ROOT / relative).resolve()
    if not path.is_relative_to(SOURCE_ROOT) or not path.is_file():
        raise ValueError(f"Missing source or path outside selected run: {relative}")
    return path


def selected_run(run_dir=None):
    if run_dir is None:
        choice = ROOT / "configs/primary_run.json"
        if not choice.is_file():
            raise ValueError("Missing configs/primary_run.json; select a completed run with --run-dir")
        run_dir = read_json(choice)["workspace"]
    candidate = Path(run_dir)
    path = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    if not path.is_relative_to(ROOT) or not path.is_dir():
        raise ValueError(f"Selected run is missing or outside repo: {run_dir}")
    return path


def copy_artifact(relative, target, expected_sha=None):
    source = source_path(relative)
    digest = sha256(source)
    if expected_sha and digest != expected_sha:
        raise ValueError(f"Source checksum mismatch: {relative}")
    shutil.copyfile(source, target)
    if sha256(target) != digest:
        raise ValueError(f"Copy checksum mismatch: {target.name}")
    return {"file": target.name, "sha256": digest, "bytes": target.stat().st_size,
            "original_source": source.relative_to(ROOT).as_posix()}


def export_models(folder, training, ptq):
    models = []
    for code in ("E2", "E3", "E4"):
        for seed in SEEDS:
            entry = (ptq if code == "E4" else training)["runs"][f"{code}_{seed}"]
            if not entry["passed"]:
                raise ValueError(f"Final artifact not verified: {code}_{seed}")
            dtype = "int8" if code == "E4" else "fp32"
            record = copy_artifact(entry["artifact"], folder / f"{code}_seed_{seed}_{dtype}.pt", entry["artifact_sha256"])
            record.update(ecode=code, seed=seed, dtype=dtype,
                          input_channels=1 if code == "E2" else 4,
                          preprocess_hash=entry["preprocess_hash"])
            if code == "E4":
                parent = training["runs"][f"E3_{seed}"]
                calibration = entry["calibration"]
                if entry["source_sha256"] != parent["artifact_sha256"] or calibration["split"] != "train":
                    raise ValueError("INT8 parent/calibration source mismatch")
                record.update(parent_sha256=entry["source_sha256"], backend=entry["backend"],
                              calibration_split="train", calibration_windows=calibration["n_windows"],
                              calibration_window_ids_sha256=calibration["window_ids_sha256"])
            else:
                record.update(selected_by=entry["selection"], best_epoch=entry["best_epoch"])
            models.append(record)
    return models


def export_results(folder, final, benchmark):
    files = []
    for name in REPORT_TABLES + FIGURES:
        files.append(copy_artifact("results/report/" + name, folder / name))
    for name in BENCHMARK_TABLES:
        files.append(copy_artifact(benchmark["tables"][name], folder / name))
    metrics = {}
    sources = {}
    for method, entry in final["methods"].items():
        path = source_path(entry["metrics"])
        metrics[method] = read_json(path)
        sources[method] = {"source": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
    metrics_path = folder / "test_metrics.json"
    write_json(metrics_path, {"split": "test", "subjects": ["S13", "S14", "S15"],
                             "primary_aggregation": "macro_subject", "metrics": metrics,
                             "original_sources": sources})
    runtime_path = folder / "benchmark_runtime.json"
    write_json(runtime_path, {"runtime": benchmark["runtime"], "dsp": benchmark["dsp"],
                              "scope": benchmark["scope"]})
    for path in (metrics_path, runtime_path):
        files.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                      "origin": "read-only summary of original final index/metrics; no new experiment"})
    return files


def export_demo_results(folder, required=False):
    verification_pointer = SOURCE_ROOT / "results/demo_final/verification_index.json"
    browser_pointer = SOURCE_ROOT / "results/demo_final/browser_index.json"
    available = verification_pointer.is_file() and browser_pointer.is_file()
    if not available:
        if required:
            raise ValueError("Verified primary-run Test/UI/firmware evidence is required")
        return []
    demo_pointer, ui_pointer = read_json(verification_pointer), read_json(browser_pointer)
    if not demo_pointer["passed"] or not ui_pointer["passed"]:
        if required:
            raise ValueError("Test demo or browser verification did not pass")
        return []
    demo = read_json(source_path(demo_pointer["verification"]))
    browser = read_json(source_path(ui_pointer["verification"]))
    run_id = read_json(source_path("workspace.json"))["execution_id"]
    if demo["source_run_id"] != run_id or browser["source_run_id"] != run_id:
        raise ValueError("Demo evidence belongs to another notebook run")
    firmware = demo.get("firmware", {})
    if not demo["passed"] or not browser["passed"] or not firmware.get("passed"):
        raise ValueError("Verified Test/UI/firmware evidence is required")
    files = [copy_artifact(browser["screenshots"]["replay"], folder / "demo_replay.png"),
             copy_artifact(browser["screenshots"]["results"], folder / "demo_results.png"),
             copy_artifact(firmware["screenshot"], folder / "firmware_lcd.png", firmware["screenshot_sha256"])]
    sources = {name: {"source": source_path(relative).relative_to(ROOT).as_posix(),
                      "sha256": sha256(source_path(relative))}
               for name, relative in (("test_demo", demo_pointer["verification"]), ("browser", ui_pointer["verification"]))}
    summary = {"passed": True, "source_run_id": run_id,
               "available_test_windows": demo["n_available_windows"], "subjects": demo["subjects"],
               "sample_windows": demo["n_sample_windows"], "fresh_cpu_predictions": demo["n_fresh_predictions"],
               "models": demo["models"], "tests_run": demo["tests_run"], "tests_passed": demo["tests_passed"],
               "waveforms": demo["waveforms"], "firmware_ack_count": firmware["n_windows"],
               "firmware_model_sha256": firmware["model_sha256"],
               "browser_passed": browser["passed"], "browser_actions": browser["actions"],
               "source_verifications": sources,
               "scope": "New host CPU inference; ESP32 firmware simulates UART/LCD; not MCU AI or real-board performance"}
    target = folder / "demo_verification.json"
    write_json(target, summary)
    files.append({"file": target.name, "sha256": sha256(target), "bytes": target.stat().st_size,
                  "origin": "read-only summary of verified primary-run Test/UI/firmware evidence"})
    return files


def export(run_dir=None, include_demo=False):
    global SOURCE_ROOT
    SOURCE_ROOT = selected_run(run_dir)
    training = read_json(source_path("results/training_index.json"))
    ptq = read_json(source_path("results/ptq_index.json"))
    final = read_json(source_path("results/final_test/index.json"))
    benchmark = read_json(source_path("results/benchmark/index.json"))
    if not all(index["passed"] for index in (training, ptq, final, benchmark)) or final["split"] != "test":
        raise ValueError("Verified final experiment indexes required")
    model_folder, result_folder = ROOT / "models/published", ROOT / "results/published"
    model_folder.mkdir(parents=True, exist_ok=True)
    result_folder.mkdir(parents=True, exist_ok=True)
    models = export_models(model_folder, training, ptq)
    result_files = export_results(result_folder, final, benchmark)
    demo_files = export_demo_results(result_folder, required=include_demo)
    result_files.extend(demo_files)
    if not demo_files:
        for name in DEMO_FILES:
            stale = result_folder / name
            if stale.is_file() and stale.resolve().is_relative_to(result_folder.resolve()):
                stale.unlink()
    provenance = {"source_workspace": SOURCE_ROOT.relative_to(ROOT).as_posix(),
                  "source_run_id": read_json(SOURCE_ROOT / "workspace.json")["execution_id"] if (SOURCE_ROOT / "workspace.json").is_file() else "historical-root-2026-10-04",
                  "source_indexes": {source_path(name).relative_to(ROOT).as_posix(): sha256(source_path(name)) for name in
                  ("results/training_index.json", "results/ptq_index.json", "results/final_test/index.json", "results/benchmark/index.json")},
                  "protocol_lock_source": source_path("results/protocol_lock.json").relative_to(ROOT).as_posix(),
                  "protocol_lock_file_sha256": sha256(source_path("results/protocol_lock.json")),
                  "protocol_hash": final["protocol_sha256"],
                  "config_sha256": sha256(source_path("configs/default.toml")),
                  "preprocess_hash": training["preprocess_hash"],
                  "exported_at": datetime.now(timezone.utc).isoformat()}
    write_json(model_folder / "manifest.json", {"schema_version": 1, "scope": "final model-only inference artifacts; not resume checkpoints",
               "models": models, "provenance": provenance})
    write_json(result_folder / "manifest.json", {"schema_version": 1, "scope": "selected results of locked Test experiment, including unfavorable cases",
               "files": result_files, "provenance": provenance, "demo_included": bool(demo_files),
               "selection_rule": "fixed filenames; all seeds/methods retained in metrics, no selection by good errors",
               "limitations": ["Host CPU benchmark, not MCU measurements", "Offline preprocessing", "No clinical or LOSO claims", "D01 same-protocol benchmark remains unverified"]})


def check():
    model_folder, result_folder = ROOT / "models/published", ROOT / "results/published"
    model_manifest = read_json(model_folder / "manifest.json")
    result_manifest = read_json(result_folder / "manifest.json")
    models = model_manifest["models"]
    results = result_manifest["files"]
    if model_manifest["provenance"] != result_manifest["provenance"]:
        raise ValueError("Published model/results provenance differs")
    if {(m["ecode"], m["seed"]) for m in models} != {(c, s) for c in ("E2", "E3", "E4") for s in SEEDS} or len(models) != 9:
        raise ValueError("Expected all nine final inference artifacts")
    expected_results = set(REPORT_TABLES + FIGURES + BENCHMARK_TABLES + ("test_metrics.json", "benchmark_runtime.json"))
    if result_manifest.get("demo_included"):
        expected_results.update(DEMO_FILES)
    if {entry["file"] for entry in results} != expected_results or len(results) != len(expected_results):
        raise ValueError("Public results do not match the selected set")
    for folder, entries in ((model_folder, models), (result_folder, results)):
        for entry in entries:
            path = folder / entry["file"]
            if not path.resolve().is_relative_to(folder.resolve()) or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Public artifact checksum/size/path mismatch: {entry['file']}")
    for model in models:
        if model["ecode"] == "E4":
            parent = next(m for m in models if m["ecode"] == "E3" and m["seed"] == model["seed"])
            if model["parent_sha256"] != parent["sha256"] or model["calibration_split"] != "train":
                raise ValueError("Public INT8 parent/calibration mismatch")
    if result_manifest.get("demo_included"):
        demo = read_json(result_folder / "demo_verification.json")
        if not demo["passed"] or demo["source_run_id"] != result_manifest["provenance"]["source_run_id"]:
            raise ValueError("Public demo does not match primary research provenance")
        for entry in demo["models"]:
            model = next(m for m in models if m["ecode"] == entry["ecode"] and m["seed"] == entry["seed"])
            if entry["artifact_sha256"] != model["sha256"] or not entry["passed"]:
                raise ValueError("Public demo uses another model checkpoint")
    print(f"PASS: {len(models)} final models and {len(results)} selected result files; checksums verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Completed workspace to export; default: configs/primary_run.json")
    parser.add_argument("--include-demo", action="store_true", help="Require verified Test/UI/firmware proof; otherwise include it automatically when available")
    parser.add_argument("--check", action="store_true", help="verify published files only; original local run is not required")
    args = parser.parse_args()
    if not args.check:
        export(args.run_dir, include_demo=args.include_demo)
    check()


if __name__ == "__main__":
    main()
