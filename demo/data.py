"""Portable Test replay from the locked notebook's continuous signal exports."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from demo.runtime import ROOT, DemoError, Window, ACTIVITIES, digest, read_json, resolve_file

TEST_SUBJECTS = ("S13", "S14", "S15")
ACTIVITY_NAMES = {1: "Ngồi", 2: "Lên/xuống cầu thang", 3: "Bóng đá bàn", 4: "Đạp xe",
                  5: "Lái xe", 6: "Ăn trưa", 7: "Đi bộ", 8: "Làm việc", 0: "Chuyển tiếp / hỗn hợp", -1: "Toàn bộ bản ghi"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_hashes(root, files):
    for relative, checksum in files.items():
        if digest(resolve_file(root, relative)) != checksum:
            raise DemoError(f"Frozen file checksum mismatch: {relative}")


def frozen_manifest(root=ROOT):
    lock = read_json(root / "results/protocol_lock.json")
    if not lock.get("passed"):
        raise DemoError("A passed research protocol lock is required before Test replay")
    for name in ("results/manifests/windows.csv", "results/manifests/acc_stats.json",
                 "results/training_index.json", "results/ptq_index.json", "configs/default.toml"):
        verify_hashes(root, {name: lock["files"][name]})
    frame = pd.read_csv(root / "results/manifests/windows.csv", float_precision="round_trip")
    selected = frame.loc[(frame.split == "test") & frame.eligible].reset_index(drop=True)
    if set(selected.subject_id) != set(TEST_SUBJECTS) or len(selected) != 13007 or selected.window_id.duplicated().any():
        raise DemoError("Expected all 13,007 eligible locked Test windows from S13–S15")
    if set(selected.activity_id.dropna().astype(int)) != set(range(1, 9)):
        raise DemoError("Expected all eight Test activities")
    return selected, lock


def export_test_bundle(root=ROOT):
    """Export every Test window in manifest order without computing predictions."""
    root = Path(root)
    frame, lock = frozen_manifest(root)
    pointer = read_json(root / "results/manifests/processed_index.json")
    source_index = resolve_file(root, pointer["cache_relative_path"])
    verify_hashes(root, {pointer["cache_relative_path"]: pointer["index_sha256"]})
    cache = read_json(source_index)
    if not cache.get("passed") or cache["preprocess_hash"] != lock["preprocess_hash"] or cache["acc_stats_sha256"] != lock["files"]["results/manifests/acc_stats.json"]:
        raise DemoError("Continuous signals do not match locked preprocessing/Train ACC statistics")
    final = read_json(root / "results/final_test/index.json")
    if not final.get("passed") or final["split"] != "test" or final["protocol_sha256"] != lock["protocol_sha256"]:
        raise DemoError("Final Test evaluation must be locked and passed before preparing presentation data")
    features = np.empty((len(frame), 4, 256), dtype=np.float32)
    sources = {}
    offsets = np.arange(256)
    for subject, group in frame.groupby("subject_id", sort=False):
        signals = {}
        for kind in ("ppg32", "acc32_normalized"):
            entry = cache["subjects"][subject][kind]
            path = source_index.parent / entry["file"]
            relative = path.relative_to(root).as_posix()
            verify_hashes(root, {relative: entry["sha256"]})
            signals[kind] = np.load(path, mmap_mode="r")
            sources[relative] = entry["sha256"]
        ppg_idx = group.ppg_start_idx.to_numpy(dtype=int)[:, None] + offsets
        acc_idx = group.acc_start_idx.to_numpy(dtype=int)[:, None] + offsets
        ppg = signals["ppg32"][ppg_idx]
        ppg = ((ppg - ppg.mean(axis=1, keepdims=True)) / np.maximum(ppg.std(axis=1, ddof=0, keepdims=True), 1e-8)).astype(np.float32)
        positions = group.index.to_numpy()
        features[positions, 0] = ppg
        features[positions, 1:] = signals["acc32_normalized"][acc_idx].transpose(0, 2, 1)
    if not np.isfinite(features).all():
        raise DemoError("Non-finite Test input: preserve and investigate rather than dropping windows")
    folder = root / "results/demo_final/data" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    feature_file, metadata = folder / "features.npz", folder / "windows.csv"
    np.savez_compressed(feature_file, features=features, window_ids=frame.window_id.to_numpy(dtype=str))
    frame.to_csv(metadata, index=False)
    frozen_files = {"results/protocol_lock.json": digest(root / "results/protocol_lock.json"),
                    "results/final_test/index.json": digest(root / "results/final_test/index.json"),
                    "results/benchmark/index.json": digest(root / "results/benchmark/index.json")}
    for name, entry in final["methods"].items():
        frozen_files[entry["metrics"]] = digest(resolve_file(root, entry["metrics"]))
        verify_hashes(root, {entry["predictions"]: entry["prediction_sha256"]})
        frozen_files[entry["predictions"]] = entry["prediction_sha256"]
    benchmark = read_json(root / "results/benchmark/index.json")
    for name in ("model_benchmark.csv", "paired_benchmark.csv"):
        relative = benchmark["tables"][name]
        frozen_files[relative] = digest(resolve_file(root, relative))
    frozen_files.update({name: lock["files"][name] for name in ("results/manifests/windows.csv", "results/manifests/acc_stats.json",
                                                              "results/training_index.json", "results/ptq_index.json", "configs/default.toml")})
    info = {"passed": True, "mode": "locked_test_presentation", "split": "test", "n_windows": len(frame),
            "subjects": list(TEST_SUBJECTS), "activities": list(range(1, 9)), "preprocess_hash": lock["preprocess_hash"],
            "selection_rule": "all eligible locked Test windows, original manifest order; no selection by predictions/errors",
            "features": feature_file.relative_to(root).as_posix(), "metadata": metadata.relative_to(root).as_posix(),
            "feature_shape": list(features.shape), "dtype": "float32", "feature_bytes": feature_file.stat().st_size,
            "bundle_files": {feature_file.relative_to(root).as_posix(): digest(feature_file), metadata.relative_to(root).as_posix(): digest(metadata)},
            "frozen_files": frozen_files, "continuous_sources": sources,
            "processed_index_sha256": digest(source_index), "at": datetime.now(timezone.utc).isoformat(),
            "scope": "presentation export; no training, calibration or new prediction; offline notebook preprocessing"}
    index_file = folder / "index.json"
    write_json(index_file, info)
    write_json(root / "results/demo_final/bundle.json", {"index": index_file.relative_to(root).as_posix(), "sha256": digest(index_file)})
    return info


def bundle_index(root=ROOT):
    root = Path(root)
    pointer_path = root / "results/demo_final/bundle.json"
    if not pointer_path.is_file():
        raise DemoError("Prepare the Test presentation data: uv run --locked --extra demo python scripts/prepare_demo.py")
    pointer = read_json(pointer_path)
    index_path = resolve_file(root, pointer["index"])
    verify_hashes(root, {pointer["index"]: pointer["sha256"]})
    index = read_json(index_path)
    if not index.get("passed") or index["split"] != "test" or index["mode"] != "locked_test_presentation":
        raise DemoError("Expected a locked Test presentation bundle")
    return index


def load_test_windows(root=ROOT):
    root = Path(root)
    index = bundle_index(root)
    verify_hashes(root, index["frozen_files"])
    verify_hashes(root, index["bundle_files"])
    frame, lock = frozen_manifest(root)
    metadata = pd.read_csv(root / index["metadata"], float_precision="round_trip")
    if metadata.window_id.tolist() != frame.window_id.tolist() or index["preprocess_hash"] != lock["preprocess_hash"]:
        raise DemoError("Test replay IDs or preprocessing differ from the locked experiment")
    pd.testing.assert_frame_equal(metadata, frame, check_dtype=False)
    with np.load(root / index["features"], allow_pickle=False) as packet:
        features, ids = packet["features"], packet["window_ids"].tolist()
    if features.shape != (13007, 4, 256) or features.dtype != np.float32 or ids != frame.window_id.tolist() or not np.isfinite(features).all():
        raise DemoError("Test feature shape/dtype/order/content is invalid")
    features.setflags(write=False)
    windows = []
    for i, row in enumerate(frame.itertuples(index=False)):
        activity = None if pd.isna(row.activity_id) else int(row.activity_id)
        windows.append(Window(row.window_id, row.subject_id, row.start_s, row.end_s, activity,
                              ACTIVITIES.get(activity, "mixed_unknown"), row.hr_ref_bpm,
                              index["preprocess_hash"], features[i], split="test"))
    return windows


def replay_segments(windows, subject, activity=-1):
    selected = sorted((w for w in windows if w.subject_id == subject and
                       (activity == -1 or (w.activity_id or 0) == activity)), key=lambda w: w.start_s)
    segments = []
    for window in selected:
        if not segments or window.start_s - segments[-1][-1].start_s != 2:
            segments.append([])
        segments[-1].append(window)
    return segments


def record_test_use(root, event, details):
    path = Path(root) / "results/demo_final/test_replay_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": datetime.now(timezone.utc).isoformat(), "event": event,
              "scope": "locked Test presentation/verification; no training or calibration", **details}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
