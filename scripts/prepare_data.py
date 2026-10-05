"""G03: continuous filtering, frozen Train ACC statistics, common manifest/cache."""
from __future__ import annotations
import argparse
from importlib import metadata
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppg_iot.config import SUBJECTS, load_config, project_path
from ppg_iot.data.loader import dataset_root, load_subject
from ppg_iot.data.alignment import validate_alignment
from ppg_iot.data.preprocessing import fit_acc_statistics, filter_continuous, normalize_acc, preflight
from ppg_iot.data.windows import build_manifest, window_counts
from ppg_iot.provenance import RunRecorder, hash_object, sha256_file, write_json, now_utc, canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--data-root")
    parser.add_argument("--sources", default="results/data_sources.json")
    parser.add_argument("--inventory", default="results/data_inventory.json")
    parser.add_argument("--alignment", default="results/alignment.json")
    parser.add_argument("--out", default="results/manifests")
    args = parser.parse_args()
    config, out = load_config(args.config), project_path(args.out)
    inputs = {name: project_path(getattr(args, name)) for name in
              ("config", "sources", "inventory", "alignment")}
    run = RunRecorder(ROOT, out / "runs", config, ecode="G03", kind="final", inputs=inputs)
    try:
        sources = json.loads(inputs["sources"].read_text(encoding="utf-8"))
        inventory = json.loads(inputs["inventory"].read_text(encoding="utf-8"))
        alignment = json.loads(inputs["alignment"].read_text(encoding="utf-8"))
        if not inventory.get("passed") or not alignment.get("passed"):
            raise ValueError("G02 must pass before preprocessing")
        if inventory["sources_sha256"] != sha256_file(inputs["sources"]):
            raise ValueError("Sources changed since G02 inspection")
        if {row["subject_id"] for row in alignment["subjects"]} != set(SUBJECTS):
            raise ValueError("Alignment does not cover all 15 subjects")
        root, records = dataset_root(args.data_root), {}
        for subject in SUBJECTS:
            if subject in config["split"]["test"]:
                ledger = out.parent / "test_access.jsonl"
                ledger.parent.mkdir(parents=True, exist_ok=True)
                with ledger.open("ab") as stream:
                    stream.write(canonical_json({"at": now_utc(), "run_id": run.record["run_id"],
                                                 "subject_id": subject, "access_type": "fixed_preprocessing",
                                                 "selection": False, "accuracy_evaluation": False}) + b"\n")
            record = load_subject(subject, root, sources["subjects"][subject])
            validate_alignment(record)
            preflight(record)
            records[subject] = record
        stats = fit_acc_statistics(records, config["signal"]["normalization_epsilon"])
        if len(stats["subjects"]) != 10:
            raise ValueError("ACC statistics must include all 10 Train subjects")
        code_files = ("loader.py", "alignment.py", "schema.py", "preprocessing.py", "windows.py")
        recipe = {"signal": config["signal"], "filter": config["filter"], "run": config["run"],
                  "alignment_sha256": sha256_file(inputs["alignment"]),
                  "source_hashes": {s: sources["subjects"][s]["sha256"] for s in SUBJECTS},
                  "acc_stats_sha256": stats["sha256"],
                  "code": {p: sha256_file(ROOT / "src/ppg_iot/data" / p) for p in code_files},
                  "runtime": {p: metadata.version(p) for p in ("numpy", "scipy")},
                  "cache_dtype": {"ppg32": "float64", "acc32_normalized": "float32"},
                  "scipy_defaults": {"sosfiltfilt": {"padtype": "odd", "padlen": None},
                                     "resample_poly": {"window": ["kaiser", 5.0],
                                                       "padtype": "constant", "cval": None}}}
        preprocess_hash = hash_object(recipe)
        cache = ROOT / "data/processed" / preprocess_hash
        cache.mkdir(parents=True, exist_ok=True)
        frames, entries = [], {}
        for subject in SUBJECTS:
            record = records[subject]
            ppg = filter_continuous(record, config)
            acc = normalize_acc(record.acc32, stats)
            current = build_manifest(record, ppg, preprocess_hash=preprocess_hash,
                                     protocol_version=config["run"]["protocol_version"],
                                     activity_min_fraction=config["signal"]["activity_min_fraction"])
            frames.append(current)
            entry = {"source_sha256": record.source_sha256}
            for name, array in (("ppg32", ppg), ("acc32_normalized", acc)):
                path = cache / f"{subject}_{name}.npy"
                np.save(path, array, allow_pickle=False)
                entry[name] = {"file": path.name, "sha256": sha256_file(path),
                               "shape": list(array.shape), "dtype": str(array.dtype)}
            entries[subject] = entry
            print(f"{subject}: {len(current)} candidate windows; eligible={int(current.eligible.sum())}", flush=True)
        manifest = pd.concat(frames, ignore_index=True)
        if manifest.window_id.duplicated().any():
            raise ValueError("Duplicate common-window IDs")
        manifest_path = out / "windows.csv"
        manifest.to_csv(manifest_path, index=False)
        counts = out / "window_counts.csv"
        window_counts(manifest).to_csv(counts, index=False)
        stats_path, recipe_path = out / "acc_stats.json", out / "preprocessing.json"
        write_json(stats_path, stats)
        write_json(recipe_path, {"preprocess_hash": preprocess_hash, "recipe": recipe,
                                 "manifest_ppg_index_rate_hz": 32, "native_ppg_index_rate_hz": 64})
        index_path = cache / "processed_index.json"
        write_json(index_path, {"schema_version": 1, "passed": True,
                                "preprocess_hash": preprocess_hash, "subjects": entries,
                                "manifest_sha256": sha256_file(manifest_path),
                                "acc_stats_sha256": stats["sha256"]})
        pointer = out / "processed_index.json"
        write_json(pointer, {"cache_relative_path": index_path.relative_to(ROOT).as_posix(),
                             "index_sha256": sha256_file(index_path), "preprocess_hash": preprocess_hash,
                             "manifest_sha256": sha256_file(manifest_path)})
        check = out / "checks.json"
        write_json(check, {"gate": "G03", "passed": True, "n_subjects": len(entries),
                           "n_candidates": len(manifest), "n_eligible": int(manifest.eligible.sum()),
                           "n_excluded": int((~manifest.eligible).sum()),
                           "n_mixed_transition": int((manifest.activity_status == "mixed_transition").sum()),
                           "preprocess_hash": preprocess_hash,
                           "manifest_sha256": sha256_file(manifest_path)})
        run.finish(evidence={"manifest": manifest_path, "counts": counts, "stats": stats_path,
                             "recipe": recipe_path, "cache_index": index_path, "checks": check})
        print(f"PASS G03: {out}", flush=True)
        return 0
    except Exception as exc:
        run.finish(error=f"{type(exc).__name__}: {exc}")
        print(f"FAIL G03: {exc}; evidence: {run.directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
