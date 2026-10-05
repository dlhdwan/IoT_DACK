"""Verify 15 real source records and the README-based label timeline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppg_iot.config import SUBJECTS, load_config, project_path
from ppg_iot.data.loader import dataset_root, load_subject
from ppg_iot.data.alignment import validate_alignment
from ppg_iot.provenance import (RunRecorder, canonical_json, now_utc,
                               sha256_file, write_json)


def ledger(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_json(event) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--data-root")
    parser.add_argument("--sources", default="results/data_sources.json")
    parser.add_argument("--out", default="results/data_inventory.json")
    args = parser.parse_args()
    config = load_config(args.config)
    out, sources_path = project_path(args.out), project_path(args.sources)
    run = RunRecorder(ROOT, out.parent / "data_inspection", config, ecode="G02", kind="final",
                      inputs={"config": project_path(args.config)})
    rows, alignments, errors = [], [], []
    try:
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
        source_root = dataset_root(args.data_root)
        readme = source_root / "readme.pdf"
        if sha256_file(readme) != sources["readme"]["sha256"]:
            raise ValueError("README source hash mismatch")
        if set(sources["subjects"]) != set(SUBJECTS):
            raise ValueError("Acquisition manifest must identify all 15 official records")
        for subject in SUBJECTS:
            event_id = uuid.uuid4().hex
            is_test = subject in config["split"]["test"]
            base = {"event_id": event_id, "run_id": run.record["run_id"], "subject_id": subject,
                    "access_type": "schema_and_alignment", "purpose": "G02 source verification",
                    "model_selection": False, "accuracy_evaluation": False}
            if is_test:
                ledger(out.parent / "test_access.jsonl", {**base, "status": "started", "at": now_utc()})
            try:
                record = load_subject(subject, source_root, sources["subjects"][subject])
                alignment = validate_alignment(record)
                split = next(name for name, ids in config["split"].items()
                             if name != "unit" and subject in ids)
                rows.append({"subject_id": subject, "split": split,
                             "source_sha256": record.source_sha256,
                             "schema": record.schema_details,
                             "rates_hz": {"PPG": 64, "ACC": 32, "activity": 4, "ECG": 700},
                             "hr_count": len(record.hr_labels),
                             "nonfinite_ppg_samples": int((~np.isfinite(record.ppg64)).sum()),
                             "nonfinite_acc_values": int((~np.isfinite(record.acc32)).sum()),
                             "invalid_hr_labels": int((~np.isfinite(record.hr_labels)
                                                       | (record.hr_labels <= 0)).sum()),
                             "activity_codes": np.unique(record.activity_labels).tolist()})
                alignments.append(alignment)
                print(f"{subject}: {len(record.hr_labels)} HR windows; alignment PASS", flush=True)
                del record
                if is_test:
                    ledger(out.parent / "test_access.jsonl", {**base, "status": "complete", "at": now_utc()})
            except Exception as exc:
                errors.append({"subject_id": subject, "error": f"{type(exc).__name__}: {exc}"})
                if is_test:
                    ledger(out.parent / "test_access.jsonl", {**base, "status": "failed", "at": now_utc(),
                                                            "error": str(exc)})
        passed = len(rows) == 15 and not errors
        alignment_path = out.parent / "alignment.json"
        write_json(out, {"schema_version": 1, "passed": passed, "gate": "G02",
                         "sources_sha256": sha256_file(sources_path), "subjects": rows,
                         "errors": errors, "scope": "data schema/alignment; no model or DSP accuracy"})
        write_json(alignment_path, {"schema_version": 1, "passed": passed,
                                   "readme_sha256": sources["readme"]["sha256"],
                                   "subjects": alignments, "errors": errors})
        run.finish(evidence={"inventory": out, "alignment": alignment_path, "sources": sources_path},
                   error=None if passed else "Real record/alignment checks failed")
        print(f"{'PASS' if passed else 'FAIL'} G02: {out}")
        return 0 if passed else 1
    except Exception as exc:
        write_json(out, {"schema_version": 1, "passed": False, "gate": "G02",
                         "subjects": rows, "errors": [{"error": f"{type(exc).__name__}: {exc}"}]})
        run.finish(evidence={"inventory": out}, error=str(exc))
        print(f"FAIL G02: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
