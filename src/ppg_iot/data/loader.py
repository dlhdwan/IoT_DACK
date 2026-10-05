"""Load only official synchronized pickles verified against the acquisition manifest."""
from __future__ import annotations
import os
from pathlib import Path
import pickle
import numpy as np
from ppg_iot.config import SUBJECTS, project_path
from ppg_iot.provenance import sha256_file
from .schema import SubjectRecord


def dataset_root(value: str | Path | None = None) -> Path:
    return project_path(value if value is not None else os.environ.get("PPG_DALIA_DIR", "data/raw"))


def _numeric(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name}: expected a real numeric array")
    return array


def decode_subject(raw: dict, subject: str, path: Path, source_sha256: str) -> SubjectRecord:
    if subject not in SUBJECTS:
        raise ValueError(f"Unsupported subject: {subject}")
    if not isinstance(raw, dict) or not {"signal", "label", "activity", "subject"}.issubset(raw):
        raise ValueError("Unknown synchronized pickle schema")
    if raw["subject"] != subject:
        raise ValueError("Pickle subject does not match requested subject/path")
    try:
        wrist = raw["signal"]["wrist"]
        bvp = _numeric(wrist["BVP"], "wrist/BVP")
        acc = _numeric(wrist["ACC"], "wrist/ACC")
        hr = _numeric(raw["label"], "ECG-derived HR label")
        activity = _numeric(raw["activity"], "activity")
        ecg = _numeric(raw["signal"]["chest"]["ECG"], "chest/ECG")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing synchronized data field: {exc}") from exc
    if bvp.ndim != 2 or bvp.shape[1] != 1 or not len(bvp):
        raise ValueError("BVP must have shape (N,1)")
    if acc.ndim != 2 or acc.shape[1] != 3 or not len(acc):
        raise ValueError("Wrist ACC must have shape (N,3)")
    if hr.ndim != 1 or not len(hr):
        raise ValueError("Official HR label must have shape (N,)")
    if activity.ndim != 2 or activity.shape[1] != 1 or not len(activity):
        raise ValueError("Activity must have shape (N,1)")
    if ecg.ndim != 2 or ecg.shape[1] != 1 or not len(ecg):
        raise ValueError("ECG must have shape (N,1)")
    if not np.isfinite(activity).all() or not np.equal(activity, np.floor(activity)).all():
        raise ValueError("Activity IDs must be finite integers")
    details = {"root_keys": sorted(raw), "wrist_keys": sorted(wrist),
               "chest_keys": sorted(raw["signal"]["chest"]),
               "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                          for name, array in (("BVP", bvp), ("ACC", acc), ("label", hr),
                                               ("activity", activity), ("ECG", ecg))}}
    # Only wrist arrays/official labels survive; do not retain large chest feature arrays.
    return SubjectRecord(subject, bvp[:, 0], acc, hr, activity[:, 0].astype(np.int64),
                         source_sha256, path, len(ecg), details)


def load_subject(subject: str, root: Path, source_info: dict) -> SubjectRecord:
    if subject not in SUBJECTS:
        raise ValueError("Subject must be S1-S15")
    path = root / subject / f"{subject}.pkl"
    if not source_info.get("sha256"):
        raise ValueError("Missing independent source-file SHA-256")
    actual = sha256_file(path)
    if actual != source_info["sha256"]:
        raise ValueError(f"Source hash mismatch: {subject}")
    with path.open("rb") as stream:
        raw = pickle.load(stream, encoding="latin1")
    return decode_subject(raw, subject, path, actual)
