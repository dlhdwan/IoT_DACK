"""Continuous per-subject filtering and continuous Train-only population moments."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from ppg_iot.config import SPLITS
from ppg_iot.provenance import hash_object
from .schema import SubjectRecord


def preflight(record: SubjectRecord) -> None:
    if not np.isfinite(record.ppg64).all() or not np.isfinite(record.acc32).all():
        raise ValueError(f"{record.subject_id}: non-finite continuous signal before filtering")


def filter_continuous(record: SubjectRecord, config: dict) -> np.ndarray:
    preflight(record)
    settings = config["filter"]
    sos = butter(settings["prototype_order"], settings["band_hz"], btype="bandpass",
                 fs=record.ppg_hz, output="sos")
    filtered = sosfiltfilt(sos, record.ppg64, padtype="odd", padlen=None)
    result = resample_poly(filtered, up=1, down=2, window=("kaiser", 5.0),
                           padtype="constant")
    if len(result) != len(record.acc32) or not np.isfinite(result).all():
        raise ValueError(f"{record.subject_id}: resampling output domain/values are invalid")
    return result


def fit_acc_statistics(records: dict[str, SubjectRecord], epsilon: float = 1e-8) -> dict:
    count, mean, moment = 0, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    sources = []
    # Merge population moments; every continuous sample contributes exactly once.
    for subject in SPLITS["train"]:
        if subject not in records:
            continue
        record = records[subject]
        if record.subject_id != subject:
            raise ValueError("Record dictionary has a mismatched subject key")
        preflight(record)
        x = record.acc32.astype(np.float64, copy=False)
        n, current_mean = len(x), x.mean(axis=0)
        current_moment = np.square(x - current_mean).sum(axis=0)
        delta = current_mean - mean
        total = count + n
        moment += current_moment + delta**2 * count * n / total
        mean += delta * n / total
        count = total
        sources.append({"subject_id": subject, "n_continuous_samples": n,
                        "source_sha256": record.source_sha256})
    if not count:
        raise ValueError("No continuous Train samples for ACC normalization")
    std = np.sqrt(moment / count)
    result = {"schema_version": 1, "fit_split": "train", "domain": "continuous_aligned_record",
              "ddof": 0, "epsilon": epsilon, "n_samples_per_axis": count,
              "mean": mean.tolist(), "std": std.tolist(),
              "denominator": np.maximum(std, epsilon).tolist(), "subjects": sources}
    result["sha256"] = hash_object(result)
    return result


def normalize_acc(acc: np.ndarray, stats: dict) -> np.ndarray:
    unhashed = {key: value for key, value in stats.items() if key != "sha256"}
    if stats.get("sha256") != hash_object(unhashed) or stats.get("fit_split") != "train":
        raise ValueError("ACC statistics are stale or do not come from Train")
    result = (acc - np.asarray(stats["mean"])) / np.asarray(stats["denominator"])
    if not np.isfinite(result).all():
        raise ValueError("Non-finite normalized ACC")
    return result.astype(np.float32)


def normalize_ppg(window: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    if window.ndim != 1 or len(window) != 256 or not np.isfinite(window).all():
        raise ValueError("PPG normalization requires one finite 256-sample window")
    return ((window - window.mean()) / max(float(window.std(ddof=0)), epsilon)).astype(np.float32)
