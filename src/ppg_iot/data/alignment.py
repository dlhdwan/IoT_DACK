"""README I.2/III.3 timeline, checked against all four synchronized domains."""
from __future__ import annotations
import numpy as np
from .schema import SubjectRecord

README_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00495/readme.pdf"


def label_interval(record: SubjectRecord, index: int) -> tuple[float, float]:
    if type(index) is not int or not 0 <= index < len(record.hr_labels):
        raise ValueError("Label index outside reference domain")
    start = record.label_start_s + 2 * index
    return float(start), float(start + 8)


def sample_bounds(record: SubjectRecord, index: int, fs: int) -> tuple[int, int]:
    start, end = label_interval(record, index)
    return round(start * fs), round(end * fs)


def validate_alignment(record: SubjectRecord) -> dict:
    if (record.ppg_hz, record.acc_hz, record.activity_hz, record.ecg_hz) != (64, 32, 4, 700):
        raise ValueError("Sampling rates differ from the official synchronized schema")
    if any(value != 0 for value in (record.ppg_start_s, record.acc_start_s,
                                   record.activity_start_s, record.label_start_s)):
        raise ValueError("Unexpected relative time origin; do not shift labels to optimize MAE")
    ppg, acc, activity = len(record.ppg64), len(record.acc32), len(record.activity_labels)
    if ppg * 32 != acc * 64 or ppg * 4 != activity * 64:
        raise ValueError("Synchronized wrist/activity durations disagree; no min-length truncation")
    if ppg * 700 != record.ecg_sample_count * 64:
        raise ValueError("ECG reference domain disagrees with synchronized wrist duration")
    if ppg < 512:
        raise ValueError("Record shorter than one complete 8-second window")
    expected_count = (ppg - 512) // 128 + 1
    if len(record.hr_labels) != expected_count:
        raise ValueError(f"HR label count mismatch: {len(record.hr_labels)} vs {expected_count}")
    indices = sorted({0, expected_count // 2, expected_count - 1})
    checks = []
    for index in indices:
        start, end = label_interval(record, index)
        p0, p1 = sample_bounds(record, index, 64)
        a0, a1 = sample_bounds(record, index, 32)
        if p1 > ppg or a1 > acc or p1 - p0 != 512 or a1 - a0 != 256:
            raise ValueError("Aligned window is incomplete")
        value = record.hr_labels[index]
        checks.append({"label_index": index, "start_s": start, "end_s": end,
                       "ppg_bounds": [p0, p1], "acc_bounds": [a0, a1],
                       "hr_ref_bpm": float(value) if np.isfinite(value) else None})
    last_end_s = label_interval(record, expected_count - 1)[1]
    return {"subject_id": record.subject_id, "source_sha256": record.source_sha256,
            "label_count": expected_count, "expected_count": expected_count,
            "relative_origin_seconds": 0, "window_seconds": 8, "step_seconds": 2,
            "interval_convention": "[2*label_index, 2*label_index+8)",
            "activity_hz": 4, "signal_duration_seconds": ppg / 64,
            "unwindowed_tail_seconds": ppg / 64 - last_end_s, "checks": checks,
            "hr_definition": "official mean instantaneous ECG HR in each window",
            "source": {"url": README_URL, "sections": ["I.2", "II", "III.3"], "pages": [1, 2, 3, 4]},
            "passed": True}
