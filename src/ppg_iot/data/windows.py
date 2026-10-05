"""One common, data-only eligibility manifest, including transitions/exclusions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ppg_iot.config import SPLITS
from .alignment import label_interval, sample_bounds, validate_alignment
from .schema import SubjectRecord


def assign_activity(labels: np.ndarray, start_s: float, end_s: float, *,
                    fs: float = 4, min_fraction: float = 0.8) -> dict:
    if not 0 <= start_s < end_s <= len(labels) / fs or fs <= 0:
        raise ValueError("Activity interval is outside its time domain")
    begin, stop = int(np.floor(start_s * fs)), int(np.ceil(end_s * fs))
    indices = np.arange(begin, stop)
    overlap = np.maximum(0, np.minimum((indices + 1) / fs, end_s)
                         - np.maximum(indices / fs, start_s))
    codes = np.unique(labels[begin:stop])
    durations = np.array([overlap[labels[begin:stop] == code].sum() for code in codes])
    if not np.isclose(durations.sum(), end_s - start_s, rtol=0, atol=1e-9):
        raise ValueError("Activity timeline does not cover the window")
    # np.unique sorts codes, so a duration tie picks the smallest code.
    dominant = int(codes[np.argmax(durations)])
    fraction = float(durations.max() / (end_s - start_s))
    if dominant not in range(0, 9):
        status, activity = "unknown", None
    elif dominant in range(1, 9) and fraction >= min_fraction:
        status, activity = "assigned", dominant
    else:
        status, activity = "mixed_transition", None
    return {"activity_id": activity, "activity_fraction": fraction,
            "activity_status": status, "dominant_activity_code": dominant}


def build_manifest(record: SubjectRecord, ppg32: np.ndarray, *,
                   preprocess_hash: str, protocol_version: str,
                   activity_min_fraction: float = 0.8) -> pd.DataFrame:
    validate_alignment(record)
    if len(ppg32) != len(record.acc32) or not np.isfinite(ppg32).all():
        raise ValueError("Preprocessed PPG must match the complete aligned ACC domain")
    split = next(name for name, ids in SPLITS.items() if record.subject_id in ids)
    rows = []
    for index, reference in enumerate(record.hr_labels):
        start, end = label_interval(record, index)
        p0, p1 = sample_bounds(record, index, 32)
        activity = assign_activity(record.activity_labels, start, end,
                                   min_fraction=activity_min_fraction)
        reason = ("nonfinite_hr_reference" if not np.isfinite(reference) else
                  "nonpositive_hr_reference" if reference <= 0 else "")
        rows.append({"window_id": f"{protocol_version}:{record.subject_id}:{index:06d}",
                     "subject_id": record.subject_id, "split": split, "label_index": index,
                     "start_s": start, "end_s": end, "ppg_start_idx": p0, "ppg_end_idx": p1,
                     "acc_start_idx": p0, "acc_end_idx": p1,
                     "ppg_native_start_idx": 2 * p0, "ppg_native_end_idx": 2 * p1,
                     "hr_ref_bpm": float(reference), **activity,
                     "eligible": not reason, "exclusion_reason": reason,
                     "ppg_flat": bool(ppg32[p0:p1].std(ddof=0) == 0),
                     "source_sha256": record.source_sha256, "preprocess_hash": preprocess_hash})
    return pd.DataFrame(rows)


def window_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, split), frame in manifest.groupby(["subject_id", "split"], sort=False):
        rows.append({"subject_id": subject, "split": split, "group": "all",
                     "n_candidates": len(frame), "n_eligible": int(frame.eligible.sum()),
                     "n_excluded": int((~frame.eligible).sum())})
        for activity in range(1, 9):
            group = frame.loc[(frame.activity_status == "assigned") & (frame.activity_id == activity)]
            rows.append({"subject_id": subject, "split": split, "group": f"activity_{activity}",
                         "n_candidates": len(group), "n_eligible": int(group.eligible.sum()),
                         "n_excluded": int((~group.eligible).sum())})
        for status in ("mixed_transition", "unknown"):
            group = frame.loc[frame.activity_status == status]
            rows.append({"subject_id": subject, "split": split, "group": status,
                         "n_candidates": len(group), "n_eligible": int(group.eligible.sum()),
                         "n_excluded": int((~group.eligible).sum())})
    return pd.DataFrame(rows)
