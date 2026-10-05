"""Strict common-manifest joins and subject-first metrics with explicit coverage."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ppg_iot.provenance import hash_object

ACTIVITIES = {1: "sitting", 2: "stairs", 3: "table_soccer", 4: "cycling",
              5: "driving", 6: "lunch", 7: "walking", 8: "working"}


def _unique(frame: pd.DataFrame, name: str) -> None:
    if "window_id" not in frame or frame.window_id.isna().any() or frame.window_id.duplicated().any():
        raise ValueError(f"{name}: missing/null/duplicate window_id")


def _boolean(series: pd.Series, name: str) -> None:
    if series.isna().any() or not pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError(f"{name}: expected non-null boolean dtype")


def join_predictions(manifest: pd.DataFrame, predictions: pd.DataFrame, *,
                     allow_invalid: bool = False) -> pd.DataFrame:
    required = {"window_id", "subject_id", "split", "hr_ref_bpm", "activity_id",
                "activity_status", "eligible"}
    if not required.issubset(manifest):
        raise ValueError(f"Manifest missing fields: {required - set(manifest)}")
    if not {"window_id", "hr_ref_bpm", "hr_pred_bpm", "prediction_valid"}.issubset(predictions):
        raise ValueError("Predictions must include ID, unchanged HR reference, value and validity")
    _unique(manifest, "manifest")
    _unique(predictions, "predictions")
    _boolean(manifest.eligible, "eligible")
    _boolean(predictions.prediction_valid, "prediction_valid")
    domain = manifest.loc[manifest.eligible].copy()
    if set(domain.window_id) != set(predictions.window_id):
        missing = set(domain.window_id) - set(predictions.window_id)
        extra = set(predictions.window_id) - set(domain.window_id)
        raise ValueError(f"Prediction ID population mismatch: {len(missing)} missing, {len(extra)} extra")
    ordered = predictions.set_index("window_id").loc[domain.window_id].reset_index()
    reference = domain.hr_ref_bpm.to_numpy(dtype=float)
    if not np.isfinite(reference).all() or (reference <= 0).any():
        raise ValueError("Eligible reference HR must be finite and positive")
    if not np.array_equal(reference, ordered.hr_ref_bpm.to_numpy(dtype=float)):
        raise ValueError("Prediction reference HR differs from the common manifest")
    if "preprocess_hash" in domain and "preprocess_hash" not in ordered:
        raise ValueError("Predictions must identify their preprocessing version")
    for name in ("subject_id", "split", "preprocess_hash"):
        if name in ordered and not np.array_equal(domain[name].to_numpy(), ordered[name].to_numpy()):
            raise ValueError(f"Prediction {name} differs from the common manifest")
    values = ordered.hr_pred_bpm.to_numpy(dtype=float, na_value=np.nan)
    valid = ordered.prediction_valid.to_numpy(dtype=bool)
    if not np.isfinite(values[valid]).all():
        raise ValueError("Valid predictions contain a non-finite output")
    if not allow_invalid and not valid.all():
        raise ValueError("Full-domain method failed; do not drop failed windows")
    if allow_invalid and not np.isnan(values[~valid]).all():
        raise ValueError("Pure invalid predictions must have an absent/NaN value")
    # Metadata always comes from the manifest, never from prediction-side copies.
    extra_columns = [column for column in ordered if column not in domain]
    for column in extra_columns:
        domain[column] = ordered[column].to_numpy()
    return domain.reset_index(drop=True)


def train_median_fallback(manifest: pd.DataFrame) -> dict:
    _unique(manifest, "manifest")
    _boolean(manifest.eligible, "eligible")
    train = manifest.loc[manifest.eligible & (manifest.split == "train")]
    values = train.hr_ref_bpm.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Cannot fit fallback without valid Train reference HR")
    return {"method": "train_hr_median", "hr_bpm": float(np.median(values)),
            "n_train_windows": len(train),
            "window_ids_sha256": hash_object(sorted(train.window_id.tolist())),
            "train_hr_sha256": hash_object(train[["window_id", "hr_ref_bpm"]]
                                            .sort_values("window_id").to_dict("records"))}


def _basic(frame: pd.DataFrame) -> dict:
    n_total = len(frame)
    valid = frame.loc[frame.prediction_valid]
    n_valid = len(valid)
    result = {"n_total": n_total, "n_valid": n_valid,
              "coverage": n_valid / n_total if n_total else None,
              "mae_bpm": None, "rmse_bpm": None, "error_gt_10_percent": None}
    if n_valid:
        error = valid.hr_pred_bpm.to_numpy(dtype=float) - valid.hr_ref_bpm.to_numpy(dtype=float)
        if not np.isfinite(error).all():
            raise ValueError("Non-finite residual in metric domain")
        result.update(mae_bpm=float(np.abs(error).mean()),
                      rmse_bpm=float(np.sqrt(np.square(error).mean())),
                      error_gt_10_percent=float(100 * (np.abs(error) > 10).mean()))
    return result


def _macro(frame: pd.DataFrame) -> dict:
    rows = [_basic(group) for _, group in frame.groupby("subject_id", sort=True)]
    observed = [row for row in rows if row["n_valid"]]
    return {"n_subjects_with_data": len(observed),
            **{key: float(np.mean([row[key] for row in observed])) if observed else None
               for key in ("mae_bpm", "rmse_bpm", "error_gt_10_percent")}}


def compute_metrics(joined: pd.DataFrame, *, expected_subjects: list[str] | None = None) -> dict:
    _unique(joined, "joined predictions")
    _boolean(joined.prediction_valid, "prediction_valid")
    subjects = expected_subjects if expected_subjects is not None else sorted(
        joined.subject_id.unique(), key=lambda x: int(x[1:]))
    if not set(joined.subject_id).issubset(subjects):
        raise ValueError("Observed subject is outside expected subject table")
    by_subject = []
    for subject in subjects:
        group = joined.loc[joined.subject_id == subject]
        splits = group.split.unique().tolist()
        if len(splits) > 1:
            raise ValueError(f"Subject {subject} crosses splits")
        by_subject.append({"subject_id": subject, "split": splits[0] if splits else None,
                           **_basic(group)})
    by_activity = []
    for activity, name in ACTIVITIES.items():
        group = joined.loc[(joined.activity_status == "assigned") & (joined.activity_id == activity)]
        by_activity.append({"activity_id": activity, "activity": name,
                            "pooled_window": _basic(group), "macro_subject": _macro(group),
                            "by_subject": [
                                {"subject_id": subject,
                                 **_basic(group.loc[group.subject_id == subject])}
                                for subject in subjects]})
    for status in ("mixed_transition", "unknown"):
        group = joined.loc[joined.activity_status == status]
        by_activity.append({"activity_id": None, "activity": status,
                            "pooled_window": _basic(group), "macro_subject": _macro(group),
                            "by_subject": [
                                {"subject_id": subject,
                                 **_basic(group.loc[group.subject_id == subject])}
                                for subject in subjects]})
    return {"schema_version": 1, "primary_aggregation": "macro_subject",
            "secondary_aggregation": "pooled_window", "large_error_threshold_bpm": 10.0,
            "overall": {"macro_subject": _macro(joined), "pooled_window": _basic(joined)},
            "by_subject": by_subject, "by_activity": by_activity}


def paired_subset_metrics(methods: dict[str, pd.DataFrame]) -> dict:
    if not methods:
        raise ValueError("No methods to compare")
    first = next(iter(methods.values()))
    _unique(first, "paired domain")
    ordered_ids = first.window_id.tolist()
    aligned = {}
    mask = np.ones(len(first), dtype=bool)
    for name, joined in methods.items():
        _unique(joined, name)
        if set(joined.window_id) != set(ordered_ids):
            raise ValueError("Methods do not share the full manifest population")
        current = joined.set_index("window_id").loc[ordered_ids].reset_index()
        if not np.array_equal(first.hr_ref_bpm.to_numpy(), current.hr_ref_bpm.to_numpy()):
            raise ValueError("Paired methods have different HR references")
        mask &= current.prediction_valid.to_numpy(dtype=bool)
        aligned[name] = current
    ids = [value for value, selected in zip(ordered_ids, mask) if selected]
    return {"scope": "paired-valid subset only; primary results retain full eligible domain",
            "n_total": len(first), "n_paired": len(ids),
            "coverage": len(ids) / len(first) if len(first) else None,
            "window_ids_sha256": hash_object(ids),
            "methods": {name: compute_metrics(frame.loc[mask].reset_index(drop=True))
                        for name, frame in aligned.items()}}


def summarize_seeds(values: dict[int, float]) -> dict:
    if set(values) != {42, 123, 999}:
        raise ValueError("Final seed statistics require 42, 123, 999")
    data = np.asarray([values[seed] for seed in (42, 123, 999)], dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Seed metrics must be finite")
    return {"seeds": [42, 123, 999], "values": data.tolist(),
            "mean": float(data.mean()), "std": float(data.std(ddof=1)), "std_ddof": 1}
