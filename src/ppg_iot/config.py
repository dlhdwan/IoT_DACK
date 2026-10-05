"""Validate the agreed protocol before any experiment starts (stdlib only)."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import tomllib
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SUBJECTS = tuple(f"S{i}" for i in range(1, 16))
SPLITS = {"train": SUBJECTS[:10], "validation": SUBJECTS[10:12], "test": SUBJECTS[12:]}
MODEL = {
    "name": "dilated_1d_window_network", "conv_channels": [32, 64, 64],
    "kernels": [7, 5, 3], "dilations": [1, 2, 4], "paddings": [3, 4, 4],
    "conv_bias": False, "batch_norm": True, "pool_after_conv": [1, 2],
    "pool_kernel": 2, "pool_stride": 2, "head_hidden": 32,
    "head_activation": "relu", "output_channels": 1,
}


class ProtocolError(ValueError):
    """The supplied configuration would change the agreed experiment."""


def project_path(value: str | Path, root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_protocol(config: dict[str, Any]) -> None:
    errors: list[str] = []

    def exact(section: str, name: str, expected: Any) -> None:
        actual = config.get(section, {}).get(name)
        # bool must not masquerade as an integer, e.g. calibration_windows=True.
        if actual != expected or type(actual) is not type(expected):
            errors.append(f"{section}.{name}: expected {expected!r}, got {actual!r}")

    for key, subjects in SPLITS.items():
        exact("split", key, list(subjects))
    configured = [s for key in SPLITS for s in config.get("split", {}).get(key, [])]
    if len(configured) != len(set(configured)):
        errors.append("split: subjects overlap or are duplicated")
    fixed = {
        "project": {"dataset": "PPG-DaLiA", "seeds": [42, 123, 999],
                    "target_macro_mae_bpm": 6.0},
        "run": {"schema_version": 1, "protocol_version": "final_32hz_v1"},
        "split": {"unit": "subject"},
        "signal": {"ppg_native_hz": 64, "acc_native_hz": 32, "model_hz": 32,
                   "window_seconds": 8, "step_seconds": 2,
                   "channels": ["PPG", "ACCx", "ACCy", "ACCz"],
                   "normalization_epsilon": 1e-8, "normalization_ddof": 0,
                   "ppg_normalization": "per_window_zscore",
                   "acc_normalization": "train_continuous_per_axis_zscore",
                   "activity_min_fraction": 0.8},
        "filter": {"family": "butterworth", "prototype_order": 2,
                   "band_hz": [0.5, 4.0], "representation": "sos",
                   "mode": "sosfiltfilt_continuous_per_subject_offline",
                   "resample_up": 1, "resample_down": 2},
        "baseline": {"protocol": "final_32hz_v1", "fft_window": "hann",
                     "fft_hann_symmetric": True, "fft_mean_center": True,
                     "fft_points": 4096, "fft_search_hz": [0.6, 3.5],
                     "ibi_min_distance_seconds": 0.33,
                     "ibi_prominence_std_factor": 0.3, "ibi_aggregation": "mean",
                     "ibi_fallback": "train_hr_median",
                     "benchmark_max_relative_deviation": 0.15},
        "model": MODEL,
        "quantization": {"method": "post_training_static", "calibration_split": "train",
                         "calibration_windows": 1000, "calibration_seed": 42,
                         "shared_calibration_manifest": True, "calibration_batch_size": 32},
        "training": {"optimizer": "AdamW", "loss": "L1_BPM", "batch_size": 64,
                     "max_epochs": 80, "patience": 10, "scheduler": "none",
                     "selection_metric": "validation_macro_subject_mae",
                     "min_delta": 0.0, "tie_policy": "earlier_epoch", "drop_last": False},
        "evaluation": {"primary_aggregation": "macro_subject",
                       "secondary_aggregation": "pooled_window",
                       "large_error_threshold_bpm": 10.0, "seed_std_ddof": 1,
                       "cpu_batch_size": 1, "cpu_threads": 1},
        "targets": {"fp32_artifact_max_bytes_exclusive": 150000,
                    "minimum_compression_ratio": 2.5, "maximum_macro_mae_increase_bpm": 0.5},
    }
    for section, fields in fixed.items():
        if not isinstance(config.get(section), dict):
            errors.append(f"{section}: missing or invalid section")
            continue
        for name, expected in fields.items():
            exact(section, name, expected)
    for name, minimum in (("warmup_iterations", 100), ("measurement_iterations", 1000)):
        value = config.get("evaluation", {}).get(name)
        if type(value) is not int or value < minimum:
            errors.append(f"evaluation.{name}: must be an integer >= {minimum}")
    for name in ("learning_rate", "weight_decay"):
        value = config.get("training", {}).get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            errors.append(f"training.{name}: must be finite and nonnegative")
        elif name == "learning_rate" and value == 0:
            errors.append("training.learning_rate: must be positive")
    backend = config.get("quantization", {}).get("backend")
    if backend not in ("onednn", "x86", "fbgemm", "qnnpack"):
        errors.append("quantization.backend: unsupported backend name")
    if errors:
        raise ProtocolError("\n".join(errors))


def load_config(path: str | Path = "configs/default.toml", *, root: Path = ROOT) -> dict[str, Any]:
    with project_path(path, root).open("rb") as stream:
        config = tomllib.load(stream)
    validate_protocol(config)
    return deepcopy(config)
