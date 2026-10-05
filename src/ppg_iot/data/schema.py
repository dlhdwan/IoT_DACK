"""Validated synchronized-record schema; reference/metadata never become features."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    ppg64: np.ndarray
    acc32: np.ndarray
    hr_labels: np.ndarray
    activity_labels: np.ndarray
    source_sha256: str
    source_path: Path
    ecg_sample_count: int
    schema_details: dict = field(default_factory=dict)
    ppg_hz: int = 64
    acc_hz: int = 32
    activity_hz: int = 4
    ecg_hz: int = 700
    ppg_start_s: float = 0.0
    acc_start_s: float = 0.0
    activity_start_s: float = 0.0
    label_start_s: float = 0.0
