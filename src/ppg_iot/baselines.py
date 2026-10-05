"""Independent final_32hz_v1 DSP; reference HR never enters these estimators."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.signal import find_peaks, windows


@dataclass(frozen=True)
class PredictionWithStatus:
    hr_pred_bpm: float | None
    prediction_valid: bool
    reason: str
    peak_count: int | None = None


def _signal(signal, fs: float) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Baseline input must be a finite 1D signal with >=2 samples")
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    return x


def fft_hr(signal, fs: float) -> PredictionWithStatus:
    x = _signal(signal, fs)
    if len(x) > 4096 or fs / 2 < 3.5:
        raise ValueError("FFT input/rate does not support the agreed search grid")
    weighted = (x - x.mean()) * windows.hann(len(x), sym=True)
    amplitude = np.abs(np.fft.rfft(weighted, n=4096))
    frequencies = np.fft.rfftfreq(4096, d=1 / fs)
    candidates = np.flatnonzero((frequencies >= 0.6) & (frequencies <= 3.5))
    if not np.isfinite(amplitude).all() or not len(candidates):
        raise ValueError("Non-finite FFT or empty search grid")
    # np.argmax chooses the first (lowest frequency) in an exact tie.
    index = candidates[np.argmax(amplitude[candidates])]
    reason = "low_signal" if amplitude[candidates].max() == 0 else "ok"
    return PredictionWithStatus(float(60 * frequencies[index]), True, reason)


def ibi_hr(signal, fs: float) -> PredictionWithStatus:
    x = _signal(signal, fs)
    distance = round(0.33 * fs)
    if distance < 1:
        raise ValueError("Sampling rate gives an invalid peak distance")
    peaks, _ = find_peaks(x, distance=distance, prominence=0.3 * x.std(ddof=0))
    count = len(peaks)
    if count < 2:
        return PredictionWithStatus(None, False, "insufficient_peaks", count)
    mean_ibi = np.diff(peaks).mean() / fs
    value = float(60 / mean_ibi)
    if not math.isfinite(value):
        raise ValueError("Non-finite IBI prediction")
    return PredictionWithStatus(value, True, "ok", count)


def apply_ibi_fallback(prediction: PredictionWithStatus, train_median_bpm: float) -> dict:
    if not math.isfinite(train_median_bpm) or train_median_bpm <= 0:
        raise ValueError("Fallback must be a frozen positive finite Train HR median")
    return {"hr_pred_bpm": (prediction.hr_pred_bpm if prediction.prediction_valid
                             else float(train_median_bpm)),
            "prediction_valid": True, "ibi_valid": prediction.prediction_valid,
            "used_fallback": not prediction.prediction_valid,
            "reason": prediction.reason, "peak_count": prediction.peak_count}
