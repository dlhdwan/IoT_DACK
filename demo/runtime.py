"""Load notebook artifacts and perform new CPU inference, without training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.ao.quantization import convert, get_default_qconfig, prepare

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
from ppg_iot.model import WindowNetwork
from ppg_iot.metrics import ACTIVITIES
from demo.paths import artifact_root

ROOT = artifact_root()


class DemoError(ValueError):
    """An artifact or window does not satisfy the notebook's data contract."""


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise DemoError(f"Artifact is missing or outside the project: {relative}")
    return path


@dataclass(frozen=True)
class Window:
    window_id: str
    subject_id: str
    start_s: float
    end_s: float
    activity_id: int | None
    activity_name: str
    hr_ref_bpm: float
    preprocess_hash: str
    features: np.ndarray
    split: str = "validation"


@dataclass(frozen=True)
class ModelInfo:
    ecode: str
    seed: int
    artifact: str
    artifact_sha256: str
    artifact_bytes: int
    backend: str
    quantized: bool
    preprocess_hash: str


@dataclass
class LoadedModel:
    info: ModelInfo
    network: nn.Module


@dataclass(frozen=True)
class Reading:
    window_id: str
    subject_id: str
    start_s: float
    activity_id: int | None
    activity_name: str
    hr_ref_bpm: float
    predicted_bpm: float
    absolute_error_bpm: float
    cpu_call_ms: float
    ecode: str
    seed: int
    artifact_sha256: str
    preprocess_hash: str
    split: str = "validation"
    prediction_origin: str = "fresh_cpu_inference"

    def as_record(self) -> dict:
        return asdict(self)


def validate_windows(records: list, frame: pd.DataFrame, preprocess_hash: str) -> list[Window]:
    """Join metadata to a verified manifest; never consume stored predictions."""
    ids = [record["window_id"] for record in records]
    if not ids or len(set(ids)) != len(ids) or frame.window_id.duplicated().any():
        raise DemoError("Empty or duplicate replay window IDs")
    lookup = frame.set_index("window_id")
    if not set(ids).issubset(lookup.index):
        raise DemoError("Replay contains windows outside the locked manifest")
    windows = []
    for record in records:
        row = lookup.loc[record["window_id"]]
        if row.split != "validation" or row.subject_id not in ("S11", "S12") or not row.eligible:
            raise DemoError("Demo development only accepts eligible Validation windows")
        if record["subject_id"] != row.subject_id or float(record["start_s"]) != row.start_s:
            raise DemoError("Replay subject/time differs from the manifest")
        reference = float(record["hr_ref_bpm"])
        if not np.isfinite(reference) or not np.isclose(reference, row.hr_ref_bpm, rtol=0, atol=1e-10):
            raise DemoError("Replay HR reference differs from the manifest")
        activity = None if pd.isna(row.activity_id) else int(row.activity_id)
        if record["activity_id"] != activity or row.end_s - row.start_s != 8:
            raise DemoError("Replay activity or window duration differs from the manifest")
        ppg = np.asarray(record["ppg"], dtype=np.float32)
        acc = np.asarray(record["acc"], dtype=np.float32)
        if ppg.shape != (256,) or acc.shape != (3, 256):
            raise DemoError("Expected 256 PPG samples and three ACC axes at 32 Hz")
        features = np.concatenate((ppg[None, :], acc), axis=0)
        if not np.isfinite(features).all():
            raise DemoError("Replay waveform contains non-finite samples")
        features.setflags(write=False)
        windows.append(Window(record["window_id"], row.subject_id, float(row.start_s), float(row.end_s),
                              activity, ACTIVITIES.get(activity, "mixed_unknown"), reference,
                              preprocess_hash, features))
    return windows


def load_windows(root: Path = ROOT) -> list[Window]:
    root = Path(root)
    index = read_json(root / "results/demo/index.json")
    lock = read_json(root / "results/protocol_lock.json")
    if not index.get("passed") or index.get("split") != "validation":
        raise DemoError("Expected a verified Validation export from the notebook")
    if index["preprocess_hash"] != lock["preprocess_hash"]:
        raise DemoError("Replay preprocessing differs from the locked protocol")
    manifest = resolve_file(root, "results/manifests/windows.csv")
    if digest(manifest) != lock["files"]["results/manifests/windows.csv"]:
        raise DemoError("Locked manifest checksum mismatch")
    frame = pd.read_csv(manifest, float_precision="round_trip")
    records = read_json(resolve_file(root, index["data"]))
    windows = validate_windows(records, frame, index["preprocess_hash"])
    if len(windows) != index["n_windows"]:
        raise DemoError("Replay window count differs from its export index")
    return windows


def model_entry(root: Path, ecode: str, seed: int) -> dict:
    if ecode not in ("E3", "E4") or seed not in (42, 123, 999):
        raise DemoError("Use E3/E4 and a trained seed 42, 123 or 999")
    training = read_json(root / "results/training_index.json")
    quantized = read_json(root / "results/ptq_index.json")
    lock = read_json(root / "results/protocol_lock.json")
    for relative in ("results/training_index.json", "results/ptq_index.json"):
        if digest(resolve_file(root, relative)) != lock["files"][relative]:
            raise DemoError("Model registry checksum differs from the locked research experiment")
    if not training.get("passed") or not quantized.get("passed"):
        raise DemoError("Training/PTQ registry is not verified")
    parent = training["runs"][f"E3_{seed}"]
    entry = parent if ecode == "E3" else quantized["runs"][f"E4_{seed}"]
    if not entry.get("passed") or entry["seed"] != seed or entry["ecode"] != ecode:
        raise DemoError("Model registry identity mismatch")
    if entry["preprocess_hash"] != lock["preprocess_hash"] or parent["preprocess_hash"] != entry["preprocess_hash"]:
        raise DemoError("Model preprocessing mismatch")
    parent_path = resolve_file(root, parent["artifact"])
    if digest(parent_path) != parent["artifact_sha256"]:
        raise DemoError("FP32 parent checksum mismatch")
    if ecode == "E4":
        if entry["source_sha256"] != parent["artifact_sha256"] or entry["source_artifact"] != parent["artifact"]:
            raise DemoError("INT8 artifact belongs to a different FP32 checkpoint")
        if not quantized["calibration"].get("passed") or quantized["calibration"]["split"] != "train":
            raise DemoError("PTQ calibration provenance is invalid")
    artifact = resolve_file(root, entry["artifact"])
    if digest(artifact) != entry["artifact_sha256"] or artifact.stat().st_size != entry["artifact_bytes"]:
        raise DemoError("Model artifact checksum/byte count mismatch")
    return entry


def load_model(root: Path = ROOT, ecode: str = "E3", seed: int = 42) -> LoadedModel:
    """Reconstruct operator topology and load all weights and calibrated qparams."""
    root = Path(root)
    if torch.__version__.split("+")[0] != "2.10.0":
        raise DemoError("Notebook artifacts require PyTorch 2.10.0")
    entry = model_entry(root, ecode, seed)
    packet = torch.load(root / entry["artifact"], map_location="cpu", weights_only=True)
    if packet.get("format") != "ppg-window-state-v1" or packet["input_channels"] != 4:
        raise DemoError("Unsupported notebook model format/channels")
    if packet["quantized"] != (ecode == "E4") or packet["backend"] != "onednn":
        raise DemoError("Artifact runtime/type differs from the research protocol")
    torch.set_num_threads(1)
    network = WindowNetwork(4).eval()
    if packet["quantized"]:
        if packet["backend"] not in torch.backends.quantized.supported_engines:
            raise DemoError("This CPU runtime does not support the notebook's oneDNN INT8 artifact")
        torch.backends.quantized.engine = packet["backend"]
        network.fuse_for_inference()
        network.qconfig = get_default_qconfig(packet["backend"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepare(network, inplace=True)
            convert(network, inplace=True)
        # No calibration here: strict loading supplies the original calibrated qparams.
    network.load_state_dict(packet["state_dict"], strict=True)
    info = ModelInfo(ecode, seed, entry["artifact"], entry["artifact_sha256"], entry["artifact_bytes"],
                     packet["backend"], packet["quantized"], entry["preprocess_hash"])
    return LoadedModel(info, network.eval())


def infer(model: LoadedModel, window: Window) -> Reading:
    if window.preprocess_hash != model.info.preprocess_hash or window.features.shape != (4, 256):
        raise DemoError("Input waveform does not match the model preprocessing contract")
    inputs = torch.from_numpy(np.array(window.features, dtype=np.float32, copy=True))[None, ...]
    with torch.inference_mode():
        start = time.perf_counter_ns()
        output = model.network(inputs)
        duration = (time.perf_counter_ns() - start) / 1e6
    if output.shape != (1, 1) or not torch.isfinite(output).all():
        raise DemoError("Model returned an invalid BPM output")
    bpm = float(output.item())
    return Reading(window.window_id, window.subject_id, window.start_s, window.activity_id,
                   window.activity_name, window.hr_ref_bpm, bpm, abs(bpm - window.hr_ref_bpm),
                   duration, model.info.ecode, model.info.seed, model.info.artifact_sha256,
                   model.info.preprocess_hash, split=window.split)
