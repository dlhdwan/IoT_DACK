"""Read-only cached signals; construct (C,256), (1,), metadata separately."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ppg_iot.provenance import sha256_file
from .preprocessing import normalize_ppg


class WindowDataset(Dataset):
    def __init__(self, manifest: Path, processed_index: Path, *, split: str,
                 input_channels: int, verify_cache: bool = True):
        if split not in ("train", "validation", "test", "all") or input_channels not in (1, 4):
            raise ValueError("Invalid split/input channel contract")
        self.index = json.loads(processed_index.read_text(encoding="utf-8"))
        if not self.index.get("passed") or sha256_file(manifest) != self.index["manifest_sha256"]:
            raise ValueError("Unverified or stale manifest/cache index")
        frame = pd.read_csv(manifest)
        if frame.window_id.duplicated().any() or frame.eligible.dtype != bool:
            raise ValueError("Manifest IDs/eligibility are invalid")
        if set(frame.preprocess_hash) != {self.index["preprocess_hash"]}:
            raise ValueError("Manifest has a different preprocessing version")
        self.rows = frame.loc[frame.eligible & ((frame.split == split) if split != "all" else True)
                              ].reset_index(drop=True)
        self.root, self.input_channels = processed_index.parent, input_channels
        self.arrays = {}
        for subject in self.rows.subject_id.unique():
            entry = self.index["subjects"][subject]
            self.arrays[subject] = {}
            if set(self.rows.loc[self.rows.subject_id == subject, "source_sha256"]) != {entry["source_sha256"]}:
                raise ValueError("Manifest has different source-file hashes")
            for channel in ("ppg32", "acc32_normalized"):
                item = entry[channel]
                path = (self.root / item["file"]).resolve()
                if not path.is_relative_to(self.root.resolve()):
                    raise ValueError("Cache path escapes the preprocessing directory")
                if verify_cache and sha256_file(path) != item["sha256"]:
                    raise ValueError(f"Stale/corrupt cache: {subject}/{channel}")
                self.arrays[subject][channel] = np.load(path, mmap_mode="r", allow_pickle=False)

    def close(self) -> None:
        """Release read-only mappings explicitly, including on Windows."""
        for arrays in self.arrays.values():
            for array in arrays.values():
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
        self.arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        arrays = self.arrays[row.subject_id]
        p0, p1, a0, a1 = (int(row.ppg_start_idx), int(row.ppg_end_idx),
                         int(row.acc_start_idx), int(row.acc_end_idx))
        ppg = normalize_ppg(np.asarray(arrays["ppg32"][p0:p1]))
        if self.input_channels == 1:
            features = ppg[None, :]
        else:
            acc = np.asarray(arrays["acc32_normalized"][a0:a1])
            if acc.shape != (256, 3):
                raise ValueError("Incomplete cached ACC window")
            features = np.concatenate([ppg[None, :], acc.T], axis=0).astype(np.float32)
        target = np.array([row.hr_ref_bpm], dtype=np.float32)
        if features.shape != (self.input_channels, 256) or not np.isfinite(features).all():
            raise ValueError("Non-finite/incomplete model features")
        if not np.isfinite(target).all():
            raise ValueError("Non-finite model target")
        return torch.from_numpy(features), torch.from_numpy(target), row.to_dict()


def collate_windows(items):
    features, targets, metadata = zip(*items)
    return torch.stack(features), torch.stack(targets), list(metadata)
