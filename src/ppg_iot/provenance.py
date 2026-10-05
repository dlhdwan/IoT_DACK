"""Content-addressed experiment records, with no dependency on Git commits."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any
import uuid


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def hash_object(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Atomic replacement, rejecting NaN/Infinity rather than invalid JSON."""
    destination = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False).encode("utf-8") + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary, destination)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    return {"path": str(source), "sha256": sha256_file(source),
            "bytes": source.stat().st_size}


def code_snapshot(root: Path) -> dict[str, Any]:
    sources = {}
    for folder in ("src", "scripts"):
        for path in sorted((root / folder).rglob("*.py")):
            sources[path.relative_to(root).as_posix()] = sha256_file(path)
    commit = None
    try:
        process = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root),
             "rev-parse", "--verify", "HEAD"], capture_output=True, text=True, timeout=5)
        if process.returncode == 0:
            commit = process.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"files": sources, "sha256": hash_object(sources), "git_commit": commit}


def runtime_snapshot() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "scipy", "pandas", "torch", "matplotlib", "scikit-learn", "tqdm"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    result = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "machine": platform.machine(),
              "processor": platform.processor(), "packages": packages}
    # Record a loaded runtime without importing ML libraries for stdlib commands.
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        result["quantized_engines"] = list(torch.backends.quantized.supported_engines)
        result["quantized_engine"] = torch.backends.quantized.engine
        result["torch_threads"] = torch.get_num_threads()
    return result


class RunRecorder:
    """A running record can end once, with actual file evidence or an error."""

    def __init__(self, root: Path, out: Path, config: dict, *, ecode: str,
                 kind: str, seed: int | None = None,
                 inputs: dict[str, str | Path] | None = None,
                 parent_checkpoint: str | Path | None = None) -> None:
        if kind not in ("smoke", "pilot", "final"):
            raise ValueError("Run kind must be smoke, pilot or final")
        if seed is not None and (type(seed) is not int or seed not in (42, 123, 999)):
            raise ValueError("Unknown experiment seed")
        sources = {name: fingerprint(path) for name, path in (inputs or {}).items()}
        parent = fingerprint(parent_checkpoint) if parent_checkpoint is not None else None
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:12]}"
        self.directory = out / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.record = {
            "schema_version": 1, "run_id": run_id, "ecode": ecode, "seed": seed,
            "kind": kind, "status": "running", "started_at": now_utc(), "ended_at": None,
            "protocol_version": config["run"]["protocol_version"],
            "config_sha256": hash_object(config), "source_config_manifest_stats": sources,
            "code": code_snapshot(root), "runtime": runtime_snapshot(),
            "parent_checkpoint": parent, "evidence": {}, "error": None,
        }
        write_json(self.directory / "resolved_config.json", config)
        self._save()

    def _save(self) -> None:
        write_json(self.directory / "run.json", self.record)

    def finish(self, *, evidence: dict[str, str | Path] | None = None,
               error: str | None = None) -> None:
        if self.record["status"] != "running":
            raise RuntimeError("Run has already ended")
        if error is None and not evidence:
            raise ValueError("A complete run requires actual file evidence")
        observed = {name: fingerprint(path) for name, path in (evidence or {}).items()}
        self.record.update(status="failed" if error is not None else "complete",
                           ended_at=now_utc(), evidence=observed, error=error,
                           runtime=runtime_snapshot())
        self._save()
