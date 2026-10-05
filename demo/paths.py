"""Select an artifact workspace independently of the demo's source/toolchain."""
import os
import json
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def artifact_root(value=None):
    choice = value if value is not None else os.environ.get("PPG_RUN_DIR")
    if not choice:
        pointer = SOURCE_ROOT / "configs/primary_run.json"
        if not pointer.is_file():
            raise FileNotFoundError("Missing configs/primary_run.json; select a completed run with PPG_RUN_DIR")
        choice = json.loads(pointer.read_text(encoding="utf-8"))["workspace"]
    selected = Path(choice).expanduser()
    return (selected if selected.is_absolute() else SOURCE_ROOT / selected).resolve()
