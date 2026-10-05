"""Replay real Validation predictions to an ESP32 firmware in Wokwi.

The trained PPG model has already produced BPM in the recorded replay artifact.
This script streams those values to firmware to demonstrate serial transport and
device display; inference remains on the host CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from demo.paths import artifact_root
ROOT = artifact_root()
DEMO = ROOT / "results" / "demo"
PROJECT = SOURCE_ROOT / "firmware" / "wokwi_display"


def load_replay() -> pd.DataFrame:
    """Load and validate the recorded Validation predictions."""
    index = json.loads((DEMO / "index.json").read_text(encoding="utf-8"))
    if not index.get("passed") or index.get("split") != "validation":
        raise RuntimeError("Expected a passed Validation replay artifact")
    frame = pd.read_csv(ROOT / index["csv"])
    required = {"window_id", "split", "activity_status", "hr_ref_bpm", "E3_bpm", "E4_bpm"}
    if not required.issubset(frame.columns) or frame.empty:
        raise RuntimeError("Replay CSV is missing required prediction/provenance columns")
    if set(frame["split"].dropna()) != {"validation"}:
        raise RuntimeError("Firmware replay accepts Validation windows only")
    if frame[["hr_ref_bpm", "E3_bpm", "E4_bpm"]].isna().any().any():
        raise RuntimeError("Replay contains a missing BPM value")
    return frame


def encode_packet(row: pd.Series, model: str) -> str:
    """Encode one provenance-tagged BPM result for the firmware UART parser."""
    bpm = float(row[f"{model}_bpm"])
    reference = float(row["hr_ref_bpm"])
    activity = str(row.get("activity_name", row.get("activity_status", "assigned"))).replace("|", "_")
    window_id = str(row["window_id"]).replace("|", "_")
    split = str(row.get("split", "validation"))
    if split not in ("test", "validation"):
        raise RuntimeError("Firmware packet must preserve Test/Validation provenance")
    packet = f"{window_id}|{split}|{activity}|{bpm:.2f}|{reference:.2f}"
    if "seed" in row:
        packet += f"|{model}|{int(row['seed'])}"
    return packet + "\n"


def show_preview(frame: pd.DataFrame, model: str) -> None:
    """Print actual replay rows and the serial packet shape."""
    print(f"Recorded replay: {len(frame)} Validation windows; model output: {model} on host")
    print("window_id | prediction BPM | ECG reference BPM | absolute error BPM")
    for _, row in frame.iterrows():
        bpm = float(row[f"{model}_bpm"])
        ref = float(row["hr_ref_bpm"])
        print(f"{row.window_id} | {bpm:.2f} | {ref:.2f} | {abs(bpm-ref):.2f}")
    print("\nUART sample:", encode_packet(frame.iloc[0], model).rstrip())


def wokwi_token() -> tuple[str, str]:
    """Read an existing token without printing or persisting its value."""
    token = os.environ.get("WOKWI_CLI_TOKEN", "").strip()
    source = "process"
    if not token and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                token = str(winreg.QueryValueEx(key, "WOKWI_CLI_TOKEN")[0]).strip()
                source = "Windows User"
        except FileNotFoundError:
            pass
    if not re.fullmatch(r"wok_[A-Za-z0-9_-]{40}", token):
        raise RuntimeError("Configure a valid WOKWI_CLI_TOKEN in the process or Windows User environment")
    return token, source


def redact_token(output: str, token: str) -> str:
    """Keep authentication values out of console output and saved logs."""
    return re.sub(r"wok_[A-Za-z0-9_-]{40}", "[REDACTED]", output.replace(token, "[REDACTED]"))


def make_scenario(frame: pd.DataFrame, model: str, folder: Path, screenshot: Path | None = None) -> tuple[Path, list[str]]:
    """Send each real row after boot and wait for its firmware acknowledgment."""
    steps = [{"wait-serial": "PPG-DaLiA recorded replay receiver ready."}]
    expected = []
    for _, row in frame.iterrows():
        packet = encode_packet(row, model)
        fields = packet.rstrip().split("|")
        ack = f"ACK|{fields[0]}|{fields[3]}|{fields[4]}"
        if len(fields) == 7:
            ack += f"|{fields[5]}|{fields[6]}|{fields[1]}"
        steps.extend([{"write-serial": packet}, {"wait-serial": ack}])
        expected.append(ack)
    if screenshot is not None:
        steps.extend([{"delay": "100ms"}, {"take-screenshot": {"part-id": "screen", "save-to": screenshot.name}}])
    # JSON scalar quoting is also valid YAML and preserves the UART newline.
    split = str(frame.iloc[0].get("split", "validation"))
    lines = [f"name: {json.dumps('PPG ' + split + ' replay ' + model)}", "version: 1", "steps:"]
    for step in steps:
        for action, value in step.items():
            lines.append(f"  - {action}: {json.dumps(value)}")
    path = folder / "replay.scenario.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, expected


def capture_simulation(command: list[str], child_env: dict[str, str], token: str, log_path: Path) -> tuple[int, str, bool]:
    """Persist redacted live output and bound an unresponsive cloud connection."""
    lines = []
    process = subprocess.Popen(command, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    def read_output() -> None:
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                safe = redact_token(line, token)
                lines.append(safe)
                log.write(safe)
                log.flush()
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait()
    reader.join(timeout=5)
    return (124 if timed_out else process.returncode), "".join(lines), timed_out


def run_wokwi(frame: pd.DataFrame, model: str) -> int:
    """Build and verify recorded BPM transport in real firmware simulation."""
    cli = shutil.which("wokwi-cli")
    pio = shutil.which("pio") or shutil.which("platformio")
    if not cli or not pio:
        raise RuntimeError("Install PlatformIO and Wokwi CLI before using --wokwi")
    token, token_source = wokwi_token()
    folder = DEMO / "firmware" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    scenario, expected = make_scenario(frame, model, folder)
    serial_path = folder / "serial.log"
    child_env = dict(os.environ, WOKWI_CLI_TOKEN=token)
    build = subprocess.run([pio, "run", "-d", str(PROJECT)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    (folder / "build.log").write_text(redact_token(build.stdout + build.stderr, token), encoding="utf-8")
    evidence = {"at": datetime.now(timezone.utc).isoformat(), "passed": False,
                "scope": "level B; real firmware receives existing host predictions; no model inference in ESP32",
                "model": model, "seed": 42, "split": "validation", "n_windows": len(frame),
                "token_source": token_source, "build_returncode": build.returncode,
                "replay_sha256": hashlib.sha256((DEMO / "replay.csv").read_bytes()).hexdigest(),
                "scenario": str(scenario.relative_to(ROOT)), "serial_log": str(serial_path.relative_to(ROOT))}
    record = folder / "verification.json"
    if build.returncode:
        record.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(f"Firmware build failed; see {folder.relative_to(ROOT)}/build.log")
        return build.returncode
    evidence["firmware_sha256"] = hashlib.sha256((PROJECT / ".pio/build/esp32dev/firmware.bin").read_bytes()).hexdigest()
    command = [cli, str(PROJECT), "--scenario", str(scenario), "--timeout", "60000", "--fail-text", "ERR|", "--serial-log-file", str(serial_path)]
    returncode, output, wall_timeout = capture_simulation(command, child_env, token, folder / "wokwi.log")
    serial = redact_token(serial_path.read_text(encoding="utf-8", errors="replace"), token) if serial_path.exists() else ""
    serial_path.write_text(serial, encoding="utf-8")
    received = [line.strip() for line in serial.splitlines() if line.startswith("ACK|")]
    evidence.update(simulator_returncode=returncode, wall_timeout=wall_timeout, expected_acknowledgments=expected,
                    received_acknowledgments=received, passed=returncode == 0 and received == expected and "ERR|" not in serial)
    record.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Wokwi verification: {'PASS' if evidence['passed'] else 'FAIL'}; {len(received)}/{len(expected)} firmware acknowledgments")
    print(f"Evidence: {record.relative_to(ROOT)}")
    if not evidence["passed"]:
        print(output[-2500:])
    return 0 if evidence["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("E3", "E4"), default="E3")
    parser.add_argument("--wokwi", action="store_true", help="Build firmware and replay through Wokwi")
    args = parser.parse_args()
    frame = load_replay()
    show_preview(frame, args.model)
    if args.wokwi:
        return run_wokwi(frame, args.model)
    print("\nPreview only. Add --wokwi to run the ESP32 simulator.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"demo_pipeline: {exc}", file=sys.stderr)
        raise SystemExit(2)
