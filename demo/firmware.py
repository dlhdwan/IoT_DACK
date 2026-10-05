"""Deliver newly inferred readings to ESP32 firmware and verify its real display."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from demo.runtime import ROOT, DemoError, Reading, digest, model_entry
from demo.paths import artifact_root
from scripts.demo_pipeline import PROJECT, capture_simulation, make_scenario, redact_token, wokwi_token


def run_firmware(readings: list[Reading], root=None) -> dict:
    root = artifact_root(root)
    if not readings or any(r.split not in ("validation", "test") or r.prediction_origin != "fresh_cpu_inference" for r in readings):
        raise DemoError("Firmware accepts newly inferred Test/Validation readings only")
    model, seed = readings[0].ecode, readings[0].seed
    if any((r.ecode, r.seed, r.artifact_sha256, r.split) != (model, seed, readings[0].artifact_sha256, readings[0].split) for r in readings):
        raise DemoError("A firmware replay must use one model checkpoint and split")
    entry = model_entry(root, model, seed)
    if entry["artifact_sha256"] != readings[0].artifact_sha256:
        raise DemoError("Firmware readings belong to a different artifact workspace")
    cli, pio = shutil.which("wokwi-cli"), shutil.which("pio") or shutil.which("platformio")
    if not cli or not pio:
        raise DemoError("Install PlatformIO and Wokwi CLI to run firmware")
    token, source = wokwi_token()
    folder = root / "results/demo_ui/firmware" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    inputs = folder / "fresh_readings.json"
    inputs.write_text(json.dumps([r.as_record() for r in readings], indent=2) + "\n", encoding="utf-8")
    frame = pd.DataFrame([dict(r.as_record(), **{f"{model}_bpm": r.predicted_bpm}) for r in readings])
    screenshot, serial = folder / "lcd.png", folder / "serial.log"
    scenario, expected = make_scenario(frame, model, folder, screenshot)
    evidence = {"passed": False, "at": datetime.now(timezone.utc).isoformat(), "scope": "level B: AI on host CPU; firmware UART and LCD",
                "model": model, "seed": seed, "split": readings[0].split, "n_windows": len(readings),
                "prediction_origin": "fresh_cpu_inference", "model_sha256": readings[0].artifact_sha256,
                "preprocess_hash": readings[0].preprocess_hash, "inputs_sha256": digest(inputs),
                "token_source": source, "expected_acknowledgments": expected,
                "artifact_workspace": str(root),
                "scenario": scenario.relative_to(root).as_posix(), "serial_log": serial.relative_to(root).as_posix()}
    record = folder / "verification.json"
    try:
        build = subprocess.run([pio, "run", "-d", str(PROJECT)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
        (folder / "build.log").write_text(redact_token(build.stdout + build.stderr, token), encoding="utf-8")
        evidence["build_returncode"] = build.returncode
        if build.returncode:
            return evidence
        evidence["firmware_sha256"] = digest(PROJECT / ".pio/build/esp32dev/firmware.bin")
        evidence["firmware_sources"] = {name: digest(PROJECT / name) for name in ("src/main.cpp", "platformio.ini", "diagram.json", "wokwi.toml")}
        command = [cli, str(PROJECT), "--scenario", str(scenario), "--timeout", "60000", "--fail-text", "ERR|", "--serial-log-file", str(serial)]
        code, _, timed_out = capture_simulation(command, dict(os.environ, WOKWI_CLI_TOKEN=token), token, folder / "wokwi.log")
        text = redact_token(serial.read_text(encoding="utf-8", errors="replace"), token) if serial.exists() else ""
        serial.write_text(text, encoding="utf-8")
        received = [line.strip() for line in text.splitlines() if line.startswith("ACK|")]
        evidence.update(simulator_returncode=code, wall_timeout=timed_out, received_acknowledgments=received,
                        passed=code == 0 and received == expected and "ERR|" not in text and screenshot.is_file())
        if screenshot.is_file():
            evidence.update(screenshot=screenshot.relative_to(root).as_posix(), screenshot_sha256=digest(screenshot))
        return evidence
    except Exception as exc:
        evidence["error"] = redact_token(str(exc), token)
        raise
    finally:
        evidence["verification"] = record.relative_to(root).as_posix()
        record.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
