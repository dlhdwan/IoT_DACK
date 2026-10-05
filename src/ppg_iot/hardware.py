"""Readable host names for measured CPU benchmarks; no device selection."""
import os
from pathlib import Path
import platform
import subprocess


def environment_names():
    cpu, system = platform.processor().strip(), platform.platform()
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
        try:
            response = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-CimInstance Win32_OperatingSystem).Caption"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if response.returncode == 0 and response.stdout.strip():
                system = response.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    elif Path("/proc/cpuinfo").is_file():
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        cpu = next((line.split(":", 1)[1].strip() for line in text.splitlines()
                    if line.startswith("model name") and ":" in line), cpu)
    return {"cpu": cpu or platform.machine(), "os": system,
            "processor_identifier": platform.processor(), "platform_python": platform.platform()}
