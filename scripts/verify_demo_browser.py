"""Optional real-browser verification; run with Playwright and an existing local app."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request
import sys

from playwright.sync_api import sync_playwright

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
from demo.paths import artifact_root
from demo.runtime import load_model, infer, read_json
from demo.data import load_test_windows, replay_segments
from demo.results import load_research_results
ROOT = artifact_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--firmware", action="store_true")
    args = parser.parse_args()
    windows = load_test_windows(ROOT)
    first = replay_segments(windows, "S13", 1)[0][0]
    initial_model = load_model(ROOT, "E3", 42)
    initial_bpm = infer(initial_model, first).predicted_bpm
    scores = load_research_results(ROOT)["summary"].set_index("method")
    summary_text = f"{scores.loc['E3', 'mae_bpm']:.3f} ± {scores.loc['E3', 'mae_bpm_std']:.3f} BPM"
    folder = ROOT / "results/demo_final/browser" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    report = {"passed": False, "at": datetime.now(timezone.utc).isoformat(), "actions": [], "errors": [], "source_run_id": read_json(ROOT / "workspace.json")["execution_id"],
              "artifact_workspace": str(ROOT), "initial_fp32_sha256": initial_model.info.artifact_sha256,
              "initial_bpm_expected": initial_bpm, "e3_summary_expected": summary_text, "screenshots": {}}
    page = None
    try:
        with urllib.request.urlopen(args.url + "/_stcore/health", timeout=10) as response:
            report.update(health_status=response.status, health_body=response.read().decode())
        with sync_playwright() as p:
            edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
            browser = p.chromium.launch(executable_path=str(edge) if edge.is_file() else None, headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1050})
            page.on("pageerror", lambda error: report["errors"].append(str(error)))

            def settle():
                # Wait for a stable idle interval, including reruns that start
                # asynchronously after a widget's click returns to Playwright.
                page.evaluate("window.__demoIdleStart = 0")
                page.wait_for_function("""() => {
                    const status = document.querySelector('[data-testid="stStatusWidget"]');
                    const running = status && status.innerText.includes('Stop');
                    if (running) window.__demoIdleStart = 0;
                    else if (!window.__demoIdleStart) window.__demoIdleStart = performance.now();
                    return !running && performance.now() - window.__demoIdleStart > 500;
                }""", timeout=60000)

            def capture(filename):
                # A metric can appear before a rerun finishes. Wait for the
                # toolbar Stop button to disappear so stale elements are not faded.
                settle()
                page.locator('[role="tabpanel"]:visible [data-testid="stImage"] img').evaluate_all(
                    "images => Promise.all(images.map(image => image.decode()))")
                page.screenshot(path=str(folder / filename), full_page=True)

            page.goto(args.url, wait_until="domcontentloaded")
            page.get_by_role("heading", name="Mạng 1D-TCN nhẹ và lượng tử hóa INT8", exact=True).wait_for(timeout=60000)
            page.get_by_text(f"{initial_bpm:.2f}", exact=True).wait_for(timeout=60000)
            # Wait for actual color plots to load before the visual capture.
            first_plot = page.locator('[data-testid="stImage"] img').first
            first_plot.wait_for(timeout=30000)
            first_plot.evaluate("image => image.decode()")
            capture("test_replay_fp32.png")
            report["screenshots"]["replay"] = (folder / "test_replay_fp32.png").relative_to(ROOT).as_posix()
            page.get_by_text("Nguồn model và dữ liệu", exact=True).click()
            page.get_by_text(initial_model.info.artifact_sha256, exact=True).wait_for(timeout=30000)
            page.get_by_text("Nguồn model và dữ liệu", exact=True).click()
            report["actions"].append(f"Primary S13 FP32 {initial_bpm:.2f} BPM, exact model SHA and color plots")

            def select(label, option):
                settle()
                widget = page.locator('[data-testid="stSelectbox"]').filter(has_text=label)
                widget.get_by_role("combobox").click()
                page.get_by_role("option", name=option, exact=True).click()
                settle()

            page.get_by_role("button", name="Cửa sổ tiếp", exact=True).click()
            page.get_by_role("heading", name="Cửa sổ 2/300", exact=False).wait_for(timeout=30000)
            settle()
            page.get_by_role("button", name="Cửa sổ trước", exact=True).click()
            page.get_by_role("heading", name="Cửa sổ 1/300", exact=False).wait_for(timeout=30000)
            settle()
            page.get_by_role("button", name="Phát", exact=True).click()
            page.get_by_role("button", name="Dừng", exact=True).wait_for(timeout=10000)
            page.get_by_role("heading", name="Cửa sổ 2/300", exact=False).wait_for(timeout=15000)
            page.get_by_role("button", name="Dừng", exact=True).click()
            settle()
            report["actions"].append("Previous/next and timer-driven play/pause advance real Test windows")
            select("Người Test", "S14")
            page.get_by_text("final_32hz_v1:S14:", exact=False).first.wait_for(timeout=30000)
            select("Người Test", "S15")
            page.get_by_text("final_32hz_v1:S15:", exact=False).first.wait_for(timeout=30000)
            select("Hoạt động", "Đi bộ")
            page.get_by_text("· Test · walking", exact=False).first.wait_for(timeout=30000)
            select("Mô hình hiển thị", "INT8 · E4")
            report["actions"].append("Selected S14 then S15 walking and INT8 artifact")
            number = page.get_by_role("spinbutton", name="Bắt đầu tại cửa sổ")
            number.fill("10")
            number.press("Enter")
            settle()
            page.get_by_role("heading", name="Cửa sổ 10/", exact=False).wait_for(timeout=30000)
            report["actions"].append("Seek to the tenth window of the selected Test episode")
            page.get_by_role("tab", name="Kết quả đồ án", exact=True).click()
            page.get_by_text(summary_text, exact=True).wait_for(timeout=30000)
            capture("research_results.png")
            report["screenshots"]["results"] = (folder / "research_results.png").relative_to(ROOT).as_posix()
            report["actions"].append("Frozen E0–E4 Test results, target, bytes and actual CPU benchmark are visible")
            page.get_by_role("tab", name="Mô hình & phương pháp", exact=True).click()
            page.get_by_text("25,857", exact=True).wait_for(timeout=30000)
            capture("architecture.png")
            report["screenshots"]["architecture"] = (folder / "architecture.png").relative_to(ROOT).as_posix()
            report["actions"].append("Architecture and 25,857 real FP32 parameters displayed")
            page.get_by_role("tab", name="Trình diễn Test", exact=True).click()
            if args.firmware:
                page.get_by_role("button", name="Gửi cửa sổ này tới Wokwi", exact=True).click()
                page.get_by_text("Firmware xác nhận 1 cửa sổ Test · E4 seed 42", exact=True).wait_for(timeout=180000)
                caption = page.get_by_text("LCD thực từ lần chạy Wokwi đã xác minh", exact=True)
                caption.wait_for(timeout=30000)
                image = page.locator('[data-testid="stImage"]').filter(has=caption).locator("img")
                image.evaluate("image => image.decode()")
                report["lcd_natural_size"] = image.evaluate("image => [image.naturalWidth, image.naturalHeight]")
                image.scroll_into_view_if_needed()
                capture("test_firmware_lcd.png")
                report["screenshots"]["firmware"] = (folder / "test_firmware_lcd.png").relative_to(ROOT).as_posix()
                report["actions"].append("UI sends new S15 INT8 Test reading; exact firmware ACK and actual color LCD image rendered")
            report["browser_version"] = browser.version
            report["passed"] = not report["errors"] and report["health_status"] == 200
            browser.close()
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        (folder / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (ROOT / "results/demo_final/browser_index.json").write_text(json.dumps({"passed": report["passed"],
            "verification": (folder / "verification.json").relative_to(ROOT).as_posix()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(folder.relative_to(ROOT))


if __name__ == "__main__":
    main()
