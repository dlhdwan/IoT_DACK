# ESP32 UART/LCD firmware for the PPG heart-rate demo

The host Streamlit app loads notebook-exported FP32/INT8 models and performs fresh CPU inference. This firmware runs in Wokwi, receives the reading over UART and displays BPM, ECG reference, absolute error, activity, window, model, seed and split on a color ILI9341 LCD. Neural-network inference and DSP run on the host.

## Run

Install PlatformIO Core and Wokwi CLI, set `WOKWI_CLI_TOKEN` in the process or Windows User environment, and provide the model/data artifacts described in the [demo guide](../../demo/README.md). The bridge reads the User token when the process has not inherited it.

From the repository root:

```powershell
uv sync --locked --extra demo
uv run --locked --extra demo streamlit run demo/app.py --server.address 127.0.0.1
```

Open localhost8501, pause replay and press **Gửi cửa sổ này tới Wokwi**. The bridge builds/runs firmware, checks exact ACK and displays the actual LCD screenshot. The screenshot is the last sent window, not continuous emulator video. Simulation requires access to Wokwi's cloud service.

## Protocol and hardware configuration

Packet: `window_id|test|activity|prediction|reference|E3/E4|seed`.
ACK: `ACK|window_id|prediction_2dp|reference_2dp|E3/E4|seed|test`.

Validation and legacy five-field packets remain supported. Finite BPM predictions, including large errors, are retained. HR reference is displayed only; it does not create the prediction.

SPI: CS5, D/C27, MOSI23, SCK18, VCC3V3/GND. See `diagram.json`, `platformio.ini` and `wokwi.toml`. Verified toolchain: PlatformIO6.2.0, espressif32@7.1.3, WokwiCLI0.28.0, ILI9341 1.6.4, GFX1.12.6, BusIO1.17.4.

Build/scenario/Serial/ACK/screenshot evidence is generated locally under `results/`, excluded from Git. Build RAM/Flash values describe the UART/LCD receiver. This level-B simulation verifies transport/display; it does not establish neural inference, latency or energy on a physical ESP32.
