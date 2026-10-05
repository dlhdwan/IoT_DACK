# Demo PPG + ACC và firmware ESP32 giả lập

Demo mặc định dùng **Run All gần nhất05/10/2026** được chọn trong `configs/primary_run.json`. E3/FP32 và E4/INT8 được nạp đúng SHA/checkpoint cùng seed; mỗi cửa sổ gọi model CPU mới. CSV predictions chỉ dùng kiểm parity, không cấp BPM cho UI.

## Chạy tại root repo

```powershell
uv sync --locked --extra demo
uv run --locked --extra demo streamlit run demo/app.py --server.address 127.0.0.1
```

Mở **http://127.0.0.1:8501**. Các trang: Trình diễn Test, Kết quả đồ án, Mô hình & phương pháp. Chọn S13/S14/S15, tám hoạt động/đoạn hỗn hợp, đoạn thời gian, vị trí, seed42/123/999, FP32/INT8; Phát/Dừng/Trước/Tiếp/Về đầu điều khiển replay. Đồ thị Matplotlib màu thể hiện PPG/ACC và diễn biến BPM; CSV tải xuống là kết quả suy luận mới của phiên.

## Nguồn và bộ artifacts

`configs/primary_run.json` chỉ tới workspace hiện tại; `PPG_RUN_DIR` hoặc `--run-dir` cho phép chọn rõ một run khác đã khóa. Thiếu pin không tự dùng bộ root lịch sử. Model, Test bundle, manifest/Train ACC stats, protocol lock, registry và metrics/benchmark phải cùng run. Trên máy nghiên cứu, bộ hiện tại có đủ13.007 Test windows và không cần raw data khi mở UI.

Clone Git có code,9 model và20 kết quả/ảnh trong `published/`; **bundle đầy đủ và các index cục bộ không đi vào Git**. Khôi phục nguyên workspace theo pin từ bộ artifacts, hoặc dùng một Run All hoàn tất của mình và chọn `PPG_RUN_DIR`. Có continuous cache thì xuất bundle bằng:

```powershell
uv run --locked --extra demo python scripts/prepare_demo.py
uv run --locked --extra demo python scripts/verify_final_demo.py --firmware
```

Xuất bundle không tính BPM/train/calibration. Kiểm demo có suy luận mới và ghi lịch sử Test riêng; không đổi đánh giá Test đã khóa. ECG/HR tham chiếu, hoạt động và ID không là đầu vào mạng.

## LCD trong Wokwi

Dừng replay và bấm **Gửi cửa sổ này tới Wokwi**. Bridge build/chạy firmware, gửi BPM vừa suy luận qua UART, kiểm ACK và chụp LCD màu. Ảnh là lần gửi gần nhất; đổi cửa sổ cần gửi lại để LCD cập nhật.

Cần PlatformIO và Wokwi CLI; token `WOKWI_CLI_TOKEN` nằm trong process hoặc Windows User, không lưu trong source/notebook. Bridge tự đọc User nếu cần; không cần MQTT/private gateway cho đường UART này. [Source firmware](../firmware/wokwi_display/README.md).

## Kiểm chứng của run chính

Waveform13.007 Test windows khớp continuous source, delta0. Quy tắc first/middle/last của mọi đoạn người/hoạt động chọn153 windows×6model=918 fresh inference. FP32 lệch tối đa0,00001526 BPM so dự đoán đã khóa; INT8 lệch0; dung sai1e-4 BPM.18/18 tests và6/6 ACK/LCD trên S13–S15 đạt. Browser Edge thật kiểm model SHA, các điều khiển, các trang và gửi một cửa sổ S15/INT8 tới LCD.

[Ảnh UI](../results/published/demo_replay.png), [trang kết quả](../results/published/demo_results.png), [LCD](../results/published/firmware_lcd.png), [JSON kiểm chứng](../results/published/demo_verification.json). Helper browser cần Playwright và trình duyệt riêng; chúng không là dependency mở app.

Luồng trình bày: phương pháp → kết quả toàn Test → replay cùng cửa sổ FP32/INT8 → dừng/gửi UART → ACK/LCD. AI chạy CPU host; firmware ESP32 giả lập thực hiện parser/UART/LCD. Lọc hai chiều và padding đối xứng là xử lý ngoại tuyến; CPU/simulator không là phép đo AI/điện năng/bộ nhớ trên board thật.
