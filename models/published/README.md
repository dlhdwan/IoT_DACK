# Trọng số suy luận cuối

Chín tệp model-only từ **Run All gần nhất ngày05/10/2026**, workspace `20261005T080438565912Z-f1bf0c7c`, được người dùng chọn làm **kết quả chính của project**. Gồm đủ seed42/123/999 cho mỗi phương pháp; sao chép nguyên byte từ artifact được chọn bằng Validation, không chứa optimizer/trạng thái resume. Bộ published trước đó đã chuyển sang archive ngoài repo.

| Tệp | Phương pháp | Đầu vào | Byte/tệp |
|---|---|---|---|
| `E2_seed_*_fp32.pt` | PPG-only FP32 | `(B,1,256)` | 109.685 |
| `E3_seed_*_fp32.pt` | PPG+ACC FP32 | `(B,4,256)` | 112.373 |
| `E4_seed_*_int8.pt` | PTQ INT8 từ E3 cùng seed | `(B,4,256)` | 39.925 |

[manifest.json](manifest.json) ghi SHA-256, byte, seed, nguồn checkpoint, preprocessing hash và quan hệ parent của INT8. Calibration E4 dùng1.000 cửa sổ Train thật. Runtime nguồn: PyTorch2.10.0, backend INT8 `onednn`. Kiến trúc ở [model.py](../../src/ppg_iot/model.py); cách nạp envelope/fusion/qparams ở [runtime.py](../../demo/runtime.py).

Các tệp chứa gói `state_dict` và metadata, không phải TorchScript hay firmware ESP32. Đầu vào cần đúng tiền xử lý32 Hz của repo. Bộ nhỏ này cho phép tải/xem trọng số; UI đầy đủ còn cần registry/manifest/Test bundle cùng run theo [hướng dẫn demo](../../demo/README.md).

Kiểm checksum trên clone chỉ chứa bộ public, không cần dữ liệu thô hoặc registry cũ:

```powershell
python scripts/export_public_artifacts.py --check
```

Từ run đầy đủ trên máy nghiên cứu, `python scripts/export_public_artifacts.py` lấy nguồn từ [primary_run.json](../../configs/primary_run.json); có thể chỉ định workspace bằng `--run-dir results/notebook_runs/<id>`. Lệnh chỉ copy/tổng hợp artifact đã có; không train, calibration hoặc tính lại Test. [Kết quả tương ứng](../../results/published/README.md).
