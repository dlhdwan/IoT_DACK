"""Lecturer-facing Test replay, frozen research results and ESP32 LCD demo."""
from pathlib import Path
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from demo.runtime import DemoError, infer, load_model
from demo.data import ACTIVITY_NAMES, TEST_SUBJECTS, load_test_windows, record_test_use, replay_segments
from demo.results import load_research_results
from demo.firmware import run_firmware
from scripts.demo_pipeline import encode_packet
from demo.paths import artifact_root

ROOT = artifact_root()

st.set_page_config(page_title="1D-TCN + INT8 · Đồ án AI-IoT", page_icon="📈", layout="wide")


@st.cache_resource
def models_for_seed(root, seed):
    return {code: load_model(Path(root), code, seed) for code in ("E3", "E4")}


@st.cache_resource
def presentation_data(root):
    return load_test_windows(Path(root)), load_research_results(Path(root))


def reset_replay():
    st.session_state.update(position=0, playing=False, history={}, firmware_evidence=None,
                            tick_at=time.monotonic(), replay_finished=False, start_at=1)


def toggle_play():
    if st.session_state.position == st.session_state.episode_length - 1 and not st.session_state.playing:
        reset_replay()
    st.session_state.playing = not st.session_state.playing
    st.session_state.tick_at = time.monotonic()


def move_window(delta):
    st.session_state.position = max(0, min(st.session_state.episode_length - 1, st.session_state.position + delta))
    st.session_state.playing = False
    st.session_state.replay_finished = False


def seek_window():
    st.session_state.position = st.session_state.start_at - 1
    st.session_state.playing = False
    st.session_state.replay_finished = False
    st.session_state.history = {}


def minute_label(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


st.title("Mạng 1D-TCN nhẹ và lượng tử hóa INT8")
st.caption("Ước lượng nhịp tim từ PPG + gia tốc · PPG-DaLiA · Đinh Lê Hoàng Danh — 23110082")
try:
    windows, research = presentation_data(str(ROOT))
    with st.sidebar:
        st.header("Trình diễn trên Test")
        subject_id = st.selectbox("Người Test", list(TEST_SUBJECTS), key="subject_id")
        activity_options = [1, 7, 2, 3, 4, 5, 6, 8, 0, -1]
        activity_options = [code for code in activity_options if code == -1 or
                            any(w.subject_id == subject_id and (w.activity_id or 0) == code for w in windows)]
        activity_id = st.selectbox("Hoạt động", activity_options, format_func=lambda code: ACTIVITY_NAMES[code], key="activity_id")
        segments = replay_segments(windows, subject_id, activity_id)
        segment_key = f"segment_{subject_id}_{activity_id}"
        saved_segment = st.session_state.get(segment_key, 0)
        if not isinstance(saved_segment, int) or not 0 <= saved_segment < len(segments):
            st.session_state[segment_key] = 0
        segment_id = st.selectbox("Đoạn thời gian", list(range(len(segments))),
            format_func=lambda i: f"Đoạn {i+1} · {minute_label(segments[i][0].start_s)}–{minute_label(segments[i][-1].end_s)} · {len(segments[i])} cửa sổ",
            key=segment_key)
        st.session_state.segment_id = segment_id
        model_code = st.selectbox("Mô hình hiển thị", ["E3", "E4"],
                                 format_func=lambda code: "FP32 · E3" if code == "E3" else "INT8 · E4", key="model_code")
        seed = st.selectbox("Seed checkpoint", [42, 123, 999], key="seed")
        speed = st.select_slider("Tốc độ phát lại", options=[0.5, 1.0, 2.0, 4.0], value=1.0, key="speed")
    models = models_for_seed(str(ROOT), seed)
except (DemoError, OSError, KeyError, RuntimeError, AssertionError) as exc:
    st.error(str(exc))
    st.stop()

episode = segments[segment_id]
selection = (subject_id, activity_id, episode[0].window_id, model_code, seed)
if st.session_state.get("selection") != selection:
    reset_replay()
    st.session_state.selection = selection
st.session_state.episode_length = len(episode)
with st.sidebar:
    st.number_input("Bắt đầu tại cửa sổ", min_value=1, max_value=len(episode), step=1, key="start_at", on_change=seek_window)
    st.caption("Cửa sổ 8 giây · bước 2 giây · 32 Hz. Mỗi đoạn liên tục giữ cả các cửa sổ sai số lớn.")
    info = models[model_code].info
    st.metric("Tệp model đang nạp", f"{info.artifact_bytes / 1024:.1f} KiB")
    st.caption("PyTorch 2.10.0 · oneDNN · CPU batch=1, 1 thread")
    with st.expander("Nguồn model và dữ liệu"):
        st.code(info.artifact, language=None)
        st.code(info.artifact_sha256, language=None)
        st.caption("Model/tiền xử lý đã khóa. Đủ 13.007 Test windows, S13–S15 và 8 hoạt động; ECG/hoạt động/ID không đưa vào mạng.")

st.info("PPG + ACC ghi sẵn → tiền xử lý offline → 1D-TCN FP32/INT8 trên CPU → BPM trên UI → UART → LCD ESP32 giả lập")
replay_tab, results_tab, method_tab = st.tabs(["Trình diễn Test", "Kết quả đồ án", "Mô hình & phương pháp"])


def waveform_plot(window):
    seconds = np.arange(256) / 32
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True, constrained_layout=True)
    axes[0].plot(seconds, window.features[0], color="#1769d2", linewidth=1.6, label="PPG")
    for axis, color in enumerate(["#ee7045", "#20a67a", "#9859c5"]):
        axes[1].plot(seconds, window.features[axis + 1], color=color, linewidth=1.2, label=f"ACC {'XYZ'[axis]}")
    axes[0].set_ylabel("PPG chuẩn hóa")
    axes[1].set_ylabel("ACC chuẩn hóa")
    axes[1].set_xlabel("Thời gian trong cửa sổ (s)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", ncol=3)
    return fig


def history_plot(history):
    rows = sorted(history.values(), key=lambda row: row["start_s"])
    fig, axis = plt.subplots(figsize=(10, 3), constrained_layout=True)
    for label, column, color in [("HR ECG tham chiếu", "hr_ref_bpm", "#20a67a"),
                                  ("FP32", "E3_bpm", "#1769d2"), ("INT8", "E4_bpm", "#ee7045")]:
        axis.plot([r["start_s"] for r in rows], [r[column] for r in rows], "o-", color=color, label=label)
    axis.set(xlabel="Thời gian bắt đầu cửa sổ (s)", ylabel="Nhịp tim (BPM)")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3)
    return fig


def render_firmware(current):
    st.subheader("ESP32 giả lập · LCD màu")
    st.caption("AI chạy CPU host; firmware thực thi UART và LCD trong Wokwi.")
    if st.button("Gửi cửa sổ này tới Wokwi", key="send_firmware", disabled=st.session_state.playing):
        with st.spinner("Đang build/chạy firmware và chụp LCD…"):
            try:
                st.session_state.firmware_evidence = run_firmware([current], root=ROOT)
            except Exception as exc:
                st.error(str(exc))
    evidence = st.session_state.firmware_evidence
    if evidence:
        if evidence["passed"]:
            st.success(f"Firmware xác nhận {evidence['n_windows']} cửa sổ Test · {evidence['model']} seed {evidence['seed']}")
            st.image(str(ROOT / evidence["screenshot"]), caption="LCD thực từ lần chạy Wokwi đã xác minh", width=300)
            sent_id = evidence["expected_acknowledgments"][-1].split("|")[1]
            st.caption(f"Ảnh lần gửi gần nhất: {sent_id}")
            if sent_id != current.window_id:
                st.caption("Cửa sổ trên UI đã thay đổi; gửi lại để cập nhật LCD.")
        else:
            st.error("Lần chạy firmware không đạt kiểm tra; xem log.")
        with st.expander("ACK và bằng chứng firmware"):
            st.json(evidence)
            if (ROOT / evidence["serial_log"]).exists():
                st.code((ROOT / evidence["serial_log"]).read_text(encoding="utf-8"), language=None)
    else:
        st.caption("Dừng ở một cửa sổ rồi gửi tới Wokwi để hiển thị BPM trên thiết bị.")
    with st.expander("Gói UART của cửa sổ đang chọn"):
        row = pd.Series(dict(current.as_record(), **{f"{current.ecode}_bpm": current.predicted_bpm}))
        st.code(encode_packet(row, current.ecode).strip(), language=None)


with replay_tab:
    st.caption("Phát lại Test đã khóa · dữ liệu người thật ghi sẵn · lọc ngoại tuyến hai chiều")
    controls = st.columns([1, 1, 1, 1, 2])
    controls[0].button("Dừng" if st.session_state.playing else "Phát", key="play", width="stretch", on_click=toggle_play)
    controls[1].button("Cửa sổ trước", key="previous", width="stretch", on_click=move_window, args=(-1,))
    controls[2].button("Cửa sổ tiếp", key="next", width="stretch", on_click=move_window, args=(1,))
    controls[3].button("Về đầu", key="reset", width="stretch", on_click=reset_replay)
    controls[4].write(f"{subject_id} · {ACTIVITY_NAMES[activity_id]} · {len(episode)} cửa sổ / đoạn")

    @st.fragment(run_every=(2.0 / speed) if st.session_state.playing else None)
    def render_replay():
        now = time.monotonic()
        if st.session_state.playing and now - st.session_state.tick_at >= 2.0 / speed:
            st.session_state.position = min(len(episode) - 1, st.session_state.position + 1)
            st.session_state.tick_at = now
            if st.session_state.position == len(episode) - 1:
                st.session_state.playing = False
                st.session_state.replay_finished = True
                st.rerun()
        window = episode[st.session_state.position]
        readings = {code: infer(model, window) for code, model in models.items()}
        current = readings[model_code]
        st.session_state.latest_readings = readings
        if window.window_id not in st.session_state.history:
            record_test_use(ROOT, "ui_window", {"window_id": window.window_id, "subject_id": window.subject_id,
                "seed": seed, "models": {code: r.artifact_sha256 for code, r in readings.items()}, "input_shape": [1, 4, 256]})
        st.session_state.history[window.window_id] = dict(current.as_record(), end_s=window.end_s,
            E3_bpm=readings["E3"].predicted_bpm, E4_bpm=readings["E4"].predicted_bpm,
            E3_artifact_sha256=readings["E3"].artifact_sha256, E4_artifact_sha256=readings["E4"].artifact_sha256)
        st.subheader(f"Cửa sổ {st.session_state.position + 1}/{len(episode)} · {minute_label(window.start_s)}–{minute_label(window.end_s)}")
        st.caption(f"{window.window_id} · Test · {window.activity_name} · tensor (1, 4, 256)")
        metrics = st.columns(4)
        metrics[0].metric("BPM mô hình", f"{current.predicted_bpm:.2f}")
        metrics[1].metric("HR ECG tham chiếu", f"{current.hr_ref_bpm:.2f}")
        metrics[2].metric("Sai số tuyệt đối (BPM)", f"{current.absolute_error_bpm:.2f}")
        metrics[3].metric("Lượt gọi CPU (ms)", f"{current.cpu_call_ms:.2f}")
        st.caption("BPM được tính mới từ waveform. Thời gian một lượt forward/Quant–DeQuant; benchmark chính thức ở tab Kết quả đồ án.")
        chart_area, firmware_area = st.columns([2, 1], gap="large")
        with chart_area:
            fig = waveform_plot(window)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            comparison = pd.DataFrame([{"Mô hình": "FP32" if code == "E3" else "INT8", "BPM mới": r.predicted_bpm,
                "Sai số (BPM)": r.absolute_error_bpm, "Tệp (byte)": models[code].info.artifact_bytes} for code, r in readings.items()])
            st.dataframe(comparison, hide_index=True, width="stretch")
            st.subheader("Diễn biến BPM của các cửa sổ đã xem")
            fig = history_plot(st.session_state.history)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.download_button("Tải kết quả phiên demo (CSV)", pd.DataFrame(st.session_state.history.values()).to_csv(index=False).encode("utf-8-sig"),
                               "test_demo_readings.csv", "text/csv", key="download")
        with firmware_area:
            render_firmware(current)
        if st.session_state.replay_finished:
            st.success("Đã phát hết đoạn được chọn.")
    render_replay()

with results_tab:
    st.subheader("Kết quả Test của thí nghiệm đã khóa")
    st.caption("S13–S15 · 13.007 cửa sổ / seed · MAE macro theo người · mean ± sample std qua 3 seed học sâu")
    scores = research["summary"].set_index("method")
    top = st.columns(3)
    top[0].metric("PPG + ACC · FP32", f"{scores.loc['E3', 'mae_bpm']:.3f} ± {scores.loc['E3', 'mae_bpm_std']:.3f} BPM")
    top[1].metric("PPG + ACC · INT8", f"{scores.loc['E4', 'mae_bpm']:.3f} ± {scores.loc['E4', 'mae_bpm_std']:.3f} BPM")
    top[2].metric("Mục tiêu MAE", "≤ 6,0 BPM", "Đạt theo giao thức đã khóa", delta_color="off")
    labels = {"E0": "FFT", "E1": "IBI + fallback Train", "E2": "PPG-only FP32", "E3": "PPG+ACC FP32", "E4": "PPG+ACC INT8"}
    fig, axis = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    table = research["summary"]
    axis.bar([labels[m] for m in table.method], table.mae_bpm, yerr=table.mae_bpm_std,
             color=["#adb5bd", "#20a67a", "#9859c5", "#1769d2", "#ee7045"], capsize=4)
    axis.axhline(6, color="#c73644", linestyle="--", label="Mục tiêu 6 BPM")
    axis.set_ylabel("MAE macro theo người (BPM)")
    axis.legend()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    display = table[["method", "mae_bpm", "mae_bpm_std", "rmse_bpm", "error_gt_10_percent", "n_windows", "coverage"]].copy()
    display.method = display.method.map(labels)
    display.columns = ["Phương pháp", "MAE (BPM)", "Std MAE", "RMSE (BPM)", "Lỗi >10 BPM (%)", "Số cửa sổ", "Coverage"]
    st.dataframe(display, hide_index=True, width="stretch")
    delta_acc = scores.loc["E3", "mae_bpm"] - scores.loc["E2", "mae_bpm"]
    delta_int8 = scores.loc["E4", "mae_bpm"] - scores.loc["E3", "mae_bpm"]
    st.write(f"Thêm ACC: E3−E2 = {delta_acc:+.4f} BPM; dấu dương nghĩa là MAE trung bình tăng. PTQ INT8: E4−E3 = {delta_int8:+.4f} BPM. Giữ đầy đủ cả các kết quả bất lợi và cửa sổ sai số lớn.")
    st.subheader(f"Dung lượng và benchmark CPU · seed {seed}")
    pairs = research["pairs"].set_index("seed").loc[seed]
    st.write(f"FP32 {int(pairs.fp32_bytes):,} byte → INT8 {int(pairs.int8_bytes):,} byte; tệp nhỏ hơn {pairs.compression_ratio:.2f} lần.")
    latency = research["latency"].loc[research["latency"].method.isin([f"E3_{seed}", f"E4_{seed}"])]
    st.dataframe(latency[["method", "mean_ms", "median_ms", "p95_ms", "warmup", "n", "batch_size", "cpu_threads"]], hide_index=True, width="stretch")
    runtime = research["runtime"]
    st.caption(f"Phép đo đã lưu: {runtime['cpu']} · Python {runtime['python']} · Torch {runtime['torch']} · {runtime['backend']} · {runtime['threads']} thread. Mạng gồm Quant/DeQuant, DSP đo riêng.")
    st.write(f"Tỷ số FP32/INT8: median {pairs.speedup_median:.2f}×, mean {pairs.speedup_mean:.2f}×. Mean <1 nghĩa là INT8 chậm hơn trong phép đo đó; không suy từ tệp nhỏ sang tốc độ hoặc bộ nhớ MCU.")
    st.subheader(f"Sai số theo người · seed {seed}")
    subjects = research["by_subject"]
    st.dataframe(subjects.loc[subjects.method.isin(["E3", "E4"]) & (subjects.seed == seed),
                            ["method", "subject_id", "mae_bpm", "rmse_bpm", "error_gt_10_percent", "n_valid", "coverage"]], hide_index=True, width="stretch")
    st.caption("Kết quả trên toàn Test từ notebook; phiên replay không là thí nghiệm độc lập mới. Benchmark Đ01 chưa xác minh cùng giao thức; không kết luận lâm sàng/LOSO/MCU từ bảng này.")

with method_tab:
    st.subheader("Mạng tích chập giãn 1D nhẹ")
    st.metric("Số tham số FP32", f"{sum(p.numel() for p in models['E3'].network.parameters()):,}")
    stages = ["PPG+ACC\n4×256", "Conv1d\n32, k7, d1", "Conv1d\n64, k5, d2", "Conv1d\n64, k3, d4", "GAP\n64", "Linear\n64→32→1 BPM"]
    fig, axis = plt.subplots(figsize=(12, 2.3), constrained_layout=True)
    for i, label in enumerate(stages):
        axis.text(i, .5, label, ha="center", va="center", fontsize=11,
                  bbox=dict(boxstyle="round,pad=.6", facecolor=["#dbeafe", "#b9e1f2", "#b9e1f2", "#b9e1f2", "#c8ead9", "#fce0b9"][i]))
        if i < len(stages) - 1:
            axis.annotate("", xy=(i + .65, .5), xytext=(i + .35, .5), arrowprops=dict(arrowstyle="->", color="#54616c"))
    axis.set(xlim=(-.6, 5.6), ylim=(0, 1))
    axis.axis("off")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.write("Ba Conv1d dilation 1/2/4, BN + ReLU; pooling sau hai tầng đầu, GAP và đầu hồi quy. PTQ tĩnh 8 bit từ từng checkpoint FP32; calibration dùng 1.000 cửa sổ Train thật cùng danh sách ID.")
    st.write("Train S1–S10; Validation S11–S12; Test S13–S15. Lọc Butterworth N=2, 0,5–4 Hz liên tục; PPG 64→32 Hz; cửa sổ8s/bước2s. PPG z-score từng cửa sổ, ACC chuẩn hóa bằng thống kê Train đã đóng băng.")
    st.write("HR từ ECG là nhãn tham chiếu. Mạng chỉ nhận tensor PPG và ba trục ACC; không nhận ECG, hoạt động hoặc mã người. Pipeline hiện là xử lý ngoại tuyến, mạng padding đối xứng và phát lại dữ liệu ghi sẵn.")
    st.write("Phần IoT mức B truyền kết quả CPU tới firmware ESP32 và LCD màu trong Wokwi. Mô hình chưa chạy trong MCU; timing/RAM/Flash/năng lượng trên board thật cần kiểm riêng.")
