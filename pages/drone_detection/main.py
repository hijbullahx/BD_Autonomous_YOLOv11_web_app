import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

sys.path.append(os.path.dirname(__file__))
from config import PROJECT_CONFIG

st.set_page_config(
    page_title=PROJECT_CONFIG["title"],
    page_icon=PROJECT_CONFIG["icon"],
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-title {
        background: linear-gradient(135deg, #7dd3fc 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .subtitle {
        color: #475569;
        font-size: 1.08rem;
        margin-bottom: 1.4rem;
    }

    .info-panel {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-left: 5px solid #0ea5e9;
        padding: 1.1rem 1.2rem;
        border-radius: 14px;
        color: #1e293b;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }

    .metric-card {
        background: linear-gradient(135deg, #0ea5e9 0%, #334155 100%);
        padding: 1rem 1.1rem;
        border-radius: 14px;
        color: #fff;
        text-align: center;
        box-shadow: 0 12px 28px rgba(14, 165, 233, 0.22);
    }

    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.9;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    .detail-card {
        background: #0b0b0b;
        color: #ffffff;
        border: 1px solid #222222;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }

    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #475569 100%) !important;
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        font-weight: 700;
        width: 100%;
    }

    .live-summary {
        background: #071124;
        color: #ffffff;
        border-radius: 10px;
        padding: 0.8rem 0.9rem;
        margin-top: 0.6rem;
        border: 1px solid #1f2a44;
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, #0f172a 0%, #334155 100%);
        color: #cbd5e1;
        text-align: center;
        padding: 10px 0;
        font-size: 12px;
        z-index: 999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<h1 class="main-title">{PROJECT_CONFIG["title"]}</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">High-altitude perception for human and vehicle detection with persistent tracking and aerial analytics.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="info-panel"><strong>Mission Profile:</strong> Drone-based monitoring with YOLO detection + ByteTrack persistence for counts and trajectory-aware summaries.</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

head_col1, head_col2, head_col3, head_col4 = st.columns(4)
with head_col1:
    st.markdown('<div class="metric-card"><div class="metric-label">Core Model</div><div class="metric-value">YOLOv26m</div></div>', unsafe_allow_html=True)
with head_col2:
    st.markdown('<div class="metric-card"><div class="metric-label">Tracking</div><div class="metric-value">ByteTrack</div></div>', unsafe_allow_html=True)
with head_col3:
    st.markdown('<div class="metric-card"><div class="metric-label">Dataset</div><div class="metric-value">VisDrone</div></div>', unsafe_allow_html=True)
with head_col4:
    st.markdown('<div class="metric-card"><div class="metric-label">Analytics</div><div class="metric-value">OpenCV</div></div>', unsafe_allow_html=True)

st.markdown("---")


@st.cache_resource
def load_model() -> YOLO:
    model_path = PROJECT_CONFIG["model_path"]
    if not os.path.exists(model_path):
        st.error(f"Model weights not found at {model_path}")
        st.stop()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    return model


def to_rgb_image(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def get_class_name(model: YOLO, class_id: int) -> str:
    names = model.names
    if isinstance(names, dict):
        return names.get(class_id, f"class_{class_id}")
    if class_id < len(names):
        return names[class_id]
    return f"class_{class_id}"


def build_detection_rows(model: YOLO, boxes, frame_index: int | None = None):
    rows = []
    class_counts = Counter()
    confidences = []

    for idx, box in enumerate(boxes):
        class_id = int(box.cls[0])
        class_name = get_class_name(model, class_id)
        confidence = float(box.conf[0])
        track_id = None
        if getattr(box, "id", None) is not None:
            track_id = int(box.id[0])

        xyxy = box.xyxy[0].tolist()
        rows.append(
            {
                "Index": idx + 1,
                "Track ID": track_id if track_id is not None else "-",
                "Class": class_name,
                "Confidence": round(confidence, 4),
                "Box X1": round(xyxy[0], 2),
                "Box Y1": round(xyxy[1], 2),
                "Box X2": round(xyxy[2], 2),
                "Box Y2": round(xyxy[3], 2),
                "Frame": frame_index if frame_index is not None else "-",
            }
        )
        class_counts[class_name] += 1
        confidences.append(confidence)

    return rows, class_counts, confidences


st.sidebar.header("⚙️ System Config")
st.sidebar.markdown("---")
st.sidebar.subheader("🛰️ Inference Mode")
mode_label = st.sidebar.radio(
    "Choose Mode",
    ["Image Inference", "Video Inference"],
    index=1,
    help="Select Video Inference to upload MP4/MOV/AVI/MKV files.",
)
video_mode = mode_label == "Video Inference"
st.sidebar.caption(f"Active mode: {mode_label}")

st.sidebar.subheader("📤 Input Feed")
if video_mode:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload drone footage for tracked video inference",
    )
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a drone frame for object detection",
    )

st.sidebar.subheader("🎯 Sensitivity")
confidence = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.30, 0.05, help="Higher values reduce false detections."
)
iou_threshold = st.sidebar.slider(
    "IOU Threshold", 0.0, 1.0, 0.45, 0.05, help="Used for non-max suppression and track association."
)

model = load_model()

if uploaded_file is not None:
    if video_mode:
        st.subheader("🎥 Drone Video Tracking")
        temp_dir = os.path.join("temp", "drone_detection")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file, "wb") as file_handle:
            file_handle.write(uploaded_file.read())

        player_col, run_col = st.columns([1.2, 0.8])

        with player_col:
            player_placeholder = st.empty()
            player_placeholder.video(temp_file)
            live_summary_placeholder = st.empty()

        with run_col:
            st.markdown("### Controls")
            run_live = st.button("▶️ Start Live Detection")
            st.caption("When started, detection runs in the same video area above.")

        if run_live:
            cap = cv2.VideoCapture(temp_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            processed_frames = 0
            class_event_counts = Counter()
            tracked_objects = {}
            inference_times = []
            warned_tracker = False
            progress = st.progress(0)

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                start_time = time.perf_counter()
                try:
                    results = model.track(
                        frame,
                        conf=confidence,
                        iou=iou_threshold,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )
                except (ModuleNotFoundError, ImportError):
                    # If tracker extras are missing (e.g. lap), fallback gracefully.
                    if not warned_tracker:
                        st.warning(
                            "Tracker dependencies missing (e.g. 'lap'). Falling back to frame-wise detection."
                        )
                        warned_tracker = True
                    results = model(frame, conf=confidence, iou=iou_threshold, verbose=False)

                inference_times.append(time.perf_counter() - start_time)
                result = results[0]
                annotated = result.plot()
                # Reuse the original player area for live annotated frames.
                player_placeholder.image(annotated, channels="BGR", use_container_width=True)

                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    rows, _, _ = build_detection_rows(model, boxes, frame_index=processed_frames + 1)
                    for row in rows:
                        class_event_counts[row["Class"]] += 1
                        track_id = row["Track ID"]
                        key = track_id if track_id != "-" else f"frame-{processed_frames + 1}-{row['Index']}"
                        if key not in tracked_objects:
                            tracked_objects[key] = {
                                "Track ID": key,
                                "Class": row["Class"],
                                "Best Confidence": row["Confidence"],
                                "First Seen": processed_frames + 1,
                                "Last Seen": processed_frames + 1,
                            }
                        else:
                            tracked_objects[key]["Best Confidence"] = max(
                                tracked_objects[key]["Best Confidence"], row["Confidence"]
                            )
                            tracked_objects[key]["Last Seen"] = processed_frames + 1

                processed_frames += 1
                if total_frames > 0:
                    progress.progress(min(int((processed_frames / total_frames) * 100), 100))

                avg_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
                fps = 1.0 / avg_time if avg_time > 0 else 0.0
                live_summary_placeholder.markdown(
                    f"""
                    <div class="live-summary">
                        <strong>Live Summary</strong><br>
                        Processed Frames: <strong>{processed_frames}</strong><br>
                        Tracked Objects: <strong>{len(tracked_objects)}</strong><br>
                        Detection Events: <strong>{sum(class_event_counts.values())}</strong><br>
                        Approx FPS: <strong>{fps:.1f}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            cap.release()

            total_time = sum(inference_times) if inference_times else 0.0
            avg_fps = processed_frames / total_time if total_time > 0 else 0.0
            avg_latency_ms = (total_time / processed_frames * 1000) if processed_frames > 0 else 0.0

            st.markdown("### 📊 Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Tracked Objects</div><div class="metric-value">{len(tracked_objects)}</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Detection Events</div><div class="metric-value">{sum(class_event_counts.values())}</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Avg FPS</div><div class="metric-value">{avg_fps:.1f}</div></div>',
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Latency / Frame</div><div class="metric-value">{avg_latency_ms:.1f} ms</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### 🧾 Tracked Object Summary")
            if tracked_objects:
                st.dataframe(list(tracked_objects.values()), use_container_width=True, hide_index=True)
            else:
                st.info("No persistent tracks were recorded in the video.")

            st.markdown("### 🎯 Class Frequency")
            if class_event_counts:
                class_summary_rows = [
                    {"Class": class_name, "Count": count}
                    for class_name, count in class_event_counts.most_common()
                ]
                st.dataframe(class_summary_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No objects were detected in the uploaded video.")

    else:
        st.subheader("🖼️ Drone Image Inference")
        image_col, result_col = st.columns(2)
        image = Image.open(uploaded_file)
        image_rgb = to_rgb_image(image)

        with image_col:
            st.image(image_rgb, caption="Input Image", use_container_width=True)

        with result_col:
            with st.spinner("Running YOLO inference on aerial frame..."):
                start_time = time.perf_counter()
                results = model(image_rgb, conf=confidence, iou=iou_threshold, verbose=False)
                inference_time = time.perf_counter() - start_time
                result = results[0]
                annotated_image = result.plot()
            st.image(annotated_image, caption="Annotated Detection", channels="BGR", use_container_width=True)

        boxes = result.boxes
        total_detections = len(boxes) if boxes is not None else 0
        rows, class_counts, _ = build_detection_rows(model, boxes if boxes is not None else [], None)
        unique_classes = len(class_counts)
        detections_per_second = (total_detections / inference_time) if inference_time > 0 else 0.0

        st.markdown("### 📊 Performance Metrics")
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Detections</div><div class="metric-value">{total_detections}</div></div>',
                unsafe_allow_html=True,
            )
        with i2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Classes</div><div class="metric-value">{unique_classes}</div></div>',
                unsafe_allow_html=True,
            )
        with i3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Inference Time</div><div class="metric-value">{inference_time * 1000:.1f} ms</div></div>',
                unsafe_allow_html=True,
            )
        with i4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Detections / Sec</div><div class="metric-value">{detections_per_second:.1f}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("### 📋 Detection Summary")
        if class_counts:
            summary_rows = [{"Class": name, "Count": count} for name, count in class_counts.most_common()]
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)
        else:
            st.warning("No objects detected in the image. Try lowering the confidence threshold.")

        st.markdown("### 🔎 Object Details")
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No detection details to show.")

        if boxes is not None and len(boxes) > 0:
            from io import BytesIO

            buffer = BytesIO()
            Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)).save(buffer, format="PNG")
            st.download_button(
                label="📥 Download Annotated Image",
                data=buffer.getvalue(),
                file_name="drone_detection_result.png",
                mime="image/png",
                use_container_width=True,
            )

else:
    st.info("Upload an aerial image or drone video to start detection.")
    st.markdown("### 📋 Detection Focus")
    st.write("Human and vehicle detection across high-altitude drone captures with persistent tracking support.")
    st.markdown("### 🧠 Technologies")
    st.write("YOLOv26m, VisDrone, ByteTrack, OpenCV Analytics")

st.markdown("---")
st.markdown("### 📦 Project Details")

tech_cols = st.columns(4)
for column, tech in zip(tech_cols, PROJECT_CONFIG["technologies"]):
    with column:
        st.markdown(f'<div class="detail-card"><strong>{tech}</strong></div>', unsafe_allow_html=True)

st.markdown("<br><br><div class='footer'>© 2026 Aerial Intelligence | Drone Detection & Counting</div>", unsafe_allow_html=True)
