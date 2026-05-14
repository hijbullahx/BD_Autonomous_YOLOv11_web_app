import os
import sys
import time
from collections import Counter, defaultdict

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
        animation: fadeInDown 0.8s ease-out;
    }

    .subtitle { color: #475569; font-size: 1.08rem; margin-bottom: 1.6rem; }

    .metric-card {
        background: linear-gradient(135deg, #0ea5e9 0%, #334155 100%);
        padding: 1.1rem 1.2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 12px 28px rgba(14, 165, 233, 0.22);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        min-height: 100%;
    }

    .info-panel {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-left: 5px solid #0ea5e9;
        padding: 1.1rem 1.2rem;
        border-radius: 14px;
        color: #1e293b;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
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
        padding: 0.85rem 1.4rem;
        font-weight: 700;
        transition: all 0.25s ease;
        width: 100%;
    }

    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 26px rgba(14, 165, 233, 0.28); }

    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background: linear-gradient(90deg, #0f172a 0%, #334155 100%);
        color: #cbd5e1; text-align: center; padding: 10px 0; font-size: 12px; z-index: 999;
        box-shadow: 0 -2px 12px rgba(15, 23, 42, 0.12);
    }

    @keyframes fadeInDown { from { opacity: 0; transform: translateY(-16px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<h1 class="main-title">{PROJECT_CONFIG["title"]}</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">High-altitude perception for human and vehicle detection with persistent tracking and aerial analytics.</p>', unsafe_allow_html=True)
st.markdown('<div class="info-panel"><strong>Mission Profile:</strong> Drone-based monitoring combining YOLO detection with ByteTrack persistence to count and follow objects across frames.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-label">Core Model</div><div class="metric-value">YOLOv26m</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-label">Tracking</div><div class="metric-value">ByteTrack</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-label">Dataset</div><div class="metric-value">VisDrone</div></div>', unsafe_allow_html=True)
with col4:
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
        rows.append({
            "Index": idx + 1,
            "Track ID": track_id if track_id is not None else "-",
            "Class": class_name,
            "Confidence": round(confidence, 4),
            "Box X1": round(xyxy[0], 2),
            "Box Y1": round(xyxy[1], 2),
            "Box X2": round(xyxy[2], 2),
            "Box Y2": round(xyxy[3], 2),
            "Frame": frame_index if frame_index is not None else "-",
        })
        class_counts[class_name] += 1
        confidences.append(confidence)

    return rows, class_counts, confidences


st.sidebar.header("⚙️ System Config")
st.sidebar.markdown("---")
st.sidebar.subheader("🛰️ Inference Mode")
video_mode = st.sidebar.checkbox("Video Inference", value=False, help="Turn on for persistent tracking over video frames.")
mode_label = "Video Inference" if video_mode else "Image Inference"
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
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.30, 0.05, help="Higher values reduce false detections.")

iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05, help="Used for non-max suppression and track association.")

model = load_model()

if uploaded_file is not None:
    if video_mode:
        st.subheader("🎥 Drone Video Tracking")
        temp_dir = os.path.join("temp", "drone_detection")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file, "wb") as file_handle:
            file_handle.write(uploaded_file.read())

        # Build a live detection player: annotated frame placeholder + controls + realtime summary
        preview_col, control_col = st.columns([1.2, 0.8])

        # Initialize placeholders
        frame_placeholder = preview_col.empty()
        summary_placeholder = preview_col.empty()

        # Small raw preview on the side
        with control_col:
            st.markdown("**Preview**")
            try:
                cap_preview = cv2.VideoCapture(temp_file)
                ret, first_frame = cap_preview.read()
                cap_preview.release()
                if ret:
                    preview_img = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                    st.image(preview_img, use_column_width=True)
                else:
                    st.info("No preview available for this file.")
            except Exception:
                st.info("Preview not available.")

            start = st.button("▶️ Start Live Detection")
            stop = st.button("⏹️ Stop")

        if start:
            cap = cv2.VideoCapture(temp_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            tracked_objects = {}
            class_event_counts = Counter()
            inference_times = []
            processed_frames = 0

            # Stream frames and update counts live
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_start = time.perf_counter()
                try:
                    results = model.track(frame, conf=confidence, iou=iou_threshold, persist=True, tracker="bytetrack.yaml",
{