import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import sys
import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks
import importlib.util

def load_project_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    spec = importlib.util.spec_from_file_location("autonomous_vehicle_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROJECT_CONFIG


PROJECT_CONFIG = load_project_config()

# Set page configuration
st.set_page_config(
    page_title=PROJECT_CONFIG['title'],
    page_icon=PROJECT_CONFIG['icon'],
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-title {
        background: linear-gradient(135deg, #006a4e 0%, #f42a41 100%); /* Bangladesh Colors */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #4a5568;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #006a4e 0%, #004d38 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 106, 78, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover { transform: translateY(-5px); }
    
    .stButton > button {
        background: linear-gradient(135deg, #006a4e 0%, #f42a41 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #1a1a1a;
        color: #999;
        text-align: center;
        padding: 10px 0;
        font-size: 12px;
        z-index: 999;
    }
    
    .contributor-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border-left: 5px solid #006a4e;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. DEFINE CUSTOM CBAM ARCHITECTURE (Must match training code) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# --- 2. REGISTER CBAM WITH ULTRALYTICS ---
import ultralytics.nn.modules.block as block
tasks.CBAM = CBAM
block.CBAM = CBAM  # Register in the correct module
block.ChannelAttention = ChannelAttention
block.SpatialAttention = SpatialAttention

# --- UI HEADER ---
st.markdown(f'<h1 class="main-title">{PROJECT_CONFIG["title"]}</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Autonomous Vehicle Obstacle Detection using <b>YOLOv11 + CBAM Attention</b></p>', unsafe_allow_html=True)
st.markdown("---")

# Technologies & Frameworks
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("**🧠 Core Model**\nYOLOv11 Medium + CBAM Attention Module")
with col2:
    st.info("**👁️ Dataset**\nBangladesh Local Traffic (Rickshaw, CNG, Truck, Cycle)")
with col3:
    st.info("**⚡ Performance**\nmAP@50: ~75% (Optimized for occlusion)")
with col4:
    st.info("**💻 Hardware**\nOptimized for CUDA (T4) & CPU Inference")

st.markdown("---")

# Load model with error handling
@st.cache_resource
def load_model():
    """Load YOLO model with Custom CBAM Classes"""
    model_path = PROJECT_CONFIG['model_path']
    
    if not os.path.exists(model_path):
        st.error("❌ Model weights not found!")
        st.warning(f"Please download your `best.pt` and place it at: `{model_path}`")
        st.stop()
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load the model
model = load_model()

# Sidebar Configuration
st.sidebar.header("⚙️ System Config")
st.sidebar.markdown("---")

# File uploader
st.sidebar.subheader("📤 Input Feed")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Footage",
    type=["jpg", "jpeg", "png", "mp4"],
    help="Upload local traffic images or dashcam video"
)

# Model parameters
st.sidebar.subheader("🎯 Sensitivity")
confidence = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
    help="Lower this if the model is missing Cycles/Trucks."
)

iou_threshold = st.sidebar.slider(
    "NMS IOU Threshold", 0.0, 1.0, 0.45, 0.05,
    help="Adjust this if you see double boxes on Rickshaws."
)

# Main content
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Image Processing
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("📸 Scene Analysis")
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Input Frame", use_container_width=True)
        
        with col2:
            with st.spinner("🔍 Running YOLOv11 + CBAM Inference..."):
                results = model(image, conf=confidence, iou=iou_threshold)
                annotated_image = results[0].plot()
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, caption="Attention-Enhanced Detection", use_container_width=True)
        
        # Stats
        detections = results[0].boxes
        if len(detections) > 0:
            st.success(f"✅ Detected {len(detections)} objects")
            
            # Specific logic for Thesis Classes
            class_counts = {}
            confidences = []
            for box in detections:
                name = model.names[int(box.cls[0])]
                class_counts[name] = class_counts.get(name, 0) + 1
                confidences.append(float(box.conf[0]))
            
            # Detection Statistics
            st.markdown("---")
            st.markdown("### 📊 Detection Statistics")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric("Total Detections", len(detections))
            with stat_col2:
                st.metric("Unique Classes", len(class_counts))
            with stat_col3:
                st.metric("Highest Confidence", f"{max(confidences)*100:.2f}%")
            
            # Display class breakdown
            st.markdown("### 🎯 Class Breakdown")
            st.json(class_counts)
        else:
            st.warning("No objects detected. Try lowering confidence.")
    
    # Video Processing
    elif file_extension == 'mp4':
        st.subheader("🎥 Real-time Traffic Analysis")
        
        tfile = os.path.join("temp", "temp_video.mp4")
        os.makedirs("temp", exist_ok=True)
        with open(tfile, "wb") as f:
            f.write(uploaded_file.read())
            
        if st.button("▶️ Start Processing"):
            cap = cv2.VideoCapture(tfile)
            st_frame = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Inference
                results = model(frame, conf=confidence, iou=iou_threshold)
                res_plotted = results[0].plot()
                
                # Display
                st_frame.image(res_plotted, channels="BGR", use_container_width=True)
            
            cap.release()

else:
    st.info("👈 Upload an image to test the YOLOv11-CBAM Model")
    
    st.markdown("### 📋 Supported Classes (Bangladesh Context)")
    st.write("`Rickshaw`, `CNG`, `Bus`, `Truck`, `Car`, `Cycle`, `Bike`, `Mini-Truck`, `People`")

# Contributors Section
st.markdown("---")
st.markdown("### 👥 Project Contributors")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="contributor-card">
            <h4 style="color: #1a1a1a; margin-bottom: 0.3rem;">Md. Taher Bin Omar Hijbullah</h4>
            <p style="color: #666; margin: 0.3rem 0;">Student of IUBAT</p>
            <p style="color: #006a4e; font-weight: 500; margin: 0;">📧 22303142@iubat.edu</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="contributor-card">
            <h4 style="color: #1a1a1a; margin-bottom: 0.3rem;">Md. Rony Mia</h4>
            <p style="color: #666; margin: 0.3rem 0;">Student of IUBAT</p>
            <p style="color: #006a4e; font-weight: 500; margin: 0;">📧 22303296@iubat.edu</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br><div class='footer'>© 2026 Thesis Project | YOLOv11-CBAM Implementation</div>", unsafe_allow_html=True)
