import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import sys
import torch
import importlib.util

def load_project_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    spec = importlib.util.spec_from_file_location("breast_cancer_detection_config", config_path)
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
        background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%);
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
        background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(255, 105, 180, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover { transform: translateY(-5px); }
    
    .stButton > button {
        background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%);
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
        border-left: 5px solid #ff69b4;
    }
    </style>
""", unsafe_allow_html=True)

# --- UI HEADER ---
st.markdown(f'<h1 class="main-title">{PROJECT_CONFIG["title"]}</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">High-Precision Tumor Detection in MRI Scans using <b>YOLOv11</b></p>', unsafe_allow_html=True)
st.markdown("---")

# Technologies & Frameworks
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("**🧠 Core Model**\nYOLOv11 Object Detection")
with col2:
    st.info("**👁️ Dataset**\nDuke MRI Dataset (1,007 slices)")
with col3:
    st.info("**⚡ Performance**\nmAP@50: 91.7% | Precision: 98%")
with col4:
    st.info("**💻 Hardware**\nOptimized for CUDA & CPU Inference")

st.markdown("---")

# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    """Load YOLOv11 model with caching"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"🔧 Loading model on **{device.upper()}**...")
        model = YOLO(model_path)
        model.to(device)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

# --- DETECTION FUNCTION ---
def detect_tumors(image, model, confidence=0.25):
    """Run inference on MRI image"""
    try:
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert grayscale to RGB if needed (MRI scans are often grayscale)
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:  # Single channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Run inference
        results = model(image, conf=confidence)
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        return annotated_image, results[0]
    except Exception as e:
        st.error(f"Detection error: {e}")
        return None, None

# --- MAIN APPLICATION ---

# Sidebar Configuration
st.sidebar.header("⚙️ System Config")
st.sidebar.markdown("---")

# File uploader
st.sidebar.subheader("📤 Input Feed")
uploaded_file = st.sidebar.file_uploader(
    "Upload MRI Scan",
    type=["jpg", "jpeg", "png", "dcm"],
    help="Upload a breast MRI scan for tumor detection"
)

# Model parameters
st.sidebar.subheader("🎯 Sensitivity")
confidence = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
    help="Higher values = more confident predictions only"
)

if uploaded_file is not None:
    st.subheader("📸 MRI Scan Analysis")
    
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input MRI Scan", use_container_width=True)
    
    with col2:
        with st.spinner("🔍 Running YOLOv11 Inference..."):
            # Load model
            model = load_model(PROJECT_CONFIG['model_path'])
            
            if model is not None:
                # Run detection
                annotated_image, results = detect_tumors(image, model, confidence)
                
                if annotated_image is not None:
                    st.image(annotated_image, caption="Tumor Detection Results", use_container_width=True)
    
    # Stats
    if 'annotated_image' in locals() and annotated_image is not None and 'results' in locals() and results is not None:
        detections = results.boxes
        if len(detections) > 0:
            st.success(f"✅ Detected {len(detections)} tumor region(s)")
            
            # Collect detection info
            confidences = []
            class_counts = {}
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
            
            # Download button
            st.markdown("---")
            from io import BytesIO
            buf = BytesIO()
            Image.fromarray(annotated_image).save(buf, format="PNG")
            st.download_button(
                label="📥 Download Annotated Image",
                data=buf.getvalue(),
                file_name="breast_cancer_detection_result.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.warning("No tumors detected. Try lowering confidence threshold.")
                        
else:
    st.info("👈 Upload an MRI scan to test the YOLOv11 Model")
    
    st.markdown("### 📋 Detection Capability")
    st.write("`Invasive Breast Cancer` detection in MRI scans")
    
    st.markdown("### 📊 Model Performance")
    st.write("- **mAP@50**: 91.7% (Research Grade)")
    st.write("- **Precision**: 98% (Minimal false positives)")
    st.write("- **Recall**: 81% (High detection rate)")
    st.write("- **Dataset**: Duke MRI (1,007 annotated slices)")

# Contributors Section
st.markdown("---")
st.markdown("### 👥 Project Contributors")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="contributor-card">
            <h4 style="color: #1a1a1a; margin-bottom: 0.3rem;">Md. Taher Bin Omar Hijbullah</h4>
            <p style="color: #666; margin: 0.3rem 0;">Student of IUBAT</p>
            <p style="color: #ff69b4; font-weight: 500; margin: 0;">📧 22303142@iubat.edu</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="contributor-card">
            <h4 style="color: #1a1a1a; margin-bottom: 0.3rem;">Md. Masud Rana</h4>
            <p style="color: #666; margin: 0.3rem 0;">Student of IUBAT</p>
            <p style="color: #ff69b4; font-weight: 500; margin: 0;">📧 22303104@iubat.edu</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br><div class='footer'>© 2026 Research Project | YOLOv11 Breast Cancer Detection</div>", unsafe_allow_html=True)