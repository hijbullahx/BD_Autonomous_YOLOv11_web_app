"""
Breast Cancer Detection Project Configuration
This config is automatically loaded by Home.py
"""

PROJECT_CONFIG = {
    "id": "breast_cancer_detection",
    "title": "Breast Cancer Detection using YOLOv11_V1",
    "status": "Active",
    "icon": "🎗️",
    "page_name": "2_🎗️_Breast_Cancer_Detection.py",
    "description": """<strong>High-Precision Breast Cancer Detection using YOLOv11</strong><br>
        Advanced automated pipeline for detecting invasive breast cancer in MRI scans with research-grade accuracy. 
        Utilizes YOLOv11 object detection model trained on Duke Breast Cancer MRI Dataset (1,007 annotated slices) 
        to identify tumor regions with exceptional precision (98%) and high recall (81%), achieving 91.7% mAP@50. 
        Optimized for medical imaging workflows with automatic grayscale-to-RGB conversion, providing reliable 
        tumor detection for early diagnosis support and research applications in challenging radiological environments.""",
    "technologies": [
        "YOLOv11 Object Detection",
        "Duke MRI Dataset (1,007 slices)",
        "mAP@50: 91.7% (Research Grade)",
        "Precision: 98% | Recall: 81%",
        "Streamlit & OpenCV"
    ],
    "github_link": "https://github.com/hijbullahx/Breast-Cancer-MRI-YOLOv11",
    "model_path": "pages/breast_cancer_detection/weights/best.pt"
}
