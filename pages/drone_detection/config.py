"""
Drone Detection Project Configuration
This config is automatically loaded by Home.py
"""

PROJECT_CONFIG = {
    "id": "drone_detection",
    "title": "Aerial Perception: Drone Detection & Counting",
    "status": "Active",
    "icon": "🚁",
    "page_name": "3_🚁_Drone_Detection.py",
    "description": "Advanced YOLOv26m pipeline for high-altitude human and vehicle detection with persistent tracking.",
    "technologies": [
        "YOLOv26m (Ultralytics)",
        "VisDrone Dataset",
        "ByteTrack",
        "OpenCV Analytics"
    ],
    "model_path": "pages/drone_detection/weights/best.pt"
}
