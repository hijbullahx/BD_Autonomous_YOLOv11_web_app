"""
Drone Detection Project Configuration
This config is automatically loaded by Home.py
"""

PROJECT_CONFIG = {
    "id": "drone_detection",
    "title": " ANTLINGS Drone Computer Vision Pipeline",
    "status": "Active",
    "icon": "🚁",
    "page_name": "3_🚁_Drone_Detection.py",
    "description": "Autonomous drone-based detection, tracking and counting using YOLOv26m and ByteTrack. High-altitude human and vehicle perception and zone-based analytics.",
    "technologies": [
        "YOLOv26m (Ultralytics)",
        "VisDrone Dataset",
        "ByteTrack",
        "OpenCV Analytics"
    ],
    "model_path": "pages/drone_detection/weights/best.pt"
}
