"""
Autonomous Vehicle Project Configuration
This config is automatically loaded by Home.py
"""

PROJECT_CONFIG = {
    "id": "autonomous_vehicle",
    "title": "Autonomous Vehicle Traffic Perception System",
    "status": "Active",
    "icon": "🚗",
    "page_name": "1_Bangladesh_Traffic.py",  # Wrapper page in pages/
    "description": """<strong>Autonomous Vehicle Obstacle Detection using YOLOv11 + CBAM Attention</strong><br>
        Advanced real-time detection system designed for dense urban traffic conditions. 
        Utilizes YOLOv11 Medium model enhanced with CBAM (Convolutional Block Attention Module) to detect 
        local vehicles including Rickshaw, CNG, Bus, Truck, Car, Cycle, Bike, Mini-Truck, and People with 
        ~75% mAP@50. Optimized for dense traffic scenarios with occlusions, providing attention-enhanced 
        detection for safer autonomous navigation in challenging urban environments.""",
    "technologies": [
        "YOLOv11 Medium + CBAM Attention Module",
        "Local Traffic Dataset",
        "mAP@50: ~75% (Optimized for occlusion)",
        "CUDA (T4) & CPU Inference"
    ],
    "github_link": "https://github.com/hijbullahx",
    "model_path": "pages/autonomous_vehicle/weights/best.pt"
}
