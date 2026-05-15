# 🚁 ANTLINGS Drone Computer Vision Pipeline

**Autonomous Drone-Based Detection, Tracking & Counting System**

A high-performance computer vision pipeline for autonomous aerial systems built on YOLOv26m for real-time multi-class object detection, persistent multi-object tracking via ByteTrack, and dynamic zone-based counting for autonomous drone surveillance.

**Performance Metrics:**
- **mAP50:** 0.46
- **Inference Speed:** 133 FPS
- **Model:** YOLOv26m (Ultralytics)

---

## ✨ Features

- **🎯 Multi-Class Object Detection** – Real-time detection of Pedestrians, Cars, Trucks, and Buses using YOLOv26m trained on VisDrone dataset
- **📍 Persistent Multi-Object Tracking** – ByteTrack integration for consistent object ID assignment across frames
- **🔢 Dynamic Zone Counting** – Full-frame polygon region-based In/Out metrics for autonomous zone monitoring
- **⚡ High Performance** – 133 FPS inference speed optimized for drone deployment
- **🎬 Video Processing** – End-to-end drone video inference with real-time visualization
- **🏆 Production-Ready** – Trained on VisDrone dataset with comprehensive validation metrics

---

## 📁 Repository Structure

```
ANTLINGS_Drone_CV/
│
├── ANTLINGS_Drone.ipynb           # Main Jupyter notebook with full pipeline
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
│
├── dataset/                       # Input data
│   ├── archive.zip               # Dataset archive
│   ├── Drone Street Traffic, New York City.mp4
│   └── Recording 2026-05-14 213602.mp4
│
├── runs/                          # Training outputs & model weights
│   ├── yolo26m_visdrone_final/    # First training run
│   │   ├── args.yaml              # Training configuration
│   │   └── weights/               # Model weights
│   │
│   └── yolo26m_visdrone_final-2/  # Final training run (best performance)
│       ├── args.yaml              # Training configuration
│       ├── weights/
│       │   ├── best.pt            # ⭐ Best model weights
│       │   └── last.pt            # Last epoch weights
│       ├── results.csv            # Training metrics
│       ├── results.png            # Metrics visualization
│       ├── confusion_matrix.png
│       └── [training visualizations & batch samples]
│
├── outputs/                       # Inference results
│   ├── result_test_drone.mp4      # Detection results
│   ├── result_sahi_drone.mp4      # SAHI inference results
│   ├── result_tracking_drone.mp4  # ByteTrack results
│   └── result_counting_drone.mp4  # Zone counting results
│
└── .git/                          # Git repository
```

---

### Folder Details

| Folder | Purpose |
|--------|---------|
| `runs/` | Contains all training runs, model weights (.pt files), training curves, and validation metrics from YOLOv26m training on VisDrone dataset |
| `outputs/` | Stores inference results from running the pipeline on drone video inputs, including detection, tracking, and counting visualizations |
| `dataset/` | Input drone video samples and dataset archives for model inference and validation |

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- GPU recommended for real-time inference (CUDA 11.8+)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ANTLINGS_Drone_CV.git
   cd ANTLINGS_Drone_CV
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights & demo videos:**
   - 🔗 **[Download Weights & Demo Videos Here](insert_drive_link)**
   - Extract `best.pt` and `last.pt` to `runs/yolo26m_visdrone_final-2/weights/`
   - Place demo videos in `outputs/` folder

### Running the Pipeline

#### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook ANTLINGS_Drone.ipynb
```
- Load your drone video
- Run inference with detection, tracking, and zone counting
- Visualize results in real-time

#### Option 2: Command Line
```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('runs/yolo26m_visdrone_final-2/weights/best.pt')

# Run inference
results = model.predict(source='your_video.mp4', conf=0.5)
```

### Quick Start Example

```python
from ultralytics import YOLO
from ultralytics_solutions import ObjectCounter
import cv2

# Initialize model
model = YOLO('runs/yolo26m_visdrone_final-2/weights/best.pt')

# Initialize zone counter
counter = ObjectCounter()

# Process video
cap = cv2.VideoCapture('dataset/Drone_Sample.mp4')

while True:
	ret, frame = cap.read()
	if not ret:
		break
    
	# Run detection
	results = model(frame)
    
	# Update counter with zone polygons
	frame = counter.start_counting(frame, results)
    
	cv2.imshow('Drone Detection & Counting', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 Training Details

- **Dataset:** VisDrone (aerial object detection benchmark)
- **Model Architecture:** YOLOv26m
- **Training Framework:** Ultralytics YOLOv8
- **Hardware:** Google Colab GPU (NVIDIA Tesla V100)
- **Best Model:** `runs/yolo26m_visdrone_final-2/weights/best.pt`

### Training Metrics
- **mAP50:** 0.46
- **Car Detection Accuracy:** 82.4%
- **Classes:** Pedestrian, Car, Truck, Bus

See `runs/yolo26m_visdrone_final-2/` for detailed training curves and validation results.

---

## 🔍 Model Classes

```
0: Pedestrian
1: Car
2: Truck
3: Bus
```

---

## 📦 Dependencies

See `requirements.txt` for complete list:
- **ultralytics** ≥ 8.4.0 – YOLO detection framework
- **opencv-python** ≥ 4.8.0 – Computer vision utilities
- **tqdm** ≥ 4.66.0 – Progress bars
- **lapx** ≥ 0.5.5 – ByteTrack integration

---

## 🎯 Features Pipeline

```
Drone Video Input
	↓
Frame Extraction
	↓
YOLOv26m Detection (133 FPS)
	↓
ByteTrack Multi-Object Tracking
	↓
Zone-Based Counting (In/Out)
	↓
Visualization & Output
	↓
.mp4 Video Output
```

---

## 📥 Download Weights & Demo Videos

⚠️ **Model weights and demo videos are too large for GitHub.**

**🔗 [Download Weights & Demo Videos Here](insert_drive_link)**

Contents:
- `best.pt` – Best YOLOv26m model (optimal performance)
- `last.pt` – Last epoch weights
- `result_test_drone.mp4` – Detection inference demo
- `result_sahi_drone.mp4` – SAHI inference demo
- `result_tracking_drone.mp4` – ByteTrack tracking demo
- `result_counting_drone.mp4` – Zone counting demo

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: ultralytics` | Run `pip install -r requirements.txt` |
| CUDA out of memory | Reduce frame resolution or use CPU inference |
| No GPU detected | Ensure CUDA drivers installed or add `device=0` to model parameters |
| Video codec error | Convert video to H.264 using `ffmpeg -i input.mp4 -c:v libx264 output.mp4` |

---

## 📝 Project Phases

- ✅ **Phase 1:** Environment Setup & Dataset Ingestion
- ✅ **Phase 2:** Model Training (YOLOv26m on VisDrone)
- 🚀 **Phase 3:** Inference Pipeline with Tracking & Counting
- 📊 **Phase 4:** Evaluation & Optimization
- 🎓 **Phase 5:** Submission & Presentation

---

## 👨‍💻 Author

**Md. Taher Bin Omar Hijbullah**

ANTS Aerial Systems – Autonomous Drone Assessment Project

**Deadline:** May 19, 2026

---

## 📄 License

This project is provided as-is for research and evaluation purposes.

---

## 🤝 Contributing

For improvements or bug reports, please open an issue or submit a pull request.

---

## 📞 Support

For questions or issues:
1. Check the Jupyter notebook (`ANTLINGS_Drone.ipynb`) for implementation details
2. Review training metrics in `runs/yolo26m_visdrone_final-2/`
3. Consult [Ultralytics Documentation](https://docs.ultralytics.com)

---

**Last Updated:** May 14, 2026
