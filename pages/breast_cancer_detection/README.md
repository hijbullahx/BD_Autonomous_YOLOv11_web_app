# Breast Cancer Detection using YOLOv11

## Project Overview

An automated pipeline utilizing **YOLOv11** to detect invasive breast cancer in MRI scans with **research-grade accuracy** (91.7% mAP@50).

## Project Structure

```
breast_cancer_detection/
├── config.py           # Project configuration (auto-loaded by Home.py)
├── main.py            # Streamlit application
├── weights/           # Model weights
│   └── best.pt       # Trained YOLOv11 model
└── README.md          # This file
```

## Model Details

- **Architecture**: YOLOv11 (You Only Look Once v11)
- **Dataset**: Duke Breast Cancer MRI Dataset (1,007 slices)
- **Performance Metrics**:
  - **mAP@50**: 91.7% (Research Grade)
  - **Precision**: 98% (Minimal false positives)
  - **Recall**: 81% (High detection rate)
- **Classes**: Invasive Breast Cancer

## Features

- 🎯 High-precision tumor detection
- 📊 Confidence score display
- 🖼️ Visual bounding box annotations
- 📥 Downloadable results
- ⚡ GPU and CPU support
- 🔬 Research-grade accuracy

## How to Use

1. Navigate to the Breast Cancer Detection page
2. Upload an MRI scan (JPG, PNG, or DICOM format)
3. Adjust confidence threshold (default: 0.25)
4. Click "🎗️ Detect Tumors"
5. View results and download annotated image

## Technical Stack

- **Detection**: YOLOv11 (Ultralytics)
- **Framework**: Streamlit
- **Image Processing**: OpenCV, Pillow
- **Deep Learning**: PyTorch
- **Dataset**: Duke Breast Cancer MRI

## Model Training

- Trained on Duke MRI dataset with 1,007 annotated slices
- Optimized for high precision to minimize false positives
- Balanced recall to ensure tumor detection
- Validated on real-world MRI scans

## How to Update Project Info

Edit `config.py` to change:
- Title
- Description
- Technologies
- Status
- GitHub link

Changes will automatically reflect on:
- Home page project card
- Project page header

## Performance Notes

- **98% Precision**: Minimizes false alarms (critical for medical applications)
- **81% Recall**: Detects majority of tumors
- **91.7% mAP@50**: Research-grade accuracy
- Suitable for: Research, screening assistance, education

## Limitations

⚠️ **Important**: This is a research/educational tool and should NOT be used for clinical diagnosis without proper validation and medical professional oversight.

## GitHub Repository

[View on GitHub](https://github.com/hijbullahx/Breast-Cancer-MRI-YOLOv11)

## Credits

- **Developer**: Md. Taher Bin Omar Hijbullah
- **Model**: YOLOv11 by Ultralytics
- **Dataset**: Duke Breast Cancer MRI Dataset
- **Framework**: Streamlit

## License

This project is part of an ML portfolio. Ensure proper attribution when using or referencing this work.
