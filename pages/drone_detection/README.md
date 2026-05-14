# Aerial Perception: Drone Detection & Counting

This project adds autonomous aerial surveillance to the portfolio with a Streamlit interface for drone-based human and vehicle detection.

## Overview

- Model: YOLOv26m via Ultralytics
- Dataset: VisDrone
- Tracking: ByteTrack for persistent IDs in video streams
- Analytics: OpenCV-based detection summaries and performance metrics

## Features

- Image inference with annotated bounding boxes
- Video inference with ByteTrack tracking and object counting
- Class summary tables for detections
- Per-object confidence breakdown
- Detection speed and object-count metrics
- Portfolio-ready Streamlit styling with aerial blue and slate gradients

## Project Structure

- `config.py` stores the project metadata used by `Home.py`
- `main.py` contains the Streamlit application
- `weights/best.pt` stores the trained model weights
- `3_🚁_Drone_Detection.py` is the wrapper page used by Streamlit navigation

## Run

```bash
streamlit run Home.py
```

The home page will automatically discover this project from `pages/drone_detection/config.py`.
