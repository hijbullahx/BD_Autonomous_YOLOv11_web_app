# Bangladesh Traffic Perception System

## Project Structure
```
autonomous_vehicle/
├── config.py           # Project configuration (auto-loaded by Home.py)
├── main.py            # Streamlit application page
├── weights/           # Model weights
│   └── best.pt       # YOLOv11 + CBAM trained model
└── README.md          # This file
```

## Model Details
- **Architecture**: YOLOv11 Medium + CBAM Attention
- **Dataset**: Local Traffic
- **Performance**: mAP@50 ~75%
- **Classes**: Rickshaw, CNG, Bus, Truck, Car, Cycle, Bike, Mini-Truck, People

## How to Update Project Info
Edit `config.py` to change:
- Title
- Description
- Technologies
- Status

Changes will automatically reflect on the home page!
