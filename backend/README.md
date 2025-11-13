# Heart Attack Detection System

Real-time heart attack detection using rPPG and posture analysis with AI.

## Quick Start

### 1. Setup Environment
```bash
# Clone/create project directory
mkdir heart_attack_detection
cd heart_attack_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

**rPPG Datasets:**
- PURE: https://www.tu-ilmenau.de (search: PURE dataset)
- UBFC-rPPG: https://sites.google.com/view/ybenezeth/ubfcrppg
- COHFACE: https://www.idiap.ch/en/dataset/cohface

**Posture Datasets:**
- UP-Fall: https://sites.google.com/up.edu.mx/har-up/
- NTU RGB+D: https://rose1.ntu.edu.sg/dataset/actionRecognition/

Place datasets in `data/raw/` following the project structure.

### 3. Train Model
```bash
python training/train_mhavh.py
```

### 4. Run Detection System
```bash
# With webcam
python main.py

# With video file
python main.py --video path/to/video.mp4
```

## Project Structure

heart_attack_detection/
├── models/              # Model implementations
├── training/            # Training scripts
├── utils/               # Utilities and evaluation
├── data/                # Datasets
├── main.py              # Main detection system
└── config.py            # Configuration

## Features
- ✅ Real-time heart rate estimation (rPPG)
- ✅ Posture-based risk detection (MHAVH)
- ✅ Multi-level alert system
- ✅ Contactless monitoring
- ✅ GPU acceleration support

## Controls
- `q`: Quit
- `r`: Reset system

## Requirements
- Python 3.8+
- CUDA (optional, for GPU)
- Webcam or video file

## Citation
Based on research from:
- Lee et al. (2024) - rPPG
- Naz et al. (2024) - MHAVH