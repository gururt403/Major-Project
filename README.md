# Heart Attack Detection & Monitoring System

A real-time AI-powered health monitoring system that detects heart abnormalities and analyzes posture using advanced computer vision and deep learning techniques.

## 🚀 Features

- **Real-time Video Feed**: Smooth 30 FPS video streaming via WebSocket
- **Contactless Heart Rate Detection**: rPPG (remote photoplethysmography) technology for non-invasive heart rate monitoring
- **Posture Analysis**: AI-powered posture classification using enhanced MHAVH model
- **GPU Acceleration**: CUDA-enabled PyTorch for high-performance inference
- **Prediction Smoothing**: Exponential moving average and majority voting for stable predictions
- **Live Metrics Dashboard**: Real-time visualization of health indicators

## 🏗️ Architecture

### Backend
- **Framework**: FastAPI with WebSocket support
- **ML Models**: 
  - MHAVH (Multi-Head Attention Vision Health) - 33.4M parameters
  - ResNet50 backbone with attention layers and Bi-LSTM
  - rPPG estimator for heart rate detection
- **Computer Vision**: OpenCV, MediaPipe for face/mesh detection
- **GPU**: CUDA 12.4 with PyTorch 2.6.0+cu124

### Frontend
- **Framework**: Next.js 16.0.0 with React 19.2.0
- **UI**: Tailwind CSS with custom components
- **Real-time Communication**: WebSocket client for live data streaming
- **Rendering**: Turbopack for fast development

## 📋 Prerequisites

- **Python**: 3.12+
- **Node.js**: 18+ with pnpm
- **GPU**: NVIDIA GPU with CUDA 12.4 (optional but recommended)
- **Webcam**: For live video feed

## 🛠️ Installation

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the backend server:
```bash
python app.py
```

The backend will run on `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd v0-heart-attack-detection-ui
```

2. Install dependencies:
```bash
pnpm install
```

3. Start the development server:
```bash
pnpm run dev
```

The frontend will run on `http://localhost:3000`

## 🎯 Usage

1. Start both backend and frontend servers
2. Open browser to `http://localhost:3000`
3. Allow camera access when prompted
4. View real-time health metrics:
   - Heart rate (BPM)
   - Posture classification
   - Confidence scores
   - Live video feed

## 🧠 Model Details

### MHAVH Model
- **Parameters**: 33,407,685
- **Architecture**: 
  - ResNet50 backbone (pre-trained on ImageNet)
  - 3 Multi-Head Attention layers (8 heads, 512 dimensions)
  - Bi-directional LSTM (256 hidden units)
  - Sinusoidal positional encoding
  - Attention-based pooling
- **Input**: 224x224 RGB images
- **Output**: Posture classification with confidence scores

### rPPG Heart Rate Estimator
- **Method**: Remote photoplethysmography
- **Techniques**: 
  - Multiple ROI extraction from face
  - Welch's method for frequency analysis
  - Median filtering for noise reduction
- **Output**: Heart rate in BPM

## ⚙️ Configuration

### Backend Configuration (`backend/config.py`)
- Model paths
- Device settings (CPU/CUDA)
- Camera settings
- Processing parameters

### Performance Optimization
- **Frame Processing**: Every 5 frames (6 FPS analysis)
- **Video Streaming**: 30 FPS smooth display
- **Smoothing**: 
  - Heart Rate: 10-frame exponential moving average
  - Posture: 5-frame majority voting

## 📊 Training

Train the model with your own data:

```bash
cd backend
python train_simple.py
```

Training features:
- Synthetic data generation for testing
- Validation metrics tracking
- Model checkpointing
- GPU acceleration

## 🔧 API Endpoints

### WebSocket
- `ws://localhost:8000/ws/stream` - Real-time video and metrics streaming

### HTTP
- `GET /` - Health check
- `GET /health` - System health status

## 📁 Project Structure

```
Major-Project/
├── backend/
│   ├── app.py                 # Main FastAPI application
│   ├── mhavh_model.py        # MHAVH model architecture
│   ├── rppg_estimator.py     # Heart rate estimation
│   ├── mesh_detector.py      # Face mesh detection
│   ├── config.py             # Configuration settings
│   ├── train_simple.py       # Training script
│   ├── requirements.txt      # Python dependencies
│   ├── models/               # Model weights
│   └── data/                 # Training data
│
└── v0-heart-attack-detection-ui/
    ├── app/                  # Next.js app directory
    ├── components/           # React components
    ├── lib/                  # Utilities
    ├── public/               # Static assets
    └── package.json          # Node dependencies
```

## 🚨 Known Issues & Solutions

### Model Loading Warning
If you see "WARNING: Loaded model with random weights", this means the trained model checkpoint doesn't match the current architecture. Retrain the model using `train_simple.py`.

### Camera Access
Ensure browser has camera permissions. Use HTTPS in production for camera access.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is created for educational and research purposes.

## 🙏 Acknowledgments

- MediaPipe for face mesh detection
- PyTorch team for deep learning framework
- FastAPI for modern API framework
- Next.js team for the React framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**⚠️ Disclaimer**: This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
