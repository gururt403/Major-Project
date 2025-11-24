import os
from pathlib import Path

class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = PROJECT_ROOT / "models" / "trained_weights"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset paths
    PURE_DATASET = RAW_DATA_DIR / "rppg_datasets" / "PURE"
    UBFC_DATASET = RAW_DATA_DIR / "rppg_datasets" / "UBFC-rPPG"
    COHFACE_DATASET = RAW_DATA_DIR / "rppg_datasets" / "COHFACE"
    UPFALL_DATASET = RAW_DATA_DIR / "posture_datasets" / "UP-Fall"
    
    # Model parameters
    RPPG_FPS = 30  # Increased FPS for better temporal resolution
    RPPG_WINDOW_SIZE = 150  # 5 seconds at 30fps
    
    # MHAVH parameters
    SEQUENCE_LENGTH = 20  # Increased for better temporal context
    NUM_CLASSES = 4  # normal, chest_pain, breathing_difficulty, collapse
    BATCH_SIZE = 16  # Increased batch size for better GPU utilization
    NUM_EPOCHS = 150  # More epochs for better convergence
    LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning
    WEIGHT_DECAY = 0.0001  # L2 regularization
    DROPOUT = 0.3  # Dropout rate
    LABEL_SMOOTHING = 0.1  # Label smoothing for better generalization
    
    # Heart rate thresholds (BPM)
    HR_NORMAL_MIN = 60
    HR_NORMAL_MAX = 100
    HR_WARNING_MIN = 50
    HR_WARNING_MAX = 120
    HR_CRITICAL_MIN = 40
    HR_CRITICAL_MAX = 150
    
    # Alert system
    CONSECUTIVE_ALERT_THRESHOLD = 3
    
    # Camera settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # Class labels
    CLASS_NAMES = ['normal', 'chest_pain', 'breathing_difficulty', 'collapse']
    
    # Device
    DEVICE = 'cuda'  # or 'cpu'
    
    # API Server settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_DEBUG = True
    
    # CORS settings
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]