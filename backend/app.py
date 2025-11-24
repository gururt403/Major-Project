import cv2
import torch
import numpy as np
from collections import deque
import mediapipe as mp
from torchvision import transforms
from pathlib import Path
import asyncio
import json
from datetime import datetime
import logging
import base64
from io import BytesIO

# Eager import scipy to avoid lazy loading issues on Windows
try:
    from scipy import signal, fft
except ImportError:
    pass

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from rppg_estimator import rPPGEstimator
from mhavh_model import MHAVH
from mesh_detector import MeshDetector
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Attack Detection System API",
    description="Real-time heart rate monitoring and posture analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HeartAttackDetectionSystem:
    """Complete heart attack detection system with WebSocket support"""
    
    def __init__(self, model_path=None):
        logger.info("Initializing Heart Attack Detection System...")
        
        # Initialize rPPG estimator
        self.rppg = rPPGEstimator(fps=Config.RPPG_FPS, 
                                  window_size=Config.RPPG_WINDOW_SIZE)
        
        # Initialize mesh detector
        self.mesh_detector = MeshDetector(max_history=15, fps=30)
        
        # Initialize device
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize MHAVH model
        self.posture_model = MHAVH(num_classes=Config.NUM_CLASSES,
                                   seq_length=Config.SEQUENCE_LENGTH).to(self.device)
        
        # Load trained weights if available
        if model_path and Path(model_path).exists():
            try:
                self.posture_model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using random weights.")
        else:
            logger.warning("No trained model loaded. Using random weights.")
        
        self.posture_model.eval()
        
        # Initialize face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Frame buffer for posture detection
        self.frame_buffer = deque(maxlen=Config.SEQUENCE_LENGTH)
        
        # Transform for posture model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Alert system
        self.alert_history = deque(maxlen=10)
        self.consecutive_alerts = 0
        
        # Current status
        self.current_hr = None
        self.current_posture = None
        self.current_alert = None
        self.current_mesh = None
        
        # Smoothing buffers for stable predictions
        self.hr_history = deque(maxlen=10)  # Last 10 heart rate readings
        self.posture_history = deque(maxlen=5)  # Last 5 posture predictions
        self.confidence_history = deque(maxlen=5)  # Last 5 confidence scores
        
        # Cache for last processed result with initial "analysing" state
        self.last_result = {
            'heart_rate': 'Analysing...',
            'hr_risk': 'Normal',
            'posture': {
                'class': 'Analysing...',
                'confidence': 0,
                'risk': 'Unknown'
            },
            'alert': {
                'level': 'info',
                'message': 'Initializing detection system...',
                'timestamp': datetime.now().isoformat()
            },
            'mesh': {
                'mesh_detected': False,
                'status': 'Analysing...',
                'current_mesh': None,
                'baseline_mesh': None,
                'difference': None,
                'metrics': {},
                'danger_frames': 0,
                'is_danger': False,
                'is_final_alert': False
            },
            'face_detected': False
        }
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.frame_count = 0
    
    def smooth_heart_rate(self, new_hr):
        """Apply exponential moving average to heart rate"""
        if new_hr is None:
            return self.current_hr
        
        self.hr_history.append(new_hr)
        
        # Use weighted average with more weight on recent values
        if len(self.hr_history) >= 3:
            weights = np.array([0.1, 0.2, 0.3, 0.4])[-len(self.hr_history):]
            weights = weights / weights.sum()
            smoothed_hr = np.average(list(self.hr_history)[-len(weights):], weights=weights)
            return round(smoothed_hr, 1)
        
        return new_hr
    
    def smooth_posture(self, new_class, new_confidence):
        """Apply majority voting to posture predictions"""
        if new_class is None:
            return self.current_posture['class'] if self.current_posture else None, \
                   self.current_posture['confidence'] if self.current_posture else 0
        
        self.posture_history.append(new_class)
        self.confidence_history.append(new_confidence)
        
        # Use majority voting if we have enough history
        if len(self.posture_history) >= 3:
            # Count occurrences
            from collections import Counter
            counts = Counter(self.posture_history)
            most_common_class = counts.most_common(1)[0][0]
            
            # Average confidence for the most common class
            conf_sum = sum(c for p, c in zip(self.posture_history, self.confidence_history) 
                          if p == most_common_class)
            conf_count = sum(1 for p in self.posture_history if p == most_common_class)
            avg_confidence = conf_sum / conf_count if conf_count > 0 else new_confidence
            
            return most_common_class, avg_confidence
        
        return new_class, new_confidence
        
    def assess_heart_rate_risk(self, hr):
        """Assess risk level based on heart rate"""
        if hr is None:
            return 'unknown', 0
        
        if Config.HR_NORMAL_MIN <= hr <= Config.HR_NORMAL_MAX:
            return 'normal', 0
        elif Config.HR_WARNING_MIN <= hr < Config.HR_NORMAL_MIN or \
             Config.HR_NORMAL_MAX < hr <= Config.HR_WARNING_MAX:
            return 'warning', 1
        else:
            return 'critical', 2
    
    def process_posture(self, frames):
        """Process frame sequence for posture classification"""
        if len(frames) < Config.SEQUENCE_LENGTH:
            return None, None
        
        try:
            # Prepare frames
            processed_frames = []
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = self.transform(frame_rgb)
                processed_frames.append(frame_tensor)
            
            # Stack and add batch dimension
            input_tensor = torch.stack(processed_frames).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output, _ = self.posture_model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Error processing posture: {e}")
            return None, None
    
    def generate_alert(self, hr, hr_risk, posture_class, posture_conf, mesh_status=None):
        """Generate alert based on combined risk assessment"""
        alert_level = 'none'
        alert_message = ''
        
        # Critical mesh status (facial strain)
        if mesh_status and mesh_status.get('is_final_alert'):
            alert_level = 'critical'
            alert_message = f"🚨 CRITICAL FACIAL STRAIN DETECTED! Status: {mesh_status.get('status', 'Unknown')}"
            self.consecutive_alerts += 1
        
        # Danger mesh status
        elif mesh_status and mesh_status.get('is_danger'):
            alert_level = 'warning'
            alert_message = f"⚠️ DANGER DETECTED! {mesh_status.get('status', 'Unknown')}"
            self.consecutive_alerts += 1
        
        # Critical posture detected
        elif posture_class in [2, 3] and posture_conf > 0.7:
            alert_level = 'critical'
            alert_message = f"CRITICAL: {Config.CLASS_NAMES[posture_class].upper().replace('_', ' ')} DETECTED!"
            self.consecutive_alerts += 1
        
        # Abnormal heart rate with warning posture
        elif hr_risk == 'critical' or (hr_risk == 'warning' and posture_class == 1):
            alert_level = 'warning'
            if posture_class is not None:
                alert_message = f"WARNING: Abnormal HR ({hr:.1f} BPM) + {Config.CLASS_NAMES[posture_class]}"
            else:
                alert_message = f"WARNING: Abnormal HR ({hr:.1f} BPM)"
            self.consecutive_alerts += 1
        
        # Normal conditions
        else:
            self.consecutive_alerts = max(0, self.consecutive_alerts - 1)
        
        # Trigger emergency if consecutive alerts
        if self.consecutive_alerts >= Config.CONSECUTIVE_ALERT_THRESHOLD:
            alert_level = 'emergency'
            alert_message = "🚨 EMERGENCY: CALL FOR MEDICAL ASSISTANCE!"
        
        self.alert_history.append({
            'level': alert_level,
            'message': alert_message,
            'timestamp': datetime.now().isoformat()
        })
        
        return alert_level, alert_message
    
    def encode_frame_to_base64(self, frame):
        """Encode frame to base64 for transmission"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return None
    
    def process_frame(self, frame):
        """Process a single frame"""
        hr = None
        hr_risk = 'unknown'
        posture_class = None
        posture_conf = 0
        face_detected = False
        mesh_result = None
        
        # Process mesh detection
        mesh_result = self.mesh_detector.process_frame(frame)
        
        # Detect face for heart rate estimation
        results = self.face_detection.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.detections:
            face_detected = True
            # Extract face ROI for rPPG
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            
            # Extract face ROI
            face_roi = frame[max(0, y):y+h, max(0, x):x+w]
            
            if face_roi.size > 0:
                # Estimate heart rate
                raw_hr = self.rppg.process_frame(face_roi)
                if raw_hr:
                    # Apply smoothing to heart rate
                    hr = self.smooth_heart_rate(raw_hr)
                    hr_risk, _ = self.assess_heart_rate_risk(hr)
        
        # Add frame to buffer for posture detection
        self.frame_buffer.append(frame)
        
        # Process posture every SEQUENCE_LENGTH frames
        if len(self.frame_buffer) == Config.SEQUENCE_LENGTH:
            raw_posture_class, raw_posture_conf = self.process_posture(
                list(self.frame_buffer))
            
            # Apply smoothing to posture predictions
            posture_class, posture_conf = self.smooth_posture(raw_posture_class, raw_posture_conf)
        
        # Generate alert with mesh data (use smoothed values)
        alert_level, alert_message = self.generate_alert(
            hr, hr_risk, posture_class, posture_conf, mesh_result)
        
        # Update current status with smoothed values
        self.current_hr = hr
        self.current_posture = {
            'class': posture_class,
            'name': Config.CLASS_NAMES[posture_class] if posture_class is not None else 'unknown',
            'confidence': round(posture_conf, 3) if posture_conf else 0
        }
        self.current_alert = {
            'level': alert_level,
            'message': alert_message,
            'timestamp': datetime.now().isoformat()
        }
        self.current_mesh = mesh_result
        
        self.frame_count += 1
        
        return {
            'heart_rate': hr,
            'hr_risk': hr_risk,
            'posture': self.current_posture,
            'alert': self.current_alert,
            'mesh': mesh_result,
            'face_detected': face_detected,
            'frame_count': self.frame_count
        }
    
    async def stream_from_camera(self, websocket: WebSocket):
        """Stream real-time data from camera with optimized processing"""
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {Config.CAMERA_INDEX}")
            await websocket.send_json({'error': 'Could not open camera'})
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        logger.info("Camera stream started")
        
        frame_count = 0
        process_every_n_frames = 5  # Process metrics every 5 frames (6 FPS analysis)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Resize for display
                frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
                
                # Only process metrics every N frames to reduce lag
                if frame_count % process_every_n_frames == 0:
                    result = self.process_frame(frame)
                    # Update cached results
                    self.last_result = result
                
                # Always encode and send the frame for smooth video
                frame_b64 = self.encode_frame_to_base64(frame)
                
                # Send data through WebSocket with last known metrics
                data = {
                    'type': 'frame',
                    'frame': frame_b64,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Always send the last known metrics (cached from processing)
                if hasattr(self, 'last_result') and self.last_result:
                    data.update({
                        'heart_rate': self.last_result['heart_rate'],
                        'hr_risk': self.last_result['hr_risk'],
                        'posture': self.last_result['posture'],
                        'alert': self.last_result['alert'],
                        'mesh': {
                            'detected': self.last_result['mesh']['mesh_detected'],
                            'status': self.last_result['mesh']['status'],
                            'current_mesh': self.last_result['mesh']['current_mesh'],
                            'baseline_mesh': self.last_result['mesh']['baseline_mesh'],
                            'difference': self.last_result['mesh']['difference'],
                            'metrics': self.last_result['mesh']['metrics'],
                            'danger_frames': self.last_result['mesh']['danger_frames'],
                            'is_danger': self.last_result['mesh']['is_danger'],
                            'is_final_alert': self.last_result['mesh']['is_final_alert']
                        },
                        'face_detected': self.last_result['face_detected']
                    })
                
                await websocket.send_json(data)
                
                frame_count += 1
                
                # Minimal delay for smooth 30 FPS video
                await asyncio.sleep(0.01)  # ~100 FPS max, limited by camera
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            cap.release()
            logger.info("Camera stream stopped")


# Initialize detection system with newly trained model
model_path = Path("models/trained_weights/training_20251124_215016/best_model.pth")
detection_system = HeartAttackDetectionSystem(model_path=str(model_path))


# ============= API Endpoints =============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(detection_system.device),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return {
        "heart_rate": detection_system.current_hr,
        "posture": detection_system.current_posture,
        "alert": detection_system.current_alert,
        "frame_count": detection_system.frame_count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/alerts")
async def get_alerts(limit: int = 10):
    """Get recent alerts"""
    alerts = list(detection_system.alert_history)[-limit:]
    return {
        "alerts": alerts,
        "count": len(alerts),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/reset")
async def reset_system():
    """Reset the detection system"""
    detection_system.rppg.reset()
    detection_system.frame_buffer.clear()
    detection_system.consecutive_alerts = 0
    return {
        "status": "reset",
        "timestamp": datetime.now().isoformat()
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time video and sensor streaming"""
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        await detection_system.stream_from_camera(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        logger.info("WebSocket client disconnected")


# ============= Root Endpoint =============

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "Heart Attack Detection System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /api/health",
            "status": "GET /api/status",
            "alerts": "GET /api/alerts",
            "reset": "POST /api/reset",
            "stream": "WS /ws/stream"
        },
        "docs": "/docs"
    }


# ============= Server Runner =============

if __name__ == "__main__":
    logger.info("Starting Heart Attack Detection System Server...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.API_PORT,
        log_level="info"
    )
