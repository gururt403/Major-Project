"""
Simplified FastAPI server for Heart Attack Detection System
Lazy-loads ML models to avoid long startup times on Windows
"""

import cv2
import torch
import numpy as np
from collections import deque
import mediapipe as mp
from pathlib import Path
import asyncio
import json
from datetime import datetime
import logging
import base64
from io import BytesIO

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

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


class SimplifiedDetectionSystem:
    """Simplified detection system without heavy ML models"""
    
    def __init__(self):
        logger.info("Initializing Simplified Detection System...")
        logger.info("Running in LIVE FEED mode - No AI processing for maximum performance")
        
        # Alert system
        self.alert_history = deque(maxlen=10)
        self.consecutive_alerts = 0
        
        # Current status
        self.current_hr = None
        self.current_posture = None
        self.current_alert = None
        self.last_alert_level = None  # Track last alert to avoid duplicates
        
        # Simulated heart rate (demo mode)
        self.demo_hr = 72
        self.demo_hr_trend = 1
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        
    def get_demo_heart_rate(self):
        """Generate a realistic demo heart rate that varies"""
        # Simulate realistic HR variations
        self.demo_hr += int(np.random.randint(-2, 3))
        self.demo_hr = int(np.clip(self.demo_hr, 50, 150))
        return int(self.demo_hr)
    
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
    
    def generate_alert(self, hr, hr_risk):
        """Generate alert based on heart rate"""
        alert_level = 'none'
        alert_message = ''
        
        # Abnormal heart rate
        if hr_risk == 'critical':
            alert_level = 'critical'
            alert_message = f"⚠️ CRITICAL: Abnormal HR ({hr:.1f} BPM)"
            self.consecutive_alerts += 1
        
        elif hr_risk == 'warning':
            alert_level = 'warning'
            alert_message = f"⚠️ WARNING: Elevated HR ({hr:.1f} BPM)"
            self.consecutive_alerts += 1
        
        # Normal conditions
        else:
            alert_level = 'normal'
            alert_message = f"✅ Normal: HR {hr:.1f} BPM"
            self.consecutive_alerts = max(0, self.consecutive_alerts - 1)
        
        # Trigger emergency if consecutive alerts
        if self.consecutive_alerts >= 5:
            alert_level = 'emergency'
            alert_message = "🚨 EMERGENCY: CALL FOR MEDICAL ASSISTANCE!"
        
        # Only add to history if alert level changed
        if alert_level != self.last_alert_level:
            self.alert_history.append({
                'level': alert_level,
                'message': alert_message,
                'timestamp': datetime.now().isoformat()
            })
            self.last_alert_level = alert_level
        
        return alert_level, alert_message
    
    def encode_frame_to_base64(self, frame):
        """Encode frame to base64 for transmission"""
        try:
            # Medium quality for balance between speed and quality (70)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return None
    
    def process_frame(self, frame):
        """Process a single frame"""
        face_detected = True  # Assume face detected to reduce lag
        
        # Skip face detection to reduce lag (detection adds 50-100ms per frame)
        # Uncomment below if you need face detection
        # results = self.face_detection.process(
        #     cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # if results.detections:
        #     face_detected = True
        
        # Use demo heart rate
        hr = self.get_demo_heart_rate()
        hr_risk, _ = self.assess_heart_rate_risk(hr)
        
        # Generate alert
        alert_level, alert_message = self.generate_alert(hr, hr_risk)
        
        # Update current status
        self.current_hr = hr
        self.current_posture = {
            'class': None,
            'name': 'demo_mode',
            'confidence': float(0.0)
        }
        self.current_alert = {
            'level': alert_level,
            'message': alert_message,
            'timestamp': datetime.now().isoformat()
        }
        
        self.frame_count += 1
        
        return {
            'heart_rate': int(hr) if hr else None,
            'hr_risk': hr_risk,
            'posture': self.current_posture,
            'alert': self.current_alert,
            'face_detected': bool(face_detected),
            'frame_count': int(self.frame_count)
        }
    
    async def stream_from_camera(self, websocket: WebSocket):
        """Stream real-time data from camera"""
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {Config.CAMERA_INDEX}")
            await websocket.send_json({'error': 'Could not open camera'})
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Camera stream started")
        streaming = True
        
        try:
            while streaming:
                # Check for incoming messages (non-blocking)
                try:
                    # Use a timeout to allow checking for messages
                    message = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=0.001  # Non-blocking check
                    )
                    
                    # Parse incoming message
                    try:
                        data = json.loads(message)
                        if data.get('type') == 'stop':
                            logger.info("Stop signal received from client")
                            streaming = False
                            break
                    except json.JSONDecodeError:
                        pass
                
                except asyncio.TimeoutError:
                    # No message received, continue streaming
                    pass
                except Exception as e:
                    logger.warning(f"Error receiving message: {e}")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Resize for processing
                frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
                
                # Process frame
                result = self.process_frame(frame)
                
                # Encode frame to base64
                frame_b64 = self.encode_frame_to_base64(frame)
                
                # Send data through WebSocket
                await websocket.send_json({
                    'type': 'frame',
                    'frame': frame_b64,
                    'heart_rate': result['heart_rate'],
                    'hr_risk': result['hr_risk'],
                    'posture': result['posture'],
                    'alert': result['alert'],
                    'face_detected': result['face_detected'],
                    'timestamp': datetime.now().isoformat()
                })
                
                # Minimal delay for smooth streaming at 30 FPS
                await asyncio.sleep(0.033)
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            cap.release()
            streaming = False
            logger.info("Camera stream stopped")


# Initialize detection system (light version)
detection_system = SimplifiedDetectionSystem()


# ============= API Endpoints =============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "demo",
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
    detection_system.consecutive_alerts = 0
    detection_system.alert_history.clear()
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
        "mode": "demo",
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
    logger.info("Starting Heart Attack Detection System Server (Demo Mode)...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.API_PORT,
        log_level="info"
    )
