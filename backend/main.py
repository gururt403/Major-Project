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

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from rppg_estimator import rPPGEstimator
from mhavh_model import MHAVH
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeartAttackDetectionSystem:
    """Complete heart attack detection system with WebSocket support"""
    
    def __init__(self, model_path=None):
        logger.info("Initializing Heart Attack Detection System...")
        
        # Initialize rPPG estimator
        self.rppg = rPPGEstimator(fps=Config.RPPG_FPS, 
                                  window_size=Config.RPPG_WINDOW_SIZE)
        
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
        
        # WebSocket connections
        self.websocket_clients = set()
        
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
    
    def generate_alert(self, hr, hr_risk, posture_class, posture_conf):
        """Generate alert based on combined risk assessment"""
        alert_level = 'none'
        alert_message = ''
        
        # Critical posture detected
        if posture_class in [2, 3] and posture_conf > 0.7:
            alert_level = 'critical'
            alert_message = f"CRITICAL: {Config.CLASS_NAMES[posture_class].upper().replace('_', ' ')} DETECTED!"
            self.consecutive_alerts += 1
        
        # Abnormal heart rate with warning posture
        elif hr_risk == 'critical' or (hr_risk == 'warning' and posture_class == 1):
            alert_level = 'warning'
            alert_message = f"WARNING: Abnormal HR ({hr:.1f} BPM) + {Config.CLASS_NAMES[posture_class]}"
            self.consecutive_alerts += 1
        
        # Normal conditions
        else:
            self.consecutive_alerts = max(0, self.consecutive_alerts - 1)
        
        # Trigger emergency if consecutive alerts
        if self.consecutive_alerts >= Config.CONSECUTIVE_ALERT_THRESHOLD:
            alert_level = 'emergency'
            alert_message = "🚨 EMERGENCY: CALL FOR MEDICAL ASSISTANCE!"
        
        self.alert_history.append(alert_level)
        
        return alert_level, alert_message
    
    def draw_info(self, frame, hr, hr_risk, posture_class, posture_conf, alert_level, alert_message):
        """Draw information overlay on frame"""
        display_frame = frame.copy()
        y_offset = 30
        
        # Heart rate
        if hr:
            color = (0, 255, 0) if hr_risk == 'normal' else \
                   (0, 255, 255) if hr_risk == 'warning' else (0, 0, 255)
            cv2.putText(display_frame, f"HR: {hr:.1f} BPM ({hr_risk})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2)
            y_offset += 35
        
        # Posture
        if posture_class is not None:
            posture_text = f"Posture: {Config.CLASS_NAMES[posture_class]} ({posture_conf:.2f})"
            cv2.putText(display_frame, posture_text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 0), 2)
            y_offset += 35
        
        # Alert
        if alert_level != 'none':
            alert_color = (0, 165, 255) if alert_level == 'warning' else \
                         (0, 0, 255) if alert_level == 'critical' else (255, 0, 255)
            
            cv2.putText(display_frame, alert_message, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, alert_color, 2)
            
            # Draw alert border
            if alert_level in ['critical', 'emergency']:
                thickness = 15 if alert_level == 'emergency' else 10
                cv2.rectangle(display_frame, (0, 0), 
                            (display_frame.shape[1], display_frame.shape[0]), 
                            alert_color, thickness)
        
        return display_frame
    
    def run(self, video_source=0):
        """
        Run the detection system
        
        Args:
            video_source: Camera index (0) or video file path
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        print("Heart Attack Detection System Started")
        print("Press 'q' to quit, 'r' to reset")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face
            results = self.face_detection.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            hr = None
            hr_risk = 'unknown'
            posture_class = None
            posture_conf = 0
            
            if results.detections:
                # Extract face ROI for rPPG
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Draw face box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract face ROI
                face_roi = frame[max(0, y):y+h, max(0, x):x+w]
                
                if face_roi.size > 0:
                    # Estimate heart rate
                    hr = self.rppg.process_frame(face_roi)
                    if hr:
                        hr_risk, _ = self.assess_heart_rate_risk(hr)
            
            # Add frame to buffer for posture detection
            self.frame_buffer.append(frame)
            
            # Process posture every SEQUENCE_LENGTH frames
            if len(self.frame_buffer) == Config.SEQUENCE_LENGTH:
                posture_class, posture_conf = self.process_posture(
                    list(self.frame_buffer))
            
            # Generate alert
            alert_level, alert_message = self.generate_alert(
                hr, hr_risk, posture_class, posture_conf)
            
            # Draw information on frame
            display_frame = self.draw_info(
                frame, hr, hr_risk, posture_class, posture_conf,
                alert_level, alert_message)
            
            # Show frame
            cv2.imshow('Heart Attack Detection System', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.rppg.reset()
                self.frame_buffer.clear()
                print("System reset")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize system
    model_path = Config.MODEL_DIR / "mhavh_posture_model.pth"
    system = HeartAttackDetectionSystem(model_path=model_path)
    
    # Run with webcam (0) or specify video file path
    system.run(video_source=0)