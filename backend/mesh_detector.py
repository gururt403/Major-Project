"""
Face Mesh Detection and Analysis Module
Detects facial mesh landmarks and analyzes facial metrics for stress/fatigue detection
Based on MediaPipe Face Mesh with custom metric calculations
"""

import cv2
import mediapipe as mp
import math
from collections import deque
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh


class MeshDetector:
    """
    Detects facial mesh landmarks and analyzes facial strain metrics
    to assess stress levels and fatigue indicators
    """
    
    def __init__(self, max_history: int = 15, fps: int = 30):
        """
        Initialize mesh detector
        
        Args:
            max_history: Number of frames to keep for smoothing
            fps: Frames per second for timing calculations
        """
        self.history = deque(maxlen=max_history)
        self.baseline_mesh = None
        self.fps = fps
        self.frame_count = 0
        
        # Thresholds for risk assessment
        self.THRESH_EFFORT = 4          # Normal gym effort
        self.THRESH_RISK = 10           # Warning level
        self.THRESH_WARN = 15           # Critical warning
        
        # Danger detection timer
        self.danger_frames = 0
        self.DANGER_LIMIT = 2 * fps * 60  # 2 minutes continuous danger
        
        # Initialize MediaPipe FaceMesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for metric calculations
        self.CHEEK_LEFT = 50
        self.CHEEK_RIGHT = 280
        self.JAW_LEFT = 127
        self.JAW_RIGHT = 356
        self.EYE_UP = 159
        self.EYE_DOWN = 145
        self.EYE_LEFT = 33
        self.EYE_RIGHT = 133
        self.MOUTH_UP = 13
        self.MOUTH_DOWN = 14
        self.MOUTH_LEFT = 78
        self.MOUTH_RIGHT = 308
    
    def dist_px(self, a: 'mp.solutions.face_mesh.NormalizedLandmark', 
                b: 'mp.solutions.face_mesh.NormalizedLandmark', 
                w: int, h: int) -> float:
        """
        Calculate distance between two landmarks in pixels
        
        Args:
            a: First landmark
            b: Second landmark
            w: Frame width
            h: Frame height
            
        Returns:
            Distance in pixels
        """
        x1, y1 = int(a.x * w), int(a.y * h)
        x2, y2 = int(b.x * w), int(b.y * h)
        return math.hypot(x2 - x1, y2 - y1)
    
    def calc_eye_ratio(self, landmarks: List, w: int, h: int) -> float:
        """
        Calculate eye opening ratio (vertical/horizontal)
        Higher ratio = more open, lower = closing (fatigue)
        
        Args:
            landmarks: List of facial landmarks
            w: Frame width
            h: Frame height
            
        Returns:
            Eye opening ratio (0-1+)
        """
        vert = self.dist_px(landmarks[self.EYE_UP], landmarks[self.EYE_DOWN], w, h)
        horiz = self.dist_px(landmarks[self.EYE_LEFT], landmarks[self.EYE_RIGHT], w, h)
        return (vert / horiz * 100) if horiz > 0 else 0.0
    
    def calc_mouth_ratio(self, landmarks: List, w: int, h: int) -> float:
        """
        Calculate mouth opening ratio
        Indicates stress/strain levels
        
        Args:
            landmarks: List of facial landmarks
            w: Frame width
            h: Frame height
            
        Returns:
            Mouth opening ratio
        """
        vert = self.dist_px(landmarks[self.MOUTH_UP], landmarks[self.MOUTH_DOWN], w, h)
        horiz = self.dist_px(landmarks[self.MOUTH_LEFT], landmarks[self.MOUTH_RIGHT], w, h)
        return (vert / horiz * 100) if horiz > 0 else 0.0
    
    def calc_cheek_distance(self, landmarks: List, w: int, h: int) -> float:
        """
        Calculate cheek-to-cheek distance
        Indicates facial tension
        
        Args:
            landmarks: List of facial landmarks
            w: Frame width
            h: Frame height
            
        Returns:
            Cheek distance in pixels
        """
        return self.dist_px(landmarks[self.CHEEK_LEFT], landmarks[self.CHEEK_RIGHT], w, h)
    
    def calc_jaw_distance(self, landmarks: List, w: int, h: int) -> float:
        """
        Calculate jaw width
        Indicates clenching/stress
        
        Args:
            landmarks: List of facial landmarks
            w: Frame width
            h: Frame height
            
        Returns:
            Jaw distance in pixels
        """
        return self.dist_px(landmarks[self.JAW_LEFT], landmarks[self.JAW_RIGHT], w, h)
    
    def calc_mesh_metric(self, landmarks: List, w: int, h: int) -> float:
        """
        Calculate combined mesh metric
        Weighted combination of multiple facial metrics
        
        Args:
            landmarks: List of facial landmarks
            w: Frame width
            h: Frame height
            
        Returns:
            Combined mesh metric score
        """
        cheek_dist = self.calc_cheek_distance(landmarks, w, h)
        jaw_dist = self.calc_jaw_distance(landmarks, w, h)
        eye_ratio = self.calc_eye_ratio(landmarks, w, h)
        mouth_ratio = self.calc_mouth_ratio(landmarks, w, h)
        
        # Weighted combination of metrics
        combined = (
            0.45 * cheek_dist +      # Facial tension
            0.25 * jaw_dist +        # Jaw clenching
            0.15 * eye_ratio +       # Eye opening
            0.15 * mouth_ratio       # Mouth opening
        )
        
        return combined
    
    def process_frame(self, frame: 'cv2.Mat') -> Dict:
        """
        Process a frame for mesh detection and analysis
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dictionary with detection results:
            {
                'mesh_detected': bool,
                'current_mesh': float,
                'baseline_mesh': float,
                'difference': float,
                'status': str,
                'status_color': tuple,
                'metrics': {
                    'cheek_dist': float,
                    'jaw_dist': float,
                    'eye_ratio': float,
                    'mouth_ratio': float
                },
                'danger_frames': int,
                'is_danger': bool,
                'is_final_alert': bool,
                'landmarks': list  # Facial landmarks if detected
            }
        """
        h, w, c = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        # Default values
        current_mesh = 0.0
        diff = 0.0
        status = "No Face"
        status_color = (0, 255, 255)  # Cyan
        landmarks = None
        is_danger = False
        is_final_alert = False
        metrics = {
            'cheek_dist': 0.0,
            'jaw_dist': 0.0,
            'eye_ratio': 0.0,
            'mouth_ratio': 0.0
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            landmarks = face_landmarks
            
            # Calculate individual metrics
            cheek_dist = self.calc_cheek_distance(face_landmarks, w, h)
            jaw_dist = self.calc_jaw_distance(face_landmarks, w, h)
            eye_ratio = self.calc_eye_ratio(face_landmarks, w, h)
            mouth_ratio = self.calc_mouth_ratio(face_landmarks, w, h)
            
            metrics = {
                'cheek_dist': cheek_dist,
                'jaw_dist': jaw_dist,
                'eye_ratio': eye_ratio,
                'mouth_ratio': mouth_ratio
            }
            
            # Calculate combined mesh metric
            combined = self.calc_mesh_metric(face_landmarks, w, h)
            self.history.append(combined)
            
            # Calculate smoothed current mesh
            current_mesh = sum(self.history) / len(self.history)
            
            # Lock baseline after collecting enough samples
            if self.baseline_mesh is None and len(self.history) == self.history.maxlen:
                self.baseline_mesh = current_mesh
                logger.info(f"[MESH] Baseline locked at: {self.baseline_mesh:.2f}")
            
            # Calculate difference from baseline
            if self.baseline_mesh is not None:
                diff = current_mesh - self.baseline_mesh
                abs_diff = abs(diff)
                
                # Risk assessment based on deviation
                if abs_diff < self.THRESH_EFFORT:
                    status = "Stable"
                    status_color = (0, 255, 0)  # Green
                    self.danger_frames = 0
                
                elif abs_diff < self.THRESH_RISK:
                    status = "Effort (Normal Gym)"
                    status_color = (0, 255, 255)  # Cyan
                    self.danger_frames = 0
                
                elif abs_diff < self.THRESH_WARN:
                    status = "Risk (Warning)"
                    status_color = (0, 165, 255)  # Orange
                    self.danger_frames = 0
                
                else:  # Critical danger
                    self.danger_frames += 1
                    is_danger = True
                    status_color = (0, 0, 255)  # Red
                    
                    if self.danger_frames > self.DANGER_LIMIT:
                        status = "FINAL ALERT"
                        is_final_alert = True
                    else:
                        status = "DANGER"
            else:
                status = "Calibrating..."
                status_color = (255, 255, 255)  # White
        
        self.frame_count += 1
        
        return {
            'mesh_detected': results.multi_face_landmarks is not None,
            'current_mesh': current_mesh,
            'baseline_mesh': self.baseline_mesh,
            'difference': diff,
            'status': status,
            'status_color': status_color,
            'metrics': metrics,
            'danger_frames': self.danger_frames,
            'is_danger': is_danger,
            'is_final_alert': is_final_alert,
            'landmarks': landmarks,
            'frame_count': self.frame_count
        }
    
    def draw_landmarks(self, frame: 'cv2.Mat', landmarks: List, 
                      color: Tuple[int, int, int] = (0, 255, 0),
                      radius: int = 1) -> 'cv2.Mat':
        """
        Draw facial landmarks on frame
        
        Args:
            frame: Input frame
            landmarks: List of landmarks
            color: Color for landmarks (BGR)
            radius: Radius of circles
            
        Returns:
            Frame with drawn landmarks
        """
        if landmarks is None:
            return frame
        
        h, w, c = frame.shape
        frame_copy = frame.copy()
        
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_copy, (x, y), radius, color, -1)
        
        return frame_copy
    
    def draw_metrics(self, frame: 'cv2.Mat', result: Dict) -> 'cv2.Mat':
        """
        Draw metrics overlay on frame
        
        Args:
            frame: Input frame
            result: Detection result dictionary
            
        Returns:
            Frame with drawn metrics
        """
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        # Position for text
        x0, y0 = 10, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_color = (200, 200, 200)
        
        # Baseline info
        baseline_text = (
            f"Baseline: {result['baseline_mesh']:.2f}" 
            if result['baseline_mesh'] is not None 
            else "Baseline: --"
        )
        cv2.putText(frame_copy, baseline_text, (x0, y0), font, font_scale, text_color, thickness)
        y0 += 25
        
        # Current mesh
        cv2.putText(
            frame_copy, 
            f"Current: {result['current_mesh']:.2f}", 
            (x0, y0), font, font_scale, (255, 255, 255), thickness
        )
        y0 += 25
        
        # Difference
        diff_color = (0, 255, 0) if abs(result['difference']) < self.THRESH_EFFORT else (0, 0, 255)
        cv2.putText(
            frame_copy, 
            f"Difference: {result['difference']:.2f}", 
            (x0, y0), font, font_scale, diff_color, thickness
        )
        y0 += 25
        
        # Status
        cv2.putText(
            frame_copy, 
            f"Status: {result['status']}", 
            (x0, y0), font, 0.7, result['status_color'], thickness
        )
        
        # Danger timer if applicable
        if result['is_danger']:
            danger_time = result['danger_frames'] / self.fps
            cv2.putText(
                frame_copy,
                f"Danger Time: {danger_time:.1f}s",
                (x0, h - 30),
                font, 0.6, (0, 0, 255), 2
            )
        
        # Final alert overlay
        if result['is_final_alert']:
            cv2.putText(
                frame_copy, 
                "FINAL ALERT", 
                (int(w * 0.18), int(h * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (0, 0, 255), 6
            )
        
        return frame_copy
    
    def reset(self):
        """Reset detector state"""
        self.history.clear()
        self.baseline_mesh = None
        self.danger_frames = 0
        self.frame_count = 0
        logger.info("[MESH] Detector reset")
