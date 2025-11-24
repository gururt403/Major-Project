"""
Remote Photoplethysmography (rPPG) Heart Rate Estimator
Implements CHROM and POS algorithms for contactless HR estimation
"""

import numpy as np
import cv2
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque


class rPPGEstimator:
    """
    Remote Photoplethysmography for heart rate estimation from facial video
    
    Attributes:
        fps (int): Frames per second of input video
        window_size (int): Number of frames to analyze (typically 5 seconds)
        frame_buffer (deque): Rolling buffer of RGB signals
    """
    
    def __init__(self, fps=30, window_size=150):
        """
        Initialize rPPG estimator
        
        Args:
            fps (int): Video frame rate (default: 30)
            window_size (int): Analysis window in frames (default: 150 = 5s at 30fps)
        """
        self.fps = fps
        self.window_size = window_size
        self.frame_buffer = deque(maxlen=window_size)
        
    def extract_rgb_signals(self, face_roi):
        """
        Extract mean RGB values from facial region of interest with improved ROI selection
        
        Args:
            face_roi (np.ndarray): Cropped face region (BGR format)
            
        Returns:
            np.ndarray: Mean RGB values [R, G, B] or None if invalid
        """
        if face_roi is None or face_roi.size == 0:
            return None
            
        h, w = face_roi.shape[:2]
        
        if h < 10 or w < 10:
            return None
        
        # Use multiple ROIs for robust signal extraction
        # Forehead region (best for rPPG)
        forehead = face_roi[int(h*0.15):int(h*0.35), int(w*0.25):int(w*0.75)]
        
        # Cheek regions (additional signal sources)
        left_cheek = face_roi[int(h*0.4):int(h*0.65), int(w*0.1):int(w*0.35)]
        right_cheek = face_roi[int(h*0.4):int(h*0.65), int(w*0.65):int(w*0.9)]
        
        rgb_signals = []
        
        for roi in [forehead, left_cheek, right_cheek]:
            if roi.size > 0:
                # Use median instead of mean for robustness to outliers
                r = np.median(roi[:, :, 2])
                g = np.median(roi[:, :, 1])
                b = np.median(roi[:, :, 0])
                rgb_signals.append([r, g, b])
        
        if len(rgb_signals) == 0:
            return None
        
        # Average signals from multiple ROIs
        return np.mean(rgb_signals, axis=0)
    
    def chrom_method(self, rgb_signals):
        """
        CHROM algorithm for rPPG signal extraction
        Reference: De Haan, G., & Jeanne, V. (2013). Robust pulse rate from 
        chrominance-based rPPG. IEEE TBME, 60(10), 2878-2886.
        
        Args:
            rgb_signals (np.ndarray): Array of RGB signals (frames x 3)
            
        Returns:
            np.ndarray: Extracted pulse signal
        """
        # Normalize RGB signals
        rgb_mean = np.mean(rgb_signals, axis=0)
        rgb_mean[rgb_mean == 0] = 1  # Avoid division by zero
        rgb_norm = rgb_signals / rgb_mean
        
        # CHROM color space transformation
        X = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
        Y = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
        
        # Bandpass filter (0.7-4 Hz corresponds to HR 42-240 BPM)
        X_filtered = self.bandpass_filter(X, 0.7, 4.0)
        Y_filtered = self.bandpass_filter(Y, 0.7, 4.0)
        
        # Calculate pulse signal with optimal weighting
        std_x = np.std(X_filtered)
        std_y = np.std(Y_filtered)
        
        if std_y == 0:
            alpha = 0
        else:
            alpha = std_x / std_y
            
        pulse_signal = X_filtered - alpha * Y_filtered
        
        return pulse_signal
    
    def pos_method(self, rgb_signals):
        """
        Plane-Orthogonal-to-Skin (POS) algorithm
        Reference: Wang, W., et al. (2017). Algorithmic principles of remote PPG. 
        IEEE TBME, 64(7), 1479-1491.
        
        Args:
            rgb_signals (np.ndarray): Array of RGB signals (frames x 3)
            
        Returns:
            np.ndarray: Extracted pulse signal
        """
        # Normalize RGB signals
        rgb_mean = np.mean(rgb_signals, axis=0)
        rgb_mean[rgb_mean == 0] = 1
        rgb_norm = rgb_signals / rgb_mean
        
        # POS projection
        S1 = rgb_norm[:, 1] - rgb_norm[:, 2]
        S2 = rgb_norm[:, 1] + rgb_norm[:, 2] - 2 * rgb_norm[:, 0]
        
        # Temporal normalization
        std_s1 = np.std(S1)
        std_s2 = np.std(S2)
        
        S1_norm = S1 / std_s1 if std_s1 != 0 else S1
        S2_norm = S2 / std_s2 if std_s2 != 0 else S2
        
        # Pulse signal
        pulse_signal = S1_norm + S2_norm
        pulse_signal = self.bandpass_filter(pulse_signal, 0.7, 4.0)
        
        return pulse_signal
    
    def bandpass_filter(self, signal_data, low_freq, high_freq):
        """
        Apply Butterworth bandpass filter to signal
        
        Args:
            signal_data (np.ndarray): Input signal
            low_freq (float): Lower cutoff frequency (Hz)
            high_freq (float): Upper cutoff frequency (Hz)
            
        Returns:
            np.ndarray: Filtered signal
        """
        if len(signal_data) < 20:
            return signal_data
            
        nyquist = 0.5 * self.fps
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure frequencies are in valid range (0, 1)
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, signal_data)
            return filtered
        except Exception as e:
            print(f"Warning: Filtering failed: {e}")
            return signal_data
    
    def estimate_heart_rate(self, pulse_signal):
        """
        Estimate heart rate using enhanced FFT with Welch's method for better frequency resolution
        
        Args:
            pulse_signal (np.ndarray): Extracted pulse signal
            
        Returns:
            float: Estimated heart rate in BPM or None if estimation fails
        """
        if len(pulse_signal) < 20:
            return None
            
        # Remove DC component and detrend
        pulse_signal = pulse_signal - np.mean(pulse_signal)
        pulse_signal = signal.detrend(pulse_signal)
        
        # Apply Hamming window to reduce spectral leakage
        window = np.hamming(len(pulse_signal))
        pulse_signal = pulse_signal * window
        
        # Use Welch's method for better spectral estimation
        try:
            freqs, psd = signal.welch(
                pulse_signal,
                fs=self.fps,
                nperseg=min(len(pulse_signal), 256),
                noverlap=min(len(pulse_signal)//2, 128),
                window='hamming'
            )
            
            # Filter to physiological heart rate range: 40-200 BPM (0.67-3.33 Hz)
            hr_mask = (freqs >= 0.67) & (freqs <= 3.33)
            hr_freqs = freqs[hr_mask]
            hr_psd = psd[hr_mask]
            
            if len(hr_psd) == 0:
                return None
            
            # Find peak frequency with quadratic interpolation for sub-bin accuracy
            peak_idx = np.argmax(hr_psd)
            
            # Parabolic interpolation for better accuracy
            if 0 < peak_idx < len(hr_psd) - 1:
                alpha = hr_psd[peak_idx - 1]
                beta = hr_psd[peak_idx]
                gamma = hr_psd[peak_idx + 1]
                p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                peak_freq = hr_freqs[peak_idx] + p * (hr_freqs[1] - hr_freqs[0])
            else:
                peak_freq = hr_freqs[peak_idx]
            
            # Convert frequency (Hz) to BPM
            heart_rate = peak_freq * 60.0
            
            # Additional validation: check if peak is prominent
            if hr_psd[peak_idx] < 2 * np.mean(hr_psd):
                return None
            
            return heart_rate
            
        except Exception as e:
            # Fallback to simple FFT if Welch's method fails
            n = len(pulse_signal)
            fft_values = fft(pulse_signal)
            fft_freqs = fftfreq(n, 1.0/self.fps)
            
            pos_mask = fft_freqs > 0
            fft_values = np.abs(fft_values[pos_mask])
            fft_freqs = fft_freqs[pos_mask]
            
            hr_mask = (fft_freqs >= 0.67) & (fft_freqs <= 3.33)
            hr_freqs = fft_freqs[hr_mask]
            hr_fft = fft_values[hr_mask]
            
            if len(hr_fft) == 0:
                return None
            
            peak_idx = np.argmax(hr_fft)
            peak_freq = hr_freqs[peak_idx]
            heart_rate = peak_freq * 60.0
            
            return heart_rate
    
    def process_frame(self, face_roi):
        """
        Process single frame and return heart rate estimate
        
        This is the main method to call for each video frame.
        
        Args:
            face_roi (np.ndarray): Cropped face region from current frame
            
        Returns:
            float: Estimated heart rate in BPM, or None if not enough data
        """
        # Extract RGB signals from current frame
        rgb = self.extract_rgb_signals(face_roi)
        
        if rgb is None:
            return None
        
        # Add to buffer
        self.frame_buffer.append(rgb)
        
        # Need full window for reliable estimate
        if len(self.frame_buffer) < self.window_size:
            return None
        
        # Convert buffer to numpy array
        rgb_signals = np.array(self.frame_buffer)
        
        # Extract pulse signal using CHROM method
        pulse_signal = self.chrom_method(rgb_signals)
        
        # Estimate heart rate
        hr = self.estimate_heart_rate(pulse_signal)
        
        # Validate heart rate is in reasonable physiological range
        if hr is not None and 40 <= hr <= 200:
            return hr
        
        return None
    
    def reset(self):
        """Clear the frame buffer (useful when switching videos or restarting)"""
        self.frame_buffer.clear()
    
    def get_buffer_size(self):
        """Get current buffer size"""
        return len(self.frame_buffer)
    
    def is_ready(self):
        """Check if enough frames have been collected for estimation"""
        return len(self.frame_buffer) >= self.window_size


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize estimator
    rppg = rPPGEstimator(fps=30, window_size=150)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # For demo: use entire frame as ROI (you should detect face first)
        hr = rppg.process_frame(frame)
        
        if hr:
            cv2.putText(frame, f"HR: {hr:.1f} BPM", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
        else:
            progress = rppg.get_buffer_size()
            cv2.putText(frame, f"Collecting data: {progress}/{rppg.window_size}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
        
        cv2.imshow('rPPG Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()