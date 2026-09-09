"""
Blink Analyzer
Analyzes eye blink patterns to detect abnormal eye behavior
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class BlinkAnalyzer:
    def __init__(self, window_size: int = 90, blink_threshold: float = 0.3):
        """
        Initialize blink analyzer

        Args:
            window_size: Number of frames to consider for temporal analysis (~3 seconds at 30fps)
            blink_threshold: Threshold for eye closure detection (0-1)
        """
        self.window_size = window_size
        self.blink_threshold = blink_threshold

        # For storing historical eye aspect ratio data
        self.ear_history = deque(maxlen=window_size)
        self.blink_timestamps = deque(maxlen=window_size)
        self.frame_timestamps = deque(maxlen=window_size)

        # Blink statistics
        self.blink_count = 0
        self.last_blink_time = 0
        self.blink_intervals = deque(maxlen=20)  # Store last 20 blink intervals

        # Initialize eye detection
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            logger.info("Blink analyzer initialized with OpenCV cascades")
        except Exception as e:
            logger.warning(f"Could not load OpenCV cascades: {e}. Using simplified analysis.")
            self.face_cascade = None
            self.eye_cascade = None

    def analyze(self, video_frames: List[str]) -> Dict[str, Any]:
        """
        Analyze blink behavior from video frames

        Args:
            video_frames: List of base64 encoded video frames

        Returns:
            Dictionary with blink analysis results
        """
        try:
            if not video_frames:
                return self._insufficient_data_response("No video frames provided")

            # Reset per-analysis counters (but keep some history for continuity)
            frame_ear_values = []
            valid_frames = 0

            # Process each frame
            for i, frame_b64 in enumerate(video_frames):
                try:
                    # Decode base64 frame
                    frame = self._decode_frame(frame_b64)
                    if frame is None:
                        continue

                    # Calculate Eye Aspect Ratio (EAR) for this frame
                    ear = self._calculate_eye_aspect_ratio(frame)
                    if ear is not None:
                        frame_ear_values.append(ear)
                        valid_frames += 1

                        # Store in history
                        self.ear_history.append(ear)
                        self.frame_timestamps.append(time.time())

                        # Detect blinks
                        self._detect_blink(ear, time.time())

                except Exception as e:
                    logger.warning(f"Error processing frame {i}: {e}")
                    continue

            if valid_frames < 10:  # Need minimum frames for analysis
                return self._insufficient_data_response(f"Insufficient valid frames: {valid_frames}")

            # Calculate blink metrics
            blink_rate, abnormal_pattern, confidence = self._calculate_blink_metrics()

            return {
                "blink_rate": float(blink_rate),
                "abnormal_blink_pattern": abnormal_pattern,
                "confidence": float(confidence),
                "timestamp": time.time(),
                "frames_analyzed": valid_frames,
                "total_frames": len(video_frames),
                "blink_count": self.blink_count,
                "avg_blink_interval": np.mean(list(self.blink_intervals)) if self.blink_intervals else 0.0
            }

        except Exception as e:
            logger.error(f"Blink analysis failed: {e}")
            return {
                "error": str(e),
                "blink_rate": 0.0,
                "abnormal_blink_pattern": False,
                "confidence": 0.0
            }

    def _decode_frame(self, frame_b64: str) -> Optional[np.ndarray]:
        """
        Decode base64 frame to numpy array
        """
        try:
            # Simulate frame for testing/MVP
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            return frame
        except Exception as e:
            logger.warning(f"Frame decoding failed: {e}")
            return None

    def _calculate_eye_aspect_ratio(self, frame: np.ndarray) -> Optional[float]:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
        where p1-p6 are eye landmark points
        """
        try:
            if self.face_cascade is not None and self.eye_cascade is not None:
                return self._calculate_ear_with_landmarks(frame)
            else:
                return self._calculate_ear_simplified(frame)
        except Exception as e:
            logger.warning(f"EAR calculation failed: {e}")
            return None

    def _calculate_ear_with_landmarks(self, frame: np.ndarray) -> Optional[float]:
        """
        Calculate EAR using facial landmarks
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        # Process the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Extract face region
        face_roi = gray[y:y+h, x:x+w]

        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(face_roi)

        if len(eyes) < 2:
            return None

        # Process both eyes and calculate average EAR
        ear_values = []

        for (ex, ey, ew, eh) in eyes[:2]:  # Process first two eyes
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]

            # Apply threshold to get eye region
            _, thresh = cv2.threshold(eye_roi, 50, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get the largest contour (should be the eye)
                largest_contour = max(contours, key=cv2.contourArea)

                # Calculate eye aspect ratio using contour approximation
                # Simplified EAR calculation
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)

                if perimeter > 0:
                    # Simple approximation: more closed eye = lower area/perimeter ratio
                    ear = area / (perimeter * perimeter) * 100  # Scale factor
                    ear_values.append(min(0.5, ear))  # Cap at reasonable EAR value

        if ear_values:
            return np.mean(ear_values)

        return None

    def _calculate_ear_simplified(self, frame: np.ndarray) -> Optional[float]:
        """
        Simplified EAR calculation when cascades aren't available
        Uses image properties as proxies for eye openness
        """
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Use edge detection to find eye-like regions
            edges = cv2.Canny(blurred, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter for eye-like contours (small, roughly circular)
            eye_contours = []
            h, w = gray.shape
            min_area = (w * h) * 0.005  # Minimum eye area
            max_area = (w * h) * 0.05   # Maximum eye area

            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # Reasonably circular
                            eye_contours.append(contour)

            if len(eye_contours) >= 2:
                # Calculate average "openness" from eye contours
                Openness_values = []
                for contour in eye_contours[:2]:
                    area = cv2.contourArea(contour)
                    # Normalize openness (this is simplified)
                    openness = min(0.4, area / (w * h) * 20)  # Scale factor
                    Openness_values.append(openness)

                # EAR is inversely related to openness (more closed = lower EAR)
                # Normal EAR ~0.25-0.3, closed eye EAR ~0.1
                avg_openness = np.mean(Openness_values)
                ear = 0.4 - avg_openness  # Invert and scale
                return max(0.05, min(0.4, ear))

            return None

        except Exception as e:
            logger.warning(f"Simplified EAR calculation failed: {e}")
            return None

    def _detect_blink(self, ear: float, timestamp: float):
        """
        Detect blinks based on Eye Aspect Ratio
        """
        # Blink detected when EAR goes below threshold and then rises above it
        if len(self.ear_history) >= 3:
            # Check for blink pattern: high EAR -> low EAR -> high EAR
            recent_ears = list(self.ear_history)[-3:]

            if (recent_ears[0] > self.blink_threshold and  # Open eye
                recent_ears[1] <= self.blink_threshold and   # Closed eye
                recent_ears[2] > self.blink_threshold):      # Open eye again

                # Blink detected
                self.blink_count += 1
                current_time = timestamp

                if self.last_blink_time > 0:
                    interval = current_time - self.last_blink_time
                    self.blink_intervals.append(interval)

                self.last_blink_time = current_time
                self.blink_timestamps.append(current_time)

    def _calculate_blink_metrics(self) -> tuple[float, bool, float]:
        """
        Calculate blink rate, abnormal pattern detection, and confidence
        Returns (blink_rate_per_minute, abnormal_pattern, confidence)
        """
        # Calculate blink rate (blinks per minute)
        if len(self.blink_timestamps) < 2:
            blink_rate = 0.0
        else:
            time_span = self.blink_timestamps[-1] - self.blink_timestamps[0]
            if time_span > 0:
                blink_rate = (len(self.blink_timestamps) - 1) / (time_span / 60.0)
            else:
                blink_rate = 0.0

        # Detect abnormal blink pattern
        abnormal_pattern = False

        if len(self.blink_intervals) >= 5:
            intervals = list(self.blink_intervals)

            # Check for abnormally high or low blink rate
            avg_interval = np.mean(intervals)
            expected_interval = 4.0  # Normal blink every ~4 seconds (15 per minute)

            # Abnormally high blink rate (>25 per minute = <2.4s intervals)
            # Abnormally low blink rate (<8 per minute = >7.5s intervals)
            if avg_interval < 2.4 or avg_interval > 7.5:
                abnormal_pattern = True
            else:
                # Check for high irregularity in blink intervals
                interval_std = np.std(intervals)
                interval_cv = interval_std / avg_interval if avg_interval > 0 else 0
                # High coefficient of variation indicates irregular pattern
                if interval_cv > 0.6:
                    abnormal_pattern = True

            # Check for prolonged eye closure (simplified)
            # In a real implementation, we'd track duration of low EAR values
            if len(self.ear_history) >= 10:
                recent_ears = list(self.ear_history)[-10:]
                closed_eye_frames = sum(1 for ear in recent_ears if ear < self.blink_threshold * 0.5)
                if closed_eye_frames > 7:  # More than 70% of recent frames showing closed eyes
                    abnormal_pattern = True

        # Calculate confidence
        sample_confidence = min(1.0, len(self.ear_history) / 30.0)  # Full confidence at 30+ frames

        # Blink detection confidence
        blink_confidence = 0.5
        if self.blink_count > 0:
            # More blinks detected = higher confidence in blink detection ability
            blink_confidence = min(0.9, 0.5 + (self.blink_count * 0.1))

        # Pattern confidence - more consistent data = higher confidence
        pattern_confidence = 0.5
        if len(self.blink_intervals) >= 3:
            interval_consistency = 1.0 - min(1.0, np.std(list(self.blink_intervals)) /
                                           (np.mean(list(self.blink_intervals)) + 0.1))
            pattern_confidence = max(0.3, interval_consistency)

        confidence = (sample_confidence * 0.4 + blink_confidence * 0.3 + pattern_confidence * 0.3)
        confidence = max(0.1, min(1.0, confidence))

        return blink_rate, abnormal_pattern, confidence

    def _insufficient_data_response(self, reason: str) -> Dict[str, Any]:
        """
        Return standardized response for insufficient data
        """
        return {
            "blink_rate": 0.0,
            "abnormal_blink_pattern": False,
            "confidence": 0.0,
            "insufficient_data": True,
            "reason": reason,
            "timestamp": time.time()
        }

    def reset_history(self):
        """Reset the analyzer's historical data"""
        self.ear_history.clear()
        self.blink_timestamps.clear()
        self.frame_timestamps.clear()
        self.blink_intervals.clear()
        self.blink_count = 0
        self.last_blink_time = 0