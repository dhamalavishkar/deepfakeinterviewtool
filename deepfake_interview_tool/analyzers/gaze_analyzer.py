"""
Gaze Diversion Analyzer
Analyzes eye gaze patterns to detect potential distraction or unnatural eye movement
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class GazeAnalyzer:
    def __init__(self, window_size: int = 30, deviation_threshold: float = 0.3):
        """
        Initialize gaze analyzer

        Args:
            window_size: Number of frames to consider for temporal analysis
            deviation_threshold: Threshold for significant gaze deviation (0-1)
        """
        self.window_size = window_size
        self.deviation_threshold = deviation_threshold

        # For storing historical gaze data
        self.gaze_history = deque(maxlen=window_size)
        self.frame_timestamps = deque(maxlen=window_size)

        # Initialize face detection and landmark models (using OpenCV's pre-trained models)
        try:
            # Load pre-trained face detection model
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # Load eye cascade for more precise eye detection
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )

            logger.info("Gaze analyzer initialized with OpenCV cascades")
        except Exception as e:
            logger.warning(f"Could not load OpenCV cascades: {e}. Using simplified analysis.")
            self.face_cascade = None
            self.eye_cascade = None

    def analyze(self, video_frames: List[str]) -> Dict[str, Any]:
        """
        Analyze gaze diversion from video frames

        Args:
            video_frames: List of base64 encoded video frames

        Returns:
            Dictionary with gaze analysis results
        """
        try:
            if not video_frames:
                return self._insufficient_data_response("No video frames provided")

            # Process each frame
            gaze_scores = []
            valid_frames = 0

            for i, frame_b64 in enumerate(video_frames):
                try:
                    # Decode base64 frame (simplified - in practice would use proper decoding)
                    frame = self._decode_frame(frame_b64)
                    if frame is None:
                        continue

                    # Analyze gaze in this frame
                    gaze_score = self._analyze_frame_gaze(frame)
                    if gaze_score is not None:
                        gaze_scores.append(gaze_score)
                        valid_frames += 1

                        # Store in history for temporal analysis
                        self.gaze_history.append(gaze_score)
                        self.frame_timestamps.append(time.time())

                except Exception as e:
                    logger.warning(f"Error processing frame {i}: {e}")
                    continue

            if valid_frames == 0:
                return self._insufficient_data_response("No valid frames processed")

            # Calculate temporal gaze diversion metrics
            gaze_diversion_rate, confidence = self._calculate_temporal_gaze_metrics(gaze_scores)

            # Determine if gaze diversion is detected (avoid false positives for normal movement)
            gaze_diversion_detected = gaze_diversion_rate > self.deviation_threshold and confidence > 0.5

            return {
                "gaze_diversion_detected": gaze_diversion_detected,
                "gaze_diversion_rate": float(gaze_diversion_rate),
                "confidence": float(confidence),
                "timestamp": time.time(),
                "frames_analyzed": valid_frames,
                "total_frames": len(video_frames)
            }

        except Exception as e:
            logger.error(f"Gaze analysis failed: {e}")
            return {
                "error": str(e),
                "gaze_diversion_detected": False,
                "gaze_diversion_rate": 0.0,
                "confidence": 0.0
            }

    def _decode_frame(self, frame_b64: str) -> Optional[np.ndarray]:
        """
        Decode base64 frame to numpy array
        In a real implementation, this would use proper base64 and image decoding
        """
        try:
            # For MVP, we'll simulate frame processing
            # In reality:
            # import base64
            # import numpy as np
            # import cv2
            # img_data = base64.b64decode(frame_b64)
            # nparr = np.frombuffer(img_data, np.uint8)
            # frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Simulate a frame for testing
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            return frame
        except Exception as e:
            logger.warning(f"Frame decoding failed: {e}")
            return None

    def _analyze_frame_gaze(self, frame: np.ndarray) -> Optional[float]:
        """
        Analyze gaze in a single frame
        Returns a gaze deviation score (0-1, where higher means more deviation)
        """
        try:
            if self.face_cascade is not None and self.eye_cascade is not None:
                return self._analyze_gaze_with_landmarks(frame)
            else:
                return self._analyze_gaze_simplified(frame)
        except Exception as e:
            logger.warning(f"Frame gaze analysis failed: {e}")
            return None

    def _analyze_gaze_with_landmarks(self, frame: np.ndarray) -> float:
        """
        Analyze gaze using facial landmarks (more accurate)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return 0.5  # No face detected - uncertain

        # Process the largest face (assumed to be the candidate)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Extract face region
        face_roi = gray[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = self.eye_cascade.detectMultiScale(face_roi)

        if leneyes < 2:
            # Less than 2 eyes detected - could be looking away or obstruction
            return 0.7  # High uncertainty

        # Calculate eye positions relative to face center
        eye_centers = []
        for (ex, ey, ew, eh) in eyes:
            center_x = ex + ew // 2
            center_y = ey + eh // 2
            eye_centers.append([center_x, center_y])

        if len(eye_centers) >= 2:
            # Calculate average eye position
            avg_eye_x = np.mean([center[0] for center in eye_centers])
            avg_eye_y = np.mean([center[1] for center in eye_centers])

            # Normalize to face coordinates (0-1)
            norm_eye_x = avg_eye_x / w
            norm_eye_y = avg_eye_y / h

            # Expected gaze direction is roughly center of face
            expected_x, expected_y = 0.5, 0.4  # Slightly above center for natural gaze

            # Calculate deviation from expected gaze
            deviation_x = abs(norm_eye_x - expected_x)
            deviation_y = abs(norm_eye_y - expected_y)

            # Combine deviations (weighted more on horizontal deviation for gaze diversion)
            gaze_deviation = min(1.0, (deviation_x * 0.7 + deviation_y * 0.3) * 2)

            return gaze_deviation

        return 0.5  # Default uncertain value

    def _analyze_gaze_simplified(self, frame: np.ndarray) -> float:
        """
        Simplified gaze analysis when OpenCV cascades aren't available
        Uses basic image properties as proxies
        """
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Calculate image moments to estimate face/eye position
            moments = cv2.moments(gray)

            if moments["m00"] == 0:
                return 0.5  # No significant features

            # Calculate centroid
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

            # Normalize to image dimensions
            h, w = gray.shape
            norm_cx = cx / w
            norm_cy = cy / h

            # Expected center for frontal face
            expected_x, expected_y = 0.5, 0.5

            # Calculate deviation
            deviation = np.sqrt((norm_cx - expected_x)**2 + (norm_cy - expected_y)**2)
            deviation = min(1.0, deviation * 2)  # Scale to 0-1 range

            return deviation

        except Exception as e:
            logger.warning(f"Simplified gaze analysis failed: {e}")
            return 0.5

    def _calculate_temporal_gaze_metrics(self, gaze_scores: List[float]) -> tuple[float, float]:
        """
        Calculate temporal gaze diversion metrics from a sequence of scores
        Returns (gaze_diversion_rate, confidence)
        """
        if not gaze_scores:
            return 0.0, 0.0

        # Convert to numpy array for easier calculation
        scores = np.array(gaze_scores)

        # Calculate gaze diversion rate (proportion of frames with significant deviation)
        significant_deviations = scores > self.deviation_threshold
        gaze_diversion_rate = np.mean(significant_deviations)

        # Calculate confidence based on:
        # 1. Number of samples (more frames = higher confidence)
        # 2. Consistency of the signal
        # 3. Magnitude of deviations

        sample_confidence = min(1.0, len(scores) / 30.0)  # Full confidence at 30+ frames

        # Consistency check - if all frames show similar deviation, higher confidence
        if len(scores) > 1:
            consistency = 1.0 - min(1.0, np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0.5
            consistency = max(0.0, consistency)
        else:
            consistency = 0.5

        # Magnitude confidence - stronger signals get higher confidence
        magnitude_confidence = min(1.0, np.mean(scores) / self.deviation_threshold) if np.mean(scores) > 0 else 0.0

        # Combine confidence factors
        confidence = (sample_confidence * 0.4 + consistency * 0.3 + magnitude_confidence * 0.3)
        confidence = max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0

        return float(gaze_diversion_rate), float(confidence)

    def _insufficient_data_response(self, reason: str) -> Dict[str, Any]:
        """
        Return standardized response for insufficient data
        """
        return {
            "gaze_diversion_detected": False,
            "gaze_diversion_rate": 0.0,
            "confidence": 0.0,
            "insufficient_data": True,
            "reason": reason,
            "timestamp": time.time()
        }

    def reset_history(self):
        """Reset the analyzer's historical data"""
        self.gaze_history.clear()
        self.frame_timestamps.clear()