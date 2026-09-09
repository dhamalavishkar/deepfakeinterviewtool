"""
Lip-Sync Variance Analyzer
Analyzes temporal alignment between audio speech and lip movement
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

class LipSyncAnalyzer:
    def __init__(self, window_size: int = 150, sample_rate: int = 16000):
        """
        Initialize lip-sync analyzer

        Args:
            window_size: Number of frames/audio chunks to consider for temporal analysis
            sample_rate: Audio sample rate in Hz
        """
        self.window_size = window_size
        self.sample_rate = sample_rate

        # For storing temporal alignment data
        self.audio_energy_history = deque(maxlen=window_size)
        self.lip_movement_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        # Cross-correlation parameters
        self.max_lag_frames = 30  # Maximum lag to consider (±30 frames at 30fps = ±1 second)
        self.sync_threshold = 0.3  # Threshold for significant lip-sync variance

        logger.info("Lip-sync analyzer initialized")

    def analyze(self, video_frames: List[str], audio_samples: List[str]) -> Dict[str, Any]:
        """
        Analyze lip-sync variance from video and audio data

        Args:
            video_frames: List of base64 encoded video frames
            audio_samples: List of base64 encoded audio samples/chunks

        Returns:
            Dictionary with lip-sync analysis results
        """
        try:
            if not video_frames or not audio_samples:
                return self._insufficient_data_response("Missing video frames or audio samples")

            # Process audio and video to extract features
            audio_energy = self._extract_audio_energy(audio_samples)
            lip_movement = self._extract_lip_movement(video_frames)

            if audio_energy is None or lip_movement is None:
                return self._insufficient_data_response("Failed to extract features from media")

            # Ensure we have matching lengths
            min_length = min(len(audio_energy), len(lip_movement))
            if min_length < 10:
                return self._insufficient_data_response(f"Insufficient aligned data: {min_length} samples")

            audio_energy = audio_energy[:min_length]
            lip_movement = lip_movement[:min_length]

            # Store in history for temporal analysis
            for i in range(min_length):
                self.audio_energy_history.append(audio_energy[i])
                self.lip_movement_history.append(lip_movement[i])
                self.timestamps.append(time.time() - (min_length - i) * 0.1)  # Approximate timestamps

            # Calculate lip-sync variance using cross-correlation
            temporal_offset_ms, sync_score, confidence = self._calculate_lip_sync_variance(
                audio_energy, lip_movement
            )

            # Determine if lip-sync variance is detected
            lip_sync_variance_detected = sync_score < self.sync_threshold and confidence > 0.5

            return {
                "lip_sync_variance_detected": lip_sync_variance_detected,
                "temporal_offset_ms": float(temporal_offset_ms),
                "confidence": float(confidence),
                "sync_score": float(sync_score),
                "analysis_window_seconds": float(min_length * 0.1),  # Assuming 10fps effective rate
                "timestamp": time.time(),
                "samples_analyzed": min_length
            }

        except Exception as e:
            logger.error(f"Lip-sync analysis failed: {e}")
            return {
                "error": str(e),
                "lip_sync_variance_detected": False,
                "temporal_offset_ms": 0.0,
                "confidence": 0.0
            }

    def _extract_audio_energy(self, audio_samples: List[str]) -> Optional[np.ndarray]:
        """
        Extract energy envelope from audio samples
        """
        try:
            # Convert audio samples to energy envelope
            energy_values = []

            for i, audio_b64 in enumerate(audio_samples):
                try:
                    # Decode audio sample (simplified)
                    audio_data = self._decode_audio_sample(audio_b64)
                    if audio_data is None or len(audio_data) == 0:
                        energy_values.append(0.0)
                        continue

                    # Calculate short-term energy
                    # In reality: energy = sum(abs(audio_data ** 2)) / len(audio_data)
                    # Simplified for MVP:
                    energy = np.sqrt(np.mean(np.array(audio_data, dtype=np.float64) ** 2))
                    energy_values.append(float(energy))

                except Exception as e:
                    logger.warning(f"Error processing audio sample {i}: {e}")
                    energy_values.append(0.0)
                    continue

            if len(energy_values) == 0:
                return None

            # Smooth the energy envelope
            energy_array = np.array(energy_values)
            if len(energy_array) > 3:
                # Simple moving average smoothing
                window_size = min(5, len(energy_array) // 3 * 2 + 1)
                if window_size >= 3:
                    from numpy import convolve
                    weights = np.ones(window_size) / window_size
                    energy_smoothed = convolve(energy_array, weights, mode='same')
                    return energy_smoothed
                else:
                    return energy_array
            else:
                return energy_array

        except Exception as e:
            logger.warning(f"Audio energy extraction failed: {e}")
            return None

    def _extract_lip_movement(self, video_frames: List[str]) -> Optional[np.ndarray]:
        """
        Extract lip movement features from video frames
        """
        try:
            # Extract lip region movement from video frames
            movement_values = []

            prev_lip_region = None

            for i, frame_b64 in enumerate(video_frames):
                try:
                    # Decode video frame
                    frame = self._decode_video_frame(frame_b64)
                    if frame is None:
                        movement_values.append(0.0)
                        continue

                    # Extract lip region features
                    lip_feature = self._extract_lip_feature(frame)

                    if prev_lip_region is not None and lip_feature is not None:
                        # Calculate movement as difference from previous frame
                        movement = np.linalg.norm(np.array(lip_feature) - np.array(prev_lip_region))
                        movement_values.append(float(movement))
                    else:
                        movement_values.append(0.0)

                    prev_lip_region = lip_feature if lip_feature is not None else prev_lip_region

                except Exception as e:
                    logger.warning(f"Error processing video frame {i}: {e}")
                    movement_values.append(0.0)
                    continue

            if len(movement_values) == 0:
                return None

            movement_array = np.array(movement_values)
            return movement_array

        except Exception as e:
            logger.warning(f"Lip movement extraction failed: {e}")
            return None

    def _decode_audio_sample(self, audio_b64: str) -> Optional[List[float]]:
        """
        Decode base64 audio sample to numerical array
        """
        try:
            # For MVP, simulate audio data
            # In reality:
            # import base64
            # import numpy as np
            # audio_data = base64.b64decode(audio_b64)
            # audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Simulate audio chunk
            chunk_size = 1024  # Typical audio chunk size
            audio_data = np.random.uniform(-0.5, 0.5, chunk_size).tolist()
            return audio_data
        except Exception as e:
            logger.warning(f"Audio decoding failed: {e}")
            return None

    def _decode_video_frame(self, frame_b64: str) -> Optional[np.ndarray]:
        """
        Decode base64 video frame to numpy array
        """
        try:
            # For MVP, simulate frame data
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            return frame
        except Exception as e:
            logger.warning(f"Video frame decoding failed: {e}")
            return None

    def _extract_lip_feature(self, frame: np.ndarray) -> Optional[List[float]]:
        """
        Extract lip region feature vector from a frame
        """
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # For MVP, use simplified lip region detection
            # In reality, would use facial landmarks to extract precise lip region

            h, w = gray.shape

            # Define approximate lip region (lower third of face, centered horizontally)
            lip_y_start = int(h * 0.6)
            lip_y_end = int(h * 0.9)
            lip_x_start = int(w * 0.3)
            lip_x_end = int(w * 0.7)

            # Extract lip region
            lip_region = gray[lip_y_start:lip_y_end, lip_x_start:lip_x_end]

            if lip_region.size == 0:
                return None

            # Extract simple features: mean intensity and horizontal gradient
            mean_intensity = np.mean(lip_region)

            # Calculate horizontal gradient (lip width changes during speech)
            horiz_gradient = np.mean(np.abs(np.diff(lip_region, axis=1)))

            # Calculate vertical gradient (lip height changes during speech)
            vert_gradient = np.mean(np.abs(np.diff(lip_region, axis=0)))

            # Return feature vector
            return [float(mean_intensity), float(horiz_gradient), float(vert_gradient)]

        except Exception as e:
            logger.warning(f"Lip feature extraction failed: {e}")
            return None

    def _calculate_lip_sync_variance(self, audio_energy: np.ndarray, lip_movement: np.ndarray) -> tuple[float, float, float]:
        """
        Calculate lip-sync variance using cross-correlation
        Returns (temporal_offset_ms, sync_score, confidence)
        """
        try:
            # Normalize the signals
            audio_norm = (audio_energy - np.mean(audio_energy)) / (np.std(audio_energy) + 1e-8)
            lip_norm = (lip_movement - np.mean(lip_movement)) / (np.std(lip_movement) + 1e-8)

            # Calculate cross-correlation
            correlation = np.correlate(audio_norm, lip_norm, mode='full')

            # Find the lag with maximum correlation
            max_corr_index = np.argmax(correlation)
            max_correlation = correlation[max_corr_index]

            # Convert index to lag
            lag = max_corr_index - (len(lip_norm) - 1)

            # Limit lag to reasonable range
            max_lag = min(self.max_lag_frames, len(audio_norm) // 4)
            lag = np.clip(lag, -max_lag, max_lag)

            # Convert lag to milliseconds
            # Assuming ~30fps for video and corresponding audio chunks
            frame_duration_ms = 1000.0 / 30.0  # ~33.3ms per frame
            temporal_offset_ms = lag * frame_duration_ms

            # Normalize correlation to 0-1 range (higher = better sync)
            # Cross-correlation can be negative, so we shift and scale
            sync_score = (max_correlation + 1.0) / 2.0  # Assuming normalized signals give correlation in [-1, 1]
            sync_score = max(0.0, min(1.0, sync_score))

            # Calculate confidence based on:
            # 1. Peak correlation strength
            # 2. Signal-to-noise ratio of the correlation peak
            # 3. Length of analysis window

            # Peak confidence
            peak_confidence = max(0.0, min(1.0, (sync_score - 0.3) / 0.7))  # Confident if sync_score > 0.3

            # Length confidence
            length_confidence = min(1.0, len(audio_energy) / 50.0)  # Full confidence at 50+ samples

            # Sharpness confidence (how peaked the correlation is)
            if len(correlation) > 10:
                # Find width of correlation peak at half maximum
                half_max = max_correlation / 2.0
                above_half = correlation >= half_max
                if np.any(above_half):
                    indices = np.where(above_half)[0]
                    width = indices[-1] - indices[0] + 1
                    sharpness = min(1.0, 20.0 / width)  # Sharper peak = higher confidence
                else:
                    sharpness = 0.5
            else:
                sharpness = 0.5

            confidence = (peak_confidence * 0.4 + length_confidence * 0.3 + sharpness * 0.3)
            confidence = max(0.1, min(1.0, confidence))

            return float(temporal_offset_ms), float(sync_score), float(confidence)

        except Exception as e:
            logger.warning(f"Lip-sync variance calculation failed: {e}")
            return 0.0, 0.5, 0.3  # Default uncertain values

    def _insufficient_data_response(self, reason: str) -> Dict[str, Any]:
        """
        Return standardized response for insufficient data
        """
        return {
            "lip_sync_variance_detected": False,
            "temporal_offset_ms": 0.0,
            "confidence": 0.0,
            "sync_score": 0.0,
            "insufficient_data": True,
            "reason": reason,
            "timestamp": time.time()
        }

    def reset_history(self):
        """Reset the analyzer's historical data"""
        self.audio_energy_history.clear()
        self.lip_movement_history.clear()
        self.timestamps.clear()