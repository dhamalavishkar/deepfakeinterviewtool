"""
Voice Analyzer
Analyzes audio for synthetic voice artifacts and unnatural characteristics
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

class VoiceAnalyzer:
    def __init__(self, window_size: int = 100, sample_rate: int = 16000):
        """
        Initialize voice analyzer

        Args:
            window_size: Number of audio chunks to consider for temporal analysis
            sample_rate: Audio sample rate in Hz
        """
        self.window_size = window_size
        self.sample_rate = sample_rate

        # For storing historical audio features
        self.spectral_centroid_history = deque(maxlen=window_size)
        self.spectral_rolloff_history = deque(maxlen=window_size)
        self.zero_crossing_rate_history = deque(maxlen=window_size)
        self.mfcc_history = deque(maxlen=window_size)
        self.pitch_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        # Artifact detection thresholds
        self.spectral_flatness_threshold = 0.8  # High flatness may indicate synthetic voice
        self.pitch_variance_threshold = 50.0    # High pitch variance may indicate artifacts
        self.mfcc_deviation_threshold = 10.0    # Deviation from natural speech patterns

        logger.info("Voice analyzer initialized")

    def analyze(self, audio_samples: List[str]) -> Dict[str, Any]:
        """
        Analyze audio for synthetic voice artifacts

        Args:
            audio_samples: List of base64 encoded audio samples/chunks

        Returns:
            Dictionary with voice analysis results
        """
        try:
            if not audio_samples:
                return self._insufficient_data_response("No audio samples provided")

            # Extract features from audio samples
            features = self._extract_audio_features(audio_samples)

            if features is None or len(features) == 0:
                return self._insufficient_data_response("Failed to extract audio features")

            # Store features in history for temporal analysis
            for feature in features:
                self.spectral_centroid_history.append(feature['spectral_centroid'])
                self.spectral_rolloff_history.append(feature['spectral_rolloff'])
                self.zero_crossing_rate_history.append(feature['zero_crossing_rate'])
                if feature['mfccs'] is not None:
                    self.mfcc_history.append(feature['mfccs'])
                self.pitch_history.append(feature['pitch'])
                self.timestamps.append(time.time())

            # Calculate voice artifact indicators
            artifact_score, confidence = self._calculate_voice_artifact_score(features)

            # Determine if synthetic voice artifacts are detected
            synthetic_voice_artifact_detected = artifact_score > 0.5 and confidence > 0.5

            return {
                "synthetic_voice_artifact_detected": synthetic_voice_artifact_detected,
                "artifact_score": float(artifact_score),
                "confidence": float(confidence),
                "timestamp": time.time(),
                "samples_analyzed": len(features),
                "features_summary": self._get_features_summary(features)
            }

        except Exception as e:
            logger.error(f"Voice analysis failed: {e}")
            return {
                "error": str(e),
                "synthetic_voice_artifact_detected": False,
                "artifact_score": 0.0,
                "confidence": 0.0
            }

    def _extract_audio_features(self, audio_samples: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        Extract audio features from samples
        """
        try:
            features_list = []

            for i, audio_b64 in enumerate(audio_samples):
                try:
                    # Decode audio sample
                    audio_data = self._decode_audio_sample(audio_b64)
                    if audio_data is None or len(audio_data) < self.sample_rate // 10:  # Less than 100ms
                        continue

                    # Extract features
                    features = self._compute_audio_features(audio_data)
                    if features is not None:
                        features_list.append(features)

                except Exception as e:
                    logger.warning(f"Error processing audio sample {i}: {e}")
                    continue

            if len(features_list) == 0:
                return None

            return features_list

        except Exception as e:
            logger.warning(f"Audio feature extraction failed: {e}")
            return None

    def _decode_audio_sample(self, audio_b64: str) -> Optional[np.ndarray]:
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

            # Simulate audio chunk (1 second of audio at 16kHz)
            audio_data = np.random.uniform(-0.8, 0.8, self.sample_rate).astype(np.float32)
            return audio_data
        except Exception as e:
            logger.warning(f"Audio decoding failed: {e}")
            return None

    def _compute_audio_features(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Compute audio features for synthetic voice detection
        """
        try:
            # Ensure we have enough data
            if len(audio_data) < 512:
                return None

            # Simple feature extraction for MVP
            # In reality, would use libraries like librosa for more sophisticated features

            # 1. Spectral Centroid
            spectral_centroid = self._compute_spectral_centroid(audio_data)

            # 2. Spectral Rolloff
            spectral_rolloff = self._compute_spectral_rolloff(audio_data)

            # 3. Zero Crossing Rate
            zero_crossing_rate = self._compute_zero_crossing_rate(audio_data)

            # 4. Pitch (fundamental frequency)
            pitch = self._estimate_pitch(audio_data)

            # 5. MFCC-like features (simplified)
            mfccs = self._compute_simple_mfcc(audio_data)

            return {
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'zero_crossing_rate': zero_crossing_rate,
                'pitch': pitch,
                'mfccs': mfccs,
                'timestamp': time.time()
            }

        except Exception as e:
            logger.warning(f"Feature computation failed: {e}")
            return None

    def _compute_spectral_centroid(self, audio_data: np.ndarray) -> float:
        """
        Compute spectral centroid (brightness of sound)
        """
        try:
            # Compute FFT
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1.0/self.sample_rate)

            # Avoid division by zero
            if np.sum(magnitude) == 0:
                return 0.0

            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            return float(centroid)
        except Exception:
            return 0.0

    def _compute_spectral_rolloff(self, audio_data: np.ndarray, rolloff_point: float = 0.85) -> float:
        """
        Compute spectral rolloff frequency
        """
        try:
            # Compute FFT
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1.0/self.sample_rate)

            # Compute cumulative sum
            cumsum = np.cumsum(magnitude)
            total_energy = cumsum[-1]

            if total_energy == 0:
                return 0.0

            # Find frequency where rolloff_point% of energy is below
            rolloff_threshold = total_energy * rolloff_point
            rolloff_index = np.where(cumsum >= rolloff_threshold)[0]

            if len(rolloff_index) > 0:
                rolloff_freq = freqs[rolloff_index[0]]
                return float(rolloff_freq)
            else:
                return float(freqs[-1])
        except Exception:
            return 0.0

    def _compute_zero_crossing_rate(self, audio_data: np.ndarray) -> float:
        """
        Compute zero crossing rate
        """
        try:
            # Count zero crossings
            signs = np.sign(audio_data)
            zero_crossings = np.where(np.diff(signs))[0]
            zcr = len(zero_crossings) / len(audio_data)
            return float(zcr)
        except Exception:
            return 0.0

    def _estimate_pitch(self, audio_data: np.ndarray) -> float:
        """
        Estimate fundamental frequency (pitch) using autocorrelation
        """
        try:
            # Autocorrelation method for pitch estimation
            # Limit to reasonable pitch range (80-400 Hz for speech)
            max_lag = int(self.sample_rate / 80)   # Minimum period for 80Hz
            min_lag = int(self.sample_rate / 400)  # Maximum period for 400Hz

            if max_lag > len(audio_data):
                max_lag = len(audio_data) // 2

            # Compute autocorrelation
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Take second half

            # Look for peak in the relevant range
            search_range = autocorr[min_lag:max_lag]
            if len(search_range) == 0:
                return 0.0

            peak_index = np.argmax(search_range) + min_lag
            if peak_index > 0:
                pitch = self.sample_rate / peak_index
                return float(pitch)
            else:
                return 0.0
        except Exception:
            return 0.0

    def _compute_simple_mfcc(self, audio_data: np.ndarray) -> Optional[List[float]]:
        """
        Compute simplified MFCC-like features
        """
        try:
            # For MVP, compute a few basic spectral features that approximate MFCCs
            # In reality, would use proper MFCC computation with mel filterbanks

            # Split audio into frames
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop

            if len(audio_data) < frame_length:
                return None

            # Extract a few frames
            num_frames = min(5, (len(audio_data) - frame_length) // hop_length + 1)
            if num_frames < 1:
                return None

            features = []
            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                frame = audio_data[start:end]

                # Apply window
                windowed = frame * np.hamming(len(frame))

                # Compute FFT
                fft = np.fft.rfft(windowed)
                magnitude = np.abs(fft)
                power = magnitude ** 2

                # Take first few coefficients as simplified MFCCs
                num_coeffs = min(5, len(power))
                frame_features = power[:num_coeffs].tolist()
                features.extend(frame_features)

            # Return averaged features
            if len(features) > 0:
                # Group by coefficient index and average
                num_features = len(features) // num_frames
                averaged = []
                for i in range(num_features):
                    coeff_values = features[i::num_frames]
                    averaged.append(np.mean(coeff_values))
                return averaged
            else:
                return None

        except Exception as e:
            logger.warning(f"MFCC computation failed: {e}")
            return None

    def _calculate_voice_artifact_score(self, features: List[Dict[str, Any]]) -> tuple[float, float]:
        """
        Calculate voice artifact score based on extracted features
        Returns (artifact_score, confidence)
        """
        try:
            if not features:
                return 0.0, 0.0

            # Extract feature arrays
            spectral_centroids = [f['spectral_centroid'] for f in features if f['spectral_centroid'] is not None]
            spectral_rolloffs = [f['spectral_rolloff'] for f in features if f['spectral_rolloff'] is not None]
            zero_crossing_rates = [f['zero_crossing_rate'] for f in features if f['zero_crossing_rate'] is not None]
            pitches = [f['pitch'] for f in features if f['pitch'] is not None and f['pitch'] > 0]
            mfccs = [f['mfccs'] for f in features if f['mfccs'] is not None]

            # Calculate artifact indicators
            artifact_indicators = []

            # 1. Spectral flatness indicator (synthetic voices often have unusually flat spectra)
            if spectral_centroids and spectral_rolloffs:
                # Ratio of centroid to rolloff - unnatural values may indicate synthesis
                ratios = [sc/sr if sr > 0 else 0 for sc, sr in zip(spectral_centroids, spectral_rolloffs)]
                avg_ratio = np.mean(ratios) if ratios else 0.5
                # Normalize to 0-1 range where extreme values indicate artifacts
                flatness_score = min(1.0, abs(avg_ratio - 0.3) * 2)  # Assuming 0.3 is natural
                artifact_indicators.append(flatness_score)

            # 2. Pitch stability indicator (synthetic voices may have unnatural pitch stability or instability)
            if len(pitches) >= 3:
                pitch_std = np.std(pitches)
                pitch_mean = np.mean(pitches)
                # Natural speech has some pitch variation but not extreme
                if pitch_mean > 0:
                    pitch_cv = pitch_std / pitch_mean  # Coefficient of variation
                    # Very low or very high CV may indicate synthetic voice
                    pitch_stability_score = min(1.0, abs(pitch_cv - 0.1) * 3)  # Assuming 0.1 is natural CV
                    artifact_indicators.append(pitch_stability_score)

            # 3. Zero crossing rate consistency
            if zero_crossing_rates:
                zcr_mean = np.mean(zero_crossing_rates)
                zcr_std = np.std(zero_crossing_rates) if len(zero_crossing_rates) > 1 else 0
                # Unnatural consistency or variation in ZCR may indicate synthesis
                zcr_score = min(1.0, (zcr_std / (zcr_mean + 0.01)) * 2) if zcr_mean > 0 else 0.5
                artifact_indicators.append(zcr_score)

            # 4. MFCC deviation from natural speech patterns (simplified)
            if mfccs and len(mfccs) >= 2:
                # Look at variance across MFCC coefficients
                mfcc_array = np.array(mfccs)
                if mfcc_array.size > 0:
                    mfcc_std = np.std(mfcc_array, axis=0)
                    avg_std = np.mean(mfcc_std)
                    # High variance across coefficients may indicate artifacts
                    mfcc_score = min(1.0, avg_std / 10.0)  # Normalize
                    artifact_indicators.append(mfcc_score)

            # Combine artifact indicators
            if artifact_indicators:
                artifact_score = np.mean(artifact_indicators)
                artifact_score = max(0.0, min(1.0, artifact_score))  # Clamp to 0-1
            else:
                artifact_score = 0.0

            # Calculate confidence
            # Based on: number of samples, feature quality, consistency
            sample_confidence = min(1.0, len(features) / 20.0)  # Full confidence at 20+ samples

            # Feature completeness confidence
            expected_features = 5  # spectral_centroid, spectral_rolloff, zcr, pitch, mfccs
            complete_features = 0
            if spectral_centroids: complete_features += 1
            if spectral_rolloffs: complete_features += 1
            if zero_crossing_rates: complete_features += 1
            if pitches: complete_features += 1
            if mfccs: complete_features += 1
            feature_confidence = complete_features / expected_features

            # Temporal consistency confidence
            consistency_confidence = 0.5
            if len(features) >= 3:
                # Check if features are reasonably consistent over time
                feature_variances = []
                if len(spectral_centroids) >= 3:
                    feature_variances.append(np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-8))
                if len(zero_crossing_rates) >= 3:
                    feature_variances.append(np.std(zero_crossing_rates) / (np.mean(zero_crossing_rates) + 1e-8))
                if feature_variances:
                    avg_variance = np.mean(feature_variances)
                    # Moderate variance is natural, too low or too high may be suspicious
                    consistency_confidence = max(0.1, min(0.9, 1.0 - abs(avg_variance - 0.1) * 5))

            confidence = (sample_confidence * 0.4 + feature_confidence * 0.3 + consistency_confidence * 0.3)
            confidence = max(0.1, min(1.0, confidence))

            return float(artifact_score), float(confidence)

        except Exception as e:
            logger.warning(f"Voice artifact score calculation failed: {e}")
            return 0.0, 0.3

    def _get_features_summary(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of extracted features for debugging
        """
        try:
            if not features:
                return {}

            spectral_centroids = [f['spectral_centroid'] for f in features if f['spectral_centroid'] is not None]
            spectral_rolloffs = [f['spectral_rolloff'] for f in features if f['spectral_rolloff'] is not None]
            zero_crossing_rates = [f['zero_crossing_rate'] for f in features if f['zero_crossing_rate'] is not None]
            pitches = [f['pitch'] for f in features if f['pitch'] is not None and f['pitch'] > 0]

            summary = {
                'avg_spectral_centroid': np.mean(spectral_centroids) if spectral_centroids else 0,
                'avg_spectral_rolloff': np.mean(spectral_rolloffs) if spectral_rolloffs else 0,
                'avg_zero_crossing_rate': np.mean(zero_crossing_rates) if zero_crossing_rates else 0,
                'avg_pitch': np.mean(pitches) if pitches else 0,
                'pitch_std': np.std(pitches) if len(pitches) >= 2 else 0,
                'sample_count': len(features)
            }

            return summary

        except Exception:
            return {}

    def _insufficient_data_response(self, reason: str) -> Dict[str, Any]:
        """
        Return standardized response for insufficient data
        """
        return {
            "synthetic_voice_artifact_detected": False,
            "artifact_score": 0.0,
            "confidence": 0.0,
            "insufficient_data": True,
            "reason": reason,
            "timestamp": time.time()
        }

    def reset_history(self):
        """Reset the analyzer's historical data"""
        self.spectral_centroid_history.clear()
        self.spectral_rolloff_history.clear()
        self.zero_crossing_rate_history.clear()
        self.mfcc_history.clear()
        self.pitch_history.clear()
        self.timestamps.clear()