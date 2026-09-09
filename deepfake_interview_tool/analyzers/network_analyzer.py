"""
Network Analyzer
Analyzes network telemetry for latency, jitter, and packet loss
"""

from typing import List, Dict, Any, Optional
import logging
import time
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class NetworkAnalyzer:
    def __init__(self, window_size: int = 30):
        """
        Initialize network analyzer

        Args:
            window_size: Number of network measurements to consider for temporal analysis
        """
        self.window_size = window_size

        # For storing historical network metrics
        self.latency_history = deque(maxlen=window_size)
        self.jitter_history = deque(maxlen=window_size)
        self.packet_loss_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        # Thresholds for network issue detection
        self.latency_threshold_ms = 150.0   # High latency threshold
        self.jitter_threshold_ms = 30.0     # High jitter threshold
        self.packet_loss_threshold_percent = 5.0  # High packet loss threshold

        logger.info("Network analyzer initialized")

    def analyze(self, network_metrics: dict) -> Dict[str, Any]:
        """
        Analyze network telemetry

        Args:
            network_metrics: Dictionary containing network metrics
                           Expected keys: 'latency_ms', 'jitter_ms', 'packet_loss_percent'

        Returns:
            Dictionary with network analysis results
        """
        try:
            if not network_metrics:
                return self._insufficient_data_response("No network metrics provided")

            # Extract metrics with defaults
            latency_ms = network_metrics.get('latency_ms', 0.0)
            jitter_ms = network_metrics.get('jitter_ms', 0.0)
            packet_loss_percent = network_metrics.get('packet_loss_percent', 0.0)

            # Validate metrics
            if latency_ms < 0 or jitter_ms < 0 or packet_loss_percent < 0 or packet_loss_percent > 100:
                return self._insufficient_data_response("Invalid network metric values")

            # Store in history for temporal analysis
            self.latency_history.append(latency_ms)
            self.jitter_history.append(jitter_ms)
            self.packet_loss_history.append(packet_loss_percent)
            self.timestamps.append(time.time())

            # Calculate network issue indicators
            network_issue_detected, confidence = self._calculate_network_issue_score(
                latency_ms, jitter_ms, packet_loss_percent
            )

            # Calculate rolling averages for smoother metrics
            avg_latency = self._calculate_rolling_average(self.latency_history)
            avg_jitter = self._calculate_rolling_average(self.jitter_history)
            avg_packet_loss = self._calculate_rolling_average(self.packet_loss_history)

            return {
                "network_issue_detected": network_issue_detected,
                "latency_ms": float(latency_ms),
                "jitter_ms": float(jitter_ms),
                "packet_loss_percent": float(packet_loss_percent),
                "avg_latency_ms": float(avg_latency) if avg_latency is not None else 0.0,
                "avg_jitter_ms": float(avg_jitter) if avg_jitter is not None else 0.0,
                "avg_packet_loss_percent": float(avg_packet_loss) if avg_packet_loss is not None else 0.0,
                "confidence": float(confidence),
                "timestamp": time.time(),
                "samples_analyzed": len(self.latency_history)
            }

        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            return {
                "error": str(e),
                "network_issue_detected": False,
                "latency_ms": 0.0,
                "jitter_ms": 0.0,
                "packet_loss_percent": 0.0,
                "confidence": 0.0
            }

    def _calculate_network_issue_score(self, latency_ms: float, jitter_ms: float, packet_loss_percent: float) -> tuple[bool, float]:
        """
        Calculate network issue detection score and confidence
        Returns (network_issue_detected, confidence)
        """
        try:
            # Calculate individual issue scores
            latency_score = min(1.0, latency_ms / self.latency_threshold_ms) if self.latency_threshold_ms > 0 else 0.0
            jitter_score = min(1.0, jitter_ms / self.jitter_threshold_ms) if self.jitter_threshold_ms > 0 else 0.0
            packet_loss_score = min(1.0, packet_loss_percent / self.packet_loss_threshold_percent) if self.packet_loss_threshold_percent > 0 else 0.0

            # Combined network issue score (weighted average)
            # Latency and jitter are more critical for real-time communication
            combined_score = (latency_score * 0.4 + jitter_score * 0.35 + packet_loss_score * 0.25)

            # Determine if network issue is detected
            network_issue_detected = combined_score > 0.5

            # Calculate confidence
            # Based on: number of samples, consistency of metrics
            sample_confidence = min(1.0, len(self.latency_history) / 10.0)  # Full confidence at 10+ samples

            # Consistency confidence - check if metrics are stable
            consistency_confidence = 0.5
            if len(self.latency_history) >= 3:
                latency_cv = self._calculate_coefficient_of_variation(list(self.latency_history))
                jitter_cv = self._calculate_coefficient_of_variation(list(self.jitter_history))
                # Lower coefficient of variation = more stable = higher confidence
                avg_cv = (latency_cv + jitter_cv) / 2 if (latency_cv is not None and jitter_cv is not None) else 0.5
                consistency_confidence = max(0.1, min(0.9, 1.0 - min(1.0, avg_cv)))

            confidence = (sample_confidence * 0.6 + consistency_confidence * 0.4)
            confidence = max(0.1, min(1.0, confidence))

            return network_issue_detected, confidence

        except Exception as e:
            logger.warning(f"Network issue score calculation failed: {e}")
            return False, 0.3

    def _calculate_rolling_average(self, data: deque) -> Optional[float]:
        """
        Calculate rolling average of data
        """
        try:
            if len(data) == 0:
                return None
            return statistics.mean(data)
        except Exception:
            return None

    def _calculate_coefficient_of_variation(self, data: List[float]) -> Optional[float]:
        """
        Calculate coefficient of variation (std/dev / mean)
        """
        try:
            if len(data) < 2:
                return None
            mean_val = statistics.mean(data)
            if mean_val == 0:
                return None
            std_val = statistics.stdev(data)
            return std_val / mean_val
        except Exception:
            return None

    def _insufficient_data_response(self, reason: str) -> Dict[str, Any]:
        """
        Return standardized response for insufficient data
        """
        return {
            "network_issue_detected": False,
            "latency_ms": 0.0,
            "jitter_ms": 0.0,
            "packet_loss_percent": 0.0,
            "confidence": 0.0,
            "insufficient_data": True,
            "reason": reason,
            "timestamp": time.time()
        }

    def reset_history(self):
        """Reset the analyzer's historical data"""
        self.latency_history.clear()
        self.jitter_history.clear()
        self.packet_loss_history.clear()
        self.timestamps.clear()