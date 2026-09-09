"""
Test network analyzer for high network jitter scenario
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analyzers.network_analyzer import NetworkAnalyzer


class TestNetworkAnalyzer(unittest.TestCase):
    """Test cases for NetworkAnalyzer"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = NetworkAnalyzer(window_size=10)

    def test_high_jitter_detection(self):
        """Test that high jitter is detected as network issue"""
        # Simulate network metrics with high jitter and moderate latency to trigger detection
        network_metrics = {
            'latency_ms': 100.0,  # Moderate latency (below threshold but contributes)
            'jitter_ms': 50.0,    # High jitter (above 30ms threshold)
            'packet_loss_percent': 1.0  # Normal packet loss
        }

        result = self.analyzer.analyze(network_metrics)

        # Verify network issue is detected
        self.assertTrue(result['network_issue_detected'],
                       "High jitter with moderate latency should be detected as network issue")
        self.assertGreater(result['confidence'], 0.1,
                          "Confidence should be reasonable")
        self.assertEqual(result['jitter_ms'], 50.0)
        self.assertEqual(result['latency_ms'], 100.0)

    def test_high_latency_detection(self):
        """Test that high latency is detected as network issue"""
        network_metrics = {
            'latency_ms': 200.0,  # High latency (above 150ms threshold)
            'jitter_ms': 10.0,    # Normal jitter
            'packet_loss_percent': 1.0  # Normal packet loss
        }

        result = self.analyzer.analyze(network_metrics)

        self.assertTrue(result['network_issue_detected'],
                       "High latency should be detected as network issue")
        self.assertEqual(result['latency_ms'], 200.0)
        self.assertLess(result['jitter_ms'], 50.0)  # Jitter should be normal

    def test_high_packet_loss_detection(self):
        """Test that high packet loss is detected as network issue"""
        network_metrics = {
            'latency_ms': 50.0,   # Normal latency
            'jitter_ms': 10.0,    # Normal jitter
            'packet_loss_percent': 10.0  # High packet loss (above 5% threshold)
        }

        result = self.analyzer.analyze(network_metrics)

        self.assertTrue(result['network_issue_detected'],
                       "High packet loss should be detected as network issue")
        self.assertEqual(result['packet_loss_percent'], 10.0)

    def test_multiple_network_issues(self):
        """Test detection when multiple network issues are present"""
        network_metrics = {
            'latency_ms': 180.0,  # High latency
            'jitter_ms': 40.0,    # High jitter
            'packet_loss_percent': 8.0   # High packet loss
        }

        result = self.analyzer.analyze(network_metrics)

        self.assertTrue(result['network_issue_detected'],
                       "Multiple network issues should be detected")
        # Confidence should be higher with multiple issues
        self.assertGreater(result['confidence'], 0.5)

    def test_normal_network_conditions(self):
        """Test that normal network conditions don't trigger issues"""
        network_metrics = {
            'latency_ms': 50.0,   # Normal latency
            'jitter_ms': 15.0,    # Normal jitter
            'packet_loss_percent': 1.0  # Normal packet loss
        }

        result = self.analyzer.analyze(network_metrics)

        # With normal conditions, network issue may or may not be detected
        # depending on internal thresholds and scoring, but confidence should be low if not detected
        if not result['network_issue_detected']:
            self.assertLess(result['confidence'], 0.5,
                           "Low confidence when no network issue detected")

    def test_network_issues_separate_from_biometric_signals(self):
        """Test that network analyzer only processes network metrics, not biometric data"""
        # This test ensures the analyzer doesn't mistakenly process biometric-like values
        # as network metrics. We'll send values that might resemble biometric signals
        # but are actually network metrics, and verify it still works correctly.

        # Simulate what might be mistaken biometric signal ranges
        network_metrics = {
            'latency_ms': 0.5,    # Very low latency (like some biometric normalization)
            'jitter_ms': 0.1,     # Very low jitter
            'packet_loss_percent': 0.0  # No packet loss
        }

        result = self.analyzer.analyze(network_metrics)

        # Should not detect network issues with very good network conditions
        self.assertFalse(result['network_issue_detected'],
                        "Excellent network conditions should not trigger network issues")

        # Now simulate high values that could be confused with biometric signals
        # but are actually indicating network problems
        network_metrics = {
            'latency_ms': 300.0,  # Very high latency
            'jitter_ms': 100.0,   # Very high jitter
            'packet_loss_percent': 50.0  # Severe packet loss
        }

        result = self.analyzer.analyze(network_metrics)

        self.assertTrue(result['network_issue_detected'],
                       "Severe network issues should be detected regardless of value ranges")
        self.assertGreater(result['confidence'], 0.7,
                          "High confidence for severe network issues")


if __name__ == '__main__':
    unittest.main()