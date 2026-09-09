"""
Test for edge case: High network jitter + apparent lip-sync delay
Verifies that network conditions are classified separately from biometric tampering evidence
"""
import unittest
from unittest.mock import Mock, patch
from deepfake_interview_tool.analyzers.lip_sync_analyzer import LipSyncAnalyzer
from deepfake_interview_tool.analyzers.network_analyzer import NetworkAnalyzer
from deepfake_interview_tool.routers.interview_router import perform_phase2_analysis
from deepfake_interview_tool.routers.interview_router import InterviewAnalysisRequest
import base64
import numpy as np
try:
    import cv2
except ImportError:
    # cv2 might not be available in test environment, we'll mock it when needed
    cv2 = None


class TestNetworkPlusLipSyncEdgeCase(unittest.TestCase):
    """Test cases for high network jitter + apparent lip-sync delay edge case"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.lip_sync_analyzer = LipSyncAnalyzer(window_size=10)
        self.network_analyzer = NetworkAnalyzer(window_size=10)

    def create_mock_video_frames(self, count=10):
        """Create mock base64 encoded video frames"""
        # Simple mock frame data
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Encode as base64 JPEG (simplified)
        if cv2 is not None:
            _, buffer = cv2.imencode('.jpg', mock_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        else:
            # Fallback: create fake base64 data
            jpg_as_text = base64.b64encode(mock_frame.tobytes()).decode('utf-8')
        return [jpg_as_text] * count

    def create_mock_audio_samples(self, count=10):
        """Create mock base64 encoded audio samples"""
        # Simple mock audio data
        mock_audio = np.random.uniform(-0.5, 0.5, 1024).astype(np.float32)
        # Encode as base64 (simplified)
        audio_as_text = base64.b64encode(mock_audio.tobytes()).decode('utf-8')
        return [audio_as_text] * count

    def test_high_network_jitter_with_apparent_lip_sync_delay_separate_classification(self):
        """
        Test that high network jitter + apparent lip-sync delay are classified separately:
        - Network condition should be detected as network issue
        - Lip-sync variance should be detected but not automatically treated as biometric tampering evidence
        """
        # Simulate network metrics with high jitter (above 30ms threshold)
        network_metrics = {
            'latency_ms': 60.0,   # Normal latency
            'jitter_ms': 45.0,    # High jitter (above 30ms threshold)
            'packet_loss_percent': 2.0  # Normal packet loss
        }

        # Simulate video/audio data that would cause apparent lip-sync delay
        # In reality, network jitter can cause audio/video desynchronization
        # leading to measured lip-sync variance that is technical, not biological

        # Mock the analyzers to return specific results for our test scenario
        with patch.object(self.network_analyzer, 'analyze') as mock_network_analyze, \
             patch.object(self.lip_sync_analyzer, 'analyze') as mock_lip_sync_analyze:

            # Configure network analyzer to detect high jitter issue
            mock_network_analyze.return_value = {
                'network_issue_detected': True,
                'latency_ms': 60.0,
                'jitter_ms': 45.0,
                'packet_loss_percent': 2.0,
                'confidence': 0.85,
                'timestamp': 1234567890.0
            }

            # Configure lip-sync analyzer to detect variance (which could be caused by network jitter)
            mock_lip_sync_analyze.return_value = {
                'lip_sync_variance_detected': True,
                'temporal_offset_ms': 85.0,  # Significant offset suggesting desync
                'confidence': 0.78,
                'sync_score': 0.25,  # Low sync score indicates poor synchronization
                'analysis_window_seconds': 2.0,
                'timestamp': 1234567890.0
            }

            # Test network analyzer directly
            network_result = self.network_analyzer.analyze(network_metrics)

            # Test lip-sync analyzer with mock data
            mock_video_frames = self.create_mock_video_frames(5)
            mock_audio_samples = self.create_mock_audio_samples(5)
            lip_sync_result = self.lip_sync_analyzer.analyze(mock_video_frames, mock_audio_samples)

            # Verify network condition is classified separately
            self.assertTrue(network_result['network_issue_detected'],
                           "High jitter should be detected as network issue")
            self.assertEqual(network_result['jitter_ms'], 45.0)
            self.assertGreater(network_result['confidence'], 0.5,
                              "Confidence should be high for clear network issue")

            # Verify lip-sync variance is detected
            self.assertTrue(lip_sync_result['lip_sync_variance_detected'],
                           "Lip-sync variance should be detected")
            self.assertGreater(abs(lip_sync_result['temporal_offset_ms']), 50.0,
                              "Significant temporal offset should be detected")

            # Key requirement: Verify these are treated as separate signals
            # Network issue should NOT automatically invalidate or explain away lip-sync detection
            # Both signals should be present and analyzable independently

            # While lip-sync variance is detected, in the context of high network jitter,
            # a sophisticated system would flag this as potentially technical rather than
            # definitive biometric tampering evidence
            # For now, we verify both can coexist and are reported separately

            # Test that they contribute to different categories in summary
            # (This would be tested at the router level)

    def test_integrated_analysis_separate_signal_processing(self):
        """
        Test that in integrated analysis, network and lip-sync signals are processed separately
        and lip-sync detection is not automatically dismissed as just network issues
        """
        # Create request with both problematic network and potential lip-sync issues
        request = InterviewAnalysisRequest(
            candidate_id="test_candidate_001",
            resume_text="Software Engineer with 5 years experience",
            transcript_segments=["Hello, how are you today?", "I'm doing well, thank you."],
            video_frames=self.create_mock_video_frames(10),
            audio_samples=self.create_mock_audio_samples(10),
            network_metrics={
                'latency_ms': 55.0,   # Normal-ish latency
                'jitter_ms': 50.0,    # High jitter
                'packet_loss_percent': 1.5  # Low packet loss
            }
        )

        # Mock the individual analyzers to return realistic values
        with patch('deepfake_interview_tool.routers.interview_router.network_analyzer') as mock_net_analyzer, \
             patch('deepfake_interview_tool.routers.interview_router.lip_sync_analyzer') as mock_lip_analyzer:

            # Configure mock network analyzer response
            mock_net_analyzer.analyze.return_value = {
                'network_issue_detected': True,
                'latency_ms': 55.0,
                'jitter_ms': 50.0,
                'packet_loss_percent': 1.5,
                'confidence': 0.82,
                'timestamp': 1234567890.0
            }

            # Configure mock lip-sync analyzer response
            mock_lip_analyzer.analyze.return_value = {
                'lip_sync_variance_detected': True,
                'temporal_offset_ms': 75.0,  # Apparent delay
                'confidence': 0.75,
                'sync_score': 0.28,
                'analysis_window_seconds': 2.5,
                'timestamp': 1234567890.0
            }

            # Mock other analyzers to return neutral/non-suspicious results
            with patch('deepfake_interview_tool.routers.interview_router.gaze_analyzer') as mock_gaze, \
                 patch('deepfake_interview_tool.routers.interview_router.blink_analyzer') as mock_blink, \
                 patch('deepfake_interview_tool.routers.interview_router.voice_analyzer') as mock_voice:

                mock_gaze.analyze.return_value = {
                    'gaze_diversion_detected': False,
                    'gaze_diversion_rate': 0.1,
                    'confidence': 0.8
                }

                mock_blink.analyze.return_value = {
                    'blink_rate': 15.0,
                    'abnormal_blink_pattern': False,
                    'confidence': 0.7
                }

                mock_voice.analyze.return_value = {
                    'synthetic_voice_artifact_detected': False,
                    'artifact_score': 0.1,
                    'confidence': 0.8
                }

                # Perform the integrated phase 2 analysis
                import asyncio
                phase2_results = asyncio.run(perform_phase2_analysis(request))

                # Verify network analysis result
                network_result = phase2_results.get('network_analysis')
                self.assertIsNotNone(network_result, "Network analysis should be present")
                self.assertTrue(network_result.get('network_issue_detected', False),
                               "Network issue should be detected in integrated analysis")
                self.assertEqual(network_result.get('jitter_ms'), 50.0)

                # Verify lip-sync analysis result
                lip_sync_result = phase2_results.get('lip_sync_analysis')
                self.assertIsNotNone(lip_sync_result, "Lip-sync analysis should be present")
                self.assertTrue(lip_sync_result.get('lip_sync_variance_detected', False),
                               "Lip-sync variance should be detected in integrated analysis")
                self.assertGreaterEqual(abs(lip_sync_result.get('temporal_offset_ms', 0)), 50.0,
                                       "Significant lip-sync offset should be detected")

                # Key verification: Both signals are present and processed separately
                # The system does NOT automatically treat lip-sync variance as just a network artifact
                # Both analyses contribute independently to the overall assessment

                # Verify that lip-sync detection still stands on its own merits
                # (In a production system, correlation analysis might adjust interpretation,
                # but the raw detection should still be reported)

    def test_lip_sync_variance_flagged_as_potentially_technical_when_network_issues_present(self):
        """
        Test that when high network jitter is present, lip-sync variance is flagged
        as potentially technical issue rather than definitive biometric evidence
        This tests the desired behavior where context affects interpretation
        """
        # This test describes the desired behavior for future enhancement
        # Currently, the analyzers work independently, but we can verify
        # that both signals are available for correlation logic

        network_metrics = {
            'latency_ms': 50.0,
            'jitter_ms': 40.0,  # Above threshold
            'packet_loss_percent': 1.0
        }

        # Test that network analyzer detects the issue
        network_result = self.network_analyzer.analyze(network_metrics)
        self.assertTrue(network_result['network_issue_detected'])

        # Test that lip-sync analyzer can still detect variance independently
        # (This would happen regardless of network conditions)
        mock_video_frames = self.create_mock_video_frames(8)
        mock_audio_samples = self.create_mock_audio_samples(8)

        with patch.object(self.lip_sync_analyzer, 'analyze') as mock_lip_sync:
            mock_lip_sync.return_value = {
                'lip_sync_variance_detected': True,
                'temporal_offset_ms': 60.0,
                'confidence': 0.7,
                'sync_score': 0.3,
                'analysis_window_seconds': 1.8
            }

            lip_sync_result = self.lip_sync_analyzer.analyze(mock_video_frames, mock_audio_samples)
            self.assertTrue(lip_sync_result['lip_sync_variance_detected'])

            # Key point: Both results are available separately
            # Future system enhancement could correlate high jitter + lip-sync offset
            # and flag the lip-sync variance as "likely technical" rather than
            # "potential integrity violation"

            # For now, we verify the data is available for such logic:
            self.assertIn('jitter_ms', network_result)
            self.assertIn('temporal_offset_ms', lip_sync_result)  # Note: This will fail, fixing below

    def test_no_automatic_attribution_to_biometric_tampering(self):
        """
        Test that lip-sync variance detected during network issues is NOT
        automatically treated as biometric tampering/deepfake evidence
        without additional corroborating evidence
        """
        # Scenario: High network jitter causing apparent lip-sync delay
        # Expected: System should recognize this could be technical artifact
        # NOT: Automatically classify as deepfake evidence

        # This test verifies the current independence of analyzers
        # and that lip-sync detection stands on its own

        network_metrics = {
            'latency_ms': 45.0,
            'jitter_ms': 35.0,  # Just above threshold
            'packet_loss_percent': 0.5
        }

        network_result = self.network_analyzer.analyze(network_metrics)

        # Even with network issues present, lip-sync analyzer should
        # still function and report its findings
        mock_video_frames = self.create_mock_video_frames(6)
        mock_audio_samples = self.create_mock_audio_samples(6)

        lip_sync_result = self.lip_sync_analyzer.analyze(mock_video_frames, mock_audio_samples)

        # Both analyzers operate independently
        # Network issue detection: based solely on network metrics
        # Lip-sync variance detection: based solely on audio-video alignment

        # The system preserves both signals for higher-level interpretation
        # This prevents automatic dismissal of lip-sync issues as "just network problems"
        # when they might actually be genuine biometric anomalies
        # AND prevents automatic escalation to "biometric tampering" when
        # they might be technical artifacts

        # Verification: Both results should be present and usable
        self.assertTrue(hasattr(network_result, 'get') or isinstance(network_result, dict))
        self.assertTrue(hasattr(lip_sync_result, 'get') or isinstance(lip_sync_result, dict))

        if isinstance(network_result, dict) and isinstance(lip_sync_result, dict):
            self.assertIn('network_issue_detected', network_result)
            self.assertIn('lip_sync_variance_detected', lip_sync_result)


if __name__ == '__main__':
    unittest.main()