"""
Test gaze diversion for frequent looking away scenario
"""
import unittest
from unittest.mock import patch
import numpy as np
from deepfake_interview_tool.analyzers.gaze_analyzer import GazeAnalyzer


class TestGazeDiversion(unittest.TestCase):
    """Test cases for GazeAnalyzer with frequent gaze diversion"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = GazeAnalyzer(window_size=10, deviation_threshold=0.3)

    def create_dummy_frame(self):
        """Create a dummy base64 encoded frame for testing"""
        # In the current implementation, _decode_frame ignores input and returns random frame
        # So we just need to provide a string that can be processed
        return "dummy_base64_frame"

    @patch.object(GazeAnalyzer, '_analyze_frame_gaze')
    def test_frequent_gaze_diversion_detection(self, mock_analyze_gaze):
        """Test that frequent looking away is detected as high gaze diversion"""
        # Mock the gaze analysis to return high deviation scores (looking away)
        # Return scores consistently above the deviation threshold (0.3)
        mock_analyze_gaze.side_effect = [0.8, 0.7, 0.9, 0.75, 0.85,  # First 5 frames: looking away
                                         0.8, 0.7, 0.9, 0.75, 0.85]  # Next 5 frames: looking away

        # Create a sequence of video frames (10 frames)
        video_frames = [self.create_dummy_frame() for _ in range(10)]

        result = self.analyzer.analyze(video_frames)

        # Verify high gaze diversion is detected
        self.assertTrue(result['gaze_diversion_detected'],
                       "Frequent gaze diversion should be detected")
        self.assertGreater(result['gaze_diversion_rate'], 0.7,
                          "Gaze diversion rate should be high for frequent looking away")
        self.assertGreater(result['confidence'], 0.5,
                          "Confidence should be reasonable with consistent signal")
        self.assertEqual(result['frames_analyzed'], 10)
        self.assertEqual(result['total_frames'], 10)

    @patch.object(GazeAnalyzer, '_analyze_frame_gaze')
    def test_occasional_gaze_diversion_not_detected(self, mock_analyze_gaze):
        """Test that occasional looking away doesn't trigger false positive"""
        # Mock the gaze analysis to return mostly low scores with occasional high scores
        mock_analyze_gaze.side_effect = [0.2, 0.1, 0.25, 0.15, 0.2,   # Normal gaze
                                         0.8, 0.2, 0.1, 0.2, 0.15]   # One frame of looking away

        video_frames = [self.create_dummy_frame() for _ in range(10)]

        result = self.analyzer.analyze(video_frames)

        # With only 1 out of 10 frames showing significant deviation,
        # gaze diversion rate should be low (~0.1)
        self.assertLess(result['gaze_diversion_rate'], 0.3,
                       "Occasional gaze diversion should not exceed threshold")
        # May or may not be detected depending on confidence, but rate should be low

    @patch.object(GazeAnalyzer, '_analyze_frame_gaze')
    def test_no_gaze_diversion(self, mock_analyze_gaze):
        """Test that normal gaze doesn't trigger diversion detection"""
        # Mock the gaze analysis to return low deviation scores (looking at center)
        mock_analyze_gaze.side_effect = [0.1, 0.15, 0.1, 0.2, 0.1,
                                         0.15, 0.1, 0.1, 0.2, 0.1]

        video_frames = [self.create_dummy_frame() for _ in range(10)]

        result = self.analyzer.analyze(video_frames)

        # Verify no gaze diversion detected
        self.assertFalse(result['gaze_diversion_detected'],
                        "Normal gaze should not be detected as diversion")
        self.assertLess(result['gaze_diversion_rate'], 0.1,
                       "Gaze diversion rate should be very low for normal gaze")

    def test_insufficient_frames(self):
        """Test handling of insufficient video frames"""
        # Test with empty frames
        result = self.analyzer.analyze([])
        self.assertTrue(result.get('insufficient_data', False),
                       "Should indicate insufficient data for empty frames")
        self.assertFalse(result['gaze_diversion_detected'])

        # Test with None or invalid frames (though current implementation handles via mock decode)
        result = self.analyzer.analyze(["invalid"] * 2)
        # Should still process since _decode_frame returns a frame
        self.assertEqual(result['frames_analyzed'], 2)

    def test_reset_history(self):
        """Test that history can be reset"""
        # Add some history
        with patch.object(GazeAnalyzer, '_analyze_frame_gaze') as mock_gaze:
            mock_gaze.return_value = 0.5
            frames = [self.create_dummy_frame() for _ in range(5)]
            self.analyzer.analyze(frames)

        # Verify history has data
        self.assertEqual(len(self.analyzer.gaze_history), 5)

        # Reset history
        self.analyzer.reset_history()

        # Verify history is cleared
        self.assertEqual(len(self.analyzer.gaze_history), 0)
        self.assertEqual(len(self.analyzer.frame_timestamps), 0)


if __name__ == '__main__':
    unittest.main()