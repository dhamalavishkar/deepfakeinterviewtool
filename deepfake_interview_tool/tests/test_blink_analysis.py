"""
Test blink analysis for abnormal blink patterns
"""
import unittest
from unittest.mock import patch
import sys
import os

# Add the project root to the path so we can import from deepfake_interview_tool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from deepfake_interview_tool.analyzers.blink_analyzer import BlinkAnalyzer


class TestBlinkAnalysis(unittest.TestCase):
    """Test cases for blink analysis"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = BlinkAnalyzer(window_size=30, blink_threshold=0.3)

    def create_mock_ear_sequence(self, ear_values):
        """
        Create a mock function that returns EAR values in sequence
        """
        def mock_ear(frame):
            if self.analyzer._call_count >= len(ear_values):
                # Return a default value if we run out of sequence
                return 0.4  # Open eye by default
            value = ear_values[self.analyzer._call_count]
            self.analyzer._call_count += 1
            return value
        return mock_ear

    def test_normal_blink_pattern(self):
        """Test that normal blink pattern is not marked as abnormal"""
        # Normal blink: open eye (0.4) -> closed eye (0.1) -> open eye (0.4)
        # Repeat every 10 frames (simulating ~3 blinks per second? Actually we need to think about timing)
        # Let's create a sequence that simulates normal blinking every ~4 seconds
        # Assuming we process 30 frames per second, 4 seconds = 120 frames
        # We'll make a shorter test for simplicity

        # Pattern: 10 frames open, 2 frames closed, 10 frames open (repeat)
        # This gives a blink about every 12 frames
        ear_sequence = []
        for _ in range(5):  # 5 cycles
            ear_sequence.extend([0.4] * 10)   # Open eye
            ear_sequence.extend([0.1] * 2)    # Closed eye (blink)

        # Mock the EAR calculation
        with patch.object(self.analyzer, '_calculate_eye_aspect_ratio',
                         side_effect=self.create_mock_ear_sequence(ear_sequence)):
            # Create dummy frames (one per EAR value)
            dummy_frames = ["dummy_frame_b64"] * len(ear_sequence)

            result = self.analyzer.analyze(dummy_frames)

            # Check that we detected blinks
            self.assertGreater(result["blink_count"], 0)
            # For normal pattern, abnormal_blink_pattern should be False
            self.assertFalse(result["abnormal_blink_pattern"],
                            f"Normal blink pattern incorrectly marked as abnormal. "
                            f"Blink rate: {result['blink_rate']}, "
                            f"Avg interval: {result.get('avg_blink_interval', 0)}")

    def test_too_frequent_blinks(self):
        """Test that too frequent blinks are detected as abnormal"""
        # Too frequent: blink every 3 frames (open 1, closed 1, open 1) -> very high blink rate
        ear_sequence = []
        for _ in range(20):  # 20 cycles
            ear_sequence.extend([0.4, 0.1, 0.4])  # open, closed, open

        with patch.object(self.analyzer, '_calculate_eye_aspect_ratio',
                         side_effect=self.create_mock_ear_sequence(ear_sequence)):
            dummy_frames = ["dummy_frame_b64"] * len(ear_sequence)

            result = self.analyzer.analyze(dummy_frames)

            self.assertGreater(result["blink_count"], 0)
            # Too frequent blinks should be marked as abnormal
            self.assertTrue(result["abnormal_blink_pattern"],
                           f"Too frequent blink pattern not detected as abnormal. "
                           f"Blink rate: {result['blink_rate']} blinks/min")

    def test_too_infrequent_blinks(self):
        """Test that too infrequent blinks are detected as abnormal"""
        # Too infrequent: long periods of open eyes with occasional blinks
        # Blink every 50 frames: 48 open, 2 closed
        ear_sequence = []
        for _ in range(10):  # 10 cycles
            ear_sequence.extend([0.4] * 48)   # Open eye
            ear_sequence.extend([0.1] * 2)    # Closed eye (blink)

        with patch.object(self.analyzer, '_calculate_eye_aspect_ratio',
                         side_effect=self.create_mock_ear_sequence(ear_sequence)):
            dummy_frames = ["dummy_frame_b64"] * len(ear_sequence)

            result = self.analyzer.analyze(dummy_frames)

            self.assertGreater(result["blink_count"], 0)
            # Too infrequent blinks should be marked as abnormal
            self.assertTrue(result["abnormal_blink_pattern"],
                           f"Too infrequent blink pattern not detected as abnormal. "
                           f"Blink rate: {result['blink_rate']} blinks/min")

    def test_irregular_blink_pattern(self):
        """Test that irregular blink pattern is detected as abnormal"""
        # Irregular: varying intervals between blinks
        # Pattern: blink, wait 5 frames, blink, wait 30 frames, blink, wait 5 frames, etc.
        ear_sequence = []
        intervals = [5, 30, 5, 30, 5]  # intervals between blinks (in frames)
        for interval in intervals:
            # Open eye for interval frames, then closed for 2 frames (blink)
            ear_sequence.extend([0.4] * interval)
            ear_sequence.extend([0.1] * 2)

        with patch.object(self.analyzer, '_calculate_eye_aspect_ratio',
                         side_effect=self.create_mock_ear_sequence(ear_sequence)):
            dummy_frames = ["dummy_frame_b64"] * len(ear_sequence)

            result = self.analyzer.analyze(dummy_frames)

            self.assertGreater(result["blink_count"], 0)
            # Irregular pattern should be marked as abnormal due to high variance
            self.assertTrue(result["abnormal_blink_pattern"],
                           f"Irregular blink pattern not detected as abnormal. "
                           f"Blink rate: {result['blink_rate']} blinks/min, "
                           f"Avg interval: {result.get('avg_blink_interval', 0)}")

    def test_insufficient_frames(self):
        """Test handling of insufficient frames"""
        # Very few frames
        dummy_frames = ["dummy_frame_b64"] * 5

        result = self.analyzer.analyze(dummy_frames)

        # Should return insufficient data response
        self.assertTrue(result.get("insufficient_data", False))
        self.assertEqual(result["blink_rate"], 0.0)
        self.assertFalse(result["abnormal_blink_pattern"])


if __name__ == '__main__':
    unittest.main()