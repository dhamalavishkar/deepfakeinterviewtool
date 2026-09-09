"""
Test lip-sync analyzer for temporal offset detection
"""
import unittest
import numpy as np
from deepfake_interview_tool.analyzers.lip_sync_analyzer import LipSyncAnalyzer


class TestableLipSyncAnalyzer(LipSyncAnalyzer):
    """LipSyncAnalyzer with mockable feature extraction for testing"""

    def __init__(self, mock_audio_energy=None, mock_lip_movement=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_audio_energy = mock_audio_energy
        self.mock_lip_movement = mock_lip_movement

    def _extract_audio_energy(self, audio_samples):
        return self.mock_audio_energy

    def _extract_lip_movement(self, video_frames):
        return self.mock_lip_movement


class TestLipSyncAnalyzer(unittest.TestCase):
    """Test cases for LipSyncAnalyzer"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = TestableLipSyncAnalyzer(window_size=150, sample_rate=16000)

    def test_zero_offset_detected_as_sync(self):
        """Test that zero temporal offset is detected as in-sync"""
        # Create matching audio and video signals
        # Simple signal: a burst in the middle
        audio_energy = np.array([0, 0, 0, 0.5, 1.0, 0.5, 0, 0, 0, 0, 0], dtype=float)
        lip_movement = np.array([0, 0, 0, 0.5, 1.0, 0.5, 0, 0, 0, 0, 0], dtype=float)

        # Mock the feature extraction
        analyzer = TestableLipSyncAnalyzer(
            mock_audio_energy=audio_energy,
            mock_lip_movement=lip_movement
        )

        # Provide dummy inputs (required by analyze method)
        video_frames = ["dummy"] * len(audio_energy)
        audio_samples = ["dummy"] * len(audio_energy)

        result = analyzer.analyze(video_frames, audio_samples)

        # Should detect good sync (high sync_score, low offset)
        self.assertAlmostEqual(result['temporal_offset_ms'], 0.0, delta=50.0,
                               msg="Temporal offset should be near zero for aligned signals")
        self.assertGreater(result['sync_score'], 0.5,
                           msg="Sync score should be high for aligned signals")
        # Depending on confidence calculation, it might not detect variance
        # We'll check that variance detected is False when sync is good
        self.assertFalse(result['lip_sync_variance_detected'],
                         msg="Should not detect variance for aligned signals")

    def test_positive_offset_detection(self):
        """Test detection of positive offset (audio leads video)"""
        # Create audio signal leading video signal by 2 samples
        # This means lip_movement is delayed relative to audio_energy
        audio_energy = np.array([0, 0, 0, 0.5, 1.0, 0.5, 0, 0, 0, 0, 0], dtype=float)
        # Shift lip_movement to the right by 2 (delay lip)
        lip_movement = np.zeros_like(audio_energy)
        lip_movement[2:] = audio_energy[:-2]  # shift right by 2

        analyzer = TestableLipSyncAnalyzer(
            mock_audio_energy=audio_energy,
            mock_lip_movement=lip_movement
        )

        video_frames = ["dummy"] * len(audio_energy)
        audio_samples = ["dummy"] * len(audio_energy)

        result = analyzer.analyze(video_frames, audio_samples)

        # Expected offset: audio leads by 2 samples -> offset negative
        # Assuming ~30fps -> ~33.3ms per frame, so offset = -2 * 33.3 = -66.6ms
        expected_offset_ms = -2 * (1000.0 / 30.0)
        self.assertAlmostEqual(result['temporal_offset_ms'], expected_offset_ms, delta=50.0,
                               msg=f"Temporal offset should be around {expected_offset_ms}ms for 2-sample lip delay")
        # Should detect variance because offset is significant
        self.assertTrue(result['lip_sync_variance_detected'],
                        msg="Should detect variance for significant offset")

    def test_negative_offset_detection(self):
        """Test detection of negative offset (video leads audio)"""
        # Create video signal leading audio signal by 3 samples
        # This means lip_movement is advanced relative to audio_energy
        audio_energy = np.array([0, 0, 0, 0, 0.5, 1.0, 0.5, 0, 0, 0, 0], dtype=float)
        # Shift lip_movement to the left by 3 (advance lip)
        lip_movement = np.zeros_like(audio_energy)
        lip_movement[:-3] = audio_energy[3:]  # shift left by 3

        analyzer = TestableLipSyncAnalyzer(
            mock_audio_energy=audio_energy,
            mock_lip_movement=lip_movement
        )

        video_frames = ["dummy"] * len(audio_energy)
        audio_samples = ["dummy"] * len(audio_energy)

        result = analyzer.analyze(video_frames, audio_samples)

        # Expected offset: video leads by 3 samples -> audio lags -> offset positive
        # Assuming ~30fps -> ~33.3ms per frame, so offset = +3 * 33.3 = +100ms
        expected_offset_ms = 3 * (1000.0 / 30.0)
        self.assertAlmostEqual(result['temporal_offset_ms'], expected_offset_ms, delta=50.0,
                               msg=f"Temporal offset should be around {expected_offset_ms}ms for 3-sample lip advance")
        # Should detect variance
        self.assertTrue(result['lip_sync_variance_detected'],
                        msg="Should detect variance for significant offset")

    def test_various_offset_amounts(self):
        """Test detection with various offset amounts"""
        offsets_samples = [1, 2, 5, 10, 15]  # in samples, positive means lip delayed
        base_signal = np.array([0, 0, 0, 0.5, 1.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

        for offset in offsets_samples:
            with self.subTest(offset=offset):
                # Create audio and video signals with lip delayed by offset samples
                audio_energy = base_signal.copy()
                lip_movement = np.zeros_like(base_signal)
                if offset < len(base_signal):
                    lip_movement[offset:] = base_signal[:-offset]
                else:
                    lip_movement = np.zeros_like(base_signal)

                analyzer = TestableLipSyncAnalyzer(
                    mock_audio_energy=audio_energy,
                    mock_lip_movement=lip_movement
                )

                video_frames = ["dummy"] * len(audio_energy)
                audio_samples = ["dummy"] * len(audio_energy)

                result = analyzer.analyze(video_frames, audio_samples)

                # Expected offset in milliseconds: negative because lip delayed
                expected_offset_ms = -offset * (1000.0 / 30.0)
                self.assertAlmostEqual(result['temporal_offset_ms'], expected_offset_ms, delta=50.0,
                                       msg=f"Lip delayed by {offset} samples should yield offset ~{expected_offset_ms}ms")
                # For larger offsets, we expect variance detection
                if offset >= 2:  # Assuming threshold is around 1-2 samples
                    self.assertTrue(result['lip_sync_variance_detected'],
                                    msg=f"Should detect variance for lip delayed by {offset} samples")

    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        # Too few samples
        audio_energy = np.array([0.5, 1.0], dtype=float)
        lip_movement = np.array([0.5, 1.0], dtype=float)

        analyzer = TestableLipSyncAnalyzer(
            mock_audio_energy=audio_energy,
            mock_lip_movement=lip_movement
        )

        video_frames = ["dummy"] * len(audio_energy)
        audio_samples = ["dummy"] * len(audio_energy)

        result = analyzer.analyze(video_frames, audio_samples)

        # Should indicate insufficient data
        self.assertTrue(result.get('insufficient_data', False),
                        msg="Should report insufficient data for too few samples")
        self.assertFalse(result['lip_sync_variance_detected'],
                         msg="Should not detect variance with insufficient data")

    def test_no_data(self):
        """Test handling of missing data"""
        analyzer = TestableLipSyncAnalyzer()

        result = analyzer.analyze([], [])

        self.assertTrue(result.get('insufficient_data', False),
                        msg="Should report insufficient data for empty inputs")
        self.assertFalse(result['lip_sync_variance_detected'],
                         msg="Should not detect variance with no data")


if __name__ == '__main__':
    unittest.main()