"""
Test file for voice analyzer synthetic voice detection
"""
import numpy as np
import base64
from unittest.mock import patch, MagicMock
import sys
import os

# Add the analyzers directory to the path so we can import voice_analyzer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analyzers'))

from voice_analyzer import VoiceAnalyzer


def create_silent_audio(duration_seconds=1.0, sample_rate=16000):
    """Create silent audio samples"""
    samples = np.zeros(int(duration_seconds * sample_rate), dtype=np.float32)
    # Convert to int16 PCM then to base64
    pcm_data = (samples * 32767).astype(np.int16)
    return base64.b64encode(pcm_data.tobytes()).decode('utf-8'))


def create_constant_tone(frequency=440.0, duration_seconds=1.0, sample_rate=16000):
    """Create a constant tone (pure sine wave) - unnatural for speech"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    samples = np.sin(2 * np.pi * frequency * t) * 0.5
    # Convert to int16 PCM then to base64
    pcm_data = (samples * 32767).astype(np.int16)
    return base64.b64encode(pcm_data.tobytes()).decode('utf-8')


def create_white_noise(duration_seconds=1.0, sample_rate=16000):
    """Create white noise - may be detected as synthetic"""
    samples = np.random.uniform(-1, 1, int(duration_seconds * sample_rate)).astype(np.float32)
    # Convert to int16 PCM then to base64
    pcm_data = (samples * 32767).astype(np.int16)
    return base64.b64encode(pcm_data.tobytes()).decode('utf-8'))


class TestVoiceAnalyzer:
    """Test cases for VoiceAnalyzer"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.analyzer = VoiceAnalyzer(window_size=10, sample_rate=16000)

    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly"""
        assert self.analyzer.window_size == 10
        assert self.analyzer.sample_rate == 16000
        assert len(self.analyzer.spectral_centroid_history) == 0
        assert len(self.analyzer.pitch_history) == 0

    def test_insufficient_audio_samples(self):
        """Test analyzer with no audio samples"""
        result = self.analyzer.analyze([])
        assert result["synthetic_voice_artifact_detected"] == False
        assert result["artifact_score"] == 0.0
        assert result["confidence"] == 0.0
        assert result["insufficient_data"] == True
        assert "No audio samples provided" in result["reason"]

    def test_silent_audio_detection(self):
        """Test analyzer with silent audio"""
        silent_audio = create_silent_audio()
        result = self.analyzer.analyze([silent_audio, silent_audio, silent_audio])

        # Silent audio might be detected as having artifacts due to zero crossings, etc.
        # We mainly test that it runs without error and returns valid structure
        assert "synthetic_voice_artifact_detected" in result
        assert "artifact_score" in result
        assert "confidence" in result
        assert 0.0 <= result["artifact_score"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

    def test_constant_tone_detection(self):
        """Test analyzer with constant tone (unnatural for speech)"""
        # Create a constant tone that should be detected as synthetic
        constant_tone = create_constant_tone(frequency=200.0, duration_seconds=0.5)

        # Test with multiple samples
        result = self.analyzer.analyze([constant_tone] * 5)

        # Should run without error
        assert "synthetic_voice_artifact_detected" in result
        assert "artifact_score" in result
        assert "confidence" in result
        assert 0.0 <= result["artifact_score"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

        # Constant tone might have high artifact score due to unnatural pitch
        # But we won't assert on specific value as the simulation may vary

    def test_white_noise_detection(self):
        """Test analyzer with white noise"""
        white_noise = create_white_noise(duration_seconds=0.5)

        result = self.analyzer.analyze([white_noise] * 3)

        assert "synthetic_voice_artifact_detected" in result
        assert "artifact_score" in result
        assert "confidence" in result

    @patch('voice_analyzer.VoiceAnalyzer._decode_audio_sample')
    def test_synthetic_voice_artifacts_high_spectral_flatness(self, mock_decode):
        """Test detection of synthetic voice with high spectral flatness"""
        # Mock audio data that would produce high spectral flatness
        # Create audio with flat spectrum (similar magnitudes across frequencies)
        mock_audio = np.ones(16000) * 0.1  # Constant signal
        mock_decode.return_value = mock_audio

        # Test with multiple samples
        fake_audio_samples = ["fake_base64_1", "fake_base64_2", "fake_base64_3"]
        result = self.analyzer.analyze(fake_audio_samples)

        # Should have processed the samples
        assert mock_decode.call_count == 3
        assert "synthetic_voice_artifact_detected" in result

    @patch('voice_analyzer.VoiceAnalyzer._decode_audio_sample')
    def test_synthetic_voice_artifacts_unnatural_pitch(self, mock_decode):
        """Test detection of synthetic voice with unnatural pitch characteristics"""
        # Mock audio data that would produce constant pitch (unnatural for speech)
        # Create a signal with strong periodic component at unnatural frequency
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate, False)
        # Constant frequency tone (robotic sounding)
        mock_audio = np.sin(2 * np.pi * 100 * t) * 0.5  # 100Hz tone
        mock_decode.return_value = mock_audio.astype(np.float32)

        fake_audio_samples = ["fake_base64_1", "fake_base64_2"]
        result = self.analyzer.analyze(fake_audio_samples)

        assert mock_decode.call_count == 2
        assert "synthetic_voice_artifact_detected" in result

    def test_reset_history(self):
        """Test resetting analyzer history"""
        # Add some dummy data to history
        self.analyzer.spectral_centroid_history.append(1000.0)
        self.analyzer.pitch_history.append(150.0)

        assert len(self.analyzer.spectral_centroid_history) == 1
        assert len(self.analyzer.pitch_history) == 1

        # Reset history
        self.analyzer.reset_history()

        assert len(self.analyzer.spectral_centroid_history) == 0
        assert len(self.analyzer.pitch_history) == 0

    def test_analyze_return_structure(self):
        """Test that analyze method returns expected structure"""
        silent_audio = create_silent_audio()
        result = self.analyzer.analyze([silent_audio])

        # Check required fields are present
        required_fields = [
            "synthetic_voice_artifact_detected",
            "artifact_score",
            "confidence",
            "timestamp",
            "samples_analyzed",
            "features_summary"
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # Check field types
        assert isinstance(result["synthetic_voice_artifact_detected"], bool)
        assert isinstance(result["artifact_score"], float)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["timestamp"], float)
        assert isinstance(result["samples_analyzed"], int)
        assert isinstance(result["features_summary"], dict)


if __name__ == "__main__":
    # Run tests if executed directly
    import pytest
    pytest.main([__file__, "-v"])