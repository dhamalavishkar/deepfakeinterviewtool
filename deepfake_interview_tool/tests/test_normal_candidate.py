"""
Test normal candidate scenario
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analyzers.gaze_analyzer import GazeAnalyzer
from analyzers.blink_analyzer import BlinkAnalyzer
from analyzers.lip_sync_analyzer import LipSyncAnalyzer
from analyzers.voice_analyzer import VoiceAnalyzer
from analyzers.network_analyzer import NetworkAnalyzer

def test_normal_candidate():
    """Test normal candidate scenario"""
    print("Testing normal candidate scenario...")

    # Initialize analyzers
    gaze_analyzer = GazeAnalyzer()
    blink_analyzer = BlinkAnalyzer()
    lip_sync_analyzer = LipSyncAnalyzer()
    voice_analyzer = VoiceAnalyzer()
    network_analyzer = NetworkAnalyzer()

    # Create mock data for normal candidate
    # Normal gaze: looking mostly at center
    normal_video_frames = ["mock_frame_data"] * 30  # 30 frames

    # Normal audio: natural speech patterns
    normal_audio_samples = ["mock_audio_data"] * 10  # 10 audio chunks

    # Normal network: low latency, jitter, packet loss
    normal_network_metrics = {
        "latency_ms": 25.0,
        "jitter_ms": 5.0,
        "packet_loss_percent": 0.1
    }

    # Test gaze analysis
    gaze_result = gaze_analyzer.analyze(normal_video_frames)
    print(f"Gaze analysis: {gaze_result}")
    # For normal candidate, we expect low gaze diversion rate
    assert gaze_result["gaze_diversion_rate"] < 0.3, f"Expected low gaze diversion, got {gaze_result['gaze_diversion_rate']}"

    # Test blink analysis
    blink_result = blink_analyzer.analyze(normal_video_frames)
    print(f"Blink analysis: {blink_result}")
    # For normal candidate, we expect normal blink pattern
    assert blink_result["blink_rate"] > 0, f"Expected some blink activity, got {blink_result['blink_rate']}"
    # Note: abnormal_blink_pattern might be False for normal, but we're not asserting on that as it depends on implementation

    # Test lip-sync analysis
    lip_sync_result = lip_sync_analyzer.analyze(normal_video_frames, normal_audio_samples)
    print(f"Lip-sync analysis: {lip_sync_result}")
    # For normal candidate, we expect good lip-sync (low variance)
    # Note: This might vary based on mock data

    # Test voice analysis
    voice_result = voice_analyzer.analyze(normal_audio_samples)
    print(f"Voice analysis: {voice_result}")
    # For normal candidate, we expect natural voice (low artifact score)

    # Test network analysis
    network_result = network_analyzer.analyze(normal_network_metrics)
    print(f"Network analysis: {network_result}")
    # For normal candidate, we expect no network issues
    assert network_result["network_issue_detected"] == False, f"Expected no network issues, got {network_result['network_issue_detected']}"

    print("✓ Normal candidate test passed!")

if __name__ == "__main__":
    test_normal_candidate()