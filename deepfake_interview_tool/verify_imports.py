#!/usr/bin/env python3
"""
Verification script to check that all analyzers can be imported and instantiated
"""

def test_imports():
    """Test that all modules can be imported"""
    try:
        print("Testing imports...")

        # Test analyzers
        from analyzers.gaze_analyzer import GazeAnalyzer
        from analyzers.blink_analyzer import BlinkAnalyzer
        from analyzers.lip_sync_analyzer import LipSyncAnalyzer
        from analyzers.voice_analyzer import VoiceAnalyzer
        from analyzers.network_analyzer import NetworkAnalyzer

        print("[PASS] All analyzers imported successfully")

        # Test instantiation
        gaze = GazeAnalyzer()
        blink = BlinkAnalyzer()
        lip_sync = LipSyncAnalyzer()
        voice = VoiceAnalyzer()
        network = NetworkAnalyzer()

        print("[PASS] All analyzers instantiated successfully")

        # Test routers
        from routers.interview_router import router
        print("[PASS] Interview router imported successfully")

        # Test main app
        from main import app
        print("[PASS] Main app imported successfully")

        return True

    except Exception as e:
        print(f"[FAIL] Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n[SUCCESS] All import tests passed!")
    else:
        print("\n[FAILURE] Import tests failed!")
        exit(1)