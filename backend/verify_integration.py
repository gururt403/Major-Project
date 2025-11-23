"""
Quick verification script to ensure all modules are importable
Run this to check if the integration is working correctly
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("INTEGRATION VERIFICATION SCRIPT")
print("=" * 60)

try:
    print("\n1. Testing MeshDetector import...")
    from mesh_detector import MeshDetector
    print("   ✓ MeshDetector imported successfully")
    
    print("\n2. Testing MeshDetector initialization...")
    detector = MeshDetector(max_history=15, fps=30)
    print("   ✓ MeshDetector initialized successfully")
    print(f"     - Max history: {detector.history.maxlen}")
    print(f"     - FPS: {detector.fps}")
    print(f"     - Threshold effort: {detector.THRESH_EFFORT}")
    print(f"     - Threshold risk: {detector.THRESH_RISK}")
    print(f"     - Threshold warning: {detector.THRESH_WARN}")
    
    print("\n3. Testing other imports...")
    from rppg_estimator import rPPGEstimator
    print("   ✓ rPPGEstimator imported")
    
    from mhavh_model import MHAVH
    print("   ✓ MHAVH imported")
    
    from config import Config
    print("   ✓ Config imported")
    
    print("\n4. Verifying FastAPI setup...")
    from fastapi import FastAPI
    app = FastAPI()
    print("   ✓ FastAPI initialized")
    
    print("\n5. Configuration Check:")
    print(f"   - API Port: {Config.API_PORT}")
    print(f"   - Frame Width: {Config.FRAME_WIDTH}")
    print(f"   - Frame Height: {Config.FRAME_HEIGHT}")
    print(f"   - HR Normal Range: {Config.HR_NORMAL_MIN}-{Config.HR_NORMAL_MAX} BPM")
    
    print("\n" + "=" * 60)
    print("✓ ALL VERIFICATIONS PASSED!")
    print("=" * 60)
    print("\nThe integration is ready. You can now:")
    print("  1. Start backend: python app.py")
    print("  2. Start frontend: cd v0-heart-attack-detection-ui && pnpm dev")
    print("  3. Open http://localhost:3000")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nStack trace:")
    import traceback
    traceback.print_exc()
    print("\nPlease check the error above and ensure:")
    print("  - mesh_detector.py is in the backend directory")
    print("  - All dependencies are installed (pip install -r requirements.txt)")
    print("  - Python version is 3.8+")
    sys.exit(1)
