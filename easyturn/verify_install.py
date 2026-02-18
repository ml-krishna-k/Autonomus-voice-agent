"""
Verification Script for EasyTurn System

Run this script to verify that the EasyTurn package is correctly installed
and all components can be initialized.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_system():
    print("=" * 60)
    print("EasyTurn System Verification")
    print("=" * 60)
    
    # 1. Check imports
    print("\n[1/4] Checking imports...", end=" ")
    try:
        from easyturn import (
            EasyTurnController, 
            AcousticExtractor, 
            DialogueOrchestrator,
            SystemConfig,
            PresetConfigs
        )
        print("OK")
    except ImportError as e:
        print("FAILED")
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        print("FAILED")
        logger.error(f"Unexpected error: {e}")
        return False

    # 2. Check Acoustic Extractor (requires librosa/numpy)
    print("[2/4] Initializing Acoustic Extractor...", end=" ")
    try:
        extractor = AcousticExtractor()
        # Test with dummy data
        import numpy as np
        dummy_frame = np.zeros(480, dtype=np.float32)
        features = extractor.extract(dummy_frame, timestamp_ms=0)
        
        if 'frame_energy' in features and 'vad_prob' in features:
            print("OK")
        else:
            print("FAILED (Missing features)")
            return False
            
    except Exception as e:
        print("FAILED")
        logger.error(f"Acoustic extractor error: {e}")
        return False

    # 3. Check Controller Logic
    print("[3/4] Initializing EasyTurn Controller...", end=" ")
    try:
        controller = EasyTurnController()
        # Verify initial state
        if controller.current_state.value == "HOLD":
            print("OK")
        else:
            print("FAILED (Incorrect initial state)")
            return False
    except Exception as e:
        print("FAILED")
        logger.error(f"Controller error: {e}")
        return False

    # 4. Check Orchestrator Configuration
    print("[4/4] Verifying Orchestrator Config...", end=" ")
    try:
        config = PresetConfigs.get_aggressive()
        if config.MIN_SILENCE_TO_SPEAK_MS < 400:
            print("OK")
        else:
            print("FAILED (Config loading issue)")
            return False
    except Exception as e:
        print("FAILED")
        logger.error(f"Config error: {e}")
        return False

    print("\n" + "=" * 60)
    print("VERIFICATION SUCCESSFUL")
    print("System is ready for integration.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    verify_system()
