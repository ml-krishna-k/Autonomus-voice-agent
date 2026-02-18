"""
Example usage and integration demo

This demonstrates how to integrate the EasyTurn system with your existing backend.
"""

import asyncio
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from easyturn.dialogue_orchestrator import DialogueOrchestrator, SystemConfig
from easyturn.integrations import BackendAdapter


async def main():
    """
    Example integration with your existing backend.
    
    This shows the complete flow from audio input to system response.
    """
    
    print("=" * 70)
    print("FULL-DUPLEX DIALOGUE SYSTEM - EXAMPLE")
    print("=" * 70)
    
    # Step 1: Create client wrappers for your existing services
    print("\n[1/5] Initializing service wrappers...")
    asr_wrapper = BackendAdapter.create_asr_wrapper()
    llm_wrapper = BackendAdapter.create_llm_wrapper()
    tts_wrapper = BackendAdapter.create_tts_wrapper()
    print("✓ Service wrappers created")
    
    # Step 2: Configure the system
    print("\n[2/5] Configuring system...")
    config = SystemConfig(
        sample_rate=16000,
        frame_duration_ms=30,
        min_silence_to_speak_ms=400,
        hysteresis_window_ms=200,
        interruption_latency_ms=50,
        log_decisions=True
    )
    print(f"✓ Configuration loaded:")
    print(f"  - Frame rate: {config.frame_duration_ms}ms")
    print(f"  - Min silence: {config.min_silence_to_speak_ms}ms")
    print(f"  - Hysteresis: {config.hysteresis_window_ms}ms")
    
    # Step 3: Create orchestrator
    print("\n[3/5] Creating dialogue orchestrator...")
    orchestrator = DialogueOrchestrator(
        config=config,
        asr_client=asr_wrapper,
        llm_client=llm_wrapper,
        tts_client=tts_wrapper
    )
    print("✓ Orchestrator created")
    
    # Step 4: Start the system
    print("\n[4/5] Starting system...")
    await orchestrator.start()
    print("✓ System started - ready for audio input")
    
    # Step 5: Simulate audio processing
    print("\n[5/5] Running simulation...")
    print("-" * 70)
    await simulate_conversation(orchestrator)
    
    # Step 6: Stop the system
    print("\n[CLEANUP] Stopping system...")
    await orchestrator.stop()
    print("✓ System stopped")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


async def simulate_conversation(orchestrator: DialogueOrchestrator):
    """
    Simulate a conversation by sending synthetic audio frames.
    
    In production, replace this with actual audio input from microphone.
    """
    
    print("\nSimulating conversation scenario...")
    print("Scenario: User says 'Hello', system responds, user interrupts")
    print()
    
    # Simulation parameters
    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 30
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    
    current_time_ms = 0.0
    
    # Phase 1: User speaking "Hello" (simulated with noise + speech)
    print(f"[{current_time_ms:07.1f}ms] 🎤 User starts speaking...")
    for i in range(50):  # ~1.5 seconds of speech
        # Simulate speech with higher energy
        audio_frame = np.random.randn(FRAME_SIZE).astype(np.float32) * 0.05
        
        await orchestrator.process_audio_frame(audio_frame, current_time_ms)
        current_time_ms += FRAME_DURATION_MS
        
        if i % 10 == 0:  # Log every ~300ms
            metrics = orchestrator.get_metrics()
            print(f"[{current_time_ms:07.1f}ms] State: {metrics['turn_state']:<5} | "
                  f"System: {metrics['system_state']}")
    
    # Phase 2: User stops speaking (silence)
    print(f"\n[{current_time_ms:07.1f}ms] 🤐 User stops speaking (silence)...")
    for i in range(20):  # ~600ms of silence
        # Simulate silence with low energy
        audio_frame = np.random.randn(FRAME_SIZE).astype(np.float32) * 0.001
        
        await orchestrator.process_audio_frame(audio_frame, current_time_ms)
        current_time_ms += FRAME_DURATION_MS
        
        if i % 5 == 0:  # Log every ~150ms
            metrics = orchestrator.get_metrics()
            print(f"[{current_time_ms:07.1f}ms] State: {metrics['turn_state']:<5} | "
                  f"System: {metrics['system_state']}")
    
    # Phase 3: System should now be responding
    print(f"\n[{current_time_ms:07.1f}ms] 🤖 System should be responding...")
    for i in range(30):  # ~900ms of system response
        # Simulate silence during system response
        audio_frame = np.random.randn(FRAME_SIZE).astype(np.float32) * 0.001
        
        await orchestrator.process_audio_frame(audio_frame, current_time_ms)
        current_time_ms += FRAME_DURATION_MS
        
        if i % 10 == 0:
            metrics = orchestrator.get_metrics()
            print(f"[{current_time_ms:07.1f}ms] State: {metrics['turn_state']:<5} | "
                  f"System: {metrics['system_state']}")
    
    # Phase 4: User interrupts
    print(f"\n[{current_time_ms:07.1f}ms] ⚡ User interrupts!")
    for i in range(20):  # ~600ms of interruption
        # Simulate user speech during system response
        audio_frame = np.random.randn(FRAME_SIZE).astype(np.float32) * 0.04
        
        await orchestrator.process_audio_frame(audio_frame, current_time_ms)
        current_time_ms += FRAME_DURATION_MS
        
        if i % 5 == 0:
            metrics = orchestrator.get_metrics()
            print(f"[{current_time_ms:07.1f}ms] State: {metrics['turn_state']:<5} | "
                  f"System: {metrics['system_state']} (should transition to HOLD)")
    
    # Final metrics
    print("\n" + "-" * 70)
    print("FINAL METRICS:")
    metrics = orchestrator.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")


async def example_real_microphone_integration():
    """
    Example of how to integrate with real microphone input.
    
    This shows the pattern for production use.
    """
    
    # This is pseudocode - adapt to your audio library (PyAudio, sounddevice, etc.)
    
    # import pyaudio
    # 
    # SAMPLE_RATE = 16000
    # FRAME_DURATION_MS = 30
    # FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    # 
    # # Initialize orchestrator (as shown in main())
    # orchestrator = ...
    # await orchestrator.start()
    # 
    # # Audio callback
    # def audio_callback(in_data, frame_count, time_info, status):
    #     # Convert bytes to numpy array
    #     audio_frame = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    #     
    #     # Get timestamp
    #     timestamp_ms = time.time() * 1000
    #     
    #     # Process frame (non-blocking)
    #     asyncio.create_task(orchestrator.process_audio_frame(audio_frame, timestamp_ms))
    #     
    #     return (in_data, pyaudio.paContinue)
    # 
    # # Start audio stream
    # p = pyaudio.PyAudio()
    # stream = p.open(
    #     format=pyaudio.paInt16,
    #     channels=1,
    #     rate=SAMPLE_RATE,
    #     input=True,
    #     frames_per_buffer=FRAME_SIZE,
    #     stream_callback=audio_callback
    # )
    # 
    # stream.start_stream()
    # 
    # # Keep running
    # while stream.is_active():
    #     await asyncio.sleep(0.1)
    
    print("See source code for real microphone integration pattern")


def example_threshold_tuning():
    """
    Example of how to tune thresholds for your environment.
    """
    
    from easyturn.config import PresetConfigs, get_tuning_guide
    
    print("\n=== THRESHOLD TUNING EXAMPLE ===\n")
    
    # Option 1: Use preset
    print("Option 1: Use preset configuration")
    config_default = PresetConfigs.get_default()
    config_aggressive = PresetConfigs.get_aggressive()
    config_conservative = PresetConfigs.get_conservative()
    
    print(f"Default MIN_SILENCE: {config_default.MIN_SILENCE_TO_SPEAK_MS}ms")
    print(f"Aggressive MIN_SILENCE: {config_aggressive.MIN_SILENCE_TO_SPEAK_MS}ms")
    print(f"Conservative MIN_SILENCE: {config_conservative.MIN_SILENCE_TO_SPEAK_MS}ms")
    
    # Option 2: Custom tuning
    print("\nOption 2: Custom configuration")
    custom_config = PresetConfigs.get_default()
    custom_config.MIN_SILENCE_TO_SPEAK_MS = 350  # Faster response
    custom_config.HYSTERESIS_WINDOW_MS = 250     # More stability
    
    print(f"Custom MIN_SILENCE: {custom_config.MIN_SILENCE_TO_SPEAK_MS}ms")
    print(f"Custom HYSTERESIS: {custom_config.HYSTERESIS_WINDOW_MS}ms")
    
    # Option 3: Get tuning guide
    print("\n" + "=" * 70)
    print(get_tuning_guide())


if __name__ == "__main__":
    # Run main demo
    print("\nRunning main demo...")
    asyncio.run(main())
    
    # Show threshold tuning example
    print("\n\n")
    example_threshold_tuning()
