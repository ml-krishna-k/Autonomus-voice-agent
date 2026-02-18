# Testing Guide for EasyTurn System

## Overview

This guide covers unit testing, integration testing, and production validation for the EasyTurn full-duplex dialogue system.

## Unit Tests

### 1. Testing the EasyTurn Controller

```python
import pytest
import numpy as np
from easyturn import EasyTurnController, AcousticFeatures, LinguisticFeatures, TurnState

def test_controller_initialization():
    """Test controller initializes correctly"""
    controller = EasyTurnController(frame_rate_ms=30)
    assert controller.current_state == TurnState.HOLD
    assert controller.frame_rate_ms == 30

def test_hold_to_speak_transition():
    """Test transition from HOLD to SPEAK when user finishes"""
    controller = EasyTurnController()
    
    # Simulate user speaking
    for i in range(50):  # 1.5 seconds
        acoustic = AcousticFeatures(
            timestamp_ms=i * 30,
            frame_energy=0.05,  # High energy
            vad_prob=0.9,       # High VAD
            silence_duration_ms=0
        )
        linguistic = LinguisticFeatures(
            timestamp_ms=i * 30,
            partial_text="Hello how are you",
            asr_confidence=0.8,
            token_stability=0.5,  # Still evolving
            text_length=17
        )
        decision = controller.update(acoustic, linguistic)
        assert decision.state == TurnState.HOLD  # Should stay in HOLD
    
    # Simulate user stops speaking
    for i in range(50, 70):  # 600ms silence
        acoustic = AcousticFeatures(
            timestamp_ms=i * 30,
            frame_energy=0.002,  # Low energy
            vad_prob=0.1,        # Low VAD
            silence_duration_ms=(i - 50) * 30
        )
        linguistic = LinguisticFeatures(
            timestamp_ms=i * 30,
            partial_text="Hello how are you",  # Stable text
            asr_confidence=0.8,
            token_stability=0.9,  # Stable now
            text_length=17
        )
        decision = controller.update(acoustic, linguistic)
    
    # After silence + hysteresis, should be SPEAK
    assert decision.state == TurnState.SPEAK

def test_interruption_detection():
    """Test system detects user interruption"""
    controller = EasyTurnController()
    
    # Force controller to SPEAK state
    controller.current_state = TurnState.SPEAK
    
    # Simulate user interrupting
    acoustic = AcousticFeatures(
        timestamp_ms=0,
        frame_energy=0.03,  # High energy (interrupt threshold)
        vad_prob=0.8,
        silence_duration_ms=0
    )
    linguistic = LinguisticFeatures(
        timestamp_ms=0,
        partial_text="Wait",
        asr_confidence=0.7,
        token_stability=0.5,
        text_length=4
    )
    
    # Process for hysteresis window
    for i in range(10):  # 300ms
        acoustic.timestamp_ms = i * 30
        linguistic.timestamp_ms = i * 30
        decision = controller.update(acoustic, linguistic)
    
    # Should transition to HOLD
    assert decision.state == TurnState.HOLD

def test_hysteresis_prevents_flapping():
    """Test hysteresis prevents rapid state changes"""
    controller = EasyTurnController()
    
    # Simulate borderline conditions that flip rapidly
    for i in range(20):
        if i % 2 == 0:
            energy = 0.018  # Below interrupt threshold
            vad = 0.4
        else:
            energy = 0.022  # Above interrupt threshold
            vad = 0.7
        
        acoustic = AcousticFeatures(
            timestamp_ms=i * 30,
            frame_energy=energy,
            vad_prob=vad,
            silence_duration_ms=0
        )
        linguistic = LinguisticFeatures(
            timestamp_ms=i * 30,
            partial_text="test",
            asr_confidence=0.7,
            token_stability=0.6,
            text_length=4
        )
        
        decision = controller.update(acoustic, linguistic)
        
        # Should stay stable due to hysteresis
        if i < 8:  # Before hysteresis window
            assert decision.state == TurnState.HOLD

### 2. Testing Acoustic Feature Extractor

def test_energy_calculation():
    """Test energy calculation from audio frame"""
    from easyturn import AcousticExtractor
    
    extractor = AcousticExtractor(sample_rate=16000, frame_duration_ms=30)
    
    # Silence
    silence_frame = np.zeros(480, dtype=np.float32)
    features = extractor.extract(silence_frame, 0)
    assert features['frame_energy'] < 0.01
    
    # Speech
    speech_frame = np.random.randn(480).astype(np.float32) * 0.1
    features = extractor.extract(speech_frame, 30)
    assert features['frame_energy'] > 0.01

def test_silence_tracking():
    """Test silence duration tracking"""
    from easyturn import AcousticExtractor
    
    extractor = AcousticExtractor()
    
    # Process speech frames
    for i in range(10):
        frame = np.random.randn(480).astype(np.float32) * 0.05
        features = extractor.extract(frame, i * 30)
        assert features['silence_duration_ms'] == 0
    
    # Process silence frames
    for i in range(10, 20):
        frame = np.zeros(480, dtype=np.float32)
        features = extractor.extract(frame, i * 30)
    
    # Silence duration should accumulate
    assert features['silence_duration_ms'] >= 270  # ~9 frames * 30ms

### 3. Testing Token Stability Tracker

def test_stability_calculation():
    """Test token stability tracking"""
    from easyturn import TokenStabilityTracker
    
    tracker = TokenStabilityTracker()
    
    # Evolving text (low stability)
    texts = ["Hel", "Hello", "Hello ho", "Hello how", "Hello how are"]
    for i, text in enumerate(texts):
        stability = tracker.update(text, 0.7, i * 100)
    
    # Should have low stability (text changing)
    assert stability < 0.7
    
    # Stable text (high stability)
    for i in range(10):
        stability = tracker.update("Hello how are you", 0.9, 500 + i * 100)
    
    # Should have high stability (text not changing)
    assert stability > 0.8

## Integration Tests

### 1. Test Complete Pipeline

def test_end_to_end_conversation():
    """Test complete conversation flow"""
    import asyncio
    from easyturn import DialogueOrchestrator, SystemConfig
    from easyturn.integrations import BackendAdapter
    
    async def run_test():
        # Setup
        config = SystemConfig(
            frame_duration_ms=30,
            min_silence_to_speak_ms=400
        )
        
        orchestrator = DialogueOrchestrator(
            config=config,
            asr_client=BackendAdapter.create_asr_wrapper(),
            llm_client=BackendAdapter.create_llm_wrapper(),
            tts_client=BackendAdapter.create_tts_wrapper()
        )
        
        await orchestrator.start()
        
        # Simulate user speaking
        for i in range(50):
            frame = np.random.randn(480).astype(np.float32) * 0.05
            await orchestrator.process_audio_frame(frame, i * 30)
        
        # Simulate silence
        for i in range(50, 70):
            frame = np.zeros(480, dtype=np.float32)
            await orchestrator.process_audio_frame(frame, i * 30)
        
        # Check state
        metrics = orchestrator.get_metrics()
        assert metrics['system_state'] in ['processing', 'speaking']
        
        await orchestrator.stop()
    
    asyncio.run(run_test())

### 2. Test Interruption Latency

def test_interruption_latency():
    """Test that interruption happens within 50ms"""
    import asyncio
    import time
    from easyturn import DialogueOrchestrator, SystemConfig
    from easyturn.integrations import BackendAdapter
    
    async def run_test():
        config = SystemConfig(interruption_latency_ms=50)
        orchestrator = DialogueOrchestrator(
            config=config,
            asr_client=BackendAdapter.create_asr_wrapper(),
            llm_client=BackendAdapter.create_llm_wrapper(),
            tts_client=BackendAdapter.create_tts_wrapper()
        )
        
        await orchestrator.start()
        
        # Set system to SPEAKING state
        orchestrator.system_state = 'speaking'
        orchestrator.current_turn_state = TurnState.SPEAK
        
        # Simulate interruption
        start_time = time.time()
        
        # Send interrupt signal
        interrupt_frame = np.random.randn(480).astype(np.float32) * 0.05
        await orchestrator.process_audio_frame(interrupt_frame, 0)
        
        # Process for several frames
        for i in range(10):
            await orchestrator.process_audio_frame(interrupt_frame, i * 30)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should handle within latency budget
        assert elapsed_ms < 100  # Allow some margin for test overhead
        
        await orchestrator.stop()
    
    asyncio.run(run_test())

## Production Validation Tests

### 1. Noise Robustness Test

```python
def test_noise_robustness():
    """Test system handles background noise"""
    controller = EasyTurnController()
    
    # Simulate speech with background noise
    for i in range(100):
        # Speech + noise
        signal = 0.05  # Speech amplitude
        noise = np.random.randn() * 0.01  # Background noise
        
        acoustic = AcousticFeatures(
            timestamp_ms=i * 30,
            frame_energy=abs(signal + noise),
            vad_prob=0.8,
            silence_duration_ms=0
        )
        linguistic = LinguisticFeatures(
            timestamp_ms=i * 30,
            partial_text="test speech",
            asr_confidence=0.75,
            token_stability=0.7,
            text_length=11
        )
        
        decision = controller.update(acoustic, linguistic)
        
        # Should maintain HOLD during speech
        assert decision.state == TurnState.HOLD
```

### 2. Edge Case Tests

```python
def test_very_short_utterance():
    """Test handling of very short utterances like 'yes'"""
    controller = EasyTurnController()
    
    # User says "yes" (very short)
    for i in range(10):  # 300ms speech
        acoustic = AcousticFeatures(
            timestamp_ms=i * 30,
            frame_energy=0.04,
            vad_prob=0.85,
            silence_duration_ms=0
        )
        linguistic = LinguisticFeatures(
            timestamp_ms=i * 30,
            partial_text="yes",
            asr_confidence=0.9,
            token_stability=0.85,
            text_length=3
        )
        controller.update(acoustic, linguistic)
    
    # Silence - should require longer wait for short utterance
    for i in range(10, 40):  # 900ms silence
        acoustic = AcousticFeatures(
            timestamp_ms=i * 30,
            frame_energy=0.001,
            vad_prob=0.1,
            silence_duration_ms=(i - 10) * 30
        )
        linguistic = LinguisticFeatures(
            timestamp_ms=i * 30,
            partial_text="yes",
            asr_confidence=0.9,
            token_stability=0.95,
            text_length=3
        )
        decision = controller.update(acoustic, linguistic)
    
    # Should eventually transition to SPEAK
    assert decision.state == TurnState.SPEAK

def test_mid_sentence_pause():
    """Test handling of pause mid-sentence"""
    controller = EasyTurnController()
    
    # User speaks part of sentence
    for i in range(20):
        acoustic = AcousticFeatures(
            timestamp_ms=i * 30,
            frame_energy=0.04,
            vad_prob=0.85,
            silence_duration_ms=0
        )
        linguistic = LinguisticFeatures(
            timestamp_ms=i * 30,
            partial_text="I was thinking",
            asr_confidence=0.75,
            token_stability=0.6,
            text_length=14
        )
        controller.update(acoustic, linguistic)
    
    # Brief pause (300ms - not enough to respond)
    for i in range(20, 30):
        acoustic = AcousticFeatures(
            timestamp_ms=i * 30,
            frame_energy=0.001,
            vad_prob=0.1,
            silence_duration_ms=(i - 20) * 30
        )
        linguistic = LinguisticFeatures(
            timestamp_ms=i * 30,
            partial_text="I was thinking",
            asr_confidence=0.75,
            token_stability=0.7,
            text_length=14
        )
        decision = controller.update(acoustic, linguistic)
    
    # Should stay in HOLD (not enough silence)
    assert decision.state == TurnState.HOLD
```

## Performance Benchmarks

```python
import time

def benchmark_controller_latency():
    """Measure EasyTurn controller latency"""
    controller = EasyTurnController()
    
    latencies = []
    for i in range(1000):
        acoustic = AcousticFeatures(
            timestamp_ms=i * 30,
            frame_energy=0.02,
            vad_prob=0.5,
            silence_duration_ms=100
        )
        linguistic = LinguisticFeatures(
            timestamp_ms=i * 30,
            partial_text="test",
            asr_confidence=0.7,
            token_stability=0.8,
            text_length=4
        )
        
        start = time.perf_counter()
        decision = controller.update(acoustic, linguistic)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
    
    print(f"Controller latency:")
    print(f"  Average: {avg_latency:.3f}ms")
    print(f"  Max: {max_latency:.3f}ms")
    print(f"  P95: {p95_latency:.3f}ms")
    
    # Should be well under frame period (30ms)
    assert avg_latency < 5.0
    assert p95_latency < 10.0
```

## Manual Testing Checklist

### Basic Functionality
- [ ] System starts without errors
- [ ] User can speak and system listens
- [ ] System responds after user stops
- [ ] System stops when user interrupts
- [ ] Multiple turns work correctly

### Edge Cases
- [ ] Very short utterances ("yes", "no")
- [ ] Very long utterances (>30 seconds)
- [ ] Mid-sentence pauses
- [ ] Rapid back-and-forth dialogue
- [ ] User speaks while thinking ("um", "uh")

### Acoustic Conditions
- [ ] Quiet environment
- [ ] Noisy environment (background music)
- [ ] Multiple speakers
- [ ] Different microphones
- [ ] Different distances from mic

### Interruption Tests
- [ ] Interrupt at start of system speech
- [ ] Interrupt in middle of system speech
- [ ] Interrupt at end of system speech
- [ ] Multiple rapid interruptions
- [ ] False start (user starts then stops immediately)

### Latency Verification
- [ ] End-to-end latency <300ms (aggressive config)
- [ ] Interruption latency <50ms
- [ ] No noticeable lag or delay
- [ ] System feels natural and responsive

### Failure Modes
- [ ] ASR fails → graceful degradation
- [ ] LLM times out → error handling
- [ ] TTS fails → error recovery
- [ ] Network issues → retry logic

## Continuous Integration

```yaml
# Example GitHub Actions workflow
name: EasyTurn Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=easyturn --cov-report=html
    
    - name: Check test coverage
      run: |
        pytest --cov=easyturn --cov-fail-under=80
    
    - name: Run benchmarks
      run: |
        python tests/benchmark.py
```

## Production Monitoring

```python
# Production metrics to track

metrics_to_monitor = {
    "turn_latency_ms": "Time from user stops to system starts",
    "interrupt_latency_ms": "Time from user speaks to TTS stops",
    "false_interrupt_rate": "% of turns with false interruption",
    "premature_response_rate": "% of turns where system spoke too early",
    "average_confidence": "Average ASR confidence when responding",
    "average_stability": "Average token stability when responding",
    "state_flap_count": "Number of rapid state changes per minute",
}

# Alert thresholds
ALERT_THRESHOLDS = {
    "turn_latency_ms": 500,  # Alert if >500ms
    "interrupt_latency_ms": 75,  # Alert if >75ms
    "false_interrupt_rate": 0.05,  # Alert if >5%
    "premature_response_rate": 0.10,  # Alert if >10%
}
```

Run tests with:
```bash
pytest tests/ -v
pytest tests/test_controller.py -v -k "interruption"
```
