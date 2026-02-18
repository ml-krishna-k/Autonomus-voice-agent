# EasyTurn: Full-Duplex Spoken Dialogue System

## Overview

EasyTurn is a production-ready implementation of a causal, low-latency turn-taking controller for full-duplex spoken dialogue systems. It provides explicit control over when the system should speak or hold, using both acoustic and linguistic signals.

## Architecture

```
Audio Stream (16kHz PCM)
    │
    ├─► Acoustic Feature Extraction (every 20-40ms)
    │       • Frame energy (RMS)
    │       • Soft VAD probability
    │       • Silence duration tracking
    │       • Optional: Pitch features
    │
    ├─► Streaming ASR
    │       • Partial text buffer
    │       • Token stability tracking
    │       • Confidence/entropy scores
    │       • Text change detection
    │
    └─► EasyTurn Controller (core decision maker)
            • Runs every 20-40ms
            • Fuses acoustic + linguistic signals
            • Hysteresis: 200ms stability requirement
            • Asymmetric thresholds (HOLD↔SPEAK)
            • Output: HOLD or SPEAK
            │
            ├─► HOLD → Buffer ASR tokens, wait for completion
            │
            └─► SPEAK → Trigger LLM generation
                    │
                    └─► Stream to TTS (interruptible)
                            │
                            └─► On user resumption: stop TTS <50ms
```

## Key Components

### 1. **EasyTurn Controller** (`controller.py`)
The core turn-taking decision module.

**Inputs (per timestep):**
- `frame_energy`: RMS energy of audio frame
- `vad_prob`: Voice activity probability [0, 1]
- `silence_duration_ms`: Consecutive silence duration
- `partial_text`: Current ASR partial output
- `asr_confidence`: ASR confidence or inverse entropy
- `token_stability`: Text stability score [0, 1]

**Output:**
- `HOLD`: System should listen
- `SPEAK`: System should respond

**Features:**
- Hysteresis with 200ms stability requirement
- Asymmetric thresholds (different for entering vs leaving states)
- No blocking operations
- Runs independently of LLM

### 2. **Acoustic Feature Extractor** (`acoustic_extractor.py`)
Extracts real-time acoustic features from audio frames.

**Features extracted:**
- Frame energy (RMS with smoothing)
- Soft VAD probability (sigmoid-based)
- Silence duration tracking
- Optional pitch (adds 5-10ms latency)

### 3. **Token Stability Tracker** (`stability_tracker.py`)
Monitors ASR output stability to detect semantic completion.

**Metrics calculated:**
- Text similarity over time (edit distance)
- Confidence stability (low variance = stable)
- Length stability (no major rewrites)

### 4. **Dialogue Orchestrator** (`dialogue_orchestrator.py`)
Main coordinator for all components.

**Responsibilities:**
- Audio frame processing pipeline
- ASR partial polling
- EasyTurn decision handling
- LLM generation control
- TTS playback management
- Interruption handling (<50ms)

### 5. **Integration Wrappers** (`integrations.py`)
Adapters for your existing ASR, LLM, and TTS services.

**Interfaces:**
- `StreamingASRWrapper`: Streaming ASR with partial results
- `CancellableLLMWrapper`: LLM with immediate cancellation
- `InterruptibleTTSWrapper`: TTS with <50ms stop latency

## Installation

```bash
# Install dependencies
pip install numpy librosa

# Or add to your requirements.txt
numpy>=1.20.0
librosa>=0.9.0
```

## Quick Start

```python
import asyncio
from easyturn import (
    DialogueOrchestrator,
    SystemConfig,
    BackendAdapter
)

async def main():
    # Create service wrappers
    asr_client = BackendAdapter.create_asr_wrapper()
    llm_client = BackendAdapter.create_llm_wrapper()
    tts_client = BackendAdapter.create_tts_wrapper()
    
    # Configure system
    config = SystemConfig(
        sample_rate=16000,
        frame_duration_ms=30,
        min_silence_to_speak_ms=400,
        hysteresis_window_ms=200,
        interruption_latency_ms=50
    )
    
    # Create orchestrator
    orchestrator = DialogueOrchestrator(
        config=config,
        asr_client=asr_client,
        llm_client=llm_client,
        tts_client=tts_client
    )
    
    # Start system
    await orchestrator.start()
    
    # Process audio frames (from microphone callback)
    # await orchestrator.process_audio_frame(audio_frame, timestamp_ms)
    
    # Stop when done
    # await orchestrator.stop()

asyncio.run(main())
```

## Integration Guide

### Integrating Your ASR

```python
from easyturn.integrations import StreamingASRWrapper

class YourASRWrapper(StreamingASRWrapper):
    def __init__(self, your_asr_service):
        super().__init__(your_asr_service)
    
    async def start_stream(self):
        await self.asr_service.start_streaming()
        self.is_streaming = True
    
    async def get_partial_result(self):
        result = await self.asr_service.get_latest_partial()
        return result.text, result.confidence
```

### Integrating Your LLM

```python
from easyturn.integrations import CancellableLLMWrapper

class YourLLMWrapper(CancellableLLMWrapper):
    async def generate_streaming(self, prompt):
        async for token in self.llm_client.stream_generate(prompt):
            if not self.is_generating:
                break
            yield token
    
    async def cancel_generation(self):
        self.is_generating = False
        await self.llm_client.cancel_request()
```

### Integrating Your TTS

```python
from easyturn.integrations import InterruptibleTTSWrapper

class YourTTSWrapper(InterruptibleTTSWrapper):
    async def stop(self):
        """Must complete in <50ms"""
        self._should_stop = True
        await self.tts_client.stop_immediate()
```

## Configuration & Tuning

### Preset Configurations

```python
from easyturn.config import PresetConfigs

# Balanced (default)
config = PresetConfigs.get_default()

# Fast/responsive
config = PresetConfigs.get_aggressive()

# Careful/accurate
config = PresetConfigs.get_conservative()

# Noisy environment
config = PresetConfigs.get_noisy_environment()

# Quiet environment
config = PresetConfigs.get_quiet_environment()
```

### Custom Thresholds

```python
from easyturn.config import ThresholdConfig

config = ThresholdConfig()

# Adjust response timing
config.MIN_SILENCE_TO_SPEAK_MS = 350  # Faster (default: 400)

# Adjust stability requirement
config.STABILITY_THRESHOLD = 0.75  # Less strict (default: 0.8)

# Adjust interruption sensitivity
config.INTERRUPT_ENERGY_THRESHOLD = 0.012  # More sensitive (default: 0.015)
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| End-to-end latency | <300ms | From user stops speaking to system starts |
| Interruption latency | <50ms | From user speaks to TTS stops |
| Controller cycle time | 20-40ms | How often EasyTurn runs |
| State stability window | 200ms | Hysteresis to prevent flapping |

## Failure Modes & Mitigations

### Premature Response
**Symptom:** System responds before user finishes

**Causes:**
- `MIN_SILENCE_TO_SPEAK_MS` too low
- `STABILITY_THRESHOLD` too low

**Fix:**
```python
config.MIN_SILENCE_TO_SPEAK_MS = 500  # Increase by 100ms
config.STABILITY_THRESHOLD = 0.85     # Require more stability
```

### Slow Response
**Symptom:** System takes too long after user stops

**Causes:**
- `MIN_SILENCE_TO_SPEAK_MS` too high
- `HYSTERESIS_WINDOW_MS` too high

**Fix:**
```python
config.MIN_SILENCE_TO_SPEAK_MS = 350  # Decrease by 50ms
config.HYSTERESIS_WINDOW_MS = 150     # Faster transitions
```

### False Interruptions
**Symptom:** System stops when user hasn't interrupted

**Causes:**
- `INTERRUPT_ENERGY_THRESHOLD` too low
- Background noise

**Fix:**
```python
config.INTERRUPT_ENERGY_THRESHOLD = 0.02  # Increase threshold
config.INTERRUPT_VAD_THRESHOLD = 0.7      # More confident VAD
```

### Missed Interruptions
**Symptom:** System ignores user interruptions

**Causes:**
- `INTERRUPT_ENERGY_THRESHOLD` too high
- TTS volume drowning out user

**Fix:**
```python
config.INTERRUPT_ENERGY_THRESHOLD = 0.01  # More sensitive
# Also: Implement acoustic echo cancellation (AEC)
```

## Monitoring & Debugging

### Enable Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('easyturn')

config = SystemConfig(
    log_decisions=True,           # Log all turn decisions
    log_detailed_features=True    # Log acoustic/linguistic features
)
```

### Get Runtime Metrics

```python
metrics = orchestrator.get_metrics()
print(f"System state: {metrics['system_state']}")
print(f"Turn state: {metrics['turn_state']}")
print(f"Turn count: {metrics['turn_count']}")
print(f"ASR buffer size: {metrics['asr_buffer_size']}")
```

## Advanced Topics

### Multi-Modal Turn-Taking
Add visual cues (user looking away, gestures) by extending `LinguisticFeatures`:

```python
@dataclass
class ExtendedLinguisticFeatures(LinguisticFeatures):
    user_looking_away: bool = False
    gesture_detected: bool = False
```

### Emotion-Aware Turn-Taking
Modulate thresholds based on detected emotion:

```python
if emotion == "excited":
    config.MIN_SILENCE_TO_SPEAK_MS = 300  # Respond faster
elif emotion == "thoughtful":
    config.MIN_SILENCE_TO_SPEAK_MS = 600  # Wait longer
```

### Multi-Party Conversation
Track speaker identity and adjust per-speaker thresholds:

```python
if speaker_id == "fast_talker":
    config.MIN_SILENCE_TO_SPEAK_MS = 250
elif speaker_id == "slow_talker":
    config.MIN_SILENCE_TO_SPEAK_MS = 500
```

## Testing

Run the example simulation:

```bash
python -m easyturn.example
```

Expected output:
```
[0000.0ms] 🎤 User starts speaking...
[0300.0ms] State: HOLD  | System: listening
[1500.0ms] 🤐 User stops speaking (silence)...
[2100.0ms] State: SPEAK | System: processing
[2100.0ms] 🤖 System should be responding...
[3000.0ms] ⚡ User interrupts!
[3050.0ms] State: HOLD  | System: interrupted
```

## Production Checklist

- [ ] Tune thresholds for your acoustic environment
- [ ] Implement acoustic echo cancellation (AEC)
- [ ] Profile ASR, LLM, and TTS latencies
- [ ] Test interruption latency (<50ms requirement)
- [ ] Log all turn transitions for analysis
- [ ] Monitor false interrupt rate
- [ ] Test with diverse speakers and accents
- [ ] Test in noisy environments
- [ ] Implement graceful degradation (if ASR fails, etc.)
- [ ] Add telemetry for production monitoring

## License

This implementation is provided as-is for production use in real-time dialogue systems.

## References

- Hysteresis prevents state flapping (control theory)
- Asymmetric thresholds improve robustness (Schmitt trigger principle)
- Token stability detects semantic completion (ASR stabilization research)
- <50ms interruption latency matches human perception thresholds
