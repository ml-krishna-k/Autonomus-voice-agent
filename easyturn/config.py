"""
Configuration and Threshold Documentation

This module documents all thresholds, parameters, and their rationale.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ThresholdConfig:
    """
    Complete threshold configuration with documentation.
    
    All thresholds are tunable based on your specific use case.
    """
    
    # =================================================================
    # ACOUSTIC THRESHOLDS
    # =================================================================
    
    # Energy thresholds (RMS amplitude, range: [0, 1])
    ENERGY_THRESHOLD_HIGH: float = 0.02
    # Rationale: Energy above this indicates active speech
    # Tune: Increase for noisier environments, decrease for quieter
    
    ENERGY_THRESHOLD_LOW: float = 0.005
    # Rationale: Energy below this indicates silence/pause
    # Tune: Adjust based on background noise floor
    
    INTERRUPT_ENERGY_THRESHOLD: float = 0.015
    # Rationale: Lower threshold for faster interruption detection
    # Tune: Critical for <50ms interruption latency
    
    # VAD (Voice Activity Detection) thresholds (probability: [0, 1])
    VAD_THRESHOLD_SPEAKING: float = 0.7
    # Rationale: VAD confidence above this means user is speaking
    # Tune: Increase to reduce false positives, decrease for sensitivity
    
    VAD_THRESHOLD_SILENCE: float = 0.3
    # Rationale: VAD confidence below this means silence
    # Tune: Should have gap with SPEAKING threshold for hysteresis
    
    INTERRUPT_VAD_THRESHOLD: float = 0.6
    # Rationale: Lower threshold for interruption (faster reaction)
    # Tune: Balance between false interrupts and responsiveness
    
    # =================================================================
    # LINGUISTIC THRESHOLDS
    # =================================================================
    
    # ASR confidence/quality thresholds
    MIN_CONFIDENCE_TO_SPEAK: float = 0.6
    # Rationale: Minimum ASR confidence before responding
    # Tune: Higher = fewer hallucinations, lower = more responsive
    # Failure mode: Too low → respond to incorrect ASR
    
    MIN_TEXT_LENGTH: int = 5
    # Rationale: Minimum characters to consider valid input
    # Tune: Depends on language and use case
    # Failure mode: Too low → respond to noise/fragments
    
    STABILITY_THRESHOLD: float = 0.8
    # Rationale: Token stability score needed for semantic completion
    # Tune: Higher = wait for more stable ASR, lower = faster response
    # Failure mode: Too low → respond before ASR stabilizes
    
    # =================================================================
    # TIMING THRESHOLDS (milliseconds)
    # =================================================================
    
    # Silence duration thresholds
    MIN_SILENCE_TO_SPEAK_MS: int = 400
    # Rationale: Minimum silence before considering turn complete
    # Tune: Increase for more deliberate pacing, decrease for snappier
    # Failure mode: Too low → interrupt user mid-thought
    
    MIN_SILENCE_SHORT_UTTERANCE_MS: int = 600
    # Rationale: Longer wait for short utterances (avoid premature interruption)
    # Tune: Should be >= MIN_SILENCE_TO_SPEAK_MS
    # Failure mode: Too low → cut off user on short responses
    
    # Hysteresis window
    HYSTERESIS_WINDOW_MS: int = 200
    # Rationale: State must be stable this long before transitioning
    # Tune: Increase for more stability, decrease for faster transitions
    # Failure mode: Too low → state flapping; too high → sluggish
    
    # Text change detection
    TEXT_CHANGE_WINDOW_MS: int = 150
    # Rationale: Wait for ASR to stop changing before responding
    # Tune: Based on your ASR's stabilization time
    # Failure mode: Too low → respond to unstable ASR
    
    # Interruption latency
    MAX_INTERRUPTION_LATENCY_MS: int = 50
    # Rationale: Maximum time to stop TTS after user speaks
    # Tune: This is a hard requirement for good UX
    # Failure mode: Higher → poor user experience
    
    # =================================================================
    # SYSTEM PARAMETERS
    # =================================================================
    
    # Frame processing rate
    FRAME_RATE_MS: int = 30
    # Rationale: How often EasyTurn controller runs
    # Tune: 20-40ms is optimal (balance latency vs CPU)
    # Failure mode: Too high → miss fast events; too low → CPU overhead
    
    # ASR polling rate
    ASR_PARTIAL_UPDATE_MS: int = 100
    # Rationale: How often to poll ASR for partial results
    # Tune: Match your ASR's update rate
    # Failure mode: Too high → stale data; too low → unnecessary overhead
    
    # History buffer sizes
    ACOUSTIC_HISTORY_MS: int = 1000
    # Rationale: How much acoustic history to keep for analysis
    # Tune: More = better smoothing, less = lower memory
    
    STABILITY_WINDOW_MS: int = 500
    # Rationale: Window for calculating token stability
    # Tune: Based on your ASR's typical stabilization time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ThresholdConfig':
        """Load from dictionary"""
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        })


# =================================================================
# PRESET CONFIGURATIONS
# =================================================================

class PresetConfigs:
    """Pre-tuned configurations for common scenarios"""
    
    @staticmethod
    def get_default() -> ThresholdConfig:
        """Default balanced configuration"""
        return ThresholdConfig()
    
    @staticmethod
    def get_aggressive() -> ThresholdConfig:
        """Aggressive/fast configuration - prioritize speed"""
        config = ThresholdConfig()
        config.MIN_SILENCE_TO_SPEAK_MS = 300  # Faster response
        config.MIN_SILENCE_SHORT_UTTERANCE_MS = 400
        config.HYSTERESIS_WINDOW_MS = 150  # Faster transitions
        config.STABILITY_THRESHOLD = 0.7  # Lower stability requirement
        config.MIN_CONFIDENCE_TO_SPEAK = 0.5  # Accept lower confidence
        return config
    
    @staticmethod
    def get_conservative() -> ThresholdConfig:
        """Conservative/careful configuration - prioritize accuracy"""
        config = ThresholdConfig()
        config.MIN_SILENCE_TO_SPEAK_MS = 600  # Longer wait
        config.MIN_SILENCE_SHORT_UTTERANCE_MS = 800
        config.HYSTERESIS_WINDOW_MS = 300  # More stability
        config.STABILITY_THRESHOLD = 0.85  # Higher stability requirement
        config.MIN_CONFIDENCE_TO_SPEAK = 0.7  # Higher confidence required
        config.MIN_TEXT_LENGTH = 10  # Longer minimum text
        return config
    
    @staticmethod
    def get_noisy_environment() -> ThresholdConfig:
        """Configuration for noisy environments"""
        config = ThresholdConfig()
        config.ENERGY_THRESHOLD_HIGH = 0.035  # Higher energy threshold
        config.ENERGY_THRESHOLD_LOW = 0.01  # Higher noise floor
        config.VAD_THRESHOLD_SPEAKING = 0.75  # More confident VAD
        config.MIN_CONFIDENCE_TO_SPEAK = 0.7  # Higher ASR confidence
        return config
    
    @staticmethod
    def get_quiet_environment() -> ThresholdConfig:
        """Configuration for quiet environments"""
        config = ThresholdConfig()
        config.ENERGY_THRESHOLD_HIGH = 0.015  # Lower energy threshold
        config.ENERGY_THRESHOLD_LOW = 0.002  # Lower noise floor
        config.VAD_THRESHOLD_SPEAKING = 0.65  # Less strict VAD
        return config


# =================================================================
# FAILURE MODES AND MITIGATIONS
# =================================================================

FAILURE_MODES = {
    "premature_response": {
        "symptoms": "System responds before user finishes speaking",
        "causes": [
            "MIN_SILENCE_TO_SPEAK_MS too low",
            "STABILITY_THRESHOLD too low",
            "MIN_CONFIDENCE_TO_SPEAK too low"
        ],
        "mitigations": [
            "Increase MIN_SILENCE_TO_SPEAK_MS by 100-200ms",
            "Increase STABILITY_THRESHOLD to 0.85-0.9",
            "Monitor ASR confidence distribution and adjust threshold"
        ]
    },
    
    "slow_response": {
        "symptoms": "System takes too long to respond after user stops",
        "causes": [
            "MIN_SILENCE_TO_SPEAK_MS too high",
            "HYSTERESIS_WINDOW_MS too high",
            "STABILITY_THRESHOLD too high"
        ],
        "mitigations": [
            "Decrease MIN_SILENCE_TO_SPEAK_MS in 50ms increments",
            "Decrease HYSTERESIS_WINDOW_MS to 150-200ms",
            "Lower STABILITY_THRESHOLD to 0.75-0.8"
        ]
    },
    
    "false_interruptions": {
        "symptoms": "System stops speaking when user hasn't actually interrupted",
        "causes": [
            "INTERRUPT_ENERGY_THRESHOLD too low",
            "INTERRUPT_VAD_THRESHOLD too low",
            "Background noise triggering VAD"
        ],
        "mitigations": [
            "Increase INTERRUPT_ENERGY_THRESHOLD",
            "Increase INTERRUPT_VAD_THRESHOLD to 0.65-0.7",
            "Implement noise cancellation",
            "Use directional microphone"
        ]
    },
    
    "missed_interruptions": {
        "symptoms": "System continues speaking when user tries to interrupt",
        "causes": [
            "INTERRUPT_ENERGY_THRESHOLD too high",
            "INTERRUPT_VAD_THRESHOLD too high",
            "TTS volume drowning out user speech"
        ],
        "mitigations": [
            "Decrease INTERRUPT_ENERGY_THRESHOLD",
            "Decrease INTERRUPT_VAD_THRESHOLD to 0.55-0.6",
            "Implement acoustic echo cancellation (AEC)",
            "Lower TTS playback volume"
        ]
    },
    
    "state_flapping": {
        "symptoms": "System rapidly switches between HOLD and SPEAK",
        "causes": [
            "HYSTERESIS_WINDOW_MS too low",
            "Borderline speech signal (user speaking softly)",
            "Inconsistent ASR confidence"
        ],
        "mitigations": [
            "Increase HYSTERESIS_WINDOW_MS to 250-300ms",
            "Add median filtering to acoustic features",
            "Increase VAD threshold gap (SPEAKING - SILENCE)"
        ]
    },
    
    "high_latency": {
        "symptoms": "Overall system feels sluggish (>300ms perceived latency)",
        "causes": [
            "FRAME_RATE_MS too high",
            "ASR latency",
            "LLM first-token latency",
            "TTS synthesis latency"
        ],
        "mitigations": [
            "Decrease FRAME_RATE_MS to 20-25ms",
            "Profile ASR latency and optimize",
            "Use streaming LLM with low first-token latency",
            "Pre-warm TTS engine",
            "Consider using faster TTS model"
        ]
    },
    
    "responding_to_noise": {
        "symptoms": "System responds to background noise or non-speech sounds",
        "causes": [
            "MIN_CONFIDENCE_TO_SPEAK too low",
            "MIN_TEXT_LENGTH too low",
            "VAD false positives"
        ],
        "mitigations": [
            "Increase MIN_CONFIDENCE_TO_SPEAK to 0.7-0.8",
            "Increase MIN_TEXT_LENGTH to 8-10 characters",
            "Implement noise gate",
            "Use better VAD model (e.g., Silero VAD)"
        ]
    }
}


def get_tuning_guide() -> str:
    """Get a comprehensive tuning guide"""
    guide = """
    ═══════════════════════════════════════════════════════════════
    EASYTURN THRESHOLD TUNING GUIDE
    ═══════════════════════════════════════════════════════════════
    
    STEP 1: Start with default configuration
    ----------------------------------------
    Use PresetConfigs.get_default() and test in your environment.
    
    STEP 2: Optimize for your acoustic environment
    -----------------------------------------------
    • Record 30s of silence in your target environment
    • Calculate noise floor energy
    • Set ENERGY_THRESHOLD_LOW = noise_floor + 0.003
    • Set ENERGY_THRESHOLD_HIGH = noise_floor + 0.015
    
    STEP 3: Tune response timing
    -----------------------------
    • Ask users to speak naturally
    • Measure time from last word to silence
    • Set MIN_SILENCE_TO_SPEAK_MS = measured_time + 100ms buffer
    
    STEP 4: Optimize interruption sensitivity
    ------------------------------------------
    • Have system speak while user tries to interrupt
    • Measure interruption detection time
    • Adjust INTERRUPT_*_THRESHOLD until <50ms detection
    
    STEP 5: Fine-tune ASR thresholds
    ---------------------------------
    • Log ASR confidence distribution for valid vs invalid speech
    • Set MIN_CONFIDENCE_TO_SPEAK at 90th percentile of valid speech
    • Monitor token stability scores and adjust STABILITY_THRESHOLD
    
    STEP 6: Test edge cases
    ------------------------
    • Short utterances ("yes", "no", "okay")
    • Long pauses mid-sentence
    • Background speech (TV, other people)
    • User interrupting system
    • Rapid back-and-forth dialogue
    
    STEP 7: Monitor and iterate
    ----------------------------
    • Log all turn transitions with timestamps
    • Calculate metrics:
        - Average turn latency
        - False interruption rate
        - Premature response rate
        - User satisfaction score
    • Adjust thresholds based on data
    
    ═══════════════════════════════════════════════════════════════
    """
    return guide
