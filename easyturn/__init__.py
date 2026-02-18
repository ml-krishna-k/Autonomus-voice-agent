"""EasyTurn: Full-Duplex Spoken Dialogue System with Explicit Turn-Taking Control

A production-ready implementation of a causal, low-latency turn-taking controller
for full-duplex spoken dialogue systems.
"""

__version__ = "1.0.0"
__author__ = "Autonomous Voice Agent Team"

from .controller import (
    EasyTurnController,
    TurnState,
    TurnDecision,
    AcousticFeatures,
    LinguisticFeatures
)

from .dialogue_orchestrator import (
    DialogueOrchestrator,
    SystemState,
    SystemConfig
)

from .acoustic_extractor import (
    AcousticExtractor,
    BatchAcousticExtractor
)

from .stability_tracker import TokenStabilityTracker

from .integrations import (
    StreamingASRWrapper,
    CancellableLLMWrapper,
    InterruptibleTTSWrapper,
    BackendAdapter
)

from .config import (
    ThresholdConfig,
    PresetConfigs,
    FAILURE_MODES,
    get_tuning_guide
)

__all__ = [
    # Core components
    'EasyTurnController',
    'DialogueOrchestrator',
    'AcousticExtractor',
    'TokenStabilityTracker',
    
    # Enums and dataclasses
    'TurnState',
    'SystemState',
    'TurnDecision',
    'AcousticFeatures',
    'LinguisticFeatures',
    'SystemConfig',
    
    # Integration wrappers
    'StreamingASRWrapper',
    'CancellableLLMWrapper',
    'InterruptibleTTSWrapper',
    'BackendAdapter',
    
    # Configuration
    'ThresholdConfig',
    'PresetConfigs',
    'FAILURE_MODES',
    'get_tuning_guide',
]
