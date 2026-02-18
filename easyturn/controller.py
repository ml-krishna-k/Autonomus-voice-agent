"""
EasyTurn Controller: Causal, low-latency turn-taking decision module.

This controller runs every 20-40ms and uses both acoustic and linguistic signals
to decide whether the system should SPEAK or HOLD.

Key features:
- Hysteresis with 200ms stability requirement
- Asymmetric thresholds for state transitions
- No blocking operations
- Independent of LLM
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from collections import deque


class TurnState(Enum):
    """Turn-taking states"""
    HOLD = "HOLD"    # System is listening, waiting for user to finish
    SPEAK = "SPEAK"  # System should respond


@dataclass
class AcousticFeatures:
    """Acoustic features extracted from audio frame"""
    timestamp_ms: float
    frame_energy: float  # RMS energy of current frame
    vad_prob: float      # Soft VAD probability [0, 1]
    silence_duration_ms: int  # Consecutive silence duration
    pitch: Optional[float] = None  # Optional pitch feature


@dataclass
class LinguisticFeatures:
    """Linguistic features from streaming ASR"""
    timestamp_ms: float
    partial_text: str
    asr_confidence: float  # Confidence or inverse entropy
    token_stability: float  # Stability score (how much text changed recently)
    text_length: int
    

class TurnCategory(Enum):
    """Semantic categorization of the turn state"""
    WAIT = "WAIT"             # User is speaking or pausing briefly
    INCOMPLETE = "INCOMPLETE" # User paused but thought is incomplete
    COMPLETE = "COMPLETE"     # User finished a valid turn
    BACKCHANNEL = "BACKCHANNEL" # User gave a short acknowledgement (e.g. "uh-huh")

@dataclass
class TurnDecision:
    """Turn-taking decision output"""
    state: TurnState
    category: TurnCategory
    confidence: float
    timestamp_ms: float
    reason: str  # For debugging/logging


class EasyTurnController:
    """
    Core turn-taking controller with hysteresis.
    
    Decision logic:
    - User is speaking → HOLD
    - User stopped speaking + semantic completion → SPEAK
    - User resumes speaking while SPEAK → interrupt and return to HOLD
    """
    
    # Acoustic thresholds
    ENERGY_THRESHOLD_HIGH = 0.02  # Energy above this = user speaking
    ENERGY_THRESHOLD_LOW = 0.005  # Energy below this = silence
    VAD_THRESHOLD_SPEAKING = 0.7  # VAD prob above this = user speaking
    VAD_THRESHOLD_SILENCE = 0.3   # VAD prob below this = silence
    
    # Linguistic thresholds
    MIN_CONFIDENCE_TO_SPEAK = 0.4  # Lowered to accept shorter/less confident inputs
    MIN_TEXT_LENGTH = 2  # Reduced to accept "Hi", "No", "Ok"
    STABILITY_THRESHOLD = 0.6  # Relaxed stability requirement
    
    # Timing thresholds (asymmetric)
    MIN_SILENCE_TO_SPEAK_MS = 400  # Minimum silence before considering SPEAK
    MIN_SILENCE_SHORT_UTTERANCE_MS = 500  # Longer wait for short utterances
    HYSTERESIS_WINDOW_MS = 100  # Faster commitment
    
    # Interruption thresholds (faster reaction)
    INTERRUPT_ENERGY_THRESHOLD = 0.015
    INTERRUPT_VAD_THRESHOLD = 0.6
    
    def __init__(self, frame_rate_ms: int = 30):
        """
        Initialize controller.
        
        Args:
            frame_rate_ms: How often the controller runs (20-40ms recommended)
        """
        self.frame_rate_ms = frame_rate_ms
        self.current_state = TurnState.HOLD
        self.last_decision_time_ms = 0
        
        # Hysteresis tracking
        self.pending_state: Optional[TurnState] = None
        self.pending_state_start_ms: Optional[float] = None
        
        # History buffers for temporal analysis
        max_history = int(1000 / frame_rate_ms)  # Keep 1 second of history
        self.energy_history = deque(maxlen=max_history)
        self.vad_history = deque(maxlen=max_history)
        self.stability_history = deque(maxlen=max_history)
        
        # State tracking
        self.last_speech_end_ms: Optional[float] = None
        self.last_text_change_ms: Optional[float] = None
        self.previous_text = ""
        
    def update(
        self,
        acoustic: AcousticFeatures,
        linguistic: LinguisticFeatures
    ) -> TurnDecision:
        """
        Main update loop - call this every 20-40ms.
        
        Args:
            acoustic: Current acoustic features
            linguistic: Current linguistic features from ASR
            
        Returns:
            TurnDecision with state and reasoning
        """
        current_time_ms = acoustic.timestamp_ms
        
        # Update history buffers
        self.energy_history.append(acoustic.frame_energy)
        self.vad_history.append(acoustic.vad_prob)
        self.stability_history.append(linguistic.token_stability)
        
        # Track text changes
        if linguistic.partial_text != self.previous_text:
            self.last_text_change_ms = current_time_ms
            self.previous_text = linguistic.partial_text
        
        # Determine if user is currently speaking
        is_speaking = self._is_user_speaking(acoustic)
        
        # Determine desired state
        if self.current_state == TurnState.HOLD:
            desired_state = self._evaluate_hold_state(
                acoustic, linguistic, is_speaking, current_time_ms
            )
        else:  # SPEAK state
            desired_state = self._evaluate_speak_state(
                acoustic, linguistic, is_speaking, current_time_ms
            )
        
        # Apply hysteresis
        final_state, reason = self._apply_hysteresis(
            desired_state, current_time_ms
        )
        
        # Update state
        self.current_state = final_state
        self.last_decision_time_ms = current_time_ms
        
        # Track when speech ends
        if not is_speaking and self.last_speech_end_ms is None:
            self.last_speech_end_ms = current_time_ms
        elif is_speaking:
            self.last_speech_end_ms = None
        
        confidence = self._calculate_confidence(acoustic, linguistic)
        category = self._determine_category(final_state, acoustic, linguistic, is_speaking)
        
        return TurnDecision(
            state=final_state,
            category=category,
            confidence=confidence,
            timestamp_ms=current_time_ms,
            reason=reason
        )

    def _determine_category(self, 
                            state: TurnState, 
                            acoustic: AcousticFeatures, 
                            linguistic: LinguisticFeatures, 
                            is_speaking: bool) -> TurnCategory:
        """Determine the semantic category of the turn"""
        if state == TurnState.SPEAK:
            return TurnCategory.COMPLETE
            
        # STATE is HOLD
        if is_speaking:
            return TurnCategory.WAIT
            
        # User is silent
        if linguistic.text_length > 0 and linguistic.text_length < 5: # Arbitrary short length
             # Check for backchannel-like words (mock logic, better with NLU or list)
             # "ok", "yeah", "uh-huh", "right"
             backchannels = {"ok", "okay", "yeah", "yep", "uh-huh", "hm", "hmm", "right", "sure"}
             words = linguistic.partial_text.lower().strip().split()
             if words and words[0] in backchannels:
                 return TurnCategory.BACKCHANNEL
        
        # If we have text but decided to HOLD, it's likely incomplete or waiting for more context
        if linguistic.text_length >= self.MIN_TEXT_LENGTH:
             return TurnCategory.INCOMPLETE
             
        return TurnCategory.WAIT
    
    def _is_user_speaking(self, acoustic: AcousticFeatures) -> bool:
        """
        Determine if user is currently speaking based on acoustic features.
        
        Uses both energy and VAD with hysteresis to avoid flapping.
        """
        energy_speaking = acoustic.frame_energy > self.ENERGY_THRESHOLD_HIGH
        vad_speaking = acoustic.vad_prob > self.VAD_THRESHOLD_SPEAKING
        
        # Either strong energy OR strong VAD indicates speech
        return energy_speaking or vad_speaking
    
    def _evaluate_hold_state(
        self,
        acoustic: AcousticFeatures,
        linguistic: LinguisticFeatures,
        is_speaking: bool,
        current_time_ms: float
    ) -> TurnState:
        """
        Evaluate whether to transition from HOLD to SPEAK.
        
        Conditions for SPEAK:
        1. User has stopped speaking (sufficient silence)
        2. We have stable, confident ASR output
        3. Text appears semantically complete (via stability)
        """
        # If user is still speaking, definitely HOLD
        if is_speaking:
            return TurnState.HOLD
        
        # Check silence duration
        silence_ms = acoustic.silence_duration_ms
        
        # Require minimum silence
        min_silence_required = self.MIN_SILENCE_TO_SPEAK_MS
        if linguistic.text_length < 20:  # Short utterance
            min_silence_required = self.MIN_SILENCE_SHORT_UTTERANCE_MS
        
        if silence_ms < min_silence_required:
            return TurnState.HOLD
        
        # Check linguistic completeness
        if linguistic.text_length < self.MIN_TEXT_LENGTH:
            return TurnState.HOLD
        
        if linguistic.asr_confidence < self.MIN_CONFIDENCE_TO_SPEAK:
            return TurnState.HOLD
        
        # Check token stability - text should have stabilized
        if linguistic.token_stability < self.STABILITY_THRESHOLD:
            return TurnState.HOLD
        
        # Check if text recently changed (still evolving)
        if self.last_text_change_ms is not None:
            time_since_change = current_time_ms - self.last_text_change_ms
            if time_since_change < 150:  # Text still evolving
                return TurnState.HOLD
        
        # All conditions met - suggest SPEAK
        return TurnState.SPEAK
    
    def _evaluate_speak_state(
        self,
        acoustic: AcousticFeatures,
        linguistic: LinguisticFeatures,
        is_speaking: bool,
        current_time_ms: float
    ) -> TurnState:
        """
        Evaluate whether to stay in SPEAK or return to HOLD (interruption).
        
        Use more sensitive thresholds for interruption detection.
        """
        # Fast interruption detection - lower thresholds
        interrupt_energy = acoustic.frame_energy > self.INTERRUPT_ENERGY_THRESHOLD
        interrupt_vad = acoustic.vad_prob > self.INTERRUPT_VAD_THRESHOLD
        
        if interrupt_energy or interrupt_vad:
            # User is interrupting - return to HOLD immediately
            return TurnState.HOLD
        
        # Continue speaking
        return TurnState.SPEAK
    
    def _apply_hysteresis(
        self,
        desired_state: TurnState,
        current_time_ms: float
    ) -> tuple[TurnState, str]:
        """
        Apply hysteresis to prevent rapid state flapping.
        
        State change requires stability for HYSTERESIS_WINDOW_MS.
        """
        if desired_state == self.current_state:
            # No change desired, reset pending state
            self.pending_state = None
            self.pending_state_start_ms = None
            return self.current_state, f"stable_{self.current_state.value}"
        
        # Desired state differs from current
        
        # SPECIAL CASE: Interruption (SPEAK -> HOLD)
        # Should be immediate for barge-in responsiveness
        if self.current_state == TurnState.SPEAK and desired_state == TurnState.HOLD:
             self.pending_state = None
             self.pending_state_start_ms = None
             return TurnState.HOLD, "immediate_interruption"

        if self.pending_state != desired_state:
            # New pending state - start tracking
            self.pending_state = desired_state
            self.pending_state_start_ms = current_time_ms
            return self.current_state, f"pending_{desired_state.value}"
        
        # Pending state has been consistent
        elapsed_ms = current_time_ms - self.pending_state_start_ms
        
        if elapsed_ms >= self.HYSTERESIS_WINDOW_MS:
            # Transition confirmed
            self.pending_state = None
            self.pending_state_start_ms = None
            
            reason = f"transition_to_{desired_state.value}"
            if desired_state == TurnState.HOLD:
                reason += "_interrupt"
            else:
                reason += "_eot"  # end of turn
            
            return desired_state, reason
        
        # Still waiting for stability
        return self.current_state, f"pending_{desired_state.value}_{int(elapsed_ms)}ms"
    
    def _calculate_confidence(
        self,
        acoustic: AcousticFeatures,
        linguistic: LinguisticFeatures
    ) -> float:
        """Calculate confidence in current decision (for debugging/logging)"""
        # Combine multiple signals
        vad_confidence = acoustic.vad_prob if self.current_state == TurnState.HOLD else (1 - acoustic.vad_prob)
        asr_confidence = linguistic.asr_confidence
        stability_confidence = linguistic.token_stability
        
        # Weighted average
        confidence = (
            0.4 * vad_confidence +
            0.3 * asr_confidence +
            0.3 * stability_confidence
        )
        
        return min(1.0, max(0.0, confidence))
    
    def reset(self):
        """Reset controller state (e.g., after conversation ends)"""
        self.current_state = TurnState.HOLD
        self.pending_state = None
        self.pending_state_start_ms = None
        self.last_speech_end_ms = None
        self.last_text_change_ms = None
        self.previous_text = ""
        self.energy_history.clear()
        self.vad_history.clear()
        self.stability_history.clear()
