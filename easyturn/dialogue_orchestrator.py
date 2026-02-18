"""
Full-Duplex Dialogue System Orchestrator

Coordinates ASR, EasyTurn Controller, LLM, and TTS in a full-duplex pipeline.
This is the main integration point for the entire system.
"""

import asyncio
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

from .controller import EasyTurnController, TurnState, TurnCategory, AcousticFeatures, LinguisticFeatures
from .acoustic_extractor import AcousticExtractor
from .stability_tracker import TokenStabilityTracker


logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Overall system state"""
    LISTENING = "listening"      # Actively listening to user
    PROCESSING = "processing"    # LLM is generating response
    SPEAKING = "speaking"        # System is speaking (TTS active)
    INTERRUPTED = "interrupted"  # System was interrupted mid-speech
    IDLE = "idle"               # No active conversation


@dataclass
class SystemConfig:
    """Configuration for the dialogue system"""
    # Audio settings
    sample_rate: int = 16000
    frame_duration_ms: int = 30
    
    # EasyTurn settings
    min_silence_to_speak_ms: int = 400
    hysteresis_window_ms: int = 200
    interruption_latency_ms: int = 50
    
    # ASR settings
    asr_partial_update_ms: int = 100  # How often to get ASR partials
    
    # LLM settings
    llm_timeout_s: float = 30.0
    
    # Logging
    log_decisions: bool = True
    log_detailed_features: bool = False


class DialogueOrchestrator:
    """
    Main orchestrator for full-duplex dialogue.
    
    Responsibilities:
    - Coordinate all components
    - Manage system state transitions
    - Handle interruptions
    - Buffer ASR tokens
    - Gate LLM generation
    - Control TTS playback
    """
    
    def __init__(
        self,
        config: SystemConfig,
        asr_client: Any,  # Your streaming ASR client
        llm_client: Any,  # Your LLM client
        tts_client: Any,  # Your TTS client
    ):
        """
        Initialize orchestrator.
        
        Args:
            config: System configuration
            asr_client: Streaming ASR client with these methods:
                - start_stream()
                - get_partial_result() -> (text, confidence)
                - stop_stream()
            llm_client: LLM client with these methods:
                - generate_streaming(text) -> async generator of tokens
                - cancel_generation()
            tts_client: TTS client with these methods:
                - speak_streaming(text_stream) -> plays audio
                - stop() -> stops within 50ms
                - is_playing() -> bool
        """
        self.config = config
        self.asr_client = asr_client
        self.llm_client = llm_client
        self.tts_client = tts_client
        
        # Core components
        self.acoustic_extractor = AcousticExtractor(
            sample_rate=config.sample_rate,
            frame_duration_ms=config.frame_duration_ms
        )
        self.stability_tracker = TokenStabilityTracker()
        self.easyturn_controller = EasyTurnController(
            frame_rate_ms=config.frame_duration_ms
        )
        
        # State management
        self.system_state = SystemState.IDLE
        self.current_turn_state = TurnState.HOLD
        
        # Buffers
        self.asr_buffer = []  # Buffered ASR tokens
        self.current_partial_text = ""
        self.current_asr_confidence = 0.0
        
        # Control flags
        self.is_running = False
        self.llm_generation_task: Optional[asyncio.Task] = None
        self.tts_playback_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.last_audio_timestamp_ms = 0
        self.conversation_start_time = None
        self.turn_count = 0
        
    async def start(self):
        """Start the dialogue system"""
        if self.is_running:
            logger.warning("System already running")
            return
        
        logger.info("Starting full-duplex dialogue system")
        self.is_running = True
        self.conversation_start_time = time.time()
        self.system_state = SystemState.LISTENING
        
        # Start ASR stream
        await self.asr_client.start_stream()
        
        # Start main processing loops
        asyncio.create_task(self._audio_processing_loop())
        asyncio.create_task(self._asr_polling_loop())
        
        logger.info("System started successfully")
    
    async def stop(self):
        """Stop the dialogue system"""
        if not self.is_running:
            return
        
        logger.info("Stopping dialogue system")
        self.is_running = False
        
        # Cancel ongoing operations
        await self._cancel_llm_generation()
        await self._stop_tts()
        
        # Stop ASR
        await self.asr_client.stop_stream()
        
        # Reset components
        self.easyturn_controller.reset()
        self.stability_tracker.reset()
        self.acoustic_extractor.reset()
        
        self.system_state = SystemState.IDLE
        logger.info("System stopped")
    
    async def process_audio_frame(self, audio_frame: 'np.ndarray', timestamp_ms: float):
        """
        Process incoming audio frame.
        
        This is called by your audio input handler.
        
        Args:
            audio_frame: Audio samples (numpy array, float32, [-1, 1])
            timestamp_ms: Timestamp of this frame
        """
        if not self.is_running:
            return
        
        self.last_audio_timestamp_ms = timestamp_ms
        
        # Extract acoustic features
        acoustic_features_dict = self.acoustic_extractor.extract(audio_frame, timestamp_ms)
        
        # Convert to dataclass
        acoustic_features = AcousticFeatures(
            timestamp_ms=acoustic_features_dict['timestamp_ms'],
            frame_energy=acoustic_features_dict['frame_energy'],
            vad_prob=acoustic_features_dict['vad_prob'],
            silence_duration_ms=acoustic_features_dict['silence_duration_ms'],
            pitch=acoustic_features_dict.get('pitch')
        )
        
        # Get linguistic features (from latest ASR)
        linguistic_features = self._get_linguistic_features(timestamp_ms)
        
        # Run EasyTurn controller
        decision = self.easyturn_controller.update(acoustic_features, linguistic_features)
        
        if self.config.log_decisions:
            logger.debug(
                f"Turn decision: {decision.state.value} "
                f"(confidence={decision.confidence:.2f}, reason={decision.reason})"
            )
        
        # Handle state transitions
        await self._handle_turn_decision(decision)
    
    async def _asr_polling_loop(self):
        """
        Poll ASR for partial results.
        
        Runs independently at ~100ms intervals.
        """
        while self.is_running:
            try:
                # Get partial ASR result
                partial_text, confidence = await self.asr_client.get_partial_result()
                
                if partial_text:
                    self.current_partial_text = partial_text
                    self.current_asr_confidence = confidence
                    
                    # Update stability tracker
                    timestamp_ms = self.last_audio_timestamp_ms
                    stability = self.stability_tracker.update(
                        partial_text, confidence, timestamp_ms
                    )
                    
                    # Buffer ASR token
                    self.asr_buffer.append({
                        'text': partial_text,
                        'confidence': confidence,
                        'stability': stability,
                        'timestamp_ms': timestamp_ms
                    })
                
                await asyncio.sleep(self.config.asr_partial_update_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Error in ASR polling loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _audio_processing_loop(self):
        """
        Main audio processing loop.
        
        Note: In production, this would be triggered by your audio input callback.
        This is a placeholder for the pattern.
        """
        logger.info("Audio processing loop started (waiting for audio frames)")
        # In practice, process_audio_frame() is called by audio callback
        # This loop is just for monitoring
        while self.is_running:
            await asyncio.sleep(1.0)
    
    def _get_linguistic_features(self, timestamp_ms: float) -> LinguisticFeatures:
        """Build linguistic features from current ASR state"""
        # Calculate token stability
        stability = self.stability_tracker._calculate_stability(timestamp_ms)
        
        return LinguisticFeatures(
            timestamp_ms=timestamp_ms,
            partial_text=self.current_partial_text,
            asr_confidence=self.current_asr_confidence,
            token_stability=stability,
            text_length=len(self.current_partial_text)
        )
    
    async def _handle_turn_decision(self, decision):
        """
        Handle turn decision from EasyTurn controller.
        
        State machine logic:
        - HOLD → SPEAK: Trigger LLM generation
        - SPEAK → HOLD: Interrupt (cancel LLM/TTS, return to listening)
        """
        # Handle Turn Categories
        category = decision.category
        
        if category == TurnCategory.BACKCHANNEL:
            logger.info("Detected backchannel - Ignoring (Stay in HOLD)")
            return
            
        if category == TurnCategory.INCOMPLETE:
            logger.debug("Detected incomplete utterance - Waiting for more context")
            return
            
        if category == TurnCategory.WAIT:
            return
            
        # Only proceed to semantic state change if COMPLETE or explicit SPEAK
        
        previous_state = self.current_turn_state
        new_state = decision.state
        
        if previous_state == new_state:
            return  # No change
        
        # State transition
        self.current_turn_state = new_state
        
        if new_state == TurnState.SPEAK:
            # Transition to SPEAK - user finished speaking
            # If system is currently SPEAKING (TTS active), it implies we previously
            # ignored a backchannel interruption (HOLD -> SPEAKING mismatch).
            # The User has now finished their backchannel (e.g. "Great" -> silence).
            # We should just clear the buffer (ignore the "Great") and continue speaking.
            if self.system_state == SystemState.SPEAKING:
                logger.debug("Backchannel finished - clearing buffer and continuing speech")
                self.asr_buffer.clear() 
                return

            logger.info(f"User turn complete ({category.value}) - initiating system response")
            await self._initiate_system_response()
            
        elif new_state == TurnState.HOLD:
            # Transition to HOLD - user started speaking (or noise)
            
            # If we are NOT speaking/processing, this is just normal listening.
            # But if we ARE speaking/processing, this is an interruption.
            if self.system_state in [SystemState.PROCESSING, SystemState.SPEAKING]:
                
                # IMMEDIATE CHECK: Do we already have text?
                # Sometimes ASR is fast.
                intent = await self._determine_interruption_intent(max_wait_ms=500)
                
                if intent == "backchannel":
                    logger.info(f"Ignoring backchannel: '{self.current_partial_text}'")
                    # We DO NOT stop TTS.
                    # We accept `self.current_turn_state` is now HOLD.
                    # When user falls silent, we will get TurnState.SPEAK event, 
                    # which will hit the block above and be ignored.
                    return
                elif intent == "stop_command":
                     logger.info(f"Stop command detected: '{self.current_partial_text}' - Aborting immediately")
                     await self._handle_interruption()
                     return
                elif intent == "substantial_speech":
                     logger.info(f"Barge-in detected: '{self.current_partial_text}' - Stopping TTS to listen")
                     await self._handle_interruption()
                     return
                else:
                     # Timeout/Unknown. 
                     # If we heard *something* but unsure, safer to stop if it's long enough?
                     # Or assume noise?
                     # If high energy but no text?
                     # User said "immediate cut off". Safer to cut off?
                     # But noise triggers this.
                     # Let's check energy/VAD confidence?
                     # For now, if we waited 500ms and got NO text, it might be noise. 
                     # Don't interrupt for noise.
                     if not self.current_partial_text:
                         logger.info("Interruption detected but no text (Noise?) - Ignoring")
                         return
                     
                     # If we have text but didn't match categories?
                     # Treat as barge-in.
                     logger.info(f"Unknown speech detected - Treating as barge-in")
                     await self._handle_interruption()

    async def _determine_interruption_intent(self, max_wait_ms: int = 300) -> str:
        """
        Check if the current speech is a backchannel, stop command, or substantial query.
        Waits up to max_wait_ms for ASR text to appear.
        """
        STOP_WORDS = {"stop", "wait", "enough", "cancel", "hold on", "shut up", "pause"}
        BACKCHANNEL_WORDS = {"great", "carry on", "super", "ok", "okay", "yes", "yeah", "nice", "uh-huh", "hmm", "sure", "cool", "wow"}
        
        start = time.time()
        
        while (time.time() - start) * 1000 < max_wait_ms:
            text = self.current_partial_text.strip().lower()
            # Clean punctuation
            text = text.replace('.', '').replace('!', '').replace('?', '').strip()
            
            if text:
                words = set(text.split())
                
                # Check for stop words
                if any(w in STOP_WORDS for w in words):
                    return "stop_command"
                
                # Check for backchannel (exact match or very short)
                if text in BACKCHANNEL_WORDS or (len(words) <= 2 and any(w in BACKCHANNEL_WORDS for w in words)):
                    # Check if it's growing into a sentence? 
                    # For now, simplistic check.
                    return "backchannel"
                
                # If we have text that is NEITHER, it's likely a question or comment
                # e.g. "What about the price?"
                if len(words) > 0:
                     return "substantial_speech"
            
            await asyncio.sleep(0.05)
            
        return "timeout_default_interrupt"
    
    async def _initiate_system_response(self):
        """
        Initiate system response (LLM + TTS).
        
        Called when EasyTurn transitions to SPEAK.
        """
        # Get buffered text
        if not self.asr_buffer:
            logger.warning("No ASR buffer to respond to")
            return
        
        # Get the final stable text
        final_text = self.current_partial_text
        if not final_text or len(final_text) < 5:
            logger.info("Text too short to respond")
            return
        
        logger.info(f"Responding to: '{final_text}'")
        
        # Clear buffer
        self.asr_buffer.clear()
        self.turn_count += 1
        
        # Change system state
        self.system_state = SystemState.PROCESSING
        
        # Start LLM generation and TTS playback
        self.llm_generation_task = asyncio.create_task(
            self._run_llm_and_tts(final_text)
        )
    
    async def _run_llm_and_tts(self, user_text: str):
        """
        Run LLM generation and stream to TTS.
        
        This is where the LLM and TTS are coordinated.
        """
        try:
            # Start TTS in streaming mode
            tts_started = False
            
            # Generate LLM response
            async for token in self.llm_client.generate_streaming(user_text):
                if not self.is_running or self.current_turn_state == TurnState.HOLD:
                    # Interrupted
                    logger.info("LLM generation interrupted")
                    break
                
                # Start TTS on first token
                if not tts_started:
                    self.system_state = SystemState.SPEAKING
                    tts_started = True
                
                # Stream token to TTS
                await self.tts_client.add_token(token)
            
            # Finish TTS
            if tts_started:
                await self.tts_client.finish()
                
                # Wait for TTS to complete
                while self.tts_client.is_playing():
                    await asyncio.sleep(0.05)
            
            # Return to listening
            if self.is_running:
                self.system_state = SystemState.LISTENING
                logger.info("System response complete - listening")
                
        except asyncio.CancelledError:
            logger.info("LLM/TTS task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in LLM/TTS pipeline: {e}")
            self.system_state = SystemState.LISTENING
    
    async def _handle_interruption(self):
        """
        Handle user interruption.
        
        Must complete in <50ms to meet latency requirement.
        """
        start_time = time.time()
        
        # Stop TTS immediately
        await self._stop_tts()
        
        # Cancel LLM generation
        await self._cancel_llm_generation()
        
        # Return to listening state
        self.system_state = SystemState.INTERRUPTED
        await asyncio.sleep(0.05)  # Brief pause
        self.system_state = SystemState.LISTENING
        
        # Clear buffers
        self.asr_buffer.clear()
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Interruption handled in {elapsed_ms:.1f}ms")
        
        if elapsed_ms > self.config.interruption_latency_ms:
            logger.warning(
                f"Interruption latency exceeded target: "
                f"{elapsed_ms:.1f}ms > {self.config.interruption_latency_ms}ms"
            )
    
    async def _stop_tts(self):
        """Stop TTS playback immediately"""
        if self.tts_client.is_playing():
            await self.tts_client.stop()
            logger.debug("TTS stopped")
    
    async def _cancel_llm_generation(self):
        """Cancel ongoing LLM generation"""
        if self.llm_generation_task and not self.llm_generation_task.done():
            self.llm_generation_task.cancel()
            try:
                await self.llm_generation_task
            except asyncio.CancelledError:
                pass
            logger.debug("LLM generation cancelled")
        
        # Also call LLM client cancel method
        await self.llm_client.cancel_generation()
    
    def get_metrics(self) -> dict:
        """Get system metrics for monitoring"""
        return {
            'system_state': self.system_state.value,
            'turn_state': self.current_turn_state.value,
            'turn_count': self.turn_count,
            'conversation_duration_s': time.time() - self.conversation_start_time if self.conversation_start_time else 0,
            'asr_buffer_size': len(self.asr_buffer),
            'current_text': self.current_partial_text,
            'current_confidence': self.current_asr_confidence,
        }
