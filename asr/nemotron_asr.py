"""
NVIDIA Nemotron-Speech-Streaming ASR Model Wrapper
Provides streaming ASR with cache-aware architecture for low-latency real-time transcription.
"""

import numpy as np
import logging
from typing import Tuple, Optional
from threading import Lock
import asyncio
from collections import deque

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("[WARNING] NeMo not installed. Install with: pip install nemo_toolkit[asr]")

logger = logging.getLogger(__name__)


class NemotronStreamingASR:
    """
    Wrapper for NVIDIA Nemotron-Speech-Streaming-En-0.6b model.
    
    Features:
    - Cache-aware streaming with configurable latency
    - Real-time partial results for EasyTurn integration
    - Thread-safe operation
    - Efficient chunk processing
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        att_context_size: Tuple[int, int] = (70, 1),  # [left, right] context in 80ms frames
        sample_rate: int = 16000,
        device: str = "cpu"
    ):
        """
        Initialize Nemotron ASR model.
        
        Args:
            model_name: HuggingFace model identifier
            att_context_size: [left_context, right_context] frames
                - [70, 0]: 80ms latency (ultra-low)
                - [70, 1]: 160ms latency (balanced) - RECOMMENDED
                - [70, 6]: 560ms latency (higher accuracy)
                - [70, 13]: 1120ms latency (batch mode)
            sample_rate: Audio sample rate (must be 16000)
            device: 'cpu' or 'cuda'
        """
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo toolkit not installed. Install with: pip install nemo_toolkit[asr]")
        
        self.model_name = model_name
        self.att_context_size = att_context_size
        self.sample_rate = sample_rate
        self.device = device
        
        # Calculate chunk size and latency
        self.chunk_size_frames = att_context_size[1] + 1  # current + right context
        self.chunk_duration_ms = self.chunk_size_frames * 80  # 80ms per frame
        self.chunk_size_samples = int((self.chunk_duration_ms / 1000.0) * sample_rate)
        
        logger.info(f"Initializing Nemotron ASR: {model_name}")
        logger.info(f"Attention context: {att_context_size} (latency: {self.chunk_duration_ms}ms)")
        
        self._lock = Lock()
        self._model = None
        self._is_loaded = False
        
        # Streaming state
        self._audio_buffer = deque()
        self._current_partial_text = ""
        self._current_confidence = 0.0
        self._total_processed_samples = 0
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the NeMo ASR model with streaming configuration."""
        try:
            with self._lock:
                logger.info("Loading Nemotron ASR model (this may take a moment)...")
                
                # Load pretrained model
                self._model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name
                )
                
                # Configure for streaming
                # Set attention context for cache-aware streaming
                if hasattr(self._model, 'change_attention_model'):
                    self._model.change_attention_model(
                        self_attention_model='rel_pos',
                        att_context_size=self.att_context_size
                    )
                
                # Move to device
                self._model = self._model.to(self.device)
                self._model.eval()
                
                self._is_loaded = True
                logger.info(f"[OK] Nemotron ASR loaded successfully on {self.device}")
                
        except Exception as e:
            logger.error(f"Failed to load Nemotron ASR: {e}")
            raise
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Tuple[str, float]:
        """
        Process an audio chunk and return partial transcription.
        
        Args:
            audio_chunk: Audio samples (numpy array, float32, normalized to [-1, 1])
        
        Returns:
            Tuple of (partial_text, confidence_score)
        """
        if not self._is_loaded:
            logger.warning("Model not loaded")
            return "", 0.0
        
        # Add to buffer
        self._audio_buffer.extend(audio_chunk)
        
        # Process if we have enough samples
        if len(self._audio_buffer) >= self.chunk_size_samples:
            # Extract chunk
            chunk_samples = []
            for _ in range(self.chunk_size_samples):
                if self._audio_buffer:
                    chunk_samples.append(self._audio_buffer.popleft())
            
            chunk_array = np.array(chunk_samples, dtype=np.float32)
            
            # Transcribe chunk
            try:
                with self._lock:
                    import torch
                    
                    # Convert to torch tensor with shape (batch, samples)
                    if chunk_array.ndim == 1:
                        chunk_tensor = torch.from_numpy(chunk_array).unsqueeze(0)  # Add batch dimension
                    else:
                        chunk_tensor = torch.from_numpy(chunk_array)
                    
                    # Ensure shape is (batch, time) - squeeze any extra dimensions
                    if chunk_tensor.ndim > 2:
                        chunk_tensor = chunk_tensor.squeeze(1)
                    
                    # Create lengths tensor (required by NeMo)
                    lengths = torch.tensor([chunk_tensor.shape[1]], dtype=torch.long)
                    
                    # Use the model's transcribe method which handles RNNT decoding properly
                    with torch.no_grad():
                        # Move tensors to device
                        chunk_tensor = chunk_tensor.to(self.device)
                        lengths = lengths.to(self.device)
                        
                        # Use model's internal transcription (handles RNNT greedy decoding)
                        if hasattr(self._model, 'transcribe_step'):
                            # For streaming models with transcribe_step
                            hypotheses = self._model.transcribe_step(
                                input_signal=chunk_tensor,
                                input_signal_length=lengths
                            )
                        else:
                            # Fallback: encode then decode
                            encoded, encoded_len = self._model.encoder(
                                audio_signal=chunk_tensor,
                                length=lengths
                            )
                            
                            # Use greedy RNNT decoding
                            if hasattr(self._model, 'decoding') and hasattr(self._model.decoding, 'rnnt_decoder_predictions_tensor'):
                                best_hyp = self._model.decoding.rnnt_decoder_predictions_tensor(
                                    encoder_output=encoded,
                                    encoded_lengths=encoded_len,
                                    return_hypotheses=False
                                )
                                hypotheses = [best_hyp[0]] if best_hyp else []
                            else:
                                hypotheses = []
                        
                        # Extract text from hypotheses
                        if hypotheses and len(hypotheses) > 0:
                            hyp = hypotheses[0]
                            if hasattr(hyp, 'text'):
                                text = hyp.text
                            elif isinstance(hyp, str):
                                text = hyp
                            else:
                                text = str(hyp)
                            
                            if text and len(text.strip()) > 0:
                                self._current_partial_text = text.strip()
                                self._current_confidence = 0.8
                        
                        self._total_processed_samples += len(chunk_samples)
                        
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                return self._current_partial_text, self._current_confidence
        
        return self._current_partial_text, self._current_confidence

    
    def transcribe(self, audio: np.ndarray) -> Tuple[list, dict]:
        """
        Transcribe complete audio (synchronous, for compatibility with existing code).
        
        Args:
            audio: Audio samples (numpy array, float32)
        
        Returns:
            Tuple of (segments, info) compatible with faster-whisper API
        """
        if not self._is_loaded:
            logger.warning("Model not loaded")
            return [], {}
        
        try:
            import torch
            
            with self._lock:
                # Ensure correct shape
                if isinstance(audio, list):
                    audio = np.array(audio, dtype=np.float32)
                
                # Convert to tensor with shape (batch, time)
                if audio.ndim == 1:
                    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                else:
                    audio_tensor = torch.from_numpy(audio)
                
                # Squeeze extra dimensions if needed
                if audio_tensor.ndim > 2:
                    audio_tensor = audio_tensor.squeeze(1)
                
                # Create lengths
                lengths = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
                
                # Use model's RNNT decoding
                with torch.no_grad():
                    # Move to device
                    audio_tensor = audio_tensor.to(self.device)
                    lengths = lengths.to(self.device)
                    
                    # Encode
                    encoded, encoded_len = self._model.encoder(
                        audio_signal=audio_tensor,
                        length=lengths
                    )
                    
                    # Use greedy RNNT decoding
                    text = ""
                    if hasattr(self._model, 'decoding') and hasattr(self._model.decoding, 'rnnt_decoder_predictions_tensor'):
                        best_hyp = self._model.decoding.rnnt_decoder_predictions_tensor(
                            encoder_output=encoded,
                            encoded_lengths=encoded_len,
                            return_hypotheses=False
                        )
                        
                        if best_hyp and len(best_hyp) > 0:
                            hyp = best_hyp[0]
                            if hasattr(hyp, 'text'):
                                text = hyp.text
                            elif isinstance(hyp, str):
                                text = hyp
                            else:
                                text = str(hyp)
                
                # Convert to faster-whisper compatible format
                segments = []
                if text and len(text.strip()) > 0:
                    # Create segment object
                    class Segment:
                        def __init__(self, text, confidence=0.8):
                            self.text = text
                            self.confidence = confidence
                            self.start = 0.0
                            self.end = audio_tensor.shape[1] / self.sample_rate
                    
                    segments.append(Segment(text.strip(), 0.8))
                
                info = {
                    'language': 'en',
                    'duration': audio_tensor.shape[1] / self.sample_rate
                }
                
                return segments, info
                
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return [], {}
    
    def get_partial_result(self) -> Tuple[str, float]:
        """
        Get the latest partial transcription result.
        
        Returns:
            Tuple of (text, confidence)
        """
        return self._current_partial_text, self._current_confidence
    
    def reset_stream(self):
        """Reset streaming state."""
        with self._lock:
            self._audio_buffer.clear()
            self._current_partial_text = ""
            self._current_confidence = 0.0
            self._total_processed_samples = 0
            
            # Reset model cache if available
            if hasattr(self._model, 'reset_state'):
                self._model.reset_state()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded


# Async wrapper for EasyTurn integration
class NemotronASRWrapper:
    """
    Async wrapper for Nemotron ASR to work with EasyTurn DialogueOrchestrator.
    """
    
    def __init__(self, nemotron_asr: NemotronStreamingASR):
        self.asr = nemotron_asr
        self.is_streaming = False
        
    async def start_stream(self):
        """Start streaming session."""
        self.asr.reset_stream()
        self.is_streaming = True
        logger.info("ASR streaming started")
    
    async def stop_stream(self):
        """Stop streaming session."""
        self.is_streaming = False
        logger.info("ASR streaming stopped")
    
    async def get_partial_result(self) -> Tuple[str, float]:
        """
        Get partial transcription result (async).
        
        Returns:
            Tuple of (text, confidence)
        """
        if not self.is_streaming:
            return "", 0.0
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        text, confidence = await loop.run_in_executor(
            None,
            self.asr.get_partial_result
        )
        return text, confidence
    
    async def process_audio_chunk(self, audio_chunk: np.ndarray) -> Tuple[str, float]:
        """
        Process audio chunk (async).
        
        Args:
            audio_chunk: Audio samples
        
        Returns:
            Tuple of (partial_text, confidence)
        """
        if not self.is_streaming:
            return "", 0.0
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.asr.process_audio_chunk,
            audio_chunk
        )


# Factory function for easy creation
def create_nemotron_asr(
    latency_mode: str = "balanced",
    device: str = "cpu"
) -> NemotronStreamingASR:
    """
    Create Nemotron ASR with preset latency configurations.
    
    Args:
        latency_mode: 'ultra_low' (80ms), 'balanced' (160ms), 'accurate' (560ms), 'batch' (1120ms)
        device: 'cpu' or 'cuda'
    
    Returns:
        Configured NemotronStreamingASR instance
    """
    latency_configs = {
        'ultra_low': (70, 0),   # 80ms
        'balanced': (70, 1),    # 160ms - RECOMMENDED
        'accurate': (70, 6),    # 560ms
        'batch': (70, 13)       # 1120ms
    }
    
    att_context = latency_configs.get(latency_mode, (70, 1))
    
    return NemotronStreamingASR(
        att_context_size=att_context,
        device=device
    )
