"""
Integration wrappers for ASR, LLM, and TTS

These wrappers adapt your existing clients to the interface expected by the orchestrator.
"""

import asyncio
from typing import Optional, AsyncGenerator
import numpy as np


class StreamingASRWrapper:
    """
    Wrapper for your streaming ASR client.
    
    Adapt this to your actual ASR implementation.
    """
    
    def __init__(self, asr_service):
        """
        Initialize wrapper.
        
        Args:
            asr_service: Your actual ASR service/client
        """
        self.asr_service = asr_service
        self.is_streaming = False
        self.latest_partial = ""
        self.latest_confidence = 0.0
        self._partial_lock = asyncio.Lock()
        
    async def start_stream(self):
        """Start ASR streaming"""
        if self.is_streaming:
            return
        
        # Start your ASR stream
        # Example: await self.asr_service.start_stream()
        self.is_streaming = True
        
        # Start background task to collect ASR results
        asyncio.create_task(self._collect_asr_results())
    
    async def stop_stream(self):
        """Stop ASR streaming"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        # Stop your ASR stream
        # Example: await self.asr_service.stop_stream()
    
    async def get_partial_result(self) -> tuple[str, float]:
        """
        Get latest partial ASR result.
        
        Returns:
            (partial_text, confidence_score)
        """
        async with self._partial_lock:
            return self.latest_partial, self.latest_confidence
    
    async def _collect_asr_results(self):
        """
        Background task to collect ASR results.
        
        Adapt this to your ASR API.
        """
        while self.is_streaming:
            try:
                # Example: Get result from your ASR
                # result = await self.asr_service.get_next_result()
                # partial_text = result.text
                # confidence = result.confidence
                
                # For demonstration - replace with actual ASR call
                partial_text = ""  # Replace with actual
                confidence = 0.0   # Replace with actual
                
                async with self._partial_lock:
                    self.latest_partial = partial_text
                    self.latest_confidence = confidence
                
                await asyncio.sleep(0.05)  # Adjust based on your ASR update rate
                
            except Exception as e:
                print(f"Error collecting ASR results: {e}")
                await asyncio.sleep(0.1)
    
    async def send_audio(self, audio_chunk: np.ndarray):
        """
        Send audio chunk to ASR.
        
        Args:
            audio_chunk: Audio samples
        """
        if not self.is_streaming:
            return
        
        # Send to your ASR service
        # Example: await self.asr_service.send_audio(audio_chunk)
        pass


class CancellableLLMWrapper:
    """
    Wrapper for LLM client with cancellation support.
    
    Critical requirement: LLM generation must be cancellable immediately.
    """
    
    def __init__(self, llm_client):
        """
        Initialize wrapper.
        
        Args:
            llm_client: Your LLM client (e.g., OpenAI, Anthropic, local model)
        """
        self.llm_client = llm_client
        self.current_generation_task: Optional[asyncio.Task] = None
        self.is_generating = False
        
    async def generate_streaming(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from LLM.
        
        Args:
            prompt: User input text
            
        Yields:
            Text tokens from LLM
        """
        self.is_generating = True
        
        try:
            # Example for OpenAI-style API
            # Replace with your actual LLM API call
            
            # response = await self.llm_client.chat.completions.create(
            #     model="gpt-4",
            #     messages=[{"role": "user", "content": prompt}],
            #     stream=True
            # )
            # 
            # async for chunk in response:
            #     if not self.is_generating:
            #         break
            #     
            #     if chunk.choices[0].delta.content:
            #         yield chunk.choices[0].delta.content
            
            # Placeholder - replace with actual implementation
            # Simulate streaming response
            demo_response = "This is a demo response. Replace with actual LLM."
            for word in demo_response.split():
                if not self.is_generating:
                    break
                yield word + " "
                await asyncio.sleep(0.05)  # Simulate token delay
                
        finally:
            self.is_generating = False
    
    async def cancel_generation(self):
        """Cancel ongoing generation immediately"""
        if self.is_generating:
            self.is_generating = False
            
            # If your LLM client has a cancel method, call it here
            # Example: await self.llm_client.cancel_current_request()
            
            if self.current_generation_task:
                self.current_generation_task.cancel()


class InterruptibleTTSWrapper:
    """
    Wrapper for TTS client with interruption support.
    
    Critical requirements:
    - Must support streaming text input
    - Must stop playback within 50ms
    - Must support mid-utterance interruption
    """
    
    def __init__(self, tts_client):
        """
        Initialize wrapper.
        
        Args:
            tts_client: Your TTS client/engine
        """
        self.tts_client = tts_client
        self._is_playing = False
        self._text_buffer = asyncio.Queue()
        self._playback_task: Optional[asyncio.Task] = None
        self._should_stop = False
        
    async def add_token(self, token: str):
        """
        Add text token to TTS stream.
        
        Args:
            token: Text token to synthesize
        """
        await self._text_buffer.put(token)
        
        # Start playback task if not already running
        if not self._playback_task or self._playback_task.done():
            self._should_stop = False
            self._playback_task = asyncio.create_task(self._playback_loop())
    
    async def finish(self):
        """Signal that no more tokens will be added"""
        await self._text_buffer.put(None)  # Sentinel
    
    async def stop(self):
        """
        Stop TTS playback immediately.
        
        MUST complete within 50ms.
        """
        self._should_stop = True
        self._is_playing = False
        
        # Clear buffer
        while not self._text_buffer.empty():
            try:
                self._text_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Stop your TTS engine
        # Example: self.tts_client.stop_immediate()
        
        # Cancel playback task
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
    
    def is_playing(self) -> bool:
        """Check if TTS is currently playing"""
        return self._is_playing
    
    async def _playback_loop(self):
        """
        Main playback loop.
        
        Accumulates tokens and sends to TTS in chunks.
        """
        self._is_playing = True
        accumulated_text = ""
        
        try:
            while not self._should_stop:
                try:
                    # Get token with timeout
                    token = await asyncio.wait_for(
                        self._text_buffer.get(),
                        timeout=0.1
                    )
                    
                    if token is None:  # Sentinel - end of stream
                        break
                    
                    accumulated_text += token
                    
                    # Send to TTS when we have enough text
                    # (or implement true streaming if your TTS supports it)
                    if len(accumulated_text) > 50 or '.' in token or '?' in token or '!' in token:
                        await self._synthesize_and_play(accumulated_text)
                        accumulated_text = ""
                    
                except asyncio.TimeoutError:
                    # No token received, continue
                    continue
            
            # Flush remaining text
            if accumulated_text and not self._should_stop:
                await self._synthesize_and_play(accumulated_text)
                
        finally:
            self._is_playing = False
    
    async def _synthesize_and_play(self, text: str):
        """
        Synthesize and play text.
        
        Replace with your actual TTS implementation.
        """
        if self._should_stop:
            return
        
        # Example: Synthesize and play
        # audio = await self.tts_client.synthesize(text)
        # await self.tts_client.play_audio(audio)
        
        # Placeholder - replace with actual TTS
        # Simulate TTS playback time
        words = text.split()
        playback_time_s = len(words) * 0.3  # ~300ms per word
        
        # Play in small chunks to allow fast interruption
        chunk_duration = 0.05  # 50ms chunks
        chunks = int(playback_time_s / chunk_duration)
        
        for _ in range(chunks):
            if self._should_stop:
                break
            await asyncio.sleep(chunk_duration)


# Example adapter for your existing backend
class BackendAdapter:
    """
    Adapter for your existing backend components.
    
    This shows how to integrate with your existing code.
    """
    
    @staticmethod
    def create_asr_wrapper(asr_service_path: str = "Backend/audio-transcription"):
        """Create ASR wrapper from your existing ASR service"""
        # Import your ASR service
        # from Backend.audio_transcription.asr_service import ASRService
        # asr_service = ASRService()
        
        # For now, create a placeholder
        class PlaceholderASR:
            pass
        
        return StreamingASRWrapper(PlaceholderASR())
    
    @staticmethod
    def create_llm_wrapper(llm_service_path: str = "Backend/llm"):
        """Create LLM wrapper from your existing LLM service"""
        # Import your LLM service
        # from Backend.llm.llm import LLMService
        # llm_service = LLMService()
        
        class PlaceholderLLM:
            pass
        
        return CancellableLLMWrapper(PlaceholderLLM())
    
    @staticmethod
    def create_tts_wrapper(tts_service_path: str = "Backend/tts"):
        """Create TTS wrapper from your existing TTS service"""
        # Import your TTS service
        # from Backend.tts.tts import TTSService
        # tts_service = TTSService()
        
        class PlaceholderTTS:
            pass
        
        return InterruptibleTTSWrapper(PlaceholderTTS())
