
import asyncio
import numpy as np
from easyturn.integrations import StreamingASRWrapper, CancellableLLMWrapper, InterruptibleTTSWrapper
from asr.wenet_asr import WeNetASR
from llm.agent import run_agent
from tts.tts import PiperTTS
import sounddevice as sd

class WeNetASRWrapper(StreamingASRWrapper):
    def __init__(self):
        # Initialize WeNet with tuned parameters
        self.asr = WeNetASR(
            chunk_size=16,
            num_left_chunks=-1,
            beam_size=10,
            ctc_weight=0.5
        )
        self.is_streaming = False
        self._partial_lock = asyncio.Lock()
        
    async def start_stream(self):
        self.is_streaming = True
        self.asr.reset()
        
    async def stop_stream(self):
        self.is_streaming = False

    async def get_partial_result(self):
        # WeNetASR.get_latest_partial returns (text, confidence)
        return self.asr.get_latest_partial()

    async def send_audio(self, audio_chunk: np.ndarray):
        if not self.is_streaming:
            return
        # Process chunk synchronously for now as WeNet is fast enough or offload to thread if needed
        # Adapting sync to async if necessary
        # For simplicity, we call it directly. In production, run_in_executor might be better.
        text, conf = self.asr.transcribe(audio_chunk)
        # WeNetASR updates its internal state, so get_partial_result will pick it up
        pass

class AgentLLMWrapper(CancellableLLMWrapper):
    def __init__(self):
        self.is_generating = False
        self._cancel_event = asyncio.Event()

    async def generate_streaming(self, prompt: str):
        self.is_generating = True
        self._cancel_event.clear()
        
        # run_agent is a sync generator, we need to iterate it.
        # Ideally run_agent should be async, but it's not.
        # We can iterate it in a separate thread to avoid blocking the event loop?
        # Or just iterate it if it's fast (it calls LLM which blocks).
        # We MUST run it in executor.
        
        queue = asyncio.Queue()
        
        def run_target():
            try:
                for chunk in run_agent(prompt):
                    if self._cancel_event.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as e:
                print(f"LLM Error: {e}")
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop) # Sentinel

        loop = asyncio.get_running_loop()
        import threading
        thread = threading.Thread(target=run_target)
        thread.start()
        
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            if not self.is_generating:
                self._cancel_event.set()
                break
            yield chunk

    async def cancel_generation(self):
        self.is_generating = False
        self._cancel_event.set()

class PiperTTSWrapper(InterruptibleTTSWrapper):
    def __init__(self, model_dir):
        self.tts = PiperTTS(model_dir)
        self.is_playing_flag = False
        super().__init__(self.tts)

    async def stop(self):
        self.tts.stop() # Assuming PiperTTS has stop()
        self.is_playing_flag = False
        
    async def _synthesize_and_play(self, text: str):
        self.is_playing_flag = True
        # piper.synthesize queues audio and plays it in background thread
        # We assume synthesize is non-blocking or fast enough
        self.tts.synthesize(text)
        
        # We need to know when it finishes? 
        # PiperTTS.is_busy can be checked.
        while self.tts.is_busy:
            if not self.is_playing_flag: # Interrupted
                self.tts.stop()
                break
            await asyncio.sleep(0.05)
        self.is_playing_flag = False
