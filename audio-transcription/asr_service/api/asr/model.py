from faster_whisper import WhisperModel
from threading import Lock
import os

class ASRModel:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    
                    # Using distil-small.en as requested for better accuracy than tiny, but distilled for speed.
                    model_id = "distil-small.en" 
                    print(f"[INFO] Loading ASR model: {model_id}")
                    
                    cls._instance.model = WhisperModel(
                        model_id,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=4,
                        num_workers=1
                    )
        return cls._instance

    def transcribe(self, pcm: list[float]):
        # Optimized for latency and noise reduction
        return self.model.transcribe(
            pcm,
            beam_size=1,             # Reduced beam_size to 1 for lowest latency (greedy search)
            temperature=0.0,
            vad_filter=True,         # Enabled VAD filter to reduce background noise sensitivity
            condition_on_previous_text=False # Reduces hallucination loops
        )