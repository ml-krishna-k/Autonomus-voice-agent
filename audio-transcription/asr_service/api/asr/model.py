from faster_whisper import WhisperModel
from threading import Lock

class ASRModel:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.model = WhisperModel(
                        "distil-small.en",
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=4,
                        num_workers=1
                    )
        return cls._instance

    def transcribe(self, pcm: list[float]):
        return self.model.transcribe(
            pcm,
            beam_size=2,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False
        )