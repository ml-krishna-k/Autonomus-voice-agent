
import torch
import numpy as np
import sounddevice as sd
import warnings

# Filter torch warnings
warnings.filterwarnings("ignore")

class SileroVAD:
    def __init__(self, sample_rate=16000, threshold=0.5):
        print("Initializing Silero VAD...")
        self.sample_rate = sample_rate
        self.threshold = threshold
        
        try:
            # Load Silero VAD from torch hub
            # We trust it's available or will be downloaded
            self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad',
                                               force_reload=False,
                                               onnx=False)
            self.model.eval()
            print("[OK] Silero VAD loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load Silero VAD: {e}")
            self.model = None

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if the given audio chunk contains speech.
        audio_chunk: float32 numpy array
        """
        if self.model is None:
            return False
            
        # Prepare tensor
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.flatten()
        
        # Ensure float32 as requested
        audio_chunk = audio_chunk.astype(np.float32)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk)
        
        # Add batch dimension if needed (Silero expect [batch, time] or just [time] depending on version, usually [1, time] is safe)
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Get probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
        return speech_prob > self.threshold

    def start_listening_for_interruption(self, stop_event, callback=None):
        """
        Listens for speech and calls callback (or sets dict flag) when speech detected.
        Blocking loop? No, this should probably be called in a loop by the main thread
        while it waits for TTS.
        """
        pass
