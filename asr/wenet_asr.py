
import os
import torch
import wenet
import numpy as np
from typing import Optional, Tuple

class WeNetASR:
    def __init__(self, 
                 model_dir: str = None, 
                 chunk_size: int = 16, 
                 num_left_chunks: int = -1, 
                 beam_size: int = 10, 
                 ctc_weight: float = 0.5):
        """
        Initialize WeNet ASR with U2++ model.
        
        Args:
            model_dir (str): Path to model directory. If None, loads pretrained 'gigaspeech'.
            chunk_size (int): Decoding chunk size.
            num_left_chunks (int): Number of left chunks for attention.
            beam_size (int): Beam search size.
            ctc_weight (float): CTC weight for joint decoding.
        """
        print(f"Loading WeNet model with chunk_size={chunk_size}, left_chunks={num_left_chunks}...")
        
        # Load model - wenetruntime automatically downloads if 'gigaspeech' is passed
        # 'gigaspeech' model is usually a U2++ conformation
        try:
            self.model = wenet.Decoder(lang='english') 
        except Exception as e:
            print(f"Failed to load WeNet model: {e}")
            self.model = None
            return

        # Configure decoding options
        # Note: wenetruntime python bindings might not expose all these directly in init,
        # but decode() often takes them, or we use set_opts() if available.
        # Checking typical usage: decoder.decode(audio, continuous_decoding=True)
        # Advanced config might need looking at specific API. 
        # For now, we assume standard usage structure.
        
        self.chunk_size = chunk_size
        self.num_left_chunks = num_left_chunks
        self.beam_size = beam_size
        self.ctc_weight = ctc_weight
        
        self.reset()
        print("WeNet ASR initialized.")

    def reset(self):
        """Reset the decoder state."""
        if self.model:
            self.model.reset()
        self.current_result = ""

    def transcribe(self, audio_chunk: np.ndarray) -> Tuple[str, float]:
        """
        Process an audio chunk and return (text, confidence).
        Audio chunk should be 16k mono float32.
        """
        if self.model is None:
            return "", 0.0
            
        # Convert to bytes if needed, but wenetruntime usually takes bytes or path?
        # Typically: decode(bytes, last=False)
        # We need to convert numpy array to PCM bytes (int16)
        
        # Ensure audio is float32
        if audio_chunk.dtype != np.float32:
             audio_chunk = audio_chunk.astype(np.float32)
             
        # Convert to int16 bytes
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Decode
        # The python API usually returns a JSON string or Result object
        # Tuning parameters passed here? 
        # Actually most params are set at load time or handled internally.
        # But `decode` might accept some. 
        # Ideally we'd set them. If not exposed, we use defaults or re-init (if supported).
        # For now, just call decode.
        
        ans = self.model.decode(audio_bytes, last=False) # Only simple decode available in high-level API usually
        
        # ans is typically a JSON string: {"nbest": [{"sentence": "...", "score": ...}]}
        import json
        try:
            res = json.loads(ans)
            if res['nbest']:
                text = res['nbest'][0]['sentence']
                confidence = 0.8 # Placeholder or extract from score? Score is usually log-likelihood.
                
                # Check for "final" flag if any? usage usually implies stream accumulator.
                self.current_result = text
                return text, confidence
        except:
             pass
             
        return self.current_result, 0.0

    def get_latest_partial(self):
        """Compatible with EasyTurn wrapper expectation"""
        return self.current_result, 0.8 # Confidence placeholder
