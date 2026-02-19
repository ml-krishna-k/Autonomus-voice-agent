
import sounddevice as sd
import numpy as np
import time

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels

    def record_until_silence(self, threshold=0.005, silence_duration=1.5, max_duration=15, min_duration=1.0, interrupt_check=None, vad_function=None):
        """
        Records audio until silence is detected for `silence_duration` seconds.
        Or until interrupt_check returns True.
        vad_function: Optional function that takes audio_chunk and returns True if speech is present.
        """
        print(f"Listening... (Threshold: {threshold}) (Press 'q' to quit)")
        audio_buffer = []
        silent_chunks = 0
        chunk_size = 512 # Silero VAD requiring 512 samples at 16k
        max_rms_seen = 0.0
        
        # Calculate chunks for silence duration
        chunks_per_second = self.sample_rate / chunk_size
        silence_chunks_limit = int(silence_duration * chunks_per_second)
        
        # Use simple RMS threshold if no VAD provided
        use_vad = vad_function is not None
        
        stream = sd.InputStream(samplerate=self.sample_rate, channels=self.channels, blocksize=chunk_size, dtype='float32')
        with stream:
            start_time = time.time()
            has_speech_started = False
            
            while True:
                # Check interruption
                if interrupt_check and interrupt_check():
                    return None
                    
                # Read chunk
                indata, overflowed = stream.read(chunk_size)
                if overflowed:
                    # print("Warning: Audio overflow")
                    pass
                
                audio_chunk = indata.flatten()
                
                # Check volume (Root Mean Square) for debug/fallback
                rms = np.sqrt(np.mean(audio_chunk**2))
                max_rms_seen = max(max_rms_seen, rms)
                
                # Check current duration
                elapsed = time.time() - start_time
                
                is_speech = False
                if use_vad:
                   # Use VAD function
                   try:
                       is_speech = vad_function(audio_chunk)
                   except Exception as e:
                       print(f"VAD Error: {e}")
                       is_speech = rms > threshold
                else:
                   is_speech = rms > threshold
                
                if is_speech:
                    silent_chunks = 0
                    if not has_speech_started:
                        tag = "VAD" if use_vad else "RMS"
                        print(f" [Speech detected ({tag})] ", end="", flush=True)
                        has_speech_started = True
                else:
                    if has_speech_started:
                        silent_chunks += 1
                        
                # Maintain a rolling buffer of pre-speech audio (approx 0.5s)
                if not has_speech_started:
                    # Keep last ~20 chunks (approx 0.6s at 16k/512)
                    if len(audio_buffer) >= 20:
                        audio_buffer.pop(0)
                    audio_buffer.append(audio_chunk)
                else:
                    audio_buffer.append(audio_chunk)

                # Stop conditions
                if has_speech_started and silent_chunks > silence_chunks_limit:
                    current_duration = (len(audio_buffer) * chunk_size) / self.sample_rate
                    if current_duration < min_duration:
                        print(f" [Too short ({current_duration:.2f}s)]", end="", flush=True)
                        has_speech_started = False
                        silent_chunks = 0
                        audio_buffer = []
                        start_time = time.time()
                        continue
                    
                    print(" [Silence detected] ")
                    break
                    
                if elapsed > max_duration:
                    if not has_speech_started:
                        return np.array([], dtype=np.float32) # Timeout with no speech
                    print(" [Max duration] ")
                    break
                    
                # If no speech detected for a long time (e.g. 5s), just return empty to retry or loop
                if not has_speech_started and elapsed > 5.0:
                    # Return empty to avoid sending silence to ASR
                    return np.array([], dtype=np.float32)

        # Concatenate all chunks
        if not audio_buffer:
            return np.array([], dtype=np.float32)
            
        full_audio = np.concatenate(audio_buffer)
        
        # Trim leading/trailing silence (optional, simple logic)
        
        return full_audio
