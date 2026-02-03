import sounddevice as sd
import numpy as np
import time

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels

    def record_until_silence(self, threshold=0.01, silence_duration=1.5, max_duration=15, min_duration=1.0, interrupt_check=None):
        """
        Records audio until silence is detected for `silence_duration` seconds.
        Or until interrupt_check returns True.
        """
        print(f"Listening... (Threshold: {threshold}) (Press 'q' to quit)")
        audio_buffer = []
        silent_chunks = 0
        chunk_size = 1024
        max_rms_seen = 0.0
        
        # Calculate chunks for silence duration
        chunks_per_second = self.sample_rate / chunk_size
        silence_chunks_limit = int(silence_duration * chunks_per_second)
        
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
                    print("Warning: Audio overflow")
                
                audio_chunk = indata.flatten()
                
                # Check volume (Root Mean Square)
                rms = np.sqrt(np.mean(audio_chunk**2))
                max_rms_seen = max(max_rms_seen, rms)
                
                # Log level occasionally if needed, or just track logic
                # if rms > 0.1: print("#", end="", flush=True)
                
                # Check current duration
                elapsed = time.time() - start_time
                
                if rms > threshold:
                    silent_chunks = 0
                    if not has_speech_started:
                        print(" [Speech detected] ", end="", flush=True)
                        has_speech_started = True
                else:
                    if has_speech_started:
                        silent_chunks += 1
                
                # Only append to buffer if speech has started or we are keeping a small lookback?
                # For simplicity, append everything, BUT if we timeout without speech, we discard it.
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
