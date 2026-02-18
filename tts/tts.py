import os
import subprocess
import sounddevice as sd
import numpy as np
import threading
import queue

class PiperTTS: # Renamed from SherpaTTS to reflect actual usage
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.sample_rate = 22050 
        self.receiving_data = False # Flag to know if we expect more audio chunks
        
        self.piper_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "piper", "piper.exe")
        self.model_path = os.path.join(model_dir, "en_US-lessac-medium.onnx")
        
        if not os.path.exists(self.piper_path):
            print(f"Error: Piper not found at {self.piper_path}")
            self.valid = False
            return
            
        if not os.path.exists(self.model_path):
             print(f"Error: Model not found at {self.model_path}")
             self.valid = False
             return
             
        self.valid = True
        print(f"Piper TTS Initialized. Model: {self.model_path}")

    def synthesize(self, text: str):
        if not self.valid:
            return

        # print(f"[TTS] Synthesizing text: '{text[:50]}...'")
        try:
            # Run piper
            # output-raw writes raw PCM (s16le) to stdout
            # echo text | piper ...
            
            process = subprocess.Popen(
                [self.piper_path, "--model", self.model_path, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE  # Capture stderr for debugging
            )
            
            stdout_data, stderr_data = process.communicate(input=text.encode('utf-8'))
            
            if process.returncode != 0:
                print(f"[TTS] Piper failed with return code {process.returncode}")
                if stderr_data:
                    print(f"[TTS] Piper Error: {stderr_data.decode('utf-8', errors='ignore')}")
                return

            if stdout_data:
                # Piper raw output is int16 (s16le)
                audio_data = np.frombuffer(stdout_data, dtype=np.int16)
                # Sounddevice expects float32 usually, or int16? 
                # sd.play supports int16 if mapping is okay, but creating a float array is safer
                # audio_float = audio_data.astype(np.float32) / 32768.0
                
                print(f"[TTS] Generated {len(audio_data)} samples. Queueing...")
                self.audio_queue.put(audio_data)
                
                if not self.is_playing:
                    self.start_playback()
            else:
                print("[TTS] Warning: Piper produced no audio output.")
                if stderr_data:
                     print(f"[TTS] Piper Stderr: {stderr_data.decode('utf-8', errors='ignore')}")
                    
        except Exception as e:
            print(f"TTS Synthesis Error: {e}")

    def start_playback(self):
        # Even if thread alive, ensure we mark as receiving
        self.receiving_data = True
        
        if self.worker_thread and self.worker_thread.is_alive():
            return
            
        self.stop_event.clear()
        self.is_playing = True
        self.worker_thread = threading.Thread(target=self._playback_worker)
        self.worker_thread.start()

    def end_turn(self):
        """Signal that no more data is coming for this turn."""
        self.receiving_data = False

    def _playback_worker(self):
        print("[TTS] Playback worker started.")
        
        # Create an OutputStream
        # We need to define the callback or just write to it.
        # Writing to a stream is blocking if buffer is full, so it handles timing.
        
        try:
             with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='int16') as stream:
                while not self.stop_event.is_set():
                    try:
                        # Get data from queue
                        samples = self.audio_queue.get(timeout=0.1)
                        
                        # Check for stop *before* writing
                        if self.stop_event.is_set():
                            break

                        # Write to stream
                        # This blocks until data is consumed by hardware
                        # We need to chunk this so we can interrupt *during* a long sentence if needed?
                        # Piper outputs sentences. If a sentence is 2 seconds, we block for 2 seconds.
                        # We should write in small chunks.
                        
                        chunk_size = 1024
                        idx = 0
                        while idx < len(samples):
                            if self.stop_event.is_set():
                                break
                            
                            end = idx + chunk_size
                            chunk = samples[idx:end]
                            stream.write(chunk)
                            idx = end
                            
                        print(f"[TTS] Finished playing chunk of {len(samples)} samples")
                        
                    except queue.Empty:
                        if not self.receiving_data and self.audio_queue.empty():
                            print("[TTS] Queue empty and turn ended. Stopping worker.")
                            break
                        continue
                    except Exception as e:
                        print(f"Playback error: {e}")
                        break
        except Exception as e:
            print(f"[TTS] Stream error: {e}")
            
        self.is_playing = False
        print("[TTS] Playback worker stopped.")

    def stop(self):
        self.stop_event.set()
        # The worker thread will exit its loop.
        # Because stream.write is blocking, it might take a moment if the buffer is large,
        # but sounddevice usually handles thread interruptions or we can rely on the chunking loop.
        if self.worker_thread:
             self.worker_thread.join()
        
        # Clear the queue so we don't play old audio later
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_busy(self):
        return not self.audio_queue.empty() or self.is_playing
