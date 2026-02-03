import os
import subprocess
import sounddevice as sd
import numpy as np
import threading
import queue

class SherpaTTS: # Keeping name to avoid changing main.py, but implementing Piper
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.sample_rate = 22050 
        self.receiving_data = False # Flag to know if we expect more audio chunks
        
        self.piper_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "piper", "piper.exe")
        self.model_path = os.path.join(model_dir, "en_US-amy-low.onnx")
        
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
            print("TTS Invalid state")
            return

        try:
            # Run piper
            # output-raw writes raw PCM (s16le) to stdout
            # echo text | piper ...
            cmd = [
                self.piper_path,
                "--model", self.model_path,
                "--output-raw",
                "--json-input"  # Safer if we format input as json, but raw text ok if simple
            ]
            
            # Simple Text input
            input_text = text
            
            process = subprocess.Popen(
                [self.piper_path, "--model", self.model_path, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            
            stdout_data, _ = process.communicate(input=text.encode('utf-8'))
            
            if stdout_data:
                # Piper raw output is int16 (s16le)
                audio_data = np.frombuffer(stdout_data, dtype=np.int16)
                # Sounddevice expects float32 usually, or int16? 
                # sd.play supports int16 if mapping is okay, but creating a float array is safer
                # audio_float = audio_data.astype(np.float32) / 32768.0
                
                self.audio_queue.put(audio_data)
                
                if not self.is_playing:
                    self.start_playback()
                    
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
        while not self.stop_event.is_set():
            try:
                samples = self.audio_queue.get(timeout=0.5)
                # Play audio. blocking=True ensures we finish this chunk
                sd.play(samples, self.sample_rate, blocking=True)
            except queue.Empty:
                # If queue empty and we are NOT expecting more data, then stop.
                if not self.receiving_data:
                    break
                continue
            except Exception as e:
                print(f"Playback error: {e}")
                break
                
        self.is_playing = False

    def stop(self):
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join()
