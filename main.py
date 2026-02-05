import os
import sys
import time
import msvcrt
import re
import atexit
from pathlib import Path

# Ensure paths are correct
sys.path.append(os.path.join(os.path.dirname(__file__), "audio-transcription"))

from audio.recorder import AudioRecorder
# Dynamic import for ASR to handle potential path issues gracefully-ish
try:
    from asr_service.api.asr.model import ASRModel
except ImportError:
    try:
        from asr.model import ASRModel
    except ImportError:
        print("Error importing ASRModel. Check paths.")
        sys.exit(1)

from tts.tts import SherpaTTS
from llm.llm import get_llm_response, clear_history

class VoiceAgent:
    def __init__(self):
        print("Initializing Autonomous Voice Agent...")
        self.running = True
        
        # 1. Components
        try:
            self.recorder = AudioRecorder()
            print("[OK] Microphone initialized.")
            
            self.asr_model = ASRModel()
            print("[OK] ASR Model loaded.")
            
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            self.tts = SherpaTTS(models_dir)
            print("[OK] TTS System initialized.")
            
        except Exception as e:
            print(f"[CRITICAL] Initialization failed: {e}")
            sys.exit(1)
            
        # State
        self.response_buffer = ""
        self.in_think_block = False
        self.filter_buffer = ""

        # Register cleanup
        atexit.register(self.cleanup)

    def cleanup(self):
        """Cleanup resources on exit."""
        print("\nCleaning up resources...")
        try:
            clear_history() 
            if hasattr(self, 'tts'):
                self.tts.stop()
        except:
            pass

    def check_quit(self):
        """Check for 'q' key."""
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key.lower() == b'q':
                self.running = False
                return True
        return False

    def filter_text(self, chunk: str) -> str:
        """
        Filter out <think> tags from chunks for TTS.
        Maintains state to handle tags split across chunks.
        """
        text = self.filter_buffer + chunk
        self.filter_buffer = ""
        
        filtered = ""
        i = 0
        while i < len(text):
            if self.in_think_block:
                close_idx = text.find("</think>", i)
                if close_idx != -1:
                    self.in_think_block = False
                    i = close_idx + 8 
                else:
                    # Check for partial closing tag at end
                    partial_found = False
                    for length in range(1, 8):
                        suffix = text[-length:]
                        if "</think>".startswith(suffix):
                            self.filter_buffer = suffix
                            partial_found = True
                            break
                    break # Discard everything else
            else:
                # Not in think block
                open_idx = text.find("<think>", i)
                close_idx = text.find("</think>", i)
                
                # Handle orphaned </think> by skipping it
                if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
                    filtered += text[i:close_idx]
                    i = close_idx + 8
                    continue

                if open_idx != -1:
                    filtered += text[i:open_idx]
                    self.in_think_block = True
                    i = open_idx + 7
                    continue
                
                # Check for partial tags (opening OR closing) at end
                partial_found = False
                for length in range(1, 9):
                    if i + length > len(text): 
                        # Cannot have a suffix longer than remaining text
                        continue
                        
                    suffix = text[-length:]
                    # Check if suffix could be start of <think> OR </think>
                    if "<think>".startswith(suffix) or "</think>".startswith(suffix):
                        filtered += text[i:-length]
                        self.filter_buffer = suffix
                        partial_found = True
                        break
                
                if not partial_found:
                    filtered += text[i:]
                break
                
        return filtered

    def run(self):
        """Main execution loop."""
        clear_history() 
        
        print("\n--- Agent Ready ---")
        print("Press 'q' to quit.")
        
        while self.running:
            if self.check_quit(): break
            
            # Listen
            audio = self.recorder.record_until_silence(interrupt_check=self.check_quit)
            
            if self.check_quit() or audio is None or len(audio) == 0:
                continue

            # Transcribe
            print("\rTranscribing...          ", end="", flush=True)
            try:
                segments, _ = self.asr_model.transcribe(audio)
                text = " ".join([s.text for s in segments]).strip()
                print("\r" + " " * 30 + "\r", end="", flush=True)
                
                if not text:
                    continue
                    
                print(f"You: {text}")
                
            except Exception as e:
                print(f"\nASR Error: {e}")
                continue

            # Response
            print("Assistant: ", end="", flush=True)
            self.process_response(text)
            print("\n")
            
            # Wait for TTS
            self.wait_for_tts()
            time.sleep(0.5)

    def process_response(self, user_text):
        """Get LLM response and feed to TTS."""
        full_buffer = ""
        sentence_buffer = ""
        full_buffer = ""
        sentence_buffer = ""
        self.in_think_block = False # Reset state for new turn
        self.filter_buffer = ""
        
        for chunk in get_llm_response(user_text):
            if self.check_quit(): break
            
            # Debug: Check if LLM sends duplicate data
            # print(f"DEBUG RAW: {repr(chunk)}")
            
            clean_chunk = self.filter_text(chunk)
            
            if clean_chunk:
                print(clean_chunk, end="", flush=True)
                full_buffer += clean_chunk
                sentence_buffer += clean_chunk
                
                if any(p in clean_chunk for p in [".", "?", "!", "\n", ":"]):
                    if sentence_buffer.strip():
                        self.tts.synthesize(sentence_buffer)
                        sentence_buffer = ""

        if sentence_buffer.strip():
            self.tts.synthesize(sentence_buffer)
            
        self.tts.end_turn()

    def wait_for_tts(self):
        """Block until TTS is done."""
        if self.tts.is_playing or not self.tts.audio_queue.empty():
            while self.tts.is_playing or not self.tts.audio_queue.empty():
                time.sleep(0.1)
                if self.check_quit():
                    self.tts.stop()
                    break

if __name__ == "__main__":
    agent = VoiceAgent()
    agent.run()
