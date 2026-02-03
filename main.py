import os
import sys
import time
from pathlib import Path
import re
import numpy as np

# Add path to ASR module
# ASR is located at Backend/audio-transcription/asr_service/api
# We need to be able to import 'asr' from there.
# If we append 'audio-transcription/asr_service/api' to sys.path, we can do `from asr.model import ...`
current_dir = Path(__file__).resolve().parent
asr_api_path = current_dir / "audio-transcription" / "asr_service" / "api"
sys.path.append(str(asr_api_path))

try:
    from asr.audio import load_mp3_as_pcm
    from asr.model import ASRModel
    from asr.utils import segments_to_text
except ImportError as e:
    print(f"Error importing ASR modules: {e}")
    sys.exit(1)

from llm.llm import get_llm_response, clear_history
from tts.tts import SherpaTTS # Using our Piper wrapper

# Initialize TTS
models_dir = os.path.join(current_dir, "models")
tts_engine = SherpaTTS(models_dir)

# Initialize ASR
asr = ASRModel()
MAX_AUDIO_SECONDS = 30 * 60


def filter_thinking(text_stream):
    in_think_block = False
    buffer = ""
    
    for chunk in text_stream:
        buffer += chunk
        
        while True:
            # Check for opening tag
            if not in_think_block:
                start_match = re.search(r'<think>', buffer)
                if start_match:
                    # Yield everything before the tag
                    pre_think = buffer[:start_match.start()]
                    if pre_think:
                        yield pre_think
                    # Update buffer to start after the opening tag
                    # But wait, looking for closing tag in the rest
                    buffer = buffer[start_match.end():]
                    in_think_block = True
                else:
                    # No opening tag found yet.
                    # Be careful not to yield partial tag (e.g. "<th")
                    # If buffer ends with partial tag, keep it.
                    # Simple heuristic: if buffer contains '<', wait.
                    if '<' in buffer:
                        # Find the last '<'
                        last_open = buffer.rfind('<')
                        if last_open > 0:
                            yield buffer[:last_open]
                            buffer = buffer[last_open:]
                        # If at 0, yield nothing, wait for more chunks
                        break
                    else:
                        yield buffer
                        buffer = ""
                        break
            
            # Check for closing tag
            if in_think_block:
                end_match = re.search(r'</think>', buffer)
                if end_match:
                    # Found closing tag. Ignore content.
                    # Buffer starts after the closing tag
                    buffer = buffer[end_match.end():]
                    in_think_block = False
                    # Loop again to see if there is valid text or another think block
                    continue
                else:
                    # No closing tag. Discard buffer (it's thinking) 
                    # except maybe keep tail if partial closing tag?
                    # For safety, just discard everything except potential partial closing "</"
                    # But simpler: just clear buffer if we are deep in thought
                    # To be robust against split tags like "</th" + "ink>", we keep a small tail
                    if len(buffer) > 10: # Keep small context
                         buffer = buffer[-10:] 
                    break

    # Yield remaining buffer if not in think block
    if buffer and not in_think_block:
        yield buffer

# Simple Text Stream Segmenter
def yield_stream_segments(text_stream):
    buffer = ""
    for chunk in text_stream:
        buffer += chunk
        while True:
            # Check for punctuation
            match = re.search(r'([.?!:;]+)', buffer)
            if match:
                end_idx = match.end()
                sentence = buffer[:end_idx]
                buffer = buffer[end_idx:]
                if sentence.strip():
                    yield sentence.strip()
            else:
                # Check word count if no punctuation
                words = buffer.split()
                if len(words) > 12: 
                    sentence = " ".join(words[:12])
                    buffer = " ".join(words[12:])
                    yield sentence
                else:
                    break 
    if buffer.strip():
        yield buffer.strip()

from audio.recorder import AudioRecorder

def process_audio_chunk(pcm: np.ndarray):
    """
    Process a raw PCM audio chunk through the pipeline.
    """
    duration_sec = len(pcm) / 16000
    if duration_sec < 0.5:
        # print("Audio too short, ignoring.")
        return

    # 1. Transcription (ASR)
    print(f"--- Processing Audio ({duration_sec:.2f}s) ---")
    try:
        start_time = time.time()
        segments, info = asr.transcribe(pcm)
        latency = time.time() - start_time
        transcribed_text = segments_to_text(segments)
        
        if not transcribed_text.strip():
            print("No speech detected.")
            return

        # Filter common hallucinations
        clean_text = transcribed_text.strip().lower()
        if clean_text in ["[you]", "you", "thank you", "thanks", "mbc", "mo", "blank"]:
             print(f"Ignored hallucination: {transcribed_text}")
             return

        print(f"Transcription Latency: {latency:.2f}s")
        print(f"User: [{transcribed_text}]")
    except Exception as e:
        print(f"ASR Error: {e}")
        return

    # 2. LLM & TTS Streaming
    full_llm_response = ""
    try:
        llm_stream = get_llm_response(transcribed_text)
        filtered_stream = filter_thinking(llm_stream)
        
        print("Agent: ", end="", flush=True)
        for segment in yield_stream_segments(filtered_stream):
            print(f"{segment} ", end="", flush=True)
            full_llm_response += segment + " "
            
            # Send to TTS
            if tts_engine:
                 tts_engine.synthesize(segment)
            else:
                pass
        print() # Newline
        
        # Signal TTS that turn is done
        if tts_engine:
            tts_engine.end_turn()
                
    except Exception as e:
        print(f"LLM/TTS Error: {e}")

    # 3. Save Output (Optional, just logging last turn)
    try:
        output_file = os.path.join(current_dir, "audio-transcription", "asr_output_text.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_llm_response)
    except Exception:
        pass

import msvcrt

def check_quit_key():
    """
    Returns True if 'q' is pressed.
    """
    if msvcrt.kbhit():
        key = msvcrt.getch()
        try:
            # Handle standard keys
            char = key.decode('utf-8').lower()
            if char == 'q':
                return True
        except UnicodeDecodeError:
            pass # Ignore special keys
    return False

def run_live_loop():
    print("Initializing Microphone...")
    recorder = AudioRecorder()
    
    # Reset history at startup
    clear_history()
    
    # Initial Greeting
    greeting_text = "Hello, How can i assit u today"
    print(f"Agent: {greeting_text}")
    if tts_engine:
        tts_engine.synthesize(greeting_text)
        tts_engine.end_turn()

    print("System Ready. Start speaking...")
    print(">> Press and hold 'q' to quit at any time.")
    
    try:
        while True:
            # Global quit check
            if check_quit_key():
                print("\nQuit signal received.")
                clear_history()
                break

            # Check if TTS is currently playing (to avoid listening to self)
            if tts_engine and tts_engine.is_playing:
                time.sleep(0.1)
                continue
                
            # Record with interrupt check
            audio_data = recorder.record_until_silence(interrupt_check=check_quit_key)
            
            # If aborted
            if audio_data is None:
                print("\nQuit signal received during recording.")
                clear_history()
                break
            
            if len(audio_data) > 0:
                process_audio_chunk(audio_data)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        clear_history()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--file":
        # Legacy file mode
        sample_file = os.path.join(current_dir, "audio-transcription", "sampleaudio", "sampleaudio.mp3")
        output_file = os.path.join(current_dir, "audio-transcription", "asr_output_text.txt")
        process_pipeline(sample_file, output_file)
    else:
        # Default to live mode
        run_live_loop()
