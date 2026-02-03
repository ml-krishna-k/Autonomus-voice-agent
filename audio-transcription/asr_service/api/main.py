from fastapi import FastAPI, UploadFile, HTTPException
import tempfile
import time
import os
import uvicorn

import sys
from pathlib import Path

# Add Backend root to sys.path to allow imports
backend_root = Path(__file__).resolve().parents[3]
sys.path.append(str(backend_root))

from asr.audio import load_mp3_as_pcm
from asr.model import ASRModel
from asr.utils import segments_to_text
from llm.llm import get_llm_response
from tts.tts import SherpaTTS
import re

# Initialize TTS (Lazy load or global)
# We assume models are in Backend/models
models_dir = os.path.join(str(backend_root), "models")
tts_engine = SherpaTTS(models_dir)

# Simple Text Stream Segmenter
def yield_stream_segments(text_stream):
    buffer = ""
    for chunk in text_stream:
        buffer += chunk
        # Split by sentence endings or roughly every 10-15 words
        # Simple regex for sentence split
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
                if len(words) > 12: # Smaller chunk needed for faster responsiveness
                    # Find nearest space to split
                    sentence = " ".join(words[:12])
                    buffer = " ".join(words[12:])
                    yield sentence
                else:
                    break # Not enough words yet
    if buffer.strip():
        yield buffer.strip()



app = FastAPI(title="CPU Distil-Whisper ASR")

asr = ASRModel()

MAX_AUDIO_SECONDS = 30 * 60  

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only MP3 files supported")

    start_time = time.time()

    # Windows: NamedTemporaryFile usually locks the file, so we can't open it in
    # subprocess (ffmpeg) while it's open here. We must close it first.
    # delete=False prevents auto-deletion on close.
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        pcm = load_mp3_as_pcm(tmp_path)

        duration_sec = len(pcm) / 16000
        if duration_sec > MAX_AUDIO_SECONDS:
            raise HTTPException(status_code=400, detail="Audio too long")

        try:
            segments, info = asr.transcribe(pcm)
        except RuntimeError:
            raise HTTPException(status_code=500, detail="ASR inference failed")

        latency = round(time.time() - start_time, 3)
        
        transcribed_text = segments_to_text(segments)
        
        # Write output to file as requested
        output_file_path = r"C:\Autonomus voice agent\Backend\audio-transcription\asr_output_text.txt"
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(transcribed_text)
        except Exception as e:
            print(f"Failed to write output to file: {e}")

        return {
            "text": transcribed_text,
            "language": info.language,
            "confidence": info.language_probability,
            "audio_duration_sec": round(duration_sec, 2),
            "latency_sec": latency,
            "rtf": round(latency / duration_sec, 3)
        }
    finally:
        # Clean up the temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def process_file_local(input_path: str, output_path: str):
    """
    Helper to run transcription locally without a server.
    """
    print(f"Processing local file: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    try:
        pcm = load_mp3_as_pcm(input_path)
        
        duration_sec = len(pcm) / 16000
        print(f"Audio duration: {duration_sec:.2f}s")
        if duration_sec > MAX_AUDIO_SECONDS:
            print("Error: Audio too long")
            return

        start_time = time.time()
        segments, info = asr.transcribe(pcm)
        latency = time.time() - start_time
        
        transcribed_text = segments_to_text(segments)
        print(f"Transcription complete. Latency: {latency:.2f}s")
        print(f"Language: {info.language} ({info.language_probability:.2f})")
        print(f"Transcribed Text: [{transcribed_text}]")
        
        print("Sending transcription to LLM...")
        
        full_llm_response = ""
        print("LLM Response Stream:")
        
        # Iterate over the generator, segmenting it in real-time
        for segment in yield_stream_segments(get_llm_response(transcribed_text)):
            print(f" [TTS] >> {segment}")
            full_llm_response += segment + " "
            
            # Send to TTS
            if tts_engine:
                tts_engine.synthesize(segment)
            
        print() # Newline after stream ends

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_llm_response)
        print(f"LLM output written to: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # If run directly, check if we want to run server or local sample
    # For now, we defaults to running the local sample as requested.
    
    sample_file = r"C:\Autonomus voice agent\Backend\audio-transcription\sampleaudio\sampleaudio.mp3"
    output_file = r"C:\Autonomus voice agent\Backend\audio-transcription\asr_output_text.txt"
    
    # Uncomment the following line to run as server instead
    # uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    
    print("Running in local script mode...")
    process_file_local(sample_file, output_file)