
import asyncio
import logging
import signal
import sys
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

from easyturn.dialogue_orchestrator import DialogueOrchestrator, SystemConfig
from easyturn.controller import EasyTurnController
from backend_adapters import WeNetASRWrapper, AgentLLMWrapper, PiperTTSWrapper

# Global orchestration object for signal handling
orchestrator = None

def signal_handler(sig, frame):
    logger.info("Received interrupt signal, shutting down...")
    if orchestrator:
        asyncio.create_task(orchestrator.stop())
    sys.exit(0)

async def main():
    global orchestrator
    
    print("Initializing Autonomous Voice Agent with WeNet & EasyTurn...")
    
    # 1. Initialize Adapters
    # WeNet ASR (U2++)
    try:
        asr_wrapper = WeNetASRWrapper()
    except Exception as e:
        logger.error(f"Failed to initialize WeNet ASR: {e}")
        return

    # LLM Agent (LangChain + Local logic)
    llm_wrapper = AgentLLMWrapper()
    
    # Piper TTS
    # Assumes models directory exists
    tts_model_dir = "C:\\Autonomus voice agent\\Backend\\models" 
    try:
        tts_wrapper = PiperTTSWrapper(model_dir=tts_model_dir)
    except Exception as e:
        logger.error(f"Failed to initialize Piper TTS: {e}")
        print(f"Make sure Piper model exists in {tts_model_dir}")
        return

    # 2. Configure System
    config = SystemConfig(
        sample_rate=16000,
        frame_duration_ms=30, # 30ms frames
        min_silence_to_speak_ms=400,
        hysteresis_window_ms=200,
        interruption_latency_ms=50,
        log_decisions=True
    )

    # 3. Create Orchestrator
    orchestrator = DialogueOrchestrator(
        config=config,
        asr_client=asr_wrapper,
        llm_client=llm_wrapper,
        tts_client=tts_wrapper
    )

    # 4. Start Orchestrator
    await orchestrator.start()
    
    # 5. Audio Input Handling
    # We need to bridge the sync sounddevice callback to the async orchestrator
    loop = asyncio.get_running_loop()
    
    def audio_callback(indata, frames, time, status):
        if status:
            logger.warning(f"Audio status: {status}")
        
        # Copy input data to ensure it's valid when processing
        audio_chunk = indata.copy().flatten()
        
        # Dispatch to async loop
        # fire-and-forget task
        asyncio.run_coroutine_threadsafe(
            orchestrator.process_audio_frame(audio_chunk, time.currentTime),
            loop
        )

    # Start independent audio stream
    try:
        with sd.InputStream(
            channels=1, 
            samplerate=16000, 
            dtype='float32',
            blocksize=int(16000 * 0.03), # 30ms block size matching frame duration
            callback=audio_callback
        ):
            print("Listening... Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(1)
                
    except Exception as e:
        logger.error(f"Audio stream error: {e}")
    finally:
        await orchestrator.stop()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
