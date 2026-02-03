from tts.tts import SherpaTTS
import os
import time

print("Testing pure TTS...")
models_dir = os.path.join(os.getcwd(), "models")
tts = SherpaTTS(models_dir)

if tts:
    print("Synthesizing...")
    tts.synthesize("Hello, this is a test.")
    time.sleep(2)
    tts.stop()
    print("Done.")
else:
    print("TTS failed to load.")
