import subprocess
import numpy as np
import imageio_ffmpeg

def load_mp3_as_pcm(
    path: str,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Decode MP3 → mono PCM16 → float32 [-1, 1]
    """
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-i", path,
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-"
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True
    )

    audio = np.frombuffer(proc.stdout, np.int16)
    return audio.astype(np.float32) / 32768.0