"""Text to speech — Piper loaded in-process, once.

The previous implementation spawned `piper.exe` for every sentence, paying
onnxruntime initialisation and a 60 MB voice load on each call (200-500 ms of
pure overhead per sentence). Here the voice is resident for the process
lifetime and synthesis is a plain function call on a worker thread.

The output sample rate is read from the voice's own config file rather than
hardcoded, which fixes the pitch/speed bug the old player had: `en_US-amy-low`
is a 16 kHz voice that was being played back at 22050 Hz.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from .config import Settings

log = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._voice = None
        self._sample_rate = 22050
        self._semaphore = asyncio.Semaphore(max(1, settings.tts_max_concurrency))
        self._synth_call = None

    @property
    def ready(self) -> bool:
        return self._voice is not None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        """Blocking voice load. Call once, off the event loop."""
        onnx_path: Path = self._settings.voice_onnx
        config_path: Path = self._settings.voice_config

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Piper voice not found: {onnx_path}. "
                "Run `python scripts/fetch_models.py` to download it."
            )

        # Trust the voice config for the sample rate — this is the single
        # source of truth and differs between low/medium/high quality voices.
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            self._sample_rate = int(config.get("audio", {}).get("sample_rate", 22050))
        else:
            log.warning("voice config missing at %s, assuming 22050 Hz", config_path)

        started = time.perf_counter()
        from piper import PiperVoice

        kwargs = {"config_path": str(config_path)} if config_path.exists() else {}
        self._voice = PiperVoice.load(str(onnx_path), **kwargs)

        # Piper's Python API changed shape between 1.2 and 1.3. Resolve the
        # right call once at load time instead of branching per sentence.
        self._synth_call = self._resolve_synth_call()

        log.info(
            "TTS ready voice=%s rate=%dHz in %.1fs",
            self._settings.tts_voice,
            self._sample_rate,
            time.perf_counter() - started,
        )

    def _resolve_synth_call(self):
        """Bind the correct Piper call once, instead of branching per sentence.

        piper-tts 1.2 exposes `synthesize_stream_raw` returning raw bytes;
        1.3+ replaced it with `synthesize` yielding `AudioChunk` objects.
        """
        voice = self._voice

        if hasattr(voice, "synthesize_stream_raw"):  # piper-tts 1.2.x
            def call(text: str) -> bytes:
                return b"".join(voice.synthesize_stream_raw(text))

            return call

        if hasattr(voice, "synthesize"):  # piper-tts 1.3+
            syn_config = self._build_synthesis_config()

            def call(text: str) -> bytes:
                parts: list[bytes] = []
                for chunk in voice.synthesize(text, syn_config=syn_config):
                    raw = getattr(chunk, "audio_int16_bytes", None)
                    if raw is None:
                        array = getattr(chunk, "audio_int16_array", None)
                        raw = array.tobytes() if array is not None else bytes(chunk)
                    parts.append(raw)
                    # The chunk itself is the most authoritative rate source.
                    rate = getattr(chunk, "sample_rate", None)
                    if rate and rate != self._sample_rate:
                        log.warning(
                            "voice config says %d Hz but audio is %d Hz; using %d",
                            self._sample_rate,
                            rate,
                            rate,
                        )
                        self._sample_rate = int(rate)
                return b"".join(parts)

            return call

        raise RuntimeError(
            "Unsupported piper-tts version: no synthesize_stream_raw/synthesize"
        )

    def _build_synthesis_config(self):
        """Optional per-synthesis tuning; absent on older piper-tts."""
        if self._settings.tts_length_scale == 1.0:
            return None
        try:
            from piper.config import SynthesisConfig
        except ImportError:
            return None
        return SynthesisConfig(length_scale=self._settings.tts_length_scale)

    async def synthesize(self, text: str) -> bytes:
        """Return int16 little-endian mono PCM at `self.sample_rate`."""
        if self._voice is None:
            raise RuntimeError("TTS voice not loaded")

        text = text.strip()
        if not text:
            return b""

        started = time.perf_counter()
        async with self._semaphore:
            audio = await asyncio.to_thread(self._synth_call, text)

        log.debug(
            "tts chars=%d bytes=%d latency=%.2fs",
            len(text),
            len(audio),
            time.perf_counter() - started,
        )
        return audio
