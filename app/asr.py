"""Speech recognition — faster-whisper behind an async, bounded-concurrency API.

The model is loaded once at startup and shared by every session. Transcription
is CPU-bound and releases the GIL inside CTranslate2, so it runs in a thread
pool; a semaphore keeps concurrent decodes from thrashing the CPU and inflating
latency for everyone.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    language: str | None
    duration_s: float
    latency_s: float


class ASREngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._semaphore = asyncio.Semaphore(max(1, settings.asr_max_concurrency))

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Blocking model load. Call once, off the event loop."""
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        log.info(
            "loading ASR model=%s device=%s compute=%s",
            self._settings.asr_model,
            self._settings.asr_device,
            self._settings.asr_compute_type,
        )
        self._model = WhisperModel(
            self._settings.asr_model,
            device=self._settings.asr_device,
            compute_type=self._settings.asr_compute_type,
            cpu_threads=self._settings.asr_cpu_threads,
            num_workers=max(1, self._settings.asr_max_concurrency),
        )
        log.info("ASR ready in %.1fs", time.perf_counter() - started)

    async def transcribe(self, pcm: np.ndarray) -> Transcript:
        if self._model is None:
            raise RuntimeError("ASR model not loaded")

        duration = len(pcm) / self._settings.input_sample_rate
        started = time.perf_counter()

        async with self._semaphore:
            text, language = await asyncio.to_thread(self._transcribe_blocking, pcm)

        latency = time.perf_counter() - started
        log.debug(
            "asr audio=%.2fs latency=%.2fs rtf=%.2f",
            duration,
            latency,
            latency / duration if duration else 0.0,
        )
        return Transcript(
            text=text, language=language, duration_s=duration, latency_s=latency
        )

    def _transcribe_blocking(self, pcm: np.ndarray) -> tuple[str, str | None]:
        segments, info = self._model.transcribe(
            pcm,
            beam_size=self._settings.asr_beam_size,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text)
        language = getattr(info, "language", None)
        return text.strip(), language
