#!/usr/bin/env python
"""Local microphone mode — the desktop development client.

This talks to the *same* engines the server uses (`app/`), so the VAD, think-tag
filter, sentence chunker and TTS behave identically to production. The only
difference is transport: sound card in, sound card out, instead of a WebSocket.

    pip install -r requirements-dev.txt
    python main.py

For the hosted service, run `uvicorn app.server:app` instead — see README.md.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading

import numpy as np

from app.asr import ASREngine
from app.config import get_settings
from app.llm import LLMClient
from app.pipeline import ConversationPipeline
from app.segmenter import UtteranceSegmenter
from app.session import SessionStore
from app.tts import TTSEngine

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sys.exit(
        "sounddevice is not installed.\n"
        "Local mic mode needs it:  pip install -r requirements-dev.txt"
    )

log = logging.getLogger("local")


class Speaker:
    """Background playback thread with an interruptible queue."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def play(self, pcm: bytes) -> None:
        self._queue.put(pcm)

    def flush(self) -> None:
        """Drop anything not yet played (barge-in)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        with sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16"
        ) as stream:
            while not self._stop.is_set():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                samples = np.frombuffer(item, dtype="<i2")
                # Write in slices so flush()/close() can interrupt a long clip.
                for start in range(0, len(samples), 1024):
                    if self._stop.is_set():
                        return
                    stream.write(samples[start : start + 1024])


async def run() -> int:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(levelname)-7s %(name)s | %(message)s",
    )

    print("Loading models…")
    asr, tts = ASREngine(settings), TTSEngine(settings)
    await asyncio.gather(asyncio.to_thread(asr.load), asyncio.to_thread(tts.load))

    llm = LLMClient(settings)
    await llm.start()
    if not llm.configured:
        print("\n! OPENROUTER_API_KEY is not set — copy .env.example to .env first.")
        return 1

    speaker = Speaker(tts.sample_rate)
    session = SessionStore(settings).create()
    loop = asyncio.get_running_loop()

    async def emit(event: dict) -> None:
        kind = event.get("type")
        if kind == "token":
            print(event["text"], end="", flush=True)
        elif kind == "response_end":
            print()
        elif kind == "error":
            print(f"\n[error] {event['message']}")

    async def emit_audio(pcm: bytes) -> None:
        speaker.play(pcm)

    pipeline = ConversationPipeline(asr, llm, tts, emit, emit_audio)
    segmenter = UtteranceSegmenter(
        sample_rate=settings.input_sample_rate,
        threshold=settings.vad_threshold,
        silence_ms=settings.vad_silence_ms,
        min_speech_ms=settings.vad_min_speech_ms,
        max_utterance_ms=settings.vad_max_utterance_ms,
        preroll_ms=settings.vad_preroll_ms,
    )

    # The audio callback runs on PortAudio's thread; hand frames to the loop.
    frames: asyncio.Queue[np.ndarray] = asyncio.Queue()

    def on_audio(indata, _frames, _time, status) -> None:
        if status:
            log.debug("input status: %s", status)
        loop.call_soon_threadsafe(frames.put_nowait, indata[:, 0].copy())

    turn: asyncio.Task | None = None
    print("\n─── Agent ready. Speak, or press Ctrl+C to quit. ───\n")

    stream = sd.InputStream(
        samplerate=settings.input_sample_rate,
        channels=1,
        dtype="float32",
        blocksize=1024,
        callback=on_audio,
    )

    try:
        with stream:
            while True:
                chunk = await frames.get()
                for event in segmenter.feed(chunk):
                    if event.speech_started:
                        if settings.allow_barge_in and turn and not turn.done():
                            turn.cancel()
                            speaker.flush()
                            print("  [interrupted]")

                    if event.utterance is not None:
                        if turn and not turn.done():
                            turn.cancel()
                        result = await asr.transcribe(event.utterance)
                        if not result.text:
                            continue
                        print(f"\nYou: {result.text}\nKrish: ", end="", flush=True)
                        turn = asyncio.create_task(
                            pipeline.respond(session, result.text)
                        )
    except KeyboardInterrupt:
        pass
    finally:
        if turn and not turn.done():
            turn.cancel()
        speaker.close()
        await llm.aclose()
        print("\nGoodbye.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run()))
    except KeyboardInterrupt:
        raise SystemExit(0)
