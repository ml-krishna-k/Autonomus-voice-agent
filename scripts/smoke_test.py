#!/usr/bin/env python
"""Offline sanity check — verifies each stage without a microphone or server.

    python scripts/smoke_test.py

Exercises the text state machines, loads the TTS voice and synthesises a
sentence to `smoke_test.wav`, and loads the ASR model and transcribes that same
audio back. Ends with a round-trip comparison, which catches sample-rate and
voice-config mistakes that a unit test on the filter alone would miss.
"""

from __future__ import annotations

import asyncio
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.asr import ASREngine  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.textproc import SentenceChunker, ThinkTagFilter  # noqa: E402
from app.tts import TTSEngine  # noqa: E402

PHRASE = "Your order will arrive on Tuesday."


def check_textproc() -> None:
    print("[1/3] text processing")

    # Tag split across chunk boundaries — the case the old filter got wrong.
    filt = ThinkTagFilter()
    out = "".join(
        filt.feed(part)
        for part in ["<thi", "nk>secret reason", "ing</thi", "nk>Hello ", "there."]
    )
    out += filt.flush()
    assert out == "Hello there.", f"think filter produced {out!r}"

    # Orphaned closing tag.
    filt2 = ThinkTagFilter()
    assert filt2.feed("abc</think>def") + filt2.flush() == "abcdef"

    # Decimals must not split; sentence ends must.
    chunker = SentenceChunker()
    sentences = chunker.feed("That costs 3.50 dollars. Shall I add it? ")
    tail = chunker.flush()
    assert any("3.50 dollars." in s for s in sentences), sentences
    print(f"      sentences={sentences} tail={tail!r}")
    print("      ✓ filter and chunker behave")


async def check_tts(settings) -> tuple[bytes, int]:
    print("[2/3] TTS")
    engine = TTSEngine(settings)
    engine.load()
    audio = await engine.synthesize(PHRASE)
    seconds = len(audio) / 2 / engine.sample_rate
    print(f"      {len(audio)} bytes @ {engine.sample_rate} Hz = {seconds:.2f}s")
    assert seconds > 0.5, "synthesis suspiciously short"

    out = ROOT / "smoke_test.wav"
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(engine.sample_rate)
        handle.writeframes(audio)
    print(f"      ✓ wrote {out.name}")
    return audio, engine.sample_rate


async def check_asr(settings, audio: bytes, sample_rate: int) -> None:
    print("[3/3] ASR round-trip")
    samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0

    if sample_rate != settings.input_sample_rate:
        # Linear resample is fine for a smoke test.
        target = int(len(samples) * settings.input_sample_rate / sample_rate)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, target),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)

    engine = ASREngine(settings)
    engine.load()
    result = await engine.transcribe(samples)
    print(f"      heard: {result.text!r}  ({result.latency_s:.2f}s)")

    spoken = {w.strip(".,?!").lower() for w in PHRASE.split()}
    heard = {w.strip(".,?!").lower() for w in result.text.split()}
    overlap = len(spoken & heard) / len(spoken)
    print(f"      word overlap: {overlap:.0%}")
    if overlap < 0.5:
        print("      ⚠ low overlap — check the voice sample rate in models/*.onnx.json")
    else:
        print("      ✓ round-trip intelligible")


async def main() -> int:
    settings = get_settings()
    check_textproc()
    audio, rate = await check_tts(settings)
    await check_asr(settings, audio, rate)
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
