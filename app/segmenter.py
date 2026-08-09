"""Server-side utterance endpointing.

The browser/phone client streams raw PCM continuously; this class decides where
one utterance ends. It is a direct port of the RMS gate that used to live in the
local recorder, reshaped to accept arbitrary-sized network frames instead of
driving a microphone.

Includes a pre-roll ring buffer so the first phoneme of a word is not clipped —
speech is usually already in progress by the time RMS crosses the threshold.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

FRAME_MS = 20


@dataclass
class SegmenterEvent:
    """Emitted when the segmenter's view of the world changes."""

    speech_started: bool = False
    utterance: np.ndarray | None = None
    reason: str | None = None  # "silence" | "max_duration"


class UtteranceSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.01,
        silence_ms: int = 700,
        min_speech_ms: int = 400,
        max_utterance_ms: int = 20000,
        preroll_ms: int = 300,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.frame_size = max(1, int(sample_rate * FRAME_MS / 1000))

        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.max_frames = max(1, max_utterance_ms // FRAME_MS)
        preroll_frames = max(0, preroll_ms // FRAME_MS)

        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll: deque[np.ndarray] = deque(maxlen=preroll_frames or 1)
        self._voiced: list[np.ndarray] = []
        self._speaking = False
        self._silent_run = 0

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._voiced.clear()
        self._speaking = False
        self._silent_run = 0

    def feed(self, pcm: np.ndarray) -> list[SegmenterEvent]:
        """Push float32 mono samples in [-1, 1]; get back zero or more events."""
        events: list[SegmenterEvent] = []
        if pcm.size:
            self._pending = np.concatenate([self._pending, pcm.astype(np.float32)])

        while self._pending.size >= self.frame_size:
            frame = self._pending[: self.frame_size]
            self._pending = self._pending[self.frame_size :]
            event = self._consume_frame(frame)
            if event is not None:
                events.append(event)

        return events

    def _consume_frame(self, frame: np.ndarray) -> SegmenterEvent | None:
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        voiced = rms > self.threshold

        if not self._speaking:
            self._preroll.append(frame)
            if not voiced:
                return None
            # Speech onset: seed the utterance with the pre-roll so we keep the
            # attack of the first word.
            self._speaking = True
            self._silent_run = 0
            self._voiced = list(self._preroll)
            self._preroll.clear()
            return SegmenterEvent(speech_started=True)

        self._voiced.append(frame)
        self._silent_run = 0 if voiced else self._silent_run + 1

        if self._silent_run >= self.silence_frames:
            return self._finish("silence")

        if len(self._voiced) >= self.max_frames:
            return self._finish("max_duration")

        return None

    def _finish(self, reason: str) -> SegmenterEvent | None:
        frames = self._voiced
        speech_frames = len(frames) - self._silent_run
        self._voiced = []
        self._speaking = False
        self._silent_run = 0

        # Too short to be a real utterance — almost certainly a cough, a door,
        # or the tail of our own TTS leaking into the mic.
        if speech_frames < self.min_speech_frames:
            return None

        return SegmenterEvent(
            utterance=np.concatenate(frames).astype(np.float32), reason=reason
        )

    def force_endpoint(self) -> np.ndarray | None:
        """Client explicitly signalled end-of-speech (push-to-talk release)."""
        if not self._voiced:
            return None
        event = self._finish("client")
        return event.utterance if event else None


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Decode little-endian int16 PCM into float32 in [-1, 1]."""
    if not raw:
        return np.zeros(0, dtype=np.float32)
    # Drop a trailing odd byte rather than raising on a torn frame.
    if len(raw) % 2:
        raw = raw[:-1]
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()
