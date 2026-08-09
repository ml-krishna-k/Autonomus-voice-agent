"""Turn orchestration: transcript -> LLM -> sentence chunks -> speech.

Two properties matter here and both are easy to lose:

*Pipelining* — synthesis of sentence N overlaps generation of sentence N+1.
Sentences are submitted to the TTS engine as soon as the chunker emits them,
but a drain task awaits them in FIFO order, so audio reaches the client in the
right sequence without waiting for the whole response.

*Cancellation* — a turn is a single asyncio task. Barge-in cancels it, which
unwinds generation, synthesis and playback together. Anything that must survive
cancellation (persisting partial history) happens in a `finally`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from .asr import ASREngine
from .llm import LLMClient, LLMError
from .session import Session
from .textproc import SentenceChunker, ThinkTagFilter
from .tts import TTSEngine

log = logging.getLogger(__name__)

# Emitted to the client; also the contract the web/telephony clients rely on.
Emit = Callable[[dict], Awaitable[None]]
EmitAudio = Callable[[bytes], Awaitable[None]]


class ConversationPipeline:
    def __init__(
        self,
        asr: ASREngine,
        llm: LLMClient,
        tts: TTSEngine,
        emit: Emit,
        emit_audio: EmitAudio,
    ) -> None:
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._emit = emit
        self._emit_audio = emit_audio

    async def transcribe(self, pcm) -> str:
        result = await self._asr.transcribe(pcm)
        return result.text

    async def respond(self, session: Session, user_text: str) -> None:
        """Run one full turn. Safe to cancel at any point."""
        session.add_user(user_text)

        think_filter = ThinkTagFilter()
        chunker = SentenceChunker()
        spoken: list[str] = []

        # Bounded so a fast LLM cannot queue unlimited synthesis work.
        synth_queue: asyncio.Queue[asyncio.Task | None] = asyncio.Queue(maxsize=8)
        drain = asyncio.create_task(self._drain_audio(synth_queue))

        async def submit(sentence: str) -> None:
            spoken.append(sentence)
            await synth_queue.put(asyncio.create_task(self._tts.synthesize(sentence)))

        try:
            await self._emit({"type": "response_start"})

            async for delta in self._llm.stream(session.messages()):
                visible = think_filter.feed(delta)
                if not visible:
                    continue
                await self._emit({"type": "token", "text": visible})
                for sentence in chunker.feed(visible):
                    await submit(sentence)

            tail = think_filter.flush()
            if tail:
                await self._emit({"type": "token", "text": tail})
                for sentence in chunker.feed(tail):
                    await submit(sentence)

            remainder = chunker.flush()
            if remainder:
                await submit(remainder)

            await synth_queue.put(None)
            await drain
            await self._emit({"type": "response_end"})

        except LLMError as exc:
            log.warning("session=%s llm error: %s", session.id, exc)
            await self._cancel_drain(drain, synth_queue)
            await self._emit({"type": "error", "message": str(exc)})
        except asyncio.CancelledError:
            await self._cancel_drain(drain, synth_queue)
            raise
        except Exception as exc:  # noqa: BLE001 - surface, never kill the socket
            log.exception("session=%s turn failed", session.id)
            await self._cancel_drain(drain, synth_queue)
            await self._emit({"type": "error", "message": f"internal error: {exc}"})
        finally:
            # Record whatever was actually said, even on barge-in, so the model
            # knows how far it got before the user cut in.
            session.add_assistant(" ".join(spoken).strip())

    async def _drain_audio(self, queue: asyncio.Queue) -> None:
        """Await synthesis tasks in order and stream their PCM to the client."""
        started = False
        try:
            while True:
                task = await queue.get()
                if task is None:
                    break
                audio = await task
                if not audio:
                    continue
                if not started:
                    started = True
                    await self._emit(
                        {"type": "tts_start", "sample_rate": self._tts.sample_rate}
                    )
                await self._emit_audio(audio)
        finally:
            if started:
                await self._emit({"type": "tts_end"})

    @staticmethod
    async def _cancel_drain(drain: asyncio.Task, queue: asyncio.Queue) -> None:
        drain.cancel()
        try:
            await drain
        except (asyncio.CancelledError, Exception):  # noqa: B014
            pass
        # Cancel any synthesis still queued behind the drain.
        while not queue.empty():
            pending = queue.get_nowait()
            if pending is not None:
                pending.cancel()
