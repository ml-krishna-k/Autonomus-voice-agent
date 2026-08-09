"""WebSocket transport — the Phase 2 integration surface.

One socket == one conversation. The client streams raw PCM up; the server
streams events and synthesised PCM back. The full message contract is
documented in README.md under "WebSocket API".

Concurrency shape: the receive loop never blocks on model work. Turns run as
detached tasks so that audio arriving mid-response is still segmented, which is
what makes barge-in possible.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect

from .asr import ASREngine
from .config import Settings
from .llm import LLMClient
from .pipeline import ConversationPipeline
from .segmenter import UtteranceSegmenter, pcm16_to_float32
from .session import SessionStore
from .tts import TTSEngine

log = logging.getLogger(__name__)


class ConnectionHandler:
    def __init__(
        self,
        websocket: WebSocket,
        settings: Settings,
        sessions: SessionStore,
        asr: ASREngine,
        llm: LLMClient,
        tts: TTSEngine,
    ) -> None:
        self._ws = websocket
        self._settings = settings
        self._sessions = sessions
        self._tts = tts
        self._session = sessions.create()
        self._send_lock = asyncio.Lock()
        self._turn: asyncio.Task | None = None

        self._segmenter = UtteranceSegmenter(
            sample_rate=settings.input_sample_rate,
            threshold=settings.vad_threshold,
            silence_ms=settings.vad_silence_ms,
            min_speech_ms=settings.vad_min_speech_ms,
            max_utterance_ms=settings.vad_max_utterance_ms,
            preroll_ms=settings.vad_preroll_ms,
        )
        self._pipeline = ConversationPipeline(
            asr=asr, llm=llm, tts=tts, emit=self._send_json, emit_audio=self._send_bytes
        )

    # ---- transport helpers ------------------------------------------------

    async def _send_json(self, payload: dict) -> None:
        async with self._send_lock:
            await self._ws.send_json(payload)

    async def _send_bytes(self, payload: bytes) -> None:
        async with self._send_lock:
            await self._ws.send_bytes(payload)

    # ---- lifecycle --------------------------------------------------------

    async def run(self) -> None:
        await self._ws.accept()
        await self._send_json(
            {
                "type": "ready",
                "session_id": self._session.id,
                "input_sample_rate": self._settings.input_sample_rate,
                "input_format": "pcm_s16le_mono",
                "output_sample_rate": self._tts.sample_rate,
                "barge_in": self._settings.allow_barge_in,
            }
        )
        log.info("session=%s connected", self._session.id)

        try:
            while True:
                message = await self._ws.receive()

                if message["type"] == "websocket.disconnect":
                    break

                if (data := message.get("bytes")) is not None:
                    await self._on_audio(data)
                elif (text := message.get("text")) is not None:
                    await self._on_text(text)

        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            log.exception("session=%s socket error", self._session.id)
        finally:
            await self._cancel_turn()
            self._sessions.drop(self._session.id)
            log.info("session=%s closed after %d turns", self._session.id, self._session.turns)

    # ---- inbound ----------------------------------------------------------

    async def _on_audio(self, raw: bytes) -> None:
        samples = pcm16_to_float32(raw)
        for event in self._segmenter.feed(samples):
            if event.speech_started:
                await self._send_json({"type": "speech_start"})
                if self._settings.allow_barge_in and self._turn_active():
                    log.debug("session=%s barge-in", self._session.id)
                    await self._cancel_turn()
                    await self._send_json({"type": "tts_cancel"})
            if event.utterance is not None:
                await self._send_json({"type": "speech_end", "reason": event.reason})
                await self._start_turn_from_audio(event.utterance)

    async def _on_text(self, raw: str) -> None:
        import json

        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_json({"type": "error", "message": "invalid JSON"})
            return

        kind = message.get("type")

        if kind == "reset":
            await self._cancel_turn()
            self._session.reset()
            self._segmenter.reset()
            await self._send_json({"type": "reset_ok"})

        elif kind == "text":
            # Text-in path: lets Phase 2 integrations drive the agent without
            # audio, and makes the LLM/TTS half testable in isolation.
            content = (message.get("text") or "").strip()
            if content:
                await self._cancel_turn()
                await self._send_json({"type": "transcript", "text": content})
                self._turn = asyncio.create_task(
                    self._pipeline.respond(self._session, content)
                )

        elif kind == "end_audio":
            utterance = self._segmenter.force_endpoint()
            if utterance is not None:
                await self._send_json({"type": "speech_end", "reason": "client"})
                await self._start_turn_from_audio(utterance)

        elif kind == "ping":
            await self._send_json({"type": "pong"})

        else:
            await self._send_json(
                {"type": "error", "message": f"unknown message type: {kind!r}"}
            )

    # ---- turns ------------------------------------------------------------

    async def _start_turn_from_audio(self, pcm) -> None:
        await self._cancel_turn()
        self._turn = asyncio.create_task(self._run_turn(pcm))

    async def _run_turn(self, pcm) -> None:
        try:
            text = await self._pipeline.transcribe(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("session=%s asr failed", self._session.id)
            await self._send_json({"type": "error", "message": f"asr failed: {exc}"})
            return

        if not text:
            await self._send_json({"type": "no_speech"})
            return

        await self._send_json({"type": "transcript", "text": text})
        await self._pipeline.respond(self._session, text)

    def _turn_active(self) -> bool:
        return self._turn is not None and not self._turn.done()

    async def _cancel_turn(self) -> None:
        if not self._turn_active():
            self._turn = None
            return
        assert self._turn is not None
        self._turn.cancel()
        try:
            await self._turn
        except (asyncio.CancelledError, Exception):  # noqa: B014
            pass
        self._turn = None
