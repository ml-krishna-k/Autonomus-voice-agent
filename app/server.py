"""FastAPI application — process entrypoint.

Models are loaded once during lifespan startup, before the server reports
ready. `/healthz` answers as soon as the process is up (liveness); `/readyz`
only answers 200 once ASR and TTS are resident (readiness). Point your load
balancer at `/readyz` so traffic is never routed into a cold worker.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .asr import ASREngine
from .config import ROOT, get_settings
from .llm import LLMClient
from .session import SessionStore
from .tts import TTSEngine
from .ws import ConnectionHandler

log = logging.getLogger("app")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)

    log.info("starting Autonomous Voice Agent v%s", __version__)

    app.state.settings = settings
    app.state.sessions = SessionStore(settings)
    app.state.asr = ASREngine(settings)
    app.state.tts = TTSEngine(settings)
    app.state.llm = LLMClient(settings)

    if not settings.openrouter_api_key:
        log.warning("OPENROUTER_API_KEY is not set — turns will fail until it is")

    # Model loads are blocking and slow; run them concurrently off the loop.
    await asyncio.gather(
        asyncio.to_thread(app.state.asr.load),
        asyncio.to_thread(app.state.tts.load),
    )
    await app.state.llm.start()

    sweeper = asyncio.create_task(_sweep_sessions(app))
    log.info("ready on %s:%s", settings.host, settings.port)

    try:
        yield
    finally:
        sweeper.cancel()
        await app.state.llm.aclose()
        log.info("shutdown complete")


async def _sweep_sessions(app: FastAPI) -> None:
    """Evict sessions whose socket died without a clean close."""
    while True:
        try:
            await asyncio.sleep(60)
            if evicted := app.state.sessions.sweep():
                log.info("swept %d stale session(s)", evicted)
        except asyncio.CancelledError:
            break
        except Exception:  # noqa: BLE001
            log.exception("session sweep failed")


app = FastAPI(
    title="Autonomous Voice Agent",
    version=__version__,
    description="Real-time speech-to-speech agent: Whisper ASR, streaming LLM, Piper TTS.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict:
    """Liveness — the process is running."""
    return {"status": "ok", "version": __version__}


@app.get("/readyz", tags=["ops"])
async def readyz() -> JSONResponse:
    """Readiness — models are resident and the agent can serve a turn."""
    settings = app.state.settings
    checks = {
        "asr": app.state.asr.ready,
        "tts": app.state.tts.ready,
        "llm_key": bool(settings.openrouter_api_key),
    }
    ok = all(checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "not_ready", "checks": checks},
    )


@app.get("/info", tags=["ops"])
async def info() -> dict:
    settings = app.state.settings
    return {
        "version": __version__,
        "asr_model": settings.asr_model,
        "llm_model": settings.llm_model,
        "tts_voice": settings.tts_voice,
        "input_sample_rate": settings.input_sample_rate,
        "output_sample_rate": app.state.tts.sample_rate,
        "history_max_turns": settings.history_max_turns,
        "barge_in": settings.allow_barge_in,
        "active_sessions": len(app.state.sessions),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    handler = ConnectionHandler(
        websocket=websocket,
        settings=app.state.settings,
        sessions=app.state.sessions,
        asr=app.state.asr,
        llm=app.state.llm,
        tts=app.state.tts,
    )
    await handler.run()


_TEST_CLIENT = ROOT / "clients" / "web" / "index.html"


@app.get("/", include_in_schema=False)
async def index():
    settings = get_settings()
    if settings.serve_test_client and _TEST_CLIENT.exists():
        return FileResponse(_TEST_CLIENT)
    return JSONResponse(
        {
            "service": "Autonomous Voice Agent",
            "version": __version__,
            "websocket": "/ws",
            "docs": "/docs",
        }
    )


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
