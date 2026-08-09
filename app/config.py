"""Environment-driven configuration.

Every tunable lives here so the container can be reconfigured without a rebuild.
Nothing in this module reads the filesystem at import time except `.env`.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, "true" if default else "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _list(name: str, default: str) -> list[str]:
    raw = _str(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


DEFAULT_SYSTEM_PROMPT = """You are Krish, a real-time AI voice assistant for the KrishCommerce application.

Core behavior:
- Speak naturally and concisely, as if talking to a human on a phone.
- Keep responses short, clear, and action-oriented.
- Do NOT use markdown, bullet points, emojis, or long explanations.
- Do NOT reveal internal reasoning to the user.
- If you need to think/reason, you MUST put it in <think> tags at the VERY BEGINNING of the response.
- The user will ONLY hear text outside of <think> tags.

Commerce behavior:
- Help users browse products, track orders, manage carts, payments, and support issues.
- Ask one clarifying question only if absolutely required.
- Prefer confirming actions before executing purchases or cancellations.
- If you don't know something, say so briefly and offer the next best step.

Voice constraints:
- Assume responses are read aloud via TTS.
- Avoid lists longer than three items.
- Avoid technical jargon unless the user is clearly technical.
- Be interruptible: never ramble.

Tone:
- Calm, confident, friendly, professional.
- Indian English neutral accent.
"""


@dataclass(frozen=True)
class Settings:
    # ---- HTTP server -----------------------------------------------------
    host: str = field(default_factory=lambda: _str("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("PORT", 8000))
    cors_origins: list[str] = field(default_factory=lambda: _list("CORS_ORIGINS", "*"))
    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO").upper())
    serve_test_client: bool = field(
        default_factory=lambda: _bool("SERVE_TEST_CLIENT", True)
    )

    # ---- LLM -------------------------------------------------------------
    openrouter_api_key: str = field(
        default_factory=lambda: _str("OPENROUTER_API_KEY", "")
    )
    llm_base_url: str = field(
        default_factory=lambda: _str("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    )
    llm_model: str = field(
        default_factory=lambda: _str("LLM_MODEL", "allenai/olmo-3.1-32b-instruct")
    )
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 400))
    llm_temperature: float = field(
        default_factory=lambda: _float("LLM_TEMPERATURE", 0.6)
    )
    llm_timeout_s: float = field(default_factory=lambda: _float("LLM_TIMEOUT_S", 60.0))
    system_prompt: str = field(
        default_factory=lambda: _str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    )
    # Number of *turns* (user+assistant pairs) retained. Caps token growth.
    history_max_turns: int = field(
        default_factory=lambda: _int("HISTORY_MAX_TURNS", 8)
    )

    # ---- ASR -------------------------------------------------------------
    asr_model: str = field(default_factory=lambda: _str("ASR_MODEL", "distil-small.en"))
    asr_device: str = field(default_factory=lambda: _str("ASR_DEVICE", "cpu"))
    asr_compute_type: str = field(
        default_factory=lambda: _str("ASR_COMPUTE_TYPE", "int8")
    )
    asr_cpu_threads: int = field(default_factory=lambda: _int("ASR_CPU_THREADS", 4))
    asr_beam_size: int = field(default_factory=lambda: _int("ASR_BEAM_SIZE", 1))
    asr_max_concurrency: int = field(
        default_factory=lambda: _int("ASR_MAX_CONCURRENCY", 2)
    )

    # ---- TTS -------------------------------------------------------------
    tts_voice: str = field(default_factory=lambda: _str("TTS_VOICE", "en_US-amy-low"))
    tts_max_concurrency: int = field(
        default_factory=lambda: _int("TTS_MAX_CONCURRENCY", 2)
    )
    tts_length_scale: float = field(
        default_factory=lambda: _float("TTS_LENGTH_SCALE", 1.0)
    )

    # ---- Audio / endpointing --------------------------------------------
    input_sample_rate: int = field(
        default_factory=lambda: _int("INPUT_SAMPLE_RATE", 16000)
    )
    vad_threshold: float = field(default_factory=lambda: _float("VAD_THRESHOLD", 0.01))
    vad_silence_ms: int = field(default_factory=lambda: _int("VAD_SILENCE_MS", 700))
    vad_min_speech_ms: int = field(
        default_factory=lambda: _int("VAD_MIN_SPEECH_MS", 400)
    )
    vad_max_utterance_ms: int = field(
        default_factory=lambda: _int("VAD_MAX_UTTERANCE_MS", 20000)
    )
    vad_preroll_ms: int = field(default_factory=lambda: _int("VAD_PREROLL_MS", 300))
    allow_barge_in: bool = field(default_factory=lambda: _bool("ALLOW_BARGE_IN", True))

    # ---- Sessions --------------------------------------------------------
    session_ttl_s: float = field(default_factory=lambda: _float("SESSION_TTL_S", 1800))
    max_sessions: int = field(default_factory=lambda: _int("MAX_SESSIONS", 200))

    # ---- Paths -----------------------------------------------------------
    models_dir: Path = field(
        default_factory=lambda: Path(_str("MODELS_DIR", str(ROOT / "models"))).resolve()
    )

    @property
    def voice_onnx(self) -> Path:
        return self.models_dir / f"{self.tts_voice}.onnx"

    @property
    def voice_config(self) -> Path:
        return self.models_dir / f"{self.tts_voice}.onnx.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
