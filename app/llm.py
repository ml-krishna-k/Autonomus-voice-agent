"""LLM client — async SSE streaming against any OpenAI-compatible endpoint.

Defaults to OpenRouter, but `LLM_BASE_URL` will point it at vLLM, Ollama,
Together, or a local gateway without code changes.

A single `httpx.AsyncClient` is reused for the process lifetime so connections
and TLS handshakes are pooled across turns — a meaningful slice of
time-to-first-token when the LLM is remote.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from .config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._settings.llm_base_url.rstrip("/"),
            timeout=httpx.Timeout(
                self._settings.llm_timeout_s, connect=10.0, read=self._settings.llm_timeout_s
            ),
            headers=self._headers(),
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def configured(self) -> bool:
        return bool(self._settings.openrouter_api_key)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._settings.openrouter_api_key:
            headers["Authorization"] = f"Bearer {self._settings.openrouter_api_key}"
        # OpenRouter uses these for attribution; harmless elsewhere.
        headers["HTTP-Referer"] = "https://github.com/ml-krishna-k/Autonomus-voice-agent"
        headers["X-Title"] = "Autonomous Voice Agent"
        return headers

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield content deltas as they arrive."""
        if self._client is None:
            raise LLMError("LLM client not started")
        if not self.configured:
            raise LLMError("OPENROUTER_API_KEY is not set")

        payload = {
            "model": self._settings.llm_model,
            "messages": messages,
            "max_tokens": self._settings.llm_max_tokens,
            "temperature": self._settings.llm_temperature,
            "stream": True,
        }

        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                log.error("LLM HTTP %s: %s", response.status_code, body[:500])
                raise LLMError(f"LLM returned HTTP {response.status_code}")

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content
