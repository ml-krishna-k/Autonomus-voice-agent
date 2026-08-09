"""Per-connection conversation state.

Replaces the old single `llm/history.json` file, which every caller shared and
raced on. History lives in memory, is keyed by session id, is bounded to a
fixed number of turns, and is evicted on disconnect or TTL expiry.

The turn window is what stops token spend growing quadratically with
conversation length: previously the full transcript was re-sent every turn.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class Session:
    id: str
    system_prompt: str
    max_turns: int
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    history: list[dict[str, str]] = field(default_factory=list)
    turns: int = 0

    def touch(self) -> None:
        self.last_seen = time.monotonic()

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self.touch()

    def add_assistant(self, text: str) -> None:
        if not text:
            return
        self.history.append({"role": "assistant", "content": text})
        self.turns += 1
        self._trim()
        self.touch()

    def _trim(self) -> None:
        # Two messages per turn. Keep the window aligned so it never starts on
        # an assistant message, which some providers reject.
        limit = self.max_turns * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]
            while self.history and self.history[0]["role"] != "user":
                self.history.pop(0)

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}, *self.history]

    def reset(self) -> None:
        self.history.clear()
        self.turns = 0
        self.touch()


class SessionStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, Session] = {}

    def create(self, session_id: str | None = None) -> Session:
        self.sweep()
        if len(self._sessions) >= self._settings.max_sessions:
            oldest = min(self._sessions.values(), key=lambda s: s.last_seen)
            log.warning("session cap reached, evicting %s", oldest.id)
            self._sessions.pop(oldest.id, None)

        sid = session_id or uuid.uuid4().hex[:16]
        session = Session(
            id=sid,
            system_prompt=self._settings.system_prompt,
            max_turns=self._settings.history_max_turns,
        )
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def sweep(self) -> int:
        cutoff = time.monotonic() - self._settings.session_ttl_s
        stale = [s.id for s in self._sessions.values() if s.last_seen < cutoff]
        for sid in stale:
            self._sessions.pop(sid, None)
        return len(stale)

    def __len__(self) -> int:
        return len(self._sessions)
