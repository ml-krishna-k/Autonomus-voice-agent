"""Streaming text processing between the LLM and the TTS engine.

Two independent, allocation-light state machines:

`ThinkTagFilter`  strips ``<think>...</think>`` reasoning from a token stream,
                  correctly handling tags split across arbitrary chunk
                  boundaries (``"<thi"`` + ``"nk>"``).

`SentenceChunker` accumulates clean text and emits speakable units as soon as a
                  sentence boundary is seen, so synthesis starts before the LLM
                  has finished generating.
"""

from __future__ import annotations

import re
from typing import Iterable

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


def _longest_partial_suffix(text: str, tags: Iterable[str]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of a tag.

    Used to hold back bytes that might turn out to be the start of a tag once
    the next chunk arrives. Returns 0 when nothing needs holding back.
    """
    best = 0
    for tag in tags:
        # A proper prefix is shorter than the tag itself; a full match would
        # already have been handled by the caller's `find`.
        limit = min(len(tag) - 1, len(text))
        for length in range(limit, 0, -1):
            if length > best and tag.startswith(text[-length:]):
                best = length
                break
    return best


class ThinkTagFilter:
    """Removes reasoning blocks from a streamed token sequence."""

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    @property
    def in_think_block(self) -> bool:
        return self._in_think

    def feed(self, chunk: str) -> str:
        """Consume a chunk, return the portion safe to speak right now."""
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []

        while True:
            if self._in_think:
                idx = self._buf.find(CLOSE_TAG)
                if idx == -1:
                    # Everything so far is reasoning. Retain only what could
                    # still become a closing tag; discard the rest.
                    keep = _longest_partial_suffix(self._buf, (CLOSE_TAG,))
                    self._buf = self._buf[-keep:] if keep else ""
                    break
                self._buf = self._buf[idx + len(CLOSE_TAG) :]
                self._in_think = False
                continue

            open_idx = self._buf.find(OPEN_TAG)
            close_idx = self._buf.find(CLOSE_TAG)

            # An orphaned closing tag (model emitted </think> without <think>):
            # drop the tag, keep the text before it.
            if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
                out.append(self._buf[:close_idx])
                self._buf = self._buf[close_idx + len(CLOSE_TAG) :]
                continue

            if open_idx != -1:
                out.append(self._buf[:open_idx])
                self._buf = self._buf[open_idx + len(OPEN_TAG) :]
                self._in_think = True
                continue

            keep = _longest_partial_suffix(self._buf, (OPEN_TAG, CLOSE_TAG))
            if keep:
                out.append(self._buf[:-keep])
                self._buf = self._buf[-keep:]
            else:
                out.append(self._buf)
                self._buf = ""
            break

        return "".join(out)

    def flush(self) -> str:
        """Return any trailing text held back, and reset state.

        Text buffered *inside* an unterminated think block is discarded — a
        truncated reasoning block must never reach the speaker.
        """
        rest = "" if self._in_think else self._buf
        self._buf = ""
        self._in_think = False
        return rest

    def reset(self) -> None:
        self._buf = ""
        self._in_think = False


# A boundary is terminal punctuation followed by whitespace or end-of-buffer.
# Requiring the trailing space keeps "3.5" and "Rs.499" from being split.
_BOUNDARY_RE = re.compile(r"[.!?;:](?=\s)|[.!?;:]$|\n")


class SentenceChunker:
    """Splits a token stream into speakable sentences."""

    def __init__(self, min_chars: int = 12, max_chars: int = 240) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        self._buf += text
        chunks: list[str] = []

        while True:
            match = None
            for match in _BOUNDARY_RE.finditer(self._buf):
                if match.end() >= self.min_chars:
                    break
                match = None

            if match is not None:
                candidate = self._buf[: match.end()].strip()
                self._buf = self._buf[match.end() :]
                if candidate:
                    chunks.append(candidate)
                continue

            # No boundary in sight but the buffer is getting long — flush at
            # the last space so a run-on sentence still starts playing.
            if len(self._buf) >= self.max_chars:
                cut = self._buf.rfind(" ", 0, self.max_chars)
                if cut <= 0:
                    cut = self.max_chars
                candidate = self._buf[:cut].strip()
                self._buf = self._buf[cut:]
                if candidate:
                    chunks.append(candidate)
                continue

            break

        return chunks

    def flush(self) -> str | None:
        rest = self._buf.strip()
        self._buf = ""
        return rest or None

    def reset(self) -> None:
        self._buf = ""


def strip_think_tags(text: str) -> str:
    """Non-streaming equivalent, for cleaning a complete response."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()
