from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import unicodedata


_WS_RE = re.compile(r"\s+")
_BIG_REPEAT_RE = re.compile(r"(.)\1{6,}")
_ACK_WORDS = frozenset(
    {
        "k",
        "kk",
        "no",
        "nope",
        "ok",
        "okay",
        "sure",
        "thanks",
        "thx",
        "y",
        "yes",
    }
)

_Q_EMPTY = "What do you want to do? (Example: “summarize this”, “plan my day”, “debug this error”)"
_Q_NO_SEMANTIC = "Can you say what you want in a sentence or two? What’s the goal?"
_Q_TOO_SHORT = "I’m not sure what you want yet — what should I help you with?"
_Q_REPEAT = "I can help — what are you trying to do right now? (One specific task.)"
_Q_UNDERSPEC = "What should I do with that? Give me a bit more detail or the exact input."


@dataclass(frozen=True)
class LowConfidenceResult:
    is_low_confidence: bool
    reason: str
    next_question: str
    debug: dict[str, Any]


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return _WS_RE.sub(" ", normalized).strip()


def detect_low_confidence(text: str) -> LowConfidenceResult:
    norm = _normalize(text)
    words = [piece for piece in norm.split(" ") if piece] if norm else []
    words_lower = [piece.lower() for piece in words]
    length = len(norm)
    word_count = len(words)
    unique_word_count = len(set(words_lower))
    has_letter_or_digit = any(char.isalnum() for char in norm)
    is_punct_only = bool(norm) and not has_letter_or_digit
    has_big_repeat = bool(_BIG_REPEAT_RE.search(norm))
    is_bare_ack = bool(words_lower) and word_count <= 2 and all(piece in _ACK_WORDS for piece in words_lower)

    debug = {
        "norm": norm,
        "length": length,
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "has_letter_or_digit": has_letter_or_digit,
        "is_punct_only": is_punct_only,
        "has_big_repeat": has_big_repeat,
    }

    if not norm:
        return LowConfidenceResult(True, "empty", _Q_EMPTY, debug)
    if not has_letter_or_digit or is_punct_only:
        return LowConfidenceResult(True, "no_semantic_tokens", _Q_NO_SEMANTIC, debug)
    if length <= 3 or is_bare_ack:
        return LowConfidenceResult(True, "too_short", _Q_TOO_SHORT, debug)
    if has_big_repeat:
        return LowConfidenceResult(True, "repetition_spam", _Q_REPEAT, debug)
    if length < 12 and word_count <= 2 and unique_word_count <= 2:
        return LowConfidenceResult(True, "underspecified", _Q_UNDERSPEC, debug)
    return LowConfidenceResult(False, "confident", "", debug)


__all__ = ["LowConfidenceResult", "detect_low_confidence"]
