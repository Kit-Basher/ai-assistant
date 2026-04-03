from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import unicodedata


_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BULLET_LINE_RE = re.compile(r"(?m)^\s*[-*•]\s+\S")

_EXPLICIT_SWITCH_PHRASES = (
    "new topic",
    "switch topics",
    "different thing",
    "unrelated",
    "separate question",
    "ignore that",
    "forget that",
    "start over",
    "reset",
    "change subject",
)
_RESET_PHRASES = ("ignore that", "forget that", "start over", "reset")


@dataclass(frozen=True)
class ThreadIntegrityResult:
    is_thread_drift: bool
    reason: str
    message: str
    next_question: str
    debug: dict[str, Any]


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return _WS_RE.sub(" ", normalized).strip()


def cheap_tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b)) / float(len(union))


def has_explicit_switch_phrase(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(phrase in lowered for phrase in _EXPLICIT_SWITCH_PHRASES)


def has_multi_intent_separators(text: str, *, raw_text: str | None = None) -> bool:
    lowered = str(text or "").lower()
    raw = str(raw_text or "")
    return (
        " and also " in lowered
        or " also " in lowered
        or lowered.count("?") >= 2
        or bool(_BULLET_LINE_RE.search(raw))
    )


def _explicit_switch_reason(text: str) -> str:
    lowered = str(text or "").lower()
    if any(phrase in lowered for phrase in _RESET_PHRASES):
        return "reset_request"
    return "explicit_switch"


def detect_thread_drift(
    *,
    user_text_raw: str,
    user_text_norm: str,
    intent: str,
    last_user_text_norm: str | None,
    last_assistant_text_norm: str | None,
) -> ThreadIntegrityResult:
    current_norm = normalize_text(user_text_norm)
    current_tokens = cheap_tokens(current_norm)
    prev_user = normalize_text(last_user_text_norm or "")
    prev_assistant = normalize_text(last_assistant_text_norm or "")
    prev_user_tokens = cheap_tokens(prev_user)
    prev_assistant_tokens = cheap_tokens(prev_assistant)
    similarity_user = jaccard(current_tokens, prev_user_tokens) if prev_user else 0.0
    similarity_assistant = jaccard(current_tokens, prev_assistant_tokens) if prev_assistant else 0.0
    max_similarity = max(similarity_user, similarity_assistant)
    explicit_switch = has_explicit_switch_phrase(current_norm)
    multi_intent = has_multi_intent_separators(current_norm, raw_text=user_text_raw)
    has_prior_context = bool(prev_user or prev_assistant)
    current_length = len(current_norm)

    debug = {
        "intent": str(intent or "").strip().lower() or "chat",
        "current_norm": current_norm,
        "current_length": current_length,
        "current_token_count": len(current_tokens),
        "last_user_present": bool(prev_user),
        "last_assistant_present": bool(prev_assistant),
        "similarity_last_user": round(similarity_user, 4),
        "similarity_last_assistant": round(similarity_assistant, 4),
        "max_similarity": round(max_similarity, 4),
        "has_explicit_switch_phrase": explicit_switch,
        "has_multi_intent_separators": multi_intent,
    }

    if explicit_switch:
        message = "Got it — do you want to start a new thread for this, or should I keep the previous context?"
        reason = _explicit_switch_reason(current_norm)
        return ThreadIntegrityResult(True, reason, message, message, debug)

    if multi_intent and current_length >= 40:
        message = "It looks like you may be switching topics. Do you want to continue the previous thread or start a new one?"
        return ThreadIntegrityResult(True, "multi_intent", message, message, debug)

    if has_prior_context and current_length >= 25 and max_similarity < 0.12:
        message = "It looks like you may be switching topics. Do you want to continue the previous thread or start a new one?"
        return ThreadIntegrityResult(True, "topic_shift", message, message, debug)

    return ThreadIntegrityResult(False, "none", "", "", debug)


__all__ = [
    "ThreadIntegrityResult",
    "cheap_tokens",
    "detect_thread_drift",
    "has_explicit_switch_phrase",
    "has_multi_intent_separators",
    "jaccard",
    "normalize_text",
]
