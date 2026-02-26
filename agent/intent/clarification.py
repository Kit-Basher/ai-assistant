from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_VAGUE_VERBS = frozenset({"help", "fix", "explain", "why"})

_MSG_EMPTY = "What should I help you with? (e.g., “debug an error”, “summarize text”, “plan a task”)"
_MSG_NO_SEMANTIC = "I didn’t catch a request there — what are you trying to do?"
_MSG_REPEAT = "I can help — what’s the specific task or problem?"
_MSG_VAGUE = "Sure — what should I help with specifically? (Paste the text/error and your goal.)"


@dataclass(frozen=True)
class ClarificationPlan:
    message: str
    next_question: str
    reason: str
    hints: list[str]
    suggested_intents: list[str]
    debug: dict[str, Any]


def build_clarification_plan(
    *,
    raw_text: str,
    norm_text: str,
    detector_reason: str,
    intent: str,
) -> ClarificationPlan:
    normalized_reason = str(detector_reason or "").strip().lower() or "unknown"
    normalized_intent = str(intent or "").strip().lower() or "chat"
    normalized_text = str(norm_text or "").strip()
    tokens = [piece for piece in normalized_text.lower().split(" ") if piece]

    if normalized_reason == "empty":
        message = _MSG_EMPTY
        hints = ["Tell me the goal and paste any relevant text/logs."]
    elif normalized_reason == "no_semantic_tokens":
        message = _MSG_NO_SEMANTIC
        hints = ["One sentence is enough.", "If it’s an error, paste the message/log."]
    elif normalized_reason == "repetition_spam":
        message = _MSG_REPEAT
        hints = ["Describe what you expected and what happened instead."]
    elif normalized_reason in {"too_short", "underspecified"}:
        if len(tokens) == 1 and tokens[0] in _VAGUE_VERBS:
            message = _MSG_VAGUE
        else:
            display_text = normalized_text or str(raw_text or "").strip() or "that"
            message = f"What should I do with: “{display_text}”?"
        hints = ["Tell me the desired outcome.", "Include any inputs or constraints."]
    else:
        # Defensive fallback to keep this deterministic for unexpected reasons.
        message = _MSG_VAGUE
        hints = ["Tell me the desired outcome.", "Include any inputs or constraints."]
        normalized_reason = "unknown"

    return ClarificationPlan(
        message=message,
        next_question=message,
        reason=normalized_reason,
        hints=hints,
        suggested_intents=[normalized_intent],
        debug={
            "raw_text": str(raw_text or ""),
            "norm_text": normalized_text,
            "detector_reason": normalized_reason,
            "intent": normalized_intent,
            "token_count": len(tokens),
            "is_vague_verb": bool(len(tokens) == 1 and tokens[0] in _VAGUE_VERBS),
        },
    )


__all__ = ["ClarificationPlan", "build_clarification_plan"]
