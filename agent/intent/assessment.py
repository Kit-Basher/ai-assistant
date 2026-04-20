from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUESTION_STARTS = ("why", "how", "what", "when", "where", "who")
_VAGUE_WORDS = frozenset({"help", "fix", "explain", "why"})

_MODEL_CHECK_PHRASES = (
    "model",
    "ollama",
    "openrouter",
    "new model",
    "recommend model",
    "update model",
)
_MODEL_SWITCH_PHRASES = ("switch to", "use model", "set model", "change model")


@dataclass(frozen=True)
class IntentCandidate:
    intent: str
    score: float
    reason: str
    details: dict[str, Any]


@dataclass(frozen=True)
class IntentAssessment:
    decision: str
    confidence: float
    candidates: list[IntentCandidate]
    next_question: str | None
    debug: dict[str, Any]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round_score(value: float) -> float:
    return round(_clamp01(value), 4)


def _cheap_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _is_question_like(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    if lowered.endswith("?"):
        return True
    return any(lowered.startswith(prefix + " ") for prefix in _QUESTION_STARTS)


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(phrase in lowered for phrase in phrases)


def _decision_from_candidates(candidates: list[IntentCandidate]) -> tuple[str, float, str | None]:
    ordered = sorted(candidates, key=lambda row: (-float(row.score), row.intent))
    top = float(ordered[0].score) if ordered else 0.0
    # Default to the conversational route instead of surfacing internal router
    # choices. The API layer can still ask a natural follow-up for other
    # clarification cases, but intent ambiguity here should not block the user.
    return "proceed", _round_score(top), None


def _candidate_dict(candidates: list[IntentCandidate]) -> list[IntentCandidate]:
    return sorted(candidates, key=lambda row: (-float(row.score), row.intent))


def assess_intent_deterministic(
    *,
    user_text_raw: str,
    user_text_norm: str,
    route_intent: str,
    context: dict[str, Any],
) -> IntentAssessment:
    norm = str(user_text_norm or "").strip()
    lowered = norm.lower()
    route = str(route_intent or "").strip().lower() or "chat"
    tokens = _cheap_tokens(norm)
    question_like = _is_question_like(norm)
    model_check_match = _contains_any(lowered, _MODEL_CHECK_PHRASES)
    model_switch_match = _contains_any(lowered, _MODEL_SWITCH_PHRASES)
    low_confidence_flag = bool(context.get("low_confidence"))
    thread_drift_flag = bool(context.get("thread_integrity"))

    if low_confidence_flag:
        return IntentAssessment(
            decision="low_confidence",
            confidence=0.0,
            candidates=[],
            next_question=None,
            debug={
                "route_intent": route,
                "reason": "low_confidence",
                "token_count": len(tokens),
            },
        )
    if thread_drift_flag:
        return IntentAssessment(
            decision="thread_integrity",
            confidence=0.0,
            candidates=[],
            next_question=None,
            debug={
                "route_intent": route,
                "reason": "thread_integrity",
                "token_count": len(tokens),
            },
        )

    candidates: list[IntentCandidate] = []
    route_score = 0.70
    first_token = tokens[0] if tokens else ""
    if question_like and route == "ask":
        route_score += 0.10
    if (
        tokens
        and (len(tokens) >= 2 or (first_token and first_token not in _VAGUE_WORDS))
        and not (model_check_match or model_switch_match)
    ):
        route_score += 0.10
    candidates.append(
        IntentCandidate(
            intent=route,
            score=_round_score(route_score),
            reason="route_default",
            details={"token_count": len(tokens), "question_like": question_like},
        )
    )

    ask_score = 0.0
    if question_like:
        ask_score = 0.65 + 0.10
    elif lowered.startswith("ask "):
        ask_score = 0.65
    if ask_score > 0:
        candidates.append(
            IntentCandidate(
                intent="ask",
                score=_round_score(ask_score),
                reason="question_like" if question_like else "ask_prefix",
                details={"question_like": question_like},
            )
        )

    if model_check_match:
        matched = [phrase for phrase in _MODEL_CHECK_PHRASES if phrase in lowered]
        candidates.append(
            IntentCandidate(
                intent="modelops_check",
                score=0.85,
                reason="model_keywords",
                details={"matched": sorted(matched)},
            )
        )
    if model_switch_match:
        matched = [phrase for phrase in _MODEL_SWITCH_PHRASES if phrase in lowered]
        candidates.append(
            IntentCandidate(
                intent="modelops_switch",
                score=0.80,
                reason="switch_keywords",
                details={"matched": sorted(matched)},
            )
        )

    # Deduplicate by intent while keeping highest score and deterministic details.
    merged: dict[str, IntentCandidate] = {}
    for row in candidates:
        existing = merged.get(row.intent)
        if existing is None or row.score > existing.score:
            merged[row.intent] = row
    ordered = _candidate_dict(list(merged.values()))
    decision, confidence, next_question = _decision_from_candidates(ordered)
    return IntentAssessment(
        decision=decision,
        confidence=confidence,
        candidates=ordered,
        next_question=next_question,
        debug={
            "route_intent": route,
            "question_like": question_like,
            "model_check_match": model_check_match,
            "model_switch_match": model_switch_match,
            "token_count": len(tokens),
            "raw_length": len(str(user_text_raw or "")),
            "norm_length": len(norm),
        },
    )


def rebuild_assessment_from_candidates(
    *,
    candidates: list[IntentCandidate],
    debug: dict[str, Any] | None = None,
) -> IntentAssessment:
    ordered = _candidate_dict(list(candidates))
    decision, confidence, next_question = _decision_from_candidates(ordered)
    payload_debug = dict(debug or {})
    payload_debug.setdefault("source", "reranked")
    return IntentAssessment(
        decision=decision,
        confidence=confidence,
        candidates=ordered,
        next_question=next_question,
        debug=payload_debug,
    )


__all__ = [
    "IntentAssessment",
    "IntentCandidate",
    "assess_intent_deterministic",
    "rebuild_assessment_from_candidates",
]
