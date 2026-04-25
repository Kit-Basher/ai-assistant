from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_AMBIGUOUS_PHRASES = {
    "can you do it",
    "what do you mean",
    "help",
    "fix it",
    "why",
}
_KNOWN_SMALLTALK = {"ping", "hello", "hi", "/help", "help"}
_NUMERIC_CHOICES = {"1", "2", "3"}
_SMALLTALK_TOKENS = {"ping", "hello", "hi"}


@dataclass(frozen=True)
class AmbiguityVerdict:
    level: str
    reason: str
    normalized_text: str
    token_count: int

    @property
    def ambiguous(self) -> bool:
        return self.level == "AMBIGUOUS"


def normalize_text(text: str | None) -> str:
    return " ".join(str(text or "").strip().lower().split())


def classify_ambiguity(
    text: str | None,
    *,
    known_smalltalk: set[str] | None = None,
) -> AmbiguityVerdict:
    normalized = normalize_text(text)
    tokens = _TOKEN_RE.findall(normalized)
    token_count = len(tokens)
    smalltalk = known_smalltalk if isinstance(known_smalltalk, set) else _KNOWN_SMALLTALK
    if not normalized:
        return AmbiguityVerdict(level="CLEAR", reason="empty", normalized_text=normalized, token_count=token_count)
    if normalized in _NUMERIC_CHOICES:
        return AmbiguityVerdict(level="CLEAR", reason="numeric_choice", normalized_text=normalized, token_count=token_count)
    if normalized in smalltalk:
        return AmbiguityVerdict(level="CLEAR", reason="smalltalk", normalized_text=normalized, token_count=token_count)
    if token_count <= 2 and tokens and tokens[0] in _SMALLTALK_TOKENS:
        return AmbiguityVerdict(
            level="CLEAR",
            reason="smalltalk_variant",
            normalized_text=normalized,
            token_count=token_count,
        )
    if normalized in _AMBIGUOUS_PHRASES:
        return AmbiguityVerdict(level="AMBIGUOUS", reason="known_vague_phrase", normalized_text=normalized, token_count=token_count)
    if token_count <= 2:
        return AmbiguityVerdict(level="CLEAR", reason="short_chat_turn", normalized_text=normalized, token_count=token_count)
    return AmbiguityVerdict(level="CLEAR", reason="clear_enough", normalized_text=normalized, token_count=token_count)


def _clarify_examples(normalized_text: str) -> tuple[str, str]:
    if "fix" in normalized_text:
        return (
            "Fix a provider/model setup issue.",
            "Fix a specific task or error you can paste here.",
        )
    if normalized_text in {"why", "what do you mean"} or "why" in normalized_text:
        return (
            "Explain a recent system/model change.",
            "Explain a specific error you are seeing.",
        )
    if "help" in normalized_text:
        return (
            "Check system and model status.",
            "Help with a concrete task you provide.",
        )
    return (
        "Check current status and recent changes.",
        "Work on a specific task you provide.",
    )


def build_clarify_message(text: str | None) -> str:
    normalized = normalize_text(text)
    option_a, option_b = _clarify_examples(normalized)
    return "\n".join(
        [
            "What do you want me to do right now?",
            "Do you mean:",
            f"A) {option_a}",
            f"B) {option_b}",
        ]
    )


def recovery_options() -> list[dict[str, str]]:
    return [
        {"id": "status", "label": "Show current model/provider status"},
        {"id": "fixit", "label": "Run LLM fix-it wizard"},
        {"id": "brief", "label": "Show a brief system summary"},
    ]


def recovery_constraint_line(
    *,
    availability_reason: str,
) -> str:
    reason = str(availability_reason or "").strip().lower()
    if reason == "safe_mode_paused":
        return "LLM unavailable: safe-mode paused."
    if reason == "tool_required":
        return "LLM unavailable: required tool path is blocked."
    if reason == "provider_unhealthy":
        return "LLM unavailable: active provider is unhealthy."
    if reason == "model_unhealthy":
        return "LLM unavailable: active model is unhealthy."
    return "LLM unavailable right now."


def build_suggest_message(*, availability_reason: str) -> str:
    line = recovery_constraint_line(availability_reason=availability_reason)
    options = recovery_options()
    return "\n".join(
        [
            line,
            f"1) {options[0]['label']}.",
            f"2) {options[1]['label']}.",
            f"3) {options[2]['label']}.",
            "Reply 1, 2, or 3.",
        ]
    )


def parse_recovery_choice(text: str | None, *, options: list[dict[str, Any]] | None = None) -> str | None:
    normalized = normalize_text(text)
    rows = options if isinstance(options, list) else recovery_options()
    if normalized in _NUMERIC_CHOICES:
        idx = int(normalized) - 1
        if 0 <= idx < len(rows):
            row = rows[idx] if isinstance(rows[idx], dict) else {}
            return str(row.get("id") or "").strip() or None
    for row in rows:
        if not isinstance(row, dict):
            continue
        choice_id = str(row.get("id") or "").strip().lower()
        label = normalize_text(str(row.get("label") or ""))
        if normalized == choice_id or (label and normalized == label):
            return str(row.get("id") or "").strip() or None
    return None
