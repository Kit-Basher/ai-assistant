from __future__ import annotations

import re
from typing import Any

_ALLOWED_SEVERITIES = {"ok", "warn", "bad"}


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return cleaned or "card"


def normalize_card(card: dict[str, Any], index: int) -> dict[str, Any]:
    title = str(card.get("title") or "Card")
    lines_raw = card.get("lines") or []
    lines = [str(line) for line in lines_raw if str(line).strip()]
    severity = str(card.get("severity") or "ok")
    if severity not in _ALLOWED_SEVERITIES:
        severity = "ok"
    details_ref = card.get("details_ref")
    stable_key = str(card.get("key") or f"{_slug(title)}-{index}")
    normalized = {
        "key": stable_key,
        "title": title,
        "lines": lines,
        "severity": severity,
    }
    if details_ref:
        normalized["details_ref"] = str(details_ref)
    return normalized


def validate_cards_payload(payload: dict[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "payload must be an object"
    cards = payload.get("cards")
    raw_available = payload.get("raw_available")
    summary = payload.get("summary")
    confidence = payload.get("confidence")
    next_questions = payload.get("next_questions")
    if not isinstance(cards, list):
        return False, "cards must be a list"
    if not isinstance(raw_available, bool):
        return False, "raw_available must be a bool"
    if not isinstance(summary, str):
        return False, "summary must be a string"
    if not isinstance(confidence, (float, int)):
        return False, "confidence must be a number"
    if float(confidence) < 0.0 or float(confidence) > 1.0:
        return False, "confidence must be between 0 and 1"
    if not isinstance(next_questions, list) or any(not isinstance(item, str) for item in next_questions):
        return False, "next_questions must be a list of strings"
    for idx, card in enumerate(cards):
        if not isinstance(card, dict):
            return False, f"cards[{idx}] must be an object"
        if not isinstance(card.get("title"), str) or not card.get("title").strip():
            return False, f"cards[{idx}].title must be a non-empty string"
        lines = card.get("lines")
        if not isinstance(lines, list) or any(not isinstance(line, str) for line in lines):
            return False, f"cards[{idx}].lines must be a list of strings"
        if card.get("severity") not in _ALLOWED_SEVERITIES:
            return False, f"cards[{idx}].severity must be one of ok|warn|bad"
    return True, None


def render_cards_markdown(payload: dict[str, Any]) -> str:
    cards = payload.get("cards") or []
    chunks: list[str] = []
    summary = str(payload.get("summary") or "").strip()
    confidence = payload.get("confidence")
    next_questions = payload.get("next_questions") or []
    show_confidence = bool(payload.get("show_confidence", True))
    if summary:
        if show_confidence and isinstance(confidence, (float, int)):
            chunks.append(f"{summary} (confidence {float(confidence):.2f})")
        else:
            chunks.append(summary)
    for idx, raw_card in enumerate(cards):
        card = normalize_card(raw_card, idx)
        # Single titled container per card avoids nested-title drop in renderers.
        chunk_lines = [f"*{card['title']}*", *[f"- {line}" for line in card["lines"]]]
        chunks.append("\n".join(chunk_lines).strip())
    if next_questions:
        chunks.append("Follow-ups:\n" + "\n".join([f"- {item}" for item in next_questions]))
    if not chunks:
        return "No cards available."
    return "\n\n".join(chunks)
