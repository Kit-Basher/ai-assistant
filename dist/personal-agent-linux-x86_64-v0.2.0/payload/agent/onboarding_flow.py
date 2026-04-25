from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from agent.packs.capability_recommendation import recommend_packs_for_capability, render_pack_capability_response
from agent.persona import normalize_persona_text


ONBOARDING_COMPLETED_KEY_PREFIX = "onboarding_completed"
ONBOARDING_INTENT_HINT_KEY_PREFIX = "user_intent_hint"
ONBOARDING_EXTERNAL_PACK_THRESHOLD = 1

CANONICAL_ONBOARDING_INTENTS: tuple[str, ...] = ("coding", "system", "creative", "general", "not_sure")
INTENT_MAP: dict[str, str | None] = {
    "coding": "dev_tools",
    "system": "system_tools",
    "creative": "creative_tools",
    "general": None,
    "not_sure": None,
}

_ENTRY_TRIGGER_RE = re.compile(r"^(hi|hello|hey)([!.?]+)?$", re.IGNORECASE)
_AFFIRMATIVE_TEXT = {
    "yes",
    "yes please",
    "yep",
    "yeah",
    "sure",
    "ok",
    "okay",
}
_SKIP_TEXT = {
    "skip",
    "no",
    "no thanks",
    "not now",
    "later",
    "pass",
}
_INTENT_PHRASE_TABLE: dict[str, tuple[str, ...]] = {
    "coding": (
        "coding",
        "code",
        "programming",
        "software",
        "dev",
        "development",
        "build apps",
        "scripts",
        "python help",
        "coding help",
    ),
    "system": (
        "system",
        "pc help",
        "computer help",
        "system tasks",
        "pc tasks",
        "linux help",
        "windows help",
        "setup help",
        "fix my pc",
        "terminal stuff",
        "drivers",
        "hardware help",
    ),
    "creative": (
        "creative",
        "writing",
        "creative writing",
        "art",
        "brainstorming",
        "worldbuilding",
        "story help",
        "ideas",
        "content writing",
    ),
    "general": (
        "general",
        "general use",
        "general help",
        "everyday stuff",
        "normal use",
        "a bit of everything",
    ),
    "not_sure": (
        "not sure",
        "unsure",
        "dont know",
        "don't know",
        "whatever",
        "anything really",
        "anything",
    ),
}
_INTENT_PRIORITY: tuple[str, ...] = ("coding", "system", "creative", "general", "not_sure")


def _normalize_text(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("’", "'").replace("'", "")
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _normalized_tokens(value: str | None) -> tuple[str, ...]:
    cleaned = _normalize_text(value)
    return tuple(cleaned.split()) if cleaned else ()


def _phrase_matches(cleaned: str, tokens: tuple[str, ...], phrase: str) -> bool:
    phrase_clean = _normalize_text(phrase)
    if not phrase_clean:
        return False
    if cleaned == phrase_clean or phrase_clean in cleaned:
        return True
    phrase_tokens = tuple(phrase_clean.split())
    if not phrase_tokens:
        return False
    token_set = set(tokens)
    return all(token in token_set for token in phrase_tokens)


def onboarding_completed_key(user_id: str) -> str:
    return f"{ONBOARDING_COMPLETED_KEY_PREFIX}:{str(user_id or '').strip()}"


def onboarding_intent_hint_key(user_id: str) -> str:
    return f"{ONBOARDING_INTENT_HINT_KEY_PREFIX}:{str(user_id or '').strip()}"


def is_onboarding_entry_trigger(text: str | None) -> bool:
    cleaned = _normalize_text(text)
    return bool(cleaned and _ENTRY_TRIGGER_RE.match(cleaned))


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = _normalize_text(str(value or ""))
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def load_onboarding_state(db: Any, user_id: str) -> dict[str, Any]:
    completed_key = onboarding_completed_key(user_id)
    intent_key = onboarding_intent_hint_key(user_id)
    try:
        completed_raw = db.get_user_pref(completed_key) if callable(getattr(db, "get_user_pref", None)) else None
        intent_hint = db.get_user_pref(intent_key) if callable(getattr(db, "get_user_pref", None)) else None
    except Exception as exc:
        logger.warning("Failed to load onboarding state: %s", exc)
        return {
            "available": False,
            "completed": True,
            "intent_hint": None,
            "ambiguous": True,
        }

    parsed = _parse_bool(completed_raw)
    intent_text = _normalize_text(intent_hint)
    if intent_text:
        return {
            "available": True,
            "completed": True,
            "intent_hint": intent_text,
            "ambiguous": False,
        }
    if parsed is True:
        return {
            "available": True,
            "completed": True,
            "intent_hint": None,
            "ambiguous": False,
        }
    if parsed is False and str(completed_raw or "").strip().lower() in {"0", "false", "no", "off"}:
        return {
            "available": True,
            "completed": False,
            "intent_hint": None,
            "ambiguous": False,
        }
    if completed_raw is None:
        return {
            "available": True,
            "completed": False,
            "intent_hint": None,
            "ambiguous": False,
        }
    return {
        "available": True,
        "completed": True,
        "intent_hint": None,
        "ambiguous": True,
    }


def onboarding_is_completed(db: Any, user_id: str) -> bool:
    return bool(load_onboarding_state(db, user_id).get("completed", False))


def mark_onboarding_completed(db: Any, user_id: str, *, intent_hint: str | None = None) -> None:
    completed_key = onboarding_completed_key(user_id)
    intent_key = onboarding_intent_hint_key(user_id)
    if callable(getattr(db, "set_user_pref", None)):
        db.set_user_pref(completed_key, "true")
        if str(intent_hint or "").strip():
            db.set_user_pref(intent_key, _normalize_text(intent_hint))
        elif callable(getattr(db, "delete_user_pref", None)):
            db.delete_user_pref(intent_key)


def should_offer_onboarding(db: Any, pack_store: Any, user_id: str) -> bool:
    state = load_onboarding_state(db, user_id)
    if not bool(state.get("available", True)):
        return False
    if bool(state.get("completed", False)):
        return False
    list_external_packs = getattr(pack_store, "list_external_packs", None)
    if not callable(list_external_packs):
        return False
    try:
        return len(list_external_packs()) < ONBOARDING_EXTERNAL_PACK_THRESHOLD
    except Exception:
        return False


def resolve_onboarding_intent(text: str | None) -> str | None:
    cleaned = _normalize_text(text)
    if not cleaned:
        return None
    if "not sure" in cleaned:
        return "not sure"
    if "general use" in cleaned:
        return "general use"
    for intent, phrases in _INTENT_PATTERNS:
        if intent == "general use":
            continue
        if any(phrase and phrase in cleaned for phrase in phrases):
            return intent
    if any(phrase and phrase in cleaned for phrase in ("general", "anything", "everything", "unsure")):
        return "general use"
    return None


def onboarding_intent_to_capability(intent: str | None) -> str | None:
    return INTENT_MAP.get(_normalize_text(intent))


def onboarding_intent_label(intent: str | None) -> str:
    normalized = _normalize_text(intent)
    if normalized == "coding":
        return "coding"
    if normalized == "system":
        return "system / PC tasks"
    if normalized == "creative":
        return "creative / writing"
    if normalized == "general use":
        return "general use"
    if normalized == "not sure":
        return "not sure"
    return "that"


def onboarding_entry_prompt() -> str:
    return normalize_persona_text(
        "I’m ready. Want me to tailor suggestions to how you’ll use this? Say yes, skip, or just ask anything."
    )


def onboarding_intent_prompt() -> str:
    return normalize_persona_text(
        "What do you mainly want help with? Coding, system / PC tasks, creative / writing, general use, or not sure?"
    )


def onboarding_skip_response() -> str:
    return normalize_persona_text("No problem — just ask me anything. I’ll suggest capabilities when relevant.")


def onboarding_decline_response() -> str:
    return normalize_persona_text("Got it. I’ll keep that in mind and suggest things when relevant.")


def normalize_onboarding_intent(text: str | None) -> str | None:
    cleaned = _normalize_text(text)
    tokens = _normalized_tokens(cleaned)
    if not cleaned:
        return None
    if cleaned in CANONICAL_ONBOARDING_INTENTS:
        return cleaned
    if cleaned in {"general use"}:
        return "general"
    if cleaned in {"not sure", "not_sure"}:
        return "not_sure"
    matches: list[str] = []
    for intent in _INTENT_PRIORITY:
        if any(_phrase_matches(cleaned, tokens, phrase) for phrase in _INTENT_PHRASE_TABLE.get(intent, ())):
            matches.append(intent)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        if "general" in matches:
            return "general"
        if "not_sure" in matches:
            return "not_sure"
        return "general"
    return None


def recommend_onboarding_capability(
    intent: str | None,
    *,
    pack_store: Any,
    pack_registry_discovery: Any,
) -> dict[str, Any] | None:
    capability = onboarding_intent_to_capability(intent)
    if capability is None:
        return None
    return recommend_packs_for_capability(
        None,
        pack_store=pack_store,
        pack_registry_discovery=pack_registry_discovery,
        capability=capability,
    )


def render_onboarding_recommendation(intent: str | None, recommendation: dict[str, Any] | None) -> str:
    if not isinstance(recommendation, dict):
        return onboarding_decline_response()
    intent_label = onboarding_intent_label(intent)
    intro = normalize_persona_text(f"I can add capabilities for {intent_label}.")
    rendered = render_pack_capability_response(recommendation)
    if not rendered:
        return intro
    return normalize_persona_text(f"{intro} {rendered}")


def resolve_onboarding_intent(text: str | None) -> str | None:
    return normalize_onboarding_intent(text)


def classify_onboarding_reply(text: str | None, *, stage: str) -> dict[str, Any]:
    cleaned = _normalize_text(text)
    normalized_stage = _normalize_text(stage)
    if normalized_stage == "entry":
        if cleaned in _SKIP_TEXT:
            return {"kind": "skip"}
        if cleaned in _AFFIRMATIVE_TEXT:
            return {"kind": "yes"}
        intent = normalize_onboarding_intent(cleaned)
        if intent is not None:
            return {"kind": "intent", "intent": intent}
        return {"kind": "normal"}
    if normalized_stage == "intent":
        if cleaned in _SKIP_TEXT:
            return {"kind": "skip"}
        intent = normalize_onboarding_intent(cleaned)
        if intent is not None:
            return {"kind": "intent", "intent": intent}
        return {"kind": "normal"}
    return {"kind": "normal"}


__all__ = [
    "INTENT_MAP",
    "classify_onboarding_reply",
    "is_onboarding_entry_trigger",
    "load_onboarding_state",
    "mark_onboarding_completed",
    "onboarding_completed_key",
    "onboarding_decline_response",
    "onboarding_entry_prompt",
    "onboarding_intent_hint_key",
    "onboarding_intent_label",
    "onboarding_intent_prompt",
    "onboarding_intent_to_capability",
    "onboarding_is_completed",
    "onboarding_skip_response",
    "recommend_onboarding_capability",
    "render_onboarding_recommendation",
    "normalize_onboarding_intent",
    "resolve_onboarding_intent",
    "should_offer_onboarding",
]
