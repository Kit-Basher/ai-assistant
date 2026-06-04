from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from agent.actions.managed_action_recovery import ManagedActionJournal
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
    mark_onboarding_completed_reliable(db, user_id, intent_hint=intent_hint)


def mark_onboarding_completed_reliable(db: Any, user_id: str, *, intent_hint: str | None = None) -> dict[str, Any]:
    completed_key = onboarding_completed_key(user_id)
    intent_key = onboarding_intent_hint_key(user_id)
    normalized_hint = _normalize_text(intent_hint)
    user_hash = hashlib.sha256(str(user_id or "").encode("utf-8")).hexdigest()
    previous_completed = _get_user_pref_entry(db, completed_key)
    previous_intent = _get_user_pref_entry(db, intent_key)

    journal = ManagedActionJournal(action_type="onboarding.completed.write", target=f"user:{user_hash}")
    journal.plan_step("preflight_onboarding_state", resource="onboarding_state")
    journal.plan_step("write_onboarding_completed_marker", resource="onboarding_completed")
    journal.plan_step("write_onboarding_intent_hint", resource="user_intent_hint")
    journal.plan_step("verify_onboarding_state", resource="onboarding_state")
    journal.record_step(
        "preflight_onboarding_state",
        ok=True,
        resource="onboarding_state",
        user_hash=user_hash,
        completed_previous=_value_metadata(previous_completed.get("value") if previous_completed else None),
        intent_previous=_value_metadata(previous_intent.get("value") if previous_intent else None),
        intent_requested=_value_metadata(normalized_hint if normalized_hint else None),
    )

    try:
        if not callable(getattr(db, "set_user_pref", None)):
            raise RuntimeError("DB does not support user preferences")
        db.set_user_pref(completed_key, "true")
        journal.record_step("write_onboarding_completed_marker", ok=True, resource="onboarding_completed")
        if normalized_hint:
            db.set_user_pref(intent_key, normalized_hint)
            journal.record_step("write_onboarding_intent_hint", ok=True, resource="user_intent_hint", mode="set")
        elif callable(getattr(db, "delete_user_pref", None)):
            db.delete_user_pref(intent_key)
            journal.record_step("write_onboarding_intent_hint", ok=True, resource="user_intent_hint", mode="delete")
        else:
            journal.record_step("write_onboarding_intent_hint", ok=True, resource="user_intent_hint", mode="none")
    except Exception as exc:
        journal.record_step(
            "write_onboarding_completed_marker",
            ok=False,
            resource="onboarding_state",
            error=exc.__class__.__name__,
        )
        rollback_ok, rollback_summary = _restore_onboarding_state(
            db,
            completed_key,
            previous_completed,
            intent_key,
            previous_intent,
            journal=journal,
        )
        return {
            "ok": False,
            "error": "onboarding_state_write_failed",
            "rollback_ok": rollback_ok,
            "rollback_summary": rollback_summary,
            "message": _onboarding_write_failure_message(rollback_ok, rollback_summary),
            "managed_action_journal": journal.to_dict(),
        }

    state = load_onboarding_state(db, user_id)
    verification_ok = bool(state.get("available") and state.get("completed"))
    if normalized_hint:
        verification_ok = verification_ok and state.get("intent_hint") == normalized_hint
    elif previous_intent is None:
        verification_ok = verification_ok and state.get("intent_hint") is None
    journal.mark_verification(
        ok=verification_ok,
        completed=bool(state.get("completed")),
        intent_present=bool(state.get("intent_hint")),
    )
    if verification_ok:
        journal.record_changed_resource("onboarding_state", f"user:{user_hash}", rollback_supported=True)
        journal.mark_rollback(ok=True, attempted=False, summary="No rollback needed.")
        return {"ok": True, "managed_action_journal": journal.to_dict()}

    rollback_ok, rollback_summary = _restore_onboarding_state(
        db,
        completed_key,
        previous_completed,
        intent_key,
        previous_intent,
        journal=journal,
    )
    return {
        "ok": False,
        "error": "onboarding_state_verification_failed",
        "rollback_ok": rollback_ok,
        "rollback_summary": rollback_summary,
        "message": _onboarding_write_failure_message(rollback_ok, rollback_summary),
        "managed_action_journal": journal.to_dict(),
    }


def _get_user_pref_entry(db: Any, key: str) -> dict[str, Any] | None:
    fn = getattr(db, "get_user_pref_entry", None)
    if callable(fn):
        row = fn(key)
        return dict(row) if isinstance(row, dict) else None
    get_fn = getattr(db, "get_user_pref", None)
    if callable(get_fn):
        value = get_fn(key)
        if value is not None:
            return {"key": key, "value": value}
    return None


def _restore_onboarding_state(
    db: Any,
    completed_key: str,
    previous_completed: dict[str, Any] | None,
    intent_key: str,
    previous_intent: dict[str, Any] | None,
    *,
    journal: ManagedActionJournal,
) -> tuple[bool, str]:
    try:
        _restore_user_pref(db, completed_key, previous_completed)
        _restore_user_pref(db, intent_key, previous_intent)
        summary = "restored previous onboarding state"
        journal.record_rollback_step("restore_onboarding_state", ok=True, resource="onboarding_state", summary=summary)
        journal.mark_rollback(ok=True, attempted=True, summary=summary)
        return True, summary
    except Exception as exc:
        summary = "could not fully restore previous onboarding state"
        journal.record_rollback_step(
            "restore_onboarding_state",
            ok=False,
            resource="onboarding_state",
            error=exc.__class__.__name__,
        )
        journal.mark_rollback(ok=False, attempted=True, summary=summary)
        return False, summary


def _restore_user_pref(db: Any, key: str, previous: dict[str, Any] | None) -> None:
    if previous is None:
        reliable_delete_fn = getattr(db, "delete_user_pref_reliable", None)
        if callable(reliable_delete_fn):
            result = reliable_delete_fn(key)
            if isinstance(result, dict) and result.get("ok") is False:
                raise RuntimeError("onboarding preference rollback delete verification failed")
            return
        delete_fn = getattr(db, "delete_user_pref", None)
        if callable(delete_fn):
            delete_fn(key)
        return
    reliable_set_fn = getattr(db, "set_user_pref_reliable", None)
    if callable(reliable_set_fn):
        result = reliable_set_fn(key, str(previous.get("value") or ""))
        if isinstance(result, dict) and result.get("ok") is False:
            raise RuntimeError("onboarding preference rollback restore verification failed")
        return
    set_fn = getattr(db, "set_user_pref", None)
    if callable(set_fn):
        set_fn(key, str(previous.get("value") or ""))


def _value_metadata(value: Any) -> dict[str, Any]:
    text = str(value or "")
    return {
        "present": value is not None,
        "length": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if value is not None else None,
    }


def _onboarding_write_failure_message(rollback_ok: bool, rollback_summary: str) -> str:
    restored = "The previous setting/state was restored." if rollback_ok else "The previous setting/state may still need attention."
    remaining = str(rollback_summary or "check the onboarding completion state").strip()
    return (
        "The onboarding update did not finish. "
        f"{restored} Attention: {remaining}. "
        "Safe next step: retry onboarding after checking local memory status."
    )


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
    "mark_onboarding_completed_reliable",
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
