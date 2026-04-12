from __future__ import annotations

from typing import Any, Iterable

from agent.runtime_contract import get_effective_llm_identity


def normalize_identity_name(value: str | None) -> str | None:
    cleaned = " ".join(str(value or "").strip().split())
    if not cleaned:
        return None
    return cleaned[:60]


def assistant_identity_label(*, assistant_name: str | None = None) -> str:
    name = normalize_identity_name(assistant_name)
    if name:
        return f"{name}, your Personal Agent"
    return "your Personal Agent"


def user_identity_label(*, user_name: str | None = None) -> str:
    name = normalize_identity_name(user_name)
    return name or "you"


def get_public_identity(
    *,
    provider: str | None,
    model: str | None,
    local_providers: Iterable[str] | None = None,
    assistant_name: str | None = None,
    user_name: str | None = None,
) -> dict[str, Any]:
    identity = get_effective_llm_identity(
        provider=provider,
        model=model,
        local_providers=local_providers,
    )
    provider_id = str(identity.get("provider") or "").strip().lower() or None
    model_id = str(identity.get("model") or "").strip() or None
    locality = str(identity.get("local_remote") or "unknown").strip().lower() or "unknown"
    known = bool(identity.get("known", False))
    reason = str(identity.get("reason") or "unknown_provider_model").strip().lower() or "unknown_provider_model"
    assistant_label = assistant_identity_label(assistant_name=assistant_name)
    user_label = user_identity_label(user_name=user_name)

    if not known:
        return {
            "provider": provider_id,
            "model": model_id,
            "locality": locality,
            "known": False,
            "reason": reason,
            "assistant_label": assistant_label,
            "user_label": user_label,
            "summary": (
                f"I’m {assistant_label}. The active model is currently unknown. "
                "Current provider/model: unknown / unknown."
            ),
        }

    return {
        "provider": provider_id,
        "model": model_id,
        "locality": locality,
        "known": True,
        "reason": "ok",
        "assistant_label": assistant_label,
        "user_label": user_label,
        "summary": (
            f"I’m {assistant_label}. "
            f"Current provider/model: {provider_id} / {model_id} ({locality})."
        ),
    }
