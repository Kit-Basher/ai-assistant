from __future__ import annotations

from typing import Any, Iterable

from agent.runtime_contract import get_effective_llm_identity

def get_public_identity(
    *,
    provider: str | None,
    model: str | None,
    local_providers: Iterable[str] | None = None,
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

    if not known:
        return {
            "provider": provider_id,
            "model": model_id,
            "locality": locality,
            "known": False,
            "reason": reason,
            "summary": (
                "I’m running inside your Personal Agent. The active model is currently unknown. "
                "Current provider/model: unknown / unknown."
            ),
        }

    return {
        "provider": provider_id,
        "model": model_id,
        "locality": locality,
        "known": True,
        "reason": "ok",
        "summary": (
            f"I’m running inside your Personal Agent. "
            f"Current provider/model: {provider_id} / {model_id} ({locality})."
        ),
    }
