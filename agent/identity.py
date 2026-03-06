from __future__ import annotations

from typing import Any, Iterable


def get_public_identity(
    *,
    provider: str | None,
    model: str | None,
    local_providers: Iterable[str] | None = None,
) -> dict[str, Any]:
    provider_id = str(provider or "").strip().lower() or None
    model_id = str(model or "").strip() or None
    local_set = {
        str(item).strip().lower()
        for item in (local_providers or [])
        if str(item).strip()
    }

    if provider_id is None or model_id is None:
        return {
            "provider": provider_id,
            "model": model_id,
            "locality": "unknown",
            "summary": (
                "I’m running inside your Personal Agent. The active model is currently unknown. "
                "Current provider/model: unknown / unknown."
            ),
        }

    locality = "local" if provider_id in local_set else "remote"
    return {
        "provider": provider_id,
        "model": model_id,
        "locality": locality,
        "summary": (
            f"I’m running inside your Personal Agent. "
            f"Current provider/model: {provider_id} / {model_id} ({locality})."
        ),
    }
