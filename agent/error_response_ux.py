from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any


_EXAMPLE_PAYLOAD = '{\n  "messages": [\n    {"role": "user", "content": "hello"}\n  ]\n}'


def _iso_utc(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _cooldown_from_state(
    *,
    health_state: dict[str, Any] | None,
    provider: str | None,
    model: str | None,
) -> tuple[str | None, int | None, int | None]:
    state = health_state if isinstance(health_state, dict) else {}
    providers = state.get("providers") if isinstance(state.get("providers"), dict) else {}
    models = state.get("models") if isinstance(state.get("models"), dict) else {}
    provider_id = str(provider or "").strip().lower()
    model_ref = str(model or "").strip()
    model_candidates: list[str] = []
    if model_ref:
        model_candidates.append(model_ref)
        if provider_id and ":" not in model_ref:
            model_candidates.append(f"{provider_id}:{model_ref}")
    model_row = {}
    for candidate in model_candidates:
        row = models.get(candidate) if isinstance(models.get(candidate), dict) else {}
        if row:
            model_row = row
            break
    provider_row = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
    model_cooldown = int(model_row.get("cooldown_until") or 0) if model_row else 0
    provider_cooldown = int(provider_row.get("cooldown_until") or 0) if provider_row else 0
    chosen = model_cooldown if model_cooldown > 0 else provider_cooldown
    if chosen <= 0:
        return provider_id or None, model_candidates[0] if model_candidates else None, None
    return provider_id or None, model_candidates[0] if model_candidates else None, int(chosen)


def upstream_down_message(
    *,
    health_state: dict[str, Any] | None,
    provider: str | None,
    model: str | None,
    now_epoch: int | None = None,
) -> str:
    provider_id, model_id, cooldown_until = _cooldown_from_state(
        health_state=health_state,
        provider=provider,
        model=model,
    )
    if model_id:
        base = f"Model {model_id} is temporarily unavailable."
    elif provider_id:
        base = f"Provider {provider_id} is temporarily unavailable."
    else:
        base = "A model provider is temporarily unavailable."
    if not isinstance(cooldown_until, int) or cooldown_until <= 0:
        return f"{base} Try again after a short delay or switch provider."
    now_value = int(time.time()) if now_epoch is None else int(now_epoch)
    remaining = max(0, int(cooldown_until - now_value))
    cooldown_iso = _iso_utc(cooldown_until)
    return (
        f"{base} Cooldown until {cooldown_iso} ({remaining}s remaining). "
        f"Try again after {cooldown_iso} or switch provider."
    )


def bad_request_next_question(
    *,
    error_message: str,
    json_error: str | None = None,
) -> str | None:
    lowered = str(error_message or "").strip().lower()
    json_issue = str(json_error or "").strip().lower()
    if "messages must be a non-empty list" in lowered:
        return f"Could you resend using this JSON shape?\n{_EXAMPLE_PAYLOAD}"
    if json_issue == "content_type_not_json":
        return (
            "Could you resend with header Content-Type: application/json and this body?\n"
            f"{_EXAMPLE_PAYLOAD}"
        )
    if json_issue == "invalid_json_body":
        return f"Could you resend valid JSON like this?\n{_EXAMPLE_PAYLOAD}"
    return None


def friendly_error_message(
    *,
    error_kind: str,
    current_message: str,
    context: dict[str, Any] | None = None,
    now_epoch: int | None = None,
) -> str:
    kind = str(error_kind or "").strip().lower()
    if kind == "payment_required":
        return (
            "Provider credits or limits blocked this request. Add credits, lower max tokens, or choose a cheaper model."
        )
    if kind != "upstream_down":
        return str(current_message or "").strip()
    data = context if isinstance(context, dict) else {}
    return upstream_down_message(
        health_state=data.get("health_state") if isinstance(data.get("health_state"), dict) else None,
        provider=str(data.get("provider") or "").strip() or None,
        model=str(data.get("model") or "").strip() or None,
        now_epoch=now_epoch,
    )


__all__ = [
    "bad_request_next_question",
    "friendly_error_message",
    "upstream_down_message",
]
