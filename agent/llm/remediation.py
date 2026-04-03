from __future__ import annotations

import re
from typing import Any


_PAYMENT_ERROR_KINDS = {
    "payment_required",
    "credits_insufficient",
    "insufficient_credits",
}


def _normalize_error_kind(value: Any, *, status_code: int | None = None) -> str | None:
    candidate = str(value or "").strip().lower() or None
    if status_code == 402:
        return "payment_required"
    if candidate in _PAYMENT_ERROR_KINDS:
        return "payment_required"
    return candidate


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _slug(text: str) -> str:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return "step"
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return normalized or "step"


def _step_id(index: int, *, kind: str, action: str, subject: str = "") -> str:
    base = f"{kind}_{action}_{subject}".strip("_")
    return f"{index:02d}_{_slug(base)}"


def _provider_map(health_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = health_summary.get("providers") if isinstance(health_summary.get("providers"), list) else []
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        provider_id = str(row.get("id") or "").strip().lower()
        if not provider_id:
            continue
        output[provider_id] = row
    return output


def _model_map(health_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = health_summary.get("models") if isinstance(health_summary.get("models"), list) else []
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        output[model_id] = row
    return output


def _ollama_model_name(raw_model: str) -> str:
    value = str(raw_model or "").strip()
    if not value:
        return ""
    if value.startswith("ollama:"):
        return value.split(":", 1)[1].strip()
    return value


def _choose_best_local_model(
    *,
    registry_models: dict[str, dict[str, Any]],
    health_models: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    local_rows: list[dict[str, Any]] = []
    for model_id, payload in sorted(registry_models.items()):
        if str(payload.get("provider") or "").strip().lower() != "ollama":
            continue
        health = health_models.get(model_id) if isinstance(health_models.get(model_id), dict) else {}
        last_error_kind = str(health.get("last_error_kind") or "").strip().lower()
        if last_error_kind == "model_not_installed":
            continue
        status = str(health.get("status") or "").strip().lower()
        local_rows.append(
            {
                "id": model_id,
                "model": _ollama_model_name(str(payload.get("model") or model_id)),
                "enabled": bool(payload.get("enabled", True)),
                "available": bool(payload.get("available", True)),
                "status": status,
            }
        )
    if not local_rows:
        return None
    local_rows.sort(
        key=lambda row: (
            0 if bool(row.get("enabled")) else 1,
            0 if bool(row.get("available")) else 1,
            0 if str(row.get("status") or "ok") != "down" else 1,
            str(row.get("id") or ""),
        )
    )
    return local_rows[0]


def _has_any_installed_local_model(
    *,
    registry_models: dict[str, dict[str, Any]],
    health_models: dict[str, dict[str, Any]],
) -> bool:
    for model_id, payload in sorted(registry_models.items()):
        if str(payload.get("provider") or "").strip().lower() != "ollama":
            continue
        health = health_models.get(model_id) if isinstance(health_models.get(model_id), dict) else {}
        last_error_kind = str(health.get("last_error_kind") or "").strip().lower()
        if last_error_kind != "model_not_installed":
            return True
    return False


def _remote_model_cost(payload: dict[str, Any]) -> float:
    pricing = payload.get("pricing") if isinstance(payload.get("pricing"), dict) else {}
    price_in = _safe_float(pricing.get("input_per_million_tokens"))
    price_out = _safe_float(pricing.get("output_per_million_tokens"))
    if price_in is None and price_out is None:
        return float("inf")
    return float(price_in or 0.0) + (2.0 * float(price_out or 0.0))


def _choose_cheapest_remote_model(
    *,
    registry_models: dict[str, dict[str, Any]],
    registry_providers: dict[str, dict[str, Any]],
    health_models: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for model_id, payload in sorted(registry_models.items()):
        provider_id = str(payload.get("provider") or "").strip().lower()
        if provider_id in {"", "ollama"}:
            continue
        provider_payload = registry_providers.get(provider_id) if isinstance(registry_providers.get(provider_id), dict) else {}
        if not bool(provider_payload.get("enabled", True)):
            continue
        if not bool(payload.get("enabled", True)):
            continue
        health = health_models.get(model_id) if isinstance(health_models.get(model_id), dict) else {}
        if str(health.get("status") or "").strip().lower() == "down":
            continue
        candidates.append(
            {
                "id": model_id,
                "provider": provider_id,
                "model": str(payload.get("model") or "").strip() or model_id,
                "cost": _remote_model_cost(payload),
            }
        )
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            1 if row.get("cost") == float("inf") else 0,
            float(row.get("cost") or 0.0),
            str(row.get("id") or ""),
        )
    )
    return candidates[0]


def _top_model_watch_local_candidate(latest_batch: dict[str, Any] | None) -> str | None:
    batch = latest_batch if isinstance(latest_batch, dict) else {}
    rows: list[dict[str, Any]] = []
    top_pick = batch.get("top_pick") if isinstance(batch.get("top_pick"), dict) else None
    if isinstance(top_pick, dict):
        rows.append(top_pick)
    candidates = batch.get("candidates") if isinstance(batch.get("candidates"), list) else []
    rows.extend([row for row in candidates if isinstance(row, dict)])
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("id") or "").strip()
        if candidate_id and candidate_id not in deduped:
            deduped[candidate_id] = row
    ordered = sorted(
        deduped.values(),
        key=lambda row: (
            -float(row.get("score") or 0.0),
            str(row.get("id") or ""),
        ),
    )
    for row in ordered:
        provider = str(row.get("provider") or "").strip().lower()
        if provider != "ollama":
            continue
        model = _ollama_model_name(str(row.get("model") or row.get("id") or ""))
        if model:
            return model
    return None


def _add_plan_step(
    *,
    steps: list[dict[str, Any]],
    kind: str,
    action: str,
    reason: str,
    params: dict[str, Any] | None = None,
    instructions: str | None = None,
    safe_to_execute: bool = False,
) -> None:
    subject = ""
    params_obj = params if isinstance(params, dict) else {}
    if "id" in params_obj:
        subject = str(params_obj.get("id") or "")
    elif "default_model" in params_obj:
        subject = str(params_obj.get("default_model") or "")
    elif "model" in params_obj:
        subject = str(params_obj.get("model") or "")
    step_index = len(steps) + 1
    steps.append(
        {
            "id": _step_id(step_index, kind=kind, action=action, subject=subject),
            "kind": kind,
            "action": action,
            "reason": str(reason or "").strip() or "no_reason",
            "safe_to_execute": bool(safe_to_execute),
            "params": dict(sorted(params_obj.items())) if params_obj else {},
            "instructions": str(instructions or "").strip() if instructions else "",
        }
    )


def build_llm_remediation_plan(
    *,
    registry_snapshot: dict[str, Any],
    defaults: dict[str, Any],
    health_summary: dict[str, Any],
    last_error_kind: str | None,
    last_status_code: int | None,
    last_error: str | None,
    safe_mode: dict[str, Any],
    routing_mode: str | None,
    latest_model_watch_batch: dict[str, Any] | None = None,
    ollama_model_fallback: str | None = None,
    target: str | None = None,
    intent: str | None = None,
) -> dict[str, Any]:
    registry = registry_snapshot if isinstance(registry_snapshot, dict) else {}
    providers = registry.get("providers") if isinstance(registry.get("providers"), dict) else {}
    models = registry.get("models") if isinstance(registry.get("models"), dict) else {}
    providers_doc = {
        str(key).strip().lower(): value
        for key, value in providers.items()
        if isinstance(value, dict) and str(key).strip()
    }
    models_doc = {
        str(key).strip(): value
        for key, value in models.items()
        if isinstance(value, dict) and str(key).strip()
    }
    defaults_doc = defaults if isinstance(defaults, dict) else {}
    effective_defaults = {
        "default_provider": str(defaults_doc.get("default_provider") or "").strip().lower() or None,
        "default_model": str(defaults_doc.get("default_model") or "").strip() or None,
        "allow_remote_fallback": bool(defaults_doc.get("allow_remote_fallback", True)),
    }
    routing_mode_value = (
        str(routing_mode or defaults_doc.get("routing_mode") or "auto").strip().lower()
        or "auto"
    )

    health = health_summary if isinstance(health_summary, dict) else {}
    provider_health = _provider_map(health)
    model_health = _model_map(health)

    status_code = _safe_int(last_status_code)
    normalized_last_error_kind = _normalize_error_kind(last_error_kind, status_code=status_code)
    openrouter_health_error = _normalize_error_kind(
        (provider_health.get("openrouter") or {}).get("last_error_kind"),
        status_code=_safe_int((provider_health.get("openrouter") or {}).get("status_code")),
    )
    payment_issue = normalized_last_error_kind == "payment_required" or openrouter_health_error == "payment_required"

    safe_mode_payload = safe_mode if isinstance(safe_mode, dict) else {}
    safe_mode_paused = bool(safe_mode_payload.get("paused"))
    safe_mode_reason = str(safe_mode_payload.get("reason") or "").strip() or "paused"

    reasons: list[str] = []
    if normalized_last_error_kind:
        reasons.append(f"last_error_kind={normalized_last_error_kind}")
    if status_code is not None:
        reasons.append(f"status_code={status_code}")
    if payment_issue:
        reasons.append("payment_issue_detected")
    if safe_mode_paused:
        reasons.append("safe_mode_paused")
    if str(last_error or "").strip():
        reasons.append(f"last_error={str(last_error).strip()[:80]}")

    steps: list[dict[str, Any]] = []

    if safe_mode_paused:
        _add_plan_step(
            steps=steps,
            kind="user_action",
            action="user.unpause_safe_mode",
            reason=f"Safe mode is paused ({safe_mode_reason}).",
            instructions=(
                "Open the Safety section in the UI and unpause automation, or call "
                "POST /llm/autopilot/unpause with {\"confirm\": true}."
            ),
            safe_to_execute=False,
        )

    openrouter_enabled = bool((providers_doc.get("openrouter") or {}).get("enabled", True))
    if payment_issue and openrouter_enabled:
        _add_plan_step(
            steps=steps,
            kind="modelops_action",
            action="modelops.enable_disable_provider_or_model",
            reason="OpenRouter reported payment_required (credits/limit issue).",
            params={"target_type": "provider", "id": "openrouter", "enabled": False},
            safe_to_execute=True,
        )

    ollama_provider_payload = providers_doc.get("ollama") if isinstance(providers_doc.get("ollama"), dict) else {}
    ollama_enabled = bool((ollama_provider_payload or {}).get("enabled", True))
    ollama_health = provider_health.get("ollama") if isinstance(provider_health.get("ollama"), dict) else {}
    ollama_down = str(ollama_health.get("status") or "").strip().lower() == "down"

    best_local = _choose_best_local_model(registry_models=models_doc, health_models=model_health)
    has_installed_local = _has_any_installed_local_model(registry_models=models_doc, health_models=model_health)

    if best_local is not None and not ollama_down:
        if not ollama_enabled:
            _add_plan_step(
                steps=steps,
                kind="modelops_action",
                action="modelops.enable_disable_provider_or_model",
                reason="Local fallback requires ollama provider enabled.",
                params={"target_type": "provider", "id": "ollama", "enabled": True},
                safe_to_execute=True,
            )
        if not bool(best_local.get("enabled")):
            _add_plan_step(
                steps=steps,
                kind="modelops_action",
                action="modelops.enable_disable_provider_or_model",
                reason="Selected local model is disabled.",
                params={"target_type": "model", "id": str(best_local.get("id") or ""), "enabled": True},
                safe_to_execute=True,
            )
        desired_default_model = str(best_local.get("id") or "").strip()
        if (
            effective_defaults.get("default_provider") != "ollama"
            or effective_defaults.get("default_model") != desired_default_model
        ):
            _add_plan_step(
                steps=steps,
                kind="modelops_action",
                action="modelops.set_default_model",
                reason="Set deterministic local default model.",
                params={"default_provider": "ollama", "default_model": desired_default_model},
                safe_to_execute=True,
            )
    else:
        if ollama_down:
            _add_plan_step(
                steps=steps,
                kind="user_action",
                action="user.start_verify_ollama",
                reason="Ollama provider is down.",
                instructions=(
                    "In the UI, open Providers, confirm Ollama base URL is http://127.0.0.1:11434, "
                    "start Ollama, then run provider test again."
                ),
                safe_to_execute=False,
            )
        if not has_installed_local and not ollama_down:
            recommended_pull = (
                _top_model_watch_local_candidate(latest_model_watch_batch)
                or _ollama_model_name(str(ollama_model_fallback or ""))
                or "llama3"
            )
            if not ollama_enabled:
                _add_plan_step(
                    steps=steps,
                    kind="modelops_action",
                    action="modelops.enable_disable_provider_or_model",
                    reason="Enable ollama provider before pulling a local model.",
                    params={"target_type": "provider", "id": "ollama", "enabled": True},
                    safe_to_execute=True,
                )
            _add_plan_step(
                steps=steps,
                kind="modelops_action",
                action="modelops.pull_ollama_model",
                reason="No installed local ollama model was found.",
                params={"model": recommended_pull},
                safe_to_execute=True,
            )
            _add_plan_step(
                steps=steps,
                kind="modelops_action",
                action="modelops.set_default_model",
                reason="Route chat traffic to the pulled local model.",
                params={"default_provider": "ollama", "default_model": f"ollama:{recommended_pull}"},
                safe_to_execute=True,
            )

    if (
        not has_installed_local
        and bool(effective_defaults.get("allow_remote_fallback"))
        and not payment_issue
    ):
        cheaper_remote = _choose_cheapest_remote_model(
            registry_models=models_doc,
            registry_providers=providers_doc,
            health_models=model_health,
        )
        if isinstance(cheaper_remote, dict):
            desired_provider = str(cheaper_remote.get("provider") or "").strip().lower()
            desired_model = str(cheaper_remote.get("id") or "").strip()
            if desired_provider and desired_model:
                if (
                    effective_defaults.get("default_provider") != desired_provider
                    or effective_defaults.get("default_model") != desired_model
                ):
                    _add_plan_step(
                        steps=steps,
                        kind="modelops_action",
                        action="modelops.set_default_model",
                        reason="No local model is currently usable; choose cheapest remote fallback.",
                        params={"default_provider": desired_provider, "default_model": desired_model},
                        safe_to_execute=True,
                    )

    warnings = [
        "Plan is deterministic and does not execute automatically.",
        "Manual confirmation is required to execute safe steps.",
    ]
    if not steps:
        warnings.append("No changes recommended for the current state.")

    return {
        "target": str(target or "").strip() or None,
        "intent": str(intent or "fix_routing").strip().lower() or "fix_routing",
        "plan_only": True,
        "routing_mode": routing_mode_value,
        "safe_mode_paused": safe_mode_paused,
        "reasons": reasons,
        "warnings": warnings,
        "steps": steps,
    }


__all__ = ["build_llm_remediation_plan"]
