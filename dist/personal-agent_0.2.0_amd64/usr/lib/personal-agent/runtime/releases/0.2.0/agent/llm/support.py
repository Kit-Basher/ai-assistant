from __future__ import annotations

import json
import re
from typing import Any
import urllib.parse

from agent.audit_log import redact as redact_audit_value
from agent.llm.notifications import sanitize_notification_text


_SECRET_KEY_HINTS = {
    "api_key",
    "authorization",
    "auth",
    "password",
    "passphrase",
    "secret",
    "token",
}
_URL_FIELD_HINTS = {"base_url", "url", "endpoint"}
_REDACTED_VALUE = "[REDACTED]"

_PROVIDER_CAUSE_ORDER = {
    "provider_not_found": 0,
    "provider_disabled": 1,
    "bad_base_url": 2,
    "misconfigured_path": 3,
    "missing_auth": 4,
    "auth_error": 5,
    "rate_limit": 6,
    "server_error": 7,
    "provider_down": 8,
    "provider_degraded": 9,
    "repeated_failures": 10,
    "provider_healthy": 99,
}

_MODEL_CAUSE_ORDER = {
    "model_not_found": 0,
    "provider_not_found": 1,
    "provider_disabled": 2,
    "provider_down": 3,
    "provider_degraded": 4,
    "model_disabled": 5,
    "model_unavailable": 6,
    "model_not_chat_capable": 7,
    "model_missing_from_catalog": 8,
    "model_unroutable": 9,
    "model_health_down": 10,
    "model_health_degraded": 11,
    "model_in_cooldown": 12,
    "model_not_applicable_for_chat": 13,
    "model_healthy": 99,
}

_ROOT_CAUSE_ACTIONS: dict[str, list[str]] = {
    "provider_disabled": [
        "Enable the provider in Setup before routing requests to it.",
    ],
    "bad_base_url": [
        "Fix provider base_url to a valid http(s) URL and rerun /llm/health/run.",
    ],
    "misconfigured_path": [
        "Fix provider chat_path to the correct OpenAI-compatible path and rerun /llm/health/run.",
    ],
    "missing_auth": [
        "Configure provider credentials in the Providers UI and rerun /llm/health/run.",
    ],
    "auth_error": [
        "Verify provider credentials and account access, then rerun /llm/health/run.",
    ],
    "rate_limit": [
        "Wait for provider rate limits to reset, or reduce request volume.",
    ],
    "server_error": [
        "Retry after provider service recovery and rerun /llm/health/run.",
    ],
    "provider_down": [
        "Run /llm/health/run to refresh status after provider recovery.",
    ],
    "provider_degraded": [
        "Keep a local fallback enabled until provider health returns to ok.",
    ],
    "repeated_failures": [
        "Run /llm/cleanup/plan to review failing-provider suppression options.",
    ],
    "model_disabled": [
        "Enable the model in Setup if it should be routable.",
    ],
    "model_unavailable": [
        "Refresh inventory with /models/refresh and rerun /llm/health/run.",
    ],
    "model_not_chat_capable": [
        "Select a chat-capable model for defaults or routing.",
    ],
    "model_missing_from_catalog": [
        "Run /llm/catalog/run to refresh provider catalog metadata.",
    ],
    "model_unroutable": [
        "Run /llm/self_heal/plan to propose a routable default replacement.",
    ],
    "model_health_down": [
        "Wait for cooldown and rerun /llm/health/run before retrying this model.",
    ],
    "model_health_degraded": [
        "Prefer another healthy model until this model stabilizes.",
    ],
    "model_in_cooldown": [
        "Wait for cooldown expiry or run /llm/health/run later.",
    ],
    "model_not_applicable_for_chat": [
        "Use this model for embeddings only; pick a chat-capable model for conversation.",
    ],
}

_INTENT_STEPS: dict[str, list[dict[str, str]]] = {
    "fix_routing": [
        {
            "id": "refresh_inventory",
            "action": "POST /models/refresh",
            "reason": "refresh provider inventory before routing decisions",
        },
        {
            "id": "refresh_catalog",
            "action": "POST /llm/catalog/run",
            "reason": "sync catalog capabilities and metadata",
        },
        {
            "id": "run_health",
            "action": "POST /llm/health/run",
            "reason": "update provider/model health and cooldown state",
        },
        {
            "id": "plan_reconcile",
            "action": "POST /llm/capabilities/reconcile/plan",
            "reason": "repair capability mismatches before applying changes",
        },
        {
            "id": "plan_self_heal",
            "action": "POST /llm/self_heal/plan",
            "reason": "repair default drift if current defaults are unroutable",
        },
        {
            "id": "plan_autoconfig",
            "action": "POST /llm/autoconfig/plan",
            "reason": "confirm final deterministic default selection",
        },
    ],
    "reduce_churn": [
        {
            "id": "check_health",
            "action": "GET /llm/health",
            "reason": "inspect drift and safe mode signals",
        },
        {
            "id": "check_ledger",
            "action": "GET /llm/autopilot/ledger?limit=20",
            "reason": "verify recent apply cadence and flip-flop behavior",
        },
        {
            "id": "plan_cleanup",
            "action": "POST /llm/cleanup/plan",
            "reason": "review stale/unroutable candidates causing churn",
        },
        {
            "id": "plan_hygiene",
            "action": "POST /llm/hygiene/plan",
            "reason": "review provider/model disable candidates",
        },
    ],
    "bootstrap": [
        {
            "id": "run_health",
            "action": "POST /llm/health/run",
            "reason": "ensure candidate status is current",
        },
        {
            "id": "plan_bootstrap",
            "action": "POST /llm/autopilot/bootstrap",
            "reason": "deterministically select a healthy local chat default",
        },
        {
            "id": "verify_defaults",
            "action": "GET /defaults",
            "reason": "confirm final default provider/model values",
        },
    ],
}


def _looks_secret_key(key: str) -> bool:
    lowered = str(key or "").strip().lower()
    return any(token in lowered for token in _SECRET_KEY_HINTS)


def _sanitize_url(raw: str) -> str:
    value = sanitize_notification_text(str(raw or "").strip())
    if not value:
        return value
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc:
        return value

    host = str(parsed.hostname or "").strip()
    if not host:
        return value
    port = f":{int(parsed.port)}" if parsed.port else ""
    netloc = f"{host}{port}"
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_query_pairs: list[tuple[str, str]] = []
    for raw_key, raw_value in query_pairs:
        key = str(raw_key)
        if _looks_secret_key(key):
            safe_query_pairs.append((key, _REDACTED_VALUE))
        else:
            safe_query_pairs.append((key, sanitize_notification_text(str(raw_value))))
    safe_query = urllib.parse.urlencode(safe_query_pairs)
    rebuilt = urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, safe_query, parsed.fragment))
    return sanitize_notification_text(rebuilt)


def _sanitize_string(value: str, *, key_hint: str = "") -> str:
    if _looks_secret_key(key_hint):
        return _REDACTED_VALUE
    hint = str(key_hint or "").strip().lower()
    if hint in _URL_FIELD_HINTS or hint.endswith("_url"):
        return _sanitize_url(value)
    cleaned = sanitize_notification_text(value)
    lowered = cleaned.lower()
    if "secrets.enc" in lowered or "secret_store" in lowered:
        return _REDACTED_VALUE
    return cleaned


def sanitize_support_payload(value: Any, *, key_hint: str = "") -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda item: str(item)):
            key = str(raw_key)
            item = value.get(raw_key)
            if _looks_secret_key(key):
                output[key] = _REDACTED_VALUE
                continue
            if key == "api_key_source":
                source = item if isinstance(item, dict) else {}
                output[key] = {
                    "configured": bool(source.get("configured")),
                    "type": str(source.get("type") or "").strip() or None,
                }
                continue
            output[key] = sanitize_support_payload(item, key_hint=key)
        return output
    if isinstance(value, list):
        return [sanitize_support_payload(item, key_hint=key_hint) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value, key_hint=key_hint)
    return value


def _ordered(values: set[str], order_map: dict[str, int]) -> list[str]:
    return sorted(
        {str(item).strip() for item in values if str(item).strip()},
        key=lambda item: (int(order_map.get(item, 500)), item),
    )


def _actions_for_root_causes(root_causes: list[str]) -> list[str]:
    actions: list[str] = []
    for cause in root_causes:
        for action in _ROOT_CAUSE_ACTIONS.get(cause, []):
            if action not in actions:
                actions.append(action)
    if not actions:
        actions.append("No immediate remediation is required from current local evidence.")
    return actions


def build_provider_diagnosis(
    *,
    provider_id: str,
    provider_payload: dict[str, Any] | None,
    provider_health: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    related_models: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    payload = provider_payload if isinstance(provider_payload, dict) else {}
    health = provider_health if isinstance(provider_health, dict) else {}
    validation_payload = validation if isinstance(validation, dict) else {}
    status = str(health.get("status") or "unknown").strip().lower() or "unknown"
    last_error_kind = str(health.get("last_error_kind") or "").strip().lower() or None
    status_code = health.get("status_code")
    failure_streak = int(health.get("failure_streak") or 0)

    root_causes: set[str] = set()
    if not payload:
        root_causes.add("provider_not_found")
    else:
        if not bool(payload.get("enabled", True)):
            root_causes.add("provider_disabled")
        if not bool(validation_payload.get("ok", True)):
            root_causes.add(str(validation_payload.get("error_kind") or "misconfigured_path").strip().lower())
        if status == "down":
            root_causes.add("provider_down")
        elif status == "degraded":
            root_causes.add("provider_degraded")
        if failure_streak >= 3 and status in {"down", "degraded"}:
            root_causes.add("repeated_failures")
        if last_error_kind:
            root_causes.add(last_error_kind)

    ordered_causes = _ordered(root_causes, _PROVIDER_CAUSE_ORDER)
    if not ordered_causes:
        ordered_causes = ["provider_healthy"]

    diagnosis = {
        "status": status,
        "last_error_kind": last_error_kind,
        "status_code": status_code,
        "failure_streak": failure_streak,
        "root_causes": ordered_causes,
        "recommended_actions": _actions_for_root_causes(ordered_causes),
        "evidence": {
            "provider": {
                "id": str(provider_id or "").strip().lower(),
                "enabled": bool(payload.get("enabled", False)),
                "local": bool(payload.get("local", False)),
                "base_url": payload.get("base_url"),
                "chat_path": payload.get("chat_path"),
                "api_key_source": payload.get("api_key_source"),
            },
            "health": {
                "status": status,
                "last_error_kind": last_error_kind,
                "status_code": status_code,
                "failure_streak": failure_streak,
                "cooldown_until_iso": health.get("cooldown_until_iso"),
            },
            "validation": {
                "ok": bool(validation_payload.get("ok", False)),
                "error_kind": str(validation_payload.get("error_kind") or "").strip() or None,
                "message": str(validation_payload.get("message") or "").strip() or None,
                "details": validation_payload.get("details") if isinstance(validation_payload.get("details"), dict) else {},
            },
            "related_models": sorted(
                [
                    {
                        "id": str(row.get("id") or "").strip(),
                        "status": str(row.get("status") or "").strip().lower() or "unknown",
                        "last_error_kind": str(row.get("last_error_kind") or "").strip().lower() or None,
                        "routable": bool(row.get("routable", False)),
                    }
                    for row in (related_models or [])
                    if isinstance(row, dict) and str(row.get("id") or "").strip()
                ],
                key=lambda item: str(item.get("id") or ""),
            )[:10],
        },
    }
    return sanitize_support_payload(redact_audit_value(diagnosis))


def build_model_diagnosis(
    *,
    model_id: str,
    model_payload: dict[str, Any] | None,
    model_health: dict[str, Any] | None,
    model_snapshot: dict[str, Any] | None,
    provider_payload: dict[str, Any] | None,
    provider_health: dict[str, Any] | None,
    catalog_entry: dict[str, Any] | None,
    provider_catalog_ids: list[str] | None = None,
) -> dict[str, Any]:
    payload = model_payload if isinstance(model_payload, dict) else {}
    health = model_health if isinstance(model_health, dict) else {}
    snapshot = model_snapshot if isinstance(model_snapshot, dict) else {}
    provider = provider_payload if isinstance(provider_payload, dict) else {}
    provider_row = provider_health if isinstance(provider_health, dict) else {}
    catalog = catalog_entry if isinstance(catalog_entry, dict) else {}
    provider_catalog = {str(item).strip() for item in (provider_catalog_ids or []) if str(item).strip()}

    model_status = str(health.get("status") or "unknown").strip().lower() or "unknown"
    provider_status = str(provider_row.get("status") or "unknown").strip().lower() or "unknown"
    last_error_kind = str(health.get("last_error_kind") or "").strip().lower() or None
    status_code = health.get("status_code")
    failure_streak = int(health.get("failure_streak") or 0)
    cooldown_until = health.get("cooldown_until_iso")

    capabilities = sorted(
        {
            str(item).strip().lower()
            for item in (payload.get("capabilities") or [])
            if str(item).strip()
        }
    )
    model_enabled = bool(payload.get("enabled", False))
    model_available = bool(payload.get("available", False))
    model_routable = bool(snapshot.get("routable", False))
    provider_enabled = bool(provider.get("enabled", False))

    root_causes: set[str] = set()
    if not payload:
        root_causes.add("model_not_found")
    if provider and not provider_enabled:
        root_causes.add("provider_disabled")
    if provider and provider_status == "down":
        root_causes.add("provider_down")
    elif provider and provider_status == "degraded":
        root_causes.add("provider_degraded")
    if payload and not model_enabled:
        root_causes.add("model_disabled")
    if payload and not model_available:
        root_causes.add("model_unavailable")
    if payload and "chat" not in set(capabilities):
        root_causes.add("model_not_chat_capable")
    if payload and provider_catalog and str(model_id) not in provider_catalog:
        root_causes.add("model_missing_from_catalog")
    if payload and not model_routable:
        root_causes.add("model_unroutable")
    if model_status == "down":
        root_causes.add("model_health_down")
    elif model_status == "degraded":
        root_causes.add("model_health_degraded")
    if cooldown_until:
        root_causes.add("model_in_cooldown")
    if last_error_kind == "not_applicable":
        root_causes.add("model_not_applicable_for_chat")

    ordered_causes = _ordered(root_causes, _MODEL_CAUSE_ORDER)
    if not ordered_causes:
        ordered_causes = ["model_healthy"]

    diagnosis = {
        "status": model_status,
        "last_error_kind": last_error_kind,
        "status_code": status_code,
        "failure_streak": failure_streak,
        "root_causes": ordered_causes,
        "recommended_actions": _actions_for_root_causes(ordered_causes),
        "evidence": {
            "model": {
                "id": str(model_id or "").strip(),
                "provider": str(payload.get("provider") or "").strip().lower() or None,
                "enabled": model_enabled,
                "available": model_available,
                "routable": model_routable,
                "capabilities": capabilities,
            },
            "health": {
                "status": model_status,
                "last_error_kind": last_error_kind,
                "status_code": status_code,
                "failure_streak": failure_streak,
                "cooldown_until_iso": cooldown_until,
            },
            "provider": {
                "id": str(payload.get("provider") or "").strip().lower() or None,
                "enabled": provider_enabled,
                "status": provider_status,
                "last_error_kind": str(provider_row.get("last_error_kind") or "").strip().lower() or None,
            },
            "catalog": {
                "present": bool(catalog),
                "entry": catalog if catalog else None,
                "provider_models_count": len(provider_catalog),
            },
        },
    }
    return sanitize_support_payload(redact_audit_value(diagnosis))


def build_support_remediation_plan(
    *,
    target: str | None,
    intent: str | None,
    diagnosis: dict[str, Any] | None,
    drift_report: dict[str, Any] | None,
    safe_mode_enabled: bool,
) -> dict[str, Any]:
    normalized_target = str(target or "").strip() or None
    normalized_intent = str(intent or "fix_routing").strip().lower() or "fix_routing"
    if normalized_intent not in _INTENT_STEPS:
        normalized_intent = "fix_routing"

    causes = []
    if isinstance(diagnosis, dict):
        causes = [
            str(item).strip()
            for item in (diagnosis.get("root_causes") or [])
            if str(item).strip()
        ]

    reasons: list[str] = []
    if normalized_target:
        reasons.append(f"target={normalized_target}")
    reasons.append(f"intent={normalized_intent}")
    if safe_mode_enabled:
        reasons.append("safe_mode_enabled")
    if isinstance(drift_report, dict) and bool(drift_report.get("has_drift")):
        drift_reasons = ",".join(
            sorted({str(item).strip() for item in (drift_report.get("reasons") or []) if str(item).strip()})
        )
        if drift_reasons:
            reasons.append(f"drift={drift_reasons}")
    if causes:
        reasons.append(f"root_causes={','.join(causes)}")

    steps = [dict(row) for row in _INTENT_STEPS[normalized_intent]]
    for cause in causes:
        actions = _ROOT_CAUSE_ACTIONS.get(cause) or []
        for idx, action in enumerate(actions):
            suggestion_step = {
                "id": f"suggest_{cause}_{idx + 1}",
                "action": "manual",
                "reason": str(action),
            }
            if suggestion_step not in steps:
                steps.append(suggestion_step)

    warnings = [
        "This endpoint is plan-only and does not apply registry changes.",
    ]
    if safe_mode_enabled:
        warnings.append("Safe mode is enabled; apply phases may be intentionally blocked.")

    plan = {
        "target": normalized_target,
        "intent": normalized_intent,
        "plan_only": True,
        "reasons": reasons,
        "steps": steps,
        "warnings": warnings,
    }
    return sanitize_support_payload(redact_audit_value(plan))


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


__all__ = [
    "build_model_diagnosis",
    "build_provider_diagnosis",
    "build_support_remediation_plan",
    "sanitize_support_payload",
    "stable_json",
]
