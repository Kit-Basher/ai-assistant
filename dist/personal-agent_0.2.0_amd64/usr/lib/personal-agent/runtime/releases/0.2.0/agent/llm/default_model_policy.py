from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent.config import Config
from agent.llm.approved_local_models import approved_local_profile_for_ref
from agent.llm.capabilities import is_vision_model_name
from agent.llm.control_contract import normalize_model_inventory
from agent.llm.model_inventory import build_model_inventory
from agent.llm.model_state import build_effective_model_state
from agent.llm.registry import Registry, parse_registry_document
from agent.llm.value_policy import ValuePolicy, normalize_policy


_DEFAULT_TASK_REQUEST = {
    "task_type": "chat",
    "requirements": ["chat"],
    "preferred_local": True,
}
_TIER_ORDER = {
    "local": 0,
    "free_remote": 1,
    "cheap_remote": 2,
    "remote": 3,
}
_DEFAULT_TIER_SEQUENCE = ("local", "free_remote", "cheap_remote")
_EXTENDED_TIER_SEQUENCE = ("local", "free_remote", "cheap_remote", "remote")


def build_default_chat_inventory(
    *,
    config: Config,
    registry_document: dict[str, Any],
    router_snapshot: dict[str, Any] | None = None,
    health_summary: dict[str, Any] | None = None,
    discovered_local_models: Iterable[str] | None = (),
) -> tuple[Registry, list[dict[str, Any]]]:
    registry = parse_registry_document(registry_document, path=config.llm_registry_path)
    snapshot = router_snapshot if isinstance(router_snapshot, dict) else _synthetic_router_snapshot(registry_document, health_summary)
    inventory = build_model_inventory(
        config=config,
        registry=registry,
        router_snapshot=snapshot,
        discovered_local_models=discovered_local_models,
    )
    return registry, inventory


def choose_best_default_chat_candidate(
    *,
    config: Config,
    registry_document: dict[str, Any],
    router_snapshot: dict[str, Any] | None = None,
    health_summary: dict[str, Any] | None = None,
    discovered_local_models: Iterable[str] | None = (),
    inventory_rows: Iterable[Mapping[str, Any]] | None = None,
    candidate_model_ids: Sequence[str] | None = None,
    allowed_tiers: Sequence[str] | None = None,
    min_improvement: float | None = None,
    policy: dict[str, Any] | ValuePolicy | None = None,
    policy_name: str = "default",
    current_model_id: str | None = None,
    allow_remote_fallback: bool | None = None,
    env: Mapping[str, str] | None = None,
    secret_lookup: Callable[[str], str | None] | None = None,
    require_auth: bool = True,
    task_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry = parse_registry_document(registry_document, path=config.llm_registry_path)
    if inventory_rows is None:
        _registry, inventory = build_default_chat_inventory(
            config=config,
            registry_document=registry_document,
            router_snapshot=router_snapshot,
            health_summary=health_summary,
            discovered_local_models=discovered_local_models,
        )
        registry = _registry
    else:
        inventory = normalize_model_inventory(inventory_rows)
    normalized_policy = (
        policy
        if isinstance(policy, ValuePolicy)
        else normalize_policy(
            policy if isinstance(policy, dict) else (config.default_policy if isinstance(config.default_policy, dict) else {}),
            name=policy_name,
        )
    )
    cheap_remote_cap = _default_switch_cheap_remote_cap(config=config, policy=normalized_policy)
    effective_allow_remote_fallback = (
        bool(registry.defaults.allow_remote_fallback)
        if allow_remote_fallback is None
        else bool(allow_remote_fallback)
    )
    allowed_tier_set = {
        str(item).strip().lower()
        for item in (allowed_tiers or _DEFAULT_TIER_SEQUENCE)
        if str(item).strip()
    }
    effective_current_model_id = str(current_model_id or "").strip() or _canonical_default_model_id(registry)
    current_state, candidate_states, rejected_candidates = _policy_states(
        inventory=inventory,
        registry=registry,
        current_model_id=effective_current_model_id,
        allow_remote_fallback=effective_allow_remote_fallback,
        policy=normalized_policy,
        cheap_remote_cap=cheap_remote_cap,
        candidate_model_ids=candidate_model_ids,
        allowed_tiers=allowed_tier_set,
        env=env,
        secret_lookup=secret_lookup,
        require_auth=require_auth,
        task_request=dict(task_request or _DEFAULT_TASK_REQUEST),
    )
    min_improvement_value = max(
        0.0,
        float(min_improvement if min_improvement is not None else getattr(config, "model_watch_min_improvement", 0.08)),
    )

    best_candidate = candidate_states[0] if candidate_states else None
    current_summary = _candidate_summary(current_state)
    best_summary = _candidate_summary(best_candidate)
    tier_candidates = {
        tier: _candidate_summary(
            next((row for row in candidate_states if str(row.get("default_tier") or "") == tier), None)
        )
        for tier in _DEFAULT_TIER_SEQUENCE
    }
    utility_delta = round(
        float((best_candidate or {}).get("utility") or 0.0) - float((current_state or {}).get("utility") or 0.0),
        6,
    )
    decision = _switch_decision(
        current_state=current_state,
        best_candidate=best_candidate,
        min_improvement=min_improvement_value,
        utility_delta=utility_delta,
    )
    selected_candidate = best_summary if decision["switch_recommended"] else current_summary
    return {
        "selected_candidate": selected_candidate,
        "recommended_candidate": best_summary,
        "current_candidate": current_summary,
        "switch_recommended": bool(decision["switch_recommended"]),
        "decision_reason": str(decision["reason"]),
        "decision_detail": str(decision["detail"]),
        "current_model_id": effective_current_model_id,
        "utility_delta": utility_delta,
        "min_improvement": round(min_improvement_value, 6),
        "cheap_remote_cap_per_1m": round(float(cheap_remote_cap), 6),
        "general_remote_cap_per_1m": round(float(normalized_policy.cost_cap_per_1m), 6),
        "candidate_count": len(candidate_states),
        "candidate_rows": [_candidate_summary(row) for row in candidate_states],
        "ordered_candidates": [_candidate_summary(row) for row in candidate_states[:10]],
        "rejected_candidates": rejected_candidates[:10],
        "tier_candidates": tier_candidates,
        "policy_name": normalized_policy.name,
        "tier_order": list(_EXTENDED_TIER_SEQUENCE if "remote" in allowed_tier_set else _DEFAULT_TIER_SEQUENCE),
        "local_first": True,
        "allow_remote_fallback": effective_allow_remote_fallback,
        "allowed_tiers": sorted(allowed_tier_set),
        "task_request": dict(task_request or _DEFAULT_TASK_REQUEST),
    }


def _policy_states(
    *,
    inventory: list[dict[str, Any]],
    registry: Registry,
    current_model_id: str | None,
    allow_remote_fallback: bool,
    policy: ValuePolicy,
    cheap_remote_cap: float,
    candidate_model_ids: Sequence[str] | None,
    allowed_tiers: set[str],
    env: Mapping[str, str] | None,
    secret_lookup: Callable[[str], str | None] | None,
    require_auth: bool,
    task_request: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_ids = {
        str(item).strip()
        for item in (candidate_model_ids or [])
        if str(item).strip()
    }
    env_map = dict(env) if env is not None else dict(os.environ)
    secret_fn = secret_lookup or (lambda _name: None)
    states: list[dict[str, Any]] = []
    for item in normalize_model_inventory(inventory):
        state = build_effective_model_state(
            item,
            task_request=dict(task_request),
            allow_remote_fallback=allow_remote_fallback,
            policy=policy,
            policy_name="default",
        )
        provider_cfg = registry.providers.get(str(state.get("provider") or "").strip().lower())
        auth_ok = _provider_auth_available(provider_cfg, env_map=env_map, secret_lookup=secret_fn)
        state["auth_ok"] = auth_ok
        if require_auth and (not auth_ok) and not bool(state.get("local", False)):
            state["approved_ok"] = False
            state["fallback_eligible"] = False
            state["suitable"] = False
            state["state_reason"] = "auth_missing"
        tier = _candidate_tier(state, cheap_remote_cap=cheap_remote_cap)
        state["default_tier"] = tier
        states.append(state)

    by_id = {str(row.get("id") or "").strip(): row for row in states if str(row.get("id") or "").strip()}
    current_state = by_id.get(str(current_model_id or "").strip())

    eligible = [
        row
        for row in states
        if bool(row.get("suitable", False))
        and str(row.get("default_tier") or "") in allowed_tiers
        and (not normalized_ids or str(row.get("id") or "").strip() in normalized_ids)
    ]
    task_type = str(task_request.get("task_type") or "chat").strip().lower() or "chat"
    eligible.sort(key=lambda row: _candidate_sort_key(row, task_type=task_type))
    rejected = [
        {
            "model_id": str(row.get("id") or ""),
            "provider": str(row.get("provider") or ""),
            "reason": _candidate_rejection_reason(
                row,
                allowed_tiers=allowed_tiers,
                cheap_remote_cap=cheap_remote_cap,
            ),
            "tier": str(row.get("default_tier") or "none") or "none",
            "expected_cost_per_1m": float(row.get("expected_cost_per_1m") or 0.0),
        }
        for row in states
        if _candidate_rejection_reason(
            row,
            allowed_tiers=allowed_tiers,
            cheap_remote_cap=cheap_remote_cap,
        )
        is not None
        and (not normalized_ids or str(row.get("id") or "").strip() in normalized_ids)
    ]
    rejected.sort(key=lambda row: (str(row.get("reason") or ""), str(row.get("model_id") or "")))
    return current_state, eligible, rejected


def _candidate_tier(state: Mapping[str, Any], *, cheap_remote_cap: float) -> str | None:
    if not bool(state.get("suitable", False)):
        return None
    if bool(state.get("local", False)):
        return "local"
    expected_cost = float(state.get("expected_cost_per_1m") or 0.0)
    if expected_cost <= 0.0:
        return "free_remote"
    if expected_cost <= float(cheap_remote_cap):
        return "cheap_remote"
    return "remote"


def _candidate_sort_key(row: Mapping[str, Any], *, task_type: str = "chat") -> tuple[Any, ...]:
    tier = str(row.get("default_tier") or "")
    local_fit_state = str(row.get("local_fit_state") or "").strip().lower()
    local_fit_priority = 0
    if bool(row.get("local", False)):
        local_fit_priority = {
            "comfortable": 0,
            "unknown": 1,
            "tight": 2,
            "memory_starved": 3,
        }.get(local_fit_state, 2)
    normalized_task_type = str(task_type or "chat").strip().lower() or "chat"
    task_specialization_priority = _task_specialization_priority(row, task_type=normalized_task_type)
    context_priority = -int(row.get("context_window") or 0) if normalized_task_type == "reasoning" else 0
    quality_priority = -int(row.get("quality_rank") or 0)
    return (
        int(_TIER_ORDER.get(tier, 99)),
        local_fit_priority,
        task_specialization_priority,
        context_priority,
        quality_priority,
        -int(row.get("context_window") or 0) if normalized_task_type != "reasoning" else 0,
        -float(row.get("utility") or 0.0),
        float(row.get("expected_cost_per_1m") or 0.0),
        int(row.get("cost_rank") or 0),
        str(row.get("provider") or ""),
        str(row.get("id") or ""),
    )


def _candidate_task_types(row: Mapping[str, Any]) -> set[str]:
    explicit = row.get("task_types")
    if isinstance(explicit, (list, tuple, set, frozenset)):
        task_types = {
            str(item).strip().lower()
            for item in explicit
            if str(item).strip()
        }
        if task_types:
            return task_types
    model_id = str(row.get("id") or "").strip()
    model_name = str(row.get("model_name") or "").strip()
    profile = approved_local_profile_for_ref(model_id) or approved_local_profile_for_ref(model_name)
    if not isinstance(profile, dict):
        return set()
    return {
        str(item).strip().lower()
        for item in (profile.get("task_types") or [])
        if str(item).strip()
    }


def _task_specialization_priority(row: Mapping[str, Any], *, task_type: str) -> int:
    normalized_task_type = str(task_type or "chat").strip().lower() or "chat"
    if normalized_task_type == "vision":
        return 0
    task_types = _candidate_task_types(row)
    capabilities = {
        str(item).strip().lower()
        for item in (row.get("capabilities") if isinstance(row.get("capabilities"), list) else [])
        if str(item).strip()
    }
    model_id = str(row.get("id") or "").strip()
    model_name = str(row.get("model_name") or "").strip()
    looks_vision_specialist = (
        "vision" in task_types
        or (
            "vision" in capabilities
            and (
                is_vision_model_name(model_id)
                or is_vision_model_name(model_name)
            )
        )
    )
    if normalized_task_type in {"chat", "coding", "reasoning"} and looks_vision_specialist:
        return 3
    if normalized_task_type == "coding":
        if "coding" in task_types:
            return 0
        if not task_types:
            return 1
        return 2
    if normalized_task_type == "reasoning":
        if "reasoning" in task_types or "research" in task_types:
            return 0
        if not task_types:
            return 1
        return 2
    return 0


def _default_switch_cheap_remote_cap(*, config: Config, policy: ValuePolicy) -> float:
    configured = float(getattr(config, "default_switch_cheap_remote_cap_per_1m", float(policy.cost_cap_per_1m)))
    return max(0.0, min(configured, float(policy.cost_cap_per_1m)))


def _candidate_rejection_reason(
    row: Mapping[str, Any],
    *,
    allowed_tiers: set[str],
    cheap_remote_cap: float,
) -> str | None:
    if not bool(row.get("suitable", False)):
        return str(row.get("state_reason") or "unsuitable")
    tier = str(row.get("default_tier") or "")
    if tier:
        if tier == "remote" and "remote" not in allowed_tiers:
            return "cheap_remote_cap_exceeded"
        if tier not in allowed_tiers:
            return "tier_not_allowed"
        return None
    if bool(row.get("local", False)):
        return "tier_not_allowed"
    expected_cost = float(row.get("expected_cost_per_1m") or 0.0)
    if expected_cost > float(cheap_remote_cap):
        return "cheap_remote_cap_exceeded"
    return "tier_not_allowed"


def _candidate_summary(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, Mapping):
        return None
    return {
        "model_id": str(row.get("id") or "").strip() or None,
        "provider_id": str(row.get("provider") or "").strip().lower() or None,
        "local": bool(row.get("local", False)),
        "tier": str(row.get("default_tier") or "").strip() or None,
        "task_types": [
            str(item).strip().lower()
            for item in (row.get("task_types") if isinstance(row.get("task_types"), list) else [])
            if str(item).strip()
        ],
        "utility": round(float(row.get("utility") or 0.0), 6),
        "utility_quality": round(float(row.get("utility_quality") or 0.0), 6),
        "utility_latency": round(float(row.get("utility_latency") or 0.0), 6),
        "utility_risk": round(float(row.get("utility_risk") or 0.0), 6),
        "expected_cost_per_1m": round(float(row.get("expected_cost_per_1m") or 0.0), 6),
        "quality_rank": int(row.get("quality_rank") or 0),
        "context_window": int(row.get("context_window") or 0),
        "state_reason": str(row.get("state_reason") or "").strip() or None,
        "health_status": str(row.get("health_status") or "").strip().lower() or None,
        "auth_ok": bool(row.get("auth_ok", True)),
        "availability_state": str(row.get("availability_state") or "").strip() or None,
        "acquisition_state": str(row.get("acquisition_state") or "").strip() or None,
        "lifecycle_state": str(row.get("lifecycle_state") or "").strip() or None,
        "provider_connection_state": str(row.get("provider_connection_state") or "").strip() or None,
        "provider_selection_state": str(row.get("provider_selection_state") or "").strip() or None,
        "comfortable_local": bool(row.get("comfortable_local", False)),
        "local_fit_state": str(row.get("local_fit_state") or "").strip() or None,
        "local_fit_reason": str(row.get("local_fit_reason") or "").strip() or None,
        "local_fit_margin_gb": (
            round(float(row.get("local_fit_margin_gb") or 0.0), 3)
            if row.get("local_fit_margin_gb") is not None
            else None
        ),
        "min_memory_gb": int(row.get("min_memory_gb") or 0) or None,
    }


def _switch_decision(
    *,
    current_state: Mapping[str, Any] | None,
    best_candidate: Mapping[str, Any] | None,
    min_improvement: float,
    utility_delta: float,
) -> dict[str, Any]:
    if not isinstance(best_candidate, Mapping):
        return {
            "switch_recommended": False,
            "reason": "no_candidate",
            "detail": "No healthy approved default-chat candidate passed policy.",
        }
    best_id = str(best_candidate.get("id") or "").strip()
    current_id = str((current_state or {}).get("id") or "").strip()
    if not isinstance(current_state, Mapping) or not bool(current_state.get("suitable", False)):
        return {
            "switch_recommended": True,
            "reason": "current_unsuitable",
            "detail": "Current default is not suitable; switch to the best policy-allowed candidate.",
        }
    if best_id == current_id:
        return {
            "switch_recommended": False,
            "reason": "current_already_best",
            "detail": "Current default already matches the best candidate in its tier.",
        }
    best_tier = str(best_candidate.get("default_tier") or "")
    current_tier = str(current_state.get("default_tier") or "")
    best_priority = int(_TIER_ORDER.get(best_tier, 99))
    current_priority = int(_TIER_ORDER.get(current_tier, 99))
    if best_priority < current_priority:
        return {
            "switch_recommended": True,
            "reason": "tier_upgrade",
            "detail": f"Candidate tier {best_tier} outranks current tier {current_tier}.",
        }
    if best_priority > current_priority:
        return {
            "switch_recommended": False,
            "reason": "current_higher_priority_tier",
            "detail": f"Current tier {current_tier} outranks candidate tier {best_tier}.",
        }
    current_quality = int(current_state.get("quality_rank") or 0)
    best_quality = int(best_candidate.get("quality_rank") or 0)
    current_context = int(current_state.get("context_window") or 0)
    best_context = int(best_candidate.get("context_window") or 0)
    current_cost = float(current_state.get("expected_cost_per_1m") or 0.0)
    best_cost = float(best_candidate.get("expected_cost_per_1m") or 0.0)
    if best_quality >= current_quality + 2 and best_cost <= current_cost + 0.05:
        return {
            "switch_recommended": True,
            "reason": "quality_upgrade",
            "detail": f"Candidate quality rank {best_quality} materially exceeds current rank {current_quality}.",
        }
    if best_context >= max(current_context + 16384, int(current_context * 1.5)) and best_quality >= current_quality:
        return {
            "switch_recommended": True,
            "reason": "context_headroom",
            "detail": f"Candidate context window {best_context} materially exceeds current window {current_context}.",
        }
    if best_cost + 0.25 < current_cost and best_quality >= current_quality:
        return {
            "switch_recommended": True,
            "reason": "cost_reduction",
            "detail": f"Candidate expected cost {best_cost:.3f} is materially below current cost {current_cost:.3f}.",
        }
    if utility_delta >= float(min_improvement):
        return {
            "switch_recommended": True,
            "reason": "material_improvement",
            "detail": f"Candidate utility improved by {utility_delta:.3f}, meeting threshold {min_improvement:.3f}.",
        }
    return {
        "switch_recommended": False,
        "reason": "improvement_below_threshold",
        "detail": f"Utility improvement {utility_delta:.3f} is below threshold {min_improvement:.3f}.",
    }


def _canonical_default_model_id(registry: Registry) -> str | None:
    raw = str(registry.defaults.default_model or "").strip()
    provider_id = str(registry.defaults.default_provider or "").strip().lower()
    if not raw:
        return None
    if raw in registry.models:
        return raw
    if provider_id and ":" not in raw:
        scoped = f"{provider_id}:{raw}"
        if scoped in registry.models:
            return scoped
    return raw


def _synthetic_router_snapshot(
    registry_document: dict[str, Any],
    health_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    document = registry_document if isinstance(registry_document, dict) else {}
    defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
    providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
    models = document.get("models") if isinstance(document.get("models"), dict) else {}
    health = health_summary if isinstance(health_summary, dict) else {}
    provider_health_rows = health.get("providers") if isinstance(health.get("providers"), list) else []
    model_health_rows = health.get("models") if isinstance(health.get("models"), list) else []
    provider_health = {
        str(row.get("id") or "").strip().lower(): row
        for row in provider_health_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    model_health = {
        str(row.get("id") or "").strip(): row
        for row in model_health_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }

    provider_rows: list[dict[str, Any]] = []
    for provider_id, payload in sorted(providers.items()):
        if not isinstance(payload, dict):
            continue
        health_row = provider_health.get(str(provider_id).strip().lower()) or {}
        status = str(health_row.get("status") or "unknown").strip().lower() or "unknown"
        provider_rows.append(
            {
                "id": str(provider_id).strip().lower(),
                "enabled": bool(payload.get("enabled", True)),
                "available": status in {"ok", "degraded", "unknown"},
                "local": bool(payload.get("local", False)),
                "health": {
                    "status": status,
                    "last_error_kind": health_row.get("last_error_kind"),
                    "status_code": health_row.get("status_code"),
                    # Provide runtime evidence so remote rows can stay authoritative in fallback mode.
                    "last_checked_at": 1 if status != "unknown" else None,
                },
            }
        )

    model_rows: list[dict[str, Any]] = []
    for model_id, payload in sorted(models.items()):
        if not isinstance(payload, dict):
            continue
        provider_id = str(payload.get("provider") or "").strip().lower()
        provider_status = str((provider_health.get(provider_id) or {}).get("status") or "unknown").strip().lower() or "unknown"
        health_row = model_health.get(str(model_id).strip()) or {}
        status = str(health_row.get("status") or "unknown").strip().lower() or "unknown"
        pricing = payload.get("pricing") if isinstance(payload.get("pricing"), dict) else {}
        available = bool(payload.get("enabled", True)) and bool(payload.get("available", True))
        routable = available and provider_status in {"ok", "degraded", "unknown"} and status == "ok"
        model_rows.append(
            {
                "id": str(model_id).strip(),
                "provider": provider_id,
                "model": str(payload.get("model") or "").strip(),
                "capabilities": list(payload.get("capabilities") or []),
                "enabled": bool(payload.get("enabled", True)),
                "available": available,
                "routable": routable,
                "max_context_tokens": payload.get("max_context_tokens"),
                "input_cost_per_million_tokens": pricing.get("input_per_million_tokens"),
                "output_cost_per_million_tokens": pricing.get("output_per_million_tokens"),
                "health": {
                    "status": status,
                    "last_error_kind": health_row.get("last_error_kind"),
                    "status_code": health_row.get("status_code"),
                    "last_checked_at": 1 if status != "unknown" else None,
                },
            }
        )

    return {
        "defaults": {
            "default_provider": defaults.get("default_provider"),
            "default_model": defaults.get("default_model"),
            "allow_remote_fallback": defaults.get("allow_remote_fallback", True),
        },
        "providers": provider_rows,
        "models": model_rows,
    }


def _provider_auth_available(
    provider: Any,
    *,
    env_map: Mapping[str, str],
    secret_lookup: Callable[[str], str | None],
) -> bool:
    if provider is None:
        return False
    if bool(getattr(provider, "local", False)):
        return True
    source = getattr(provider, "api_key_source", None)
    if source is None:
        return True
    source_type = str(getattr(source, "source_type", "") or "").strip().lower()
    source_name = str(getattr(source, "name", "") or "").strip()
    if source_type == "env":
        return bool(str(env_map.get(source_name, "")).strip())
    if source_type == "secret":
        return bool(str(secret_lookup(source_name) or "").strip())
    return False


__all__ = [
    "build_default_chat_inventory",
    "choose_best_default_chat_candidate",
]
