from __future__ import annotations

import copy
import os
from typing import Any, Callable

from agent.config import Config
from agent.llm.default_model_policy import choose_best_default_chat_candidate


def build_autoconfig_plan(
    registry_document: dict[str, Any],
    health_summary: dict[str, Any] | None = None,
    *,
    config: Config,
    router_snapshot: dict[str, Any] | None = None,
    secret_lookup: Callable[[str], str | None] | None = None,
    disable_auth_failed_providers: bool = True,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    document = copy.deepcopy(registry_document if isinstance(registry_document, dict) else {})
    providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
    defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
    env_map = env if env is not None else dict(os.environ)
    secret_fn = secret_lookup or (lambda _name: None)
    health = _health_maps(health_summary)

    changes: list[dict[str, Any]] = []
    reasons: list[str] = []
    proposed_defaults = {
        "routing_mode": str(defaults.get("routing_mode") or "auto").strip().lower() or "auto",
        "default_provider": str(defaults.get("default_provider") or "").strip().lower() or None,
        "default_model": str(defaults.get("default_model") or "").strip() or None,
        "allow_remote_fallback": bool(defaults.get("allow_remote_fallback", True)),
    }

    policy_result = choose_best_default_chat_candidate(
        config=config,
        registry_document=document,
        router_snapshot=router_snapshot,
        health_summary=health_summary,
        env=env_map,
        secret_lookup=secret_fn,
    )
    selected = (
        policy_result.get("recommended_candidate")
        if bool(policy_result.get("switch_recommended"))
        else policy_result.get("current_candidate")
    )
    if selected is not None:
        selected_provider = str(selected["provider_id"])
        selected_model = str(selected["model_id"])
        selected_is_local = bool(selected["local"])
        desired_routing_mode = "prefer_local_lowest_cost_capable" if selected_is_local else "prefer_best"
        desired_allow_remote = bool(proposed_defaults["allow_remote_fallback"])
        if not selected_is_local:
            desired_allow_remote = True

        if not bool(policy_result.get("switch_recommended")):
            reasons.append("keep_current_default_model")
        else:
            if proposed_defaults["routing_mode"] != desired_routing_mode:
                changes.append(
                    {
                        "kind": "defaults",
                        "field": "routing_mode",
                        "before": proposed_defaults["routing_mode"],
                        "after": desired_routing_mode,
                        "reason": "selected_best_available_candidate",
                    }
                )
                proposed_defaults["routing_mode"] = desired_routing_mode
            if proposed_defaults["default_provider"] != selected_provider:
                changes.append(
                    {
                        "kind": "defaults",
                        "field": "default_provider",
                        "before": proposed_defaults["default_provider"],
                        "after": selected_provider,
                        "reason": "selected_best_available_candidate",
                    }
                )
                proposed_defaults["default_provider"] = selected_provider
            if proposed_defaults["default_model"] != selected_model:
                changes.append(
                    {
                        "kind": "defaults",
                        "field": "default_model",
                        "before": proposed_defaults["default_model"],
                        "after": selected_model,
                        "reason": "selected_best_available_candidate",
                    }
                )
                proposed_defaults["default_model"] = selected_model
            if bool(proposed_defaults["allow_remote_fallback"]) != bool(desired_allow_remote):
                changes.append(
                    {
                        "kind": "defaults",
                        "field": "allow_remote_fallback",
                        "before": bool(proposed_defaults["allow_remote_fallback"]),
                        "after": bool(desired_allow_remote),
                        "reason": "selected_best_available_candidate",
                    }
                )
                proposed_defaults["allow_remote_fallback"] = bool(desired_allow_remote)

        reasons.append(
            "selected {scope} model {model_id} (provider={provider_id}, tier={tier}, cost={cost:.3f}, utility={utility:.3f})".format(
                scope="local" if selected_is_local else "remote",
                model_id=selected_model,
                provider_id=selected_provider,
                tier=str(selected.get("tier") or "unknown"),
                cost=float(selected.get("expected_cost_per_1m") or 0.0),
                utility=float(selected.get("utility") or 0.0),
            )
        )
        reasons.append(str(policy_result.get("decision_detail") or ""))
    else:
        reasons.append("no_working_candidate_found")

    providers_to_disable: list[str] = []
    if disable_auth_failed_providers:
        for provider_id, provider_payload in sorted(providers.items()):
            if not isinstance(provider_payload, dict):
                continue
            if not bool(provider_payload.get("enabled", True)):
                continue
            provider_health = health["providers"].get(str(provider_id)) if isinstance(health["providers"], dict) else None
            if not isinstance(provider_health, dict):
                continue
            status = str(provider_health.get("status") or "unknown").strip().lower()
            last_error = str(provider_health.get("last_error_kind") or "").strip().lower()
            status_code = provider_health.get("status_code")
            auth_failed = last_error == "auth_error" or int(status_code or 0) in {401, 403}
            if status == "down" and auth_failed:
                providers_to_disable.append(str(provider_id))

    for provider_id in sorted(set(providers_to_disable)):
        before_enabled = bool((providers.get(provider_id) or {}).get("enabled", True))
        if not before_enabled:
            continue
        changes.append(
            {
                "kind": "provider",
                "id": provider_id,
                "field": "enabled",
                "before": True,
                "after": False,
                "reason": "auth_hard_failure",
            }
        )
        reasons.append(f"disable provider {provider_id} due to auth_hard_failure")

    changes.sort(key=_change_sort_key)
    reasons = sorted(set(reasons))
    return {
        "ok": True,
        "changes": changes,
        "reasons": reasons,
        "selected_candidate": selected,
        "selection_policy": policy_result,
        "proposed_defaults": proposed_defaults,
        "impact": {
            "changes_count": len(changes),
            "providers_to_disable": sorted(set(providers_to_disable)),
            "selected_model": (selected or {}).get("model_id"),
        },
    }


def apply_autoconfig_plan(registry_document: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(registry_document if isinstance(registry_document, dict) else {})
    providers = updated.get("providers") if isinstance(updated.get("providers"), dict) else {}
    defaults = updated.get("defaults") if isinstance(updated.get("defaults"), dict) else {}
    changes = plan.get("changes") if isinstance(plan.get("changes"), list) else []

    for row in sorted((item for item in changes if isinstance(item, dict)), key=_change_sort_key):
        kind = str(row.get("kind") or "").strip().lower()
        field = str(row.get("field") or "").strip()
        if kind == "defaults":
            defaults[field] = row.get("after")
            continue
        if kind == "provider":
            provider_id = str(row.get("id") or "").strip().lower()
            if not provider_id or provider_id not in providers or not isinstance(providers.get(provider_id), dict):
                continue
            providers[provider_id] = {
                **providers[provider_id],
                field: row.get("after"),
            }

    updated["providers"] = providers
    updated["defaults"] = defaults
    return updated


def _health_maps(summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    payload = summary if isinstance(summary, dict) else {}
    provider_rows = payload.get("providers") if isinstance(payload.get("providers"), list) else []
    model_rows = payload.get("models") if isinstance(payload.get("models"), list) else []
    providers = {
        str(row.get("id") or "").strip().lower(): row
        for row in provider_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    models = {
        str(row.get("id") or "").strip(): row
        for row in model_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    return {"providers": providers, "models": models}
def _change_sort_key(change: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(change.get("kind") or ""),
        str(change.get("id") or ""),
        str(change.get("field") or ""),
    )
