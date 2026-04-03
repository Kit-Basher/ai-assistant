from __future__ import annotations

from typing import Any, Iterable

from agent.llm.control_contract import normalize_model_inventory, normalize_task_request
from agent.llm.model_latency import telegram_text_model_gate
from agent.llm.value_policy import ValuePolicy, normalize_policy, score_candidate_utility


def _caps(item: dict[str, Any]) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (item.get("capabilities") or [])
        if str(value).strip()
    }


def required_capabilities(task_request: dict[str, Any]) -> set[str]:
    normalized = normalize_task_request(task_request)
    return {
        str(item).strip().lower()
        for item in (normalized.get("requirements") or [])
        if str(item).strip() and str(item).strip().lower() in {"chat", "vision", "json", "tools"}
    }


def needs_long_context(task_request: dict[str, Any]) -> bool:
    normalized = normalize_task_request(task_request)
    return "long_context" in {
        str(item).strip().lower()
        for item in (normalized.get("requirements") or [])
        if str(item).strip()
    }


def capability_match(item: dict[str, Any], task_request: dict[str, Any]) -> bool:
    normalized_task = normalize_task_request(task_request)
    caps = _caps(item)
    task_type = str(normalized_task.get("task_type") or "chat")
    required = required_capabilities(normalized_task)
    if required and not required.issubset(caps):
        return False
    if task_type in {"chat", "coding", "reasoning", "tool_use", "health"} and "chat" not in caps:
        return False
    if task_type == "vision" and "vision" not in caps:
        return False
    return True


def build_effective_model_state(
    item: dict[str, Any],
    *,
    task_request: dict[str, Any],
    allow_remote_fallback: bool,
    channel: str | None = None,
    latency_fallback: bool = False,
    policy: dict[str, Any] | ValuePolicy | None = None,
    policy_name: str = "default",
) -> dict[str, Any]:
    normalized_item = normalize_model_inventory([item])[0]
    normalized_task = normalize_task_request(task_request)
    normalized_policy = policy if isinstance(policy, ValuePolicy) else normalize_policy(policy or {}, name=policy_name)
    caps = _caps(normalized_item)
    context_window = normalized_item.get("context_window")
    context_ok = True
    if needs_long_context(normalized_task):
        context_ok = isinstance(context_window, int) and int(context_window) >= 32000
    utility = score_candidate_utility(
        {
            "model_id": normalized_item.get("id"),
            "provider": normalized_item.get("provider"),
            "local": bool(normalized_item.get("local", False)),
            "routable": bool(normalized_item.get("available", False)) and bool(normalized_item.get("healthy", False)),
            "price_in": normalized_item.get("price_in"),
            "price_out": normalized_item.get("price_out"),
            "health_status": str(normalized_item.get("health_status") or ("ok" if bool(normalized_item.get("healthy", False)) else "down")),
            "quality_rank": normalized_item.get("quality_rank"),
            "context_tokens": normalized_item.get("context_window"),
            "model_name": normalized_item.get("model_name"),
            "size": normalized_item.get("size"),
        },
        policy=normalized_policy,
        allow_remote_fallback=allow_remote_fallback,
        channel=channel,
        latency_fallback=latency_fallback,
    )
    available_ok = bool(normalized_item.get("available", False))
    health_ok = bool(normalized_item.get("healthy", False))
    approved_ok = bool(normalized_item.get("approved", False))
    local_ok = allow_remote_fallback or bool(normalized_item.get("local", False))
    capability_ok = capability_match(normalized_item, normalized_task)
    channel_ok, channel_reason = telegram_text_model_gate(
        channel=channel,
        task_type=str(normalized_task.get("task_type") or "chat"),
        required_capabilities=required_capabilities(normalized_task),
        model_id=normalized_item.get("id"),
        model_name=normalized_item.get("model_name"),
        capabilities=normalized_item.get("capabilities"),
        speed_class=str(utility.speed_class or "unknown"),
        latency_fallback=latency_fallback,
    )
    policy_ok = bool(utility.allowed)
    suitable = (
        available_ok
        and health_ok
        and approved_ok
        and local_ok
        and capability_ok
        and channel_ok
        and context_ok
        and policy_ok
    )
    fallback_eligible = (
        available_ok
        and health_ok
        and approved_ok
        and local_ok
        and capability_ok
        and channel_ok
        and context_ok
    )
    if suitable:
        reason = "suitable"
    elif not capability_ok:
        reason = "capability_mismatch"
    elif not channel_ok:
        reason = str(channel_reason or "channel_blocked")
    elif not available_ok:
        reason = str(normalized_item.get("reason") or "unavailable")
    elif not health_ok:
        reason = str(normalized_item.get("health_failure_kind") or normalized_item.get("reason") or "degraded_health")
    elif not approved_ok:
        reason = "not_approved"
    elif not local_ok:
        reason = "remote_disabled"
    elif not context_ok:
        reason = "context_inadequate"
    elif not policy_ok:
        reason = str(utility.rejected_by or "policy_blocked")
    else:
        reason = str(normalized_item.get("reason") or "unsuitable")
    return {
        **normalized_item,
        "task_type": str(normalized_task.get("task_type") or "chat"),
        "capability_ok": capability_ok,
        "context_ok": context_ok,
        "available_ok": available_ok,
        "health_ok": health_ok,
        "approved_ok": approved_ok,
        "local_ok": local_ok,
        "channel_ok": channel_ok,
        "policy_ok": policy_ok,
        "suitable": suitable,
        "fallback_eligible": fallback_eligible,
        "state_reason": reason,
        "utility": float(utility.utility),
        "expected_cost_per_1m": float(utility.expected_cost_per_1m),
        "utility_quality": float(utility.quality),
        "utility_latency": float(utility.latency),
        "utility_risk": float(utility.risk),
        "utility_speed_class": str(utility.speed_class or "unknown"),
        "policy_rejected_by": utility.rejected_by,
        "capability_score": sum(1 for cap in required_capabilities(normalized_task) if cap in caps),
    }


def effective_state_sort_key(
    state: dict[str, Any],
    *,
    preferred_local: bool,
    task_type: str,
    channel: str | None = None,
) -> tuple[Any, ...]:
    normalized_channel = str(channel or "").strip().lower()
    quality_rank = int(state.get("quality_rank") or 0)
    quality_bias = quality_rank if normalized_channel != "telegram" and task_type in {"coding", "reasoning", "vision"} else 0
    speed_class = str(state.get("utility_speed_class") or "unknown").strip().lower()
    telegram_speed_rank = {
        "fast": 0,
        "unknown": 1,
        "medium": 2,
        "slow": 3,
    }.get(speed_class, 1)
    return (
        0 if bool(state.get("suitable", False)) else 1,
        0 if bool(state.get("health_ok", False)) else 1,
        0 if bool(state.get("approved_ok", False)) else 1,
        0 if (preferred_local and bool(state.get("local", False))) or (not preferred_local) else 1,
        telegram_speed_rank if normalized_channel == "telegram" else 0,
        -int(state.get("capability_score") or 0),
        0 if bool(state.get("context_ok", False)) else 1,
        -quality_bias,
        -float(state.get("utility") or 0.0),
        int(state.get("cost_rank") or 0),
        str(state.get("provider") or ""),
        str(state.get("id") or ""),
    )


def build_fallback_candidates(
    inventory: Iterable[dict[str, Any]],
    *,
    task_request: dict[str, Any],
    allow_remote_fallback: bool,
    channel: str | None = None,
    latency_fallback: bool = False,
    policy: dict[str, Any] | ValuePolicy | None = None,
    policy_name: str = "default",
    limit: int = 3,
    exclude_model_id: str | None = None,
) -> list[dict[str, Any]]:
    normalized_task = normalize_task_request(task_request)
    preferred_local = bool(normalized_task.get("preferred_local", True))
    task_type = str(normalized_task.get("task_type") or "chat")
    states = [
        build_effective_model_state(
            item,
            task_request=normalized_task,
            allow_remote_fallback=allow_remote_fallback,
            channel=channel,
            latency_fallback=latency_fallback,
            policy=policy,
            policy_name=policy_name,
        )
        for item in normalize_model_inventory(inventory)
    ]
    filtered = [
        state
        for state in states
        if bool(state.get("fallback_eligible", False)) and str(state.get("id") or "") != str(exclude_model_id or "")
    ]
    filtered.sort(
        key=lambda state: effective_state_sort_key(
            state,
            preferred_local=preferred_local,
            task_type=task_type,
            channel=channel,
        )
    )
    return filtered[: max(0, int(limit))]


def explain_no_selection_reason(
    inventory: Iterable[dict[str, Any]],
    *,
    task_request: dict[str, Any],
    allow_remote_fallback: bool,
    channel: str | None = None,
    latency_fallback: bool = False,
    policy: dict[str, Any] | ValuePolicy | None = None,
    policy_name: str = "default",
) -> str:
    normalized_task = normalize_task_request(task_request)
    states = [
        build_effective_model_state(
            item,
            task_request=normalized_task,
            allow_remote_fallback=allow_remote_fallback,
            channel=channel,
            latency_fallback=latency_fallback,
            policy=policy,
            policy_name=policy_name,
        )
        for item in normalize_model_inventory(inventory)
    ]
    if str(channel or "").strip().lower() == "telegram":
        channel_allowed = [state for state in states if bool(state.get("channel_ok", False))]
        if not channel_allowed:
            return "no_telegram_fast_text_model"
    local_states = [state for state in states if bool(state.get("local", False))]
    capable_local = [state for state in local_states if bool(state.get("capability_ok", False)) and bool(state.get("context_ok", False))]
    if not capable_local:
        return "no_local_model_with_required_capabilities"
    healthy_local = [
        state
        for state in capable_local
        if bool(state.get("available_ok", False)) and bool(state.get("health_ok", False))
    ]
    if not healthy_local:
        return "no_healthy_local_model"
    approved_local = [state for state in healthy_local if bool(state.get("approved_ok", False))]
    if not approved_local:
        return "no_approved_local_model"
    policy_local = [state for state in approved_local if bool(state.get("policy_ok", False))]
    if not policy_local:
        return "no_local_model_allowed_by_policy"
    if not allow_remote_fallback:
        remote_capable = [
            state
            for state in states
            if not bool(state.get("local", False))
            and bool(state.get("capability_ok", False))
            and bool(state.get("context_ok", False))
            and bool(state.get("available_ok", False))
            and bool(state.get("health_ok", False))
            and bool(state.get("approved_ok", False))
            and bool(state.get("policy_ok", False))
        ]
        if remote_capable:
            return "remote_fallback_disabled"
    return "no_suitable_model"


__all__ = [
    "build_effective_model_state",
    "build_fallback_candidates",
    "capability_match",
    "explain_no_selection_reason",
    "effective_state_sort_key",
    "needs_long_context",
    "required_capabilities",
]
