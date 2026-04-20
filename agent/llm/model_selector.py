from __future__ import annotations

import time
from typing import Any, Iterable

from agent.llm.control_contract import normalize_model_inventory, normalize_selection_result, normalize_task_request
from agent.llm.model_state import (
    build_effective_model_state,
    build_fallback_candidates,
    effective_state_sort_key,
    explain_no_selection_reason,
)
from agent.llm.value_policy import ValuePolicy, normalize_policy


def select_model_for_task(
    inventory: Iterable[dict[str, Any]],
    task_request: dict[str, Any],
    *,
    allow_remote_fallback: bool,
    channel: str | None = None,
    latency_fallback: bool = False,
    trace_id: str | None = None,
    policy_name: str = "default",
    policy: dict[str, Any] | ValuePolicy | None = None,
    optimization_profile: str | None = None,
) -> dict[str, Any]:
    normalized_inventory = normalize_model_inventory(inventory)
    normalized_task = normalize_task_request(task_request)
    if isinstance(policy, ValuePolicy):
        normalized_policy = policy
    else:
        policy_payload = dict(policy or {})
        if optimization_profile:
            policy_payload["optimization_profile"] = optimization_profile
        normalized_policy = normalize_policy(policy_payload, name=policy_name)
    preferred_local = bool(normalized_task.get("preferred_local", True))
    task_type = str(normalized_task.get("task_type") or "chat")
    states = [
        build_effective_model_state(
            item,
            task_request=normalized_task,
            allow_remote_fallback=allow_remote_fallback,
            channel=channel,
            latency_fallback=latency_fallback,
            policy=normalized_policy,
            policy_name=policy_name,
        )
        for item in normalized_inventory
    ]
    states.sort(
        key=lambda state: effective_state_sort_key(
            state,
            preferred_local=preferred_local,
            task_type=task_type,
            channel=channel,
        )
    )
    selected = next((state for state in states if bool(state.get("suitable", False))), None)
    if selected is None:
        fallback_states = build_fallback_candidates(
            normalized_inventory,
            task_request=normalized_task,
            allow_remote_fallback=allow_remote_fallback,
            channel=channel,
            latency_fallback=latency_fallback,
            policy=normalized_policy,
            policy_name=policy_name,
            limit=3,
        )
        return normalize_selection_result(
            {
                "selected_model": None,
                "provider": None,
                "reason": explain_no_selection_reason(
                    normalized_inventory,
                    task_request=normalized_task,
                    allow_remote_fallback=allow_remote_fallback,
                    channel=channel,
                    latency_fallback=latency_fallback,
                    policy=normalized_policy,
                    policy_name=policy_name,
                ),
                "fallbacks": [state.get("id") for state in fallback_states],
                "trace_id": trace_id or f"llm-select-{int(time.time())}",
            }
        )

    reason_bits = [selected.get("state_reason") or "selected"]
    if bool(selected.get("health_ok", False)):
        reason_bits.insert(0, "healthy")
    if bool(selected.get("approved_ok", False)):
        reason_bits.append("approved")
    if preferred_local and bool(selected.get("local", False)):
        reason_bits.append("local_first")
    reason_bits.append(f"task={task_type}")
    reason_bits.append(f"profile={normalized_policy.optimization_profile}")
    fallback_states = build_fallback_candidates(
        normalized_inventory,
        task_request=normalized_task,
        allow_remote_fallback=allow_remote_fallback,
        channel=channel,
        latency_fallback=latency_fallback,
        policy=normalized_policy,
        policy_name=policy_name,
        limit=3,
        exclude_model_id=str(selected.get("id") or ""),
    )
    return normalize_selection_result(
        {
            "selected_model": selected.get("id"),
            "provider": selected.get("provider"),
            "reason": "+".join(str(item) for item in reason_bits if str(item).strip()),
            "fallbacks": [state.get("id") for state in fallback_states],
            "trace_id": trace_id or f"llm-select-{int(time.time())}",
        }
    )


__all__ = ["select_model_for_task"]
