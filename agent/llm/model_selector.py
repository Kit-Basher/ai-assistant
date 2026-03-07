from __future__ import annotations

import time
from typing import Any, Iterable

from agent.llm.control_contract import normalize_model_inventory, normalize_selection_result, normalize_task_request
from agent.llm.value_policy import normalize_policy, score_candidate_utility


def _required_capabilities(task_request: dict[str, Any]) -> set[str]:
    values = {
        str(item).strip().lower()
        for item in (task_request.get("requirements") or [])
        if str(item).strip()
    }
    return {item for item in values if item in {"chat", "vision", "json", "tools"}}


def _needs_long_context(task_request: dict[str, Any]) -> bool:
    return "long_context" in {
        str(item).strip().lower()
        for item in (task_request.get("requirements") or [])
        if str(item).strip()
    }


def _capability_score(item: dict[str, Any], required_capabilities: set[str]) -> int:
    caps = {
        str(value).strip().lower()
        for value in (item.get("capabilities") or [])
        if str(value).strip()
    }
    return sum(1 for capability in required_capabilities if capability in caps)


def _context_score(item: dict[str, Any], needs_long_context: bool) -> int:
    if not needs_long_context:
        return 1
    context_window = item.get("context_window")
    if isinstance(context_window, int) and context_window >= 32000:
        return 2
    if context_window is None:
        return 1
    return 0


def _quality_bias(item: dict[str, Any], task_type: str) -> int:
    quality_rank = int(item.get("quality_rank") or 0)
    if task_type in {"reasoning", "coding", "vision"}:
        return quality_rank
    return 0


def _selection_sort_key(
    item: dict[str, Any],
    *,
    task_request: dict[str, Any],
    normalized_policy: Any,
    allow_remote_fallback: bool,
) -> tuple[Any, ...]:
    required_capabilities = _required_capabilities(task_request)
    preferred_local = bool(task_request.get("preferred_local", True))
    needs_long_context = _needs_long_context(task_request)
    utility = score_candidate_utility(
        {
            "model_id": item.get("id"),
            "provider": item.get("provider"),
            "local": bool(item.get("local", False)),
            "routable": bool(item.get("available", False)) and bool(item.get("healthy", False)),
            "price_in": item.get("price_in"),
            "price_out": item.get("price_out"),
            "health_status": "ok" if bool(item.get("healthy", False)) else "down",
            "quality_rank": item.get("quality_rank"),
            "context_tokens": item.get("context_window"),
        },
        policy=normalized_policy,
        allow_remote_fallback=allow_remote_fallback,
    )
    return (
        0 if bool(item.get("healthy", False)) else 1,
        0 if bool(item.get("approved", False)) else 1,
        0 if (preferred_local and bool(item.get("local", False))) or (not preferred_local) else 1,
        -_capability_score(item, required_capabilities),
        -_context_score(item, needs_long_context),
        -_quality_bias(item, str(task_request.get("task_type") or "chat")),
        -float(utility.utility),
        int(item.get("cost_rank") or 0),
        str(item.get("provider") or ""),
        str(item.get("id") or ""),
    )


def _candidate_allowed(
    item: dict[str, Any],
    *,
    task_request: dict[str, Any],
    normalized_policy: Any,
    allow_remote_fallback: bool,
) -> bool:
    required_capabilities = _required_capabilities(task_request)
    caps = {
        str(value).strip().lower()
        for value in (item.get("capabilities") or [])
        if str(value).strip()
    }
    if required_capabilities and not required_capabilities.issubset(caps):
        return False
    if _needs_long_context(task_request):
        context_window = item.get("context_window")
        if isinstance(context_window, int) and context_window < 32000:
            return False
    if not bool(item.get("approved", False)):
        return False
    if not bool(item.get("available", False)):
        return False
    utility = score_candidate_utility(
        {
            "model_id": item.get("id"),
            "provider": item.get("provider"),
            "local": bool(item.get("local", False)),
            "routable": bool(item.get("available", False)) and bool(item.get("healthy", False)),
            "price_in": item.get("price_in"),
            "price_out": item.get("price_out"),
            "health_status": "ok" if bool(item.get("healthy", False)) else "down",
            "quality_rank": item.get("quality_rank"),
            "context_tokens": item.get("context_window"),
        },
        policy=normalized_policy,
        allow_remote_fallback=allow_remote_fallback,
    )
    return bool(utility.allowed)


def select_model_for_task(
    inventory: Iterable[dict[str, Any]],
    task_request: dict[str, Any],
    *,
    allow_remote_fallback: bool,
    trace_id: str | None = None,
    policy_name: str = "default",
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_inventory = normalize_model_inventory(inventory)
    normalized_task = normalize_task_request(task_request)
    normalized_policy = normalize_policy(policy or {}, name=policy_name)
    candidates = [
        item
        for item in normalized_inventory
        if _candidate_allowed(
            item,
            task_request=normalized_task,
            normalized_policy=normalized_policy,
            allow_remote_fallback=allow_remote_fallback,
        )
    ]
    sorted_candidates = sorted(
        candidates,
        key=lambda item: _selection_sort_key(
            item,
            task_request=normalized_task,
            normalized_policy=normalized_policy,
            allow_remote_fallback=allow_remote_fallback,
        ),
    )
    selected = sorted_candidates[0] if sorted_candidates else None
    if selected is None:
        return normalize_selection_result(
            {
                "selected_model": None,
                "provider": None,
                "reason": "no_suitable_model",
                "fallbacks": [item.get("id") for item in normalized_inventory[:3]],
                "trace_id": trace_id or f"llm-select-{int(time.time())}",
            }
        )
    preferred_local = bool(normalized_task.get("preferred_local", True))
    reason_bits = ["healthy", "approved"]
    if preferred_local and bool(selected.get("local", False)):
        reason_bits.append("local_first")
    reason_bits.append(f"task={normalized_task.get('task_type')}")
    return normalize_selection_result(
        {
            "selected_model": selected.get("id"),
            "provider": selected.get("provider"),
            "reason": "+".join(reason_bits),
            "fallbacks": [item.get("id") for item in sorted_candidates[1:4]],
            "trace_id": trace_id or f"llm-select-{int(time.time())}",
        }
    )


__all__ = ["select_model_for_task"]
