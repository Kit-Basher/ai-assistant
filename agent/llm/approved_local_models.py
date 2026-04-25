from __future__ import annotations

from typing import Any

from agent.llm.control_contract import normalize_task_request
from agent.llm.model_state import build_effective_model_state


_APPROVED_LOCAL_MODEL_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "id": "ollama:qwen2.5:3b-instruct",
        "task_types": ("chat", "health", "tool_use"),
        "capabilities": ("chat",),
        "context_window": 32768,
        "min_memory_gb": 8,
        "preferred": True,
        "size_hint": "3B",
        "install_name": "qwen2.5:3b-instruct",
        "reason": "lightweight local-first baseline for everyday chat and PC health requests",
        "quality_rank": 4,
        "cost_rank": 1,
    },
    {
        "id": "ollama:qwen2.5-coder:7b",
        "task_types": ("coding",),
        "capabilities": ("chat",),
        "context_window": 32768,
        "min_memory_gb": 12,
        "preferred": True,
        "size_hint": "7B",
        "install_name": "qwen2.5-coder:7b",
        "reason": "approved local coding baseline for debugging and code review",
        "quality_rank": 7,
        "cost_rank": 2,
    },
    {
        "id": "ollama:llava:7b",
        "task_types": ("vision",),
        "capabilities": ("chat", "vision"),
        "context_window": 32768,
        "min_memory_gb": 12,
        "preferred": True,
        "size_hint": "7B",
        "install_name": "llava:7b",
        "reason": "approved local vision baseline for screenshots and image analysis",
        "quality_rank": 6,
        "cost_rank": 2,
    },
    {
        "id": "ollama:qwen2.5:7b-instruct",
        "task_types": ("chat", "reasoning"),
        "capabilities": ("chat",),
        "context_window": 32768,
        "min_memory_gb": 12,
        "preferred": True,
        "size_hint": "7B",
        "install_name": "qwen2.5:7b-instruct",
        "reason": "approved local chat and reasoning baseline for longer comparisons and analysis",
        "quality_rank": 7,
        "cost_rank": 2,
    },
)


def approved_local_model_profiles() -> list[dict[str, Any]]:
    rows = [dict(row) for row in _APPROVED_LOCAL_MODEL_PROFILES]
    rows.sort(
        key=lambda row: (
            0 if bool(row.get("preferred", False)) else 1,
            int(row.get("min_memory_gb") or 0),
            str(row.get("install_name") or ""),
            str(row.get("id") or ""),
        )
    )
    return rows


def approved_local_profile_for_ref(model_ref: str) -> dict[str, Any] | None:
    normalized = str(model_ref or "").strip()
    if not normalized:
        return None
    for profile in approved_local_model_profiles():
        if normalized in {
            str(profile.get("id") or "").strip(),
            str(profile.get("install_name") or "").strip(),
        }:
            return dict(profile)
    return None


def _profile_inventory_row(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(profile.get("id") or "").strip(),
        "provider": "ollama",
        "installed": False,
        "available": True,
        "healthy": True,
        "capabilities": list(profile.get("capabilities") or []),
        "size": str(profile.get("size_hint") or "").strip() or None,
        "context_window": int(profile.get("context_window") or 0) or None,
        "local": True,
        "approved": True,
        "reason": str(profile.get("reason") or "approved_local_profile").strip() or "approved_local_profile",
        "quality_rank": int(profile.get("quality_rank") or 0),
        "cost_rank": int(profile.get("cost_rank") or 0),
        "source": "approved_local_profile",
    }


def approved_local_profiles_for_task(task_request: dict[str, Any], *, limit: int = 2) -> list[dict[str, Any]]:
    normalized_task = normalize_task_request(task_request)
    task_type = str(normalized_task.get("task_type") or "chat")
    matched: list[dict[str, Any]] = []
    for profile in approved_local_model_profiles():
        task_types = {
            str(value).strip().lower()
            for value in (profile.get("task_types") or [])
            if str(value).strip()
        }
        if task_type not in task_types:
            continue
        candidate_state = build_effective_model_state(
            _profile_inventory_row(profile),
            task_request=normalized_task,
            allow_remote_fallback=True,
            policy=None,
            policy_name="default",
        )
        if not bool(candidate_state.get("capability_ok", False)) or not bool(candidate_state.get("context_ok", False)):
            continue
        matched.append(profile)
    matched.sort(
        key=lambda row: (
            0 if bool(row.get("preferred", False)) else 1,
            int(row.get("min_memory_gb") or 0),
            str(row.get("install_name") or ""),
            str(row.get("id") or ""),
        )
    )
    return matched[: max(0, int(limit))]


__all__ = [
    "approved_local_model_profiles",
    "approved_local_profile_for_ref",
    "approved_local_profiles_for_task",
]
