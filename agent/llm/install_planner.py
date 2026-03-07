from __future__ import annotations

from typing import Any, Iterable

from agent.llm.control_contract import normalize_model_inventory, normalize_selection_result, normalize_task_request
from agent.llm.model_state import build_effective_model_state


_APPROVED_INSTALLS: dict[str, dict[str, Any]] = {
    "chat": {
        "model": "qwen2.5:3b-instruct",
        "size": "3B",
        "tasks": {"chat", "health", "tool_use"},
    },
    "coding": {
        "model": "qwen2.5:7b-instruct",
        "size": "7B",
        "tasks": {"coding", "reasoning"},
    },
}


def _plan_target(task_type: str) -> dict[str, Any] | None:
    normalized = str(task_type or "chat").strip().lower() or "chat"
    for row in _APPROVED_INSTALLS.values():
        tasks = row.get("tasks") if isinstance(row.get("tasks"), set) else set()
        if normalized in tasks:
            return dict(row)
    if normalized == "vision":
        return None
    return dict(_APPROVED_INSTALLS["chat"])


def build_install_plan(
    *,
    inventory: Iterable[dict[str, Any]],
    task_request: dict[str, Any],
    selection_result: dict[str, Any] | None,
    allow_remote_fallback: bool = True,
    policy: dict[str, Any] | None = None,
    policy_name: str = "default",
) -> dict[str, Any]:
    normalized_inventory = normalize_model_inventory(inventory)
    normalized_task = normalize_task_request(task_request)
    normalized_selection = normalize_selection_result(selection_result)

    selected_model = str(normalized_selection.get("selected_model") or "").strip()
    if selected_model:
        selected_item = next((item for item in normalized_inventory if str(item.get("id") or "") == selected_model), None)
        if isinstance(selected_item, dict):
            selected_state = build_effective_model_state(
                selected_item,
                task_request=normalized_task,
                allow_remote_fallback=allow_remote_fallback,
                policy=policy,
                policy_name=policy_name,
            )
            if bool(selected_state.get("local", False)) and bool(selected_state.get("suitable", False)):
                return {
                    "needed": False,
                    "approved": True,
                    "plan": [],
                    "next_action": "No action needed.",
                }

    suitable_local = [
        build_effective_model_state(
            item,
            task_request=normalized_task,
            allow_remote_fallback=allow_remote_fallback,
            policy=policy,
            policy_name=policy_name,
        )
        for item in normalized_inventory
        if bool(item.get("local", False))
    ]
    if any(bool(item.get("suitable", False)) for item in suitable_local):
        return {
            "needed": False,
            "approved": True,
            "plan": [],
            "next_action": "No action needed.",
        }

    target = _plan_target(str(normalized_task.get("task_type") or "chat"))
    if target is None:
        return {
            "needed": True,
            "approved": False,
            "plan": [],
            "next_action": "No approved local install plan exists for this task yet.",
        }

    model_name = str(target.get("model") or "").strip()
    canonical_id = f"ollama:{model_name}"
    return {
        "needed": True,
        "approved": True,
        "plan": [
            {
                "id": "01_pull_model",
                "action": "ollama.pull_model",
                "provider": "ollama",
                "model": model_name,
                "reason": "approved_local_model_missing",
                "size": str(target.get("size") or "unknown"),
            },
            {
                "id": "02_set_chat_default",
                "action": "defaults.set_chat_model",
                "provider": "ollama",
                "model": canonical_id,
                "reason": "promote_installed_local_model",
            },
            {
                "id": "03_probe_model",
                "action": "provider.test",
                "provider": "ollama",
                "model": model_name,
                "reason": "verify_model_callable",
            },
        ],
        "next_action": f"Review install plan for {canonical_id}.",
    }


__all__ = ["build_install_plan"]
