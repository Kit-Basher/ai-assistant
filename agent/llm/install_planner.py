from __future__ import annotations

from typing import Any, Iterable

from agent.llm.approved_local_models import approved_local_profile_for_ref, approved_local_profiles_for_task
from agent.llm.control_contract import normalize_model_inventory, normalize_selection_result, normalize_task_request
from agent.llm.model_state import build_effective_model_state


def _build_plan_steps(*, profile: dict[str, Any], task_type: str) -> list[dict[str, Any]]:
    model_name = str(profile.get("install_name") or "").strip()
    canonical_id = str(profile.get("id") or "").strip()
    steps: list[dict[str, Any]] = [
        {
            "id": "01_pull_model",
            "action": "ollama.pull_model",
            "provider": "ollama",
            "model": model_name,
            "reason": "approved_local_model_missing",
            "size": str(profile.get("size_hint") or "unknown"),
        },
        {
            "id": "02_probe_model",
            "action": "provider.test",
            "provider": "ollama",
            "model": model_name,
            "reason": "verify_model_callable",
        },
    ]
    if task_type in {"chat", "health", "tool_use"}:
        steps.insert(
            1,
            {
                "id": "02_set_chat_default",
                "action": "defaults.set_chat_model",
                "provider": "ollama",
                "model": canonical_id,
                "reason": "promote_installed_local_model",
            },
        )
        steps[2]["id"] = "03_probe_model"
    return steps


def _already_satisfied_plan(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    candidates = _profile_candidates(profile) if isinstance(profile, dict) else []
    return {
        "needed": False,
        "approved": True,
        "plan": [],
        "candidates": candidates,
        "install_command": None,
        "approval_required": False,
        "reason": "already_satisfied",
        "next_action": "No action needed.",
    }


def _profile_candidates(profile: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "model_id": str(profile.get("id") or "").strip(),
            "install_name": str(profile.get("install_name") or "").strip(),
            "reason": str(profile.get("reason") or "approved_local_profile").strip() or "approved_local_profile",
            "size_hint": str(profile.get("size_hint") or "unknown"),
            "preferred": bool(profile.get("preferred", False)),
            "min_memory_gb": int(profile.get("min_memory_gb") or 0),
        }
    ]


def _profile_install_plan(*, profile: dict[str, Any], task_type: str, reason: str) -> dict[str, Any]:
    model_id = str(profile.get("id") or "").strip()
    install_name = str(profile.get("install_name") or "").strip()
    return {
        "needed": True,
        "approved": True,
        "reason": str(reason or "no_suitable_model"),
        "candidates": _profile_candidates(profile),
        "install_command": f"ollama pull {install_name}",
        "approval_required": True,
        "plan": _build_plan_steps(profile=profile, task_type=task_type),
        "next_action": f"Run: python -m agent llm_install --model {model_id} --approve",
    }


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
                return _already_satisfied_plan()

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
        return _already_satisfied_plan()

    profiles = approved_local_profiles_for_task(normalized_task, limit=2)
    if not profiles:
        return {
            "needed": True,
            "approved": False,
            "plan": [],
            "candidates": [],
            "install_command": None,
            "approval_required": False,
            "reason": str(normalized_selection.get("reason") or "no_approved_local_model"),
            "next_action": "No approved local install plan exists for this task yet.",
        }

    task_type = str(normalized_task.get("task_type") or "chat")
    preferred = profiles[0]
    plan = _profile_install_plan(
        profile=preferred,
        task_type=task_type,
        reason=str(normalized_selection.get("reason") or "no_suitable_model"),
    )
    plan["candidates"] = [
        {
            "model_id": str(profile.get("id") or "").strip(),
            "install_name": str(profile.get("install_name") or "").strip(),
            "reason": str(profile.get("reason") or "approved_local_profile").strip() or "approved_local_profile",
            "size_hint": str(profile.get("size_hint") or "unknown"),
            "preferred": bool(profile.get("preferred", False)),
            "min_memory_gb": int(profile.get("min_memory_gb") or 0),
        }
        for profile in profiles
    ]
    return plan


def build_install_plan_for_model(
    *,
    inventory: Iterable[dict[str, Any]],
    model_ref: str,
) -> dict[str, Any]:
    normalized_inventory = normalize_model_inventory(inventory)
    profile = approved_local_profile_for_ref(model_ref)
    if profile is None:
        return {
            "needed": True,
            "approved": False,
            "plan": [],
            "candidates": [],
            "install_command": None,
            "approval_required": False,
            "reason": "model_not_approved",
            "next_action": "No approved local install plan exists for this model.",
        }
    model_id = str(profile.get("id") or "").strip()
    existing = next((item for item in normalized_inventory if str(item.get("id") or "") == model_id), None)
    if isinstance(existing, dict):
        installed = bool(existing.get("installed", False))
        healthy = bool(existing.get("healthy", False))
        available = bool(existing.get("available", False))
        if installed and healthy and available:
            return _already_satisfied_plan(profile=profile)
    task_types = profile.get("task_types") if isinstance(profile.get("task_types"), tuple) else tuple()
    task_type = str(task_types[0] if task_types else "chat")
    return _profile_install_plan(profile=profile, task_type=task_type, reason="explicit_model_install")


__all__ = ["build_install_plan", "build_install_plan_for_model"]
