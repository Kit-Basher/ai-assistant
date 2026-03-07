from __future__ import annotations

from typing import Any

from agent.llm.approved_local_models import approved_local_profile_for_ref


_ALLOWED_ACTIONS = {"ollama.pull_model", "provider.test", "defaults.set_chat_model"}


def validate_install_approval(plan: dict[str, Any] | None, *, approve: bool) -> dict[str, Any]:
    payload = dict(plan or {})
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else None
    model_id = str((candidate or {}).get("model_id") or "").strip() or None
    install_name = str((candidate or {}).get("install_name") or "").strip() or None
    if not bool(payload.get("needed", False)):
        return {
            "allowed": False,
            "approved": bool(payload.get("approved", False)),
            "error_kind": "already_satisfied",
            "message": "No install is needed.",
            "model_id": model_id,
            "install_name": install_name,
        }
    if not bool(payload.get("approved", False)):
        return {
            "allowed": False,
            "approved": False,
            "error_kind": "plan_not_approved",
            "message": "Only approved local Ollama install plans can be executed.",
            "model_id": model_id,
            "install_name": install_name,
        }
    if not model_id or not install_name:
        return {
            "allowed": False,
            "approved": True,
            "error_kind": "invalid_plan",
            "message": "Install plan is missing a concrete approved model candidate.",
            "model_id": model_id,
            "install_name": install_name,
        }
    if approved_local_profile_for_ref(model_id) is None or approved_local_profile_for_ref(install_name) is None:
        return {
            "allowed": False,
            "approved": True,
            "error_kind": "model_not_approved",
            "message": "Install plan refers to a model outside the approved local shortlist.",
            "model_id": model_id,
            "install_name": install_name,
        }
    steps = payload.get("plan") if isinstance(payload.get("plan"), list) else []
    for step in steps:
        if not isinstance(step, dict):
            return {
                "allowed": False,
                "approved": True,
                "error_kind": "invalid_plan",
                "message": "Install plan contains a malformed step.",
                "model_id": model_id,
                "install_name": install_name,
            }
        action = str(step.get("action") or "").strip()
        provider = str(step.get("provider") or "ollama").strip().lower()
        if action not in _ALLOWED_ACTIONS or provider != "ollama":
            return {
                "allowed": False,
                "approved": True,
                "error_kind": "unsupported_plan",
                "message": "Install plan contains an unsupported non-local action.",
                "model_id": model_id,
                "install_name": install_name,
            }
    if not approve:
        return {
            "allowed": False,
            "approved": True,
            "error_kind": "approval_required",
            "message": "Explicit approval is required before executing this local install.",
            "model_id": model_id,
            "install_name": install_name,
        }
    return {
        "allowed": True,
        "approved": True,
        "error_kind": None,
        "message": "Approved local Ollama install plan.",
        "model_id": model_id,
        "install_name": install_name,
    }


__all__ = ["validate_install_approval"]
