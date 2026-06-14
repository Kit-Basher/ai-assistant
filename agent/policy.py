from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


@dataclass
class PolicyDecision:
    allowed: bool
    requires_confirmation: bool
    reason: str | None = None


@dataclass(frozen=True)
class OperationPolicyDecision:
    action_type: str
    classification: str
    allowed_without_confirmation: bool
    requires_plan: bool
    requires_confirmation: bool
    reason: str | None = None


READ_ONLY_ACTION_TYPES = {
    "read",
    "list",
    "search",
    "status",
    "preview",
    "query",
    "diagnose",
    "compare",
    "check",
    "health",
    "get",
}

MUTATING_ACTION_TYPES = {
    "apply",
    "approve",
    "clear",
    "configure",
    "create",
    "delete",
    "disable",
    "enable",
    "grant",
    "import",
    "install",
    "pull",
    "record",
    "remove",
    "reset",
    "restart",
    "run",
    "setup",
    "start",
    "stop",
    "update",
    "write",
}

READ_ONLY_OPERATIONS = {
    "external_pack.list",
    "external_pack.preview",
    "external_pack.search",
    "external_pack.status",
    "managed_local_service.status",
    "managed_local_service.setup_plan",
    "safe_web_search.query",
    "safe_web_search.status",
}

MUTATING_OPERATIONS = {
    "external_pack.approve",
    "external_pack.enable",
    "external_pack.grant",
    "external_pack.install",
    "external_pack.remove",
    "managed_local_service.setup_apply",
}


def check_permissions(skill_permissions: list[str], requested: list[str]) -> bool:
    return all(perm in skill_permissions for perm in requested)


def classify_operation(action_type: str | None) -> OperationPolicyDecision:
    normalized = str(action_type or "").strip().lower()
    if normalized in READ_ONLY_OPERATIONS:
        return OperationPolicyDecision(
            action_type=normalized,
            classification="read_only",
            allowed_without_confirmation=True,
            requires_plan=False,
            requires_confirmation=False,
        )
    if normalized in MUTATING_OPERATIONS:
        return OperationPolicyDecision(
            action_type=normalized,
            classification="mutating",
            allowed_without_confirmation=False,
            requires_plan=True,
            requires_confirmation=True,
        )
    tokens = {item for item in normalized.replace("-", "_").replace("/", ".").split(".") if item}
    if tokens & READ_ONLY_ACTION_TYPES and not (tokens & MUTATING_ACTION_TYPES):
        return OperationPolicyDecision(
            action_type=normalized,
            classification="read_only",
            allowed_without_confirmation=True,
            requires_plan=False,
            requires_confirmation=False,
        )
    return OperationPolicyDecision(
        action_type=normalized or "unknown",
        classification="mutating",
        allowed_without_confirmation=False,
        requires_plan=True,
        requires_confirmation=True,
        reason="unknown_operations_default_to_mutating" if normalized not in MUTATING_OPERATIONS else None,
    )


def build_mutator_plan(
    *,
    action_type: str,
    resources: dict[str, list[str]] | None = None,
    rollback_scope: str,
    rollback_supported: bool,
    confirmation_token: str,
    expires_at: int | float,
    plan_id: str | None = None,
) -> dict[str, Any]:
    decision = classify_operation(action_type)
    if decision.classification != "mutating":
        raise ValueError("read_only_operations_do_not_need_mutator_plans")
    return {
        "policy_layer": "plan_mode",
        "action_type": str(action_type or "").strip().lower(),
        "classification": "mutating",
        "resources": {
            "created": [str(item) for item in (resources or {}).get("created", [])],
            "changed": [str(item) for item in (resources or {}).get("changed", [])],
            "deleted": [str(item) for item in (resources or {}).get("deleted", [])],
        },
        "rollback_scope": str(rollback_scope or "").strip(),
        "rollback_supported": bool(rollback_supported),
        "requires_confirmation": True,
        "confirmation_token": str(confirmation_token or "").strip(),
        "expires_at": int(expires_at),
        "plan_id": str(plan_id or "").strip() or None,
    }


def validate_mutator_apply(
    plan: dict[str, Any] | None,
    *,
    expected_action_type: str,
    confirmation_token: str,
    now: int | float | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(plan, dict):
        return False, "plan_required"
    if str(plan.get("policy_layer") or "") != "plan_mode":
        return False, "plan_policy_missing"
    if str(plan.get("classification") or "") != "mutating":
        return False, "plan_not_mutating"
    if str(plan.get("action_type") or "").strip().lower() != str(expected_action_type or "").strip().lower():
        return False, "plan_action_type_mismatch"
    if not str(plan.get("rollback_scope") or "").strip():
        return False, "plan_rollback_scope_missing"
    if "rollback_supported" not in plan:
        return False, "plan_rollback_support_missing"
    resources = plan.get("resources")
    if not isinstance(resources, dict):
        return False, "plan_resources_missing"
    for key in ("created", "changed", "deleted"):
        if not isinstance(resources.get(key), list):
            return False, "plan_resources_invalid"
    if str(plan.get("confirmation_token") or "").strip() != str(confirmation_token or "").strip():
        return False, "invalid_confirmation"
    try:
        expires_at = float(plan.get("expires_at") or 0)
    except (TypeError, ValueError):
        return False, "plan_expiration_invalid"
    if expires_at <= float(now if now is not None else time.time()):
        return False, "confirmation_expired"
    return True, None


def mutator_confirmation_required_payload(action_type: str, *, reason: str | None = None) -> dict[str, Any]:
    decision = classify_operation(action_type)
    return {
        "ok": False,
        "error": "confirmation_required",
        "error_kind": "confirmation_required",
        "action_type": decision.action_type,
        "classification": decision.classification,
        "requires_plan": decision.requires_plan,
        "requires_confirmation": decision.requires_confirmation,
        "reason": reason or decision.reason or "mutating_operation_requires_plan_preview_confirmation",
    }


def requires_confirmation(action: dict[str, Any]) -> bool:
    # Treat delete/overwrite and ops restarts as sensitive.
    action_type = action.get("action_type")
    if action_type in {"delete", "overwrite", "ops_restart"}:
        return True
    return False


def evaluate_policy(skill_permissions: list[str], requested: list[str], action: dict[str, Any]) -> PolicyDecision:
    if not check_permissions(skill_permissions, requested):
        return PolicyDecision(False, False, "permission_denied")
    if requires_confirmation(action):
        return PolicyDecision(True, True, None)
    return PolicyDecision(True, False, None)
