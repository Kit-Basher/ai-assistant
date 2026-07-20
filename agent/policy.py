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


PLAN_MODE_SCHEMA_VERSION = "plan_mode.v2"


@dataclass(frozen=True)
class CanonicalPlan:
    plan_id: str
    action_type: str
    target: str
    scope: str
    mutation_level: str
    resources_affected: list[str]
    risk_level: str
    rollback_scope: str
    rollback_supported: bool
    executor_status: str
    confirmation_words: list[str]
    expires_at: int
    staleness_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_layer": "plan_mode",
            "schema_version": PLAN_MODE_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "action_type": self.action_type,
            "target": self.target,
            "scope": self.scope,
            "mutation_level": self.mutation_level,
            "resources_affected": list(self.resources_affected),
            "risk_level": self.risk_level,
            "rollback_scope": self.rollback_scope,
            "rollback_supported": bool(self.rollback_supported),
            "executor_status": self.executor_status,
            "confirmation_words": list(self.confirmation_words),
            "allowed_confirmation_words": list(self.confirmation_words),
            "expires_at": int(self.expires_at),
            "staleness_policy": self.staleness_policy,
            "requires_confirmation": True,
        }


def build_canonical_plan(
    *,
    plan_id: str,
    action_type: str,
    target: str,
    scope: str,
    mutation_level: str,
    resources_affected: list[str] | tuple[str, ...] | None,
    risk_level: str,
    rollback_scope: str,
    rollback_supported: bool,
    executor_status: str,
    expires_at: int | float,
    confirmation_words: list[str] | tuple[str, ...] | None = None,
    staleness_policy: str | None = None,
) -> dict[str, Any]:
    clean_words = [str(item).strip().lower() for item in (confirmation_words or ("yes", "confirm")) if str(item).strip()]
    if not clean_words:
        clean_words = ["yes", "confirm"]
    plan = CanonicalPlan(
        plan_id=str(plan_id or "").strip(),
        action_type=str(action_type or "unknown").strip().lower() or "unknown",
        target=str(target or "unspecified").strip() or "unspecified",
        scope=str(scope or "current user/session").strip() or "current user/session",
        mutation_level=str(mutation_level or "mutating").strip().lower() or "mutating",
        resources_affected=[str(item).strip() for item in (resources_affected or []) if str(item).strip()],
        risk_level=str(risk_level or "medium").strip().lower() or "medium",
        rollback_scope=str(rollback_scope or "Only the resources listed in this plan.").strip()
        or "Only the resources listed in this plan.",
        rollback_supported=bool(rollback_supported),
        executor_status=str(executor_status or "unavailable").strip().lower() or "unavailable",
        confirmation_words=clean_words,
        expires_at=int(expires_at),
        staleness_policy=str(staleness_policy or "Plan expires, is canceled, or is lost on service restart.").strip()
        or "Plan expires, is canceled, or is lost on service restart.",
    )
    return plan.to_dict()


READ_ONLY_ACTION_TYPES = {
    "read",
    "list",
    "observe",
    "report",
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
    # Unknown operations fail closed as mutating. This legacy helper remains for
    # built-in skills, but it must follow the same classification baseline as
    # Universal Plan Mode.
    action_type = str(action.get("action_type") or "").strip()
    if not action_type:
        return False
    return classify_operation(action_type).requires_confirmation


def evaluate_policy(skill_permissions: list[str], requested: list[str], action: dict[str, Any]) -> PolicyDecision:
    if not check_permissions(skill_permissions, requested):
        return PolicyDecision(False, False, "permission_denied")
    if requires_confirmation(action):
        return PolicyDecision(True, True, None)
    return PolicyDecision(True, False, None)
