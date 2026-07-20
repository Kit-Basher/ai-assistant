from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import re
import time
import uuid
from typing import Any


POLICY_SCHEMA_VERSION = 1

CAPABILITY_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")

EFFECTS = {"read_only", "mutating"}
SCOPES = {"local_process", "local_filesystem", "local_host", "external_service"}
REVERSIBILITY = {"reversible", "conditionally_reversible", "irreversible"}
RISK_LEVELS = {"low", "medium", "high", "critical"}
AUTHORIZATION_MODES = {"allow", "plan", "plan_and_confirm", "local_activation_and_confirm", "deny"}
IMPLEMENTATION_STATUSES = {"implemented", "legacy_unmigrated", "unimplemented"}

REASON_ALLOWED = "allowed"
REASON_READ_ONLY_ALLOWED = "read_only_allowed"
REASON_PLAN_REQUIRED = "plan_required"
REASON_CONFIRMATION_REQUIRED = "confirmation_required"
REASON_LOCAL_ACTIVATION_REQUIRED = "local_activation_required"
REASON_CAPABILITY_DENIED = "capability_denied"
REASON_CAPABILITY_UNKNOWN = "capability_unknown"
REASON_CAPABILITY_UNIMPLEMENTED = "capability_unimplemented"
REASON_STALE_PLAN = "stale_plan"
REASON_TARGET_CHANGED = "target_changed"
REASON_POLICY_CHANGED = "policy_changed"
REASON_ACTIVATION_INVALID = "activation_invalid"
REASON_CONFLICTING_OPERATION = "conflicting_operation"
REASON_GENERIC_BYPASS_BLOCKED = "generic_bypass_blocked"
REASON_MUTATION_PLAN_MISSING = "mutation_plan_missing"
REASON_MUTATION_PLAN_INVALID = "mutation_plan_invalid"
REASON_MUTATION_PLAN_EXPIRED = "mutation_plan_expired"
REASON_MUTATION_PLAN_CANCELLED = "mutation_plan_cancelled"
REASON_MUTATION_PLAN_FINGERPRINT_MISMATCH = "mutation_plan_fingerprint_mismatch"
REASON_MUTATION_PLAN_TARGET_CHANGED = "mutation_plan_target_changed"
REASON_MUTATION_PLAN_POLICY_CHANGED = "mutation_plan_policy_changed"
REASON_MUTATION_PLAN_ACTIVATION_CHANGED = "mutation_plan_activation_changed"
REASON_MUTATION_PLAN_THREAD_MISMATCH = "mutation_plan_thread_mismatch"
REASON_MUTATION_PLAN_SESSION_MISMATCH = "mutation_plan_session_mismatch"

TRUSTED_CALLER_TYPES = {"core", "executor", "skill_pack", "lifecycle_runner", "fixture"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_capability_id(capability_id: str) -> str:
    normalized = str(capability_id or "").strip()
    if CAPABILITY_ID_RE.fullmatch(normalized) is None:
        raise ValueError("invalid_capability_id")
    return normalized


@dataclass(frozen=True)
class CapabilityDefinition:
    schema_version: int
    capability_id: str
    title: str
    effect: str
    scope: str
    reversibility: str
    risk_level: str
    authorization_mode: str
    receipt_required: bool
    runtime_revalidation_required: bool
    target_binding_required: bool
    external_side_effect: bool
    generic_bypass_forbidden: bool
    implementation_status: str = "implemented"
    activation_policy_id: str | None = None
    recovery_capability: str | None = None

    def validate(self) -> None:
        if int(self.schema_version) != POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported_capability_schema_version")
        validate_capability_id(self.capability_id)
        if not str(self.title or "").strip():
            raise ValueError("capability_title_required")
        _require_enum("effect", self.effect, EFFECTS)
        _require_enum("scope", self.scope, SCOPES)
        _require_enum("reversibility", self.reversibility, REVERSIBILITY)
        _require_enum("risk_level", self.risk_level, RISK_LEVELS)
        _require_enum("authorization_mode", self.authorization_mode, AUTHORIZATION_MODES)
        _require_enum("implementation_status", self.implementation_status, IMPLEMENTATION_STATUSES)
        if self.effect == "read_only" and self.authorization_mode not in {"allow", "plan", "deny"}:
            raise ValueError("read_only_capability_has_mutation_authorization")
        if self.effect == "read_only" and self.receipt_required:
            raise ValueError("read_only_capability_requires_mutation_receipt")
        if self.effect == "mutating" and self.authorization_mode == "allow":
            raise ValueError("mutating_capability_cannot_be_unconditionally_allowed")
        if self.risk_level == "critical" and self.authorization_mode not in {"local_activation_and_confirm", "deny"}:
            raise ValueError("critical_capability_requires_activation_or_deny")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "capability_id": self.capability_id,
            "title": self.title,
            "effect": self.effect,
            "scope": self.scope,
            "reversibility": self.reversibility,
            "risk_level": self.risk_level,
            "authorization_mode": self.authorization_mode,
            "receipt_required": bool(self.receipt_required),
            "runtime_revalidation_required": bool(self.runtime_revalidation_required),
            "target_binding_required": bool(self.target_binding_required),
            "external_side_effect": bool(self.external_side_effect),
            "generic_bypass_forbidden": bool(self.generic_bypass_forbidden),
            "implementation_status": self.implementation_status,
            "activation_policy_id": self.activation_policy_id,
            "recovery_capability": self.recovery_capability,
        }


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    capability_id: str
    authorization_mode: str
    decision: str
    reason_code: str
    mutation_allowed: bool
    plan_required: bool
    confirmation_required: bool
    local_activation_required: bool
    receipt_required: bool
    policy_version: int = POLICY_SCHEMA_VERSION
    decision_id: str = field(default_factory=lambda: f"authz-{uuid.uuid4().hex[:12]}")
    risk_level: str = ""
    capability_title: str = ""
    target_fingerprint: str = ""
    plan_fingerprint: str = ""
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": bool(self.allowed),
            "capability_id": self.capability_id,
            "authorization_mode": self.authorization_mode,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "mutation_allowed": bool(self.mutation_allowed),
            "plan_required": bool(self.plan_required),
            "confirmation_required": bool(self.confirmation_required),
            "local_activation_required": bool(self.local_activation_required),
            "receipt_required": bool(self.receipt_required),
            "policy_version": int(self.policy_version),
            "decision_id": self.decision_id,
            "risk_level": self.risk_level,
            "capability_title": self.capability_title,
            "target_fingerprint": self.target_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TrustedInvocationContext:
    capability_id: str
    executor_id: str
    authorization_decision_id: str
    plan_fingerprint: str
    operation_id: str
    policy_version: int = POLICY_SCHEMA_VERSION
    caller_type: str = "core"
    skill_pack_id: str = ""
    skill_pack_version: str = ""
    skill_pack_fingerprint: str = ""
    permission_id: str = ""
    grant_id: str = ""
    caller_id: str = ""
    source_module: str = ""
    source_surface: str = ""
    issued_at: str = field(default_factory=utc_now_iso)
    expires_at: str = ""
    single_use: bool = True
    parent_operation_id: str = ""
    target_fingerprint: str = ""
    consumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "executor_id": self.executor_id,
            "authorization_decision_id": self.authorization_decision_id,
            "plan_fingerprint": self.plan_fingerprint,
            "operation_id": self.operation_id,
            "policy_version": int(self.policy_version),
            "caller_type": self.caller_type,
            "skill_pack_id": self.skill_pack_id,
            "skill_pack_version": self.skill_pack_version,
            "skill_pack_fingerprint": self.skill_pack_fingerprint,
            "permission_id": self.permission_id,
            "grant_id": self.grant_id,
            "caller_id": self.caller_id,
            "source_module": self.source_module,
            "source_surface": self.source_surface,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "single_use": bool(self.single_use),
            "parent_operation_id": self.parent_operation_id,
            "target_fingerprint": self.target_fingerprint,
            "consumed": bool(self.consumed),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TrustedInvocationContext | None":
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                capability_id=str(payload.get("capability_id") or ""),
                executor_id=str(payload.get("executor_id") or ""),
                authorization_decision_id=str(payload.get("authorization_decision_id") or ""),
                plan_fingerprint=str(payload.get("plan_fingerprint") or ""),
                operation_id=str(payload.get("operation_id") or ""),
                policy_version=int(payload.get("policy_version") or 0),
                caller_type=str(payload.get("caller_type") or "core"),
                skill_pack_id=str(payload.get("skill_pack_id") or ""),
                skill_pack_version=str(payload.get("skill_pack_version") or ""),
                skill_pack_fingerprint=str(payload.get("skill_pack_fingerprint") or ""),
                permission_id=str(payload.get("permission_id") or ""),
                grant_id=str(payload.get("grant_id") or ""),
                caller_id=str(payload.get("caller_id") or ""),
                source_module=str(payload.get("source_module") or ""),
                source_surface=str(payload.get("source_surface") or ""),
                issued_at=str(payload.get("issued_at") or utc_now_iso()),
                expires_at=str(payload.get("expires_at") or ""),
                single_use=bool(payload.get("single_use", True)),
                parent_operation_id=str(payload.get("parent_operation_id") or ""),
                target_fingerprint=str(payload.get("target_fingerprint") or ""),
                consumed=bool(payload.get("consumed", False)),
            )
        except (TypeError, ValueError):
            return None


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, CapabilityDefinition] = {}

    def register(self, definition: CapabilityDefinition) -> None:
        definition.validate()
        capability_id = definition.capability_id
        if capability_id in self._capabilities:
            raise ValueError(f"duplicate_capability_id:{capability_id}")
        self._capabilities[capability_id] = definition

    def get(self, capability_id: str) -> CapabilityDefinition | None:
        try:
            normalized = validate_capability_id(capability_id)
        except ValueError:
            return None
        return self._capabilities.get(normalized)

    def list(self) -> list[CapabilityDefinition]:
        return [self._capabilities[key] for key in sorted(self._capabilities)]

    def status_categories(self) -> dict[str, list[dict[str, Any]]]:
        buckets = {
            "available_without_confirmation": [],
            "requires_plan": [],
            "requires_confirmation": [],
            "requires_local_activation": [],
            "unavailable": [],
            "legacy_unmigrated": [],
        }
        for definition in self.list():
            item = {
                "capability_id": definition.capability_id,
                "title": definition.title,
                "risk_level": definition.risk_level,
                "effect": definition.effect,
                "authorization_mode": definition.authorization_mode,
                "implementation_status": definition.implementation_status,
            }
            if definition.implementation_status == "legacy_unmigrated":
                buckets["legacy_unmigrated"].append(item)
            elif definition.implementation_status == "unimplemented" or definition.authorization_mode == "deny":
                buckets["unavailable"].append(item)
            elif definition.authorization_mode == "allow":
                buckets["available_without_confirmation"].append(item)
            elif definition.authorization_mode == "plan":
                buckets["requires_plan"].append(item)
            elif definition.authorization_mode == "plan_and_confirm":
                buckets["requires_confirmation"].append(item)
            elif definition.authorization_mode == "local_activation_and_confirm":
                buckets["requires_local_activation"].append(item)
        return buckets


def build_default_capability_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    for definition in _default_capabilities():
        registry.register(definition)
    return registry


def capability_for_action_type(action_type: str) -> str | None:
    return {
        "package.install": "system.package.install",
        "operator.support_bundle": "support_bundle.create",
        "operator.backup": "backup.create",
        "operator.restore": "restore.execute",
        "operator.cleanup": "cleanup.execute",
        "operator.update": "system.update",
        "operator.uninstall": "system.uninstall",
        "memory.delete_all": "memory.forget",
        "memory.export": "memory.export",
        "memory.redact": "memory.redact",
        "memory.cleanup": "memory.compact",
        "operator.file.create": "files.create",
        "operator.file.modify": "files.modify",
        "operator.file.delete": "files.delete",
        "operator.git.commit": "git.commit",
        "operator.git.push": "git.push",
        "operator.service.restart": "system.service.restart",
        "operator.notification.local.send": "notification.local.send",
        "operator.notification.telegram.send": "notification.external.send",
        "operator.notification.mark_read": "notification.mark_read",
        "operator.notification.prune": "notification.prune",
        "operator.skill_pack.permission.grant": "skill_pack.permission.grant",
        "operator.skill_pack.permission.revoke": "skill_pack.permission.revoke",
        "system.package.inspect": "system.package.inspect",
        "system.service.inspect": "system.service.inspect",
        "system.lifecycle.status": "system.lifecycle.status",
        "notification.inspect": "notification.inspect",
        "skill_pack.inspect": "skill_pack.inspect",
        "skill_pack.permissions.inspect": "skill_pack.permissions.inspect",
    }.get(str(action_type or "").strip().lower())


def authorize_capability(
    capability_id: str,
    *,
    request_context: dict[str, Any] | None = None,
    target_snapshot: dict[str, Any] | None = None,
    plan_context: dict[str, Any] | None = None,
    confirmation_context: dict[str, Any] | None = None,
    activation_context: dict[str, Any] | None = None,
    registry: CapabilityRegistry | None = None,
) -> AuthorizationDecision:
    reg = registry or build_default_capability_registry()
    definition = reg.get(capability_id)
    target_snapshot = dict(target_snapshot or {})
    plan_context = dict(plan_context or {})
    confirmation_context = dict(confirmation_context or {})
    activation_context = dict(activation_context or {})
    if definition is None:
        return _decision(None, capability_id, False, "denied", REASON_CAPABILITY_UNKNOWN)
    if definition.implementation_status == "unimplemented":
        return _decision(definition, capability_id, False, "denied", REASON_CAPABILITY_UNIMPLEMENTED)
    if definition.implementation_status == "legacy_unmigrated":
        return _decision(definition, capability_id, False, "denied", REASON_CAPABILITY_UNIMPLEMENTED)
    if definition.authorization_mode == "deny":
        return _decision(definition, capability_id, False, "denied", REASON_CAPABILITY_DENIED)
    if definition.effect == "read_only" and definition.authorization_mode == "allow":
        return _decision(definition, capability_id, True, "allowed", REASON_READ_ONLY_ALLOWED)

    plan_fingerprint = str(plan_context.get("plan_fingerprint") or "").strip()
    target_fingerprint = str(target_snapshot.get("target_fingerprint") or target_snapshot.get("fingerprint") or "").strip()
    expected_target = str(plan_context.get("target_fingerprint") or "").strip()
    if definition.authorization_mode in {"plan", "plan_and_confirm", "local_activation_and_confirm"} and not plan_fingerprint:
        return _decision(definition, capability_id, False, "blocked", REASON_PLAN_REQUIRED, target_fingerprint=target_fingerprint)
    if expected_target and target_fingerprint and expected_target != target_fingerprint:
        return _decision(definition, capability_id, False, "blocked", REASON_TARGET_CHANGED, target_fingerprint=target_fingerprint, plan_fingerprint=plan_fingerprint)
    plan_policy_version = int(plan_context.get("policy_version") or POLICY_SCHEMA_VERSION)
    if plan_policy_version != POLICY_SCHEMA_VERSION:
        return _decision(definition, capability_id, False, "blocked", REASON_POLICY_CHANGED, target_fingerprint=target_fingerprint, plan_fingerprint=plan_fingerprint)
    if bool(plan_context.get("stale")):
        return _decision(definition, capability_id, False, "blocked", REASON_STALE_PLAN, target_fingerprint=target_fingerprint, plan_fingerprint=plan_fingerprint)

    if definition.authorization_mode == "plan":
        return _decision(definition, capability_id, True, "allowed", REASON_ALLOWED, target_fingerprint=target_fingerprint, plan_fingerprint=plan_fingerprint)
    if definition.authorization_mode in {"plan_and_confirm", "local_activation_and_confirm"} and not bool(confirmation_context.get("confirmed")):
        return _decision(definition, capability_id, False, "blocked", REASON_CONFIRMATION_REQUIRED, target_fingerprint=target_fingerprint, plan_fingerprint=plan_fingerprint)
    if definition.authorization_mode == "local_activation_and_confirm":
        if not bool(activation_context.get("valid")):
            reason = str(activation_context.get("reason_code") or REASON_LOCAL_ACTIVATION_REQUIRED)
            if reason == "allowed":
                reason = REASON_LOCAL_ACTIVATION_REQUIRED
            return _decision(definition, capability_id, False, "blocked", reason, target_fingerprint=target_fingerprint, plan_fingerprint=plan_fingerprint)

    return _decision(definition, capability_id, True, "allowed", REASON_ALLOWED, target_fingerprint=target_fingerprint, plan_fingerprint=plan_fingerprint)


def validate_trusted_invocation_context(
    payload: dict[str, Any] | None,
    *,
    capability_id: str,
    executor_id: str,
    plan_fingerprint: str,
    operation_id: str = "",
    target_fingerprint: str = "",
    runtime_mode: str = "production",
) -> tuple[bool, str, TrustedInvocationContext | None]:
    context = TrustedInvocationContext.from_dict(payload)
    if context is None:
        return False, REASON_GENERIC_BYPASS_BLOCKED, None
    if context.consumed:
        return False, "trusted_context_consumed", context
    if context.policy_version != POLICY_SCHEMA_VERSION:
        return False, REASON_POLICY_CHANGED, context
    if context.caller_type not in TRUSTED_CALLER_TYPES:
        return False, "caller_type_invalid", context
    if context.caller_type == "fixture" and str(runtime_mode or "production") == "production":
        return False, "fixture_context_denied_in_production", context
    if context.capability_id != capability_id:
        return False, "capability_mismatch", context
    if context.executor_id != executor_id:
        return False, "executor_mismatch", context
    if not context.authorization_decision_id.startswith("authz-"):
        return False, "authorization_decision_missing", context
    if not context.operation_id:
        return False, "operation_id_missing", context
    if operation_id and context.operation_id != operation_id:
        return False, "operation_id_mismatch", context
    if plan_fingerprint and context.plan_fingerprint != plan_fingerprint:
        return False, REASON_STALE_PLAN, context
    if target_fingerprint and context.target_fingerprint and context.target_fingerprint != target_fingerprint:
        return False, REASON_TARGET_CHANGED, context
    if context.expires_at:
        try:
            parsed = datetime.fromisoformat(context.expires_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if parsed.timestamp() <= time.time():
                return False, "trusted_context_expired", context
        except ValueError:
            return False, "trusted_context_expiry_invalid", context
    return True, REASON_ALLOWED, context


def _require_enum(field_name: str, value: str, allowed: set[str]) -> None:
    if str(value or "").strip() not in allowed:
        raise ValueError(f"invalid_{field_name}")


def _decision(
    definition: CapabilityDefinition | None,
    capability_id: str,
    allowed: bool,
    decision: str,
    reason_code: str,
    *,
    target_fingerprint: str = "",
    plan_fingerprint: str = "",
) -> AuthorizationDecision:
    mode = definition.authorization_mode if definition is not None else "deny"
    return AuthorizationDecision(
        allowed=allowed,
        capability_id=capability_id,
        authorization_mode=mode,
        decision=decision,
        reason_code=reason_code,
        mutation_allowed=bool(allowed and definition is not None and definition.effect == "mutating"),
        plan_required=bool(definition is not None and definition.authorization_mode in {"plan", "plan_and_confirm", "local_activation_and_confirm"}),
        confirmation_required=bool(definition is not None and definition.authorization_mode in {"plan_and_confirm", "local_activation_and_confirm"}),
        local_activation_required=bool(definition is not None and definition.authorization_mode == "local_activation_and_confirm"),
        receipt_required=bool(definition.receipt_required) if definition is not None else False,
        risk_level=definition.risk_level if definition is not None else "",
        capability_title=definition.title if definition is not None else "",
        target_fingerprint=target_fingerprint,
        plan_fingerprint=plan_fingerprint,
    )


def _capability(
    capability_id: str,
    title: str,
    effect: str,
    scope: str,
    reversibility: str,
    risk_level: str,
    authorization_mode: str,
    *,
    receipt_required: bool = False,
    runtime_revalidation_required: bool = False,
    target_binding_required: bool = False,
    external_side_effect: bool = False,
    generic_bypass_forbidden: bool = False,
    implementation_status: str = "implemented",
    activation_policy_id: str | None = None,
    recovery_capability: str | None = None,
) -> CapabilityDefinition:
    return CapabilityDefinition(
        schema_version=POLICY_SCHEMA_VERSION,
        capability_id=capability_id,
        title=title,
        effect=effect,
        scope=scope,
        reversibility=reversibility,
        risk_level=risk_level,
        authorization_mode=authorization_mode,
        receipt_required=receipt_required,
        runtime_revalidation_required=runtime_revalidation_required,
        target_binding_required=target_binding_required,
        external_side_effect=external_side_effect,
        generic_bypass_forbidden=generic_bypass_forbidden,
        implementation_status=implementation_status,
        activation_policy_id=activation_policy_id,
        recovery_capability=recovery_capability,
    )


def _default_capabilities() -> list[CapabilityDefinition]:
    return [
        _capability("system.inspect", "Inspect local system state", "read_only", "local_host", "reversible", "low", "allow"),
        _capability("system.package.inspect", "Inspect package state", "read_only", "local_host", "reversible", "low", "allow"),
        _capability("system.service.inspect", "Inspect service status", "read_only", "local_host", "reversible", "low", "allow"),
        _capability("system.lifecycle.status", "Inspect lifecycle operation status", "read_only", "local_host", "reversible", "low", "allow"),
        _capability("notification.inspect", "Inspect notification state", "read_only", "local_process", "reversible", "low", "allow"),
        _capability("skill_pack.inspect", "Inspect installed skill packs", "read_only", "local_process", "reversible", "low", "allow"),
        _capability("skill_pack.permissions.inspect", "Inspect skill-pack permissions", "read_only", "local_process", "reversible", "low", "allow"),
        _capability("provider.inspect", "Inspect provider configuration", "read_only", "local_process", "reversible", "low", "allow"),
        _capability("model.inspect", "Inspect model inventory and recommendations", "read_only", "local_process", "reversible", "low", "allow"),
        _capability("setup.inspect", "Inspect setup and recovery state", "read_only", "local_host", "reversible", "low", "allow"),
        _capability(
            "provider.configure", "Change provider configuration", "mutating", "local_filesystem",
            "conditionally_reversible", "high", "plan_and_confirm", receipt_required=True,
            runtime_revalidation_required=True, target_binding_required=True, generic_bypass_forbidden=True,
        ),
        _capability(
            "provider.secret.manage", "Change a provider secret", "mutating", "local_filesystem",
            "conditionally_reversible", "high", "plan_and_confirm", receipt_required=True,
            runtime_revalidation_required=True, target_binding_required=True, generic_bypass_forbidden=True,
        ),
        _capability(
            "secret.manage", "Change a bounded application secret", "mutating", "local_filesystem",
            "conditionally_reversible", "high", "plan_and_confirm", receipt_required=True,
            runtime_revalidation_required=True, target_binding_required=True, generic_bypass_forbidden=True,
        ),
        _capability(
            "model.configure", "Change model selection or routing", "mutating", "local_filesystem",
            "reversible", "high", "plan_and_confirm", receipt_required=True,
            runtime_revalidation_required=True, target_binding_required=True, generic_bypass_forbidden=True,
        ),
        _capability(
            "model.acquire", "Acquire or install a model", "mutating", "external_service",
            "conditionally_reversible", "high", "plan_and_confirm", receipt_required=True,
            runtime_revalidation_required=True, target_binding_required=True, external_side_effect=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "model.maintain", "Mutate model inventory or maintenance state", "mutating", "local_host",
            "conditionally_reversible", "high", "plan_and_confirm", receipt_required=True,
            runtime_revalidation_required=True, target_binding_required=True, generic_bypass_forbidden=True,
        ),
        _capability(
            "runtime.policy.configure", "Change runtime provider/model policy", "mutating", "local_filesystem",
            "reversible", "high", "plan_and_confirm", receipt_required=True,
            runtime_revalidation_required=True, target_binding_required=True, generic_bypass_forbidden=True,
        ),
        _capability(
            "setup.repair", "Apply a bounded setup or recovery repair", "mutating", "local_host",
            "conditionally_reversible", "high", "plan_and_confirm", receipt_required=True,
            runtime_revalidation_required=True, target_binding_required=True, generic_bypass_forbidden=True,
        ),
        _capability(
            "system.package.install",
            "Install local package",
            "mutating",
            "local_host",
            "conditionally_reversible",
            "high",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "cleanup.execute",
            "Execute local cleanup",
            "mutating",
            "local_filesystem",
            "conditionally_reversible",
            "high",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "system.update",
            "Update Personal Agent runtime",
            "mutating",
            "local_host",
            "conditionally_reversible",
            "high",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
            recovery_capability="system.update",
        ),
        _capability(
            "system.uninstall",
            "Uninstall Personal Agent runtime",
            "mutating",
            "local_host",
            "irreversible",
            "critical",
            "local_activation_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
            activation_policy_id="primary_uninstall_activation_policy.v1",
        ),
        _capability("backup.inspect", "Inspect Backup v1 artifacts", "read_only", "local_filesystem", "reversible", "low", "allow"),
        _capability("backup.validate", "Validate Backup v1 artifact", "read_only", "local_filesystem", "reversible", "low", "allow"),
        _capability(
            "backup.create",
            "Create Backup v1 artifact",
            "mutating",
            "local_filesystem",
            "reversible",
            "medium",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability("restore.inspect", "Inspect restore candidates", "read_only", "local_filesystem", "reversible", "low", "allow"),
        _capability(
            "restore.execute",
            "Restore Backup v1 preferences",
            "mutating",
            "local_filesystem",
            "conditionally_reversible",
            "high",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
            recovery_capability="backup.create",
        ),
        _capability("support_bundle.inspect", "Preview support bundle contents", "read_only", "local_filesystem", "reversible", "low", "allow"),
        _capability(
            "support_bundle.create",
            "Create redacted support bundle",
            "mutating",
            "local_filesystem",
            "reversible",
            "medium",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability("memory.inspect", "Inspect saved memory state", "read_only", "local_process", "reversible", "low", "allow"),
        _capability(
            "memory.forget",
            "Forget saved memory",
            "mutating",
            "local_process",
            "irreversible",
            "high",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "memory.export",
            "Export redacted memory",
            "mutating",
            "local_filesystem",
            "reversible",
            "medium",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "memory.redact",
            "Redact sensitive saved memory",
            "mutating",
            "local_process",
            "irreversible",
            "high",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "memory.compact",
            "Compact saved memory",
            "mutating",
            "local_process",
            "conditionally_reversible",
            "medium",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability("files.inspect", "Inspect local files", "read_only", "local_filesystem", "reversible", "low", "allow"),
        _capability("files.list", "List local files", "read_only", "local_filesystem", "reversible", "low", "allow"),
        _capability("files.diff", "Inspect file differences", "read_only", "local_filesystem", "reversible", "low", "allow"),
        _capability(
            "files.create",
            "Create local file",
            "mutating",
            "local_filesystem",
            "reversible",
            "medium",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "files.modify",
            "Modify local file",
            "mutating",
            "local_filesystem",
            "conditionally_reversible",
            "medium",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "files.delete",
            "Delete local file",
            "mutating",
            "local_filesystem",
            "conditionally_reversible",
            "high",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability("git.inspect", "Inspect Git repository", "read_only", "local_filesystem", "reversible", "low", "allow"),
        _capability("git.status", "Inspect Git status", "read_only", "local_filesystem", "reversible", "low", "allow"),
        _capability("git.diff", "Inspect Git diff", "read_only", "local_filesystem", "reversible", "low", "allow"),
        _capability("git.log", "Inspect Git log", "read_only", "local_filesystem", "reversible", "low", "allow"),
        _capability(
            "git.commit",
            "Create Git commit",
            "mutating",
            "local_filesystem",
            "conditionally_reversible",
            "medium",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "git.push",
            "Push Git commits",
            "mutating",
            "external_service",
            "conditionally_reversible",
            "high",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            external_side_effect=True,
            generic_bypass_forbidden=True,
        ),
        _capability("git.force_push", "Force push Git commits", "mutating", "external_service", "irreversible", "critical", "deny", implementation_status="unimplemented"),
        _capability("git.reset", "Reset Git repository", "mutating", "local_filesystem", "conditionally_reversible", "high", "deny", implementation_status="unimplemented"),
        _capability("git.clean", "Clean Git repository", "mutating", "local_filesystem", "irreversible", "critical", "deny", implementation_status="unimplemented"),
        _capability("system.service.logs.inspect", "Inspect service logs", "read_only", "local_host", "reversible", "low", "allow"),
        _capability(
            "system.service.restart",
            "Restart local service",
            "mutating",
            "local_host",
            "reversible",
            "high",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability("system.service.start", "Start local service", "mutating", "local_host", "reversible", "high", "deny", receipt_required=True, runtime_revalidation_required=True, target_binding_required=True, generic_bypass_forbidden=True, implementation_status="unimplemented"),
        _capability("system.service.stop", "Stop local service", "mutating", "local_host", "conditionally_reversible", "high", "deny", receipt_required=True, runtime_revalidation_required=True, target_binding_required=True, generic_bypass_forbidden=True, implementation_status="unimplemented"),
        _capability("system.service.disable", "Disable local service", "mutating", "local_host", "conditionally_reversible", "high", "deny", receipt_required=True, runtime_revalidation_required=True, target_binding_required=True, generic_bypass_forbidden=True, implementation_status="unimplemented"),
        _capability(
            "notification.local.send",
            "Create local notification record",
            "mutating",
            "local_filesystem",
            "reversible",
            "medium",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "notification.external.send",
            "Send configured external notification",
            "mutating",
            "external_service",
            "irreversible",
            "high",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            external_side_effect=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "notification.mark_read",
            "Mark notification history read",
            "mutating",
            "local_filesystem",
            "reversible",
            "medium",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "notification.prune",
            "Prune notification history",
            "mutating",
            "local_filesystem",
            "conditionally_reversible",
            "medium",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "skill_pack.permission.grant",
            "Grant skill-pack permission",
            "mutating",
            "local_process",
            "conditionally_reversible",
            "high",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
        _capability(
            "skill_pack.permission.revoke",
            "Revoke skill-pack permission",
            "mutating",
            "local_process",
            "reversible",
            "medium",
            "plan_and_confirm",
            receipt_required=True,
            runtime_revalidation_required=True,
            target_binding_required=True,
            generic_bypass_forbidden=True,
        ),
    ]
