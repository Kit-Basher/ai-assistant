from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent.capability_policy import (
    POLICY_SCHEMA_VERSION,
    REASON_ALLOWED,
    REASON_GENERIC_BYPASS_BLOCKED,
    TrustedInvocationContext,
    validate_trusted_invocation_context,
)


class MutationPrimitiveClass(str, Enum):
    READ_ONLY = "READ_ONLY"
    INTERNAL_STATE_MUTATION = "INTERNAL_STATE_MUTATION"
    LOCAL_FILESYSTEM_MUTATION = "LOCAL_FILESYSTEM_MUTATION"
    HOST_CONTROL_MUTATION = "HOST_CONTROL_MUTATION"
    EXTERNAL_MUTATION = "EXTERNAL_MUTATION"
    SECURITY_SENSITIVE_MUTATION = "SECURITY_SENSITIVE_MUTATION"
    DENIED_PRIMITIVE = "DENIED_PRIMITIVE"


@dataclass(frozen=True)
class MutationPrimitivePolicy:
    primitive_id: str
    mutation_class: MutationPrimitiveClass
    allowed_caller_types: tuple[str, ...]
    trusted_context_required: bool
    capability_required: str
    executor_required: str
    target_binding_required: bool
    receipt_required: bool
    direct_use_prohibited: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "mutation_class": self.mutation_class.value,
            "allowed_caller_types": list(self.allowed_caller_types),
            "trusted_context_required": bool(self.trusted_context_required),
            "capability_required": self.capability_required,
            "executor_required": self.executor_required,
            "target_binding_required": bool(self.target_binding_required),
            "receipt_required": bool(self.receipt_required),
            "direct_use_prohibited": bool(self.direct_use_prohibited),
        }


def primitive_policy_registry() -> dict[str, MutationPrimitivePolicy]:
    entries = (
        MutationPrimitivePolicy("filesystem.write", MutationPrimitiveClass.LOCAL_FILESYSTEM_MUTATION, ("core", "executor", "skill_pack"), True, "files.create", "operator.file.create.v1", True, True, True),
        MutationPrimitivePolicy("filesystem.modify", MutationPrimitiveClass.LOCAL_FILESYSTEM_MUTATION, ("core", "executor", "skill_pack"), True, "files.modify", "operator.file.modify.v1", True, True, True),
        MutationPrimitivePolicy("filesystem.delete", MutationPrimitiveClass.LOCAL_FILESYSTEM_MUTATION, ("core", "executor", "skill_pack"), True, "files.delete", "operator.file.delete.v1", True, True, True),
        MutationPrimitivePolicy("git.commit", MutationPrimitiveClass.LOCAL_FILESYSTEM_MUTATION, ("core", "executor", "skill_pack"), True, "git.commit", "operator.git.commit.v1", True, True, True),
        MutationPrimitivePolicy("git.push", MutationPrimitiveClass.EXTERNAL_MUTATION, ("core", "executor"), True, "git.push", "operator.git.push.v1", True, True, True),
        MutationPrimitivePolicy("service.restart", MutationPrimitiveClass.HOST_CONTROL_MUTATION, ("core", "executor"), True, "system.service.restart", "operator.service.restart.v1", True, True, True),
        MutationPrimitivePolicy("notification.local.send", MutationPrimitiveClass.LOCAL_FILESYSTEM_MUTATION, ("core", "executor", "skill_pack"), True, "notification.local.send", "operator.notification.local.send.v1", True, True, True),
        MutationPrimitivePolicy("notification.external.send", MutationPrimitiveClass.EXTERNAL_MUTATION, ("core", "executor", "skill_pack"), True, "notification.external.send", "operator.notification.telegram.send.v1", True, True, True),
        MutationPrimitivePolicy("backup.create", MutationPrimitiveClass.LOCAL_FILESYSTEM_MUTATION, ("core", "executor", "skill_pack"), True, "backup.create", "operator.backup.v1", True, True, True),
        MutationPrimitivePolicy("restore.execute", MutationPrimitiveClass.INTERNAL_STATE_MUTATION, ("core", "executor"), True, "restore.execute", "operator.restore.v1", True, True, True),
        MutationPrimitivePolicy("support_bundle.create", MutationPrimitiveClass.LOCAL_FILESYSTEM_MUTATION, ("core", "executor"), True, "support_bundle.create", "operator.support_bundle.v1", True, True, True),
        MutationPrimitivePolicy("skill_pack.permission.grant", MutationPrimitiveClass.SECURITY_SENSITIVE_MUTATION, ("core", "executor"), True, "skill_pack.permission.grant", "operator.skill_pack.permission.grant.v1", True, True, True),
        MutationPrimitivePolicy("skill_pack.permission.revoke", MutationPrimitiveClass.SECURITY_SENSITIVE_MUTATION, ("core", "executor"), True, "skill_pack.permission.revoke", "operator.skill_pack.permission.revoke.v1", True, True, True),
        MutationPrimitivePolicy("generic.http.mutate", MutationPrimitiveClass.DENIED_PRIMITIVE, tuple(), True, "", "", True, False, True),
        MutationPrimitivePolicy("generic.shell.mutate", MutationPrimitiveClass.DENIED_PRIMITIVE, tuple(), True, "", "", True, False, True),
        MutationPrimitivePolicy("generic.db.domain_write", MutationPrimitiveClass.DENIED_PRIMITIVE, tuple(), True, "", "", True, False, True),
        MutationPrimitivePolicy("secret.raw_read", MutationPrimitiveClass.DENIED_PRIMITIVE, tuple(), True, "", "", True, False, True),
    )
    return {entry.primitive_id: entry for entry in entries}


def denial_result(reason: str = REASON_GENERIC_BYPASS_BLOCKED, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "mutated": False,
        "reason": reason or REASON_GENERIC_BYPASS_BLOCKED,
        "operation_id": None,
        "retryable": False,
        "details": dict(details or {}),
    }


def assert_authorized_mutation(
    context_payload: dict[str, Any] | None,
    *,
    expected_capability: str,
    expected_executor: str,
    expected_operation: str = "",
    expected_plan_fingerprint: str = "",
    expected_target_fingerprint: str = "",
    allowed_caller_types: tuple[str, ...] | None = None,
    runtime_mode: str = "production",
) -> tuple[bool, str, TrustedInvocationContext | None]:
    ok, reason, context = validate_trusted_invocation_context(
        context_payload,
        capability_id=expected_capability,
        executor_id=expected_executor,
        plan_fingerprint=expected_plan_fingerprint,
        operation_id=expected_operation,
        target_fingerprint=expected_target_fingerprint,
        runtime_mode=runtime_mode,
    )
    if not ok or context is None:
        return False, reason or REASON_GENERIC_BYPASS_BLOCKED, context
    allowed = allowed_caller_types or ("core", "executor", "skill_pack", "lifecycle_runner")
    if context.caller_type not in allowed:
        return False, "caller_type_not_allowed", context
    if context.single_use and context.consumed:
        return False, "trusted_context_consumed", context
    return True, REASON_ALLOWED, context


def deny_generic_http_mutation() -> dict[str, Any]:
    return denial_result("generic_http_mutation_denied", details={"primitive_class": MutationPrimitiveClass.DENIED_PRIMITIVE.value})


def deny_raw_secret_read() -> dict[str, Any]:
    return denial_result("raw_secret_read_denied", details={"primitive_class": MutationPrimitiveClass.DENIED_PRIMITIVE.value})


def deny_direct_domain_db_mutation() -> dict[str, Any]:
    return denial_result("direct_domain_db_mutation_denied", details={"primitive_class": MutationPrimitiveClass.DENIED_PRIMITIVE.value})


def deny_arbitrary_shell_mutation() -> dict[str, Any]:
    return denial_result("generic_shell_mutation_denied", details={"primitive_class": MutationPrimitiveClass.DENIED_PRIMITIVE.value})


__all__ = [
    "MutationPrimitiveClass",
    "MutationPrimitivePolicy",
    "assert_authorized_mutation",
    "denial_result",
    "deny_arbitrary_shell_mutation",
    "deny_direct_domain_db_mutation",
    "deny_generic_http_mutation",
    "deny_raw_secret_read",
    "primitive_policy_registry",
]
