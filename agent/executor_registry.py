from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Callable
import uuid

from .capability_policy import (
    POLICY_SCHEMA_VERSION,
    TrustedInvocationContext,
    authorize_capability,
    build_default_capability_registry,
    capability_for_action_type,
    stable_fingerprint,
    validate_trusted_invocation_context,
)
from .host_lifecycle import (
    HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION,
    HOST_LIFECYCLE_RUNNER_VERSION,
    attach_approved_hash,
)
from .mutation_plan import (
    MUTATION_PLAN_SCHEMA_VERSION,
    build_mutation_plan,
    validate_mutation_plan,
)
from .primary_uninstall_policy import (
    build_policy_context,
    consume_primary_uninstall_marker,
    validate_primary_uninstall_marker,
)


SUPPORT_BUNDLE_SCHEMA_VERSION = "support_bundle.v2"
BACKUP_SCHEMA_VERSION = "backup.v1"
BACKUP_MAX_TOTAL_BYTES = 2 * 1024 * 1024
BACKUP_MAX_FILE_BYTES = 256 * 1024
BACKUP_MAX_JOURNAL_ENTRIES = 8
EXECUTOR_JOURNAL_MAX_RECORD_BYTES = 64 * 1024
EXECUTOR_JOURNAL_MAX_STRING_BYTES = 1024
SUPPORT_BUNDLE_MAX_TOTAL_BYTES = 2 * 1024 * 1024
SUPPORT_BUNDLE_MAX_FILE_BYTES = 256 * 1024
CLEANUP_MAX_CANDIDATES = 50
CLEANUP_MAX_SCAN_ENTRIES = 10000
RESTORE_SNAPSHOT_SCHEMA_VERSION = "restore_snapshot.v1"
RESTORE_STAGE_SCHEMA_VERSION = "restore_stage.v1"
UPDATE_OPERATION_SCHEMA_VERSION = "update_operation.v1"
UPDATE_CHECKPOINT_SCHEMA_VERSION = "update_checkpoint.v1"
UNINSTALL_OPERATION_SCHEMA_VERSION = "uninstall_operation.v1"
UNINSTALL_RECEIPT_SCHEMA_VERSION = "uninstall_receipt.v1"
UNINSTALL_MODE_PRESERVE_DATA = "preserve_data"
RESTORE_ALLOWED_PREFERENCE_KEYS = {
    "system_resource_baseline_v1",
    "system_resource_baseline_context_v1",
}
RESTORE_V1_CAPABILITY = "restore_v1_allowlisted_preferences_only"

SECRET_KEY_HINTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "confirmation_token",
    "cookie",
    "password",
    "private_key",
    "secret",
    "server.secret_key",
    "sudo",
    "token",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trusted_context_failure(
    plan: dict[str, Any],
    action: dict[str, Any],
    *,
    capability_id: str,
    executor_id: str,
) -> dict[str, Any] | None:
    """Return a structured no-mutation failure when a migrated executor is called directly."""
    valid, reason, _context = validate_trusted_invocation_context(
        action.get("trusted_invocation_context") if isinstance(action.get("trusted_invocation_context"), dict) else None,
        capability_id=capability_id,
        executor_id=executor_id,
        plan_fingerprint=str(plan.get("plan_fingerprint") or ""),
    )
    if valid:
        return None
    return {
        "ok": False,
        "mutated": False,
        "executor_id": executor_id,
        "error_code": reason or "generic_bypass_blocked",
        "user_message": f"Capability policy blocked direct execution of {capability_id}. Use the Executor Registry Plan flow.",
        "rollback_available": False,
        "rollback_hint": "No rollback needed because nothing changed.",
        "details": {
            "capability_id": capability_id,
            "policy_schema_version": POLICY_SCHEMA_VERSION,
            "blocked_before_mutation": True,
        },
    }


def redact_executor_value(value: Any, *, key_hint: str = "") -> Any:
    normalized_key = str(key_hint or "").lower()
    if normalized_key in {"authorization_mode", "authorization_decision_id", "capability_id", "policy_schema_version", "plan_fingerprint", "target_fingerprint"}:
        return value
    if normalized_key in {"authorization", "authorization_decision"} and isinstance(value, dict) and "capability_id" in value and "reason_code" in value:
        return {str(key): redact_executor_value(item, key_hint=str(key)) for key, item in value.items()}
    if any(hint in normalized_key for hint in SECRET_KEY_HINTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            redacted[str(key)] = redact_executor_value(item, key_hint=str(key))
        return redacted
    if isinstance(value, list):
        return [redact_executor_value(item, key_hint=key_hint) for item in value[:100]]
    if isinstance(value, tuple):
        return [redact_executor_value(item, key_hint=key_hint) for item in value[:100]]
    if isinstance(value, str):
        lowered = value.lower()
        if any(
            hint in lowered
            for hint in (
                "bot token",
                "bearer ",
                "api_key=",
                "apikey=",
                "authorization:",
                "password=",
                "secret=",
                "token=",
                "x-api-key",
            )
        ):
            return "[REDACTED]"
        if len(value) > 512:
            return value[:512] + "...[truncated]"
    return value


def safe_path_label(path: Any) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    home = str(Path.home())
    if raw == home:
        return "~"
    if raw.startswith(home + "/"):
        raw = "~/" + raw[len(home) + 1 :]
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if len(parts) <= 4:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{'/'.join(parts[:2])}/.../{parts[-1]}#sha256:{digest}"


def support_bundle_redact(value: Any, *, key_hint: str = "") -> Any:
    redacted = redact_executor_value(value, key_hint=key_hint)
    normalized_key = str(key_hint or "").lower()
    if isinstance(redacted, dict):
        safe: dict[str, Any] = {}
        for key, item in redacted.items():
            key_text = str(key)
            safe[key_text] = support_bundle_redact(item, key_hint=key_text)
        return safe
    if isinstance(redacted, list):
        return [support_bundle_redact(item, key_hint=key_hint) for item in redacted[:80]]
    if isinstance(redacted, str):
        if any(token in normalized_key for token in ("path", "file", "dir", "root", "cwd")):
            return safe_path_label(redacted)
    return redacted


@dataclass(frozen=True)
class ExecutorResult:
    ok: bool
    mutated: bool
    executor_id: str
    plan_id: str
    action_type: str
    target: str
    started_at: str
    finished_at: str
    resources_touched: list[str] = field(default_factory=list)
    journal_id: str = ""
    rollback_available: bool = False
    rollback_hint: str = ""
    error_code: str | None = None
    user_message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    capability_id: str = ""
    policy_schema_version: int = POLICY_SCHEMA_VERSION
    authorization_mode: str = ""
    risk_level: str = ""
    plan_fingerprint: str = ""
    target_fingerprint: str = ""
    authorization_decision_id: str = ""
    confirmation_timestamp: str = ""
    mutation_plan_schema_version: int = MUTATION_PLAN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "ok": bool(self.ok),
            "mutated": bool(self.mutated),
            "executor_id": self.executor_id,
            "plan_id": self.plan_id,
            "action_type": self.action_type,
            "target": self.target,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "resources_touched": list(self.resources_touched),
            "journal_id": self.journal_id,
            "rollback_available": bool(self.rollback_available),
            "rollback_hint": self.rollback_hint,
            "error_code": self.error_code,
            "user_message": self.user_message,
            "details": dict(self.details),
            "capability_id": self.capability_id,
            "policy_schema_version": int(self.policy_schema_version),
            "authorization_mode": self.authorization_mode,
            "risk_level": self.risk_level,
            "plan_fingerprint": self.plan_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "authorization_decision_id": self.authorization_decision_id,
            "confirmation_timestamp": self.confirmation_timestamp,
            "mutation_plan_schema_version": int(self.mutation_plan_schema_version),
        }
        return redact_executor_value(payload)


ExecutorFn = Callable[[dict[str, Any], dict[str, Any]], ExecutorResult | dict[str, Any]]


class ExecutorPartialFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        resources_touched: list[str] | None = None,
        rollback_hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.resources_touched = list(resources_touched or [])
        self.rollback_hint = rollback_hint
        self.details = dict(details or {})


@dataclass(frozen=True)
class ExecutorSpec:
    executor_id: str
    action_type: str
    status: str
    run: ExecutorFn | None = None
    rollback_available: bool = False
    rollback_hint: str = ""
    capability_id: str | None = None


class MutationJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> str:
        journal_id = str(record.get("journal_id") or f"executor-{uuid.uuid4().hex[:12]}")
        payload = _bounded_journal_record(redact_executor_value({**record, "journal_id": journal_id}))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
        return journal_id

    def recent(self, limit: int = 20, *, max_tail_bytes: int = 512 * 1024, max_line_bytes: int = 64 * 1024) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - max_tail_bytes))
                data = handle.read(max_tail_bytes)
        except OSError:
            return []
        if size > max_tail_bytes:
            _, _, data = data.partition(b"\n")
        lines = data.decode("utf-8", errors="replace").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-max(0, int(limit)) :]:
            if len(line.encode("utf-8", errors="replace")) > max_line_bytes:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
        return out


class ExecutorRegistry:
    def __init__(self, journal_path: str | Path) -> None:
        self.journal = MutationJournal(journal_path)
        self._executors: dict[str, ExecutorSpec] = {}

    def register(self, spec: ExecutorSpec) -> None:
        action_type = str(spec.action_type or "").strip().lower()
        if not action_type:
            raise ValueError("executor action_type is required")
        self._executors[action_type] = spec

    def lookup(self, action_type: str) -> ExecutorSpec | None:
        return self._executors.get(str(action_type or "").strip().lower())

    def execute_confirmed_plan(
        self,
        *,
        plan: dict[str, Any],
        action: dict[str, Any],
        high_risk_confirmed: bool = True,
    ) -> ExecutorResult:
        started_at = utc_now_iso()
        plan_id = str(plan.get("plan_id") or action.get("pending_id") or "").strip()
        action_type = str(plan.get("action_type") or "").strip().lower()
        target = str(plan.get("target") or "unspecified").strip() or "unspecified"
        executor_status = str(plan.get("executor_status") or "unavailable").strip().lower() or "unavailable"
        risk_level = str(plan.get("risk_level") or "").strip().lower()
        journal_id = f"executor-{uuid.uuid4().hex[:12]}"
        capability_id = str(plan.get("capability_id") or capability_for_action_type(action_type) or "").strip()
        plan_fingerprint = str(plan.get("plan_fingerprint") or "").strip()
        target_fingerprint = str(plan.get("target_fingerprint") or "").strip()
        authorization_metadata: dict[str, Any] = {}

        def _result(
            *,
            ok: bool,
            mutated: bool,
            executor_id: str,
            error_code: str | None,
            user_message: str,
            resources_touched: list[str] | None = None,
            rollback_available: bool = False,
            rollback_hint: str = "",
            details: dict[str, Any] | None = None,
        ) -> ExecutorResult:
            finished_at = utc_now_iso()
            result = ExecutorResult(
                ok=ok,
                mutated=mutated,
                executor_id=executor_id,
                plan_id=plan_id,
                action_type=action_type,
                target=target,
                started_at=started_at,
                finished_at=finished_at,
                resources_touched=list(resources_touched or []),
                journal_id=journal_id,
                rollback_available=rollback_available,
                rollback_hint=rollback_hint,
                error_code=error_code,
                user_message=user_message,
                details=dict(details or {}),
                capability_id=str(authorization_metadata.get("capability_id") or capability_id),
                policy_schema_version=int(authorization_metadata.get("policy_schema_version") or POLICY_SCHEMA_VERSION),
                authorization_mode=str(authorization_metadata.get("authorization_mode") or ""),
                risk_level=str(authorization_metadata.get("risk_level") or risk_level),
                plan_fingerprint=str(authorization_metadata.get("plan_fingerprint") or plan_fingerprint),
                target_fingerprint=str(authorization_metadata.get("target_fingerprint") or target_fingerprint),
                authorization_decision_id=str(authorization_metadata.get("authorization_decision_id") or ""),
                confirmation_timestamp=str(authorization_metadata.get("confirmation_timestamp") or ""),
                mutation_plan_schema_version=int(authorization_metadata.get("mutation_plan_schema_version") or MUTATION_PLAN_SCHEMA_VERSION),
            )
            self.journal.append(
                {
                    "journal_id": journal_id,
                    "event": "executor_result",
                    "plan": plan,
                    "action": action,
                    "result": result.to_dict(),
                }
            )
            return result

        if not plan_id:
            return _result(
                ok=False,
                mutated=False,
                executor_id="executor_registry",
                error_code="plan_id_missing",
                user_message="I blocked execution because the confirmed plan had no plan_id.",
            )
        if str(action.get("pending_id") or "").strip() and plan_id != str(action.get("pending_id") or "").strip():
            return _result(
                ok=False,
                mutated=False,
                executor_id="executor_registry",
                error_code="plan_id_mismatch",
                user_message="I blocked execution because the confirmed plan_id did not match the pending action.",
            )
        if risk_level == "high" and not high_risk_confirmed:
            return _result(
                ok=False,
                mutated=False,
                executor_id="executor_registry",
                error_code="high_risk_confirmation_required",
                user_message="I blocked this high-risk action because it did not have explicit confirmation.",
            )
        if executor_status == "preview_only":
            return _result(
                ok=False,
                mutated=False,
                executor_id="preview_only",
                error_code="executor_not_enabled",
                user_message="This plan was confirmed, but its executor is preview-only. I did not mutate state.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing was changed.",
            )
        if executor_status == "unavailable":
            return _result(
                ok=False,
                mutated=False,
                executor_id="unavailable",
                error_code="executor_unavailable",
                user_message="This plan was confirmed, but no executor is available. I did not mutate state.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing was changed.",
            )
        if executor_status != "enabled":
            return _result(
                ok=False,
                mutated=False,
                executor_id="executor_registry",
                error_code="executor_status_unknown",
                user_message="This plan used an unknown executor status, so I blocked it.",
            )
        spec = self.lookup(action_type)
        if spec is None or str(spec.status or "").strip().lower() != "enabled" or not callable(spec.run):
            return _result(
                ok=False,
                mutated=False,
                executor_id=(spec.executor_id if spec else "unregistered"),
                error_code="executor_unavailable",
                user_message="This action has no enabled executor in the registry. I did not mutate state.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing was changed.",
            )
        trusted_capability_id = str(spec.capability_id or capability_id or "").strip()
        if trusted_capability_id:
            capability_id = trusted_capability_id
            try:
                mutation_plan = _ensure_universal_mutation_plan(
                    plan,
                    action=action,
                    capability_id=capability_id,
                    executor_id=spec.executor_id,
                    action_type=action_type,
                    target=target,
                )
            except Exception as exc:  # noqa: BLE001 - malformed plans must fail closed.
                return _result(
                    ok=False,
                    mutated=False,
                    executor_id=spec.executor_id,
                    error_code="mutation_plan_invalid",
                    user_message="I blocked execution because the confirmed mutation Plan was invalid.",
                    rollback_available=False,
                    rollback_hint="No rollback needed because nothing changed.",
                    details={"exception": exc.__class__.__name__},
                )
            if not plan_fingerprint:
                plan_fingerprint = str(mutation_plan.get("plan_fingerprint") or "")
                plan["plan_fingerprint"] = plan_fingerprint
            if not target_fingerprint:
                target_fingerprint = str(mutation_plan.get("target_fingerprint") or "")
                plan["target_fingerprint"] = target_fingerprint
            activation_context = _activation_context_for_capability(capability_id, plan=plan, action=action)
            decision = authorize_capability(
                capability_id,
                request_context={
                    "action_type": action_type,
                    "executor_id": spec.executor_id,
                    "origin": str(action.get("origin") or "executor_registry"),
                },
                target_snapshot={"target_fingerprint": target_fingerprint},
                plan_context={
                    "plan_id": plan_id,
                    "plan_fingerprint": plan_fingerprint,
                    "target_fingerprint": target_fingerprint,
                    "policy_version": int(plan.get("policy_schema_version") or POLICY_SCHEMA_VERSION),
                    "stale": bool(plan.get("stale")),
                },
                confirmation_context={
                    "confirmed": True,
                    "pending_id": str(action.get("pending_id") or ""),
                },
                activation_context=activation_context,
                registry=build_default_capability_registry(),
            )
            authorization_metadata = {
                "capability_id": capability_id,
                "policy_schema_version": POLICY_SCHEMA_VERSION,
                "authorization_mode": decision.authorization_mode,
                "risk_level": decision.risk_level,
                "plan_fingerprint": plan_fingerprint,
                "target_fingerprint": target_fingerprint,
                "authorization_decision_id": decision.decision_id,
                "confirmation_timestamp": utc_now_iso(),
                "mutation_plan_schema_version": MUTATION_PLAN_SCHEMA_VERSION,
            }
            action["authorization_decision"] = decision.to_dict()
            action["trusted_invocation_context"] = TrustedInvocationContext(
                capability_id=capability_id,
                executor_id=spec.executor_id,
                authorization_decision_id=decision.decision_id,
                plan_fingerprint=plan_fingerprint,
                operation_id=plan_id,
            ).to_dict()
            if not decision.allowed:
                return _result(
                    ok=False,
                    mutated=False,
                    executor_id=spec.executor_id,
                    error_code=decision.reason_code,
                    user_message=f"Capability policy blocked {capability_id}: {decision.reason_code}.",
                    rollback_available=False,
                    rollback_hint="No rollback needed because nothing was changed.",
                    details={"authorization": decision.to_dict()},
                )
        try:
            raw = spec.run(plan, action)
            if isinstance(raw, ExecutorResult):
                result = raw
            else:
                finished_at = utc_now_iso()
                result = ExecutorResult(
                    ok=bool(raw.get("ok")),
                    mutated=bool(raw.get("mutated")),
                    executor_id=str(raw.get("executor_id") or spec.executor_id),
                    plan_id=plan_id,
                    action_type=action_type,
                    target=target,
                    started_at=started_at,
                    finished_at=finished_at,
                    resources_touched=[str(item) for item in raw.get("resources_touched", []) if str(item).strip()]
                    if isinstance(raw.get("resources_touched"), list)
                    else [],
                    journal_id=journal_id,
                    rollback_available=bool(raw.get("rollback_available", spec.rollback_available)),
                    rollback_hint=str(raw.get("rollback_hint") or spec.rollback_hint),
                    error_code=str(raw.get("error_code") or "") or None,
                    user_message=str(raw.get("user_message") or ""),
                    details={
                        **dict(raw.get("details") if isinstance(raw.get("details"), dict) else {}),
                        **({"authorization": action.get("authorization_decision")} if isinstance(action.get("authorization_decision"), dict) else {}),
                    },
                    capability_id=str(authorization_metadata.get("capability_id") or capability_id),
                    policy_schema_version=int(authorization_metadata.get("policy_schema_version") or POLICY_SCHEMA_VERSION),
                    authorization_mode=str(authorization_metadata.get("authorization_mode") or ""),
                    risk_level=str(authorization_metadata.get("risk_level") or risk_level),
                    plan_fingerprint=str(authorization_metadata.get("plan_fingerprint") or plan_fingerprint),
                    target_fingerprint=str(authorization_metadata.get("target_fingerprint") or target_fingerprint),
                    authorization_decision_id=str(authorization_metadata.get("authorization_decision_id") or ""),
                    confirmation_timestamp=str(authorization_metadata.get("confirmation_timestamp") or ""),
                    mutation_plan_schema_version=int(authorization_metadata.get("mutation_plan_schema_version") or MUTATION_PLAN_SCHEMA_VERSION),
                )
        except ExecutorPartialFailure as exc:
            return _result(
                ok=False,
                mutated=False,
                executor_id=spec.executor_id,
                error_code="executor_partial_failure",
                user_message=f"{target} did not finish. I recorded partial artifacts and did not verify a completed mutation.",
                resources_touched=exc.resources_touched,
                rollback_available=True,
                rollback_hint=exc.rollback_hint or spec.rollback_hint,
                details={"exception": exc.__class__.__name__, **exc.details},
            )
        except Exception as exc:  # noqa: BLE001 - executor boundary must return safe failure.
            return _result(
                ok=False,
                mutated=False,
                executor_id=spec.executor_id,
                error_code="executor_exception_before_verified_mutation",
                user_message=f"{target} did not finish. I did not verify any mutation.",
                rollback_available=spec.rollback_available,
                rollback_hint=spec.rollback_hint,
                details={"exception": exc.__class__.__name__},
            )
        result = ExecutorResult(
            **{
                **result.to_dict(),
                "journal_id": journal_id,
                "started_at": result.started_at or started_at,
                "finished_at": result.finished_at or utc_now_iso(),
                "capability_id": result.capability_id or str(authorization_metadata.get("capability_id") or capability_id),
                "policy_schema_version": int(authorization_metadata.get("policy_schema_version") or result.policy_schema_version),
                "authorization_mode": result.authorization_mode or str(authorization_metadata.get("authorization_mode") or ""),
                "risk_level": result.risk_level or str(authorization_metadata.get("risk_level") or risk_level),
                "plan_fingerprint": result.plan_fingerprint or str(authorization_metadata.get("plan_fingerprint") or plan_fingerprint),
                "target_fingerprint": result.target_fingerprint or str(authorization_metadata.get("target_fingerprint") or target_fingerprint),
                "authorization_decision_id": result.authorization_decision_id or str(authorization_metadata.get("authorization_decision_id") or ""),
                "confirmation_timestamp": result.confirmation_timestamp or str(authorization_metadata.get("confirmation_timestamp") or ""),
                "mutation_plan_schema_version": int(authorization_metadata.get("mutation_plan_schema_version") or result.mutation_plan_schema_version),
            }
        )
        self.journal.append(
            {
                "journal_id": journal_id,
                "event": "executor_result",
                "plan": plan,
                "action": action,
                "result": result.to_dict(),
            }
        )
        return result


def _activation_context_for_capability(capability_id: str, *, plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    if capability_id != "system.uninstall":
        return {"valid": True, "reason_code": "allowed"}
    snapshot = action.get("target_snapshot") if isinstance(action.get("target_snapshot"), dict) else {}
    execution_mode = str(action.get("uninstall_execution_mode") or "").strip().lower()
    target = str(plan.get("target") or "").strip().lower()
    if bool(snapshot.get("fixture_marker")) or execution_mode in {"fixture_preserve_data", "production_shaped_preserve_data"} or "fixture" in target or "proof" in target:
        return {"valid": True, "reason_code": "allowed", "activation_source": "isolated_fixture"}
    try:
        status = validate_primary_uninstall_marker()
    except Exception as exc:  # noqa: BLE001 - activation validator failures must fail closed.
        return {"valid": False, "reason_code": "activation_invalid", "error": exc.__class__.__name__}
    if bool(getattr(status, "valid", False)):
        return {"valid": True, "reason_code": "allowed", "activation_source": "primary_marker"}
    reason = str(getattr(status, "reason_code", "") or "local_activation_required")
    return {"valid": False, "reason_code": reason}


def _ensure_universal_mutation_plan(
    plan: dict[str, Any],
    *,
    action: dict[str, Any],
    capability_id: str,
    executor_id: str,
    action_type: str,
    target: str,
) -> dict[str, Any]:
    existing = plan.get("mutation_plan") if isinstance(plan.get("mutation_plan"), dict) else None
    if existing is not None:
        validate_mutation_plan(existing)
        if existing.get("capability_id") != capability_id or existing.get("executor_id") != executor_id:
            raise ValueError("mutation_plan_executor_or_capability_mismatch")
        return existing
    expires_at = int(plan.get("expires_at") or (int(datetime.now(timezone.utc).timestamp()) + 600))
    resources = plan.get("resources_affected") if isinstance(plan.get("resources_affected"), list) else []
    target_snapshot = {
        "action_type": action_type,
        "target": target,
        "params": action.get("params") if isinstance(action.get("params"), dict) else {},
        "resources_affected": resources,
    }
    mutation_inventory = [{"resource": str(item), "effect": "may_change"} for item in resources]
    if not mutation_inventory:
        mutation_inventory = [{"target": target, "effect": "capability_specific"}]
    universal = build_mutation_plan(
        plan_id=str(plan.get("plan_id") or action.get("pending_id") or ""),
        capability_id=capability_id,
        executor_id=executor_id,
        expires_at_epoch=expires_at,
        thread_id=str(action.get("thread_id") or action.get("session_thread_id") or ""),
        session_id=str(action.get("session_id") or ""),
        actor_id=str(action.get("user_id") or action.get("actor_id") or ""),
        target_snapshot=target_snapshot,
        mutation_inventory=mutation_inventory,
        preserved_resources=plan.get("preserved_resources") if isinstance(plan.get("preserved_resources"), list) else [],
        expected_side_effects=plan.get("expected_side_effects") if isinstance(plan.get("expected_side_effects"), list) else [],
        recovery={
            "rollback_supported": bool(plan.get("rollback_supported")),
            "rollback_scope": str(plan.get("rollback_scope") or ""),
        },
        activation_fingerprint=str(plan.get("activation_fingerprint") or "") or None,
    )
    plan["mutation_plan"] = universal
    plan["target_fingerprint"] = str(universal.get("target_fingerprint") or "")
    plan["plan_fingerprint"] = str(universal.get("plan_fingerprint") or "")
    plan["mutation_plan_schema_version"] = MUTATION_PLAN_SCHEMA_VERSION
    return universal


def _capability_policy_record(action: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any] | None:
    decision = action.get("authorization_decision") if isinstance(action.get("authorization_decision"), dict) else {}
    capability_id = str(decision.get("capability_id") or plan.get("capability_id") or "").strip()
    if not capability_id:
        return None
    mutation_plan = plan.get("mutation_plan") if isinstance(plan.get("mutation_plan"), dict) else {}
    return {
        "capability_id": capability_id,
        "policy_schema_version": int(decision.get("policy_version") or plan.get("policy_schema_version") or POLICY_SCHEMA_VERSION),
        "mutation_plan_schema_version": int(mutation_plan.get("schema_version") or plan.get("mutation_plan_schema_version") or MUTATION_PLAN_SCHEMA_VERSION),
        "authorization_mode": str(decision.get("authorization_mode") or ""),
        "risk_level": str(decision.get("risk_level") or plan.get("risk_level") or ""),
        "plan_fingerprint": str(decision.get("plan_fingerprint") or plan.get("plan_fingerprint") or ""),
        "target_fingerprint": str(decision.get("target_fingerprint") or plan.get("target_fingerprint") or ""),
        "authorization_decision_id": str(decision.get("decision_id") or ""),
        "receipt_required": bool(decision.get("receipt_required", True)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(support_bundle_redact(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_json_unredacted(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def _bounded_journal_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated:max_depth]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 40:
                out["truncated_keys"] = True
                break
            out[str(key)] = _bounded_journal_value(item, depth=depth + 1)
        return out
    if isinstance(value, list):
        out = [_bounded_journal_value(item, depth=depth + 1) for item in value[:40]]
        if len(value) > 40:
            out.append("[truncated:list]")
        return out
    if isinstance(value, tuple):
        return _bounded_journal_value(list(value), depth=depth)
    if isinstance(value, str):
        encoded = value.encode("utf-8", errors="replace")
        if len(encoded) > EXECUTOR_JOURNAL_MAX_STRING_BYTES:
            return encoded[:EXECUTOR_JOURNAL_MAX_STRING_BYTES].decode("utf-8", errors="replace") + "...[truncated]"
    return value


def _compact_journal_record(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    action = record.get("action") if isinstance(record.get("action"), dict) else {}
    resources = result.get("resources_touched") if isinstance(result.get("resources_touched"), list) else []
    return redact_executor_value(
        {
            "journal_id": record.get("journal_id") or result.get("journal_id"),
            "event": record.get("event"),
            "compacted": True,
            "compaction_reason": "journal_record_size_cap",
            "plan_id": result.get("plan_id") or plan.get("plan_id") or action.get("pending_id"),
            "action_type": result.get("action_type") or plan.get("action_type"),
            "target": result.get("target") or plan.get("target"),
            "executor_id": result.get("executor_id"),
            "capability_id": result.get("capability_id") or plan.get("capability_id"),
            "policy_schema_version": result.get("policy_schema_version") or plan.get("policy_schema_version"),
            "authorization_decision_id": result.get("authorization_decision_id"),
            "ok": result.get("ok"),
            "mutated": result.get("mutated"),
            "error_code": result.get("error_code"),
            "resources_touched_count": len(resources),
            "rollback_available": result.get("rollback_available"),
        }
    )


def _bounded_journal_record(record: dict[str, Any]) -> dict[str, Any]:
    bounded = _bounded_journal_value(record)
    encoded = json.dumps(bounded, sort_keys=True, ensure_ascii=True).encode("utf-8")
    if len(encoded) <= EXECUTOR_JOURNAL_MAX_RECORD_BYTES:
        return bounded if isinstance(bounded, dict) else {"record": bounded}
    return _compact_journal_record(record)


def _status_summary(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


def _bounded_backup_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "[truncated:max_depth]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 40:
                out["truncated_keys"] = True
                break
            out[str(key)] = _bounded_backup_value(item, depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_bounded_backup_value(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, tuple):
        return [_bounded_backup_value(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, str):
        if len(value) > 512:
            return value[:512] + "...[truncated]"
    return value


def _write_backup_json(path: Path, payload: dict[str, Any]) -> int:
    text = json.dumps(support_bundle_redact(_bounded_backup_value(payload)), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    encoded = text.encode("utf-8")
    if len(encoded) > BACKUP_MAX_FILE_BYTES:
        raise ValueError(f"backup_file_size_cap_exceeded:{path.name}:{len(encoded)}")
    path.write_bytes(encoded)
    return len(encoded)


def _write_support_json(path: Path, payload: dict[str, Any]) -> int:
    text = json.dumps(support_bundle_redact(_bounded_backup_value(payload)), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    encoded = text.encode("utf-8")
    if len(encoded) > SUPPORT_BUNDLE_MAX_FILE_BYTES:
        raise ValueError(f"support_bundle_file_size_cap_exceeded:{path.name}:{len(encoded)}")
    path.write_bytes(encoded)
    return len(encoded)


def _summarize_executor_journal_entries(entries: list[Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for entry in entries[-BACKUP_MAX_JOURNAL_ENTRIES:]:
        if not isinstance(entry, dict):
            continue
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        plan = entry.get("plan") if isinstance(entry.get("plan"), dict) else {}
        resources = result.get("resources_touched") if isinstance(result.get("resources_touched"), list) else []
        summaries.append(
            support_bundle_redact(
                {
                    "journal_id": entry.get("journal_id") or result.get("journal_id"),
                    "event": entry.get("event"),
                    "plan_id": result.get("plan_id") or plan.get("plan_id"),
                    "action_type": result.get("action_type") or plan.get("action_type"),
                    "target": result.get("target") or plan.get("target"),
                    "executor_id": result.get("executor_id"),
                    "ok": result.get("ok"),
                    "mutated": result.get("mutated"),
                    "error_code": result.get("error_code"),
                    "resources_touched_count": len(resources),
                    "rollback_available": result.get("rollback_available"),
                    "started_at": result.get("started_at"),
                    "finished_at": result.get("finished_at"),
                }
            )
        )
    return summaries


def build_support_bundle_manifest(
    *,
    root: Path,
    diagnostics: dict[str, Any],
    included_files: list[str],
) -> dict[str, Any]:
    version = diagnostics.get("version") if isinstance(diagnostics.get("version"), dict) else {}
    return support_bundle_redact(
        {
            "bundle_schema_version": SUPPORT_BUNDLE_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "runtime_commit": version.get("git_commit"),
            "checkout_commit": diagnostics.get("checkout_commit"),
            "runtime_instance": version.get("runtime_instance"),
            "included_files": included_files,
            "bundle_path": str(root),
            "redaction_policy": (
                "Token, API key, password, bearer, secret, confirmation-token, raw secret-file, "
                "raw log, and broad private-path values are redacted or summarized."
            ),
        }
    )


def _approved_backup_root(action: dict[str, Any]) -> Path:
    raw = str(action.get("backup_root") or "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    else:
        root = (Path.home() / ".local/share/personal-agent/backups").resolve()
    home = Path.home().resolve()
    state_root = (home / ".local/share/personal-agent").resolve()
    tmp_root = Path(tempfile.gettempdir()).resolve()
    if root != state_root and state_root not in root.parents and root != tmp_root and tmp_root not in root.parents:
        raise ValueError("backup_root_not_approved")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _artifact_dir(root: Path, *, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def _tree_size_and_count(path: Path, *, max_entries: int = CLEANUP_MAX_SCAN_ENTRIES) -> tuple[int, int, bool]:
    total = 0
    count = 0
    truncated = False
    try:
        if path.is_file() or path.is_symlink():
            return int(path.lstat().st_size), 1, False
        for child in path.rglob("*"):
            count += 1
            if count > max_entries:
                truncated = True
                break
            try:
                stat = child.lstat()
            except OSError:
                continue
            total += int(stat.st_size)
    except OSError:
        return 0, count, True
    return total, count, truncated


def _cleanup_allowed_roots() -> dict[str, Path]:
    state_root = (Path.home() / ".local/share/personal-agent").resolve()
    return {
        "backup": (state_root / "backups").resolve(),
        "runtime_release": (state_root / "runtime/releases").resolve(),
        "support_tmp": Path(tempfile.gettempdir()).resolve(),
    }


def _is_contained(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve(strict=False)
        root_resolved = root.resolve(strict=False)
    except OSError:
        return False
    return resolved == root_resolved or root_resolved in resolved.parents


def _cleanup_candidate_kind(classification: str) -> str | None:
    normalized = str(classification or "").strip().lower()
    if normalized in {"oversized backup artifact", "old backup artifact"}:
        return "backup"
    if normalized == "old support bundle artifact":
        return "support_tmp"
    if normalized == "old runtime release":
        return "runtime_release"
    return None


def _cleanup_path_allowed(path: Path, *, kind: str) -> tuple[bool, str]:
    roots = _cleanup_allowed_roots()
    root = roots.get(kind)
    if root is None:
        return False, "cleanup_candidate_kind_not_allowed"
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return False, "cleanup_path_unresolvable"
    if not _is_contained(resolved, root):
        return False, "cleanup_path_outside_owned_root"
    if kind == "backup" and not resolved.name.startswith("personal-agent-backup-"):
        return False, "cleanup_backup_name_not_owned"
    if kind == "support_tmp" and not (
        resolved.name.startswith("personal-agent-support-") or resolved.name.startswith("agent-support-")
    ):
        return False, "cleanup_support_name_not_owned"
    if kind == "runtime_release" and resolved == root:
        return False, "cleanup_runtime_release_root_protected"
    return True, ""


def _cleanup_has_symlink(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        if path.is_dir():
            for index, child in enumerate(path.rglob("*")):
                if index > CLEANUP_MAX_SCAN_ENTRIES:
                    return True
                if child.is_symlink():
                    return True
    except OSError:
        return True
    return False


def execute_cleanup(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    preview = action.get("cleanup_preview") if isinstance(action.get("cleanup_preview"), dict) else {}
    candidates = preview.get("candidates") if isinstance(preview.get("candidates"), list) else []
    protected = preview.get("protected") if isinstance(preview.get("protected"), list) else []
    protected_paths = {
        str(item.get("canonical_path") or "").strip()
        for item in protected
        if isinstance(item, dict) and str(item.get("canonical_path") or "").strip()
    }
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    bytes_recovered = 0

    for raw in candidates[:CLEANUP_MAX_CANDIDATES]:
        if not isinstance(raw, dict):
            skipped.append({"reason": "cleanup_candidate_malformed"})
            continue
        label = str(raw.get("path") or raw.get("canonical_path") or "").strip()
        canonical = str(raw.get("canonical_path") or "").strip()
        classification = str(raw.get("classification") or "").strip()
        if not canonical:
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_canonical_path_missing"})
            continue
        if not bool(raw.get("safe_to_delete_later")):
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_candidate_not_marked_safe"})
            continue
        if canonical in protected_paths:
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_candidate_became_protected"})
            continue
        kind = _cleanup_candidate_kind(classification)
        if kind is None:
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_classification_not_deletable"})
            continue
        path = Path(canonical).expanduser()
        allowed, reason = _cleanup_path_allowed(path, kind=kind)
        if not allowed:
            skipped.append({"path": label, "classification": classification, "reason": reason})
            continue
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_candidate_missing"})
            continue
        if os.path.ismount(str(resolved)):
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_mount_point_protected"})
            continue
        if _cleanup_has_symlink(resolved):
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_symlink_protected"})
            continue
        current_size, current_count, truncated = _tree_size_and_count(resolved)
        preview_size = raw.get("size_bytes")
        preview_count = raw.get("file_count")
        if isinstance(preview_size, int) and current_size != preview_size:
            skipped.append(
                {
                    "path": label,
                    "classification": classification,
                    "reason": "cleanup_candidate_changed_after_preview",
                    "preview_size_bytes": preview_size,
                    "current_size_bytes": current_size,
                }
            )
            continue
        if isinstance(preview_count, int) and current_count != preview_count:
            skipped.append(
                {
                    "path": label,
                    "classification": classification,
                    "reason": "cleanup_candidate_changed_after_preview",
                    "preview_file_count": preview_count,
                    "current_file_count": current_count,
                }
            )
            continue
        if truncated:
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_scan_truncated"})
            continue
        try:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
        except Exception as exc:  # noqa: BLE001 - independent cleanup failures are reported, not thrown.
            failures.append({"path": label, "classification": classification, "reason": exc.__class__.__name__})
            continue
        bytes_recovered += current_size
        deleted.append({"path": label, "classification": classification, "size_bytes": current_size})

    ok = not failures
    mutated = bool(deleted)
    if deleted and failures:
        status = "partial_failure"
        message = (
            f"Cleanup partially finished. Removed {len(deleted)} Personal Agent artifact(s) and recovered "
            f"{bytes_recovered} bytes, but {len(failures)} candidate(s) failed."
        )
        error_code = "cleanup_partial_failure"
    elif deleted:
        status = "completed"
        message = f"Cleanup finished. Removed {len(deleted)} old Personal Agent artifact(s) and recovered {bytes_recovered} bytes."
        error_code = None
    elif failures:
        status = "failed"
        message = "Cleanup did not remove anything because every attempted candidate failed."
        error_code = "cleanup_failed"
    else:
        status = "no_op"
        message = "Cleanup found no eligible candidates to delete after revalidation. I did not remove anything."
        error_code = None
    return {
        "ok": ok,
        "mutated": mutated,
        "executor_id": "operator.cleanup.v1",
        "resources_touched": [str(item.get("path") or "") for item in deleted if str(item.get("path") or "").strip()],
        "rollback_available": False,
        "rollback_hint": (
            "Cleanup deletion is not automatically reversible. The latest valid backup, current runtime, secret store, "
            "and active service files were protected by policy."
        ),
        "error_code": error_code,
        "user_message": message,
        "details": {
            "status": status,
            "deleted": deleted,
            "skipped": skipped[:50],
            "protected_count": len(protected),
            "failures": failures[:50],
            "bytes_recovered": bytes_recovered,
        },
    }


def _approved_restore_root(action: dict[str, Any]) -> Path:
    raw = str(action.get("state_root") or "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    else:
        root = (Path.home() / ".local/share/personal-agent").resolve()
    home_state = (Path.home() / ".local/share/personal-agent").resolve()
    tmp_root = Path(tempfile.gettempdir()).resolve()
    if root != home_state and home_state not in root.parents and root != tmp_root and tmp_root not in root.parents:
        raise ValueError("restore_state_root_not_approved")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _restore_backup_path(action: dict[str, Any]) -> Path:
    raw = str(action.get("restore_backup_path") or "").strip()
    if not raw:
        raise ValueError("restore_backup_path_missing")
    path = Path(raw).expanduser().resolve()
    backup_root = str(action.get("backup_root") or "").strip()
    approved_roots = [(Path.home() / ".local/share/personal-agent/backups").resolve(), Path(tempfile.gettempdir()).resolve()]
    if backup_root:
        approved_roots.append(Path(backup_root).expanduser().resolve())
    if not any(path == root or root in path.parents for root in approved_roots):
        raise ValueError("restore_backup_path_outside_approved_locations")
    return path


def _read_json_file(path: Path, *, max_bytes: int = BACKUP_MAX_FILE_BYTES) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"restore_symlink_rejected:{path.name}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"restore_file_size_cap_exceeded:{path.name}")
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise ValueError(f"restore_json_not_object:{path.name}")
    return payload


def _backup_fingerprint(path: Path, included_files: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(included_files):
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"restore_unsafe_included_file:{name}")
        file_path = path / name
        if file_path.is_symlink():
            raise ValueError(f"restore_symlink_rejected:{name}")
        stat = file_path.stat()
        digest.update(name.encode("utf-8"))
        digest.update(str(int(stat.st_size)).encode("ascii"))
        digest.update(hashlib.sha256(file_path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def _validate_restore_backup(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"valid": False, "error_code": "backup_path_not_directory"}
    if path.is_symlink():
        return {"valid": False, "error_code": "backup_path_symlink_rejected"}
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return {"valid": False, "error_code": "manifest_missing"}
    try:
        manifest = _read_json_file(manifest_path)
    except Exception as exc:  # noqa: BLE001 - restore reports validation failure.
        return {"valid": False, "error_code": "manifest_unreadable", "exception": exc.__class__.__name__}
    if manifest.get("backup_schema_version") != BACKUP_SCHEMA_VERSION:
        return {"valid": False, "error_code": "unsupported_backup_schema", "schema_version": manifest.get("backup_schema_version")}
    included = [str(item) for item in manifest.get("included_files", []) if str(item).strip()] if isinstance(manifest.get("included_files"), list) else []
    required = {
        "backup_summary.json",
        "diagnostics_summary.json",
        "executor_registry_journal_summary.json",
        "manifest.json",
        "memory_anchors_summary.json",
        "pack_metadata_summary.json",
        "preferences_summary.json",
        "runtime_config_summary.json",
        "state_database_summary.json",
        "support_bundle_style_summary.json",
    }
    missing = sorted(name for name in required if name not in set(included) or not (path / name).is_file())
    if missing:
        return {"valid": False, "error_code": "required_files_missing", "missing_files": missing}
    total_size = 0
    parsed_files: dict[str, dict[str, Any]] = {}
    try:
        for name in sorted(set(included) | required):
            if "/" in name or "\\" in name or name.startswith("."):
                return {"valid": False, "error_code": "unsafe_included_file_name", "file": name}
            file_path = path / name
            if not file_path.is_file():
                continue
            if file_path.is_symlink():
                return {"valid": False, "error_code": "backup_contains_symlink", "file": name}
            size = file_path.stat().st_size
            total_size += size
            if size > BACKUP_MAX_FILE_BYTES:
                return {"valid": False, "error_code": "backup_file_size_cap_exceeded", "file": name}
            parsed_files[name] = _read_json_file(file_path)
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error_code": "backup_file_validation_failed", "exception": exc.__class__.__name__}
    if total_size > BACKUP_MAX_TOTAL_BYTES:
        return {"valid": False, "error_code": "backup_total_size_cap_exceeded"}
    preferences = parsed_files.get("preferences_summary.json", {})
    preference_items = preferences.get("preferences") if isinstance(preferences.get("preferences"), list) else []
    supported_preferences: list[dict[str, str]] = []
    skipped_preferences: list[str] = []
    for item in preference_items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        value = item.get("value")
        if key not in RESTORE_ALLOWED_PREFERENCE_KEYS:
            skipped_preferences.append(key or "unknown")
            continue
        if not isinstance(value, str) or len(value.encode("utf-8", errors="replace")) > 64 * 1024:
            return {"valid": False, "error_code": "preference_value_invalid", "preference_key": key}
        supported_preferences.append({"key": key, "value": value})
    try:
        fingerprint = _backup_fingerprint(path, included)
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error_code": "backup_fingerprint_failed", "exception": exc.__class__.__name__}
    return {
        "valid": True,
        "manifest": manifest,
        "included_files": sorted(set(included)),
        "fingerprint": fingerprint,
        "supported_preferences": supported_preferences,
        "skipped_preferences": sorted(set(skipped_preferences)),
        "created_at": manifest.get("created_at"),
        "runtime_commit": manifest.get("runtime_commit"),
        "total_size_bytes": total_size,
    }


def _restore_lock_path(state_root: Path) -> Path:
    return state_root / "lifecycle_locks" / "restore.lock"


def _acquire_restore_lock(state_root: Path, *, operation_id: str, target: str) -> tuple[Path | None, str | None]:
    lock_path = _restore_lock_path(state_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "restore",
        "operation_id": operation_id,
        "target": target,
        "started_at": utc_now_iso(),
        "pid": os.getpid(),
    }
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return None, "restore_lock_active"
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(support_bundle_redact(payload), sort_keys=True, ensure_ascii=True) + "\n")
    return lock_path, None


def _release_restore_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    try:
        lock_path.unlink()
    except OSError:
        pass


def _current_preferences(db_path: Path, keys: list[str]) -> dict[str, str | None]:
    conn = sqlite3.connect(str(db_path))
    try:
        out: dict[str, str | None] = {}
        for key in keys:
            row = conn.execute("SELECT value FROM preferences WHERE key = ?", (key,)).fetchone()
            out[key] = str(row[0]) if row is not None else None
        return out
    finally:
        conn.close()


def _write_preferences(db_path: Path, values: dict[str, str | None]) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        now = utc_now_iso()
        with conn:
            for key, value in values.items():
                if value is None:
                    conn.execute("DELETE FROM preferences WHERE key = ?", (key,))
                else:
                    conn.execute(
                        "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                        (key, value, now),
                    )
    finally:
        conn.close()


def _make_restore_snapshot(*, state_root: Path, db_path: Path, before: dict[str, str | None], plan_id: str, backup_path: Path) -> Path:
    root = _artifact_dir(state_root / "restore_snapshots", prefix="personal-agent-restore-snapshot")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    preferences_payload = {
        "schema": RESTORE_SNAPSHOT_SCHEMA_VERSION,
        "preferences": [{"key": key, "value": value, "present": value is not None} for key, value in sorted(before.items())],
    }
    _write_backup_json(root / "preferences_snapshot.json", preferences_payload)
    manifest = {
        "snapshot_schema_version": RESTORE_SNAPSHOT_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "plan_id": plan_id,
        "source_backup": str(backup_path),
        "captured_categories": ["allowlisted_preferences"],
        "excluded": ["raw secrets", "raw logs", "arbitrary home data", "model caches", "untrusted pack source text"],
    }
    text = json.dumps(support_bundle_redact(_bounded_backup_value(manifest)), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if len(text.encode("utf-8")) > BACKUP_MAX_FILE_BYTES:
        raise ValueError("restore_snapshot_manifest_size_cap_exceeded")
    (root / "manifest.json").write_text(text, encoding="utf-8")
    return root


def _restore_from_snapshot(snapshot_root: Path, db_path: Path) -> bool:
    try:
        payload = _read_json_file(snapshot_root / "preferences_snapshot.json")
        rows = payload.get("preferences") if isinstance(payload.get("preferences"), list) else []
        values: dict[str, str | None] = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            if key in RESTORE_ALLOWED_PREFERENCE_KEYS:
                values[key] = str(item.get("value")) if item.get("present") else None
        _write_preferences(db_path, values)
        return _current_preferences(db_path, list(values.keys())) == values
    except Exception:
        return False


def restore_backup_v1(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    context_failure = _trusted_context_failure(
        plan,
        action,
        capability_id="restore.execute",
        executor_id="operator.restore.v1",
    )
    if context_failure is not None:
        return context_failure
    plan_id = str(plan.get("plan_id") or action.get("pending_id") or "unknown").strip() or "unknown"
    state_root = _approved_restore_root(action)
    db_path = Path(str(action.get("db_path") or "")).expanduser().resolve()
    if not db_path.is_file() or not _is_contained(db_path, state_root):
        return {
            "ok": False,
            "mutated": False,
            "executor_id": "operator.restore.v1",
            "error_code": "restore_db_path_not_approved",
            "user_message": "Restore could not start because the state database path is not approved.",
        }
    backup_path = _restore_backup_path(action)
    lock_path, lock_error = _acquire_restore_lock(state_root, operation_id=plan_id, target=str(backup_path))
    if lock_error:
        return {
            "ok": False,
            "mutated": False,
            "executor_id": "operator.restore.v1",
            "error_code": lock_error,
            "user_message": "Restore is blocked because another restore/lifecycle operation appears active. I did not change state.",
            "rollback_available": False,
            "rollback_hint": "Do not clear a live restore lock automatically. Check restore status first.",
        }
    snapshot_root: Path | None = None
    staging_root: Path | None = None
    resources: list[str] = []
    try:
        validation = _validate_restore_backup(backup_path)
        if not validation.get("valid"):
            return {
                "ok": False,
                "mutated": False,
                "executor_id": "operator.restore.v1",
                "error_code": str(validation.get("error_code") or "restore_validation_failed"),
                "user_message": "Restore did not start because the Backup v1 artifact failed validation.",
                "details": {"validation": validation},
            }
        preview_fingerprint = str(action.get("restore_fingerprint") or "").strip()
        if preview_fingerprint and preview_fingerprint != validation.get("fingerprint"):
            return {
                "ok": False,
                "mutated": False,
                "executor_id": "operator.restore.v1",
                "error_code": "backup_changed_since_preview",
                "user_message": "Restore did not start because the backup changed after preview. Ask for a fresh restore preview.",
                "details": {"current_fingerprint": validation.get("fingerprint")},
            }
        supported = validation.get("supported_preferences") if isinstance(validation.get("supported_preferences"), list) else []
        target_values = {str(item["key"]): str(item["value"]) for item in supported if isinstance(item, dict) and str(item.get("key") or "") in RESTORE_ALLOWED_PREFERENCE_KEYS}
        staging_root = _artifact_dir(state_root / "restore_staging", prefix="personal-agent-restore-stage")
        staging_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        _write_backup_json(
            staging_root / "stage.json",
            {
                "stage_schema_version": RESTORE_STAGE_SCHEMA_VERSION,
                "created_at": utc_now_iso(),
                "source_backup": str(backup_path),
                "target_preferences": [{"key": key, "value": value} for key, value in sorted(target_values.items())],
                "excluded": ["secrets", "logs", "arbitrary files", "model caches", "untrusted pack source text"],
            },
        )
        resources.append(str(staging_root / "stage.json"))
        before = _current_preferences(db_path, sorted(target_values.keys()))
        snapshot_root = _make_restore_snapshot(state_root=state_root, db_path=db_path, before=before, plan_id=plan_id, backup_path=backup_path)
        resources.extend([str(snapshot_root / "manifest.json"), str(snapshot_root / "preferences_snapshot.json")])
        if not target_values:
            return {
                "ok": True,
                "mutated": False,
                "executor_id": "operator.restore.v1",
                "resources_touched": resources,
                "rollback_available": True,
                "rollback_hint": f"Pre-restore safety snapshot is at {snapshot_root}. No live state was changed.",
                "user_message": "Nothing needed restoring. This Backup v1 artifact contains no supported restorable state.",
                "details": {
                    "status": "no_op",
                    "snapshot_path": str(snapshot_root),
                    "staging_path": str(staging_root),
                    "restored_categories": [],
                    "skipped_categories": ["secrets", "logs", "runtime metadata", "pack source text", "model caches"],
                },
            }
        if before == target_values:
            return {
                "ok": True,
                "mutated": False,
                "executor_id": "operator.restore.v1",
                "resources_touched": resources,
                "rollback_available": True,
                "rollback_hint": f"Pre-restore safety snapshot is at {snapshot_root}. No live state was changed.",
                "user_message": "Nothing needed restoring. Your current supported state already matches that backup.",
                "details": {
                    "status": "no_op_already_matches",
                    "snapshot_path": str(snapshot_root),
                    "staging_path": str(staging_root),
                    "restored_categories": [],
                    "preference_keys": sorted(target_values.keys()),
                },
            }
        _write_preferences(db_path, target_values)
        after = _current_preferences(db_path, sorted(target_values.keys()))
        if after != target_values:
            rolled_back = _restore_from_snapshot(snapshot_root, db_path)
            return {
                "ok": False,
                "mutated": False,
                "executor_id": "operator.restore.v1",
                "resources_touched": resources,
                "rollback_available": True,
                "rollback_hint": f"Safety snapshot preserved at {snapshot_root}.",
                "error_code": "restore_failed_rolled_back" if rolled_back else "restore_failed_rollback_failed",
                "user_message": (
                    "The restore failed during verification, so I restored your previous state from the safety snapshot."
                    if rolled_back
                    else "The restore did not complete, and automatic rollback could not be fully verified. I preserved the safety snapshot."
                ),
                "details": {
                    "status": "restore_failed_rolled_back" if rolled_back else "restore_failed_rollback_failed",
                    "snapshot_path": str(snapshot_root),
                    "staging_path": str(staging_root),
                    "preference_keys": sorted(target_values.keys()),
                },
            }
        return {
            "ok": True,
            "mutated": True,
            "executor_id": "operator.restore.v1",
            "resources_touched": [*resources, str(db_path)],
            "rollback_available": True,
            "rollback_hint": f"Use the pre-restore safety snapshot at {snapshot_root} to restore the previous supported preferences.",
            "user_message": (
                "Restore completed and verified. I restored the supported settings and memory baseline state from the Backup v1 artifact. "
                "Secrets, logs, model files, and unrelated personal files were not restored."
            ),
            "details": {
                "status": "completed_verified",
                "snapshot_path": str(snapshot_root),
                "staging_path": str(staging_root),
                "source_backup": str(backup_path),
                "restored_categories": ["allowlisted_preferences"],
                "preference_keys": sorted(target_values.keys()),
                "skipped_categories": ["secrets", "logs", "runtime metadata", "pack source text", "model caches"],
            },
        }
    except Exception as exc:  # noqa: BLE001
        rolled_back = _restore_from_snapshot(snapshot_root, db_path) if snapshot_root is not None else False
        return {
            "ok": False,
            "mutated": False,
            "executor_id": "operator.restore.v1",
            "resources_touched": resources,
            "rollback_available": snapshot_root is not None,
            "rollback_hint": f"Safety snapshot preserved at {snapshot_root}." if snapshot_root is not None else "No rollback needed because mutation did not start.",
            "error_code": "restore_failed_rolled_back" if rolled_back else "restore_executor_exception",
            "user_message": (
                "Restore failed and I restored the previous supported state from the safety snapshot."
                if rolled_back
                else "Restore failed before I could verify completion. I preserved any safety snapshot that was created."
            ),
            "details": {"exception": exc.__class__.__name__, "snapshot_path": str(snapshot_root) if snapshot_root else None},
        }
    finally:
        _release_restore_lock(lock_path)


def _release_commit(path: Path) -> str:
    for candidate in (
        path / "agent" / "BUILD_INFO.json",
        path / "BUILD_INFO.json",
        path / "build_info.json",
    ):
        if candidate.exists() and candidate.is_file():
            try:
                payload = _read_json(candidate)
            except (OSError, json.JSONDecodeError):
                continue
            commit = str(payload.get("git_commit") or payload.get("commit") or "").strip()
            if commit:
                return commit
    marker = path / "VERSION_COMMIT"
    if marker.exists() and marker.is_file():
        try:
            return marker.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return ""


def _replace_symlink(link: Path, target: Path) -> None:
    temp_link = link.with_name(f"{link.name}.tmp-{uuid.uuid4().hex[:8]}")
    temp_link.symlink_to(target)
    os.replace(temp_link, link)


def _acquire_operation_lock(lock_path: Path, payload: dict[str, Any]) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(support_bundle_redact(payload), sort_keys=True, ensure_ascii=True) + "\n")
    return True


def _release_operation_lock(lock_path: Path, operation_id: str) -> None:
    try:
        existing = _read_json(lock_path)
    except (OSError, json.JSONDecodeError):
        return
    if str(existing.get("operation_id") or "") == operation_id:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _update_state_write(state_dir: Path, stage: str, payload: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "update_operation_schema_version": UPDATE_OPERATION_SCHEMA_VERSION,
        "stage": stage,
        "updated_at": utc_now_iso(),
        **payload,
    }
    temp = state_dir / "state.json.tmp"
    _write_json(temp, state)
    os.replace(temp, state_dir / "state.json")


def _update_result(
    plan: dict[str, Any],
    action: dict[str, Any],
    *,
    ok: bool,
    mutated: bool,
    status: str,
    message: str,
    resources: list[str] | None = None,
    rollback_available: bool = True,
    rollback_hint: str = "",
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": bool(ok),
        "mutated": bool(mutated),
        "executor_id": "operator.update.v1",
        "resources_touched": [safe_path_label(item) for item in list(resources or [])],
        "rollback_available": bool(rollback_available),
        "rollback_hint": rollback_hint,
        "error_code": error_code,
        "user_message": message,
        "details": {
            "status": status,
            "update_operation_schema_version": UPDATE_OPERATION_SCHEMA_VERSION,
            **(details or {}),
        },
    }


def execute_update_v1(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Bounded Update v1 executor using trusted staged-release inputs only."""

    action_type = str(plan.get("action_type") or "").strip().lower()
    if action_type != "operator.update":
        return _update_result(
            plan,
            action,
            ok=False,
            mutated=False,
            status="blocked",
            message="I blocked update execution because the plan action type was not operator.update.",
            rollback_available=False,
            rollback_hint="No rollback needed because nothing changed.",
            error_code="update_action_type_mismatch",
        )

    mode = str(action.get("update_mode") or "live_guarded").strip().lower()
    operation_id = str(action.get("operation_id") or plan.get("plan_id") or f"update-{uuid.uuid4().hex[:12]}").strip()
    target_commit = str(action.get("target_commit") or "").strip()
    preview_target_commit = str(action.get("preview_target_commit") or target_commit).strip()
    expected_current_commit = str(action.get("expected_current_commit") or "").strip()

    if action.get("working_tree_clean") is False:
        changed = action.get("dirty_files") if isinstance(action.get("dirty_files"), list) else []
        return _update_result(
            plan,
            action,
            ok=False,
            mutated=False,
            status="blocked_dirty_working_tree",
            message="I can’t update yet because the Personal Agent repository has uncommitted changes. I left them untouched.",
            rollback_available=False,
            rollback_hint="No rollback needed because nothing changed.",
            error_code="update_dirty_working_tree",
            details={"dirty_files": [str(item) for item in changed[:20]]},
        )

    if preview_target_commit and target_commit and preview_target_commit != target_commit:
        return _update_result(
            plan,
            action,
            ok=False,
            mutated=False,
            status="blocked_target_changed",
            message="The approved update target changed after the preview, so I blocked execution. Ask for a fresh update preview.",
            rollback_available=False,
            rollback_hint="No rollback needed because nothing changed.",
            error_code="update_target_changed_since_preview",
            details={"preview_target_commit": preview_target_commit, "target_commit": target_commit},
        )

    if mode == "live_noop":
        current_commit = str(action.get("current_runtime_commit") or "").strip()
        if not current_commit or not target_commit or current_commit != target_commit:
            return _update_result(
                plan,
                action,
                ok=False,
                mutated=False,
                status="blocked_live_update_not_enabled",
                message=(
                    "Live update execution is blocked because this is not a verified no-op. "
                    "I did not fetch code, promote a runtime, restart services, or change files."
                ),
                rollback_available=False,
                rollback_hint="No rollback needed because nothing changed.",
                error_code="update_live_promotion_not_enabled",
                details={"current_runtime_commit": current_commit, "target_commit": target_commit},
            )
        return _update_result(
            plan,
            action,
            ok=True,
            mutated=False,
            status="already_current",
            message="You’re already running the current approved version. Nothing needed updating.",
            rollback_available=True,
            rollback_hint="The current runtime was not changed.",
            details={"current_runtime_commit": current_commit, "target_commit": target_commit},
        )

    if mode not in {"fixture_staged_release", "primary_staged_release"}:
        return _update_result(
            plan,
            action,
            ok=False,
            mutated=False,
            status="blocked_live_update_not_enabled",
            message=(
                "Update execution is enabled only for the trusted staged-release runner. "
                "This request did not include an approved internal update target, so I did not mutate state."
            ),
            rollback_available=False,
            rollback_hint="No rollback needed because nothing changed.",
            error_code="update_live_promotion_not_enabled",
        )

    state_root = Path(str(action.get("state_root") or "")).expanduser()
    runtime_root = Path(str(action.get("runtime_root") or "")).expanduser()
    releases_root = Path(str(action.get("releases_root") or runtime_root / "releases")).expanduser()
    current_link = Path(str(action.get("current_link") or runtime_root / "current")).expanduser()
    source_release = Path(str(action.get("staged_source_path") or "")).expanduser()
    target_release_id = str(action.get("target_release_id") or f"update-{target_commit[:12]}").strip()

    try:
        state_root = state_root.resolve()
        runtime_root = runtime_root.resolve()
        releases_root = releases_root.resolve()
        current_link_parent = current_link.parent.resolve()
        source_release = source_release.resolve()
    except OSError as exc:
        return _update_result(
            plan,
            action,
            ok=False,
            mutated=False,
            status="blocked_invalid_path",
            message="I blocked the update because one of the internal update paths could not be resolved.",
            rollback_available=False,
            rollback_hint="No rollback needed because nothing changed.",
            error_code="update_invalid_path",
            details={"exception": exc.__class__.__name__},
        )

    current_link = current_link_parent / current_link.name
    if not _is_contained(releases_root, runtime_root) or not _is_contained(current_link_parent, runtime_root):
        return _update_result(
            plan,
            action,
            ok=False,
            mutated=False,
            status="blocked_path_escape",
            message="I blocked the update because runtime paths were outside the approved Personal Agent runtime root.",
            rollback_available=False,
            rollback_hint="No rollback needed because nothing changed.",
            error_code="update_path_escape",
        )
    if not source_release.exists() or not source_release.is_dir() or source_release.is_symlink():
        return _update_result(
            plan,
            action,
            ok=False,
            mutated=False,
            status="blocked_missing_staged_source",
            message="I blocked the update because the approved staged release source was missing.",
            rollback_available=False,
            rollback_hint="No rollback needed because nothing changed.",
            error_code="update_staged_source_missing",
        )
    if not target_release_id or any(ch in target_release_id for ch in {"/", "\\", "\x00"}) or ".." in Path(target_release_id).parts:
        return _update_result(
            plan,
            action,
            ok=False,
            mutated=False,
            status="blocked_invalid_release_id",
            message="I blocked the update because the internal target release id was not safe.",
            rollback_available=False,
            rollback_hint="No rollback needed because nothing changed.",
            error_code="update_invalid_release_id",
        )

    fixture_mode = "primary_update_proof" if mode == "primary_staged_release" else "strict"
    api_service_name = str(action.get("api_service_name") or "").strip()
    verify_base_url = str(action.get("verify_base_url") or "").strip()
    proof_marker_path = str(action.get("proof_marker_path") or "").strip()
    if mode == "primary_staged_release":
        expected_state_root = (Path.home() / ".local/share/personal-agent").expanduser().resolve()
        expected_runtime_root = (expected_state_root / "runtime").resolve()
        if state_root != expected_state_root or runtime_root != expected_runtime_root:
            return _update_result(
                plan,
                action,
                ok=False,
                mutated=False,
                status="blocked_primary_path_mismatch",
                message="I blocked primary update handoff because the approved target was not the primary Personal Agent runtime.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing changed.",
                error_code="update_primary_path_mismatch",
            )
        staged_root = expected_state_root / "host_lifecycle" / "staged_sources"
        if not _is_contained(source_release, staged_root.resolve()):
            return _update_result(
                plan,
                action,
                ok=False,
                mutated=False,
                status="blocked_primary_staged_source_escape",
                message="I blocked primary update handoff because the staged release source was outside the approved host lifecycle staging root.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing changed.",
                error_code="update_primary_staged_source_escape",
            )
        if api_service_name != "personal-agent-api.service" or verify_base_url != "http://127.0.0.1:8765":
            return _update_result(
                plan,
                action,
                ok=False,
                mutated=False,
                status="blocked_primary_service_mismatch",
                message="I blocked primary update handoff because the service or API endpoint was not the approved primary target.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing changed.",
                error_code="update_primary_service_mismatch",
            )
        marker = Path(proof_marker_path).expanduser() if proof_marker_path else expected_state_root / "host_lifecycle" / "primary_update_enablement.marker"
        try:
            marker = marker.resolve()
        except OSError:
            marker = expected_state_root / "host_lifecycle" / "primary_update_enablement.marker"
        if not marker.is_file() or not _is_contained(marker, expected_state_root):
            return _update_result(
                plan,
                action,
                ok=False,
                mutated=False,
                status="blocked_primary_proof_marker_missing",
                message="I blocked primary update handoff because the required installed-host proof marker is missing.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing changed.",
                error_code="update_primary_proof_marker_missing",
            )
        proof_marker_path = str(marker)

    state_dir = state_root / "host_lifecycle" / "operations" / operation_id
    receipt_path = state_dir / "receipt.json"
    operation_state_path = state_dir / "state.json"
    operation_record_path = state_dir / "operation.json"
    expires_at_raw = action.get("expires_at")
    if isinstance(expires_at_raw, (int, float)):
        expires_at = datetime.fromtimestamp(float(expires_at_raw), timezone.utc).isoformat()
    elif isinstance(expires_at_raw, str) and expires_at_raw.strip():
        expires_at = expires_at_raw.strip()
    else:
        expires_at = ""
    operation_record = attach_approved_hash(
        {
            "schema_version": HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION,
            "runner_version": HOST_LIFECYCLE_RUNNER_VERSION,
            "operation_id": operation_id,
            "operation_type": "update",
            "plan_id": plan.get("plan_id"),
            "created_at": utc_now_iso(),
            "expires_at": expires_at,
            "current_stage": "created",
            "fixture_mode": fixture_mode,
            "state_root": str(state_root),
            "runtime_root": str(runtime_root),
            "releases_root": str(releases_root),
            "current_link": str(current_link),
            "staged_source_path": str(source_release),
            "target_release_id": target_release_id,
            "current_runtime_commit": expected_current_commit,
            "target_commit": target_commit,
            "operation_state_path": str(operation_state_path),
            "receipt_path": str(receipt_path),
            "force_post_promotion_failure": bool(action.get("force_post_promotion_failure")),
            "api_service_name": api_service_name,
            "verify_base_url": verify_base_url,
            "proof_marker_path": proof_marker_path,
            "capability_policy": _capability_policy_record(action, plan),
        }
    )
    _write_json_unredacted(operation_record_path, operation_record)
    resources = [str(operation_record_path), str(operation_state_path), str(receipt_path)]
    if mode == "primary_staged_release":
        try:
            handoff = _launch_host_lifecycle_runner_systemd("update", operation_record_path, operation_id=operation_id, timeout=20)
        except Exception as exc:  # noqa: BLE001
            return _update_result(
                plan,
                action,
                ok=False,
                mutated=False,
                status="host_runner_handoff_failed",
                message="The trusted host lifecycle runner could not be started. I left the current runtime untouched.",
                rollback_available=False,
                rollback_hint="No rollback needed because host handoff did not start.",
                error_code="update_host_runner_handoff_failed",
                resources=resources,
                details={"exception": exc.__class__.__name__},
            )
        if not handoff.get("ok"):
            return _update_result(
                plan,
                action,
                ok=False,
                mutated=False,
                status="host_runner_handoff_failed",
                message="The trusted host lifecycle runner refused the update handoff. I left the current runtime untouched.",
                rollback_available=False,
                rollback_hint="No rollback needed because host handoff did not start.",
                error_code="update_host_runner_handoff_failed",
                resources=resources,
                details={"handoff": handoff},
            )
        return _update_result(
            plan,
            action,
            ok=True,
            mutated=True,
            status="in_progress",
            message=(
                "The update has started. I built and checked the new release, created a rollback checkpoint, "
                "and handed the switch to the trusted host runner. The assistant may disconnect briefly."
            ),
            rollback_available=True,
            rollback_hint="The host lifecycle runner will keep the previous working release as a rollback checkpoint.",
            resources=resources,
            details={
                "operation_id": operation_id,
                "operation_record_path": str(operation_record_path),
                "operation_state_path": str(operation_state_path),
                "receipt_path": str(receipt_path),
                "target_commit": target_commit,
                "target_release_id": target_release_id,
                "handoff": handoff,
            },
        )
    try:
        runner_result = _launch_host_lifecycle_runner("update", operation_record_path, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return _update_result(
            plan,
            action,
            ok=False,
            mutated=False,
            status="host_runner_failed",
            message="The trusted host lifecycle runner failed before update mutation. I left the current runtime untouched.",
            rollback_available=False,
            rollback_hint="No rollback needed because the runner did not report mutation.",
            error_code="update_host_runner_failed",
            resources=resources,
            details={"exception": exc.__class__.__name__},
        )
    status = str(runner_result.get("status") or "runner_unknown")
    mutated = bool(runner_result.get("mutated"))
    ok = bool(runner_result.get("ok"))
    if ok:
        return _update_result(
            plan,
            action,
            ok=True,
            mutated=mutated,
            status=status,
            message=(
                f"Update completed and verified. Personal Agent is now running commit {target_commit[:12]}. "
                "The previous working release was kept as a rollback checkpoint."
            ),
            rollback_available=True,
            rollback_hint="Use the host lifecycle receipt and checkpoint for manual rollback if needed.",
            resources=resources,
            details=runner_result,
        )
    rollback_verified = bool(runner_result.get("rollback_verified"))
    return _update_result(
        plan,
        action,
        ok=False,
        mutated=mutated,
        status=status,
        message=(
            "The update failed its readiness check, so the host runner restored the previous working release."
            if rollback_verified
            else "The update failed before completion. The host runner preserved the operation receipt and stopped further changes."
        ),
        rollback_available=rollback_verified or not mutated,
        rollback_hint="See the host lifecycle receipt for checkpoint and recovery details.",
        error_code=status,
        resources=resources,
        details=runner_result,
    )

    lock_path = state_root / "lifecycle_locks" / "update.lock"
    state_dir = state_root / "update_operations" / operation_id
    lock_payload = {
        "operation_type": "operator.update",
        "operation_id": operation_id,
        "started_at": utc_now_iso(),
        "source_commit": expected_current_commit,
        "target_commit": target_commit,
        "stage": "starting",
    }
    if not _acquire_operation_lock(lock_path, lock_payload):
        return _update_result(
            plan,
            action,
            ok=False,
            mutated=False,
            status="blocked_lock_conflict",
            message="Another lifecycle operation is already using the update lock. I did not start a second update.",
            rollback_available=False,
            rollback_hint="No rollback needed because nothing changed.",
            error_code="update_lock_conflict",
        )

    resources: list[str] = [str(lock_path), str(state_dir)]
    previous_target: Path | None = None
    promoted = False
    checkpoint: dict[str, Any] = {}
    try:
        _update_state_write(state_dir, "validating", lock_payload)
        if not current_link.is_symlink():
            return _update_result(
                plan,
                action,
                ok=False,
                mutated=False,
                status="blocked_current_not_symlink",
                message="I blocked the update because runtime/current is not the expected Personal Agent symlink.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing changed.",
                error_code="update_current_not_symlink",
            )
        previous_target = current_link.resolve()
        previous_commit = _release_commit(previous_target)
        if expected_current_commit and previous_commit and previous_commit != expected_current_commit:
            return _update_result(
                plan,
                action,
                ok=False,
                mutated=False,
                status="blocked_current_changed",
                message="The running runtime changed after the update preview, so I blocked execution. Ask for a fresh update preview.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing changed.",
                error_code="update_current_changed_since_preview",
                details={"expected_current_commit": expected_current_commit, "actual_current_commit": previous_commit},
            )
        source_commit = _release_commit(source_release)
        if target_commit and source_commit != target_commit:
            return _update_result(
                plan,
                action,
                ok=False,
                mutated=False,
                status="blocked_target_metadata_mismatch",
                message="I blocked the update because the staged release metadata did not match the approved target commit.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing changed.",
                error_code="update_target_metadata_mismatch",
                details={"target_commit": target_commit, "staged_commit": source_commit},
            )
        if previous_commit and target_commit and previous_commit == target_commit:
            return _update_result(
                plan,
                action,
                ok=True,
                mutated=False,
                status="already_current",
                message="You’re already running the approved target version. Nothing needed updating.",
                rollback_available=True,
                rollback_hint="The current runtime was not changed.",
                details={"current_runtime_commit": previous_commit, "target_commit": target_commit},
            )

        checkpoint_dir = state_root / "update_checkpoints" / operation_id
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = {
            "update_checkpoint_schema_version": UPDATE_CHECKPOINT_SCHEMA_VERSION,
            "operation_id": operation_id,
            "created_at": utc_now_iso(),
            "previous_release_path": str(previous_target),
            "previous_runtime_commit": previous_commit,
            "current_link": str(current_link),
            "target_commit": target_commit,
        }
        _write_json(checkpoint_dir / "manifest.json", checkpoint)
        resources.append(str(checkpoint_dir))
        _update_state_write(state_dir, "checkpoint_created", {**lock_payload, "checkpoint": checkpoint})

        final_release = releases_root / target_release_id
        stage_release = releases_root / f"{target_release_id}.staging-{uuid.uuid4().hex[:8]}"
        if final_release.exists():
            existing_commit = _release_commit(final_release)
            if existing_commit != target_commit:
                return _update_result(
                    plan,
                    action,
                    ok=False,
                    mutated=False,
                    status="blocked_existing_release_mismatch",
                    message="I blocked the update because an existing staged release id points at a different commit.",
                    rollback_available=True,
                    rollback_hint=f"Previous runtime remains at {safe_path_label(previous_target)}.",
                    error_code="update_existing_release_mismatch",
                    resources=resources,
                    details={"existing_commit": existing_commit, "target_commit": target_commit},
                )
        else:
            _update_state_write(state_dir, "staging", {**lock_payload, "checkpoint": checkpoint})
            shutil.copytree(source_release, stage_release, symlinks=False)
            resources.append(str(stage_release))
            staged_commit = _release_commit(stage_release)
            if staged_commit != target_commit:
                shutil.rmtree(stage_release, ignore_errors=True)
                return _update_result(
                    plan,
                    action,
                    ok=False,
                    mutated=False,
                    status="failed_before_promotion",
                    message="The staged release failed its metadata check, so I left the current runtime untouched.",
                    rollback_available=True,
                    rollback_hint=f"Previous runtime remains at {safe_path_label(previous_target)}.",
                    error_code="update_pre_promotion_check_failed",
                    resources=resources,
                    details={"staged_commit": staged_commit, "target_commit": target_commit},
                )
            os.replace(stage_release, final_release)
            resources.append(str(final_release))

        _update_state_write(state_dir, "promoting", {**lock_payload, "checkpoint": checkpoint, "staged_release": str(final_release)})
        _replace_symlink(current_link, final_release)
        promoted = True
        resources.append(str(current_link))

        _update_state_write(state_dir, "verifying", {**lock_payload, "checkpoint": checkpoint, "promoted_release": str(final_release)})
        verified_commit = _release_commit(current_link.resolve())
        if bool(action.get("force_post_promotion_failure")) or verified_commit != target_commit:
            _replace_symlink(current_link, previous_target)
            rollback_commit = _release_commit(current_link.resolve())
            rollback_verified = bool(previous_commit and rollback_commit == previous_commit)
            status = "update_failed_rolled_back" if rollback_verified else "update_failed_rollback_failed"
            _update_state_write(
                state_dir,
                status,
                {
                    **lock_payload,
                    "checkpoint": checkpoint,
                    "verified_commit": verified_commit,
                    "rollback_commit": rollback_commit,
                    "rollback_verified": rollback_verified,
                },
            )
            return _update_result(
                plan,
                action,
                ok=False,
                mutated=True,
                status=status,
                message=(
                    "The update failed its readiness check, so I restored the previous working release."
                    if rollback_verified
                    else "The update failed, and I could not fully verify the previous release after rollback. I stopped further changes."
                ),
                rollback_available=rollback_verified,
                rollback_hint=(
                    f"Runtime/current is back at {safe_path_label(previous_target)}."
                    if rollback_verified
                    else f"Use the rollback checkpoint at {safe_path_label(checkpoint_dir)}."
                ),
                error_code=status,
                resources=resources,
                details={
                    "checkpoint": checkpoint,
                    "verified_commit": verified_commit,
                    "rollback_commit": rollback_commit,
                    "rollback_verified": rollback_verified,
                },
            )

        _update_state_write(
            state_dir,
            "completed_verified",
            {**lock_payload, "checkpoint": checkpoint, "promoted_release": str(final_release), "verified_commit": verified_commit},
        )
        return _update_result(
            plan,
            action,
            ok=True,
            mutated=True,
            status="completed_verified",
            message=(
                f"Update completed and verified. Personal Agent is now running commit {target_commit[:12]}. "
                "The previous working release was kept as a rollback checkpoint."
            ),
            rollback_available=True,
            rollback_hint=f"Switch runtime/current back to {safe_path_label(previous_target)} if manual rollback is needed.",
            resources=resources,
            details={
                "checkpoint": checkpoint,
                "previous_runtime_commit": previous_commit,
                "target_commit": target_commit,
                "promoted_release": str(final_release),
                "verified_commit": verified_commit,
            },
        )
    except Exception as exc:  # noqa: BLE001 - executor boundary must stay structured.
        rollback_verified = False
        rollback_commit = ""
        if promoted and previous_target is not None:
            try:
                _replace_symlink(current_link, previous_target)
                rollback_commit = _release_commit(current_link.resolve())
                rollback_verified = not expected_current_commit or rollback_commit == expected_current_commit
            except Exception:  # noqa: BLE001 - report rollback failure below.
                rollback_verified = False
        status = "update_failed_rolled_back" if rollback_verified else ("update_failed_rollback_failed" if promoted else "failed_before_promotion")
        _update_state_write(
            state_dir,
            status,
            {
                **lock_payload,
                "checkpoint": checkpoint,
                "exception": exc.__class__.__name__,
                "rollback_commit": rollback_commit,
                "rollback_verified": rollback_verified,
            },
        )
        return _update_result(
            plan,
            action,
            ok=False,
            mutated=bool(promoted),
            status=status,
            message=(
                "The update failed before promotion, so I left the current runtime untouched."
                if not promoted
                else (
                    "The update failed after promotion, so I restored the previous working release."
                    if rollback_verified
                    else "The update failed after promotion, and rollback could not be fully verified."
                )
            ),
            rollback_available=rollback_verified or not promoted,
            rollback_hint=(
                "No rollback needed because promotion did not happen."
                if not promoted
                else f"Use the rollback checkpoint at {safe_path_label(state_root / 'update_checkpoints' / operation_id)}."
            ),
            error_code=status,
            resources=resources,
            details={"exception": exc.__class__.__name__, "rollback_commit": rollback_commit, "rollback_verified": rollback_verified},
        )
    finally:
        _release_operation_lock(lock_path, operation_id)


def _uninstall_result(
    *,
    ok: bool,
    mutated: bool,
    status: str,
    message: str,
    resources: list[str] | None = None,
    rollback_available: bool = False,
    rollback_hint: str = "",
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": bool(ok),
        "mutated": bool(mutated),
        "executor_id": "operator.uninstall.v1",
        "resources_touched": [safe_path_label(item) for item in list(resources or [])],
        "rollback_available": bool(rollback_available),
        "rollback_hint": rollback_hint,
        "error_code": error_code,
        "user_message": message,
        "details": {
            "status": status,
            "uninstall_operation_schema_version": UNINSTALL_OPERATION_SCHEMA_VERSION,
            **(details or {}),
        },
    }


def _snapshot_hash(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(support_bundle_redact(snapshot), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _uninstall_resource_path(resource: dict[str, Any]) -> Path:
    return Path(str(resource.get("path") or "")).expanduser()


def _validate_uninstall_resource(
    resource: dict[str, Any],
    *,
    containment_root: Path,
    removable_roots: list[Path],
    preserved_roots: list[Path] | None = None,
) -> tuple[bool, str, Path | None]:
    if not isinstance(resource, dict):
        return False, "resource_not_object", None
    if not bool(resource.get("owned")):
        return False, "resource_not_owned", None
    raw = str(resource.get("path") or "").strip()
    if not raw or ".." in Path(raw).parts:
        return False, "resource_path_invalid", None
    try:
        path = _uninstall_resource_path(resource)
        resolved = path.resolve(strict=False)
        lexical = path if path.is_absolute() else path.resolve(strict=False)
    except OSError:
        return False, "resource_path_unresolvable", None
    def _lexically_contained(path_value: Path, root_value: Path) -> bool:
        path_text = str(path_value)
        root_text = str(root_value)
        return path_text == root_text or path_text.startswith(root_text.rstrip("/") + "/")

    if not _lexically_contained(lexical, containment_root):
        return False, "resource_outside_uninstall_root", lexical
    if not any(_lexically_contained(lexical, root) for root in removable_roots):
        return False, "resource_outside_removable_roots", lexical
    expected_type = str(resource.get("expected_type") or "").strip().lower()
    if lexical.exists() or lexical.is_symlink():
        try:
            if lexical.is_mount():
                return False, "resource_is_mount_point", lexical
        except OSError:
            return False, "resource_mount_check_failed", lexical
        if lexical.is_symlink():
            try:
                target = lexical.resolve(strict=True)
            except OSError:
                target = lexical.resolve(strict=False)
            if not any(_is_contained(target, root) for root in removable_roots):
                return False, "resource_symlink_escape", lexical
        if expected_type == "directory" and not lexical.is_dir():
            return False, "resource_type_mismatch", lexical
        if expected_type == "file" and not lexical.is_file():
            return False, "resource_type_mismatch", lexical
        if expected_type == "symlink" and not lexical.is_symlink():
            return False, "resource_type_mismatch", lexical
    for preserved in list(preserved_roots or []):
        if _lexically_contained(lexical, preserved):
            return False, "resource_overlaps_preserved_root", lexical
    return True, "ok", lexical


def _create_uninstall_final_backup(
    *,
    backup_root: Path,
    operation_id: str,
    snapshot: dict[str, Any],
    preserved_resources: list[dict[str, Any]],
) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    root = backup_root / f"personal-agent-uninstall-backup-{operation_id}"
    if root.exists():
        manifest = root / "manifest.json"
        if manifest.is_file():
            return root
        raise ValueError("uninstall_backup_existing_invalid")
    root.mkdir(mode=0o700)
    files: dict[str, dict[str, Any]] = {
        "backup_summary.json": {
            "backup_schema_version": BACKUP_SCHEMA_VERSION,
            "purpose": "final_uninstall_safety_backup",
            "restore_status": RESTORE_V1_CAPABILITY,
        },
        "uninstall_target_snapshot.json": snapshot,
        "preserved_inventory.json": {"preserved_resources": preserved_resources},
        "preferences_summary.json": {
            "mode": "summary_only",
            "note": "Fixture uninstall backup preserves supported state summaries only; raw secrets are excluded.",
        },
        "runtime_config_summary.json": {
            "mode": "summary_only",
            "install_metadata": snapshot.get("install_metadata") if isinstance(snapshot.get("install_metadata"), dict) else {},
        },
    }
    file_sizes: dict[str, int] = {}
    for name, payload in files.items():
        target = root / name
        _write_json(target, payload)
        file_sizes[name] = target.stat().st_size
    included_files = sorted([*files.keys(), "manifest.json"])
    manifest = {
        "backup_schema_version": BACKUP_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "runtime_commit": snapshot.get("runtime_commit"),
        "runtime_instance": "uninstall-fixture",
        "included_files": included_files,
        "excluded_files": [
            "raw secret-store files",
            "raw logs",
            "arbitrary home directory files",
            "model caches",
            "external pack source text",
        ],
        "file_sizes": file_sizes,
        "total_size_bytes": sum(file_sizes.values()),
        "restore_status": RESTORE_V1_CAPABILITY,
        "live_restore": RESTORE_V1_CAPABILITY,
        "uninstall_operation_id": operation_id,
    }
    _write_json(root / "manifest.json", manifest)
    return root


def _remove_uninstall_target(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    try:
        path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass


def _run_uninstall_helper(operation_state: dict[str, Any]) -> dict[str, Any]:
    operation_id = str(operation_state.get("operation_id") or "").strip()
    snapshot = operation_state.get("target_snapshot") if isinstance(operation_state.get("target_snapshot"), dict) else {}
    receipt_path = Path(str(operation_state.get("receipt_path") or "")).expanduser().resolve()
    backup_path = Path(str(operation_state.get("final_backup_path") or "")).expanduser().resolve()
    resources = snapshot.get("removable_resources") if isinstance(snapshot.get("removable_resources"), list) else []
    preserved = snapshot.get("preserved_resources") if isinstance(snapshot.get("preserved_resources"), list) else []
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started_at = utc_now_iso()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    initial_receipt = {
        "uninstall_receipt_schema_version": UNINSTALL_RECEIPT_SCHEMA_VERSION,
        "operation_id": operation_id,
        "mode": UNINSTALL_MODE_PRESERVE_DATA,
        "started_at": started_at,
        "status": "running",
        "final_backup_path": str(backup_path),
        "removed_resources": [],
        "skipped_resources": [],
        "preserved_resources": preserved,
        "capability_policy": operation_state.get("capability_policy") if isinstance(operation_state.get("capability_policy"), dict) else None,
        "reinstall_guidance": "Reinstall from the preserved Personal Agent repository, then restore supported state from the final uninstall backup.",
    }
    _write_json(receipt_path, initial_receipt)

    force_failure_after = str(operation_state.get("force_failure_after_resource_id") or "").strip()
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        resource_id = str(resource.get("id") or resource.get("path") or "unknown")
        path = _uninstall_resource_path(resource)
        if not path.is_absolute():
            path = path.resolve(strict=False)
        if not path.exists() and not path.is_symlink():
            skipped.append({"id": resource_id, "path": str(path), "reason": "already_absent"})
            continue
        try:
            _remove_uninstall_target(path)
            removed.append({"id": resource_id, "path": str(path), "class": resource.get("class")})
            if force_failure_after and resource_id == force_failure_after:
                raise RuntimeError("forced_uninstall_partial_failure")
        except Exception as exc:  # noqa: BLE001 - receipt must stay truthful.
            failures.append({"id": resource_id, "path": str(path), "error": exc.__class__.__name__})
            break

    preserved_checks: list[dict[str, Any]] = []
    for resource in preserved:
        if not isinstance(resource, dict):
            continue
        raw = str(resource.get("path") or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        preserved_checks.append(
            {
                "id": str(resource.get("id") or raw),
                "path": str(path),
                "exists": path.exists() or path.is_symlink(),
                "class": resource.get("class"),
            }
        )

    completed = not failures
    status = "completed_verified" if completed else "partial_uninstall"
    receipt = {
        **initial_receipt,
        "finished_at": utc_now_iso(),
        "status": status,
        "removed_resources": removed,
        "skipped_resources": skipped,
        "preserved_resources": preserved_checks,
        "final_backup_path": str(backup_path),
        "final_backup_exists": backup_path.exists(),
        "warnings": ["Uninstall is not automatically reversible; reinstall then restore from the final backup."],
        "partial_failures": failures,
        "verification": {
            "removed_absent": all(not Path(str(item["path"])).exists() for item in removed),
            "preserved_checked": preserved_checks,
            "receipt_finalized": True,
        },
    }
    _write_json(receipt_path, receipt)
    return receipt


def run_uninstall_helper_state_file(path: str | Path) -> dict[str, Any]:
    state_path = Path(path).expanduser().resolve()
    operation_state = _read_json(state_path)
    if not isinstance(operation_state, dict):
        raise ValueError("uninstall_helper_state_not_object")
    return _run_uninstall_helper(operation_state)


def _launch_uninstall_helper_state_file(state_path: Path) -> dict[str, Any]:
    helper = Path(__file__).resolve().parents[1] / "scripts" / "host_lifecycle_runner.py"
    if not helper.is_file():
        raise FileNotFoundError(f"host_lifecycle_runner_missing:{helper}")
    proc = subprocess.run(
        [sys.executable, str(helper), "uninstall", "--operation-record", str(state_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=45,
        check=False,
    )
    if proc.returncode not in {0, 2}:
        raise RuntimeError(f"host_lifecycle_runner_failed:{proc.returncode}:{(proc.stdout or '')[:1000]}")
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"host_lifecycle_runner_bad_json:{(proc.stdout or '')[:1000]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("host_lifecycle_runner_non_object_json")
    return payload


def _launch_host_lifecycle_runner(operation: str, record_path: Path, *, timeout: int = 60) -> dict[str, Any]:
    runner = Path(__file__).resolve().parents[1] / "scripts" / "host_lifecycle_runner.py"
    if not runner.is_file():
        raise FileNotFoundError(f"host_lifecycle_runner_missing:{runner}")
    proc = subprocess.run(
        [sys.executable, str(runner), operation, "--operation-record", str(record_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if proc.returncode not in {0, 2}:
        raise RuntimeError(f"host_lifecycle_runner_failed:{proc.returncode}:{(proc.stdout or '')[:1000]}")
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"host_lifecycle_runner_bad_json:{(proc.stdout or '')[:1000]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("host_lifecycle_runner_non_object_json")
    return payload


def _launch_host_lifecycle_runner_systemd(operation: str, record_path: Path, *, operation_id: str, timeout: int = 20) -> dict[str, Any]:
    if operation not in {"update", "uninstall"}:
        raise ValueError("host_lifecycle_operation_not_supported")
    runner_candidates = [
        Path(str(os.environ.get("PERSONAL_AGENT_HOST_LIFECYCLE_RUNNER") or "")),
        Path(__file__).resolve().parents[1] / "scripts" / "host_lifecycle_runner.py",
        Path.home() / "personal-agent" / "scripts" / "host_lifecycle_runner.py",
    ]
    runner = next((candidate.expanduser().resolve() for candidate in runner_candidates if str(candidate) and candidate.expanduser().is_file()), None)
    if runner is None:
        raise FileNotFoundError("host_lifecycle_runner_missing")
    allowed_runner_roots = {
        (Path(__file__).resolve().parents[1] / "scripts").resolve(),
        (Path.home() / "personal-agent" / "scripts").resolve(),
    }
    if runner.parent not in allowed_runner_roots:
        raise FileNotFoundError("host_lifecycle_runner_untrusted_path")
    systemd_run = shutil.which("systemd-run") or "/usr/bin/systemd-run"
    if not Path(systemd_run).is_file():
        raise FileNotFoundError(f"systemd_run_missing:{systemd_run}")
    safe_operation_id = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in operation_id.lower())[:48] or uuid.uuid4().hex[:12]
    unit_name = f"personal-agent-host-lifecycle-{operation}-{safe_operation_id}.service"
    proc = subprocess.run(
        [
            systemd_run,
            "--user",
            "--collect",
            f"--unit={unit_name.removesuffix('.service')}",
            sys.executable,
            str(runner),
            operation,
            "--operation-record",
            str(record_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "unit": unit_name,
        "returncode": proc.returncode,
        "output": (proc.stdout or "")[:1000],
    }


def execute_uninstall_v1(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    action_type = str(plan.get("action_type") or "").strip().lower()
    if action_type != "operator.uninstall":
        return _uninstall_result(
            ok=False,
            mutated=False,
            status="blocked",
            message="I blocked uninstall because the plan action type was not operator.uninstall.",
            rollback_hint="No rollback needed because nothing changed.",
            error_code="uninstall_action_type_mismatch",
        )
    mode = str(action.get("uninstall_mode") or UNINSTALL_MODE_PRESERVE_DATA).strip().lower()
    if mode != UNINSTALL_MODE_PRESERVE_DATA:
        return _uninstall_result(
            ok=False,
            mutated=False,
            status="blocked_unsupported_mode",
            message="I blocked uninstall because purge/removal of user data is not enabled in Uninstall v1.",
            rollback_hint="No rollback needed because nothing changed.",
            error_code="uninstall_mode_not_supported",
        )
    execution_mode = str(action.get("uninstall_execution_mode") or "live_guarded").strip().lower()
    allowed_execution_modes = {"fixture_preserve_data", "primary_preserve_data", "production_shaped_preserve_data"}
    if execution_mode not in allowed_execution_modes:
        return _uninstall_result(
            ok=False,
            mutated=False,
            status="blocked_live_uninstall",
            message=(
                "Live uninstall is blocked from this chat unless the target is an approved isolated fixture. "
                "I did not stop services, remove runtime files, delete state, or uninstall anything."
            ),
            rollback_hint="No rollback needed because nothing changed.",
            error_code="uninstall_live_execution_not_enabled",
        )
    is_primary = execution_mode == "primary_preserve_data"
    is_production_shaped = execution_mode == "production_shaped_preserve_data"

    operation_id = str(action.get("operation_id") or plan.get("plan_id") or f"uninstall-{uuid.uuid4().hex[:12]}").strip()
    snapshot = action.get("target_snapshot") if isinstance(action.get("target_snapshot"), dict) else {}
    expected_hash = str(action.get("target_snapshot_hash") or "").strip()
    actual_hash = _snapshot_hash(snapshot)
    if expected_hash and actual_hash != expected_hash:
        return _uninstall_result(
            ok=False,
            mutated=False,
            status="blocked_snapshot_changed",
            message="The uninstall target changed after preview, so I blocked execution. Ask for a fresh uninstall preview.",
            rollback_hint="No rollback needed because nothing changed.",
            error_code="uninstall_target_changed_since_preview",
            details={"expected_hash": expected_hash, "actual_hash": actual_hash},
        )
    if str(snapshot.get("mode") or "") != UNINSTALL_MODE_PRESERVE_DATA:
        return _uninstall_result(
            ok=False,
            mutated=False,
            status="blocked_missing_preserve_data_mode",
            message="I blocked uninstall because the target snapshot was not preserve-data mode.",
            rollback_hint="No rollback needed because nothing changed.",
            error_code="uninstall_preserve_data_required",
        )
    if not is_primary and not bool(snapshot.get("fixture_marker")):
        return _uninstall_result(
            ok=False,
            mutated=False,
            status="blocked_missing_fixture_marker",
            message="I blocked uninstall because the target snapshot was not marked as an approved isolated fixture.",
            rollback_hint="No rollback needed because nothing changed.",
            error_code="uninstall_fixture_marker_missing",
        )

    try:
        fixture_root = Path(str(snapshot.get("fixture_root") or "")).expanduser().resolve()
        state_root = Path(str(action.get("state_root") or snapshot.get("state_root") or "")).expanduser().resolve()
        receipt_root = Path(str(action.get("receipt_root") or snapshot.get("receipt_root") or state_root / "uninstall_receipts")).expanduser().resolve()
        backup_root = Path(str(action.get("backup_root") or snapshot.get("backup_root") or state_root / "backups")).expanduser().resolve()
        runtime_root = Path(str(snapshot.get("runtime_root") or "")).expanduser().resolve()
    except OSError as exc:
        return _uninstall_result(
            ok=False,
            mutated=False,
            status="blocked_invalid_path",
            message="I blocked uninstall because one of the internal paths could not be resolved.",
            rollback_hint="No rollback needed because nothing changed.",
            error_code="uninstall_invalid_path",
            details={"exception": exc.__class__.__name__},
        )
    containment_root = fixture_root
    if is_primary:
        expected_state = (Path.home() / ".local/share/personal-agent").expanduser().resolve()
        expected_runtime = (expected_state / "runtime").resolve()
        policy_status = validate_primary_uninstall_marker(
            build_policy_context(state_root=expected_state),
            expected_fingerprint=str(action.get("primary_uninstall_marker_fingerprint") or "") or None,
        )
        if state_root != expected_state or runtime_root != expected_runtime or not policy_status.enabled:
            return _uninstall_result(
                ok=False,
                mutated=False,
                status="blocked_primary_enablement_invalid",
                message=(
                    "Primary preserve-data uninstall is not enabled on this host "
                    f"({policy_status.reason}). "
                    "I did not stop services, remove runtime files, delete state, or uninstall anything."
                ),
                rollback_hint="No rollback needed because nothing changed.",
                error_code="uninstall_live_execution_not_enabled",
                details={
                    "policy_status": policy_status.redacted_dict(),
                    "operator_status_command": "python scripts/primary_uninstall_policy.py status",
                },
            )
        snapshot_fingerprint = str(snapshot.get("primary_uninstall_marker_fingerprint") or "")
        if snapshot_fingerprint and snapshot_fingerprint != str(policy_status.fingerprint or ""):
            return _uninstall_result(
                ok=False,
                mutated=False,
                status="blocked_primary_enablement_changed",
                message=(
                    "Primary uninstall enablement changed after preview, so I blocked execution. "
                    "Ask for a fresh uninstall preview."
                ),
                rollback_hint="No rollback needed because nothing changed.",
                error_code="uninstall_primary_enablement_changed_since_preview",
                details={"policy_status": policy_status.redacted_dict()},
            )
        containment_root = Path.home().expanduser().resolve()
    if not is_primary and (
        not fixture_root.exists()
        or not _is_contained(state_root, fixture_root)
        or not _is_contained(receipt_root, fixture_root)
        or not _is_contained(backup_root, fixture_root)
    ):
        return _uninstall_result(
            ok=False,
            mutated=False,
            status="blocked_path_escape",
            message="I blocked uninstall because state, receipt, or backup roots were outside the approved fixture root.",
            rollback_hint="No rollback needed because nothing changed.",
            error_code="uninstall_path_escape",
        )

    removable_roots = []
    for raw in snapshot.get("removable_roots") if isinstance(snapshot.get("removable_roots"), list) else []:
        try:
            root = Path(str(raw)).expanduser().resolve()
        except OSError:
            continue
        if is_primary:
            allowed_primary_roots = {
                runtime_root,
                (Path.home() / ".config/systemd/user").expanduser().resolve(),
                (Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "applications").expanduser().resolve(),
                (Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "icons").expanduser().resolve(),
                (state_root / "bin").resolve(),
            }
            if root in allowed_primary_roots or any(_is_contained(root, allowed) for allowed in allowed_primary_roots):
                removable_roots.append(root)
        elif _is_contained(root, fixture_root):
            removable_roots.append(root)
    if not removable_roots:
        return _uninstall_result(
            ok=False,
            mutated=False,
            status="blocked_no_removable_roots",
            message="I blocked uninstall because no approved removable roots were listed.",
            rollback_hint="No rollback needed because nothing changed.",
            error_code="uninstall_no_removable_roots",
        )

    resources = snapshot.get("removable_resources") if isinstance(snapshot.get("removable_resources"), list) else []
    preserved_roots: list[Path] = []
    for resource in snapshot.get("preserved_resources") if isinstance(snapshot.get("preserved_resources"), list) else []:
        if not isinstance(resource, dict):
            continue
        raw = str(resource.get("path") or "")
        resource_id = str(resource.get("id") or "")
        resource_class = str(resource.get("class") or "").lower()
        if not raw or raw.startswith("~/.cache") or resource_id in {"state-root", "state"} or resource_class == "state root":
            continue
        try:
            preserved_roots.append(Path(raw).expanduser().resolve())
        except OSError:
            continue
    validated_paths: list[str] = []
    for resource in resources:
        ok, reason, resolved = _validate_uninstall_resource(
            resource,
            containment_root=containment_root,
            removable_roots=removable_roots,
            preserved_roots=preserved_roots,
        )
        if not ok:
            return _uninstall_result(
                ok=False,
                mutated=False,
                status="blocked_invalid_resource",
                message=f"I blocked uninstall because a removable resource failed safety validation: {reason}.",
                rollback_hint="No rollback needed because nothing changed.",
                error_code=reason,
                details={"resource": support_bundle_redact(resource), "path": str(resolved) if resolved else None},
            )
        if resolved is not None:
            validated_paths.append(str(resolved))

    lock_path = state_root / "lifecycle_locks" / "uninstall.lock"
    receipt_path = receipt_root / f"personal-agent-uninstall-{operation_id}.json"
    lock_payload = {
        "operation_type": "operator.uninstall",
        "operation_id": operation_id,
        "plan_id": plan.get("plan_id"),
        "started_at": utc_now_iso(),
        "stage": "starting",
        "target_snapshot_hash": actual_hash,
    }
    if not _acquire_operation_lock(lock_path, lock_payload):
        return _uninstall_result(
            ok=False,
            mutated=False,
            status="blocked_lock_conflict",
            message="Another lifecycle operation already holds the uninstall lock. I did not start a second uninstall.",
            rollback_hint="No rollback needed because nothing changed.",
            error_code="uninstall_lock_conflict",
        )

    touched = [str(lock_path), str(receipt_path)]
    try:
        final_backup = _create_uninstall_final_backup(
            backup_root=backup_root,
            operation_id=operation_id,
            snapshot=snapshot,
            preserved_resources=snapshot.get("preserved_resources") if isinstance(snapshot.get("preserved_resources"), list) else [],
        )
        manifest = final_backup / "manifest.json"
        manifest_payload = _read_json(manifest)
        if manifest_payload.get("backup_schema_version") != BACKUP_SCHEMA_VERSION:
            raise ValueError("uninstall_final_backup_invalid")
        touched.append(str(final_backup))
        state_dir = state_root / "uninstall_operations" / operation_id
        operation_state = {
            "schema_version": HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION,
            "runner_version": HOST_LIFECYCLE_RUNNER_VERSION,
            "operation_id": operation_id,
            "operation_type": "uninstall",
            "plan_id": plan.get("plan_id"),
            "created_at": utc_now_iso(),
            "current_stage": "created",
            "fixture_mode": "primary_uninstall" if is_primary else ("primary_uninstall_shaped_proof" if is_production_shaped else "strict"),
            "proof_marker_path": snapshot.get("proof_marker_path") or action.get("proof_marker_path"),
            "state_root": str(state_root),
            "operation_state_path": str(state_dir / "state.json"),
            "receipt_path": str(receipt_path),
            "final_backup_path": str(final_backup),
            "target_snapshot_hash": actual_hash,
            "target_snapshot": snapshot,
            "service_names": action.get("service_names") if isinstance(action.get("service_names"), list) else snapshot.get("service_names"),
            "force_failure_after_resource_id": action.get("force_failure_after_resource_id"),
            "capability_policy": _capability_policy_record(action, plan),
        }
        if is_primary:
            policy_summary = {
                "schema_version": policy_status.schema_version,
                "capability": "primary_preserve_data_uninstall",
                "mode": "preserve_data",
                "unsupported_modes": ["purge"],
                "expires_at": policy_status.expires_at,
                "marker_fingerprint": policy_status.fingerprint,
            }
            operation_state["primary_uninstall_policy"] = policy_summary
        operation_state = attach_approved_hash(operation_state)
        touched.append(str(state_dir))
        helper_state = state_dir / "helper_state.json"
        _write_json_unredacted(helper_state, operation_state)
        if is_primary:
            consumed = consume_primary_uninstall_marker(build_policy_context(state_root=expected_state))
            if not consumed.get("consumed"):
                raise RuntimeError(f"primary_uninstall_marker_consume_failed:{consumed.get('reason')}")
            operation_state["primary_uninstall_policy_consumed"] = consumed
            operation_state = attach_approved_hash(operation_state)
            _write_json_unredacted(helper_state, operation_state)
            handoff = _launch_host_lifecycle_runner_systemd("uninstall", helper_state, operation_id=operation_id, timeout=20)
            if not handoff.get("ok"):
                raise RuntimeError(f"host_lifecycle_uninstall_handoff_failed:{handoff.get('returncode')}")
            _update_state_write(state_dir, "in_progress", {**operation_state, "handoff": handoff})
            message = (
                "Uninstall has started.\n\n"
                "I created and verified a final safety backup and handed the removal to the trusted host runner.\n\n"
                "Your data will be preserved. This chat will disconnect shortly."
            )
            return _uninstall_result(
                ok=True,
                mutated=True,
                status="in_progress",
                message=message,
                resources=touched,
                rollback_available=False,
                rollback_hint="Uninstall is not automatically reversible. Reinstall Personal Agent, then restore supported state from the final uninstall backup.",
                details={
                    "operation_id": operation_id,
                    "target_snapshot_hash": actual_hash,
                    "final_backup_path": str(final_backup),
                    "receipt_path": str(receipt_path),
                    "handoff": support_bundle_redact(handoff),
                    "preserved_resource_count": len(snapshot.get("preserved_resources") if isinstance(snapshot.get("preserved_resources"), list) else []),
                },
            )
        receipt = _launch_uninstall_helper_state_file(helper_state)
        status = str(receipt.get("status") or "verification_incomplete")
        completed = status == "completed_verified"
        _update_state_write(state_dir, status, {**operation_state, "receipt": receipt})
        if completed:
            message = (
                "Uninstall completed and verified in the isolated fixture. "
                "The application runtime and generated service files were removed; user data was preserved."
            )
            error_code = None
        else:
            message = (
                "Uninstall only partially completed in the isolated fixture. "
                "User data and the final backup were preserved, and the receipt lists what remains."
            )
            error_code = "uninstall_partial"
        return _uninstall_result(
            ok=completed,
            mutated=True,
            status=status,
            message=message,
            resources=[*touched, *validated_paths],
            rollback_available=False,
            rollback_hint="Uninstall is not automatically reversible. Reinstall Personal Agent, then restore supported state from the final uninstall backup.",
            error_code=error_code,
            details={
                "operation_id": operation_id,
                "target_snapshot_hash": actual_hash,
                "final_backup_path": str(final_backup),
                "receipt_path": str(receipt_path),
                "removed_count": len(receipt.get("removed_resources") if isinstance(receipt.get("removed_resources"), list) else []),
                "preserved_count": len(receipt.get("preserved_resources") if isinstance(receipt.get("preserved_resources"), list) else []),
                "receipt": receipt,
            },
        )
    except Exception as exc:  # noqa: BLE001 - executor boundary must stay structured.
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        if not receipt_path.exists():
            _write_json(
                receipt_path,
                {
                    "uninstall_receipt_schema_version": UNINSTALL_RECEIPT_SCHEMA_VERSION,
                    "operation_id": operation_id,
                    "status": "failed_before_runtime_removal",
                    "error": exc.__class__.__name__,
                    "finished_at": utc_now_iso(),
                    "preserved_resources": snapshot.get("preserved_resources") if isinstance(snapshot.get("preserved_resources"), list) else [],
                },
            )
        return _uninstall_result(
            ok=False,
            mutated=False,
            status="failed_before_runtime_removal",
            message="Uninstall failed before runtime removal, so the installation was left in place and user data was preserved.",
            resources=touched,
            rollback_available=False,
            rollback_hint="No rollback needed because destructive removal did not complete.",
            error_code="uninstall_executor_exception",
            details={"exception": exc.__class__.__name__, "receipt_path": str(receipt_path)},
        )
    finally:
        _release_operation_lock(lock_path, operation_id)


def build_backup_manifest(
    *,
    root: Path,
    diagnostics: dict[str, Any],
    included_files: list[str],
    excluded_files: list[str],
    file_sizes: dict[str, int] | None = None,
    total_size_bytes: int | None = None,
) -> dict[str, Any]:
    version = diagnostics.get("version") if isinstance(diagnostics.get("version"), dict) else {}
    policy = (
        "Backup v1 stores redacted JSON summaries only. Raw secret-store files, "
        "tokens, API keys, passwords, raw logs, model caches, arbitrary home data, "
        "and unreviewed pack/source contents are excluded. No encryption is applied "
        "because raw secret material is not included; treat the backup as local-sensitive."
    )
    return support_bundle_redact(
        {
            "backup_schema_version": BACKUP_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "runtime_commit": version.get("git_commit"),
            "runtime_instance": version.get("runtime_instance"),
            "included_files": included_files,
            "excluded_files": excluded_files,
            "file_sizes": file_sizes or {},
            "total_size_bytes": total_size_bytes,
            "size_caps": {
                "max_total_bytes": BACKUP_MAX_TOTAL_BYTES,
                "max_file_bytes": BACKUP_MAX_FILE_BYTES,
                "max_journal_entries": BACKUP_MAX_JOURNAL_ENTRIES,
            },
            "redaction/encryption policy": policy,
            "redaction_encryption_policy": policy,
            "restore_status": RESTORE_V1_CAPABILITY,
            "live_restore": RESTORE_V1_CAPABILITY,
            "backup_path": str(root),
        }
    )


def create_additive_backup(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    context_failure = _trusted_context_failure(
        plan,
        action,
        capability_id="backup.create",
        executor_id="operator.backup.v1",
    )
    if context_failure is not None:
        return context_failure
    plan_id = str(plan.get("plan_id") or action.get("pending_id") or "unknown").strip() or "unknown"
    backup_root = _approved_backup_root(action)
    root = _artifact_dir(backup_root, prefix="personal-agent-backup")
    root.mkdir(mode=0o700, parents=False, exist_ok=False)

    diagnostics = action.get("diagnostics") if isinstance(action.get("diagnostics"), dict) else {}
    executor_recent = action.get("executor_journal_recent") if isinstance(action.get("executor_journal_recent"), list) else []
    backup_sources = action.get("backup_sources") if isinstance(action.get("backup_sources"), dict) else {}
    version = diagnostics.get("version") if isinstance(diagnostics.get("version"), dict) else {}
    ready = diagnostics.get("ready") if isinstance(diagnostics.get("ready"), dict) else {}
    state = diagnostics.get("state") if isinstance(diagnostics.get("state"), dict) else {}
    search = diagnostics.get("search_status") if isinstance(diagnostics.get("search_status"), dict) else {}
    telegram = diagnostics.get("telegram_status") if isinstance(diagnostics.get("telegram_status"), dict) else {}
    packs = diagnostics.get("packs_state") if isinstance(diagnostics.get("packs_state"), dict) else {}
    doctor = diagnostics.get("doctor") if isinstance(diagnostics.get("doctor"), dict) else {}
    git = diagnostics.get("git") if isinstance(diagnostics.get("git"), dict) else {}

    excluded_files = [
        "raw secret-store files",
        "raw credentials such as Telegram/provider/API authentication values and passwords",
        "raw logs and full support bundles",
        "arbitrary home directory files",
        "model caches/downloads and GGUF/model artifacts",
        "raw external pack archives, SKILL.md, AGENTS.md, and untrusted source text",
        "browser caches, downloaded pages, and search result page contents",
        "unsupported restore payloads; Restore v1 only applies allowlisted non-secret preferences",
    ]
    files: dict[str, dict[str, Any]] = {}
    files["state_database_summary.json"] = {
        "source": "runtime state database summary",
        "mode": "summary_only_raw_database_excluded",
        "state_database": backup_sources.get("state_database"),
    }
    preference_source = backup_sources.get("preferences") if isinstance(backup_sources.get("preferences"), dict) else {}
    preference_items = preference_source.get("preferences") if isinstance(preference_source.get("preferences"), list) else []
    files["preferences_summary.json"] = {
        "source": "preferences summary",
        "mode": "allowlisted_restore_export",
        "preferences": preference_items,
        "restore_supported_keys": preference_source.get("restore_supported_keys", sorted(RESTORE_ALLOWED_PREFERENCE_KEYS)),
        "excluded": preference_source.get("excluded", "All non-allowlisted preferences are excluded from Restore v1."),
    }
    files["memory_anchors_summary.json"] = {
        "source": "memory/anchors summary",
        "mode": "summary_only_raw_memory_text_excluded",
        "memory": backup_sources.get("memory", {"status": "summary_only"}),
    }
    pack_counts = packs.get("counts") if isinstance(packs.get("counts"), dict) else {}
    files["pack_metadata_summary.json"] = {
        "source": "pack metadata summary",
        "ok": packs.get("ok", True),
        "counts": pack_counts,
        "state": packs.get("state"),
        "warnings": packs.get("warnings") if isinstance(packs.get("warnings"), list) else [],
        "raw_pack_text": "excluded",
    }
    files["runtime_config_summary.json"] = {
        "version": version,
        "ready": _status_summary(
            ready,
            ("ready", "phase", "startup_phase", "runtime_mode", "failure_code", "next_action", "state_label", "reason", "blocker", "next_step"),
        ),
        "state": _status_summary(state, ("ok", "ready", "runtime_mode", "state_label", "reason", "next_action", "search", "telegram", "packs", "memory")),
        "search_status": _status_summary(
            search,
            ("ok", "enabled", "provider", "endpoint_configured", "available", "reason", "next_action", "search_state", "base_url", "managed_service"),
        ),
        "telegram_status": _status_summary(
            telegram,
            (
                "ok",
                "enabled",
                "configured",
                "token_source",
                "state",
                "effective_state",
                "service_installed",
                "service_active",
                "service_enabled",
                "lock_present",
                "lock_live",
                "lock_stale",
                "next_action",
            ),
        ),
    }
    journal_entries = _summarize_executor_journal_entries(executor_recent)
    files["executor_registry_journal_summary.json"] = {
        "entries": journal_entries,
        "entry_count": len(executor_recent),
        "included_entry_count": len(journal_entries),
        "source": "executor_registry_journal_recent_summary_only",
    }
    files["diagnostics_summary.json"] = {
        "doctor": doctor or _status_summary(ready, ("ready", "runtime_mode", "state_label", "reason", "next_action")),
        "git_runtime_freshness": {
            "runtime_commit": version.get("git_commit"),
            "checkout_commit": diagnostics.get("checkout_commit"),
            "runtime_instance": version.get("runtime_instance"),
            "git": git,
        },
        "readiness_proof": diagnostics.get("readiness_proof"),
        "docs_truth": diagnostics.get("docs_truth"),
    }
    files["support_bundle_style_summary.json"] = {
        "source": "support-bundle-style redacted summaries",
        "included": sorted(files.keys()),
        "redaction": "same redaction helper as Support Bundle v2",
    }
    files["backup_summary.json"] = {
        "created_at": utc_now_iso(),
        "plan_id": plan_id,
        "action_type": str(plan.get("action_type") or "operator.backup"),
        "target": str(plan.get("target") or "backup assistant"),
        "backup_schema_version": BACKUP_SCHEMA_VERSION,
        "restore_status": RESTORE_V1_CAPABILITY,
        "live_restore": RESTORE_V1_CAPABILITY,
        "contents": sorted(files.keys()),
        "size_caps": {
            "max_total_bytes": BACKUP_MAX_TOTAL_BYTES,
            "max_file_bytes": BACKUP_MAX_FILE_BYTES,
            "max_journal_entries": BACKUP_MAX_JOURNAL_ENTRIES,
        },
    }

    included_files = sorted([*files.keys(), "manifest.json"])
    file_sizes: dict[str, int] = {}
    try:
        total_size = 0
        for name, payload in files.items():
            size = _write_backup_json(root / name, payload)
            file_sizes[name] = size
            total_size += size
            if total_size > BACKUP_MAX_TOTAL_BYTES:
                raise ValueError(f"backup_total_size_cap_exceeded:{total_size}")
        manifest = build_backup_manifest(
            root=root,
            diagnostics=diagnostics,
            included_files=included_files,
            excluded_files=excluded_files,
            file_sizes=file_sizes,
            total_size_bytes=total_size,
        )
        manifest_text = json.dumps(support_bundle_redact(_bounded_backup_value(manifest)), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        manifest_size = len(manifest_text.encode("utf-8"))
        if manifest_size > BACKUP_MAX_FILE_BYTES:
            raise ValueError(f"backup_file_size_cap_exceeded:manifest.json:{manifest_size}")
        if total_size + manifest_size > BACKUP_MAX_TOTAL_BYTES:
            raise ValueError(f"backup_total_size_cap_exceeded:{total_size + manifest_size}")
        (root / "manifest.json").write_text(manifest_text, encoding="utf-8")
        file_sizes["manifest.json"] = manifest_size
        total_size += manifest_size
    except Exception as exc:
        resources = [str(path) for path in sorted(root.glob("*.json"))]
        return {
            "ok": False,
            "mutated": False,
            "executor_id": "operator.backup.v1",
            "resources_touched": resources,
            "rollback_available": True,
            "rollback_hint": f"Remove only the partial backup directory created for this failed action: {root}",
            "error_code": "backup_v1_failed_before_final_manifest",
            "user_message": "Backup v1 did not finish. I did not write a final manifest or verify a usable backup.",
            "details": {"artifact_path": str(root), "partial": True, "error": exc.__class__.__name__},
        }
    resources = [str(root / name) for name in included_files]
    return {
        "ok": True,
        "mutated": True,
        "executor_id": "operator.backup.v1",
        "resources_touched": resources,
        "rollback_available": True,
        "rollback_hint": f"Remove only the newly created backup directory: {root}",
        "user_message": (
            f"Backup v1 created at {root}. It contains redacted summaries only. "
            "Restore v1 can apply only allowlisted non-secret preferences."
        ),
        "details": {"artifact_path": str(root), "files": included_files, "manifest_path": str(root / "manifest.json")},
    }


def create_redacted_support_bundle(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    context_failure = _trusted_context_failure(
        plan,
        action,
        capability_id="support_bundle.create",
        executor_id="operator.support_bundle.v1",
    )
    if context_failure is not None:
        return context_failure
    plan_id = str(plan.get("plan_id") or action.get("pending_id") or "unknown").strip() or "unknown"
    root = Path(tempfile.mkdtemp(prefix="personal-agent-support-"))
    diagnostics = action.get("diagnostics") if isinstance(action.get("diagnostics"), dict) else {}
    executor_recent = action.get("executor_journal_recent") if isinstance(action.get("executor_journal_recent"), list) else []
    files: dict[str, dict[str, Any]] = {}
    version = diagnostics.get("version") if isinstance(diagnostics.get("version"), dict) else {}
    ready = diagnostics.get("ready") if isinstance(diagnostics.get("ready"), dict) else {}
    state = diagnostics.get("state") if isinstance(diagnostics.get("state"), dict) else {}
    search = diagnostics.get("search_status") if isinstance(diagnostics.get("search_status"), dict) else {}
    telegram = diagnostics.get("telegram_status") if isinstance(diagnostics.get("telegram_status"), dict) else {}
    packs = diagnostics.get("packs_state") if isinstance(diagnostics.get("packs_state"), dict) else {}
    doctor = diagnostics.get("doctor") if isinstance(diagnostics.get("doctor"), dict) else {}
    proof = diagnostics.get("readiness_proof") if isinstance(diagnostics.get("readiness_proof"), dict) else {}
    docs_truth = diagnostics.get("docs_truth") if isinstance(diagnostics.get("docs_truth"), dict) else {}
    git = diagnostics.get("git") if isinstance(diagnostics.get("git"), dict) else {}

    files["doctor_summary.json"] = {
        "source": "doctor/runtime summary",
        "summary": doctor or _status_summary(ready, ("ready", "runtime_mode", "state_label", "reason", "next_action")),
    }
    files["version.json"] = version
    files["ready.json"] = _status_summary(
        ready,
        ("ready", "phase", "startup_phase", "runtime_mode", "failure_code", "next_action", "state_label", "reason", "blocker", "next_step", "message"),
    )
    files["state_summary.json"] = _status_summary(
        state,
        ("ok", "ready", "runtime_mode", "state_label", "reason", "next_action", "search", "telegram", "packs", "memory"),
    )
    files["search_status.json"] = _status_summary(
        search,
        ("ok", "enabled", "provider", "endpoint_configured", "available", "reason", "next_action", "search_state", "base_url", "managed_service"),
    )
    files["telegram_status.json"] = _status_summary(
        telegram,
        (
            "ok",
            "enabled",
            "configured",
            "token_source",
            "state",
            "effective_state",
            "service_installed",
            "service_active",
            "service_enabled",
            "lock_present",
            "lock_live",
            "lock_stale",
            "next_action",
        ),
    )
    pack_counts = packs.get("counts") if isinstance(packs.get("counts"), dict) else {}
    files["packs_state_summary.json"] = {
        "ok": packs.get("ok", True),
        "counts": pack_counts,
        "state": packs.get("state"),
        "warnings": packs.get("warnings") if isinstance(packs.get("warnings"), list) else [],
    }
    files["executor_registry_journal_summary.json"] = {
        "entries": _summarize_executor_journal_entries(executor_recent),
        "entry_count": len(executor_recent),
        "included_entry_count": min(len(executor_recent), BACKUP_MAX_JOURNAL_ENTRIES),
        "source": "executor_registry_journal_recent_summary_only",
    }
    files["readiness_proof_summary.json"] = {
        "prove_ready": proof,
        "docs_truth": docs_truth,
    }
    files["git_runtime_freshness.json"] = {
        "runtime_commit": version.get("git_commit"),
        "checkout_commit": diagnostics.get("checkout_commit"),
        "runtime_instance": version.get("runtime_instance"),
        "git": git,
    }
    files["support_summary.json"] = {
        "created_at": utc_now_iso(),
        "plan_id": plan_id,
        "action_type": str(plan.get("action_type") or "operator.support_bundle"),
        "target": str(plan.get("target") or "support bundle"),
        "bundle_schema_version": SUPPORT_BUNDLE_SCHEMA_VERSION,
        "redaction": "secrets, tokens, API keys, raw private values, raw logs, and broad private paths are redacted or summarized",
        "contents": sorted(files.keys()),
    }

    included_files = sorted([*files.keys(), "manifest.json"])
    file_sizes: dict[str, int] = {}
    try:
        total_size = 0
        for name, payload in files.items():
            size = _write_support_json(root / name, payload)
            file_sizes[name] = size
            total_size += size
            if total_size > SUPPORT_BUNDLE_MAX_TOTAL_BYTES:
                raise ValueError(f"support_bundle_total_size_cap_exceeded:{total_size}")
        manifest = build_support_bundle_manifest(root=root, diagnostics=diagnostics, included_files=included_files)
        manifest["file_sizes"] = file_sizes
        manifest["total_size_bytes"] = total_size
        manifest["size_caps"] = {
            "max_total_bytes": SUPPORT_BUNDLE_MAX_TOTAL_BYTES,
            "max_file_bytes": SUPPORT_BUNDLE_MAX_FILE_BYTES,
            "max_journal_entries": BACKUP_MAX_JOURNAL_ENTRIES,
        }
        manifest_text = json.dumps(support_bundle_redact(_bounded_backup_value(manifest)), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        manifest_size = len(manifest_text.encode("utf-8"))
        if manifest_size > SUPPORT_BUNDLE_MAX_FILE_BYTES:
            raise ValueError(f"support_bundle_file_size_cap_exceeded:manifest.json:{manifest_size}")
        if total_size + manifest_size > SUPPORT_BUNDLE_MAX_TOTAL_BYTES:
            raise ValueError(f"support_bundle_total_size_cap_exceeded:{total_size + manifest_size}")
        (root / "manifest.json").write_text(manifest_text, encoding="utf-8")
        file_sizes["manifest.json"] = manifest_size
        total_size += manifest_size
    except Exception as exc:
        resources = [str(path) for path in sorted(root.glob("*.json"))]
        return {
            "ok": False,
            "mutated": False,
            "executor_id": "operator.support_bundle.v1",
            "resources_touched": resources,
            "rollback_available": True,
            "rollback_hint": f"Remove only the partial support bundle directory created for this failed action: {root}",
            "error_code": "support_bundle_v2_failed_before_final_manifest",
            "user_message": "Support bundle creation did not finish. I did not write a final manifest or verify a usable bundle.",
            "details": {"artifact_path": str(root), "partial": True, "error": exc.__class__.__name__},
        }
    resources = [str(root / name) for name in included_files]
    return {
        "ok": True,
        "mutated": True,
        "executor_id": "operator.support_bundle.v1",
        "resources_touched": resources,
        "rollback_available": True,
        "rollback_hint": f"Remove only the newly created support bundle directory: {root}",
        "user_message": f"Support bundle created at {root}. It contains redacted diagnostics only.",
        "details": {"artifact_path": str(root), "files": included_files, "manifest_path": str(root / "manifest.json")},
    }


_BLOCKED_FILE_ROOTS = tuple(Path(path) for path in ("/proc", "/sys", "/dev", "/run"))
_MAX_FILE_EXECUTOR_BYTES = 256 * 1024


def _path_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _approved_file_target(action: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    raw = str(action.get("target_path") or "").strip()
    if not raw:
        return None, {"error_code": "file_target_missing", "message": "No target file path was provided."}
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        base = Path(str(action.get("base_dir") or Path.cwd())).expanduser().resolve(strict=False)
        candidate = base / candidate
    if candidate.is_symlink():
        return None, {"error_code": "file_symlink_target_blocked", "message": "Symlink file targets are blocked."}
    for parent_candidate in candidate.parents:
        if parent_candidate.is_symlink():
            return None, {"error_code": "file_symlink_parent_blocked", "message": "Symlink parent directories are blocked."}
        if parent_candidate.exists():
            break
    resolved = candidate.resolve(strict=False)
    if any(resolved == root or root in resolved.parents for root in _BLOCKED_FILE_ROOTS):
        return None, {"error_code": "file_pseudo_filesystem_blocked", "message": "Pseudo-filesystem file mutation is blocked."}
    roots = [Path(str(item)).expanduser().resolve(strict=False) for item in action.get("approved_roots", []) if str(item).strip()] if isinstance(action.get("approved_roots"), list) else []
    if not roots:
        roots = [Path(tempfile.gettempdir()).resolve()]
    if not any(_is_under(resolved, root) or resolved == root for root in roots):
        return None, {"error_code": "file_target_outside_approved_roots", "message": "The target is outside approved file mutation roots."}
    parent = resolved.parent
    if resolved.exists() and not resolved.is_file():
        return None, {"error_code": "file_target_not_regular_file", "message": "Only regular file targets are supported."}
    return resolved, None


def _file_snapshot(path: Path) -> dict[str, Any]:
    exists = path.exists()
    stat_result = path.stat() if exists else None
    return {
        "canonical_path": str(path),
        "resource_type": "file" if path.is_file() else ("missing" if not exists else "other"),
        "exists": exists,
        "owner": int(stat_result.st_uid) if stat_result is not None else None,
        "mode": oct(stat_result.st_mode & 0o777) if stat_result is not None else None,
        "size": int(stat_result.st_size) if stat_result is not None else 0,
        "mtime": int(stat_result.st_mtime_ns) if stat_result is not None else None,
        "content_hash": _path_digest(path) if exists and path.is_file() and int(stat_result.st_size) <= _MAX_FILE_EXECUTOR_BYTES else None,
        "parent_root": str(path.parent),
        "protected": False,
        "symlink_status": "symlink" if path.is_symlink() else "not_symlink",
        "mount_status": "mount_point" if path.exists() and path.is_mount() else "not_mount_point",
    }


def execute_file_create(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    context_failure = _trusted_context_failure(plan, action, capability_id="files.create", executor_id="operator.file.create.v1")
    if context_failure is not None:
        return context_failure
    target, error = _approved_file_target(action)
    if error is not None:
        return {"ok": False, "mutated": False, "executor_id": "operator.file.create.v1", **error}
    assert target is not None
    if target.exists() and not bool(action.get("overwrite")):
        return {"ok": False, "mutated": False, "executor_id": "operator.file.create.v1", "error_code": "file_exists_overwrite_not_allowed", "user_message": "File creation was blocked because the target already exists."}
    content = str(action.get("content") or "")
    if len(content.encode("utf-8")) > _MAX_FILE_EXECUTOR_BYTES:
        return {"ok": False, "mutated": False, "executor_id": "operator.file.create.v1", "error_code": "file_content_too_large", "user_message": "File content exceeded the bounded write size."}
    before = _file_snapshot(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    rollback_path = None
    if target.exists():
        rollback_path = target.with_name(f".{target.name}.rollback-{uuid.uuid4().hex[:8]}")
        shutil.copy2(target, rollback_path)
    temp = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex[:8]}")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, target)
    after = _file_snapshot(target)
    return {
        "ok": True,
        "mutated": True,
        "executor_id": "operator.file.create.v1",
        "resources_touched": [str(target), *([str(rollback_path)] if rollback_path else [])],
        "rollback_available": rollback_path is not None,
        "rollback_hint": f"Restore rollback copy at {rollback_path}." if rollback_path else f"Remove newly created file {target}.",
        "user_message": f"Created file {target}.",
        "details": {"before": before, "after": after, "rollback_path": str(rollback_path) if rollback_path else None},
    }


def execute_file_modify(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    context_failure = _trusted_context_failure(plan, action, capability_id="files.modify", executor_id="operator.file.modify.v1")
    if context_failure is not None:
        return context_failure
    target, error = _approved_file_target(action)
    if error is not None:
        return {"ok": False, "mutated": False, "executor_id": "operator.file.modify.v1", **error}
    assert target is not None
    if not target.is_file():
        return {"ok": False, "mutated": False, "executor_id": "operator.file.modify.v1", "error_code": "file_modify_target_missing", "user_message": "File modification requires an existing regular file."}
    expected_hash = str(action.get("expected_hash") or "").strip()
    current_hash = _path_digest(target) or ""
    if expected_hash and expected_hash != current_hash:
        return {"ok": False, "mutated": False, "executor_id": "operator.file.modify.v1", "error_code": "file_changed_since_preview", "user_message": "The file changed after preview. Ask for a fresh Plan.", "details": {"current_hash": current_hash}}
    content = str(action.get("content") or "")
    if len(content.encode("utf-8")) > _MAX_FILE_EXECUTOR_BYTES:
        return {"ok": False, "mutated": False, "executor_id": "operator.file.modify.v1", "error_code": "file_content_too_large", "user_message": "File content exceeded the bounded write size."}
    before = _file_snapshot(target)
    rollback_path = target.with_name(f".{target.name}.rollback-{uuid.uuid4().hex[:8]}")
    shutil.copy2(target, rollback_path)
    temp = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex[:8]}")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, target)
    after = _file_snapshot(target)
    return {
        "ok": True,
        "mutated": True,
        "executor_id": "operator.file.modify.v1",
        "resources_touched": [str(target), str(rollback_path)],
        "rollback_available": True,
        "rollback_hint": f"Restore rollback copy at {rollback_path}.",
        "user_message": f"Modified file {target}.",
        "details": {"before": before, "after": after, "rollback_path": str(rollback_path)},
    }


def execute_file_delete(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    context_failure = _trusted_context_failure(plan, action, capability_id="files.delete", executor_id="operator.file.delete.v1")
    if context_failure is not None:
        return context_failure
    target, error = _approved_file_target(action)
    if error is not None:
        return {"ok": False, "mutated": False, "executor_id": "operator.file.delete.v1", **error}
    assert target is not None
    if not target.is_file():
        return {"ok": False, "mutated": False, "executor_id": "operator.file.delete.v1", "error_code": "file_delete_target_missing", "user_message": "File deletion requires an existing regular file."}
    expected_hash = str(action.get("expected_hash") or "").strip()
    current_hash = _path_digest(target) or ""
    if expected_hash and expected_hash != current_hash:
        return {"ok": False, "mutated": False, "executor_id": "operator.file.delete.v1", "error_code": "file_changed_since_preview", "user_message": "The file changed after preview. Ask for a fresh Plan.", "details": {"current_hash": current_hash}}
    staging_root = Path(str(action.get("delete_staging_root") or target.parent / ".personal-agent-deleted")).expanduser().resolve(strict=False)
    if not (_is_under(staging_root, target.parent) or staging_root == target.parent):
        return {"ok": False, "mutated": False, "executor_id": "operator.file.delete.v1", "error_code": "delete_staging_outside_parent", "user_message": "Delete staging must stay under the target parent."}
    before = _file_snapshot(target)
    staging_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    staged = staging_root / f"{target.name}.{uuid.uuid4().hex[:8]}"
    os.replace(target, staged)
    return {
        "ok": True,
        "mutated": True,
        "executor_id": "operator.file.delete.v1",
        "resources_touched": [str(target), str(staged)],
        "rollback_available": True,
        "rollback_hint": f"Move {staged} back to {target}.",
        "user_message": f"Moved deleted file {target} to recoverable staging.",
        "details": {"before": before, "staged_path": str(staged), "permanent_delete": False},
    }


def _run_git(repo: Path, args: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["/usr/bin/git", *args], cwd=str(repo), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def _git_repository(action: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    raw = str(action.get("repository_root") or "").strip()
    if not raw:
        return None, {"error_code": "git_repository_missing", "user_message": "No Git repository root was provided."}
    repo = Path(raw).expanduser().resolve(strict=False)
    if repo.is_symlink() or not (repo / ".git").exists():
        return None, {"error_code": "git_repository_invalid", "user_message": "The Git target is not an approved repository root."}
    roots = [Path(str(item)).expanduser().resolve(strict=False) for item in action.get("approved_roots", []) if str(item).strip()] if isinstance(action.get("approved_roots"), list) else []
    if roots and not any(repo == root or _is_under(repo, root) for root in roots):
        return None, {"error_code": "git_repository_outside_approved_roots", "user_message": "The repository is outside approved roots."}
    return repo, None


def _git_fingerprint(repo: Path) -> dict[str, Any]:
    head = _run_git(repo, ["rev-parse", "HEAD"])
    branch = _run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    staged = _run_git(repo, ["diff", "--cached", "--name-status"])
    staged_patch = _run_git(repo, ["diff", "--cached", "--binary"])
    status = _run_git(repo, ["status", "--porcelain=v1"])
    return {
        "repository_root": str(repo),
        "head_commit": (head.stdout or "").strip(),
        "current_branch": (branch.stdout or "").strip(),
        "staged_files": (staged.stdout or "").strip().splitlines(),
        "staged_diff_sha256": hashlib.sha256((staged_patch.stdout or "").encode("utf-8", errors="replace")).hexdigest(),
        "working_tree_status": (status.stdout or "").strip().splitlines(),
    }


def execute_git_commit(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    context_failure = _trusted_context_failure(plan, action, capability_id="git.commit", executor_id="operator.git.commit.v1")
    if context_failure is not None:
        return context_failure
    repo, error = _git_repository(action)
    if error is not None:
        return {"ok": False, "mutated": False, "executor_id": "operator.git.commit.v1", **error}
    assert repo is not None
    before = _git_fingerprint(repo)
    expected = str(action.get("staged_diff_sha256") or "").strip()
    if expected and expected != before["staged_diff_sha256"]:
        return {"ok": False, "mutated": False, "executor_id": "operator.git.commit.v1", "error_code": "git_staged_diff_changed", "user_message": "The staged Git diff changed after preview. Ask for a fresh Plan.", "details": {"current": before}}
    if not before["staged_files"]:
        return {"ok": False, "mutated": False, "executor_id": "operator.git.commit.v1", "error_code": "git_nothing_staged", "user_message": "No staged changes were available to commit.", "details": {"current": before}}
    message = str(action.get("commit_message") or "").strip()
    if not message or "\x00" in message or len(message.encode("utf-8")) > 4096:
        return {"ok": False, "mutated": False, "executor_id": "operator.git.commit.v1", "error_code": "git_commit_message_invalid", "user_message": "The commit message was missing or invalid."}
    proc = _run_git(repo, ["commit", "-m", message], timeout=20.0)
    if proc.returncode != 0:
        return {"ok": False, "mutated": False, "executor_id": "operator.git.commit.v1", "error_code": "git_commit_failed", "user_message": "Git commit failed before verification.", "details": {"stderr": (proc.stderr or "")[-1000:]}}
    after = _git_fingerprint(repo)
    return {
        "ok": True,
        "mutated": True,
        "executor_id": "operator.git.commit.v1",
        "resources_touched": [str(repo)],
        "rollback_available": True,
        "rollback_hint": f"Previous HEAD was {before['head_commit']}; use a reviewed revert/reset plan if needed.",
        "user_message": f"Created Git commit {after['head_commit'][:12]} on {after['current_branch']}.",
        "details": {"before": before, "after": after, "commit_message_sha256": hashlib.sha256(message.encode('utf-8')).hexdigest()},
    }


def execute_git_push(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    context_failure = _trusted_context_failure(plan, action, capability_id="git.push", executor_id="operator.git.push.v1")
    if context_failure is not None:
        return context_failure
    if bool(action.get("force")):
        return {"ok": False, "mutated": False, "executor_id": "operator.git.push.v1", "error_code": "git_force_push_denied", "user_message": "Force push is denied by policy."}
    return {"ok": False, "mutated": False, "executor_id": "operator.git.push.v1", "error_code": "git_push_external_side_effect_not_enabled", "user_message": "Git push remains preview-only until an external remote proof is configured."}


def execute_service_restart(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    context_failure = _trusted_context_failure(plan, action, capability_id="system.service.restart", executor_id="operator.service.restart.v1")
    if context_failure is not None:
        return context_failure
    service = str(action.get("service_name") or "").strip()
    allowed = {str(item).strip() for item in action.get("allowed_services", []) if str(item).strip()} if isinstance(action.get("allowed_services"), list) else set()
    if not service or service not in allowed:
        return {"ok": False, "mutated": False, "executor_id": "operator.service.restart.v1", "error_code": "service_not_allowlisted", "user_message": "Service restart was blocked because the unit is not allowlisted."}
    if service == "personal-agent-api.service" and not bool(action.get("allow_primary_api_restart")):
        return {"ok": False, "mutated": False, "executor_id": "operator.service.restart.v1", "error_code": "service_protected", "user_message": "Primary API restart requires the dedicated restart/reconnect path."}
    fixture_root = str(action.get("service_fixture_root") or "").strip()
    if not fixture_root:
        return {"ok": False, "mutated": False, "executor_id": "operator.service.restart.v1", "error_code": "service_fixture_required", "user_message": "This proof executor only restarts fixture services."}
    root = Path(fixture_root).expanduser().resolve(strict=False)
    if root.is_symlink() or not (root == Path(tempfile.gettempdir()).resolve() or _is_under(root, Path(tempfile.gettempdir()).resolve())):
        return {"ok": False, "mutated": False, "executor_id": "operator.service.restart.v1", "error_code": "service_fixture_root_unapproved", "user_message": "Fixture service root is not approved."}
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / f"{service}.state.json"
    before = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"active_state": "inactive", "restart_count": 0}
    after = {
        "service_name": service,
        "active_state": "active",
        "restart_count": int(before.get("restart_count") or 0) + 1,
        "restarted_at": utc_now_iso(),
    }
    state_path.write_text(json.dumps(after, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "mutated": True,
        "executor_id": "operator.service.restart.v1",
        "resources_touched": [str(state_path)],
        "rollback_available": False,
        "rollback_hint": "Fixture restart has no rollback requirement.",
        "user_message": f"Restarted fixture service {service}.",
        "details": {"service_name": service, "before": before, "after": after, "command_class": "fixture_systemctl_user_restart"},
    }
