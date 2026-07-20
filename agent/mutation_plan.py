from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any

from agent.capability_policy import POLICY_SCHEMA_VERSION, build_default_capability_registry, stable_fingerprint, validate_capability_id


MUTATION_PLAN_SCHEMA_VERSION = 1
MUTATION_PLAN_STATUS_PENDING = "pending"
MUTATION_PLAN_STATUS_CONFIRMED = "confirmed"
MUTATION_PLAN_STATUS_EXECUTING = "executing"
MUTATION_PLAN_STATUS_COMPLETED = "completed"
MUTATION_PLAN_STATUS_FAILED = "failed"
MUTATION_PLAN_STATUS_CANCELLED = "cancelled"
MUTATION_PLAN_STATUS_EXPIRED = "expired"
MUTATION_PLAN_STATUS_INVALIDATED = "invalidated"

VALID_PLAN_STATUSES = {
    MUTATION_PLAN_STATUS_PENDING,
    MUTATION_PLAN_STATUS_CONFIRMED,
    MUTATION_PLAN_STATUS_EXECUTING,
    MUTATION_PLAN_STATUS_COMPLETED,
    MUTATION_PLAN_STATUS_FAILED,
    MUTATION_PLAN_STATUS_CANCELLED,
    MUTATION_PLAN_STATUS_EXPIRED,
    MUTATION_PLAN_STATUS_INVALIDATED,
}

MUTATION_PLAN_REASON_CODES = {
    "mutation_plan_missing",
    "mutation_plan_invalid",
    "mutation_plan_expired",
    "mutation_plan_cancelled",
    "mutation_plan_fingerprint_mismatch",
    "mutation_plan_target_changed",
    "mutation_plan_policy_changed",
    "mutation_plan_activation_changed",
    "mutation_plan_thread_mismatch",
    "mutation_plan_session_mismatch",
}

_EXECUTOR_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]+)*\.v[0-9]+$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_timestamp(value: str | int | float | None) -> str:
    if value is None or value == "":
        return utc_now_iso()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("mutation_plan_timestamp_invalid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def expires_at_epoch(plan: dict[str, Any]) -> int:
    raw = plan.get("expires_at")
    if isinstance(raw, (int, float)):
        return int(raw)
    try:
        parsed = datetime.fromisoformat(str(raw or "").replace("Z", "+00:00"))
        return int(parsed.timestamp())
    except ValueError:
        return 0


def normalize_path_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_path_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [normalize_path_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_path_value(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("~") or "/" in text:
            try:
                return str(Path(text).expanduser().resolve(strict=False))
            except OSError:
                return text
        return text
    return value


def canonical_plan_security_payload(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(plan.get("schema_version") or MUTATION_PLAN_SCHEMA_VERSION),
        "plan_id": str(plan.get("plan_id") or ""),
        "capability_id": str(plan.get("capability_id") or ""),
        "executor_id": str(plan.get("executor_id") or ""),
        "policy_version": int(plan.get("policy_version") or POLICY_SCHEMA_VERSION),
        "authorization_mode": str(plan.get("authorization_mode") or ""),
        "risk_level": str(plan.get("risk_level") or ""),
        "scope": str(plan.get("scope") or ""),
        "reversibility": str(plan.get("reversibility") or ""),
        "expires_at": normalize_timestamp(plan.get("expires_at")),
        "thread_id": str(plan.get("thread_id") or ""),
        "session_id": str(plan.get("session_id") or ""),
        "actor_id": str(plan.get("actor_id") or ""),
        "target_snapshot": normalize_path_value(plan.get("target_snapshot") if isinstance(plan.get("target_snapshot"), dict) else {}),
        "mutation_inventory": normalize_path_value(plan.get("mutation_inventory") if isinstance(plan.get("mutation_inventory"), list) else []),
        "preserved_resources": normalize_path_value(plan.get("preserved_resources") if isinstance(plan.get("preserved_resources"), list) else []),
        "expected_side_effects": normalize_path_value(plan.get("expected_side_effects") if isinstance(plan.get("expected_side_effects"), list) else []),
        "recovery": normalize_path_value(plan.get("recovery") if isinstance(plan.get("recovery"), dict) else {}),
        "activation_fingerprint": str(plan.get("activation_fingerprint") or ""),
        "confirmation_requirement": normalize_path_value(plan.get("confirmation_requirement") if isinstance(plan.get("confirmation_requirement"), dict) else {}),
        "receipt_required": bool(plan.get("receipt_required")),
        "runtime_revalidation_required": bool(plan.get("runtime_revalidation_required")),
    }


def mutation_plan_fingerprint(plan: dict[str, Any]) -> str:
    return stable_fingerprint(canonical_plan_security_payload(plan))


def target_fingerprint_for_snapshot(target_snapshot: dict[str, Any]) -> str:
    return stable_fingerprint(normalize_path_value(target_snapshot))


@dataclass(frozen=True)
class MutationPlan:
    plan_id: str
    capability_id: str
    executor_id: str
    authorization_mode: str
    risk_level: str
    scope: str
    reversibility: str
    created_at: str
    expires_at: str
    thread_id: str
    session_id: str
    actor_id: str
    target_snapshot: dict[str, Any]
    mutation_inventory: list[Any]
    preserved_resources: list[Any]
    expected_side_effects: list[Any]
    recovery: dict[str, Any]
    confirmation_requirement: dict[str, Any]
    receipt_required: bool
    runtime_revalidation_required: bool
    policy_version: int = POLICY_SCHEMA_VERSION
    schema_version: int = MUTATION_PLAN_SCHEMA_VERSION
    activation_fingerprint: str | None = None
    status: str = MUTATION_PLAN_STATUS_PENDING
    target_fingerprint: str = ""
    plan_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "plan_id": self.plan_id,
            "capability_id": self.capability_id,
            "executor_id": self.executor_id,
            "policy_version": int(self.policy_version),
            "authorization_mode": self.authorization_mode,
            "risk_level": self.risk_level,
            "scope": self.scope,
            "reversibility": self.reversibility,
            "created_at": normalize_timestamp(self.created_at),
            "expires_at": normalize_timestamp(self.expires_at),
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "actor_id": self.actor_id,
            "target_snapshot": normalize_path_value(self.target_snapshot),
            "target_fingerprint": self.target_fingerprint or target_fingerprint_for_snapshot(self.target_snapshot),
            "mutation_inventory": normalize_path_value(self.mutation_inventory),
            "preserved_resources": normalize_path_value(self.preserved_resources),
            "expected_side_effects": normalize_path_value(self.expected_side_effects),
            "recovery": normalize_path_value(self.recovery),
            "activation_fingerprint": self.activation_fingerprint,
            "confirmation_requirement": normalize_path_value(self.confirmation_requirement),
            "receipt_required": bool(self.receipt_required),
            "runtime_revalidation_required": bool(self.runtime_revalidation_required),
            "status": self.status,
        }
        payload["plan_fingerprint"] = self.plan_fingerprint or mutation_plan_fingerprint(payload)
        return payload

    def validate(self) -> None:
        validate_mutation_plan(self.to_dict())


def build_mutation_plan(
    *,
    plan_id: str,
    capability_id: str,
    executor_id: str,
    expires_at_epoch: int,
    thread_id: str = "",
    session_id: str = "",
    actor_id: str = "",
    target_snapshot: dict[str, Any] | None = None,
    mutation_inventory: list[Any] | None = None,
    preserved_resources: list[Any] | None = None,
    expected_side_effects: list[Any] | None = None,
    recovery: dict[str, Any] | None = None,
    activation_fingerprint: str | None = None,
) -> dict[str, Any]:
    definition = build_default_capability_registry().get(capability_id)
    if definition is None:
        raise ValueError("mutation_plan_unknown_capability")
    plan = MutationPlan(
        plan_id=str(plan_id or "").strip(),
        capability_id=definition.capability_id,
        executor_id=str(executor_id or "").strip(),
        authorization_mode=definition.authorization_mode,
        risk_level=definition.risk_level,
        scope=definition.scope,
        reversibility=definition.reversibility,
        created_at=utc_now_iso(),
        expires_at=normalize_timestamp(int(expires_at_epoch)),
        thread_id=str(thread_id or ""),
        session_id=str(session_id or ""),
        actor_id=str(actor_id or ""),
        target_snapshot=dict(target_snapshot or {}),
        mutation_inventory=list(mutation_inventory or []),
        preserved_resources=list(preserved_resources or []),
        expected_side_effects=list(expected_side_effects or []),
        recovery=dict(recovery or {}),
        activation_fingerprint=activation_fingerprint,
        confirmation_requirement={
            "required": definition.authorization_mode in {"plan_and_confirm", "local_activation_and_confirm"},
            "mode": definition.authorization_mode,
            "allowed_phrase_classes": ["affirmative"],
        },
        receipt_required=bool(definition.receipt_required),
        runtime_revalidation_required=bool(definition.runtime_revalidation_required),
    ).to_dict()
    validate_mutation_plan(plan)
    return plan


def validate_mutation_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise ValueError("mutation_plan_invalid")
    if int(plan.get("schema_version") or 0) != MUTATION_PLAN_SCHEMA_VERSION:
        raise ValueError("mutation_plan_schema_version_unsupported")
    validate_capability_id(str(plan.get("capability_id") or ""))
    executor_id = str(plan.get("executor_id") or "")
    if _EXECUTOR_ID_RE.fullmatch(executor_id) is None:
        raise ValueError("mutation_plan_executor_id_invalid")
    if str(plan.get("status") or MUTATION_PLAN_STATUS_PENDING) not in VALID_PLAN_STATUSES:
        raise ValueError("mutation_plan_status_invalid")
    if not str(plan.get("plan_id") or "").strip():
        raise ValueError("mutation_plan_id_required")
    normalize_timestamp(plan.get("created_at"))
    normalize_timestamp(plan.get("expires_at"))
    expected_target = target_fingerprint_for_snapshot(plan.get("target_snapshot") if isinstance(plan.get("target_snapshot"), dict) else {})
    if str(plan.get("target_fingerprint") or "") != expected_target:
        raise ValueError("mutation_plan_target_fingerprint_mismatch")
    expected_plan = mutation_plan_fingerprint(plan)
    if str(plan.get("plan_fingerprint") or "") != expected_plan:
        raise ValueError("mutation_plan_fingerprint_mismatch")


@dataclass(frozen=True)
class MutationConfirmation:
    confirmation_id: str
    plan_id: str
    plan_fingerprint: str
    capability_id: str
    executor_id: str
    thread_id: str
    session_id: str
    actor_id: str
    confirmed_at: str
    confirmation_phrase_class: str
    activation_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmation_id": self.confirmation_id,
            "plan_id": self.plan_id,
            "plan_fingerprint": self.plan_fingerprint,
            "capability_id": self.capability_id,
            "executor_id": self.executor_id,
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "actor_id": self.actor_id,
            "confirmed_at": normalize_timestamp(self.confirmed_at),
            "confirmation_phrase_class": self.confirmation_phrase_class,
            "activation_fingerprint": self.activation_fingerprint,
        }


def build_mutation_confirmation(
    plan: dict[str, Any],
    *,
    confirmation_id: str,
    actor_id: str | None = None,
    thread_id: str | None = None,
    session_id: str | None = None,
    confirmation_phrase_class: str = "affirmative",
) -> dict[str, Any]:
    """Build confirmation metadata bound to one exact Universal Mutation Plan.

    This is deliberately separate from the Plan.  The Plan is safe to preview;
    the confirmation is issued only after the front door has validated the
    pending user/thread scope.
    """
    validate_mutation_plan(plan)
    return MutationConfirmation(
        confirmation_id=str(confirmation_id or "").strip(),
        plan_id=str(plan.get("plan_id") or ""),
        plan_fingerprint=str(plan.get("plan_fingerprint") or ""),
        capability_id=str(plan.get("capability_id") or ""),
        executor_id=str(plan.get("executor_id") or ""),
        thread_id=str(plan.get("thread_id") if thread_id is None else thread_id),
        session_id=str(plan.get("session_id") if session_id is None else session_id),
        actor_id=str(plan.get("actor_id") if actor_id is None else actor_id),
        confirmed_at=utc_now_iso(),
        confirmation_phrase_class=str(confirmation_phrase_class or "").strip().lower(),
        activation_fingerprint=str(plan.get("activation_fingerprint") or "") or None,
    ).to_dict()


def validate_mutation_confirmation(
    plan: dict[str, Any],
    confirmation: dict[str, Any] | None,
    *,
    now: int | None = None,
) -> None:
    validate_mutation_plan(plan)
    if not isinstance(confirmation, dict):
        raise ValueError("mutation_confirmation_missing")
    if not str(confirmation.get("confirmation_id") or "").strip():
        raise ValueError("mutation_confirmation_token_missing")
    for field in (
        "plan_id",
        "plan_fingerprint",
        "capability_id",
        "executor_id",
        "thread_id",
        "session_id",
        "actor_id",
        "activation_fingerprint",
    ):
        expected = str(plan.get(field) or "")
        actual = str(confirmation.get(field) or "")
        if actual != expected:
            raise ValueError(f"mutation_confirmation_{field}_mismatch")
    if str(confirmation.get("confirmation_phrase_class") or "").strip().lower() != "affirmative":
        raise ValueError("mutation_confirmation_phrase_invalid")
    confirmed_at = normalize_timestamp(confirmation.get("confirmed_at"))
    confirmed_epoch = int(datetime.fromisoformat(confirmed_at).timestamp())
    current = int(now or time.time())
    if confirmed_epoch > current + 30:
        raise ValueError("mutation_confirmation_timestamp_invalid")
    if expires_at_epoch(plan) <= current:
        raise ValueError("mutation_confirmation_expired")


class MutationPlanStore:
    def __init__(self, path: str | Path | None = None, *, max_records: int = 200) -> None:
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self.max_records = max(1, int(max_records))
        self._records: dict[str, dict[str, Any]] = {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    def save(self, plan: dict[str, Any]) -> None:
        validate_mutation_plan(plan)
        plan_id = str(plan.get("plan_id") or "")
        existing = self._records.get(plan_id)
        if existing and existing.get("plan_fingerprint") != plan.get("plan_fingerprint"):
            raise ValueError("mutation_plan_id_reuse")
        self._records[plan_id] = dict(plan)
        self.prune()
        self._persist()

    def load(self, plan_id: str) -> dict[str, Any] | None:
        record = self._records.get(str(plan_id or ""))
        return dict(record) if record else None

    def transition(self, plan_id: str, status: str) -> dict[str, Any] | None:
        if status not in VALID_PLAN_STATUSES:
            raise ValueError("mutation_plan_status_invalid")
        record = self._records.get(str(plan_id or ""))
        if record is None:
            return None
        record = dict(record)
        record["status"] = status
        self._records[str(plan_id)] = record
        self._persist()
        return dict(record)

    def cancel(self, plan_id: str) -> dict[str, Any] | None:
        return self.transition(plan_id, MUTATION_PLAN_STATUS_CANCELLED)

    def prune(self, *, now: int | None = None) -> None:
        current = int(now or time.time())
        for plan_id, record in list(self._records.items()):
            if expires_at_epoch(record) and expires_at_epoch(record) < current:
                record = dict(record)
                record["status"] = MUTATION_PLAN_STATUS_EXPIRED
                self._records[plan_id] = record
        if len(self._records) > self.max_records:
            ordered = sorted(self._records.items(), key=lambda item: str(item[1].get("created_at") or ""))
            for plan_id, _record in ordered[: len(self._records) - self.max_records]:
                self._records.pop(plan_id, None)

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        records = parsed.get("plans") if isinstance(parsed, dict) else []
        if not isinstance(records, list):
            return
        for record in records:
            if isinstance(record, dict):
                try:
                    validate_mutation_plan(record)
                except ValueError:
                    continue
                self._records[str(record.get("plan_id") or "")] = dict(record)

    def _persist(self) -> None:
        if self.path is None:
            return
        payload = {"schema_version": MUTATION_PLAN_SCHEMA_VERSION, "plans": list(self._records.values())}
        self.path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
