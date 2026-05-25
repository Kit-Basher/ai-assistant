from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass
class ManagedActionStep:
    """One planned, executed, or rollback step in a managed action."""

    name: str
    status: str = "planned"
    resource: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "resource": self.resource,
            "details": dict(self.details),
        }


@dataclass
class ManagedActionJournal:
    """In-memory journal for a bounded managed action attempt.

    The journal records what the runtime planned, what it actually changed, and
    which rollback steps were attempted. It is intentionally data-only: callers
    own all validation and execution so rollback stays scoped to owned changes.
    """

    action_type: str
    target: str
    action_id: str = field(default_factory=lambda: f"managed-action-{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    planned_steps: list[ManagedActionStep] = field(default_factory=list)
    executed_steps: list[ManagedActionStep] = field(default_factory=list)
    created_resources: list[dict[str, Any]] = field(default_factory=list)
    changed_resources: list[dict[str, Any]] = field(default_factory=list)
    rollback_steps: list[ManagedActionStep] = field(default_factory=list)
    verification_result: dict[str, Any] = field(default_factory=dict)
    rollback_result: dict[str, Any] = field(default_factory=dict)

    def plan_step(self, name: str, *, resource: str | None = None, **details: Any) -> None:
        self.planned_steps.append(ManagedActionStep(name=name, resource=resource, details=_clean(details)))

    def record_step(self, name: str, *, ok: bool, resource: str | None = None, **details: Any) -> None:
        self.executed_steps.append(
            ManagedActionStep(name=name, status="ok" if ok else "failed", resource=resource, details=_clean(details))
        )

    def record_created_resource(self, kind: str, identifier: str, **details: Any) -> None:
        self.created_resources.append({"kind": kind, "identifier": identifier, **_clean(details)})

    def record_changed_resource(self, kind: str, identifier: str, **details: Any) -> None:
        self.changed_resources.append({"kind": kind, "identifier": identifier, **_clean(details)})

    def record_rollback_step(self, name: str, *, ok: bool, resource: str | None = None, **details: Any) -> None:
        self.rollback_steps.append(
            ManagedActionStep(name=name, status="ok" if ok else "failed", resource=resource, details=_clean(details))
        )

    def mark_verification(self, *, ok: bool, **details: Any) -> None:
        self.verification_result = {"ok": bool(ok), **_clean(details)}

    def mark_rollback(self, *, ok: bool, **details: Any) -> None:
        self.rollback_result = {"ok": bool(ok), **_clean(details)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "target": self.target,
            "created_at": self.created_at,
            "planned_steps": [step.to_dict() for step in self.planned_steps],
            "executed_steps": [step.to_dict() for step in self.executed_steps],
            "created_resources": [dict(item) for item in self.created_resources],
            "changed_resources": [dict(item) for item in self.changed_resources],
            "rollback_steps": [step.to_dict() for step in self.rollback_steps],
            "verification_result": dict(self.verification_result),
            "rollback_result": dict(self.rollback_result),
        }


def _clean(values: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [item if isinstance(item, (str, int, float, bool)) else str(item) for item in value]
        elif isinstance(value, dict):
            cleaned[key] = {
                str(inner_key): inner_value if isinstance(inner_value, (str, int, float, bool)) else str(inner_value)
                for inner_key, inner_value in value.items()
            }
        else:
            cleaned[key] = str(value)
    return cleaned


__all__ = ["ManagedActionJournal", "ManagedActionStep"]
