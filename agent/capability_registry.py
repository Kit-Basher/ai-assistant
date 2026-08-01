from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


class CapabilityMode(str, Enum):
    READ_ONLY = "read_only"
    MUTATING = "mutating"


class ApprovalPolicy(str, Enum):
    NEVER = "never"
    REQUIRED = "required"


class CapabilityProvenance(str, Enum):
    NATIVE = "native"
    PACK = "pack"


InvocationHook = Callable[[Mapping[str, Any]], Any]
VerificationHook = Callable[[Any], bool]
HealthHook = Callable[[], tuple[bool, str | None]]


@dataclass(frozen=True)
class CapabilityContract:
    """Small JSON-schema subset used at the model/registry trust boundary."""

    properties: Mapping[str, type | tuple[type, ...]] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    allow_extra: bool = False

    def validate(self, value: Mapping[str, Any] | None) -> dict[str, Any]:
        candidate = dict(value or {})
        unknown = sorted(set(candidate) - set(self.properties))
        if unknown and not self.allow_extra:
            raise ValueError(f"unknown_input_fields:{','.join(unknown)}")
        missing = [name for name in self.required if name not in candidate]
        if missing:
            raise ValueError(f"missing_input_fields:{','.join(missing)}")
        for name, expected in self.properties.items():
            if name not in candidate or candidate[name] is None:
                continue
            if not isinstance(candidate[name], expected):
                raise ValueError(f"invalid_input_type:{name}")
        return candidate

    def public_schema(self) -> dict[str, Any]:
        def _type_name(value: type | tuple[type, ...]) -> str:
            if isinstance(value, tuple):
                return "|".join(item.__name__ for item in value)
            return value.__name__

        return {
            "properties": {name: _type_name(value) for name, value in self.properties.items()},
            "required": list(self.required),
            "allow_extra": self.allow_extra,
        }


@dataclass(frozen=True)
class CapabilityDefinition:
    capability_id: str
    description: str
    example_goals: tuple[str, ...]
    input_contract: CapabilityContract
    output_contract: CapabilityContract
    mode: CapabilityMode
    approval_policy: ApprovalPolicy
    invocation_hook: InvocationHook
    verification_hook: VerificationHook
    health_hook: HealthHook
    provenance: CapabilityProvenance = CapabilityProvenance.NATIVE
    capability_type: str = "native"
    material_group: str = "general"

    def availability(self) -> tuple[bool, str | None]:
        try:
            available, reason = self.health_hook()
            return bool(available), str(reason).strip() or None if reason is not None else None
        except Exception as exc:
            return False, f"health_check_failed:{exc.__class__.__name__}"


class CapabilityRegistry:
    """Runtime registry and the only authority for selectable capability IDs."""

    def __init__(self) -> None:
        self._items: dict[str, CapabilityDefinition] = {}

    def register(self, definition: CapabilityDefinition) -> None:
        capability_id = str(definition.capability_id or "").strip().lower()
        if not capability_id or capability_id != definition.capability_id:
            raise ValueError("capability_id_must_be_stable_lowercase")
        if capability_id in self._items:
            raise ValueError(f"duplicate_capability_id:{capability_id}")
        if not definition.description.strip() or not definition.example_goals:
            raise ValueError(f"capability_metadata_incomplete:{capability_id}")
        if definition.mode is CapabilityMode.MUTATING and definition.approval_policy is not ApprovalPolicy.REQUIRED:
            raise ValueError(f"mutating_capability_requires_approval:{capability_id}")
        self._items[capability_id] = definition

    def get(self, capability_id: str) -> CapabilityDefinition | None:
        return self._items.get(str(capability_id or "").strip().lower())

    def require(self, capability_id: str) -> CapabilityDefinition:
        definition = self.get(capability_id)
        if definition is None:
            raise ValueError("unknown_capability_id")
        return definition

    def validate_selection(self, capability_id: str, inputs: Mapping[str, Any] | None) -> tuple[CapabilityDefinition, dict[str, Any]]:
        definition = self.require(capability_id)
        available, reason = definition.availability()
        if not available:
            raise RuntimeError(reason or "capability_unavailable")
        return definition, definition.input_contract.validate(inputs)

    def invoke(self, capability_id: str, inputs: Mapping[str, Any] | None, *, approved: bool = False) -> Any:
        definition, validated = self.validate_selection(capability_id, inputs)
        if definition.approval_policy is ApprovalPolicy.REQUIRED and not approved:
            raise PermissionError("capability_approval_required")
        result = definition.invocation_hook(validated)
        if not definition.verification_hook(result):
            raise RuntimeError("capability_result_verification_failed")
        return result

    def definitions(self, *, available_only: bool = False) -> tuple[CapabilityDefinition, ...]:
        values = tuple(self._items[key] for key in sorted(self._items))
        if not available_only:
            return values
        return tuple(item for item in values if item.availability()[0])

    def public_snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.definitions():
            available, reason = item.availability()
            rows.append(
                {
                    "id": item.capability_id,
                    "description": item.description,
                    "mode": item.mode.value,
                    "approval_policy": item.approval_policy.value,
                    "available": available,
                    "health_reason": reason,
                    "provenance": item.provenance.value,
                    "type": item.capability_type,
                    "input_contract": item.input_contract.public_schema(),
                    "output_contract": item.output_contract.public_schema(),
                    "verification": "hook",
                }
            )
        return rows
