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
SelfTestHook = Callable[[], Mapping[str, Any]]
IndependentVerificationHook = Callable[[Mapping[str, Any], Any], Mapping[str, Any]]
TaskInputValidationHook = Callable[[Mapping[str, Any]], tuple[bool, str | None]]
InputNormalizerHook = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class CapabilityHealthState(str, Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CapabilityHealth:
    state: CapabilityHealthState
    reason: str | None = None
    next_step: str | None = None

    @property
    def available(self) -> bool:
        return self.state in {CapabilityHealthState.AVAILABLE, CapabilityHealthState.DEGRADED}

    def public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "available": self.available,
            "reason": self.reason,
            "next_step": self.next_step,
        }


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
    permission_requirements: tuple[str, ...] = ()
    mode_requirements: tuple[str, ...] = ()
    unavailable_message: str = "This capability is not available in the current runtime."
    unavailable_invocation_safe: bool = False
    proof_requirements: tuple[str, ...] = ()
    proof_nodes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    self_test_hook: SelfTestHook | None = None
    chat_selectable: bool = True
    task_composable: bool = True
    retry_safety: str = "never"
    max_task_retries: int = 0
    resume_policy: str = "never"
    independent_verification_hook: IndependentVerificationHook | None = None
    compensation_capability_id: str | None = None
    task_input_validation_hook: TaskInputValidationHook | None = None
    # The runtime input contract can retain compatibility-only fields.  This
    # optional subset is the exact model-visible surface for native tool
    # calling; ``None`` means the full contract (used by dynamic packs).
    model_input_fields: tuple[str, ...] | None = None
    # A model-facing contract may deliberately rename or narrow compatibility
    # inputs retained by the native implementation.  Translation remains a
    # deterministic registry-bound operation, never model authority.
    model_input_contract: CapabilityContract | None = None
    input_normalizer_hook: InputNormalizerHook | None = None

    def model_contract(self) -> CapabilityContract:
        if self.model_input_contract is not None:
            return self.model_input_contract
        fields = self.model_input_fields
        properties = {
            name: expected
            for name, expected in self.input_contract.properties.items()
            if name not in {"user_id", "text"} and (fields is None or name in fields)
        }
        required = tuple(name for name in self.input_contract.required if name in properties)
        return CapabilityContract(properties=properties, required=required)

    def availability(self) -> tuple[bool, str | None]:
        health = self.health()
        return health.available, health.reason

    def health(self) -> CapabilityHealth:
        try:
            available, reason = self.health_hook()
            clean_reason = str(reason).strip() or None if reason is not None else None
            return CapabilityHealth(
                (
                    CapabilityHealthState.DEGRADED
                    if bool(available) and clean_reason
                    else CapabilityHealthState.AVAILABLE
                    if bool(available)
                    else CapabilityHealthState.UNAVAILABLE
                ),
                clean_reason,
            )
        except Exception as exc:
            return CapabilityHealth(CapabilityHealthState.UNAVAILABLE, f"health_check_failed:{exc.__class__.__name__}")

    def self_test(self) -> dict[str, Any]:
        if self.self_test_hook is None:
            return {"ok": False, "status": "missing", "reason": "self_test_hook_missing"}
        try:
            result = dict(self.self_test_hook())
        except Exception as exc:
            return {"ok": False, "status": "fail", "reason": f"self_test_failed:{exc.__class__.__name__}"}
        result.setdefault("ok", False)
        result.setdefault("status", "pass" if bool(result["ok"]) else "fail")
        return result


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
        if definition.retry_safety not in {"read_only_safe", "idempotent", "never", "reconcile_first"}:
            raise ValueError(f"capability_retry_safety_invalid:{capability_id}")
        if definition.mode is CapabilityMode.MUTATING and definition.retry_safety == "read_only_safe":
            raise ValueError(f"mutating_capability_retry_safety_invalid:{capability_id}")
        if not 0 <= int(definition.max_task_retries) <= 2:
            raise ValueError(f"capability_task_retry_bound_invalid:{capability_id}")
        if definition.resume_policy not in {"revalidate", "never", "reconcile_first"}:
            raise ValueError(f"capability_resume_policy_invalid:{capability_id}")
        self._items[capability_id] = definition

    def unregister_external(self, capability_id: str) -> bool:
        """Remove only dynamically registered external authority.

        Native inventory entries cannot be removed through the pack lifecycle.
        """
        key = str(capability_id or "").strip().lower()
        definition = self._items.get(key)
        if definition is None:
            return False
        if definition.provenance is not CapabilityProvenance.PACK:
            raise PermissionError("native_capability_cannot_be_unregistered")
        del self._items[key]
        return True

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
        if not available and not definition.unavailable_invocation_safe:
            raise RuntimeError(reason or "capability_unavailable")
        validated = definition.input_contract.validate(inputs)
        if definition.input_normalizer_hook is not None:
            validated = definition.input_contract.validate(definition.input_normalizer_hook(validated))
        return definition, validated

    def invoke(self, capability_id: str, inputs: Mapping[str, Any] | None, *, approved: bool = False) -> Any:
        definition, validated = self.validate_selection(capability_id, inputs)
        if definition.approval_policy is ApprovalPolicy.REQUIRED and not approved:
            raise PermissionError("capability_approval_required")
        result = definition.invocation_hook(validated)
        if not definition.verification_hook(result):
            raise RuntimeError("capability_result_verification_failed")
        return result

    def preview_mutation(self, capability_id: str, inputs: Mapping[str, Any] | None) -> Any:
        """Invoke only the registered mutation preview boundary.

        This does not grant approval.  The capability hook must return its
        canonical preview; execution remains owned by the confirmation path.
        """
        definition, validated = self.validate_selection(capability_id, inputs)
        if definition.mode is not CapabilityMode.MUTATING:
            raise ValueError("capability_mutation_preview_requires_mutating_capability")
        return definition.invocation_hook(validated)

    def definitions(
        self,
        *,
        available_only: bool = False,
        chat_selectable_only: bool = False,
    ) -> tuple[CapabilityDefinition, ...]:
        values = tuple(self._items[key] for key in sorted(self._items))
        if chat_selectable_only:
            values = tuple(item for item in values if item.chat_selectable)
        if not available_only:
            return values
        return tuple(item for item in values if item.availability()[0])

    def public_snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.definitions():
            health = item.health()
            rows.append(
                {
                    "id": item.capability_id,
                    "description": item.description,
                    "mode": item.mode.value,
                    "approval_policy": item.approval_policy.value,
                    "available": health.state is CapabilityHealthState.AVAILABLE,
                    "usable": health.available,
                    "health": health.public_dict(),
                    "health_reason": health.reason,
                    "provenance": item.provenance.value,
                    "type": item.capability_type,
                    "material_group": item.material_group,
                    "input_contract": item.input_contract.public_schema(),
                    "model_input_contract": item.model_contract().public_schema(),
                    "output_contract": item.output_contract.public_schema(),
                    "verification": "hook",
                    "self_test": "hook" if item.self_test_hook is not None else "missing",
                    "permissions": list(item.permission_requirements),
                    "mode_requirements": list(item.mode_requirements),
                    "chat_selectable": item.chat_selectable,
                    "proof_requirements": list(item.proof_requirements),
                    "unavailable_invocation_safe": item.unavailable_invocation_safe,
                    "proof_categories": sorted(item.proof_nodes),
                    "task_composable": item.task_composable,
                    "task_retry_safety": item.retry_safety,
                    "task_max_retries": item.max_task_retries,
                    "task_resume_policy": item.resume_policy,
                    "task_independent_verifier": "hook" if item.independent_verification_hook is not None else "registry_result_only",
                    "task_compensation_capability_id": item.compensation_capability_id,
                }
            )
        return rows
