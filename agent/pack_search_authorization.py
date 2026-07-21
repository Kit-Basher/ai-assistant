from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import time
from typing import Any, Callable
import uuid

from agent.capability_policy import stable_fingerprint
from agent.executor_registry import ExecutorRegistry, ExecutorSpec, redact_executor_value
from agent.mutation_boundary import assert_authorized_mutation
from agent.mutation_plan import build_mutation_plan, target_fingerprint_for_snapshot, validate_mutation_plan
from agent.permissions import MODEL_OPS_ACTIONS


CONTROL_FIELDS = {
    "operation", "mutation_plan", "confirmation", "confirm", "confirmed",
    "confirmation_token", "plan_id", "approve", "yes", "actor_id", "thread_id", "session_id",
}


@dataclass(frozen=True)
class V2FMutationSpec:
    operation: str
    capability_id: str
    executor_id: str
    rollback: str
    safe_mode_blocked: bool = False


def _spec(operation: str, capability: str, rollback: str, *, safe: bool = False) -> V2FMutationSpec:
    return V2FMutationSpec(operation, capability, f"{operation}.v1", rollback, safe)


SPECS: dict[str, V2FMutationSpec] = {
    "pack_source.catalog.create": _spec("pack_source.catalog.create", "pack.source.configure", "Restore the previous source catalog document."),
    "pack_source.catalog.update": _spec("pack_source.catalog.update", "pack.source.configure", "Restore the previous source catalog entry."),
    "pack_source.catalog.delete": _spec("pack_source.catalog.delete", "pack.source.configure", "Restore the deleted source and its scoped policy through a new Plan."),
    "pack_source.policy.update": _spec("pack_source.policy.update", "pack.source.configure", "Restore the previous global source policy."),
    "pack_source.scoped_policy.update": _spec("pack_source.scoped_policy.update", "pack.source.configure", "Restore the previous source-specific policy."),
    "permission.policy.update": _spec("permission.policy.update", "permission.policy.configure", "Restore the previous bounded permission policy through a separately authorized Plan."),
    "external_pack.install": _spec("external_pack.install", "pack.lifecycle.install", "Remove only the exact installed/quarantined pack version through a new Plan.", safe=True),
    "external_pack.approve": _spec("external_pack.approve", "pack.lifecycle.approve", "Restore the previous review state through a new Plan."),
    "external_pack.enable": _spec("external_pack.enable", "pack.lifecycle.enable", "Restore the previous enabled state through a new Plan."),
    "external_pack.grant": _spec("external_pack.grant", "pack.permission.grant", "Revoke the exact bounded grant through a new Plan."),
    "external_pack.remove": _spec("external_pack.remove", "pack.lifecycle.remove", "Restore only from a separately reviewed backup; deletion is otherwise irreversible."),
    "search.setup": _spec("search.setup", "search.service.configure", "Remove only the owned service and restore prior search configuration.", safe=True),
    "search.prerequisite": _spec("search.prerequisite", "search.prerequisite.install", "System package installation is not automatically reversible.", safe=True),
}


class PackSearchAuthorizationService:
    """Canonical durable boundary for remaining v2F public mutations."""

    def __init__(self, runtime: Any, *, state_root: str | Path) -> None:
        self.runtime = runtime
        root = Path(state_root).expanduser().resolve()
        self.registry = ExecutorRegistry(
            root / "executor_registry_journal.jsonl",
            confirmation_store_path=root / "confirmation_transactions.sqlite3",
        )
        for spec in SPECS.values():
            self.registry.register(ExecutorSpec(
                executor_id=spec.executor_id,
                action_type=spec.operation,
                status="enabled",
                run=self._executor(spec),
                rollback_available=bool(spec.rollback),
                rollback_hint=spec.rollback,
                capability_id=spec.capability_id,
            ))
        self.registry.freeze()

    @staticmethod
    def _scope(payload: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(payload.get("actor_id") or "loopback_operator"),
            str(payload.get("thread_id") or "operator"),
            str(payload.get("session_id") or "local"),
        )

    @staticmethod
    def _request(payload: dict[str, Any]) -> dict[str, Any]:
        return {str(k): copy.deepcopy(v) for k, v in payload.items() if str(k) not in CONTROL_FIELDS}

    def _runtime_snapshot(self) -> dict[str, Any]:
        truth = self.runtime.runtime_truth_service()
        target = dict(truth.current_chat_target_status())
        target.pop("truth_timing_ms", None)
        search = self.runtime.search_status()
        return {
            "mode": "safe" if self.runtime._safe_mode_enabled() else str(self.runtime.llm_control_mode_status().get("mode") or "controlled").lower(),
            "chat_target": target,
            "search_status": {
                key: search.get(key)
                for key in ("enabled", "provider", "base_url", "endpoint_configured", "available", "search_state")
            },
        }

    def _target_snapshot(self, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        source_id = str(request.get("source_id") or "").strip()
        pack_id = str(request.get("pack_id") or "").strip()
        snapshot: dict[str, Any] = {
            "operation": operation,
            "request": request,
            "runtime": self._runtime_snapshot(),
        }
        if operation.startswith("pack_source."):
            snapshot["catalog"] = self.runtime._pack_registry_discovery().get_catalog()
            snapshot["policy"] = self.runtime._pack_registry_discovery().get_policy()
            snapshot["source_id"] = source_id or str(request.get("id") or "").strip() or None
        elif operation == "permission.policy.update":
            snapshot["current_policy"] = self.runtime.permission_store.load()
        elif operation.startswith("external_pack."):
            snapshot["pack_id"] = pack_id or None
            snapshot["pack"] = self.runtime.pack_store.get_external_pack(pack_id) or self.runtime.pack_store.get_pack(pack_id) if pack_id else None
            grant_path = Path(self.runtime.pack_store.external_storage_root()).joinpath("managed_adapter_grants.json")
            snapshot["grants_fingerprint"] = stable_fingerprint(grant_path.read_text(encoding="utf-8") if grant_path.is_file() else "absent")
            snapshot["permission_policy"] = self.runtime.permission_store.load()
            snapshot["source_policy"] = self.runtime._pack_registry_discovery().get_policy()
        elif operation == "search.setup":
            built = self.runtime._build_search_setup_execution_plan(request)
            snapshot["execution_plan"] = built.get("_execution_plan") or built.get("plan") or built
        elif operation == "search.prerequisite":
            built = self.runtime._build_podman_prerequisite_execution_plan(request)
            snapshot["execution_plan"] = built.get("plan") or built
        return snapshot

    @staticmethod
    def _validate(operation: str, request: dict[str, Any]) -> str | None:
        if len(str(request)) > 512 * 1024:
            return "mutation_request_too_large"
        if any(key in request for key in ("executor", "executor_id", "internal_context", "trusted_invocation_context", "callback", "command", "argv")):
            return "caller_selected_execution_metadata_forbidden"
        if operation.startswith("pack_source."):
            source_id = str(request.get("source_id") or request.get("id") or "").strip()
            if operation != "pack_source.policy.update" and not source_id:
                return "pack_source_id_required"
        if operation.startswith("external_pack.") and operation != "external_pack.install" and not str(request.get("pack_id") or "").strip():
            return "pack_id_required"
        if operation == "external_pack.install":
            source = request.get("source")
            source_text = str(source or "") if not isinstance(source, dict) else str(source.get("url") or "")
            if "://" in source_text or str(request.get("url") or "").strip():
                return "remote_pack_fetch_stage_unimplemented_denied"
            if not str(request.get("path") or request.get("pack_path") or source_text).strip():
                return "local_pack_path_required"
        if operation == "external_pack.approve" and request.get("enable") is not None:
            return "approval_enablement_combination_forbidden"
        if operation == "external_pack.grant" and not isinstance(request.get("adapter"), dict):
            return "exact_adapter_grant_required"
        if operation == "permission.policy.update":
            allowed = {"version", "mode", "actions", "constraints"}
            if not request or any(str(k) not in allowed for k in request):
                return "permission_policy_field_unknown"
            actions = request.get("actions")
            if isinstance(actions, dict):
                if any("*" in str(k) for k in actions):
                    return "permission_wildcard_forbidden"
                if any(str(k) not in set(MODEL_OPS_ACTIONS) for k in actions):
                    return "permission_capability_unknown"
                if any(value is not True and value is not False for value in actions.values()):
                    return "permission_action_value_invalid"
        return None

    def preview(self, operation: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        spec = SPECS.get(operation)
        if spec is None:
            return False, {"ok": False, "error": "mutation_operation_unknown", "mutated": False}
        request = self._request(payload)
        invalid = self._validate(operation, request)
        if invalid:
            return False, {"ok": False, "error": invalid, "mutated": False}
        actor, thread, session = self._scope(payload)
        snapshot = self._target_snapshot(operation, request)
        if operation.startswith("search."):
            execution = snapshot.get("execution_plan") if isinstance(snapshot.get("execution_plan"), dict) else {}
            if execution.get("ok") is False or execution.get("blocked") is True:
                return False, {**execution, "ok": False, "mutated": False}
            if str(execution.get("setup_mode") or "") == "already_configured":
                return True, {"ok": True, "requires_confirmation": False, "plan": execution, "execution_preview": execution, "mutated": False}
        plan = build_mutation_plan(
            plan_id=f"mutation-{uuid.uuid4().hex}", capability_id=spec.capability_id,
            executor_id=spec.executor_id, expires_at_epoch=int(time.time()) + 600,
            actor_id=actor, thread_id=thread, session_id=session, target_snapshot=snapshot,
            mutation_inventory=[{"operation": operation, "resources": self._resources(operation, request)}],
            preserved_resources=["unrelated packs and sources", "secrets", "recovery artifacts"],
            expected_side_effects=[operation],
            recovery={"rollback_available": bool(spec.rollback), "scope": spec.rollback},
            activation_fingerprint=stable_fingerprint(self._runtime_snapshot()),
        )
        plan.update({"action_type": operation, "executor_status": "enabled", "target": operation})
        plan["operation_payload"] = request
        response = {"ok": True, "requires_confirmation": True, "operation": operation, "operation_payload": request, "plan": plan, "mutated": False}
        if operation.startswith("search."):
            execution_preview = snapshot.get("execution_plan") if isinstance(snapshot.get("execution_plan"), dict) else {}
            safe_preview = {
                key: value for key, value in execution_preview.items()
                if key not in {"raw_base_url", "commands"}
            }
            plan.update(safe_preview)
            response["execution_preview"] = safe_preview
        return True, response

    def apply(self, operation: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        spec = SPECS.get(operation)
        plan = payload.get("mutation_plan") if isinstance(payload.get("mutation_plan"), dict) else None
        confirmation = payload.get("confirmation") if isinstance(payload.get("confirmation"), dict) else None
        if spec is None or plan is None or confirmation is None:
            return False, {"ok": False, "error": "scoped_mutation_plan_and_confirmation_required", "mutated": False}
        request = self._request(payload)
        invalid = self._validate(operation, request)
        if invalid:
            return False, {"ok": False, "error": invalid, "mutated": False}
        try:
            validate_mutation_plan(plan)
        except ValueError as exc:
            return False, {"ok": False, "error": str(exc), "mutated": False}
        if str(plan.get("capability_id")) != spec.capability_id or str(plan.get("executor_id")) != spec.executor_id:
            return False, {"ok": False, "error": "mutation_operation_scope_mismatch", "mutated": False}
        snapshot = self._target_snapshot(operation, request)
        if target_fingerprint_for_snapshot(snapshot) != str(plan.get("target_fingerprint") or ""):
            return False, {"ok": False, "error": "mutation_plan_target_changed", "mutated": False}
        execution_plan = {
            "plan_id": str(plan.get("plan_id") or ""), "action_type": operation,
            "executor_status": "enabled", "target": operation, "risk_level": "high",
            "capability_id": spec.capability_id, "policy_schema_version": int(plan.get("policy_version") or 1),
            "plan_fingerprint": str(plan.get("plan_fingerprint") or ""),
            "target_fingerprint": str(plan.get("target_fingerprint") or ""), "mutation_plan": plan,
        }
        action = {
            "type": operation, "origin": "pack_search_authorization", "pending_id": str(plan.get("plan_id") or ""),
            "target_snapshot": snapshot, "parameters": request, "runtime_mode": "production",
        }
        result = self.registry.execute_confirmed_plan(plan=execution_plan, action=action, confirmation=confirmation, high_risk_confirmed=True)
        receipt = redact_executor_value(result.to_dict())
        details = result.details if isinstance(result.details, dict) else {}
        legacy = details.get("result") if isinstance(details.get("result"), dict) else {}
        body = {**redact_executor_value(legacy), "authorization_receipt": receipt, "ok": bool(result.ok), "mutated": bool(result.mutated), "capability_id": result.capability_id, "executor_id": result.executor_id, "error_code": result.error_code}
        if not result.ok and result.error_code:
            body.setdefault("error", result.error_code)
        return bool(result.ok), body

    def route(self, operation: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if any(payload.get(key) is True for key in ("confirm", "confirmed", "yes", "approve")) and not isinstance(payload.get("confirmation"), dict):
            return False, {"ok": False, "error": "boolean_or_legacy_confirmation_not_authorization", "mutated": False}
        if "confirmation_token" in payload or ("plan_id" in payload and "mutation_plan" not in payload):
            return False, {"ok": False, "error": "legacy_confirmation_token_rejected", "mutated": False}
        return self.apply(operation, payload) if ("mutation_plan" in payload or "confirmation" in payload) else self.preview(operation, payload)

    @staticmethod
    def _resources(operation: str, request: dict[str, Any]) -> list[str]:
        identifiers = [f"operation:{operation}"]
        for key in ("source_id", "pack_id"):
            if str(request.get(key) or "").strip():
                identifiers.append(f"{key}:{str(request[key]).strip()}")
        if operation.startswith("search."):
            identifiers.extend(["service:personal-agent-searxng", "bind:127.0.0.1"])
        return identifiers

    def _executor(self, spec: V2FMutationSpec) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
        def run(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
            valid, reason, _ = assert_authorized_mutation(
                action.get("trusted_invocation_context"), expected_capability=spec.capability_id,
                expected_executor=spec.executor_id, expected_operation=str(plan.get("plan_id") or ""),
                expected_plan_fingerprint=str(plan.get("plan_fingerprint") or ""),
                expected_target_fingerprint=str(plan.get("target_fingerprint") or ""), runtime_mode="production",
            )
            if not valid:
                return {"ok": False, "mutated": False, "error_code": reason or "generic_bypass_blocked"}
            if spec.safe_mode_blocked and self.runtime._safe_mode_enabled():
                return {"ok": False, "mutated": False, "error_code": "safe_mode_mutation_blocked"}
            p = dict(action.get("parameters") if isinstance(action.get("parameters"), dict) else {})
            result = self._invoke(spec.operation, p, snapshot=action.get("target_snapshot"))
            ok = bool(result[0]) if isinstance(result, tuple) else bool(result.get("ok"))
            body = result[1] if isinstance(result, tuple) else result
            return {"ok": ok, "mutated": bool(ok and body.get("mutated", True)), "resources_touched": self._resources(spec.operation, p), "details": {"result": body}, "error_code": None if ok else str(body.get("error") or "mutation_failed"), "user_message": str(body.get("message") or "")}
        return run

    def _invoke(self, operation: str, p: dict[str, Any], *, snapshot: Any = None) -> tuple[bool, dict[str, Any]] | dict[str, Any]:
        actor = str(p.pop("changed_by", "loopback_operator") or "loopback_operator")
        if operation == "pack_source.catalog.create": return self.runtime.create_pack_source_catalog(p, changed_by=actor)
        if operation == "pack_source.catalog.update": return self.runtime.update_pack_source_catalog(str(p.pop("source_id")), p, changed_by=actor)
        if operation == "pack_source.catalog.delete": return self.runtime.delete_pack_source_catalog(str(p.pop("source_id")), changed_by=actor)
        if operation == "pack_source.policy.update": return self.runtime.update_pack_sources_policy(p, changed_by=actor)
        if operation == "pack_source.scoped_policy.update": return self.runtime.update_pack_source_policy(str(p.pop("source_id")), p, changed_by=actor)
        if operation == "permission.policy.update": return self.runtime.update_permissions(p)
        if operation == "external_pack.install": return self.runtime.packs_install(p)
        if operation == "external_pack.approve": p["approve"] = True; return self.runtime.packs_approve(p)
        if operation == "external_pack.enable": return self.runtime.packs_enable(p)
        if operation == "external_pack.grant": return self.runtime.packs_grant(p)
        if operation == "external_pack.remove": return self.runtime.delete_external_pack(str(p.get("pack_id") or ""), changed_by=actor)
        if operation == "search.setup":
            built = self.runtime._build_search_setup_execution_plan(p)
            execution_plan = built.get("_execution_plan") or built.get("plan") or built
            return self.runtime._execute_search_setup_plan(execution_plan)
        if operation == "search.prerequisite":
            built = self.runtime._build_podman_prerequisite_execution_plan(p)
            execution_plan = built.get("plan") or built
            return self.runtime._execute_podman_prerequisite_plan(execution_plan)
        return {"ok": False, "mutated": False, "error": "mutation_operation_unknown"}
