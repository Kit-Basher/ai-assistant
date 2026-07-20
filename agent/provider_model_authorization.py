from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import hmac
import ipaddress
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid
from urllib.parse import urlsplit

from agent.capability_policy import stable_fingerprint
from agent.executor_registry import ExecutorRegistry, ExecutorSpec, redact_executor_value
from agent.mutation_boundary import assert_authorized_mutation
from agent.mutation_plan import build_mutation_plan, target_fingerprint_for_snapshot, validate_mutation_plan
from agent.secret_store import SecretStore
from agent.llm.model_discovery_policy import allowed_model_discovery_role_hints, allowed_model_discovery_statuses


SECRET_FIELDS = {"api_key", "secret", "token", "password", "authorization"}


@dataclass(frozen=True)
class DomainMutationSpec:
    operation: str
    capability_id: str
    executor_id: str
    action_type: str
    rollback_available: bool
    rollback_hint: str
    safe_mode_blocked: bool = True
    secret_field: str = ""


SPECS: dict[str, DomainMutationSpec] = {
    "provider.add": DomainMutationSpec("provider.add", "provider.configure", "provider.add.v1", "provider.add", True, "Restore the prior provider registry snapshot."),
    "provider.update": DomainMutationSpec("provider.update", "provider.configure", "provider.update.v1", "provider.update", True, "Restore the prior provider registry snapshot."),
    "provider.delete": DomainMutationSpec("provider.delete", "provider.configure", "provider.delete.v1", "provider.delete", True, "Restore the prior provider registry snapshot."),
    "provider.model.add": DomainMutationSpec("provider.model.add", "model.configure", "provider.model.add.v1", "provider.model.add", True, "Restore the prior provider/model registry snapshot."),
    "provider.secret.set": DomainMutationSpec("provider.secret.set", "provider.secret.manage", "provider.secret.set.v1", "provider.secret.set", False, "The old plaintext is never journalled; restore only through a separately authorized secret replacement.", secret_field="api_key"),
    "telegram.secret.set": DomainMutationSpec("telegram.secret.set", "secret.manage", "telegram.secret.set.v1", "telegram.secret.set", False, "The old plaintext is never journalled; replace it only through a new authorized Plan.", safe_mode_blocked=False, secret_field="bot_token"),
    "config.update": DomainMutationSpec("config.update", "runtime.policy.configure", "config.update.v1", "config.update", True, "Restore the prior bounded configuration fields."),
    "defaults.update": DomainMutationSpec("defaults.update", "model.configure", "defaults.update.v1", "defaults.update", True, "Use an independently authorized rollback against the recorded registry snapshot."),
    "defaults.rollback": DomainMutationSpec("defaults.rollback", "model.configure", "defaults.rollback.v1", "defaults.rollback", False, "Rollback itself requires a new Plan; automatic retry is forbidden."),
    "model.switch": DomainMutationSpec("model.switch", "model.configure", "model.switch.v1", "model.switch", True, "Authorize a switch to the previously recorded verified target."),
    "model.switch_temporary": DomainMutationSpec("model.switch_temporary", "model.configure", "model.switch_temporary.v1", "model.switch_temporary", True, "Clear the temporary override through a new authorized Plan."),
    "model.acquire": DomainMutationSpec("model.acquire", "model.acquire", "model.acquire.v1", "model.acquire", False, "Downloaded artifacts may require a separately authorized removal after reconciliation."),
    "model.policy": DomainMutationSpec("model.policy", "runtime.policy.configure", "model.policy.v1", "model.policy", True, "Restore the previous policy snapshot through a new Plan."),
    "runtime.control_mode": DomainMutationSpec("runtime.control_mode", "runtime.policy.configure", "runtime.control_mode.v1", "runtime.control_mode", True, "Restore the prior policy mode through a new confirmed Plan.", safe_mode_blocked=False),
    "model.refresh": DomainMutationSpec("model.refresh", "model.maintain", "model.refresh.v1", "model.refresh", False, "Refresh cache state again after reconciliation."),
    "model_watch.run": DomainMutationSpec("model_watch.run", "model.maintain", "model_watch.run.v1", "model_watch.run", False, "Advisory/bookkeeping output is append-only; reconcile before retry."),
    "model_watch.refresh": DomainMutationSpec("model_watch.refresh", "model.maintain", "model_watch.refresh.v1", "model_watch.refresh", False, "Refresh again only with a new Plan."),
    "model_watch.hf_scan": DomainMutationSpec("model_watch.hf_scan", "model.maintain", "model_watch.hf_scan.v1", "model_watch.hf_scan", False, "Scan bookkeeping cannot be rolled back automatically."),
    "llm.fix": DomainMutationSpec("llm.fix", "setup.repair", "llm.fix.v1", "llm.fix", True, "Use the operation-specific recovery receipt."),
    "setup.bootstrap": DomainMutationSpec("setup.bootstrap", "setup.repair", "setup.bootstrap.v1", "setup.bootstrap", True, "Restore only bootstrap-owned state through a new Plan.", safe_mode_blocked=False),
    "llm.autoconfig": DomainMutationSpec("llm.autoconfig", "setup.repair", "llm.autoconfig.v1", "llm.autoconfig", True, "Restore the prior registry snapshot through a new Plan."),
    "llm.reconcile": DomainMutationSpec("llm.reconcile", "setup.repair", "llm.reconcile.v1", "llm.reconcile", True, "Restore prior capability metadata through a new Plan."),
    "llm.hygiene": DomainMutationSpec("llm.hygiene", "model.maintain", "llm.hygiene.v1", "llm.hygiene", False, "Destructive hygiene is not automatically reversible."),
    "llm.cleanup": DomainMutationSpec("llm.cleanup", "model.maintain", "llm.cleanup.v1", "llm.cleanup", False, "Cleanup is not automatically reversible."),
    "llm.self_heal": DomainMutationSpec("llm.self_heal", "setup.repair", "llm.self_heal.v1", "llm.self_heal", True, "Use the operation receipt to authorize a bounded reversal."),
    "llm.support.remediate": DomainMutationSpec("llm.support.remediate", "setup.repair", "llm.support.remediate.v1", "llm.support.remediate", True, "Use the remediation receipt to authorize a bounded reversal."),
    "llm.registry.rollback": DomainMutationSpec("llm.registry.rollback", "runtime.policy.configure", "llm.registry.rollback.v1", "llm.registry.rollback", False, "A rollback is itself single-use and cannot be replayed."),
    "llm.autopilot.undo": DomainMutationSpec("llm.autopilot.undo", "runtime.policy.configure", "llm.autopilot.undo.v1", "llm.autopilot.undo", False, "Undo cannot be retried automatically."),
    "llm.autopilot.unpause": DomainMutationSpec("llm.autopilot.unpause", "runtime.policy.configure", "llm.autopilot.unpause.v1", "llm.autopilot.unpause", True, "Re-pause through a new confirmed Plan."),
    "llm.autopilot.bootstrap": DomainMutationSpec("llm.autopilot.bootstrap", "setup.repair", "llm.autopilot.bootstrap.v1", "llm.autopilot.bootstrap", True, "Restore the prior autopilot state through a new Plan."),
    "modelops.execute": DomainMutationSpec("modelops.execute", "model.acquire", "modelops.execute.v1", "modelops.execute", False, "Reconcile installed artifacts before any follow-up."),
}


def _without_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).strip().lower() not in {"secret_namespace", "current_secret_version", "proposed_secret_version"}
                and any(marker in str(key).strip().lower() for marker in SECRET_FIELDS)
                else _without_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return value


class ProviderModelAuthorizationService:
    """Canonical Plan/confirm/executor boundary for provider/model/setup mutations."""

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
                action_type=spec.action_type,
                status="enabled",
                run=self._executor(spec),
                rollback_available=spec.rollback_available,
                rollback_hint=spec.rollback_hint,
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

    def _runtime_snapshot(self) -> dict[str, Any]:
        truth = self.runtime.runtime_truth_service()
        target = dict(truth.current_chat_target_status())
        # Diagnostic timing varies from call to call and is not configuration
        # state. Binding it would make every otherwise valid confirmation stale.
        target.pop("truth_timing_ms", None)
        registry = copy.deepcopy(self.runtime.registry_document)
        mode = "safe" if bool(self.runtime._safe_mode_enabled()) else str(self.runtime.llm_control_mode_status().get("mode") or "controlled").lower()
        return {
            "mode": mode,
            "chat_target": _without_secrets(target),
            "registry_fingerprint": stable_fingerprint(_without_secrets(registry)),
        }

    @staticmethod
    def _secret_version(secret_store: SecretStore, namespace_value: str, value: str) -> str:
        _ = secret_store
        namespace = str(namespace_value).encode("utf-8")
        key = hmac.new(SecretStore._machine_secret(), b"personal-agent:secret-plan:v1", hashlib.sha256).digest()
        return "opaque-v1:" + hmac.new(key, namespace + b"\0" + value.encode("utf-8"), hashlib.sha256).hexdigest()

    def _bound_request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        spec = SPECS[operation]
        request = {str(k): v for k, v in payload.items() if k not in {"operation", "mutation_plan", "confirmation", "confirm", "actor_id", "thread_id", "session_id"}}
        if spec.secret_field:
            secret = str(request.pop(spec.secret_field, "") or "")
            provider_id = str(request.get("provider_id") or request.get("provider") or "").strip().lower()
            namespace = f"provider:{provider_id}:api_key" if operation == "provider.secret.set" else "telegram:bot_token"
            request["secret_namespace"] = namespace
            request["proposed_secret_version"] = self._secret_version(self.runtime.secret_store, namespace, secret)
            current = self.runtime.secret_store.get_secret(namespace) or ""
            request["current_secret_version"] = self._secret_version(self.runtime.secret_store, namespace, current) if current else "absent"
        if operation in {"model.acquire", "modelops.execute"}:
            request["acquisition_binding"] = {
                "source": str(request.get("source") or "ollama").strip().lower(),
                "artifact_digest": str(request.get("artifact_digest") or "unavailable").strip(),
                "estimated_download_bytes": request.get("estimated_download_bytes"),
                "estimated_cost": request.get("estimated_cost"),
                "disk_target": "provider-managed-model-store",
                "rollback_limit": "reconcile outcome before separately authorized removal",
            }
        return _without_secrets(request)

    def _operation_state(self, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        document = self.runtime.registry_document if isinstance(self.runtime.registry_document, dict) else {}
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
        provider = str(request.get("provider_id") or request.get("provider") or request.get("id") or "").strip().lower()
        model = str(request.get("model_id") or request.get("model") or request.get("model_ref") or "").strip()
        current = {
            "provider": _without_secrets(providers.get(provider)) if provider else None,
            "model": _without_secrets(models.get(model) or models.get(f"{provider}:{model}")) if model else None,
            "defaults": _without_secrets(defaults),
        }
        if operation == "llm.fix":
            fixit_store = getattr(self.runtime, "_llm_fixit_store", None)
            current["fixit_state"] = _without_secrets(
                copy.deepcopy(getattr(fixit_store, "state", {}))
            )
        return {
            "provider_identity": provider or None,
            "model_identity": model or None,
            "current_state_hash": stable_fingerprint(current),
            "proposed_state_hash": stable_fingerprint({"current": current, "request": request, "operation": operation}),
            "affected_routing_roles": ["default_provider", "default_model", "chat_model", "fallback_policy"],
        }

    def preview(self, operation: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        spec = SPECS.get(operation)
        if spec is None:
            return False, {"ok": False, "error": "mutation_operation_unknown", "mutated": False}
        if operation == "setup.bootstrap" and payload.get("force") is not True:
            question = 'Include {"force": true} to run bootstrap snapshot?'
            return True, {
                "ok": True,
                "intent": "bootstrap",
                "did_work": False,
                "error_kind": "needs_clarification",
                "message": question,
                "next_question": question,
                "requires_confirmation": False,
                "mutated": False,
            }
        if operation == "setup.bootstrap" and (
            not bool(self.runtime.config.memory_v2_enabled) or self.runtime._memory_v2_store is None
        ):
            error = "memory_v2_disabled" if not bool(self.runtime.config.memory_v2_enabled) else "memory_v2_store_unavailable"
            message = "memory_v2 is disabled." if error == "memory_v2_disabled" else "memory_v2 store is unavailable."
            return False, {
                "ok": False,
                "error": error,
                "error_kind": "feature_disabled",
                "message": message,
                "mutated": False,
                "envelope": {
                    "ok": False,
                    "intent": "bootstrap",
                    "did_work": False,
                    "error_kind": "feature_disabled",
                    "message": message,
                    "errors": [error],
                },
            }
        invalid = self._validate_request(operation, payload)
        if invalid:
            body: dict[str, Any] = {"ok": False, "error": invalid, "error_kind": "bad_request", "message": "The mutation request is invalid.", "mutated": False}
            if operation == "model.policy":
                body["envelope"] = {"allowed_statuses": allowed_model_discovery_statuses(), "allowed_role_hints": allowed_model_discovery_role_hints()}
            return False, body
        if spec.safe_mode_blocked and bool(self.runtime._safe_mode_enabled()):
            provider = str(payload.get("provider") or payload.get("provider_id") or "").strip().lower()
            model = str(payload.get("model_id") or payload.get("model") or "").strip()
            if operation in {"model.switch", "model.switch_temporary"} and provider != "ollama":
                return False, {**self.runtime._safe_mode_remote_switch_response(model), "mutated": False, "operation": operation}
            return False, {"ok": False, "error": "safe_mode_mutation_blocked", "error_kind": "safe_mode_mutation_blocked", "message": "SAFE MODE blocked this mutation.", "mutated": False, "operation": operation}
        actor, thread, session = self._scope(payload)
        bound_request = self._bound_request(operation, payload)
        runtime_snapshot = self._runtime_snapshot()
        snapshot = {**runtime_snapshot, "operation": operation, "request": bound_request, "operation_state": self._operation_state(operation, bound_request)}
        plan = build_mutation_plan(
            plan_id=f"mutation-{uuid.uuid4().hex}", capability_id=spec.capability_id,
            executor_id=spec.executor_id, expires_at_epoch=int(time.time()) + 600,
            actor_id=actor, thread_id=thread, session_id=session, target_snapshot=snapshot,
            mutation_inventory=[{"operation": operation, "resources": self._resources(operation, bound_request)}],
            preserved_resources=["secret plaintext", "unrelated providers/models", "recovery artifacts"],
            expected_side_effects=[operation],
            recovery={"rollback_available": spec.rollback_available, "scope": spec.rollback_hint},
            activation_fingerprint=stable_fingerprint(runtime_snapshot),
        )
        plan.update({
            "action_type": spec.action_type,
            "executor_status": "enabled",
            "target": operation,
        })
        return True, {"ok": True, "requires_confirmation": True, "operation": operation, "plan": plan, "mutated": False}

    @staticmethod
    def _validate_request(operation: str, payload: dict[str, Any]) -> str | None:
        if len(str(payload)) > 128 * 1024:
            return "mutation_request_too_large"
        provider = str(payload.get("provider_id") or payload.get("provider") or payload.get("id") or "").strip().lower()
        if provider and (len(provider) > 64 or not all(ch.isalnum() or ch in "_-" for ch in provider)):
            return "provider_identity_invalid"
        for key in ("model", "model_id", "model_ref"):
            if key in payload and len(str(payload.get(key) or "")) > 256:
                return "model_identity_invalid"
        if operation == "model.policy":
            action = str(payload.get("action") or "upsert").strip().lower()
            if action not in {"upsert", "remove"}:
                return "model_policy_action_invalid"
            if not str(payload.get("model_id") or "").strip():
                return "model_identity_invalid"
            if action == "upsert" and str(payload.get("status") or "").strip().lower() not in set(allowed_model_discovery_statuses()):
                return "model_policy_status_invalid"
            role_hints = payload.get("role_hints") if isinstance(payload.get("role_hints"), list) else []
            if any(str(item).strip().lower() not in set(allowed_model_discovery_role_hints()) for item in role_hints):
                return "model_policy_role_hint_invalid"
        if operation not in {"provider.add", "provider.update"} or "base_url" not in payload:
            return None
        try:
            parsed = urlsplit(str(payload.get("base_url") or ""))
        except ValueError:
            return "provider_url_invalid"
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return "provider_url_scheme_invalid"
        if parsed.username or parsed.password:
            return "provider_url_credentials_forbidden"
        host = str(parsed.hostname).lower()
        is_local_declared = bool(payload.get("local"))
        if host in {"localhost", "127.0.0.1", "::1"}:
            return None if is_local_declared else "provider_private_url_requires_local_provider"
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return None
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            return "provider_private_url_forbidden"
        return None

    @staticmethod
    def _resources(operation: str, request: dict[str, Any]) -> list[str]:
        provider = str(request.get("provider_id") or request.get("provider") or "").strip()
        model = str(request.get("model") or request.get("model_id") or request.get("model_ref") or "").strip()
        return [item for item in (f"operation:{operation}", f"provider:{provider}" if provider else "", f"model:{model}" if model else "") if item]

    def apply(self, operation: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        spec = SPECS.get(operation)
        plan = payload.get("mutation_plan") if isinstance(payload.get("mutation_plan"), dict) else None
        confirmation = payload.get("confirmation") if isinstance(payload.get("confirmation"), dict) else None
        if spec is None or plan is None or confirmation is None:
            return False, {"ok": False, "error": "scoped_mutation_plan_and_confirmation_required", "mutated": False}
        invalid = self._validate_request(operation, payload)
        if invalid:
            return False, {"ok": False, "error": invalid, "error_kind": "bad_request", "message": "The mutation request is invalid.", "mutated": False}
        try:
            validate_mutation_plan(plan)
        except ValueError as exc:
            return False, {"ok": False, "error": str(exc), "mutated": False}
        if str(plan.get("capability_id")) != spec.capability_id or str(plan.get("executor_id")) != spec.executor_id:
            return False, {"ok": False, "error": "mutation_operation_scope_mismatch", "mutated": False}
        bound_request = self._bound_request(operation, payload)
        current_runtime = self._runtime_snapshot()
        current_snapshot = {**current_runtime, "operation": operation, "request": bound_request, "operation_state": self._operation_state(operation, bound_request)}
        if target_fingerprint_for_snapshot(current_snapshot) != str(plan.get("target_fingerprint") or ""):
            return False, {"ok": False, "error": "mutation_plan_target_changed", "mutated": False}
        action = {
            "type": spec.action_type, "origin": "provider_model_authorization",
            "pending_id": str(plan.get("plan_id") or ""), "target_snapshot": current_snapshot,
            "parameters": {
                str(k): v
                for k, v in payload.items()
                if k not in {"operation", "mutation_plan", "confirmation", "confirm", "actor_id", "thread_id", "session_id"}
            },
            "runtime_mode": "production",
        }
        execution_plan = {
            "plan_id": str(plan.get("plan_id") or ""),
            "action_type": spec.action_type,
            "executor_status": "enabled",
            "target": operation,
            "risk_level": str(plan.get("risk_level") or "high"),
            "capability_id": spec.capability_id,
            "policy_schema_version": int(plan.get("policy_version") or 1),
            "plan_fingerprint": str(plan.get("plan_fingerprint") or ""),
            "target_fingerprint": str(plan.get("target_fingerprint") or ""),
            "mutation_plan": plan,
        }
        result = self.registry.execute_confirmed_plan(
            plan=execution_plan,
            action=action,
            confirmation=confirmation,
            high_risk_confirmed=True,
        )
        body = redact_executor_value(result.to_dict())
        legacy_result = (
            result.details.get("result")
            if isinstance(result.details, dict) and isinstance(result.details.get("result"), dict)
            else {}
        )
        response = {**_without_secrets(legacy_result), "authorization_receipt": body}
        response["ok"] = bool(result.ok)
        response["mutated"] = bool(result.mutated)
        response["capability_id"] = result.capability_id
        response["executor_id"] = result.executor_id
        response["error_code"] = result.error_code
        response["runtime_truth"] = _without_secrets(self._runtime_snapshot())
        return bool(result.ok), response

    def route(self, operation: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        spec = SPECS.get(operation)
        if spec is not None and spec.safe_mode_blocked and bool(self.runtime._safe_mode_enabled()):
            return self.preview(operation, payload)
        if payload.get("confirm") is True and not isinstance(payload.get("confirmation"), dict):
            return False, {"ok": False, "error": "boolean_confirmation_not_authorization", "mutated": False}
        if "mutation_plan" in payload or "confirmation" in payload:
            return self.apply(operation, payload)
        return self.preview(operation, payload)

    def _executor(self, spec: DomainMutationSpec) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
        def run(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
            valid, reason, _ = assert_authorized_mutation(
                action.get("trusted_invocation_context"), expected_capability=spec.capability_id,
                expected_executor=spec.executor_id, expected_operation=str(plan.get("plan_id") or ""),
                expected_plan_fingerprint=str(plan.get("plan_fingerprint") or ""),
                expected_target_fingerprint=str(plan.get("target_fingerprint") or ""), runtime_mode="production",
            )
            if not valid:
                return {"ok": False, "mutated": False, "error_code": reason or "generic_bypass_blocked", "user_message": "Direct mutation execution was blocked."}
            params = action.get("parameters") if isinstance(action.get("parameters"), dict) else {}
            ok, body = self._dispatch(spec.operation, params)
            return {
                "ok": bool(ok), "mutated": bool(ok), "executor_id": spec.executor_id,
                "error_code": None if ok else str(body.get("error") or "mutation_failed"),
                "user_message": str(body.get("message") or ("Mutation completed." if ok else "Mutation failed.")),
                "resources_touched": self._resources(spec.operation, self._bound_request(spec.operation, params)),
                "rollback_available": spec.rollback_available, "rollback_hint": spec.rollback_hint,
                "details": {"operation": spec.operation, "result": _without_secrets(body)},
            }
        return run

    def _dispatch(self, operation: str, p: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        r = self.runtime
        provider = str(p.get("provider_id") or p.get("provider") or "").strip().lower()
        model_value = str(p.get("model_id") or p.get("model") or "").strip()
        model_for_runtime = (
            model_value.split(":", 1)[1]
            if provider and model_value.lower().startswith(f"{provider}:")
            else model_value
        )
        model_payload = {**p, "model_id": model_for_runtime, "model": model_for_runtime}
        return {
            "provider.add": lambda: r.add_provider(p),
            "provider.update": lambda: r.update_provider(provider, p),
            "provider.delete": lambda: r.delete_provider(provider),
            "provider.model.add": lambda: r.add_provider_model(provider, p),
            "provider.secret.set": lambda: r.set_provider_secret(provider, p),
            "telegram.secret.set": lambda: r.set_telegram_secret({"bot_token": p.get("bot_token")}),
            "config.update": lambda: r.update_config(p),
            "defaults.update": lambda: r.update_defaults(p),
            "defaults.rollback": lambda: r.rollback_defaults(),
            "model.switch": lambda: r.llm_models_switch({**model_payload, "confirm": True}),
            "model.switch_temporary": lambda: r.llm_models_switch_temporary({**model_payload, "confirm": True}),
            "model.acquire": lambda: r.pull_ollama_model({**p, "confirm": True}),
            "model.policy": lambda: r.llm_models_policy(p),
            "runtime.control_mode": lambda: r.llm_control_mode_set({**p, "confirm": True}),
            "model.refresh": lambda: r.refresh_models(p),
            "model_watch.run": lambda: r.run_model_watch_once(trigger="manual"),
            "model_watch.refresh": lambda: r.model_watch_refresh(p),
            "model_watch.hf_scan": lambda: r.model_watch_hf_scan(trigger="manual"),
            "llm.fix": lambda: r.llm_fixit({**p, "confirm": True}),
            "setup.bootstrap": lambda: r.run_memory_v2_bootstrap(
                source_ref=f"authorized_bootstrap:{str(p.get('actor_id') or 'operator')}",
                promote_semantic=bool(p.get("promote_semantic", True)),
                reason=str(p.get("reason") or "").strip() or None,
            ),
            "llm.autoconfig": lambda: r.llm_autoconfig_apply(p),
            "llm.reconcile": lambda: r.llm_capabilities_reconcile_apply(p, trigger="manual"),
            "llm.hygiene": lambda: r.llm_hygiene_apply(p),
            "llm.cleanup": lambda: r.llm_cleanup_apply(p, trigger="manual"),
            "llm.self_heal": lambda: r.llm_self_heal_apply(p, trigger="manual"),
            "llm.support.remediate": lambda: r.llm_support_remediate_execute({**p, "confirm": True}),
            "llm.registry.rollback": lambda: r.llm_registry_rollback(p),
            "llm.autopilot.undo": lambda: r.llm_autopilot_undo({**p, "confirm": True}),
            "llm.autopilot.unpause": lambda: r.llm_autopilot_unpause({**p, "confirm": True}),
            "llm.autopilot.bootstrap": lambda: r.llm_autopilot_bootstrap({**p, "confirm": True}, trigger="manual"),
            "modelops.execute": lambda: r.modelops_execute({**p, "confirm": True}),
        }[operation]()


__all__ = ["DomainMutationSpec", "ProviderModelAuthorizationService", "SPECS"]
