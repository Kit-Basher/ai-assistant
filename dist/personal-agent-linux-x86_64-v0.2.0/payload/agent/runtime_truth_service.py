from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from copy import deepcopy
from typing import Any
import re

from agent.error_response_ux import compose_actionable_message
from agent.failure_ux import build_failure_recovery
from agent.filesystem_skill import FileSystemSkill
from agent.persona import normalize_persona_text
from agent.onboarding_contract import (
    detect_onboarding_state,
    onboarding_next_action,
    onboarding_steps,
    onboarding_summary,
)
from agent.llm.approved_local_models import approved_local_profile_for_ref
from agent.llm.control_contract import normalize_task_request
from agent.llm.default_model_policy import choose_best_default_chat_candidate
from agent.llm.model_discovery import (
    allowed_model_discovery_proposal_kinds,
    allowed_model_discovery_proposal_sources,
    build_model_discovery_proposals,
)
from agent.llm.model_discovery_external import load_external_model_discovery_rows
from agent.llm.model_discovery_manager import ModelDiscoveryManager
from agent.llm.model_latency import infer_parameter_size_b, resolve_speed_class
from agent.llm.model_inventory import build_model_inventory
from agent.llm.model_manager import (
    build_model_lifecycle_rows,
    load_model_manager_state,
    model_manager_state_path_for_runtime,
)
from agent.llm.registry import _default_registry_document, parse_registry_document
from agent.recovery_contract import detect_recovery_mode, recovery_next_action, recovery_summary
from agent.runtime_lifecycle import RuntimeLifecyclePhase
from agent.shell_skill import ShellSkill


class RuntimeTruthService:
    """Deterministic adapter over canonical runtime state and setup actions.

    This service must never call the LLM. It reads runtime state from AgentRuntime
    and exposes structured data/actions that higher-level agent layers can use.
    """

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    _SNAPSHOT_CACHE_TTL_SECONDS = 2.0

    def _snapshot_cache(self) -> dict[str, dict[str, Any]]:
        cache = getattr(self, "_snapshot_cache_store", None)
        if not isinstance(cache, dict):
            cache = {}
            self._snapshot_cache_store = cache
        return cache

    def _cached_snapshot(self, key: str, builder: Any) -> Any:
        cache = self._snapshot_cache()
        now = time.monotonic()
        entry = cache.get(key)
        if isinstance(entry, dict):
            created_at = float(entry.get("created_at") or 0.0)
            if created_at and now - created_at <= self._SNAPSHOT_CACHE_TTL_SECONDS:
                return deepcopy(entry.get("value"))
        value = builder()
        cache[key] = {"created_at": now, "value": deepcopy(value)}
        return deepcopy(value)

    def _timed_cached_snapshot(self, key: str, builder: Any) -> tuple[Any, int]:
        cache = self._snapshot_cache()
        now = time.monotonic()
        entry = cache.get(key)
        if isinstance(entry, dict):
            created_at = float(entry.get("created_at") or 0.0)
            if created_at and now - created_at <= self._SNAPSHOT_CACHE_TTL_SECONDS:
                value = entry.get("value")
                return deepcopy(value), 0
        started = time.monotonic()
        value = builder()
        elapsed_ms = int(max(0.0, time.monotonic() - started) * 1000)
        cache[key] = {"created_at": time.monotonic(), "value": deepcopy(value)}
        return deepcopy(value), elapsed_ms

    def _invalidate_snapshot_cache(self) -> None:
        cache = getattr(self, "_snapshot_cache_store", None)
        if isinstance(cache, dict):
            cache.clear()

    def _filesystem_allowed_roots(self) -> list[str]:
        config = getattr(self.runtime, "config", None)
        configured = list(getattr(config, "perception_roots", ()) or [])
        repo_root = str(getattr(self.runtime, "_repo_root", "") or "").strip()
        if repo_root:
            configured.append(repo_root)
        return [item for item in configured if str(item).strip()]

    def _filesystem_skill(self) -> FileSystemSkill:
        cached = getattr(self, "_filesystem_skill_cache", None)
        if isinstance(cached, FileSystemSkill):
            return cached
        base_dir = str(getattr(self.runtime, "_repo_root", "") or "").strip() or os.getcwd()
        skill = FileSystemSkill(
            allowed_roots=self._filesystem_allowed_roots(),
            base_dir=base_dir,
        )
        self._filesystem_skill_cache = skill
        return skill

    def _shell_skill(self) -> ShellSkill:
        cached = getattr(self, "_shell_skill_cache", None)
        if isinstance(cached, ShellSkill):
            return cached
        base_dir = str(getattr(self.runtime, "_repo_root", "") or "").strip() or os.getcwd()
        skill = ShellSkill(
            allowed_roots=self._filesystem_allowed_roots(),
            base_dir=base_dir,
        )
        self._shell_skill_cache = skill
        return skill

    def _model_discovery_manager(self) -> ModelDiscoveryManager:
        cached = getattr(self, "_model_discovery_manager_cache", None)
        if isinstance(cached, ModelDiscoveryManager):
            return cached
        secret_store = getattr(self.runtime, "secret_store", None)
        secret_lookup = getattr(secret_store, "get_secret", None)
        manager = ModelDiscoveryManager(
            runtime=self.runtime,
            secret_lookup=secret_lookup if callable(secret_lookup) else None,
        )
        self._model_discovery_manager_cache = manager
        return manager

    def _defaults_snapshot(self) -> dict[str, Any]:
        defaults = self.runtime.get_defaults() if callable(getattr(self.runtime, "get_defaults", None)) else {}
        return dict(defaults) if isinstance(defaults, dict) else {}

    def _health_state_snapshot(self) -> dict[str, Any]:
        health_monitor = getattr(self.runtime, "_health_monitor", None)
        state = health_monitor.state if isinstance(getattr(health_monitor, "state", None), dict) else {}
        return dict(state) if isinstance(state, dict) else {}

    def _router_snapshot(self) -> dict[str, Any]:
        cache = self._snapshot_cache().get("router_snapshot")
        if isinstance(cache, dict):
            created_at = float(cache.get("created_at") or 0.0)
            if created_at and time.monotonic() - created_at <= self._SNAPSHOT_CACHE_TTL_SECONDS:
                value = cache.get("value")
                return dict(value) if isinstance(value, dict) else {}
        router = getattr(self.runtime, "_router", None)
        snapshot_reader = getattr(router, "doctor_snapshot", None)
        if not callable(snapshot_reader):
            return {}
        try:
            snapshot = snapshot_reader()
        except Exception:
            return {}
        payload = dict(snapshot) if isinstance(snapshot, dict) else {}
        self._snapshot_cache()["router_snapshot"] = {
            "created_at": time.monotonic(),
            "value": dict(payload),
        }
        return dict(payload)

    def _router_model_status(
        self,
        model_id: str | None,
        *,
        router_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model_key = str(model_id or "").strip()
        if not model_key:
            return {}
        snapshot = router_snapshot if isinstance(router_snapshot, dict) else self._router_snapshot()
        rows = snapshot.get("models")
        if not isinstance(rows, list):
            return {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("id") or "").strip() != model_key:
                continue
            health = row.get("health") if isinstance(row.get("health"), dict) else {}
            status = str(health.get("status") or row.get("status") or "").strip().lower() or None
            return {
                "status": status,
                "routable": bool(row.get("routable", False)),
                "available": row.get("available"),
            }
        return {}

    def _provider_disabled_by_runtime(
        self,
        provider_id: str | None,
        *,
        router_snapshot: dict[str, Any] | None = None,
    ) -> bool:
        provider_key = str(provider_id or "").strip().lower()
        if not provider_key:
            return False
        registry_document = (
            self.runtime.registry_document
            if isinstance(getattr(self.runtime, "registry_document", None), dict)
            else {}
        )
        providers_doc = (
            registry_document.get("providers")
            if isinstance(registry_document.get("providers"), dict)
            else {}
        )
        provider_payload = providers_doc.get(provider_key) if isinstance(providers_doc.get(provider_key), dict) else None
        if isinstance(provider_payload, dict) and not bool(provider_payload.get("enabled", True)):
            return True
        snapshot = router_snapshot if isinstance(router_snapshot, dict) else self._router_snapshot()
        provider_rows = snapshot.get("providers") if isinstance(snapshot.get("providers"), list) else []
        for row in provider_rows:
            if not isinstance(row, dict):
                continue
            row_provider_id = str(row.get("id") or "").strip().lower()
            if row_provider_id != provider_key:
                continue
            if not bool(row.get("enabled", True)):
                return True
            health = row.get("health") if isinstance(row.get("health"), dict) else {}
            error_kind = str(health.get("last_error_kind") or "").strip().lower()
            if error_kind == "provider_disabled":
                return True
        return False

    def _provider_health_row(
        self,
        provider_id: str | None,
        *,
        router_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provider_key = str(provider_id or "").strip().lower()
        if not provider_key:
            return {}
        providers = self._health_state_snapshot().get("providers")
        if not isinstance(providers, dict):
            return {}
        row = providers.get(provider_key)
        payload = dict(row) if isinstance(row, dict) else {}
        effective_provider_health = getattr(self.runtime, "_effective_provider_health", None)
        if callable(effective_provider_health):
            try:
                effective = effective_provider_health(provider_key, payload)
            except Exception:
                effective = payload
            if isinstance(effective, dict):
                payload = dict(effective)
        if self._provider_disabled_by_runtime(provider_key, router_snapshot=router_snapshot):
            payload = dict(payload)
            payload["status"] = "down"
            payload["last_error_kind"] = "provider_disabled"
        return payload

    def _model_health_row(
        self,
        model_id: str | None,
        *,
        router_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model_key = str(model_id or "").strip()
        if not model_key:
            return {}
        models = self._health_state_snapshot().get("models")
        row = models.get(model_key) if isinstance(models, dict) else None
        payload = dict(row) if isinstance(row, dict) else {}
        registry_document = (
            self.runtime.registry_document
            if isinstance(getattr(self.runtime, "registry_document", None), dict)
            else {}
        )
        models_doc = (
            registry_document.get("models")
            if isinstance(registry_document.get("models"), dict)
            else {}
        )
        model_payload = models_doc.get(model_key) if isinstance(models_doc.get(model_key), dict) else {}
        provider_id = str(model_payload.get("provider") or "").strip().lower()
        if not provider_id and ":" in model_key:
            provider_id = model_key.split(":", 1)[0].strip().lower()
        if provider_id and self._provider_disabled_by_runtime(provider_id, router_snapshot=router_snapshot):
            payload = dict(payload)
            payload["status"] = "down"
            payload["last_error_kind"] = "provider_disabled"
        return payload

    @staticmethod
    def _health_status(row: dict[str, Any] | None, default: str = "unknown") -> str:
        status = str((row or {}).get("status") or "").strip().lower()
        return status or default

    @staticmethod
    def _health_reason(row: dict[str, Any] | None) -> str | None:
        payload = dict(row) if isinstance(row, dict) else {}
        for key in ("reason", "message", "detail", "error", "error_kind", "last_error", "failure_reason"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
        return None

    def _model_manager_state(self) -> dict[str, Any]:
        cache = self._snapshot_cache().get("model_manager_state")
        if isinstance(cache, dict):
            created_at = float(cache.get("created_at") or 0.0)
            if created_at and time.monotonic() - created_at <= self._SNAPSHOT_CACHE_TTL_SECONDS:
                value = cache.get("value")
                return dict(value) if isinstance(value, dict) else {"schema_version": 1, "targets": {}}
        try:
            path = model_manager_state_path_for_runtime(self.runtime)
            payload = load_model_manager_state(path)
        except Exception:
            payload = {"schema_version": 1, "targets": {}}
        normalized = dict(payload) if isinstance(payload, dict) else {"schema_version": 1, "targets": {}}
        self._snapshot_cache()["model_manager_state"] = {
            "created_at": time.monotonic(),
            "value": dict(normalized),
        }
        return dict(normalized)

    def _policy_flags(self) -> dict[str, Any]:
        runtime_policy_reader = getattr(self.runtime, "_chat_control_policy", None)
        if callable(runtime_policy_reader):
            try:
                runtime_policy = runtime_policy_reader()
            except Exception:
                runtime_policy = {}
            if isinstance(runtime_policy, dict) and runtime_policy:
                return dict(runtime_policy)
        defaults = self._defaults_snapshot()
        safe_mode = bool(
            callable(getattr(self.runtime, "_safe_mode_enabled", None)) and self.runtime._safe_mode_enabled()
        )
        allow_remote_fallback = bool(defaults.get("allow_remote_fallback", True))
        if safe_mode:
            allow_remote_fallback = False
        allow_remote_recommendation = not safe_mode
        mode = "safe" if safe_mode else "controlled"
        return {
            "mode": mode,
            "mode_label": "SAFE MODE" if safe_mode else "Controlled Mode",
            "safe_mode": safe_mode,
            "mode_source": "config_default",
            "config_default_mode": mode,
            "config_default_mode_label": "SAFE MODE" if safe_mode else "Controlled Mode",
            "override_mode": None,
            "override_active": False,
            "allow_remote_fallback": allow_remote_fallback,
            "allow_remote_recommendation": allow_remote_recommendation,
            "allow_remote_switch": not safe_mode,
            "allow_install_pull": not safe_mode,
            "scout_advisory_only": safe_mode,
            "approval_required_actions": [
                "test_model",
                "switch_temporarily",
                "make_default",
                "acquire_model",
            ],
            "forbidden_actions": (
                [
                    "remote_switch",
                    "install_download_import",
                ]
                if safe_mode
                else []
            ),
            "transition": {
                "endpoint": "/llm/control_mode",
                "confirm_required": True,
                "loopback_only": True,
                "can_enter_controlled_mode": mode != "controlled",
                "can_enter_safe_mode": mode != "safe",
                "can_clear_override": False,
            },
        }

    @staticmethod
    def _health_label(provider_status: str, model_status: str) -> str:
        provider_state = str(provider_status or "").strip().lower() or "unknown"
        model_state = str(model_status or "").strip().lower() or "unknown"
        if provider_state == "ok" and model_state == "ok":
            return "up"
        if provider_state in {"down", "degraded"}:
            return provider_state
        if model_state in {"down", "degraded"}:
            return model_state
        return "unknown"

    @staticmethod
    def _runtime_failure_recovery(
        *,
        ready: dict[str, Any],
        llm_status: dict[str, Any],
        normalized_status: dict[str, Any],
        runtime_mode: str,
        startup_phase: str,
    ) -> dict[str, Any] | None:
        if str(runtime_mode or "").strip().lower() == "ready":
            return None
        failure_code = (
            str(normalized_status.get("failure_code") or "").strip().lower()
            or str(ready.get("failure_code") or "").strip().lower()
            or str(ready.get("llm_reason") or "").strip().lower()
        )
        provider_health = llm_status.get("active_provider_health") if isinstance(llm_status.get("active_provider_health"), dict) else {}
        model_health = llm_status.get("active_model_health") if isinstance(llm_status.get("active_model_health"), dict) else {}
        if str(startup_phase or "").strip().lower() in {"starting", "listening", "warming"}:
            return build_failure_recovery(
                "runtime_initializing",
                current_state=startup_phase,
                details=failure_code or None,
            )
        if failure_code in {"lock_unavailable", "lock_path_unavailable", "telegram_conflict", "lock_conflict"}:
            return build_failure_recovery("db_busy", current_state=runtime_mode, details=failure_code or None)
        if failure_code in {"telegram_token_missing", "missing_token", "telegram_token_invalid", "token_invalid"}:
            return build_failure_recovery("confirm_token_expired", current_state=runtime_mode, details=failure_code or None)
        if failure_code in {"llm_unavailable", "provider_unhealthy", "model_unhealthy", "no_chat_model", "router_unavailable"}:
            return build_failure_recovery(
                "runtime_degraded",
                current_state=runtime_mode,
                details=failure_code or None,
                blocker=str(ready.get("next_action") or "").strip() or None,
            )
        provider_status = str(provider_health.get("status") or "").strip().lower()
        model_status = str(model_health.get("status") or "").strip().lower()
        if provider_status in {"down", "degraded"} or model_status in {"down", "degraded"}:
            return build_failure_recovery(
                "dependency_unavailable",
                subject="active provider" if provider_status in {"down", "degraded"} else "active model",
                reason="The active runtime dependency is not healthy.",
                current_state=runtime_mode,
                details=failure_code or None,
            )
        if str(runtime_mode or "").strip().lower() == "degraded":
            return build_failure_recovery("runtime_degraded", current_state=runtime_mode, details=failure_code or None)
        if str(runtime_mode or "").strip().lower() == "blocked":
            return build_failure_recovery("runtime_blocked", current_state=runtime_mode, details=failure_code or None)
        return build_failure_recovery("runtime_not_ready", current_state=runtime_mode, details=failure_code or None)

    def ready_status(self) -> dict[str, Any]:
        observability = self.runtime._runtime_observability_context()
        safe_mode_target = self.runtime.safe_mode_target_status()
        telegram = observability["telegram"] if isinstance(observability.get("telegram"), dict) else {}
        telegram_state = str(observability.get("telegram_state") or "stopped")
        telegram_enabled = bool(observability.get("telegram_enabled", False))
        llm_status = observability["llm_status"] if isinstance(observability.get("llm_status"), dict) else {}
        canonical_llm_runtime_status = (
            observability["canonical_llm_runtime_status"]
            if isinstance(observability.get("canonical_llm_runtime_status"), dict)
            else {}
        )
        hf_status = self.runtime._model_watch_hf_status_snapshot()
        startup_phase = str(observability.get("startup_phase") or "starting").strip().lower() or "starting"
        phase = (
            str(observability.get("phase") or RuntimeLifecyclePhase.READY.value).strip().lower()
            or RuntimeLifecyclePhase.READY.value
        )
        warmup_remaining = list(observability.get("warmup_remaining") or [])
        normalized_status = (
            observability["normalized_status"] if isinstance(observability.get("normalized_status"), dict) else {}
        )
        ready = bool(observability.get("ready", False))
        setup_context_ready_payload: dict[str, Any] = {
            "ok": True,
            "ready": bool(ready),
            "phase": phase,
            "startup_phase": startup_phase,
            "failure_code": str(normalized_status.get("failure_code") or "").strip() or None,
            "runtime_mode": str(normalized_status.get("runtime_mode") or "DEGRADED"),
            "runtime_status": normalized_status,
            "telegram": {
                "enabled": telegram_enabled,
                "configured": bool(telegram.get("configured", False)),
                "state": telegram_state,
                "effective_state": str(telegram.get("effective_state") or "unknown"),
            },
        }
        onboarding_state = detect_onboarding_state(
            ready_payload=setup_context_ready_payload,
            llm_status=llm_status,
        )
        recovery_mode = detect_recovery_mode(
            ready_payload=setup_context_ready_payload,
            llm_status=llm_status,
            failure_code=str(normalized_status.get("failure_code") or "").strip() or None,
            api_reachable=True,
        )
        onboarding_next = onboarding_next_action(
            onboarding_state,
            ready_payload=setup_context_ready_payload,
        )
        if str(onboarding_state).strip().upper() == "DEGRADED":
            onboarding_next = recovery_next_action(recovery_mode)
        onboarding_payload = {
            "state": onboarding_state,
            "summary": onboarding_summary(
                onboarding_state,
                ready_payload=setup_context_ready_payload,
                llm_status=llm_status,
            ),
            "next_action": onboarding_next,
            "steps": onboarding_steps(onboarding_state),
        }
        recovery_payload = {
            "mode": recovery_mode,
            "summary": recovery_summary(recovery_mode),
            "next_action": recovery_next_action(recovery_mode),
        }
        failure_recovery = self._runtime_failure_recovery(
            ready=setup_context_ready_payload,
            llm_status=llm_status if isinstance(llm_status, dict) else {},
            normalized_status=normalized_status if isinstance(normalized_status, dict) else {},
            runtime_mode=str(normalized_status.get("runtime_mode") or "DEGRADED"),
            startup_phase=startup_phase,
        )
        chat_usable = bool(ready)
        llm_available = getattr(self.runtime, "llm_available", None)
        if not chat_usable and callable(llm_available):
            try:
                chat_usable = bool(llm_available())
            except Exception:
                chat_usable = bool(ready)
        uptime_seconds = max(0, int((datetime.now(timezone.utc) - self.runtime.started_at).total_seconds()))
        recent_messages = self.runtime._ready_recent_telegram_messages(limit=5)
        runtime_state_label = str(failure_recovery.get("state_label") or "").strip() if isinstance(failure_recovery, dict) else ""
        runtime_reason = str(failure_recovery.get("reason") or "").strip() if isinstance(failure_recovery, dict) else ""
        runtime_blocker = str(failure_recovery.get("blocker") or "").strip() if isinstance(failure_recovery, dict) else ""
        runtime_next_step = str(failure_recovery.get("next_step") or "").strip() if isinstance(failure_recovery, dict) else ""
        message = self.runtime._runtime_surface_message(
            startup_phase=startup_phase,
            normalized_status=normalized_status,
            onboarding_summary=None if str(onboarding_state).strip().upper() == "READY" else str(onboarding_payload.get("summary") or ""),
            onboarding_next_action=None if str(onboarding_state).strip().upper() == "READY" else str(onboarding_payload.get("next_action") or ""),
            safe_mode_target=safe_mode_target,
            failure_recovery=failure_recovery,
        )
        return {
            "ok": True,
            "ready": bool(ready),
            "chat_usable": bool(chat_usable),
            "phase": phase,
            "startup_phase": startup_phase,
            "runtime_mode": str(normalized_status.get("runtime_mode") or "DEGRADED"),
            "next_action": normalized_status.get("next_action"),
            "state_label": runtime_state_label or ("Ready" if chat_usable or ready else "Not ready"),
            "reason": runtime_reason or None,
            "blocker": runtime_blocker or None,
            "next_step": runtime_next_step or None,
            "runtime_status": normalized_status,
            "onboarding": onboarding_payload,
            "recovery": recovery_payload,
            "failure_recovery": failure_recovery,
            "warmup_remaining": list(warmup_remaining),
            "last_error": str(self.runtime._startup_last_error or "") or None,
            "api": {
                "version": self.runtime.version,
                "git_commit": self.runtime.git_commit,
                "pid": self.runtime.pid,
                "started_at": self.runtime.started_at_iso,
                "uptime_seconds": uptime_seconds,
            },
            "telegram": {
                **telegram,
                "status": telegram_state,
                "recent_messages": recent_messages,
            },
            "model_watch": {
                "hf": hf_status,
            },
            "safe_mode_target": safe_mode_target,
            "llm": {
                "provider": canonical_llm_runtime_status.get("provider"),
                "model": canonical_llm_runtime_status.get("model"),
                "local_remote": canonical_llm_runtime_status.get("local_remote"),
                "known": bool(canonical_llm_runtime_status.get("identity_known", False)),
                "reason": canonical_llm_runtime_status.get("identity_reason"),
                "runtime_status": canonical_llm_runtime_status,
                "active_provider_health": llm_status.get("active_provider_health") if isinstance(llm_status.get("active_provider_health"), dict) else {},
                "active_model_health": llm_status.get("active_model_health") if isinstance(llm_status.get("active_model_health"), dict) else {},
                "default_provider": llm_status.get("default_provider"),
                "resolved_default_model": llm_status.get("resolved_default_model"),
                "default_model": llm_status.get("default_model"),
                "allow_remote_fallback": bool(llm_status.get("allow_remote_fallback", True)),
                "policy": llm_status.get("policy") if isinstance(llm_status.get("policy"), dict) else {},
            },
            "message": message,
        }

    def ui_state(self, *, ready_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        ready = dict(ready_payload) if isinstance(ready_payload, dict) else self.ready_status()

        try:
            def _norm_text(value: Any) -> str:
                return str(value or "").strip()

            ready_runtime_status = (
                ready.get("runtime_status") if isinstance(ready.get("runtime_status"), dict) else {}
            )
            truth = self.runtime.runtime_truth_service()
            current_target = (
                truth.current_chat_target_status()
                if callable(getattr(truth, "current_chat_target_status", None))
                else {}
            )
            defaults = self.runtime.get_defaults()
            control_mode = self.runtime.llm_control_mode_status()
            model_provider = (
                str(current_target.get("effective_provider") or current_target.get("provider") or "").strip().lower()
                or str(defaults.get("default_provider") or "").strip().lower()
                or None
            )
            model_id = (
                str(current_target.get("effective_model") or current_target.get("model") or "").strip()
                or str(defaults.get("resolved_default_model") or defaults.get("default_model") or "").strip()
                or None
            )
            provider_health = str(
                current_target.get("effective_provider_health_status")
                or current_target.get("provider_health_status")
                or ""
            ).strip().lower()
            model_health = str(
                current_target.get("effective_model_health_status")
                or current_target.get("health_status")
                or ""
            ).strip().lower()
            health = self._health_label(provider_health, model_health)
            runtime_mode = (
                str(ready_runtime_status.get("runtime_mode") or ready.get("runtime_mode") or "DEGRADED")
                .strip()
                .lower()
                or "degraded"
            )
            recovery_row = ready.get("failure_recovery") if isinstance(ready.get("failure_recovery"), dict) else {}
            summary = normalize_persona_text(
                str(ready.get("message") or ready_runtime_status.get("summary") or "Ready.").strip()
            )
            if runtime_mode != "ready" and isinstance(recovery_row, dict) and str(recovery_row.get("message") or "").strip():
                summary = normalize_persona_text(str(recovery_row.get("message") or "").strip())
            state_label = (
                str(ready.get("state_label") or ready_runtime_status.get("state_label") or "").strip()
                or str(recovery_row.get("state_label") or "").strip()
                or ("Ready" if runtime_mode == "ready" else "Not ready")
            )
            if bool(ready.get("chat_usable", ready.get("ready", False))) and state_label != "Ready":
                state_label = "Ready"
            reason = (
                str(ready.get("reason") or ready_runtime_status.get("reason") or "").strip()
                or str(recovery_row.get("reason") or "").strip()
                or None
            )
            next_step = (
                str(ready.get("next_step") or ready_runtime_status.get("next_step") or "").strip()
                or str(recovery_row.get("next_step") or "").strip()
                or None
            )
            blocked_reason = None
            if runtime_mode != "ready":
                blocked_reason = (
                    _norm_text(recovery_row.get("blocker"))
                    or _norm_text(ready_runtime_status.get("failure_code"))
                    or None
                )
            safe_mode_target = ready.get("safe_mode_target") if isinstance(ready.get("safe_mode_target"), dict) else {}
            if not blocked_reason and isinstance(safe_mode_target, dict) and bool(safe_mode_target.get("enabled", False)):
                if safe_mode_target.get("configured_valid") is False:
                    blocked_reason = (
                        _norm_text(safe_mode_target.get("reason"))
                        or _norm_text(safe_mode_target.get("message"))
                        or None
                    )
            if not blocked_reason:
                blocked_payload = ready.get("blocked") if isinstance(ready.get("blocked"), dict) else {}
                if isinstance(blocked_payload, dict) and bool(blocked_payload.get("blocked", False)):
                    blocked_reason = (
                        _norm_text(blocked_payload.get("reason"))
                        or _norm_text(blocked_payload.get("kind"))
                        or None
                    )
            failure_recovery = self._runtime_failure_recovery(
                ready=ready,
                llm_status=ready.get("llm") if isinstance(ready.get("llm"), dict) else {},
                normalized_status=ready_runtime_status if isinstance(ready_runtime_status, dict) else {},
                runtime_mode=runtime_mode,
                startup_phase=str(ready.get("startup_phase") or ready.get("phase") or "starting"),
            )
            next_step = normalize_persona_text(
                str(
                    ready_runtime_status.get("next_action")
                    or (
                        ready.get("onboarding", {}).get("next_action")
                        if isinstance(ready.get("onboarding"), dict)
                        else ""
                    )
                    or (
                        ready.get("recovery", {}).get("next_action")
                        if isinstance(ready.get("recovery"), dict)
                        else ""
                    )
                    or ""
                ).strip()
            ) or None
            memory_db = None
            try:
                memory_db = self.runtime._ensure_memory_db()
            except Exception:
                memory_db = None
            show_conf_pref = "on"
            response_style = "concise"
            if memory_db is not None:
                show_conf_pref = (memory_db.get_preference("show_confidence") or "on").strip().lower()
                response_style = _norm_text(memory_db.get_preference("response_style")) or "concise"
            return {
            "ok": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "chat_usable": bool(ready.get("chat_usable", ready.get("ready", False))),
            "runtime": {
                "status": runtime_mode,
                "state_label": state_label,
                "summary": summary,
                "reason": reason,
                "next_action": next_step,
                "next_step": next_step,
                "blocker": blocked_reason,
                "recovery": recovery_row if recovery_row else ready.get("failure_recovery"),
                "chat_usable": bool(ready.get("chat_usable", ready.get("ready", False))),
            },
                "model": {
                    "provider": model_provider,
                    "model": model_id,
                    "path": f"{model_provider} / {model_id}" if model_provider and model_id else None,
                    "routing_mode": str(defaults.get("routing_mode") or self.runtime.config.llm_routing_mode or "auto")
                    .strip()
                    .lower()
                    or "auto",
                    "health": health,
                    "reason": str(
                        current_target.get("health_reason")
                        or current_target.get("qualification_reason")
                        or current_target.get("degraded_reason")
                        or ""
                    ).strip() or None,
                    "health_reason": str(current_target.get("health_reason") or "").strip() or None,
                    "qualification_reason": str(current_target.get("qualification_reason") or "").strip() or None,
                    "degraded_reason": str(current_target.get("degraded_reason") or "").strip() or None,
                },
                "conversation": {
                    "topic": None,
                    "recent_request": None,
                    "open_loop": None,
                },
                "action": {
                    "pending_approval": False,
                    "blocked_reason": blocked_reason,
                    "next_step": next_step,
                    "last_action": None,
                },
                "failure_recovery": failure_recovery,
                "signals": {
                    "response_style": response_style,
                    "confidence_visible": show_conf_pref not in {"off", "false", "0", "no"},
                },
                "source": "ready+runtime_truth+defaults+preferences",
            }
        except Exception as exc:  # pragma: no cover - defensive UI state fallback
            return {
                "ok": False,
                "error": exc.__class__.__name__,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "runtime": {
                    "status": "unknown",
                    "summary": "I couldn’t read the live state snapshot right now.",
                    "next_action": None,
                },
                "model": {
                    "provider": None,
                    "model": None,
                    "path": None,
                    "routing_mode": "auto",
                    "health": "unknown",
                },
                "conversation": {
                    "topic": None,
                    "recent_request": None,
                    "open_loop": None,
                },
                "action": {
                    "pending_approval": False,
                    "blocked_reason": "state_snapshot_unavailable",
                    "last_action": None,
                },
                "failure_recovery": {
                    "kind": "unknown",
                    "status": "unknown",
                    "state_label": "Unknown",
                    "summary": "I couldn’t read the live state snapshot right now.",
                    "reason": "The runtime state snapshot failed to load.",
                    "next_step": "Try again after the runtime finishes starting.",
                    "retryable": True,
                    "recoverability": "retryable",
                    "blocker": "state_snapshot_unavailable",
                    "current_state": "unknown",
                    "details": exc.__class__.__name__,
                    "message": "I couldn’t read the live state snapshot right now. The runtime state snapshot failed to load. Next: Try again after the runtime finishes starting.",
                },
                "signals": {
                    "response_style": None,
                    "confidence_visible": None,
                },
                "source": "fallback",
            }

    def _hardware_capacity_snapshot(self) -> dict[str, Any]:
        cached = getattr(self, "_hardware_capacity_cache", None)
        if isinstance(cached, dict) and cached:
            return dict(cached)

        memory_total_bytes = 0
        open_db = getattr(self.runtime, "_open_perception_db", None)
        if callable(open_db):
            db = None
            try:
                db = open_db()
                latest_reader = getattr(db, "get_latest_metrics_snapshot", None)
                latest = latest_reader() if callable(latest_reader) else None
                if isinstance(latest, dict):
                    memory_total_bytes = max(
                        0,
                        int(latest.get("mem_used") or 0) + int(latest.get("mem_available") or 0),
                    )
            except Exception:
                memory_total_bytes = 0
            finally:
                close_db = getattr(db, "close", None)
                if callable(close_db):
                    try:
                        close_db()
                    except Exception:
                        pass

        payload = {
            "memory_total_bytes": int(memory_total_bytes),
            "memory_total_gb": (
                round(float(memory_total_bytes) / float(1024 ** 3), 3) if memory_total_bytes > 0 else None
            ),
        }
        self._hardware_capacity_cache = dict(payload)
        return payload

    @staticmethod
    def _local_model_min_memory_gb(row: dict[str, Any], install_profile: dict[str, Any] | None) -> int | None:
        if isinstance(install_profile, dict):
            value = int(install_profile.get("min_memory_gb") or 0)
            if value > 0:
                return value
        size_b = infer_parameter_size_b(
            row.get("model_id"),
            row.get("model_name"),
            row.get("size"),
        )
        if size_b is None or size_b <= 0.0:
            return None
        if size_b <= 3.5:
            return 8
        if size_b <= 7.5:
            return 12
        if size_b <= 13.0:
            return 16
        return 24

    def _local_model_fit(self, row: dict[str, Any], install_profile: dict[str, Any] | None) -> dict[str, Any]:
        if not bool(row.get("local", False)):
            return {
                "comfortable_local": False,
                "local_fit_state": "not_applicable",
                "local_fit_reason": None,
                "local_fit_margin_gb": None,
                "min_memory_gb": None,
            }

        min_memory_gb = self._local_model_min_memory_gb(row, install_profile)
        hardware = self._hardware_capacity_snapshot()
        memory_total_gb = hardware.get("memory_total_gb")
        speed_class = resolve_speed_class(
            model_id=row.get("model_id"),
            model_name=row.get("model_name"),
            size_label=row.get("size"),
        )

        if min_memory_gb is None:
            state = "unknown"
            reason = "hardware fit is unknown for this local model"
            comfortable = speed_class != "slow"
            margin_gb = None
        elif memory_total_gb is None:
            state = "unknown"
            reason = f"this local model usually wants about {min_memory_gb} GiB, but hardware memory is unknown"
            comfortable = speed_class != "slow"
            margin_gb = None
        else:
            margin_gb = round(float(memory_total_gb) - float(min_memory_gb), 3)
            if margin_gb >= 4.0 and speed_class != "slow":
                state = "comfortable"
                reason = f"it fits comfortably on this machine with about {margin_gb:.1f} GiB of headroom"
                comfortable = True
            elif margin_gb >= 0.0:
                state = "tight"
                reason = f"it should fit, but only with about {margin_gb:.1f} GiB of memory headroom"
                comfortable = False
            else:
                state = "memory_starved"
                reason = f"it likely wants about {min_memory_gb} GiB, which is beyond this machine's comfortable memory budget"
                comfortable = False

        return {
            "comfortable_local": bool(comfortable),
            "local_fit_state": state,
            "local_fit_reason": reason,
            "local_fit_margin_gb": margin_gb,
            "min_memory_gb": int(min_memory_gb) if min_memory_gb is not None else None,
        }

    @staticmethod
    def _normalize_chat_task_request(task_request: dict[str, Any] | None) -> dict[str, Any]:
        normalized = normalize_task_request(task_request or {})
        requirements = [
            str(item).strip().lower()
            for item in (normalized.get("requirements") or [])
            if str(item).strip()
        ]
        if "chat" not in requirements:
            requirements.append("chat")
        return {
            "task_type": str(normalized.get("task_type") or "chat").strip().lower() or "chat",
            "requirements": sorted(set(requirements)),
            "preferred_local": bool(normalized.get("preferred_local", True)),
        }

    def _runtime_inventory_rows(self) -> list[dict[str, Any]]:
        cache = self._snapshot_cache().get("runtime_inventory_rows")
        if isinstance(cache, dict):
            created_at = float(cache.get("created_at") or 0.0)
            if created_at and time.monotonic() - created_at <= self._SNAPSHOT_CACHE_TTL_SECONDS:
                value = cache.get("value")
                return [dict(row) for row in value] if isinstance(value, list) else []
        document = (
            self.runtime.registry_document
            if isinstance(getattr(self.runtime, "registry_document", None), dict)
            else {}
        )
        default_document = _default_registry_document()
        models_doc = document.get("models") if isinstance(document.get("models"), dict) else {}
        defaults_doc = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
        referenced_model_ids = {
            str(item).strip()
            for item in (
                defaults_doc.get("chat_model"),
                defaults_doc.get("default_model"),
                defaults_doc.get("embed_model"),
                defaults_doc.get("last_chat_model"),
            )
            if str(item or "").strip()
        }
        baseline_models = (
            default_document.get("models")
            if isinstance(default_document.get("models"), dict)
            else {}
        )
        runtime_defaults = self._defaults_snapshot()
        referenced_model_ids.update(
            {
                model_id
                for model_id in (
                    str(runtime_defaults.get("resolved_default_model") or "").strip(),
                    str(runtime_defaults.get("chat_model") or "").strip(),
                    str(runtime_defaults.get("default_model") or "").strip(),
                    str(runtime_defaults.get("embed_model") or "").strip(),
                    str(runtime_defaults.get("last_chat_model") or "").strip(),
                )
                if model_id
                and isinstance(models_doc.get(model_id), dict)
                and (
                    not isinstance(baseline_models.get(model_id), dict)
                    or dict(models_doc.get(model_id) or {}) != dict(baseline_models.get(model_id) or {})
                )
            }
        )
        manager_state = self._model_manager_state()
        registered_model_ids: set[str] = set()
        for model_id, payload in models_doc.items():
            normalized_model_id = str(model_id).strip()
            if not normalized_model_id or not isinstance(payload, dict):
                continue
            baseline_payload = baseline_models.get(normalized_model_id)
            if normalized_model_id in referenced_model_ids:
                registered_model_ids.add(normalized_model_id)
                continue
            if not isinstance(baseline_payload, dict) or dict(payload) != dict(baseline_payload):
                registered_model_ids.add(normalized_model_id)
        manager_target_ids = {
            str(row.get("model_id") or "").strip()
            for row in (
                (
                    manager_state.get("targets")
                    if isinstance(manager_state.get("targets"), dict)
                    else {}
                ).values()
            )
            if isinstance(row, dict) and str(row.get("model_id") or "").strip()
        }
        try:
            registry = parse_registry_document(
                document,
                path=getattr(self.runtime.config, "llm_registry_path", None),
            )
        except Exception:
            router = getattr(self.runtime, "_router", None)
            registry = getattr(router, "registry", None)
        if registry is None:
            return []
        try:
            rows = build_model_inventory(
                config=self.runtime.config,
                registry=registry,
                router_snapshot=self._router_snapshot(),
                # Canonical runtime truth should not discover extra local artifacts here.
                discovered_local_models=(),
            )
        except Exception:
            return []
        canonical_ids = registered_model_ids | manager_target_ids
        if not canonical_ids:
            return []
        filtered_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id") or "").strip()
            if not model_id:
                continue
            if model_id not in canonical_ids:
                continue
            filtered_rows.append(dict(row))
        self._snapshot_cache()["runtime_inventory_rows"] = {
            "created_at": time.monotonic(),
            "value": [dict(row) for row in filtered_rows],
        }
        return filtered_rows

    def _resolved_default_model(self) -> str | None:
        defaults = self._defaults_snapshot()
        return (
            str(defaults.get("resolved_default_model") or "").strip()
            or str(defaults.get("chat_model") or "").strip()
            or str(defaults.get("default_model") or "").strip()
            or None
        )

    def _default_provider_id(self, model_id: str | None = None) -> str | None:
        defaults = self._defaults_snapshot()
        resolved_model = str(model_id or self._resolved_default_model() or "").strip() or None
        provider = str(defaults.get("default_provider") or "").strip().lower() or None
        if provider:
            return provider
        if resolved_model and ":" in resolved_model:
            return str(resolved_model.split(":", 1)[0]).strip().lower() or None
        return None

    def _chat_capable_model_rows(self, provider_id: str) -> list[dict[str, Any]]:
        provider_key = str(provider_id or "").strip().lower()
        models_doc = (
            self.runtime.registry_document.get("models")
            if isinstance(self.runtime.registry_document.get("models"), dict)
            else {}
        )
        rows: list[dict[str, Any]] = []
        for model_id, model_payload in models_doc.items():
            if not isinstance(model_payload, dict):
                continue
            if str(model_payload.get("provider") or "").strip().lower() != provider_key:
                continue
            capabilities = {
                str(item).strip().lower()
                for item in (model_payload.get("capabilities") if isinstance(model_payload.get("capabilities"), list) else [])
                if str(item).strip()
            }
            if not capabilities:
                capabilities = {"chat"}
            if "chat" not in capabilities:
                continue
            rows.append(
                {
                    "model_id": str(model_id or "").strip(),
                    "model": str(model_payload.get("model") or model_id).strip(),
                    "enabled": bool(model_payload.get("enabled", True)),
                    "available": bool(model_payload.get("available", True)),
                    "quality_rank": int(model_payload.get("quality_rank") or 0),
                }
            )
        return rows

    @staticmethod
    def _model_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            0 if bool(row.get("enabled", True)) else 1,
            0 if bool(row.get("available", True)) else 1,
            -int(row.get("quality_rank") or 0),
            str(row.get("model_id") or ""),
        )

    def _configured_chat_target_status(self) -> dict[str, Any]:
        model_id = self._resolved_default_model()
        provider_id = self._default_provider_id(model_id)
        models_doc = (
            self.runtime.registry_document.get("models")
            if isinstance(self.runtime.registry_document.get("models"), dict)
            else {}
        )
        current_row = models_doc.get(model_id) if model_id and isinstance(models_doc.get(model_id), dict) else {}
        capabilities = {
            str(item).strip().lower()
            for item in (current_row.get("capabilities") if isinstance(current_row, dict) and isinstance(current_row.get("capabilities"), list) else [])
            if str(item).strip()
        }
        if current_row and not capabilities:
            capabilities = {"chat"}
        model_health = self._model_health_row(model_id)
        provider_health = self._provider_health_row(provider_id)
        model_health_status = self._health_status(model_health)
        provider_health_status = self._health_status(provider_health)
        ready = bool(
            model_id
            and isinstance(current_row, dict)
            and bool(current_row.get("enabled", True))
            and bool(current_row.get("available", True))
            and "chat" in capabilities
            and model_health_status == "ok"
            and provider_health_status == "ok"
        )
        return {
            "provider": provider_id,
            "model": model_id,
            "ready": ready if model_id else False,
            "health_status": model_health_status,
            "provider_health_status": provider_health_status,
            "source": "defaults+health_state",
        }

    def _live_chat_target_status(self) -> dict[str, Any]:
        model_id = self._resolved_default_model()
        provider_id = self._default_provider_id(model_id)
        provider_health = self._provider_health_row(provider_id)
        model_health = self._model_health_row(model_id)
        router_status = self._router_model_status(model_id)
        provider_health_status = self._health_status(provider_health)
        model_health_status = self._health_status(model_health)
        router_health_status = str(router_status.get("status") or "").strip().lower() or "unknown"
        if model_health_status == "unknown" and router_health_status in {"ok", "degraded", "down"}:
            model_health_status = router_health_status
        ready = bool(
            model_id
            and provider_id
            and bool(
                router_status.get("available") is True
                or model_health_status == "ok"
            )
            and provider_health_status == "ok"
            and model_health_status == "ok"
        )
        return {
            "provider": provider_id,
            "model": model_id,
            "ready": ready,
            "health_status": model_health_status,
            "provider_health_status": provider_health_status,
            "source": "defaults+health_state+router_snapshot",
        }

    def _safe_mode_pin_exists_live(self, safe_mode_target: dict[str, Any]) -> bool | None:
        if not bool(safe_mode_target.get("enabled", False)):
            return None
        configured_model = str(safe_mode_target.get("configured_model") or "").strip() or None
        configured_provider = (
            str(safe_mode_target.get("configured_provider") or "").strip().lower() or None
        )
        if not configured_model or configured_provider != "ollama":
            return None
        tags_reader = getattr(self.runtime, "_ollama_tags_models", None)
        if not callable(tags_reader):
            return None
        try:
            tags_payload = tags_reader(timeout_seconds=1.0)
        except Exception:
            return None
        if not isinstance(tags_payload, dict) or not bool(tags_payload.get("ok", False)):
            return None
        live_models = {
            str(item).strip()
            for item in (tags_payload.get("models") if isinstance(tags_payload.get("models"), list) else [])
            if str(item).strip()
        }
        if not live_models:
            return None
        normalized_model = (
            configured_model.split(":", 1)[1].strip()
            if configured_model.startswith("ollama:")
            else configured_model
        )
        return normalized_model in live_models or configured_model in live_models

    def chat_target_truth(
        self,
        *,
        selection: dict[str, Any] | None = None,
        safe_mode_target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        configured = self._configured_chat_target_status()
        live_current = self._live_chat_target_status()
        resolved_selection = (
            dict(selection)
            if isinstance(selection, dict)
            else self._default_chat_policy_selection()
        )
        resolved_safe_mode_target = (
            dict(safe_mode_target)
            if isinstance(safe_mode_target, dict)
            else (
                self.runtime.safe_mode_target_status()
                if callable(getattr(self.runtime, "safe_mode_target_status", None))
                else {}
            )
        )
        selected_candidate = (
            resolved_selection.get("selected_candidate")
            if isinstance(resolved_selection.get("selected_candidate"), dict)
            else None
        )
        configured_provider = str(configured.get("provider") or "").strip().lower() or None
        configured_model = str(configured.get("model") or "").strip() or None
        configured_ready = bool(configured.get("ready", False))
        live_provider = str(live_current.get("provider") or "").strip().lower() or None
        live_model = str(live_current.get("model") or "").strip() or None
        live_ready = bool(live_current.get("ready", False))
        effective_provider = (
            live_provider
            or str((selected_candidate or {}).get("provider_id") or "").strip().lower()
            or configured_provider
        )
        effective_model = (
            live_model
            or str((selected_candidate or {}).get("model_id") or "").strip()
            or configured_model
        )
        provider_health_status = str(configured.get("provider_health_status") or "unknown").strip().lower() or "unknown"
        model_health_status = str(configured.get("health_status") or "unknown").strip().lower() or "unknown"
        effective_provider_health_status = (
            str(live_current.get("provider_health_status") or "").strip().lower()
            or provider_health_status
        )
        effective_model_health_status = (
            str(live_current.get("health_status") or "").strip().lower()
            or model_health_status
        )
        if live_model and live_provider:
            effective_ready = bool(live_ready)
        else:
            effective_ready = bool(
                configured_ready
                or (
                    selected_candidate is not None
                    and str((selected_candidate or {}).get("model_id") or "").strip()
                )
            )
        qualification_reason: str | None = None
        degraded_reason: str | None = None
        if effective_ready and effective_model and effective_provider:
            qualification_reason = (
                f"Current active target {effective_model} on {effective_provider} is healthy and ready."
            )
        elif configured_ready and configured_model and configured_provider:
            qualification_reason = (
                f"Configured default {configured_model} on {configured_provider} is healthy and ready."
            )
        elif configured_model and effective_model and effective_model != configured_model:
            degraded_reason = (
                f"Configured default {configured_model} on {configured_provider or 'the configured provider'} "
                f"is not currently healthy. The best healthy target would be "
                f"{effective_model} on {effective_provider or 'another provider'}."
            )
            qualification_reason = degraded_reason
        elif configured_model and provider_health_status == "down":
            degraded_reason = (
                f"Configured default {configured_model} on {configured_provider or 'the configured provider'} "
                f"is not responding right now."
            )
            qualification_reason = degraded_reason
        elif configured_model and provider_health_status == "degraded":
            degraded_reason = (
                f"Configured default {configured_model} on {configured_provider or 'the configured provider'} "
                f"needs attention right now."
            )
            qualification_reason = degraded_reason
        elif configured_model and model_health_status == "down":
            degraded_reason = (
                f"Configured default {configured_model} is not healthy right now."
            )
            qualification_reason = degraded_reason
        elif effective_model and effective_provider and not configured_model:
            qualification_reason = (
                f"No chat default is configured. The best healthy target would be "
                f"{effective_model} on {effective_provider}."
            )
        elif configured_model:
            degraded_reason = (
                f"Configured default {configured_model} on {configured_provider or 'the configured provider'} "
                f"is not ready right now."
            )
            qualification_reason = degraded_reason
        safe_mode_pin_exists_live = self._safe_mode_pin_exists_live(
            resolved_safe_mode_target if isinstance(resolved_safe_mode_target, dict) else {}
        )
        live_target_healthy = bool(
            effective_model
            and effective_provider
            and effective_ready
            and effective_provider_health_status == "ok"
            and effective_model_health_status == "ok"
        )
        if (
            bool(resolved_safe_mode_target.get("enabled", False))
            and resolved_safe_mode_target.get("configured_valid") is False
            and safe_mode_pin_exists_live is not True
            and not live_target_healthy
        ):
            safe_mode_message = str(resolved_safe_mode_target.get("message") or "").strip() or None
            if safe_mode_message:
                degraded_reason = safe_mode_message
                qualification_reason = safe_mode_message
        decision_detail = str(resolved_selection.get("decision_detail") or "").strip() or None
        if qualification_reason is None and decision_detail:
            qualification_reason = decision_detail
        return {
            "configured_provider": configured_provider,
            "configured_model": configured_model,
            "configured_ready": configured_ready,
            "configured_provider_health_status": provider_health_status,
            "configured_model_health_status": model_health_status,
            "effective_provider": effective_provider,
            "effective_model": effective_model,
            "effective_ready": effective_ready,
            "effective_provider_health_status": effective_provider_health_status,
            "effective_model_health_status": effective_model_health_status,
            "qualification_reason": qualification_reason,
            "degraded_reason": degraded_reason,
            "selection_decision_reason": str(resolved_selection.get("decision_reason") or "").strip() or None,
            "selection_decision_detail": decision_detail,
            "selected_candidate": dict(selected_candidate) if isinstance(selected_candidate, dict) else None,
            "selection": dict(resolved_selection),
            "safe_mode_pin_exists_live": safe_mode_pin_exists_live,
            "safe_mode_target": dict(resolved_safe_mode_target) if isinstance(resolved_safe_mode_target, dict) else {},
            "source": "runtime_truth.current_target+defaults+health_state+policy",
        }

    def current_chat_target_status(self) -> dict[str, Any]:
        cache_key = "current_chat_target_status"
        cached = self._snapshot_cache().get(cache_key)
        if isinstance(cached, dict):
            created_at = float(cached.get("created_at") or 0.0)
            if created_at and time.monotonic() - created_at <= self._SNAPSHOT_CACHE_TTL_SECONDS:
                value = cached.get("value")
                return deepcopy(value) if isinstance(value, dict) else {}

        started = time.monotonic()
        configured = self._configured_chat_target_status()
        truth = self.chat_target_truth()
        payload = {
            **configured,
            "provider": truth.get("effective_provider") or configured.get("provider"),
            "model": truth.get("effective_model") or configured.get("model"),
            "ready": bool(
                truth.get("effective_ready", False)
                if truth.get("effective_model")
                else configured.get("ready", False)
            ),
            "health_status": truth.get("effective_model_health_status") or configured.get("health_status"),
            "provider_health_status": truth.get("effective_provider_health_status")
            or configured.get("provider_health_status"),
            "configured_provider": truth.get("configured_provider"),
            "configured_model": truth.get("configured_model"),
            "configured_ready": truth.get("configured_ready"),
            "effective_provider": truth.get("effective_provider"),
            "effective_model": truth.get("effective_model"),
            "effective_ready": truth.get("effective_ready"),
            "qualification_reason": truth.get("qualification_reason"),
            "degraded_reason": truth.get("degraded_reason"),
            "source": "runtime_truth.chat_target_truth",
        }
        payload["truth_timing_ms"] = {
            "current_chat_target_status_ms": int(max(0.0, time.monotonic() - started) * 1000),
            "cache_hit": False,
        }
        self._snapshot_cache()[cache_key] = {
            "created_at": time.monotonic(),
            "value": deepcopy(payload),
        }
        return deepcopy(payload)

    def _provider_status_snapshot(
        self,
        provider_id: str,
        *,
        include_target_truth: bool,
        router_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provider_key = str(provider_id or "").strip().lower()
        target_truth = self.chat_target_truth() if include_target_truth else {}
        providers_doc = (
            self.runtime.registry_document.get("providers")
            if isinstance(self.runtime.registry_document.get("providers"), dict)
            else {}
        )
        provider_payload = (
            providers_doc.get(provider_key)
            if isinstance(providers_doc.get(provider_key), dict)
            else {}
        )
        inventory_rows = [
            dict(row)
            for row in self._runtime_inventory_rows()
            if isinstance(row, dict)
            and str(row.get("provider") or "").strip().lower() == provider_key
        ]
        model_rows = []
        for row in inventory_rows:
            capabilities = {
                str(item).strip().lower()
                for item in (row.get("capabilities") if isinstance(row.get("capabilities"), list) else [])
                if str(item).strip()
            }
            if not capabilities:
                capabilities = {"chat"}
            if "chat" not in capabilities:
                continue
            model_rows.append(
                {
                    "model_id": str(row.get("id") or "").strip(),
                    "model": str(row.get("model_name") or row.get("id") or "").strip(),
                    "routable": bool(row.get("routable", False)),
                    "available": bool(row.get("available", False)),
                    "enabled": True,
                    "quality_rank": int(row.get("quality_rank") or 0),
                    "health_status": str(row.get("health_status") or "unknown").strip().lower() or "unknown",
                    "installed": bool(row.get("installed", False)),
                }
            )
        current_model_id = self._resolved_default_model()
        current_provider = self._default_provider_id(current_model_id)
        provider_known = bool(provider_payload) or bool(model_rows)
        provider_local = bool(
            (provider_payload if provider_payload else {}).get("local", provider_key == "ollama")
        )
        provider_enabled = bool(
            (provider_payload if provider_payload else {}).get("enabled", provider_known)
        )
        api_key_source = (
            provider_payload.get("api_key_source")
            if isinstance(provider_payload.get("api_key_source"), dict)
            else None
        )
        auth_required = bool((provider_payload if provider_payload else {}) and not provider_local and api_key_source is not None)
        secret_present = bool(self.runtime._provider_api_key(provider_payload)) if auth_required else False
        preferred_row = None
        if current_provider == provider_key and current_model_id:
            preferred_row = next(
                (row for row in model_rows if str(row.get("model_id") or "") == str(current_model_id)),
                None,
            )
        if preferred_row is None and model_rows:
            preferred_row = sorted(model_rows, key=self._model_sort_key)[0]
        preferred_model_id = str((preferred_row or {}).get("model_id") or "").strip() or None
        configured = bool(
            provider_known
            and provider_enabled
            and model_rows
            and (provider_local or not auth_required or secret_present)
        )
        provider_health = self._provider_health_row(provider_key)
        current_model_health = self._model_health_row(
            current_model_id if current_provider == provider_key else preferred_model_id,
        )
        health_status = self._health_status(provider_health)
        if health_status == "unknown" and any(str(row.get("health_status") or "") == "ok" for row in model_rows):
            health_status = "ok"
        if bool(current_provider == provider_key and current_model_id) and self._health_status(current_model_health) == "ok":
            health_status = "ok"
        health_reason = None if health_status == "ok" else (
            self._health_reason(provider_health) or self._health_reason(current_model_health)
        )
        policy = self._policy_flags()
        policy_blocked = bool(
            not provider_local and not bool(policy.get("allow_remote_recommendation", policy.get("allow_remote_fallback", True)))
        )
        if not provider_known or not provider_enabled or not model_rows:
            connection_state = "discovered_but_not_usable"
        elif auth_required and not secret_present:
            connection_state = "configured_but_auth_missing"
        elif health_status in {"down", "degraded"}:
            connection_state = "configured_but_unhealthy"
        else:
            connection_state = "configured_and_usable"
        if not provider_enabled or policy_blocked:
            selection_state = "ignored_by_policy"
        else:
            selection_state = connection_state
        return {
            "provider": provider_key,
            "provider_label": self.runtime._setup_provider_label(provider_key),
            "known": provider_known,
            "enabled": provider_enabled,
            "local": provider_local,
            "configured": configured,
            "auth_required": auth_required,
            "active": bool(current_provider == provider_key and current_model_id),
            "secret_present": secret_present,
            "health_status": health_status,
            "health_reason": health_reason,
            "connection_state": connection_state,
            "selection_state": selection_state,
            "usable_for_selection": selection_state == "configured_and_usable",
            "policy_blocked": policy_blocked,
            "model_id": preferred_model_id,
            "model_ids": [
                str(row.get("model_id") or "").strip()
                for row in sorted(model_rows, key=self._model_sort_key)
                if str(row.get("model_id") or "").strip()
            ],
            "current_provider": current_provider,
            "current_model_id": current_model_id,
            "effective_provider": str(target_truth.get("effective_provider") or "").strip().lower() or None,
            "effective_model_id": str(target_truth.get("effective_model") or "").strip() or None,
            "effective_active": bool(
                str(target_truth.get("effective_provider") or "").strip().lower() == provider_key
            ),
            "qualification_reason": str(target_truth.get("qualification_reason") or "").strip() or None,
            "degraded_reason": str(target_truth.get("degraded_reason") or "").strip() or None,
        }

    def provider_status(self, provider_id: str) -> dict[str, Any]:
        return self._provider_status_snapshot(provider_id, include_target_truth=True)

    def providers_status(self) -> dict[str, Any]:
        provider_ids = {
            "ollama",
            "openrouter",
            "openai",
            *(
                str(provider_id).strip().lower()
                for provider_id in (
                    self.runtime.registry_document.get("providers").keys()
                    if isinstance(self.runtime.registry_document.get("providers"), dict)
                    else []
                )
                if str(provider_id).strip()
            ),
        }
        rows = [
            self.provider_status(provider_id)
            for provider_id in sorted(provider_ids)
            if str(provider_id).strip()
        ]
        configured_rows = [row for row in rows if bool(row.get("configured", False))]
        active_row = next((row for row in rows if bool(row.get("active", False))), None)
        return {
            "providers": rows,
            "configured_providers": [
                str(row.get("provider") or "").strip().lower()
                for row in configured_rows
                if str(row.get("provider") or "").strip()
            ],
            "active_provider": str((active_row or {}).get("provider") or "").strip().lower() or None,
            "active_model_id": str((active_row or {}).get("model_id") or "").strip() or None,
        }

    def _approved_chat_model_ids(self) -> set[str]:
        approved: set[str] = set()
        for policy_name in ("default_policy", "premium_policy"):
            policy = getattr(self.runtime.config, policy_name, None)
            allowlist = policy.get("allowlist") if isinstance(policy, dict) else []
            if not isinstance(allowlist, list):
                continue
            approved.update(str(item).strip() for item in allowlist if str(item).strip())
        return approved

    def model_inventory_status(self) -> dict[str, Any]:
        cache_key = "model_inventory_status"
        cached = self._snapshot_cache().get(cache_key)
        if isinstance(cached, dict):
            created_at = float(cached.get("created_at") or 0.0)
            if created_at and time.monotonic() - created_at <= self._SNAPSHOT_CACHE_TTL_SECONDS:
                value = cached.get("value")
                return deepcopy(value) if isinstance(value, dict) else {}

        started = time.monotonic()
        target_status = self._configured_chat_target_status()
        rows: list[dict[str, Any]] = []
        models_doc = (
            self.runtime.registry_document.get("models")
            if isinstance(self.runtime.registry_document.get("models"), dict)
            else {}
        )
        for inventory_row in self._runtime_inventory_rows():
            if not isinstance(inventory_row, dict):
                continue
            capabilities = {
                str(item).strip().lower()
                for item in (inventory_row.get("capabilities") if isinstance(inventory_row.get("capabilities"), list) else [])
                if str(item).strip()
            }
            if not capabilities:
                capabilities = {"chat"}
            if "chat" not in capabilities:
                continue
            model_key = str(inventory_row.get("id") or "").strip()
            provider_id = (
                str(inventory_row.get("provider") or "").strip().lower()
                or (str(model_key).split(":", 1)[0].strip().lower() if ":" in str(model_key) else "")
            )
            if not provider_id:
                continue
            model_payload = models_doc.get(model_key) if isinstance(models_doc.get(model_key), dict) else {}
            enabled = bool(model_payload.get("enabled", True))
            available = bool(inventory_row.get("available", False))
            local = bool(inventory_row.get("local", False))
            active = bool(
                str(target_status.get("model") or "").strip() == model_key
                and str(target_status.get("provider") or "").strip().lower() == provider_id
            )
            rows.append(
                {
                    "model_id": model_key,
                    "provider_id": provider_id,
                    "model_name": str(inventory_row.get("model_name") or model_payload.get("model") or model_key).strip(),
                    "capabilities": sorted(capabilities),
                    "task_types": [
                        str(item).strip().lower()
                        for item in (
                            inventory_row.get("task_types")
                            if isinstance(inventory_row.get("task_types"), list)
                            else []
                        )
                        if str(item).strip()
                    ],
                    "architecture_modality": str(inventory_row.get("architecture_modality") or "").strip().lower() or None,
                    "input_modalities": list(inventory_row.get("input_modalities") or []) if isinstance(inventory_row.get("input_modalities"), list) else [],
                    "output_modalities": list(inventory_row.get("output_modalities") or []) if isinstance(inventory_row.get("output_modalities"), list) else [],
                    "local": local,
                    "active": active,
                    "enabled": enabled,
                    "available": available,
                    "known": True,
                    "registered": bool(inventory_row.get("runtime_known", True)),
                    "runtime_known": bool(inventory_row.get("runtime_known", True)),
                    "installed": bool(inventory_row.get("installed", False)),
                    "installed_local": bool(local and inventory_row.get("installed", False)),
                    "quality_rank": int(inventory_row.get("quality_rank") or 0),
                    "cost_rank": int(inventory_row.get("cost_rank") or 0),
                    "context_window": (
                        int(inventory_row.get("context_window"))
                        if inventory_row.get("context_window") is not None
                        else None
                    ),
                    "price_in": inventory_row.get("price_in"),
                    "price_out": inventory_row.get("price_out"),
                    "health_status": str(inventory_row.get("health_status") or "unknown").strip().lower() or "unknown",
                    "health_failure_kind": str(inventory_row.get("health_failure_kind") or "").strip().lower() or None,
                    "health_reason": str(inventory_row.get("health_reason") or "").strip() or None,
                    "source": str(inventory_row.get("source") or "runtime_inventory").strip() or "runtime_inventory",
                }
            )
        rows.sort(
            key=lambda row: (
                0 if bool(row.get("active", False)) else 1,
                0 if bool(row.get("installed_local", False)) else 1,
                0 if bool(row.get("local", False)) else 1,
                -int(row.get("quality_rank") or 0),
                str(row.get("model_id") or ""),
            )
        )
        lifecycle_rows = build_model_lifecycle_rows(
            inventory_rows=rows,
            readiness_rows=[],
            manager_state=self._model_manager_state(),
        )
        lifecycle_by_model = {
            str(row.get("model_id") or "").strip(): dict(row)
            for row in lifecycle_rows
            if isinstance(row, dict) and str(row.get("model_id") or "").strip()
        }
        rows = [
            {
                **dict(row),
                "lifecycle_state": str((lifecycle_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("lifecycle_state") or "not_installed"),
                "lifecycle_message": (lifecycle_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("message"),
                "lifecycle_error_kind": (lifecycle_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("error_kind"),
            }
            for row in rows
        ]
        payload = {
            "active_provider": str(target_status.get("provider") or "").strip().lower() or None,
            "active_model": str(target_status.get("model") or "").strip() or None,
            "configured_provider": str(target_status.get("provider") or "").strip().lower() or None,
            "configured_model": str(target_status.get("model") or "").strip() or None,
            "known_models": [dict(row) for row in rows],
            "local_installed_models": [
                dict(row) for row in rows if bool(row.get("installed_local", False))
            ],
            "remote_registered_models": [
                dict(row) for row in rows if not bool(row.get("local", False))
            ],
            "lifecycle_rows": lifecycle_rows,
            "models": rows,
            "source": "runtime_inventory+model_manager_lifecycle",
        }
        payload["truth_timing_ms"] = {
            "model_inventory_status_ms": int(max(0.0, time.monotonic() - started) * 1000),
            "cache_hit": False,
        }
        self._snapshot_cache()[cache_key] = {
            "created_at": time.monotonic(),
            "value": deepcopy(payload),
        }
        return deepcopy(payload)

    def filesystem_list_directory(
        self,
        path: str | None,
        *,
        max_entries: int = 200,
    ) -> dict[str, Any]:
        payload = self._filesystem_skill().list_directory(path, max_entries=max_entries)
        return {
            **dict(payload),
            "type": "filesystem_list_directory",
            "source": "runtime_truth.filesystem",
        }

    def filesystem_stat_path(self, path: str | None) -> dict[str, Any]:
        payload = self._filesystem_skill().stat_path(path)
        return {
            **dict(payload),
            "type": "filesystem_stat_path",
            "source": "runtime_truth.filesystem",
        }

    def filesystem_read_text_file(
        self,
        path: str | None,
        *,
        max_bytes: int = 8192,
        offset: int = 0,
    ) -> dict[str, Any]:
        payload = self._filesystem_skill().read_text_file(path, max_bytes=max_bytes, offset=offset)
        return {
            **dict(payload),
            "type": "filesystem_read_text_file",
            "source": "runtime_truth.filesystem",
        }

    def filesystem_search_filenames(
        self,
        root: str | None,
        query: str | None,
        *,
        max_results: int = 25,
        max_depth: int = 4,
    ) -> dict[str, Any]:
        payload = self._filesystem_skill().search_filenames(
            root,
            query,
            max_results=max_results,
            max_depth=max_depth,
        )
        return {
            **dict(payload),
            "type": "filesystem_search_filenames",
            "source": "runtime_truth.filesystem",
        }

    def filesystem_search_text(
        self,
        root: str | None,
        query: str | None,
        *,
        max_results: int = 25,
        max_files: int = 200,
        max_bytes_per_file: int = 8192,
    ) -> dict[str, Any]:
        payload = self._filesystem_skill().search_text(
            root,
            query,
            max_results=max_results,
            max_files=max_files,
            max_bytes_per_file=max_bytes_per_file,
        )
        return {
            **dict(payload),
            "type": "filesystem_search_text",
            "source": "runtime_truth.filesystem",
        }

    def shell_execute_safe_command(
        self,
        command_name: str | None,
        *,
        subject: str | None = None,
        query: str | None = None,
        cwd: str | None = None,
        timeout_s: float = 2.0,
        max_output_chars: int = 4000,
    ) -> dict[str, Any]:
        payload = self._shell_skill().execute_safe_command(
            command_name,
            subject=subject,
            query=query,
            cwd=cwd,
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
        )
        return {
            **dict(payload),
            "type": "shell_execute_safe_command",
            "source": "runtime_truth.shell",
        }

    def shell_install_package(
        self,
        *,
        manager: str | None,
        package: str | None,
        scope: str | None = None,
        dry_run: bool = False,
        cwd: str | None = None,
        timeout_s: float = 10.0,
        max_output_chars: int = 4000,
    ) -> dict[str, Any]:
        payload = self._shell_skill().install_package(
            manager=manager,
            package=package,
            scope=scope,
            dry_run=dry_run,
            cwd=cwd,
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
        )
        return {
            **dict(payload),
            "type": "shell_install_package",
            "source": "runtime_truth.shell",
        }

    def shell_preview_install_package(
        self,
        *,
        manager: str | None,
        package: str | None,
        scope: str | None = None,
        dry_run: bool = False,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        payload = self._shell_skill().preview_install_package(
            manager=manager,
            package=package,
            scope=scope,
            dry_run=dry_run,
            cwd=cwd,
        )
        return {
            **dict(payload),
            "type": "shell_install_package_preview",
            "source": "runtime_truth.shell",
        }

    def shell_create_directory(self, path: str | None) -> dict[str, Any]:
        payload = self._shell_skill().create_directory(path)
        return {
            **dict(payload),
            "type": "shell_create_directory",
            "source": "runtime_truth.shell",
        }

    def shell_preview_create_directory(self, path: str | None) -> dict[str, Any]:
        payload = self._shell_skill().preview_create_directory(path)
        return {
            **dict(payload),
            "type": "shell_create_directory_preview",
            "source": "runtime_truth.shell",
        }

    def model_discovery_proposals_status(
        self,
        *,
        provider_id: str | None = None,
        source: str | None = None,
        proposal_kind: str | None = None,
    ) -> dict[str, Any]:
        inventory = self.model_inventory_status()
        inventory_rows = inventory.get("models") if isinstance(inventory.get("models"), list) else []
        policy_entries_reader = getattr(self.runtime, "model_discovery_policy_entries", None)
        policy_entries = policy_entries_reader() if callable(policy_entries_reader) else []
        policy_status = self.model_policy_status()
        normalized_provider = str(provider_id or "").strip().lower() or None
        external_discovery = load_external_model_discovery_rows(
            provider_id=normalized_provider,
            openrouter_snapshot_path=getattr(self.runtime, "_model_watch_catalog_path", None),
        )
        external_rows = external_discovery.get("models") if isinstance(external_discovery.get("models"), list) else []
        seen_model_ids = {
            str(row.get("model_id") or "").strip()
            for row in inventory_rows
            if isinstance(row, dict) and str(row.get("model_id") or "").strip()
        }
        merged_inventory_rows = [dict(row) for row in inventory_rows if isinstance(row, dict)]
        for row in external_rows:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("model_id") or "").strip()
            if not model_id or model_id in seen_model_ids:
                continue
            merged_inventory_rows.append(dict(row))
            seen_model_ids.add(model_id)
        proposals = build_model_discovery_proposals(
            inventory_rows=merged_inventory_rows,
            policy_entries=[dict(row) for row in policy_entries if isinstance(row, dict)],
            cheap_remote_cap_per_1m=float(policy_status.get("cheap_remote_cap_per_1m") or 0.0),
        )
        normalized_source = str(source or "").strip().lower() or None
        normalized_kind = str(proposal_kind or "").strip().lower() or None
        available_sources = sorted(
            {
                str(row.get("source") or "").strip()
                for row in proposals
                if isinstance(row, dict) and str(row.get("source") or "").strip()
            }
        )
        available_kinds = sorted(
            {
                str(row.get("proposal_kind") or "").strip().lower()
                for row in proposals
                if isinstance(row, dict) and str(row.get("proposal_kind") or "").strip()
            }
        )
        filtered = [
            dict(row)
            for row in proposals
            if isinstance(row, dict)
            and (not normalized_provider or str(row.get("provider_id") or "").strip().lower() == normalized_provider)
            and (not normalized_source or str(row.get("source") or "").strip().lower() == normalized_source)
            and (not normalized_kind or str(row.get("proposal_kind") or "").strip().lower() == normalized_kind)
        ]
        policy_rows = [dict(row) for row in policy_entries if isinstance(row, dict)]
        return {
            "type": "model_discovery_proposals",
            "source": "runtime_truth.model_discovery_proposals",
            "non_canonical": True,
            "review_required": True,
            "canonical_status": "not_adopted",
            "canonical_selector_surface": "runtime_truth.model_scout_v2_status.recommendation_roles",
            "inventory_model_count": len([row for row in inventory_rows if isinstance(row, dict)]),
            "external_model_count": len([row for row in external_rows if isinstance(row, dict)]),
            "external_sources": [
                dict(row)
                for row in (external_discovery.get("sources") if isinstance(external_discovery.get("sources"), list) else [])
                if isinstance(row, dict)
            ],
            "policy_entries": policy_rows,
            "policy_summary": {
                "known_good": len(
                    [row for row in policy_rows if str(row.get("status") or "").strip().lower() == "known_good"]
                ),
                "known_stale": len(
                    [row for row in policy_rows if str(row.get("status") or "").strip().lower() == "known_stale"]
                ),
                "avoid": len([row for row in policy_rows if str(row.get("status") or "").strip().lower() == "avoid"]),
            },
            "proposal_count": len(filtered),
            "filters": {
                "provider_id": normalized_provider,
                "source": normalized_source,
                "proposal_kind": normalized_kind,
            },
            "available_sources": available_sources,
            "available_proposal_kinds": available_kinds,
            "allowed_sources": allowed_model_discovery_proposal_sources(),
            "allowed_proposal_kinds": allowed_model_discovery_proposal_kinds(),
            "proposals": filtered,
        }

    def model_readiness_status(self) -> dict[str, Any]:
        cache_key = "model_readiness_status"
        cached = self._snapshot_cache().get(cache_key)
        if isinstance(cached, dict):
            created_at = float(cached.get("created_at") or 0.0)
            if created_at and time.monotonic() - created_at <= self._SNAPSHOT_CACHE_TTL_SECONDS:
                value = cached.get("value")
                return deepcopy(value) if isinstance(value, dict) else {}

        started = time.monotonic()
        inventory = self.model_inventory_status()
        policy = self._policy_flags()
        router_snapshot = self._router_snapshot()
        self._router_snapshot_cache = router_snapshot
        provider_cache: dict[str, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        for inventory_row in (
            inventory.get("models") if isinstance(inventory.get("models"), list) else []
        ):
            if not isinstance(inventory_row, dict):
                continue
            row = dict(inventory_row)
            provider_id = str(row.get("provider_id") or "").strip().lower()
            model_id = str(row.get("model_id") or "").strip()
            provider_status = provider_cache.get(provider_id)
            if provider_status is None:
                provider_status = self._provider_status_snapshot(
                    provider_id,
                    include_target_truth=False,
                )
                provider_cache[provider_id] = provider_status
            configured = bool(provider_status.get("configured", False))
            provider_connection_state = str(provider_status.get("connection_state") or "discovered_but_not_usable").strip().lower() or "discovered_but_not_usable"
            provider_selection_state = str(provider_status.get("selection_state") or provider_connection_state).strip().lower() or provider_connection_state
            provider_policy_blocked = bool(provider_status.get("policy_blocked", False))
            auth_required = bool(provider_status.get("auth_required", False))
            secret_present = bool(provider_status.get("secret_present", False))
            model_health_status = self._health_status(self._model_health_row(model_id))
            router_status = self._router_model_status(model_id, router_snapshot=router_snapshot)
            router_health_status = str(router_status.get("status") or "").strip().lower() or "unknown"
            if model_health_status == "unknown" and router_health_status in {"ok", "degraded", "down"}:
                model_health_status = router_health_status
            provider_health_status = str(provider_status.get("health_status") or "unknown").strip().lower() or "unknown"
            if model_health_status == "ok" and provider_health_status == "unknown":
                provider_health_status = "ok"
            enabled = bool(row.get("enabled", True))
            available = bool(row.get("available", False))
            if router_status.get("available") is True:
                available = True
            lifecycle_state = str(row.get("lifecycle_state") or "not_installed").strip().lower() or "not_installed"
            local = bool(row.get("local", False))
            install_profile = approved_local_profile_for_ref(model_id) if local else None
            acquirable = bool(install_profile)
            acquisition_state = "not_applicable"
            acquisition_reason = None
            if lifecycle_state == "ready":
                acquisition_state = "ready_now"
            elif lifecycle_state == "installed_not_ready":
                acquisition_state = "installed_not_ready"
                acquisition_reason = "installed, but it is not ready to use yet"
            elif lifecycle_state == "installed":
                acquisition_state = "installed"
                acquisition_reason = "installed, but not yet verified as ready"
            elif lifecycle_state == "queued":
                acquisition_state = "queued"
                acquisition_reason = "queued and waiting for explicit approval"
            elif lifecycle_state == "downloading":
                acquisition_state = "downloading"
                acquisition_reason = "download or import is in progress"
            elif lifecycle_state == "failed":
                if acquirable and bool(policy.get("allow_install_pull", True)):
                    acquisition_state = "acquirable"
                    acquisition_reason = "the last acquisition attempt failed, but it can be retried through the canonical model manager"
                elif acquirable:
                    acquisition_state = "blocked_by_policy"
                    acquisition_reason = "the last acquisition attempt failed, and SAFE MODE currently blocks retrying it"
                else:
                    acquisition_state = "not_acquirable"
                    acquisition_reason = "the last acquisition attempt failed, and this model is not on the canonical acquisition path"
            elif local and acquirable:
                if bool(policy.get("allow_install_pull", True)):
                    acquisition_state = "acquirable"
                    acquisition_reason = "approved for local acquisition through the canonical model manager"
                else:
                    acquisition_state = "blocked_by_policy"
                    acquisition_reason = "approved for local acquisition, but SAFE MODE currently blocks installs"
            elif local:
                acquisition_state = "not_acquirable"
                acquisition_reason = "not installed and not approved for canonical acquisition"
            usable_now = bool(
                enabled
                and available
                and configured
                and provider_selection_state == "configured_and_usable"
                and model_health_status == "ok"
                and provider_health_status == "ok"
            )
            if not available and model_health_status == "ok":
                available = True
                usable_now = bool(
                    enabled
                    and configured
                    and provider_selection_state == "configured_and_usable"
                    and provider_health_status == "ok"
                )
            if usable_now:
                acquisition_state = "ready_now"
                acquisition_reason = None
            if usable_now:
                availability_reason = "healthy and ready now"
                availability_state = "usable_now"
                eligibility_state = "usable_now"
                eligibility_reason = availability_reason
            elif lifecycle_state == "downloading":
                availability_reason = str(acquisition_reason or "download or import is in progress")
                availability_state = "downloading"
                eligibility_state = "downloading"
                eligibility_reason = availability_reason
            elif lifecycle_state == "queued":
                availability_reason = str(acquisition_reason or "queued and waiting for explicit approval")
                availability_state = "queued"
                eligibility_state = "queued"
                eligibility_reason = availability_reason
            elif lifecycle_state == "failed":
                availability_reason = str(acquisition_reason or "the last install or download attempt failed")
                availability_state = "failed"
                eligibility_state = "failed"
                eligibility_reason = availability_reason
            elif not enabled:
                availability_reason = "disabled"
                availability_state = "disabled"
                eligibility_state = "disabled"
                eligibility_reason = availability_reason
            elif provider_selection_state == "ignored_by_policy" or provider_policy_blocked:
                availability_reason = "ignored by current SAFE MODE or routing policy"
                availability_state = "ignored_by_policy"
                eligibility_state = "blocked_by_policy"
                eligibility_reason = availability_reason
            elif auth_required and not secret_present:
                availability_reason = "provider is configured but still missing auth"
                availability_state = "auth_missing"
                eligibility_state = "auth_missing"
                eligibility_reason = availability_reason
            elif not configured or provider_connection_state == "discovered_but_not_usable":
                availability_reason = "provider setup is still required"
                availability_state = "needs_setup"
                eligibility_state = "needs_setup"
                eligibility_reason = availability_reason
            elif provider_health_status in {"down", "degraded"}:
                availability_reason = f"provider is {provider_health_status}"
                availability_state = "provider_unhealthy"
                eligibility_state = "provider_unhealthy"
                eligibility_reason = availability_reason
            elif acquisition_state == "installed_not_ready":
                availability_reason = str(acquisition_reason or "installed, but it is not ready to use yet")
                availability_state = "installed_not_ready"
                eligibility_state = "installed_not_ready"
                eligibility_reason = availability_reason
            elif acquisition_state == "blocked_by_policy":
                availability_reason = str(acquisition_reason or "install is blocked by SAFE MODE or policy")
                availability_state = "install_blocked"
                eligibility_state = "blocked_by_policy"
                eligibility_reason = availability_reason
            elif acquisition_state == "acquirable":
                availability_reason = str(acquisition_reason or "approved for local acquisition through the canonical model manager")
                availability_state = "acquirable"
                eligibility_state = "acquirable"
                eligibility_reason = availability_reason
            elif acquisition_state == "not_acquirable":
                availability_reason = str(acquisition_reason or "not installed and not acquirable through the canonical model manager")
                availability_state = "not_acquirable"
                eligibility_state = "not_acquirable"
                eligibility_reason = availability_reason
            elif not available:
                availability_reason = "not available in the current runtime"
                availability_state = "unavailable"
                eligibility_state = "unavailable"
                eligibility_reason = availability_reason
            elif model_health_status in {"down", "degraded"}:
                availability_reason = f"model health is {model_health_status}"
                availability_state = "model_unhealthy"
                eligibility_state = "model_unhealthy"
                eligibility_reason = availability_reason
            else:
                availability_reason = "not ready right now"
                availability_state = "not_ready"
                eligibility_state = "not_ready"
                eligibility_reason = availability_reason
            rows.append(
                {
                    **row,
                    "configured": configured,
                    "usable_now": usable_now,
                    "provider_health_status": provider_health_status,
                    "model_health_status": model_health_status,
                    "provider_connection_state": provider_connection_state,
                    "provider_selection_state": provider_selection_state,
                    "auth_required": auth_required,
                    "acquirable": acquirable,
                    "acquisition_state": acquisition_state,
                    "acquisition_reason": acquisition_reason,
                    "availability_state": availability_state,
                    "availability_reason": availability_reason,
                    "eligibility_state": eligibility_state,
                    "eligibility_reason": eligibility_reason,
                }
            )
        rows.sort(
            key=lambda row: (
                0 if bool(row.get("active", False)) else 1,
                0 if bool(row.get("usable_now", False)) else 1,
                0 if bool(row.get("local", False)) else 1,
                -int(row.get("quality_rank") or 0),
                str(row.get("model_id") or ""),
            )
        )
        lifecycle_rows = build_model_lifecycle_rows(
            inventory_rows=[
                dict(row)
                for row in (
                    inventory.get("models")
                    if isinstance(inventory.get("models"), list)
                    else []
                )
                if isinstance(row, dict)
            ],
            readiness_rows=rows,
            manager_state=self._model_manager_state(),
        )
        lifecycle_by_model = {
            str(row.get("model_id") or "").strip(): dict(row)
            for row in lifecycle_rows
            if isinstance(row, dict) and str(row.get("model_id") or "").strip()
        }
        rows = [
            {
                **dict(row),
                "lifecycle_state": str((lifecycle_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("lifecycle_state") or "not_installed"),
                "lifecycle_message": (lifecycle_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("message"),
                "lifecycle_error_kind": (lifecycle_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("error_kind"),
            }
            for row in rows
        ]
        ready_rows = [dict(row) for row in rows if bool(row.get("usable_now", False))]
        result = {
            "active_provider": str(inventory.get("active_provider") or "").strip().lower() or None,
            "active_model": str(inventory.get("active_model") or "").strip() or None,
            "configured_provider": str(inventory.get("configured_provider") or "").strip().lower() or None,
            "configured_model": str(inventory.get("configured_model") or "").strip() or None,
            "models": rows,
            "ready_now_models": ready_rows,
            "usable_models": ready_rows,
            "other_ready_now_models": [
                dict(row) for row in ready_rows if not bool(row.get("active", False))
            ],
            "other_usable_models": [
                dict(row) for row in ready_rows if not bool(row.get("active", False))
            ],
            "not_ready_models": [
                dict(row) for row in rows if not bool(row.get("usable_now", False))
            ],
            "lifecycle_rows": lifecycle_rows,
            "source": "inventory+provider_health+model_health+model_manager_lifecycle",
        }
        if hasattr(self, "_router_snapshot_cache"):
            delattr(self, "_router_snapshot_cache")
        result["truth_timing_ms"] = {
            "model_readiness_status_ms": int(max(0.0, time.monotonic() - started) * 1000),
            "cache_hit": False,
        }
        self._snapshot_cache()[cache_key] = {
            "created_at": time.monotonic(),
            "value": deepcopy(result),
        }
        return deepcopy(result)

    @staticmethod
    def _selection_candidate_health_status(row: dict[str, Any]) -> str:
        if bool(row.get("usable_now", False)):
            return "ok"
        provider_health_status = str(row.get("provider_health_status") or "unknown").strip().lower() or "unknown"
        model_health_status = str(row.get("model_health_status") or "unknown").strip().lower() or "unknown"
        if provider_health_status in {"down", "degraded"}:
            return provider_health_status
        if model_health_status in {"down", "degraded"}:
            return model_health_status
        return "unknown"

    def canonical_chat_candidate_inventory(self) -> list[dict[str, Any]]:
        readiness = self.model_readiness_status()
        approved_ids = self._approved_chat_model_ids()
        policy = self._policy_flags()
        allow_remote_fallback = bool(policy.get("allow_remote_fallback", True))
        allow_remote_recommendation = bool(
            policy.get("allow_remote_recommendation", allow_remote_fallback)
        )
        rows: list[dict[str, Any]] = []
        for raw_row in (
            readiness.get("models") if isinstance(readiness.get("models"), list) else []
        ):
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            model_id = str(row.get("model_id") or "").strip()
            provider_id = str(row.get("provider_id") or "").strip().lower()
            if not model_id or not provider_id:
                continue
            local = bool(row.get("local", False))
            lifecycle_state = str(row.get("lifecycle_state") or "not_installed").strip().lower() or "not_installed"
            install_profile = approved_local_profile_for_ref(model_id) if local else None
            local_fit = self._local_model_fit(row, install_profile)
            installed = bool(
                row.get("installed", False)
                or lifecycle_state in {"installed", "installed_not_ready", "ready"}
                or bool(row.get("installed_local", False))
            )
            approved_remote = allow_remote_recommendation
            usable_now = bool(row.get("usable_now", False))
            reason = str(
                row.get("eligibility_reason")
                or row.get("availability_reason")
                or row.get("acquisition_reason")
                or row.get("lifecycle_message")
                or row.get("lifecycle_state")
                or "unknown"
            ).strip() or "unknown"
            rows.append(
                {
                    "id": model_id,
                    "provider": provider_id,
                    "installed": installed,
                    "available": usable_now,
                    "healthy": usable_now,
                    "capabilities": list(row.get("capabilities") or []),
                    "task_types": list(row.get("task_types") or []),
                    "architecture_modality": str(row.get("architecture_modality") or "").strip().lower() or None,
                    "input_modalities": list(row.get("input_modalities") or []) if isinstance(row.get("input_modalities"), list) else [],
                    "output_modalities": list(row.get("output_modalities") or []) if isinstance(row.get("output_modalities"), list) else [],
                    "size": row.get("size"),
                    "context_window": row.get("context_window"),
                    "local": local,
                    "approved": bool(local or approved_remote),
                    "reason": reason,
                    "quality_rank": int(row.get("quality_rank") or 0),
                    "cost_rank": int(row.get("cost_rank") or 0),
                    "price_in": row.get("price_in"),
                    "price_out": row.get("price_out"),
                    "health_status": self._selection_candidate_health_status(row),
                    "health_failure_kind": str(
                        row.get("availability_state")
                        or row.get("eligibility_state")
                        or row.get("lifecycle_error_kind")
                        or ""
                    ).strip() or None,
                    "health_reason": str(reason).strip() or None,
                    "model_name": str(row.get("model_name") or model_id).strip() or model_id,
                    "source": "runtime_truth.model_readiness_status",
                    "configured": bool(row.get("configured", False)),
                    "capability_source": "registry",
                    "capability_provenance": [],
                    "runtime_known": bool(row.get("runtime_known", True)),
                    "routable": usable_now,
                    "usable_now": usable_now,
                    "availability_state": str(row.get("availability_state") or "").strip() or None,
                    "availability_reason": str(row.get("availability_reason") or "").strip() or None,
                    "eligibility_state": str(row.get("eligibility_state") or "").strip() or None,
                    "eligibility_reason": str(row.get("eligibility_reason") or "").strip() or None,
                    "acquirable": bool(row.get("acquirable", False)),
                    "acquisition_state": str(row.get("acquisition_state") or "").strip() or None,
                    "acquisition_reason": str(row.get("acquisition_reason") or "").strip() or None,
                    "provider_health_status": str(row.get("provider_health_status") or "").strip().lower() or None,
                    "model_health_status": str(row.get("model_health_status") or "").strip().lower() or None,
                    "provider_connection_state": str(row.get("provider_connection_state") or "").strip().lower() or None,
                    "provider_selection_state": str(row.get("provider_selection_state") or "").strip().lower() or None,
                    "auth_required": bool(row.get("auth_required", False)),
                    "lifecycle_state": lifecycle_state,
                    "lifecycle_message": row.get("lifecycle_message"),
                    "lifecycle_error_kind": row.get("lifecycle_error_kind"),
                    "active": bool(row.get("active", False)),
                    "params_b": infer_parameter_size_b(model_id, row.get("model_name"), row.get("size")),
                    **local_fit,
                }
            )
        rows.sort(
            key=lambda row: (
                0 if bool(row.get("active", False)) else 1,
                0 if bool(row.get("local", False)) else 1,
                0 if bool(row.get("healthy", False)) else 1,
                -int(row.get("quality_rank") or 0),
                str(row.get("id") or ""),
            )
        )
        return rows

    def select_chat_candidates(
        self,
        *,
        policy: dict[str, Any] | Any | None = None,
        policy_name: str = "default",
        inventory_rows: list[dict[str, Any]] | None = None,
        candidate_model_ids: list[str] | None = None,
        allowed_tiers: tuple[str, ...] | None = None,
        min_improvement: float | None = None,
        current_model_id: str | None = None,
        allow_remote_fallback_override: bool | None = None,
        require_auth: bool = True,
        task_request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        defaults = self._defaults_snapshot()
        safe_mode = bool(
            callable(getattr(self.runtime, "_safe_mode_enabled", None)) and self.runtime._safe_mode_enabled()
        )
        allow_remote_fallback = (
            bool(defaults.get("allow_remote_fallback", True))
            if allow_remote_fallback_override is None
            else bool(allow_remote_fallback_override)
        )
        if safe_mode:
            allow_remote_fallback = False
        secret_store = getattr(self.runtime, "secret_store", None)
        secret_lookup = getattr(secret_store, "get_secret", None)
        document = self.runtime.registry_document if isinstance(self.runtime.registry_document, dict) else {}
        normalized_task_request = self._normalize_chat_task_request(task_request)
        candidate_inventory = (
            [
                dict(row)
                for row in inventory_rows
                if isinstance(row, dict)
            ]
            if isinstance(inventory_rows, list)
            else self.canonical_chat_candidate_inventory()
        )
        return choose_best_default_chat_candidate(
            config=self.runtime.config,
            registry_document=document,
            inventory_rows=candidate_inventory,
            candidate_model_ids=candidate_model_ids,
            allowed_tiers=allowed_tiers,
            min_improvement=min_improvement,
            policy=policy,
            policy_name=policy_name,
            current_model_id=current_model_id,
            allow_remote_fallback=allow_remote_fallback,
            env=os.environ,
            secret_lookup=secret_lookup if callable(secret_lookup) else None,
            require_auth=require_auth,
            task_request=normalized_task_request,
        )

    def model_controller_policy_status(
        self,
        *,
        target_truth: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        policy_flags = self._policy_flags()
        resolved_target_truth = (
            dict(target_truth)
            if isinstance(target_truth, dict)
            else self.chat_target_truth()
        )
        return {
            "type": "model_controller_policy",
            "mode": str(policy_flags.get("mode") or "safe").strip().lower() or "safe",
            "mode_label": str(policy_flags.get("mode_label") or "SAFE MODE").strip() or "SAFE MODE",
            "safe_mode": bool(policy_flags.get("safe_mode", False)),
            "mode_source": str(policy_flags.get("mode_source") or "config_default").strip().lower() or "config_default",
            "config_default_mode": str(policy_flags.get("config_default_mode") or policy_flags.get("mode") or "safe").strip().lower() or "safe",
            "config_default_mode_label": str(
                policy_flags.get("config_default_mode_label")
                or policy_flags.get("mode_label")
                or "SAFE MODE"
            ).strip() or "SAFE MODE",
            "override_mode": str(policy_flags.get("override_mode") or "").strip().lower() or None,
            "override_active": bool(policy_flags.get("override_active", False)),
            "allow_remote_fallback": bool(policy_flags.get("allow_remote_fallback", True)),
            "allow_remote_recommendation": bool(policy_flags.get("allow_remote_recommendation", policy_flags.get("allow_remote_fallback", True))),
            "allow_remote_switch": bool(policy_flags.get("allow_remote_switch", not bool(policy_flags.get("safe_mode", False)))),
            "allow_install_pull": bool(policy_flags.get("allow_install_pull", True)),
            "scout_advisory_only": bool(policy_flags.get("scout_advisory_only", False)),
            "approval_required_actions": [
                str(item).strip()
                for item in (
                    policy_flags.get("approval_required_actions")
                    if isinstance(policy_flags.get("approval_required_actions"), list)
                    else []
                )
                if str(item).strip()
            ],
            "forbidden_actions": [
                str(item).strip()
                for item in (
                    policy_flags.get("forbidden_actions")
                    if isinstance(policy_flags.get("forbidden_actions"), list)
                    else []
                )
                if str(item).strip()
            ],
            "transition": dict(policy_flags.get("transition")) if isinstance(policy_flags.get("transition"), dict) else {},
            "exact_target_preserved": True,
            "effective_provider": str(resolved_target_truth.get("effective_provider") or "").strip().lower() or None,
            "effective_model": str(resolved_target_truth.get("effective_model") or "").strip() or None,
            "configured_provider": str(resolved_target_truth.get("configured_provider") or "").strip().lower() or None,
            "configured_model": str(resolved_target_truth.get("configured_model") or "").strip() or None,
            "source": "runtime.control_mode_policy",
        }

    def setup_status(self) -> dict[str, Any]:
        current_target = self.current_chat_target_status()
        target_truth = self.chat_target_truth()
        inventory = self.model_inventory_status()
        active_provider = (
            str(
                current_target.get("provider")
                or target_truth.get("effective_provider")
                or inventory.get("active_provider")
                or ""
            ).strip().lower()
            or None
        )
        active_model = (
            str(
                current_target.get("model")
                or target_truth.get("effective_model")
                or inventory.get("active_model")
                or ""
            ).strip()
            or None
        )
        configured_provider = (
            str(
                current_target.get("configured_provider")
                or target_truth.get("configured_provider")
                or inventory.get("configured_provider")
                or ""
            ).strip().lower()
            or None
        )
        configured_model = (
            str(
                current_target.get("configured_model")
                or target_truth.get("configured_model")
                or inventory.get("configured_model")
                or ""
            ).strip()
            or None
        )
        provider_snapshot = self.provider_status(active_provider) if active_provider else {}
        provider_health_status = (
            str(
                provider_snapshot.get("health_status")
                or current_target.get("provider_health_status")
                or "unknown"
            ).strip().lower()
            or "unknown"
        )
        provider_health_reason = str(provider_snapshot.get("health_reason") or "").strip() or None
        model_health_status = (
            str(current_target.get("health_status") or "unknown").strip().lower() or "unknown"
        )
        local_installed_rows = [
            dict(row)
            for row in (
                inventory.get("local_installed_models")
                if isinstance(inventory.get("local_installed_models"), list)
                else []
            )
            if isinstance(row, dict)
        ]
        other_local_rows = [
            dict(row)
            for row in local_installed_rows
            if str(row.get("model_id") or "").strip() != str(active_model or "")
        ]
        ready = bool(current_target.get("ready", False) and active_model)
        if ready:
            setup_state = "ready"
            attention_kind = None
        elif active_model and active_provider:
            setup_state = "attention"
            if provider_health_status == "down":
                attention_kind = "provider_down"
            elif provider_health_status == "degraded":
                attention_kind = "provider_degraded"
            elif model_health_status in {"down", "degraded"}:
                attention_kind = "model_unhealthy"
            else:
                attention_kind = "not_ready"
        elif local_installed_rows:
            setup_state = "inventory_only"
            attention_kind = None
        else:
            setup_state = "unavailable"
            attention_kind = None
        return {
            "setup_state": setup_state,
            "attention_kind": attention_kind,
            "ready": ready,
            "active_provider": active_provider,
            "active_model": active_model,
            "configured_provider": configured_provider,
            "configured_model": configured_model,
            "effective_provider": str(target_truth.get("effective_provider") or "").strip().lower() or None,
            "effective_model": str(target_truth.get("effective_model") or "").strip() or None,
            "provider_health_status": provider_health_status,
            "provider_health_reason": provider_health_reason,
            "model_health_status": model_health_status,
            "qualification_reason": str(target_truth.get("qualification_reason") or "").strip() or None,
            "degraded_reason": str(target_truth.get("degraded_reason") or "").strip() or None,
            "local_installed_models": local_installed_rows,
            "other_local_models": other_local_rows,
            "source": "current_chat_target+chat_target_truth+inventory+provider_status",
        }

    def model_lifecycle_status(self) -> dict[str, Any]:
        inventory = self.model_inventory_status()
        readiness = self.model_readiness_status()
        lifecycle_rows = build_model_lifecycle_rows(
            inventory_rows=[
                dict(row)
                for row in (
                    inventory.get("models")
                    if isinstance(inventory.get("models"), list)
                    else []
                )
                if isinstance(row, dict)
            ],
            readiness_rows=[
                dict(row)
                for row in (
                    readiness.get("models")
                    if isinstance(readiness.get("models"), list)
                    else []
                )
                if isinstance(row, dict)
            ],
            manager_state=self._model_manager_state(),
        )
        readiness_by_model = {
            str(row.get("model_id") or "").strip(): dict(row)
            for row in (
                readiness.get("models")
                if isinstance(readiness.get("models"), list)
                else []
            )
            if isinstance(row, dict) and str(row.get("model_id") or "").strip()
        }
        lifecycle_rows = [
            {
                **dict(row),
                "acquirable": bool((readiness_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("acquirable", False)),
                "acquisition_state": (readiness_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("acquisition_state"),
                "acquisition_reason": (readiness_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("acquisition_reason"),
                "availability_state": (readiness_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("availability_state"),
                "availability_reason": (readiness_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("availability_reason"),
                "provider_connection_state": (readiness_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("provider_connection_state"),
                "provider_selection_state": (readiness_by_model.get(str(row.get("model_id") or "").strip()) or {}).get("provider_selection_state"),
            }
            for row in lifecycle_rows
        ]
        state_order = (
            "not_installed",
            "queued",
            "downloading",
            "installed",
            "installed_not_ready",
            "ready",
            "failed",
        )
        counts = {state: 0 for state in state_order}
        grouped: dict[str, list[dict[str, Any]]] = {state: [] for state in state_order}
        for row in lifecycle_rows:
            if not isinstance(row, dict):
                continue
            lifecycle_state = str(row.get("lifecycle_state") or "not_installed").strip().lower() or "not_installed"
            if lifecycle_state not in counts:
                counts[lifecycle_state] = 0
                grouped[lifecycle_state] = []
            counts[lifecycle_state] += 1
            grouped[lifecycle_state].append(dict(row))
        return {
            "active_provider": str(inventory.get("active_provider") or "").strip().lower() or None,
            "active_model": str(inventory.get("active_model") or "").strip() or None,
            "configured_provider": str(inventory.get("configured_provider") or "").strip().lower() or None,
            "configured_model": str(inventory.get("configured_model") or "").strip() or None,
            "tracked_targets": len(lifecycle_rows),
            "counts": counts,
            "queued_targets": grouped.get("queued", []),
            "downloading_targets": grouped.get("downloading", []),
            "installed_targets": grouped.get("installed", []),
            "installed_not_ready_targets": grouped.get("installed_not_ready", []),
            "ready_targets": grouped.get("ready", []),
            "failed_targets": grouped.get("failed", []),
            "active_operations": [
                *[dict(row) for row in grouped.get("queued", [])],
                *[dict(row) for row in grouped.get("downloading", [])],
            ],
            "policy": self.model_controller_policy_status(),
            "models": lifecycle_rows,
            "source": "inventory+readiness+model_manager_state",
        }

    def runtime_status(self, kind: str = "runtime_status") -> dict[str, Any]:
        normalized_kind = str(kind or "runtime_status").strip().lower() or "runtime_status"
        if normalized_kind == "telegram_status":
            telegram = self.runtime.telegram_status()
            state = str(telegram.get("state") or "unknown").strip().lower() or "unknown"
            configured = bool(telegram.get("configured", False))
            return {
                "scope": "telegram",
                "configured": configured,
                "state": state,
                "summary": normalize_persona_text(
                    f"Telegram is {state.replace('_', ' ')}." if configured else "Telegram is not configured yet."
                ),
            }

        ready = self.ready_status()
        current_target = self.current_chat_target_status()
        target_truth = self.chat_target_truth()
        runtime_status = (
            ready.get("runtime_status")
            if isinstance(ready.get("runtime_status"), dict)
            else {}
        )
        provider = str(target_truth.get("effective_provider") or current_target.get("provider") or self._default_provider_id() or "").strip().lower() or None
        model = str(target_truth.get("effective_model") or current_target.get("model") or self._resolved_default_model() or "").strip() or None
        summary = normalize_persona_text(
            str(ready.get("message") or "").strip()
            if bool(ready.get("ready", False))
            else (
                str(target_truth.get("qualification_reason") or "").strip()
                or str(ready.get("message") or "").strip()
                or str(runtime_status.get("summary") or "").strip()
                or "I can't read a clean runtime status from the current state yet."
            )
        )
        return {
            "scope": "ready",
            "ready": bool(ready.get("ready", False)),
            "runtime_mode": str(
                ready.get("runtime_mode") or runtime_status.get("runtime_mode") or "DEGRADED"
            ).strip()
            or "DEGRADED",
            "failure_code": str(
                ready.get("failure_code") or runtime_status.get("failure_code") or ""
            ).strip()
            or None,
            "provider": provider,
            "model": model,
            "configured_provider": str(target_truth.get("configured_provider") or "").strip().lower() or None,
            "configured_model": str(target_truth.get("configured_model") or "").strip() or None,
            "qualification_reason": str(target_truth.get("qualification_reason") or "").strip() or None,
            "summary": summary,
        }

    def skill_governance_status(self) -> dict[str, Any]:
        payload = self.runtime.skill_governance_status()
        return dict(payload) if isinstance(payload, dict) else {}

    def managed_adapters_status(self) -> dict[str, Any]:
        payload = self.runtime.managed_adapters_status()
        return dict(payload) if isinstance(payload, dict) else {}

    def background_tasks_status(self) -> dict[str, Any]:
        payload = self.runtime.background_tasks_status()
        return dict(payload) if isinstance(payload, dict) else {}

    @staticmethod
    def _normalize_key(value: str | None) -> str:
        lowered = str(value or "").strip().lower()
        cleaned = re.sub(r"[^a-z0-9_-]+", "_", lowered)
        return re.sub(r"_+", "_", cleaned).strip("_")

    def list_managed_adapters(self) -> dict[str, Any]:
        payload = self.managed_adapters_status()
        rows = payload.get("managed_adapters") if isinstance(payload.get("managed_adapters"), list) else []
        normalized_rows = [dict(row) for row in rows if isinstance(row, dict)]
        active_rows = [
            row
            for row in normalized_rows
            if bool(row.get("approved", False))
            and bool(row.get("enabled", False))
        ]
        return {
            "managed_adapters": normalized_rows,
            "active_adapters": active_rows,
            "active_adapter_ids": [
                str(row.get("adapter_id") or "").strip()
                for row in active_rows
                if str(row.get("adapter_id") or "").strip()
            ],
        }

    def get_managed_adapter_status(self, adapter_id: str | None) -> dict[str, Any]:
        normalized = self._normalize_key(adapter_id)
        adapters_payload = self.list_managed_adapters()
        rows = adapters_payload.get("managed_adapters") if isinstance(adapters_payload, dict) else []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            row_id = self._normalize_key(str(row.get("adapter_id") or ""))
            row_type = self._normalize_key(str(row.get("adapter_type") or ""))
            if normalized and normalized in {row_id, row_type}:
                return {
                    "found": True,
                    "adapter": dict(row),
                }
        return {
            "found": False,
            "adapter_id": str(adapter_id or "").strip() or None,
        }

    def list_background_tasks(self) -> dict[str, Any]:
        payload = self.background_tasks_status()
        rows = payload.get("background_tasks") if isinstance(payload.get("background_tasks"), list) else []
        normalized_rows = [dict(row) for row in rows if isinstance(row, dict)]
        active_rows = [
            row
            for row in normalized_rows
            if bool(row.get("approved", False))
            and bool(row.get("enabled", False))
        ]
        return {
            "background_tasks": normalized_rows,
            "active_tasks": active_rows,
            "active_task_ids": [
                str(row.get("task_id") or "").strip()
                for row in active_rows
                if str(row.get("task_id") or "").strip()
            ],
        }

    def get_background_task_status(self, task_id: str | None) -> dict[str, Any]:
        normalized = self._normalize_key(task_id)
        tasks_payload = self.list_background_tasks()
        rows = tasks_payload.get("background_tasks") if isinstance(tasks_payload, dict) else []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            row_id = self._normalize_key(str(row.get("task_id") or ""))
            if normalized and normalized == row_id:
                return {
                    "found": True,
                    "task": dict(row),
                }
        return {
            "found": False,
            "task_id": str(task_id or "").strip() or None,
        }

    def list_governance_blocks(self) -> dict[str, Any]:
        payload = self.skill_governance_status()
        rows = payload.get("skills") if isinstance(payload.get("skills"), list) else []
        blocked = [
            dict(row)
            for row in rows
            if isinstance(row, dict)
            and not bool(row.get("allowed", False))
            and not bool(row.get("requires_user_approval", False))
        ]
        return {
            "blocked_skills": blocked,
        }

    def list_pending_governance_requests(self) -> dict[str, Any]:
        status = self.skill_governance_status()
        skill_rows = status.get("skills") if isinstance(status.get("skills"), list) else []
        adapter_rows = status.get("managed_adapters") if isinstance(status.get("managed_adapters"), list) else []
        task_rows = status.get("background_tasks") if isinstance(status.get("background_tasks"), list) else []
        pending_skills = [
            dict(row)
            for row in skill_rows
            if isinstance(row, dict)
            and not bool(row.get("allowed", False))
            and bool(row.get("requires_user_approval", False))
        ]
        pending_adapters = [
            dict(row)
            for row in adapter_rows
            if isinstance(row, dict) and not bool(row.get("approved", False))
        ]
        pending_tasks = [
            dict(row)
            for row in task_rows
            if isinstance(row, dict) and not bool(row.get("approved", False))
        ]
        return {
            "pending_skills": pending_skills,
            "pending_adapters": pending_adapters,
            "pending_background_tasks": pending_tasks,
        }

    def get_skill_governance_status(self, skill_id: str | None) -> dict[str, Any]:
        normalized = self._normalize_key(skill_id)
        payload = self.skill_governance_status()
        rows = payload.get("skills") if isinstance(payload.get("skills"), list) else []
        if not normalized:
            return {
                "found": False,
                "needs_skill_id": True,
            }
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            row_id = self._normalize_key(str(row.get("skill_id") or ""))
            if normalized == row_id:
                return {
                    "found": True,
                    "skill": dict(row),
                }
        return {
            "found": False,
            "skill_id": str(skill_id or "").strip() or None,
        }

    def _default_chat_policy_selection(
        self,
        *,
        candidate_model_ids: list[str] | None = None,
        allowed_tiers: tuple[str, ...] | None = None,
        require_auth: bool = True,
    ) -> dict[str, Any]:
        if bool(callable(getattr(self.runtime, "_safe_mode_enabled", None)) and self.runtime._safe_mode_enabled()):
            safe_mode_target = (
                self.runtime.safe_mode_target_status()
                if callable(getattr(self.runtime, "safe_mode_target_status", None))
                else {}
            )
            pinned_model = str(safe_mode_target.get("effective_model") or "").strip() or None
            pinned_provider = str(safe_mode_target.get("effective_provider") or "").strip().lower() or None
            pinned_local = bool(safe_mode_target.get("effective_local", True))
            if pinned_model and pinned_provider:
                candidate = {
                    "model_id": pinned_model,
                    "provider_id": pinned_provider,
                    "tier": "local" if pinned_local else "free_remote",
                    "local": pinned_local,
                    "approved": True,
                    "expected_cost_per_1m": 0.0 if pinned_local else None,
                    "utility": 0.0,
                    "selection_reason": str(safe_mode_target.get("reason") or "safe_mode_pinned_target"),
                }
                tier_key = "local" if pinned_local else "free_remote"
                decision_detail = str(safe_mode_target.get("message") or "").strip() or f"Safe mode pins chat to {pinned_model}."
                return {
                    "policy_name": "safe_mode_pinned_target",
                    "local_first": True,
                    "tier_order": ["local", "free_remote", "cheap_remote"],
                    "general_remote_cap_per_1m": 0.0,
                    "cheap_remote_cap_per_1m": 0.0,
                    "current_candidate": dict(candidate),
                    "selected_candidate": dict(candidate),
                    "recommended_candidate": dict(candidate),
                    "switch_recommended": False,
                    "decision_reason": str(safe_mode_target.get("reason") or "safe_mode_pinned_target"),
                    "decision_detail": decision_detail,
                    "utility_delta": 0.0,
                    "min_improvement": 0.0,
                    "tier_candidates": {tier_key: dict(candidate)},
                    "ordered_candidates": [dict(candidate)],
                    "rejected_candidates": [],
                }
        return self.select_chat_candidates(
            candidate_model_ids=candidate_model_ids,
            allowed_tiers=allowed_tiers,
            require_auth=require_auth,
        )

    @staticmethod
    def _model_scout_recommendation_reason(
        current: dict[str, Any] | None,
        candidate: dict[str, Any] | None,
        *,
        min_improvement: float,
    ) -> str | None:
        if not isinstance(candidate, dict):
            return None
        if not isinstance(current, dict):
            return "current_unavailable"
        current_model = str(current.get("model_id") or "").strip()
        candidate_model = str(candidate.get("model_id") or "").strip()
        if not candidate_model or candidate_model == current_model:
            return None
        if not bool(current.get("usable_now", False)):
            return "current_unavailable"

        current_quality = int(current.get("quality_rank") or 0)
        candidate_quality = int(candidate.get("quality_rank") or 0)
        current_context = int(current.get("context_window") or 0)
        candidate_context = int(candidate.get("context_window") or 0)
        current_cost = float(current.get("expected_cost_per_1m") or 0.0)
        candidate_cost = float(candidate.get("expected_cost_per_1m") or 0.0)
        current_utility = float(current.get("utility") or 0.0)
        candidate_utility = float(candidate.get("utility") or 0.0)

        if candidate_quality >= current_quality + 2 and candidate_cost <= current_cost + 0.5:
            return "quality_upgrade"
        if candidate_context >= max(current_context + 16384, int(current_context * 1.5)) and candidate_quality >= current_quality:
            return "context_headroom"
        if candidate_cost + 0.25 < current_cost and candidate_quality >= current_quality:
            return "cost_reduction"
        if candidate_utility >= current_utility + float(min_improvement):
            return "overall_fit"
        return None

    @staticmethod
    def _model_scout_reason_message(
        reason: str | None,
        *,
        candidate: dict[str, Any] | None = None,
    ) -> str:
        normalized = str(reason or "").strip().lower()
        if normalized == "best_local":
            if isinstance(candidate, dict) and bool(candidate.get("comfortable_local", False)):
                return "strongest local option that still fits comfortably on this machine"
            return "strongest local option currently available"
        if normalized == "cheap_remote_value":
            if isinstance(candidate, dict) and str(candidate.get("tier") or "").strip().lower() == "free_remote":
                return "lowest-cost remote option currently available"
            return "lower-cost remote option for general use"
        if normalized == "premium_coding_tier":
            return "qualifies for the premium coding tier"
        if normalized == "premium_research_tier":
            return "meets the premium quality and large-context requirements for research"
        if normalized == "best_task_chat":
            return "strongest available option currently visible for chat"
        if normalized == "best_task_coding":
            return "strongest available option currently visible for coding"
        if normalized == "best_task_research":
            return "best available research option currently visible"
        if normalized == "current_unavailable":
            return "the current model is not fully usable right now"
        if normalized == "quality_upgrade":
            return "higher-quality option than the current model"
        if normalized == "context_headroom":
            return "more context headroom than the current model"
        if normalized == "cost_reduction":
            return "lower-cost option that still qualifies for general use"
        return "strongest practical option currently visible"

    @staticmethod
    def _model_scout_comparison_message(
        state: str | None,
        basis: str | None,
    ) -> str:
        normalized_basis = str(basis or "").strip().lower()
        normalized_state = str(state or "").strip().lower()
        if normalized_basis == "same_as_current":
            return "already using this model"
        if normalized_basis == "higher_premium_role_than_current":
            return "upgrade for coding quality"
        if normalized_basis == "larger_context_research_fit_than_current":
            return "upgrade for research quality and context"
        if normalized_basis == "stronger_task_fit_than_current":
            return "upgrade for this task"
        if normalized_basis == "lower_cost_alternative":
            return "alternative option, not a clear overall upgrade"
        if normalized_basis == "current_already_best_local":
            return "already using the best local option"
        if normalized_basis == "stronger_local_than_current":
            return "upgrade within the local options"
        if normalized_basis == "weaker_local_alternative":
            return "local-first alternative, not a clear overall upgrade"
        if normalized_basis == "no_meaningful_difference":
            return "alternative option, not a clear overall upgrade"
        if normalized_basis == "weaker_than_current":
            return "weaker overall; treat it as a situational alternative"
        if normalized_state == "upgrade":
            return "upgrade over the current model"
        if normalized_state == "downgrade":
            return "not a clear replacement for the current model"
        if normalized_state == "lateral":
            return "alternative option, not a clear overall upgrade"
        return "not directly comparable right now"

    def _model_scout_candidate_comparison(
        self,
        *,
        current: dict[str, Any] | None,
        candidate: dict[str, Any] | None,
        role_key: str | None,
    ) -> dict[str, Any] | None:
        if not isinstance(candidate, dict):
            return None
        candidate_model_id = str(candidate.get("model_id") or "").strip()
        if not candidate_model_id:
            return None
        current_model_id = str((current or {}).get("model_id") or "").strip()
        current_quality = int((current or {}).get("quality_rank") or 0)
        candidate_quality = int(candidate.get("quality_rank") or 0)
        current_context = int((current or {}).get("context_window") or 0)
        candidate_context = int(candidate.get("context_window") or 0)
        current_local = bool((current or {}).get("local", False))
        normalized_role = str(role_key or candidate.get("recommendation_basis") or "").strip().lower()

        if current_model_id and candidate_model_id == current_model_id:
            state = "lateral"
            basis = "same_as_current"
        elif not current_model_id:
            state = "not_comparable"
            basis = "not_comparable"
        elif normalized_role == "cheap_cloud":
            state = "lateral"
            basis = "lower_cost_alternative"
        elif normalized_role in {"premium_coding", "premium_coding_tier", "best_task_coding"}:
            if current_quality < 7 or candidate_quality > current_quality:
                state = "upgrade"
                basis = (
                    "higher_premium_role_than_current"
                    if normalized_role in {"premium_coding", "premium_coding_tier"}
                    else "stronger_task_fit_than_current"
                )
            elif candidate_quality + 2 <= current_quality:
                state = "downgrade"
                basis = "weaker_than_current"
            else:
                state = "lateral"
                basis = "no_meaningful_difference"
        elif normalized_role in {"premium_research", "premium_research_tier", "best_task_research"}:
            if (
                current_quality < 7
                or current_context < 131072
                or candidate_quality > current_quality
                or candidate_context > current_context
            ):
                state = "upgrade"
                basis = "larger_context_research_fit_than_current"
            elif candidate_quality + 2 <= current_quality and candidate_context <= current_context:
                state = "downgrade"
                basis = "weaker_than_current"
            else:
                state = "lateral"
                basis = "no_meaningful_difference"
        elif normalized_role == "best_local":
            current_fit_state = str((current or {}).get("local_fit_state") or "").strip().lower()
            candidate_fit_state = str(candidate.get("local_fit_state") or "").strip().lower()
            if current_local and candidate_quality <= current_quality and candidate_fit_state == current_fit_state:
                state = "lateral"
                basis = "no_meaningful_difference"
            elif candidate_quality > current_quality or (
                current_local and candidate_fit_state == "comfortable" and current_fit_state in {"tight", "memory_starved"}
            ):
                state = "upgrade"
                basis = "stronger_local_than_current"
            elif not current_local and candidate_quality + 2 <= current_quality:
                state = "downgrade"
                basis = "weaker_local_alternative"
            else:
                state = "lateral"
                basis = "weaker_local_alternative"
        elif normalized_role in {"best_task_chat", "best_task_coding", "best_task_research"}:
            if candidate_quality > current_quality or candidate_context > current_context:
                state = "upgrade"
                basis = "stronger_task_fit_than_current"
            elif candidate_quality + 2 <= current_quality and candidate_context <= current_context:
                state = "downgrade"
                basis = "weaker_than_current"
            else:
                state = "lateral"
                basis = "no_meaningful_difference"
        else:
            state = "not_comparable"
            basis = "not_comparable"

        return {
            "state": state,
            "basis": basis,
            "explanation": self._model_scout_comparison_message(state, basis),
        }

    def _annotate_model_scout_candidate(
        self,
        candidate: dict[str, Any] | None,
        *,
        role: str | None = None,
        recommendation_reason: str | None = None,
        recommendation_basis: str | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(candidate, dict):
            return None
        payload = dict(candidate)
        if role:
            payload["role"] = role
        if recommendation_reason:
            payload["recommendation_reason"] = recommendation_reason
        basis = str(recommendation_basis or recommendation_reason or "").strip().lower()
        if basis:
            payload["recommendation_basis"] = basis
            payload["recommendation_explanation"] = self._model_scout_reason_message(
                basis,
                candidate=payload,
            )
        elif not str(payload.get("recommendation_explanation") or "").strip():
            payload["recommendation_explanation"] = self._model_scout_reason_message(
                recommendation_reason,
                candidate=payload,
            )
        return payload

    @staticmethod
    def _model_scout_resolution_payload(
        *,
        state: str,
        candidate: dict[str, Any] | None = None,
        reason_code: str | None = None,
        explanation: str | None = None,
        comparison: dict[str, Any] | None = None,
        advisory_actions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": str(state or "unavailable").strip().lower() or "unavailable",
        }
        if isinstance(candidate, dict):
            model_id = str(candidate.get("model_id") or "").strip()
            if model_id:
                payload["model_id"] = model_id
            provider_id = str(candidate.get("provider_id") or "").strip().lower()
            if provider_id:
                payload["provider_id"] = provider_id
            basis = str(candidate.get("recommendation_basis") or "").strip().lower()
            if basis:
                payload["recommendation_basis"] = basis
            candidate_role = str(candidate.get("role") or "").strip().lower()
            if candidate_role:
                payload["candidate_role"] = candidate_role
            explanation_text = str(explanation or candidate.get("recommendation_explanation") or "").strip()
            if explanation_text:
                payload["explanation"] = explanation_text
        else:
            explanation_text = str(explanation or "").strip()
            if explanation_text:
                payload["explanation"] = explanation_text
        normalized_reason_code = str(reason_code or "").strip().lower()
        if normalized_reason_code:
            payload["reason_code"] = normalized_reason_code
        if isinstance(comparison, dict):
            payload["comparison"] = dict(comparison)
        if isinstance(advisory_actions, dict):
            payload["advisory_actions"] = dict(advisory_actions)
        return payload

    @staticmethod
    def _model_scout_action_payload(
        *,
        state: str,
        reason_code: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": str(state or "not_applicable").strip().lower() or "not_applicable",
        }
        normalized_reason_code = str(reason_code or "").strip().lower()
        if normalized_reason_code:
            payload["reason_code"] = normalized_reason_code
        if payload["state"] == "available":
            payload["approval_required"] = True
        return payload

    def _model_scout_advisory_actions(
        self,
        *,
        resolution_state: str,
        resolution_reason_code: str | None,
        candidate: dict[str, Any] | None,
        current_model_id: str | None,
        default_model_id: str | None,
        allow_remote_switch: bool,
        allow_install_pull: bool,
        safe_mode: bool,
    ) -> dict[str, Any]:
        normalized_state = str(resolution_state or "").strip().lower()
        normalized_reason_code = str(resolution_reason_code or "").strip().lower()
        action_names = ("test", "switch_temporarily", "make_default", "acquire")
        if normalized_state == "blocked_by_mode":
            block_code = normalized_reason_code or ("safe_mode_remote_block" if safe_mode else "remote_switch_disabled")
            return {
                name: self._model_scout_action_payload(state="blocked", reason_code=block_code)
                for name in action_names
            }
        if normalized_state != "selected" or not isinstance(candidate, dict):
            return {
                name: self._model_scout_action_payload(
                    state="not_applicable",
                    reason_code="no_selected_candidate",
                )
                for name in action_names
            }

        candidate_model_id = str(candidate.get("model_id") or "").strip()
        candidate_local = bool(candidate.get("local", False))
        current_id = str(current_model_id or "").strip()
        default_id = str(default_model_id or "").strip()
        remote_switch_blocked = (not candidate_local) and (not allow_remote_switch)
        remote_block_code = "safe_mode_remote_block" if safe_mode else "remote_switch_disabled"
        install_block_code = "safe_mode_install_block" if safe_mode else "install_disabled_by_policy"

        if remote_switch_blocked:
            test_action = self._model_scout_action_payload(state="blocked", reason_code=remote_block_code)
            switch_action = self._model_scout_action_payload(state="blocked", reason_code=remote_block_code)
            default_action = self._model_scout_action_payload(state="blocked", reason_code=remote_block_code)
        else:
            if candidate_model_id and candidate_model_id == current_id:
                test_action = self._model_scout_action_payload(
                    state="not_applicable",
                    reason_code="already_current_model",
                )
                switch_action = self._model_scout_action_payload(
                    state="not_applicable",
                    reason_code="already_current_model",
                )
            else:
                test_action = self._model_scout_action_payload(state="available")
                switch_action = self._model_scout_action_payload(state="available")
            if candidate_model_id and candidate_model_id == default_id:
                default_action = self._model_scout_action_payload(
                    state="not_applicable",
                    reason_code="already_default_model",
                )
            else:
                default_action = self._model_scout_action_payload(state="available")

        acquisition_state = str(candidate.get("acquisition_state") or "").strip().lower()
        if acquisition_state in {"acquirable", "installed_not_ready", "queued", "downloading"}:
            if allow_install_pull:
                acquire_action = self._model_scout_action_payload(state="available")
            else:
                acquire_action = self._model_scout_action_payload(state="blocked", reason_code=install_block_code)
        elif candidate_local:
            acquire_action = self._model_scout_action_payload(
                state="not_applicable",
                reason_code="local_model_no_acquire_needed",
            )
        else:
            acquire_action = self._model_scout_action_payload(
                state="not_applicable",
                reason_code="already_available_remote",
            )

        return {
            "test": test_action,
            "switch_temporarily": switch_action,
            "make_default": default_action,
            "acquire": acquire_action,
        }

    def _model_scout_remote_role_resolution(
        self,
        *,
        role_key: str,
        candidate: dict[str, Any] | None,
        current_candidate: dict[str, Any] | None,
        default_model_id: str | None,
        allow_remote_recommendation: bool,
        allow_remote_switch: bool,
        allow_install_pull: bool,
        safe_mode: bool,
    ) -> dict[str, Any]:
        if isinstance(candidate, dict):
            return self._model_scout_resolution_payload(
                state="selected",
                candidate=candidate,
                comparison=self._model_scout_candidate_comparison(
                    current=current_candidate,
                    candidate=candidate,
                    role_key=role_key,
                ),
                advisory_actions=self._model_scout_advisory_actions(
                    resolution_state="selected",
                    resolution_reason_code=None,
                    candidate=candidate,
                    current_model_id=str((current_candidate or {}).get("model_id") or "").strip() or None,
                    default_model_id=default_model_id,
                    allow_remote_switch=allow_remote_switch,
                    allow_install_pull=allow_install_pull,
                    safe_mode=safe_mode,
                ),
            )
        blocked_reason_code = "safe_mode_remote_block" if safe_mode else "remote_recommendation_disabled"
        if not allow_remote_recommendation:
            return self._model_scout_resolution_payload(
                state="blocked_by_mode",
                reason_code=blocked_reason_code,
                explanation=(
                    "remote recommendations are not usable in this mode"
                    if safe_mode
                    else "remote recommendations are disabled by policy"
                ),
                advisory_actions=self._model_scout_advisory_actions(
                    resolution_state="blocked_by_mode",
                    resolution_reason_code=blocked_reason_code,
                    candidate=None,
                    current_model_id=str((current_candidate or {}).get("model_id") or "").strip() or None,
                    default_model_id=default_model_id,
                    allow_remote_switch=allow_remote_switch,
                    allow_install_pull=allow_install_pull,
                    safe_mode=safe_mode,
                ),
            )
        if role_key == "cheap_cloud":
            return self._model_scout_resolution_payload(
                state="no_qualifying_candidate",
                reason_code="no_cheap_remote_candidate",
                explanation="no usable low-cost remote model is currently available",
                advisory_actions=self._model_scout_advisory_actions(
                    resolution_state="no_qualifying_candidate",
                    resolution_reason_code="no_cheap_remote_candidate",
                    candidate=None,
                    current_model_id=str((current_candidate or {}).get("model_id") or "").strip() or None,
                    default_model_id=default_model_id,
                    allow_remote_switch=allow_remote_switch,
                    allow_install_pull=allow_install_pull,
                    safe_mode=safe_mode,
                ),
            )
        if role_key == "premium_coding":
            return self._model_scout_resolution_payload(
                state="no_qualifying_candidate",
                reason_code="premium_coding_threshold_unmet",
                explanation="no remote model currently meets the required premium quality threshold",
                advisory_actions=self._model_scout_advisory_actions(
                    resolution_state="no_qualifying_candidate",
                    resolution_reason_code="premium_coding_threshold_unmet",
                    candidate=None,
                    current_model_id=str((current_candidate or {}).get("model_id") or "").strip() or None,
                    default_model_id=default_model_id,
                    allow_remote_switch=allow_remote_switch,
                    allow_install_pull=allow_install_pull,
                    safe_mode=safe_mode,
                ),
            )
        return self._model_scout_resolution_payload(
            state="no_qualifying_candidate",
            reason_code="premium_research_threshold_unmet",
            explanation="no remote model currently meets the required premium quality and context thresholds",
            advisory_actions=self._model_scout_advisory_actions(
                resolution_state="no_qualifying_candidate",
                resolution_reason_code="premium_research_threshold_unmet",
                candidate=None,
                current_model_id=str((current_candidate or {}).get("model_id") or "").strip() or None,
                default_model_id=default_model_id,
                allow_remote_switch=allow_remote_switch,
                allow_install_pull=allow_install_pull,
                safe_mode=safe_mode,
            ),
        )

    def _model_scout_best_task_role_resolution(
        self,
        *,
        role_key: str,
        task_request: dict[str, Any],
        active_model: str | None,
        current_candidate: dict[str, Any] | None,
        default_model_id: str | None,
        candidate_lookup: dict[str, Any],
        candidate_model_ids: list[str],
        allow_remote_recommendation: bool,
        allow_remote_switch: bool,
        allow_install_pull: bool,
        safe_mode: bool,
        select_candidates_fn: Any | None = None,
        role_candidates_resolver: Any | None = None,
    ) -> dict[str, Any]:
        task_basis = str(role_key or "").strip().lower()
        if task_basis not in {"best_task_chat", "best_task_coding", "best_task_research"}:
            return self._model_scout_resolution_payload(
                state="unavailable",
                reason_code="no_usable_candidate",
                explanation="no usable model is currently available for this task",
            )
        if not candidate_model_ids:
            return self._model_scout_resolution_payload(
                state="unavailable",
                reason_code="no_usable_candidate",
                explanation="no usable model is currently available for this task",
            )
        allowed_tiers = (
            ("local",)
            if not allow_remote_recommendation
            else ("local", "free_remote", "cheap_remote", "remote")
        )
        selection_getter = select_candidates_fn if callable(select_candidates_fn) else self.select_chat_candidates
        selection = selection_getter(
            candidate_model_ids=candidate_model_ids,
            allowed_tiers=allowed_tiers,
            current_model_id=active_model,
            allow_remote_fallback_override=allow_remote_recommendation,
            require_auth=True,
            task_request=task_request,
        )
        if callable(role_candidates_resolver):
            role_candidates = role_candidates_resolver(task_request)
        else:
            role_candidates = self._model_scout_role_candidates(
                active_model=active_model,
                allow_remote_recommendation=allow_remote_recommendation,
                task_request=task_request,
                select_candidates_fn=selection_getter,
            )
        selected = self._model_scout_task_recommendation(
            task_request=task_request,
            active_model=active_model,
            selection=selection,
            role_candidates=role_candidates,
        )
        if not isinstance(selected, dict):
            selected = (
                selection.get("selected_candidate")
                if isinstance(selection.get("selected_candidate"), dict)
                else selection.get("current_candidate")
            )
        annotated = self._annotate_model_scout_candidate(
            dict(selected) if isinstance(selected, dict) else None,
            role="current_task",
            recommendation_basis=task_basis,
        )
        selected_model_id = str((annotated or {}).get("model_id") or "").strip()
        resolved_candidate = (
            {
                **dict(candidate_lookup.get(selected_model_id) or {}),
                **dict(annotated),
            }
            if isinstance(annotated, dict) and selected_model_id
            else annotated
        )
        if isinstance(annotated, dict):
            return self._model_scout_resolution_payload(
                state="selected",
                candidate=resolved_candidate if isinstance(resolved_candidate, dict) else annotated,
                comparison=self._model_scout_candidate_comparison(
                    current=current_candidate,
                    candidate=resolved_candidate if isinstance(resolved_candidate, dict) else annotated,
                    role_key=role_key,
                ),
                advisory_actions=self._model_scout_advisory_actions(
                    resolution_state="selected",
                    resolution_reason_code=None,
                    candidate=resolved_candidate if isinstance(resolved_candidate, dict) else annotated,
                    current_model_id=str((current_candidate or {}).get("model_id") or "").strip() or None,
                    default_model_id=default_model_id,
                    allow_remote_switch=allow_remote_switch,
                    allow_install_pull=allow_install_pull,
                    safe_mode=safe_mode,
                ),
            )
        return self._model_scout_resolution_payload(
            state="unavailable",
            reason_code="no_usable_candidate",
            explanation="no usable model is currently available for this task",
            advisory_actions=self._model_scout_advisory_actions(
                resolution_state="unavailable",
                resolution_reason_code="no_usable_candidate",
                candidate=None,
                current_model_id=str((current_candidate or {}).get("model_id") or "").strip() or None,
                default_model_id=default_model_id,
                allow_remote_switch=allow_remote_switch,
                allow_install_pull=allow_install_pull,
                safe_mode=safe_mode,
            ),
        )

    def _model_scout_recommendation_roles(
        self,
        *,
        active_model: str | None,
        current_candidate: dict[str, Any] | None,
        default_model_id: str | None,
        candidate_lookup: dict[str, Any],
        candidate_model_ids: list[str],
        allow_remote_recommendation: bool,
        allow_remote_switch: bool,
        allow_install_pull: bool,
        safe_mode: bool,
        role_candidates: dict[str, Any],
        select_candidates_fn: Any | None = None,
        role_candidates_resolver: Any | None = None,
        included_role_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        included = {str(item).strip().lower() for item in included_role_keys or set() if str(item).strip()}
        roles: dict[str, Any] = {}
        best_local_candidate = role_candidates.get("comfortable_local_default")
        best_local_model_id = str((best_local_candidate or {}).get("model_id") or "").strip()
        resolved_best_local = (
            {
                **dict(candidate_lookup.get(best_local_model_id) or {}),
                **dict(best_local_candidate),
            }
            if isinstance(best_local_candidate, dict) and best_local_model_id
            else best_local_candidate
        )
        if not included or "best_local" in included:
            roles["best_local"] = (
                self._model_scout_resolution_payload(
                    state="selected",
                    candidate=resolved_best_local if isinstance(resolved_best_local, dict) else best_local_candidate,
                    comparison=self._model_scout_candidate_comparison(
                        current=current_candidate,
                        candidate=resolved_best_local if isinstance(resolved_best_local, dict) else best_local_candidate,
                        role_key="best_local",
                    ),
                    advisory_actions=self._model_scout_advisory_actions(
                        resolution_state="selected",
                        resolution_reason_code=None,
                        candidate=resolved_best_local if isinstance(resolved_best_local, dict) else best_local_candidate,
                        current_model_id=str((current_candidate or {}).get("model_id") or "").strip() or None,
                        default_model_id=default_model_id,
                        allow_remote_switch=allow_remote_switch,
                        allow_install_pull=allow_install_pull,
                        safe_mode=safe_mode,
                    ),
                )
                if isinstance(best_local_candidate, dict)
                else self._model_scout_resolution_payload(
                    state="unavailable",
                    reason_code="no_local_candidate",
                    explanation="no usable local model is currently available",
                    advisory_actions=self._model_scout_advisory_actions(
                        resolution_state="unavailable",
                        resolution_reason_code="no_local_candidate",
                        candidate=None,
                        current_model_id=str((current_candidate or {}).get("model_id") or "").strip() or None,
                        default_model_id=default_model_id,
                        allow_remote_switch=allow_remote_switch,
                        allow_install_pull=allow_install_pull,
                        safe_mode=safe_mode,
                    ),
                )
            )
        if not included or "cheap_cloud" in included:
            roles["cheap_cloud"] = self._model_scout_remote_role_resolution(
                role_key="cheap_cloud",
                candidate=role_candidates.get("cheap_cloud") if isinstance(role_candidates.get("cheap_cloud"), dict) else None,
                current_candidate=current_candidate,
                default_model_id=default_model_id,
                allow_remote_recommendation=allow_remote_recommendation,
                allow_remote_switch=allow_remote_switch,
                allow_install_pull=allow_install_pull,
                safe_mode=safe_mode,
            )
        if not included or "premium_coding" in included:
            roles["premium_coding"] = self._model_scout_remote_role_resolution(
                role_key="premium_coding",
                candidate=(
                    role_candidates.get("premium_coding_cloud")
                    if isinstance(role_candidates.get("premium_coding_cloud"), dict)
                    else None
                ),
                current_candidate=current_candidate,
                default_model_id=default_model_id,
                allow_remote_recommendation=allow_remote_recommendation,
                allow_remote_switch=allow_remote_switch,
                allow_install_pull=allow_install_pull,
                safe_mode=safe_mode,
            )
        if not included or "premium_research" in included:
            roles["premium_research"] = self._model_scout_remote_role_resolution(
                role_key="premium_research",
                candidate=(
                    role_candidates.get("premium_research_cloud")
                    if isinstance(role_candidates.get("premium_research_cloud"), dict)
                    else None
                ),
                current_candidate=current_candidate,
                default_model_id=default_model_id,
                allow_remote_recommendation=allow_remote_recommendation,
                allow_remote_switch=allow_remote_switch,
                allow_install_pull=allow_install_pull,
                safe_mode=safe_mode,
            )

        task_role_requests = {
            "best_task_chat": {
                "task_type": "chat",
                "requirements": ["chat"],
                "preferred_local": True,
            },
            "best_task_coding": {
                "task_type": "coding",
                "requirements": ["chat"],
                "preferred_local": True,
            },
            "best_task_research": {
                "task_type": "reasoning",
                "requirements": ["chat", "long_context"],
                "preferred_local": True,
            },
        }
        for role_key, task_request in task_role_requests.items():
            if included and role_key not in included:
                continue
            roles[role_key] = self._model_scout_best_task_role_resolution(
                role_key=role_key,
                task_request=task_request,
                active_model=active_model,
                current_candidate=current_candidate,
                default_model_id=default_model_id,
                candidate_lookup=candidate_lookup,
                candidate_model_ids=candidate_model_ids,
                allow_remote_recommendation=allow_remote_recommendation,
                allow_remote_switch=allow_remote_switch,
                allow_install_pull=allow_install_pull,
                safe_mode=safe_mode,
                select_candidates_fn=select_candidates_fn,
                role_candidates_resolver=role_candidates_resolver,
            )
        return roles

    @staticmethod
    def _scout_role_candidate(
        selection: dict[str, Any],
        *,
        prefer_tiers: tuple[str, ...],
        require_comfortable_local: bool = False,
    ) -> dict[str, Any] | None:
        ordered = selection.get("ordered_candidates") if isinstance(selection.get("ordered_candidates"), list) else []
        for tier in prefer_tiers:
            for row in ordered:
                if not isinstance(row, dict):
                    continue
                if str(row.get("tier") or "").strip().lower() != str(tier).strip().lower():
                    continue
                if require_comfortable_local and not bool(row.get("comfortable_local", False)):
                    continue
                return dict(row)
        tier_candidates = selection.get("tier_candidates") if isinstance(selection.get("tier_candidates"), dict) else {}
        for tier in prefer_tiers:
            candidate = tier_candidates.get(tier)
            if not isinstance(candidate, dict):
                continue
            if require_comfortable_local and not bool(candidate.get("comfortable_local", False)):
                continue
            return dict(candidate)
        return None

    @staticmethod
    def _premium_role_candidate(
        selection: dict[str, Any],
        *,
        task_type: str,
    ) -> dict[str, Any] | None:
        ordered = selection.get("ordered_candidates") if isinstance(selection.get("ordered_candidates"), list) else []
        normalized_task_type = str(task_type or "chat").strip().lower() or "chat"
        for row in ordered:
            if not isinstance(row, dict):
                continue
            if str(row.get("tier") or "").strip().lower() != "remote":
                continue
            quality_rank = int(row.get("quality_rank") or 0)
            context_window = int(row.get("context_window") or 0)
            if normalized_task_type == "coding":
                if quality_rank < 7:
                    continue
            elif normalized_task_type == "reasoning":
                if quality_rank < 7 or context_window < 131072:
                    continue
            return dict(row)
        return None

    def _model_scout_role_candidates(
        self,
        *,
        active_model: str | None,
        allow_remote_recommendation: bool,
        task_request: dict[str, Any],
        select_candidates_fn: Any | None = None,
        included_role_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        included = {str(item).strip().lower() for item in included_role_keys or set() if str(item).strip()}
        role_candidates: dict[str, Any] = {}
        selection_getter = select_candidates_fn if callable(select_candidates_fn) else self.select_chat_candidates
        if not included or included.intersection({"best_local", "best_task_chat", "best_task_coding", "best_task_research"}):
            local_selection = selection_getter(
                current_model_id=active_model,
                allowed_tiers=("local",),
                allow_remote_fallback_override=False,
                require_auth=True,
                task_request={**dict(task_request), "preferred_local": True},
            )
            local_candidate = self._scout_role_candidate(
                local_selection,
                prefer_tiers=("local",),
                require_comfortable_local=True,
            )
            if local_candidate is None:
                local_candidate = self._scout_role_candidate(
                    local_selection,
                    prefer_tiers=("local",),
            )
            role_candidates["comfortable_local_default"] = self._annotate_model_scout_candidate(
                local_candidate,
                role="comfortable_local_default",
                recommendation_basis="best_local",
            )
        else:
            role_candidates["comfortable_local_default"] = None

        if allow_remote_recommendation:
            if not included or "cheap_cloud" in included:
                cheap_cloud_selection = selection_getter(
                    current_model_id=active_model,
                    allowed_tiers=("free_remote", "cheap_remote"),
                    allow_remote_fallback_override=True,
                    require_auth=True,
                    task_request={
                        "task_type": "chat",
                        "requirements": ["chat"],
                        "preferred_local": False,
                    },
                )
                role_candidates["cheap_cloud"] = self._annotate_model_scout_candidate(
                    self._scout_role_candidate(
                        cheap_cloud_selection,
                        prefer_tiers=("free_remote", "cheap_remote"),
                    ),
                    role="cheap_cloud",
                    recommendation_basis="cheap_remote_value",
                )
            else:
                role_candidates["cheap_cloud"] = None

            premium_policy = (
                self.runtime.config.premium_policy
                if isinstance(getattr(self.runtime.config, "premium_policy", None), dict)
                else {}
            )
            if not included or included.intersection({"premium_coding", "best_task_coding"}):
                premium_coding_selection = selection_getter(
                    policy=premium_policy,
                    policy_name="premium",
                    current_model_id=active_model,
                    allowed_tiers=("remote",),
                    allow_remote_fallback_override=True,
                    require_auth=True,
                    task_request={
                        "task_type": "coding",
                        "requirements": ["chat"],
                        "preferred_local": False,
                    },
                )
                role_candidates["premium_coding_cloud"] = self._annotate_model_scout_candidate(
                    self._premium_role_candidate(
                        premium_coding_selection,
                        task_type="coding",
                    ),
                    role="premium_coding_cloud",
                    recommendation_reason="task_specialist_upgrade",
                    recommendation_basis="premium_coding_tier",
                )
            else:
                role_candidates["premium_coding_cloud"] = None
            if not included or included.intersection({"premium_research", "best_task_research"}):
                premium_research_selection = selection_getter(
                    policy=premium_policy,
                    policy_name="premium",
                    current_model_id=active_model,
                    allowed_tiers=("remote",),
                    allow_remote_fallback_override=True,
                    require_auth=True,
                    task_request={
                        "task_type": "reasoning",
                        "requirements": ["chat", "long_context"],
                        "preferred_local": False,
                    },
                )
                role_candidates["premium_research_cloud"] = self._annotate_model_scout_candidate(
                    self._premium_role_candidate(
                        premium_research_selection,
                        task_type="reasoning",
                    ),
                    role="premium_research_cloud",
                    recommendation_reason="task_specialist_upgrade",
                    recommendation_basis="premium_research_tier",
                )
            else:
                role_candidates["premium_research_cloud"] = None
        else:
            role_candidates["cheap_cloud"] = None
            role_candidates["premium_coding_cloud"] = None
            role_candidates["premium_research_cloud"] = None
        return role_candidates

    def _model_scout_task_recommendation(
        self,
        *,
        task_request: dict[str, Any],
        active_model: str | None,
        selection: dict[str, Any],
        role_candidates: dict[str, Any],
    ) -> dict[str, Any] | None:
        task_type = str(task_request.get("task_type") or "chat").strip().lower() or "chat"
        preferred_role = None
        if task_type == "coding":
            preferred_role = "premium_coding_cloud"
        elif task_type == "reasoning":
            preferred_role = "premium_research_cloud"

        if preferred_role:
            candidate = role_candidates.get(preferred_role)
            if isinstance(candidate, dict):
                candidate_id = str(candidate.get("model_id") or "").strip()
                if candidate_id and candidate_id != str(active_model or "").strip():
                    task_basis = "best_task_coding" if task_type == "coding" else "best_task_research"
                    return self._annotate_model_scout_candidate(
                        candidate,
                        role=preferred_role,
                        recommendation_reason="task_specialist_upgrade",
                        recommendation_basis=task_basis,
                    )

        recommended = selection.get("recommended_candidate")
        if isinstance(recommended, dict):
            if task_type == "coding":
                task_basis = "best_task_coding"
            elif task_type == "reasoning":
                task_basis = "best_task_research"
            elif bool(recommended.get("local", False)):
                task_basis = "best_local"
            else:
                task_basis = "best_task_chat"
            return self._annotate_model_scout_candidate(
                recommended,
                role="current_task",
                recommendation_reason=str(recommended.get("recommendation_reason") or "overall_fit"),
                recommendation_basis=task_basis,
            )
        return None

    def model_scout_v2_status(
        self,
        *,
        task_request: dict[str, Any] | None = None,
        included_role_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        inventory = self.model_inventory_status()
        readiness = self.model_readiness_status()
        normalized_task_request = self._normalize_chat_task_request(task_request)
        scoped_role_keys = {
            str(item).strip().lower()
            for item in (included_role_keys if isinstance(included_role_keys, list) else [])
            if str(item).strip()
        }
        canonical_candidate_inventory = self.canonical_chat_candidate_inventory()
        default_selection = self.select_chat_candidates(inventory_rows=canonical_candidate_inventory)
        target_truth = self.chat_target_truth(selection=default_selection)
        policy = self.model_controller_policy_status(target_truth=target_truth)
        allow_remote_fallback = bool(policy.get("allow_remote_fallback", True))
        allow_remote_recommendation = bool(
            policy.get("allow_remote_recommendation", allow_remote_fallback)
        )
        allow_remote_switch = bool(policy.get("allow_remote_switch", not bool(policy.get("safe_mode", False))))
        allow_install_pull = bool(policy.get("allow_install_pull", True))
        safe_mode = bool(policy.get("safe_mode", False))
        default_model_id = str(target_truth.get("configured_model") or "").strip() or None
        select_cache: dict[str, dict[str, Any]] = {}
        role_candidates_cache: dict[str, dict[str, Any]] = {}

        def _cached_select_chat_candidates(
            *,
            policy: dict[str, Any] | Any | None = None,
            policy_name: str = "default",
            candidate_model_ids: list[str] | None = None,
            allowed_tiers: tuple[str, ...] | None = None,
            min_improvement: float | None = None,
            current_model_id: str | None = None,
            allow_remote_fallback_override: bool | None = None,
            require_auth: bool = True,
            task_request: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            normalized_task = self._normalize_chat_task_request(task_request)
            cache_key = json.dumps(
                {
                    "policy_name": str(policy_name or "default"),
                    "policy": dict(policy) if isinstance(policy, dict) else policy,
                    "candidate_model_ids": list(candidate_model_ids or []),
                    "allowed_tiers": list(allowed_tiers or []),
                    "min_improvement": min_improvement,
                    "current_model_id": str(current_model_id or "").strip() or None,
                    "allow_remote_fallback_override": allow_remote_fallback_override,
                    "require_auth": bool(require_auth),
                    "task_request": normalized_task,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            cached = select_cache.get(cache_key)
            if isinstance(cached, dict):
                return dict(cached)
            resolved = self.select_chat_candidates(
                policy=policy,
                policy_name=policy_name,
                inventory_rows=canonical_candidate_inventory,
                candidate_model_ids=candidate_model_ids,
                allowed_tiers=allowed_tiers,
                min_improvement=min_improvement,
                current_model_id=current_model_id,
                allow_remote_fallback_override=allow_remote_fallback_override,
                require_auth=require_auth,
                task_request=normalized_task,
            )
            select_cache[cache_key] = dict(resolved)
            return dict(resolved)

        def _cached_role_candidates(task_request_payload: dict[str, Any]) -> dict[str, Any]:
            normalized_task = self._normalize_chat_task_request(task_request_payload)
            cache_key = json.dumps(
                {
                    "task_request": normalized_task,
                    "included_role_keys": sorted(scoped_role_keys),
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            cached = role_candidates_cache.get(cache_key)
            if isinstance(cached, dict):
                return dict(cached)
            resolved = self._model_scout_role_candidates(
                active_model=active_model,
                allow_remote_recommendation=allow_remote_recommendation,
                task_request=normalized_task,
                select_candidates_fn=_cached_select_chat_candidates,
                included_role_keys=scoped_role_keys or None,
            )
            role_candidates_cache[cache_key] = dict(resolved)
            return dict(resolved)
        rows = [
            dict(row)
            for row in (readiness.get("models") if isinstance(readiness.get("models"), list) else [])
            if isinstance(row, dict)
        ]
        if not rows:
            recommendation_roles = self._model_scout_recommendation_roles(
                active_model=str(target_truth.get("effective_model") or "").strip() or None,
                current_candidate=None,
                default_model_id=default_model_id,
                candidate_lookup={},
                candidate_model_ids=[],
                allow_remote_recommendation=allow_remote_recommendation,
                allow_remote_switch=allow_remote_switch,
                allow_install_pull=allow_install_pull,
                safe_mode=safe_mode,
                role_candidates={},
                included_role_keys=scoped_role_keys or None,
            )
            return {
                "type": "model_scout_v2",
                "active_provider": str(target_truth.get("effective_provider") or "").strip().lower() or None,
                "active_model": str(target_truth.get("effective_model") or "").strip() or None,
                "current_candidate": None,
                "recommended_candidate": None,
                "better_candidates": [],
                "candidate_rows": [],
                "not_ready_models": [],
                "policy": dict(policy),
                "task_request": dict(normalized_task_request),
                "role_candidates": {},
                "recommendation_roles": recommendation_roles,
                "task_recommendation": None,
                "selection": {
                    "ordered_candidates": [],
                    "rejected_candidates": [],
                    "switch_recommended": False,
                    "decision_reason": "no_candidate",
                    "decision_detail": "No chat-capable models are currently available.",
                },
                "source": "runtime_truth.model_scout_v2",
            }

        active_model = str(target_truth.get("effective_model") or readiness.get("active_model") or "").strip() or None
        active_provider = (
            str(target_truth.get("effective_provider") or readiness.get("active_provider") or "").strip().lower()
            or None
        )
        allowed_tiers = (
            ("local",)
            if not allow_remote_recommendation
            else ("local", "free_remote", "cheap_remote", "remote")
        )
        candidate_model_ids = [
            str(row.get("model_id") or "").strip()
            for row in rows
            if bool(row.get("usable_now", False))
            and str(row.get("model_id") or "").strip()
            and (allow_remote_recommendation or bool(row.get("local", False)))
        ]

        selection = _cached_select_chat_candidates(
            candidate_model_ids=candidate_model_ids,
            allowed_tiers=allowed_tiers,
            current_model_id=active_model,
            allow_remote_fallback_override=allow_remote_recommendation,
            require_auth=True,
            task_request=normalized_task_request,
        )
        role_candidates = _cached_role_candidates(normalized_task_request)

        summary_rows: list[dict[str, Any]] = []
        for key in ("current_candidate", "selected_candidate", "recommended_candidate"):
            row = selection.get(key)
            if isinstance(row, dict):
                summary_rows.append(dict(row))
        tier_candidates = selection.get("tier_candidates") if isinstance(selection.get("tier_candidates"), dict) else {}
        summary_rows.extend(
            dict(value)
            for value in tier_candidates.values()
            if isinstance(value, dict)
        )
        summary_rows.extend(
            dict(row)
            for row in (selection.get("ordered_candidates") if isinstance(selection.get("ordered_candidates"), list) else [])
            if isinstance(row, dict)
        )
        summary_by_model: dict[str, dict[str, Any]] = {}
        for row in summary_rows:
            model_id = str(row.get("model_id") or "").strip()
            if not model_id:
                continue
            summary_by_model[model_id] = dict(row)

        merged_rows: list[dict[str, Any]] = []
        for row in rows:
            model_id = str(row.get("model_id") or "").strip()
            summary = summary_by_model.get(model_id, {})
            merged_rows.append(
                {
                    **dict(row),
                    "tier": str(summary.get("tier") or "").strip() or None,
                    "utility": float(summary.get("utility") or 0.0),
                    "utility_quality": float(summary.get("utility_quality") or 0.0),
                    "utility_latency": float(summary.get("utility_latency") or 0.0),
                    "utility_risk": float(summary.get("utility_risk") or 0.0),
                    "expected_cost_per_1m": float(summary.get("expected_cost_per_1m") or 0.0),
                    "context_window": int(summary.get("context_window") or 0),
                    "auth_ok": bool(summary.get("auth_ok", True if bool(row.get("local", False)) else False)),
                }
            )
        merged_by_model = {
            str(row.get("model_id") or "").strip(): dict(row)
            for row in merged_rows
            if isinstance(row, dict) and str(row.get("model_id") or "").strip()
        }

        current_row = next(
            (
                dict(row)
                for row in merged_rows
                if str(row.get("model_id") or "").strip() == str(active_model or "")
            ),
            None,
        )
        ordered_selection_rows = [
            dict(row)
            for row in (selection.get("ordered_candidates") if isinstance(selection.get("ordered_candidates"), list) else [])
            if isinstance(row, dict)
        ]
        eligible_candidates: list[dict[str, Any]] = []
        seen_candidate_ids: set[str] = set()
        for summary in ordered_selection_rows:
            model_id = str(summary.get("model_id") or "").strip()
            if not model_id or model_id in seen_candidate_ids:
                continue
            merged = merged_by_model.get(model_id)
            if isinstance(merged, dict):
                seen_candidate_ids.add(model_id)
                eligible_candidates.append(dict(merged))
        if not eligible_candidates:
            eligible_candidates = [
                dict(row)
                for row in merged_rows
                if bool(row.get("usable_now", False))
                and bool(row.get("auth_ok", True))
                and (
                    bool(row.get("local", False))
                    or (
                        allow_remote_recommendation
                        and str(row.get("tier") or "").strip().lower() in {"free_remote", "cheap_remote"}
                    )
                )
            ]

        min_improvement = float(selection.get("min_improvement") or 0.08)
        better_candidates: list[dict[str, Any]] = []
        for row in eligible_candidates:
            reason = self._model_scout_recommendation_reason(current_row, row, min_improvement=min_improvement)
            if reason is None:
                continue
            better_candidates.append(
                {
                    **dict(row),
                    "recommendation_reason": reason,
                    "recommendation_explanation": self._model_scout_reason_message(reason),
                }
            )

        recommended_candidate = None
        selected_recommendation = (
            selection.get("recommended_candidate")
            if bool(selection.get("switch_recommended")) and isinstance(selection.get("recommended_candidate"), dict)
            else None
        )
        selected_recommendation_id = str((selected_recommendation or {}).get("model_id") or "").strip()
        if selected_recommendation_id:
            recommended_candidate = next(
                (
                    dict(row)
                    for row in better_candidates
                    if str(row.get("model_id") or "").strip() == selected_recommendation_id
                ),
                None,
            )
            if recommended_candidate is None:
                recommended_candidate = dict(merged_by_model.get(selected_recommendation_id) or {})
                if recommended_candidate:
                    recommendation_reason = self._model_scout_recommendation_reason(
                        current_row,
                        recommended_candidate,
                        min_improvement=min_improvement,
                    )
                    recommended_candidate["recommendation_reason"] = recommendation_reason or "overall_fit"
                    recommended_candidate["recommendation_explanation"] = self._model_scout_reason_message(
                        recommendation_reason
                    )
        if recommended_candidate is None and better_candidates:
            recommended_candidate = dict(better_candidates[0])
        current_candidate = (
            dict(current_row)
            if isinstance(current_row, dict)
            else (
                dict(selection.get("current_candidate"))
                if isinstance(selection.get("current_candidate"), dict)
                else None
            )
        )
        task_recommendation = self._model_scout_task_recommendation(
            task_request=normalized_task_request,
            active_model=active_model,
            selection=selection,
            role_candidates=role_candidates,
        )
        recommendation_roles = self._model_scout_recommendation_roles(
            active_model=active_model,
            current_candidate=current_candidate,
            default_model_id=default_model_id,
            candidate_lookup=merged_by_model,
            candidate_model_ids=candidate_model_ids,
            allow_remote_recommendation=allow_remote_recommendation,
            allow_remote_switch=allow_remote_switch,
            allow_install_pull=allow_install_pull,
            safe_mode=safe_mode,
            role_candidates=role_candidates,
            select_candidates_fn=_cached_select_chat_candidates,
            role_candidates_resolver=_cached_role_candidates,
            included_role_keys=scoped_role_keys or None,
        )
        return {
            "type": "model_scout_v2",
            "active_provider": active_provider,
            "active_model": active_model,
            "current_candidate": current_candidate,
            "recommended_candidate": recommended_candidate,
            "better_candidates": better_candidates[:3],
            "candidate_rows": eligible_candidates,
            "not_ready_models": [
                dict(row)
                for row in merged_rows
                if not bool(row.get("usable_now", False))
            ][:4],
            "inventory": dict(inventory),
            "readiness": dict(readiness),
            "policy": dict(policy),
            "task_request": dict(normalized_task_request),
            "role_candidates": dict(role_candidates),
            "recommendation_roles": dict(recommendation_roles),
            "task_recommendation": dict(task_recommendation) if isinstance(task_recommendation, dict) else None,
            "advisory_only": bool(policy.get("scout_advisory_only", safe_mode)),
            "selection": dict(selection),
            "source": "runtime_truth.model_scout_v2",
        }

    def model_watch_hf_status(self) -> dict[str, Any]:
        status = getattr(self.runtime, "model_watch_hf_status", None)
        if not callable(status):
            return {
                "ok": False,
                "enabled": False,
                "error": "hf_status_unavailable",
            }
        payload = status()
        return dict(payload) if isinstance(payload, dict) else {"ok": False, "enabled": False}

    def model_watch_hf_scan(
        self,
        *,
        trigger: str = "manual",
        notify_proposal: bool = False,
        persist_proposal: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        scanner = getattr(self.runtime, "model_watch_hf_scan", None)
        if not callable(scanner):
            return False, {"ok": False, "error": "hf_scan_unavailable"}
        ok, body = scanner(
            trigger=trigger,
            notify_proposal=notify_proposal,
            persist_proposal=persist_proposal,
        )
        return bool(ok), dict(body) if isinstance(body, dict) else {"ok": bool(ok)}

    def model_discovery_query(
        self,
        query: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._model_discovery_manager().query(query, filters)

    def model_policy_status(
        self,
        *,
        selection: dict[str, Any] | None = None,
        target_truth: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_selection = (
            dict(selection)
            if isinstance(selection, dict)
            else self._default_chat_policy_selection()
        )
        current_target = (
            dict(target_truth)
            if isinstance(target_truth, dict)
            else self.chat_target_truth(selection=resolved_selection)
        )
        tier_candidates = resolved_selection.get("tier_candidates") if isinstance(resolved_selection.get("tier_candidates"), dict) else {}
        return {
            "type": "model_policy_status",
            "policy_name": str(resolved_selection.get("policy_name") or "default"),
            "local_first": bool(resolved_selection.get("local_first", True)),
            "tier_order": list(resolved_selection.get("tier_order") or ["local", "free_remote", "cheap_remote"]),
            "general_remote_cap_per_1m": float(resolved_selection.get("general_remote_cap_per_1m") or 0.0),
            "cheap_remote_cap_per_1m": float(resolved_selection.get("cheap_remote_cap_per_1m") or 0.0),
            "current_active_model": str(current_target.get("effective_model") or "").strip() or None,
            "current_active_provider": str(current_target.get("effective_provider") or "").strip().lower() or None,
            "current_default_model": str(self._resolved_default_model() or "").strip() or None,
            "current_default_provider": str(self._default_provider_id() or "").strip().lower() or None,
            "current_candidate": dict(resolved_selection.get("current_candidate") or {}) if isinstance(resolved_selection.get("current_candidate"), dict) else None,
            "selected_candidate": dict(resolved_selection.get("selected_candidate") or {}) if isinstance(resolved_selection.get("selected_candidate"), dict) else None,
            "recommended_candidate": dict(resolved_selection.get("recommended_candidate") or {}) if isinstance(resolved_selection.get("recommended_candidate"), dict) else None,
            "switch_recommended": bool(resolved_selection.get("switch_recommended", False)),
            "decision_reason": str(resolved_selection.get("decision_reason") or "").strip() or None,
            "decision_detail": str(resolved_selection.get("decision_detail") or "").strip() or None,
            "utility_delta": float(resolved_selection.get("utility_delta") or 0.0),
            "min_improvement": float(resolved_selection.get("min_improvement") or 0.0),
            "tier_candidates": {
                key: dict(value)
                for key, value in tier_candidates.items()
                if isinstance(value, dict)
            },
            "ordered_candidates": [
                dict(row)
                for row in (resolved_selection.get("ordered_candidates") if isinstance(resolved_selection.get("ordered_candidates"), list) else [])
                if isinstance(row, dict)
            ],
            "rejected_candidates": [
                dict(row)
                for row in (resolved_selection.get("rejected_candidates") if isinstance(resolved_selection.get("rejected_candidates"), list) else [])
                if isinstance(row, dict)
            ],
        }

    def model_policy_candidate(
        self,
        tier: str | None = None,
        *,
        status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        status = dict(status) if isinstance(status, dict) else self.model_policy_status()
        tier_key = str(tier or "").strip().lower() or None
        if tier_key:
            tier_candidates = status.get("tier_candidates") if isinstance(status.get("tier_candidates"), dict) else {}
            candidate = tier_candidates.get(tier_key) if isinstance(tier_candidates.get(tier_key), dict) else None
            return {
                "type": "model_policy_candidate",
                "tier": tier_key,
                "found": bool(candidate),
                "candidate": dict(candidate) if isinstance(candidate, dict) else None,
                "cheap_remote_cap_per_1m": float(status.get("cheap_remote_cap_per_1m") or 0.0),
                "general_remote_cap_per_1m": float(status.get("general_remote_cap_per_1m") or 0.0),
                "selection": status,
            }
        candidate = (
            status.get("recommended_candidate")
            if isinstance(status.get("recommended_candidate"), dict)
            else status.get("selected_candidate")
        )
        return {
            "type": "model_policy_candidate",
            "tier": None,
            "found": bool(candidate),
            "candidate": dict(candidate) if isinstance(candidate, dict) else None,
            "switch_recommended": bool(status.get("switch_recommended", False)),
            "decision_reason": str(status.get("decision_reason") or "").strip() or None,
            "decision_detail": str(status.get("decision_detail") or "").strip() or None,
            "selection": status,
        }

    def model_policy_provider_candidate(self, provider_id: str | None) -> dict[str, Any]:
        provider_key = str(provider_id or "").strip().lower()
        provider_payload = self.provider_status(provider_key)
        model_ids = [
            str(row.get("model_id") or "").strip()
            for row in self._chat_capable_model_rows(provider_key)
            if str(row.get("model_id") or "").strip()
        ]
        if not provider_key:
            return {
                "type": "model_policy_candidate",
                "provider_id": None,
                "found": False,
                "provider_status": provider_payload,
                "selection": self.model_policy_status(),
            }
        allowed_tiers = ("local",) if bool(provider_payload.get("local", False)) else ("free_remote", "cheap_remote")
        selection = self._default_chat_policy_selection(
            candidate_model_ids=model_ids,
            allowed_tiers=allowed_tiers,
        ) if model_ids else {
            "recommended_candidate": None,
            "selected_candidate": None,
            "rejected_candidates": [],
            "decision_reason": "no_candidate",
            "decision_detail": "No provider candidate passed the default-switch policy.",
        }
        candidate = (
            selection.get("recommended_candidate")
            if isinstance(selection.get("recommended_candidate"), dict)
            else selection.get("selected_candidate")
        )
        return {
            "type": "model_policy_candidate",
            "provider_id": provider_key,
            "found": bool(candidate),
            "candidate": dict(candidate) if isinstance(candidate, dict) else None,
            "provider_status": provider_payload,
            "selection": self.model_policy_status(),
            "provider_selection": dict(selection),
        }

    def choose_best_local_chat_model(
        self,
        payload: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        return self.runtime.choose_best_local_chat_model(payload)

    def test_chat_model_target(
        self,
        model_id: str,
        *,
        provider_id: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        tester = getattr(self.runtime, "test_provider", None)
        if not callable(tester):
            return False, {
                "ok": False,
                "error": "provider_test_unavailable",
                "message": "I couldn't test that model from the current runtime.",
            }
        requested_model = str(model_id or "").strip()
        requested_provider = str(provider_id or "").strip().lower() or None
        if not requested_model:
            return False, {
                "ok": False,
                "error": "model_required",
                "message": "I need an exact model to test.",
            }
        if not requested_provider and ":" in requested_model:
            requested_provider = str(requested_model.split(":", 1)[0]).strip().lower() or None
        if not requested_provider:
            inventory = self.model_inventory_status()
            matches = [
                dict(row)
                for row in (
                    inventory.get("models")
                    if isinstance(inventory.get("models"), list)
                    else []
                )
                if isinstance(row, dict)
                and (
                    str(row.get("model_id") or "").strip() == requested_model
                    or (
                        ":" in str(row.get("model_id") or "")
                        and str(row.get("model_id") or "").split(":", 1)[1].strip() == requested_model
                    )
                )
            ]
            provider_ids = {
                str(row.get("provider_id") or "").strip().lower()
                for row in matches
                if str(row.get("provider_id") or "").strip()
            }
            if len(provider_ids) == 1:
                requested_provider = next(iter(provider_ids))
                requested_model = str(matches[0].get("model_id") or requested_model).strip()
            elif len(provider_ids) > 1:
                options = ", ".join(sorted(str(row.get("model_id") or "").strip() for row in matches if str(row.get("model_id") or "").strip())[:3])
                return False, {
                    "ok": False,
                    "error": "model_target_ambiguous",
                    "message": f"I can test more than one model matching {requested_model}: {options}. Which exact model do you want?",
                    "matches": [str(row.get("model_id") or "").strip() for row in matches if str(row.get("model_id") or "").strip()],
                }
        if not requested_provider:
            return False, {
                "ok": False,
                "error": "provider_required",
                "message": "I couldn't tell which provider that model belongs to.",
            }
        ok, body = tester(requested_provider, {"model": requested_model})
        response = dict(body) if isinstance(body, dict) else {"ok": bool(ok)}
        response.setdefault("provider", requested_provider)
        response.setdefault("model_id", requested_model)
        return bool(ok), response

    def acquire_chat_model_target(
        self,
        model_id: str,
        *,
        provider_id: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        requested_model = str(model_id or "").strip()
        requested_provider = str(provider_id or "").strip().lower() or None
        if not requested_model:
            return False, {
                "ok": False,
                "error": "model_required",
                "message": "I need one exact model before I can acquire it.",
            }
        if not requested_provider and ":" in requested_model:
            requested_provider = str(requested_model.split(":", 1)[0]).strip().lower() or None
        canonical_model_id = requested_model
        if requested_provider and ":" not in canonical_model_id:
            canonical_model_id = f"{requested_provider}:{canonical_model_id}"

        lifecycle = self.model_lifecycle_status()
        lifecycle_rows = lifecycle.get("models") if isinstance(lifecycle.get("models"), list) else []
        row = next(
            (
                dict(item)
                for item in lifecycle_rows
                if isinstance(item, dict) and str(item.get("model_id") or "").strip() == canonical_model_id
            ),
            None,
        )
        lifecycle_state = str((row or {}).get("lifecycle_state") or "").strip().lower()
        if lifecycle_state == "ready":
            return True, {
                "ok": True,
                "executed": False,
                "provider": requested_provider,
                "model_id": canonical_model_id,
                "message": f"{canonical_model_id} is already ready to use.",
            }
        if lifecycle_state == "installed_not_ready":
            return True, {
                "ok": True,
                "executed": False,
                "provider": requested_provider,
                "model_id": canonical_model_id,
                "message": f"{canonical_model_id} is already installed, but it is not ready yet.",
            }
        if lifecycle_state == "queued":
            return True, {
                "ok": True,
                "executed": False,
                "provider": requested_provider,
                "model_id": canonical_model_id,
                "message": f"{canonical_model_id} is already queued. I am waiting for approval before I start.",
            }
        if lifecycle_state == "downloading":
            return True, {
                "ok": True,
                "executed": False,
                "provider": requested_provider,
                "model_id": canonical_model_id,
                "message": f"{canonical_model_id} is already downloading now.",
            }

        acquisition_state = str((row or {}).get("acquisition_state") or "").strip().lower()
        acquisition_reason = str((row or {}).get("acquisition_reason") or "").strip() or None
        if acquisition_state == "blocked_by_policy":
            why = acquisition_reason or f"Current mode does not allow downloading or installing {canonical_model_id} here."
            next_action = "Stay on the current model, or switch mode explicitly before retrying."
            return False, {
                "ok": False,
                "error": "safe_mode_blocked",
                "error_kind": "safe_mode_blocked",
                "provider": requested_provider,
                "model_id": canonical_model_id,
                "message": compose_actionable_message(
                    what_happened=f"I did not start downloading or installing {canonical_model_id}",
                    why=why,
                    next_action=next_action,
                ),
                "why": why,
                "next_action": next_action,
            }
        if requested_provider != "ollama" or not approved_local_profile_for_ref(canonical_model_id):
            why = acquisition_reason or f"{canonical_model_id} is not on the supported local acquire/install path."
            next_action = "Choose a supported local model, or use the canonical model manager path for approved installs."
            return False, {
                "ok": False,
                "error": "not_acquirable",
                "error_kind": "not_acquirable",
                "provider": requested_provider,
                "model_id": canonical_model_id,
                "message": compose_actionable_message(
                    what_happened=f"I cannot acquire {canonical_model_id} from this runtime",
                    why=why,
                    next_action=next_action,
                ),
                "why": why,
                "next_action": next_action,
            }

        puller = getattr(self.runtime, "pull_ollama_model", None)
        if not callable(puller):
            why = "The canonical local model manager is unavailable in this runtime."
            next_action = "Retry from the main runtime, or run: python -m agent doctor."
            return False, {
                "ok": False,
                "error": "acquisition_unavailable",
                "message": compose_actionable_message(
                    what_happened=f"I could not start acquiring {canonical_model_id}",
                    why=why,
                    next_action=next_action,
                ),
                "why": why,
                "next_action": next_action,
            }
        bare_model = canonical_model_id.split(":", 1)[1] if ":" in canonical_model_id else canonical_model_id
        ok, body = puller({"model": bare_model, "confirm": True})
        response = dict(body) if isinstance(body, dict) else {"ok": bool(ok)}
        response.setdefault("provider", "ollama")
        response.setdefault("model_id", canonical_model_id)
        return bool(ok), response

    def configure_local_chat_model(self, model_id: str) -> tuple[bool, dict[str, Any]]:
        return self.runtime.configure_local_chat_model(model_id)

    def configure_openrouter(
        self,
        api_key: str | None,
        payload: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        normalized_key = str(api_key or "").strip()
        if not normalized_key:
            providers = (
                self.runtime.registry_document.get("providers")
                if isinstance(self.runtime.registry_document.get("providers"), dict)
                else {}
            )
            provider_payload = (
                providers.get("openrouter")
                if isinstance(providers.get("openrouter"), dict)
                else {}
            )
            normalized_key = str(self.runtime._provider_api_key(provider_payload) or "").strip()
        if not normalized_key:
            return False, {
                "ok": False,
                "error": "api_key_required",
                "message": "Paste your OpenRouter API key and I will finish the setup.",
            }
        return self.runtime.configure_openrouter(normalized_key, payload)

    def set_default_chat_model(self, model_id: str) -> tuple[bool, dict[str, Any]]:
        ok, body = self.runtime.set_default_chat_model(model_id)
        if ok:
            self._invalidate_snapshot_cache()
        return ok, body

    def set_confirmed_chat_model_target(
        self,
        model_id: str,
        *,
        provider_id: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        setter = getattr(self.runtime, "set_confirmed_chat_model_target", None)
        if callable(setter):
            ok, body = setter(model_id, provider_id=provider_id)
        else:
            ok, body = self.runtime.set_default_chat_model(model_id)
        if ok:
            self._invalidate_snapshot_cache()
        return ok, body

    def set_temporary_chat_model_target(
        self,
        model_id: str,
        *,
        provider_id: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        setter = getattr(self.runtime, "set_temporary_chat_model_target", None)
        if callable(setter):
            ok, body = setter(model_id, provider_id=provider_id)
        else:
            ok, body = self.set_confirmed_chat_model_target(model_id, provider_id=provider_id)
        if ok:
            self._invalidate_snapshot_cache()
        return ok, body

    def restore_temporary_chat_model_target(
        self,
        model_id: str,
        *,
        provider_id: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        restorer = getattr(self.runtime, "restore_temporary_chat_model_target", None)
        if callable(restorer):
            ok, body = restorer(model_id, provider_id=provider_id)
        else:
            ok, body = self.set_confirmed_chat_model_target(model_id, provider_id=provider_id)
        if ok:
            self._invalidate_snapshot_cache()
        return ok, body
