import os
import subprocess
import tempfile
import unittest
import json
import time
from pathlib import Path
from unittest.mock import patch

from agent.llm.chat_preflight import PreparedChatRequest
from agent.filesystem_skill import FileSystemSkill
from agent.knowledge_cache import facts_hash
from agent.onboarding_flow import onboarding_completed_key
from agent.orchestrator import Orchestrator, OrchestratorResponse
from skills.resource_governor import collector
from agent.shell_skill import ShellSkill
from agent.public_chat import build_no_llm_public_message
from agent.tool_contract import normalize_tool_request
from agent.working_memory import WorkingMemoryState, append_turn
from memory.db import MemoryDB


class _FakeChatLLM:
    def __init__(self, *, enabled: bool, text: str = "LLM reply") -> None:
        self._enabled = bool(enabled)
        self._text = text
        self.chat_calls: list[dict[str, object]] = []

    def enabled(self) -> bool:
        return self._enabled

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.chat_calls.append(
            {
                "messages": messages,
                "kwargs": kwargs,
            }
        )
        return {"ok": True, "text": self._text, "provider": "ollama", "model": "llama3"}

    def intent_from_text(self, text: str) -> dict[str, object] | None:
        raise AssertionError(f"intent_from_text should not be called: {text}")


class _RaisingChatLLM:
    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self.chat_calls = 0

    def enabled(self) -> bool:
        return self._enabled

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        _ = messages
        _ = kwargs
        self.chat_calls += 1
        raise RuntimeError("llm chat failure")


class _FrontdoorRuntimeAdapter:
    def should_use_assistant_frontdoor(  # type: ignore[no-untyped-def]
        self,
        *,
        text=None,
        route_decision=None,
        is_user_chat=True,
    ):
        _ = text
        _ = route_decision
        return bool(is_user_chat)

    def _safe_mode_enabled(self) -> bool:
        return True

    def assistant_chat_available(self) -> bool:
        return True


class _RuntimeChatAvailableAdapter:
    def _safe_mode_enabled(self) -> bool:
        return False


class _PreparedChatAdapter:
    def _safe_mode_enabled(self) -> bool:
        return False

    def prepare_orchestrator_chat_request(self, request):  # type: ignore[no-untyped-def]
        payload = dict(request.get("payload") or {})
        messages = list(request.get("messages") or [])
        return {
            "prepared": PreparedChatRequest(
                messages=messages,
                last_user_text=str(((messages or [{}])[-1] or {}).get("content") or ""),
                provider_override="ollama",
                model_override="ollama:qwen2.5:7b-instruct",
                require_tools=False,
                selection_reason="default_target_pin",
                escalation_reasons=(),
                premium_selected_model=None,
            ),
            "defaults": {
                "default_provider": "ollama",
                "default_model": "ollama:qwen2.5:7b-instruct",
                "chat_model": "ollama:qwen2.5:7b-instruct",
                "allow_remote_fallback": False,
                "source_surface": payload.get("source_surface"),
            },
        }

    def assistant_chat_available(self) -> bool:
        return True


class _ControlModeRuntimeAdapter(_FrontdoorRuntimeAdapter):
    def __init__(self, truth: "_FakeRuntimeTruthService") -> None:
        self._truth = truth
        self.control_mode_calls: list[dict[str, object]] = []

    def llm_control_mode_set(self, payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
        request = dict(payload)
        self.control_mode_calls.append(request)
        mode = str(request.get("mode") or "").strip().lower()
        if mode == "controlled":
            self._truth.safe_mode = False
            self._truth.mode_source = "explicit_override"
            self._truth.allow_remote_fallback = True
            self._truth.allow_remote_recommendation = True
            self._truth.allow_remote_switch = True
            self._truth.allow_install_pull = True
            self._truth.scout_advisory_only = False
        elif mode == "safe":
            self._truth.safe_mode = True
            self._truth.mode_source = "explicit_override"
            self._truth.scout_advisory_only = True
        elif mode == "baseline":
            self._truth.safe_mode = True
            self._truth.mode_source = "config_default"
            self._truth.scout_advisory_only = True
        else:
            return False, {
                "ok": False,
                "error": "invalid_mode",
                "error_kind": "invalid_mode",
                "message": 'mode must be "safe", "controlled", or "baseline".',
            }
        return True, {
            "ok": True,
            "changed": True,
            "policy": self._truth.model_controller_policy_status(),
        }

    def llm_control_mode_status(self) -> dict[str, object]:
        return self._truth.model_controller_policy_status()


class _FakeRuntimeTruthService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []
        self.current_provider = "ollama"
        self.current_model = "ollama:qwen3.5:4b"
        self.default_provider = "ollama"
        self.default_model = "ollama:qwen3.5:4b"
        self.effective_provider = "ollama"
        self.effective_model = "ollama:qwen3.5:4b"
        self.temporary_override_active = False
        self.current_ready = True
        self.current_model_health_status = "ok"
        self.current_provider_health_status = "ok"
        self.openrouter_cheap_model = "openrouter:openai/gpt-4o-mini"
        self.openrouter_premium_model = "openrouter:anthropic/claude-opus-4"
        self.openrouter_cheap_quality_rank = 6
        self.openrouter_cheap_context_window = 128000
        self.openrouter_cheap_expected_cost_per_1m = 0.15
        self.openrouter_premium_quality_rank = 9
        self.openrouter_premium_context_window = 200000
        self.openrouter_premium_expected_cost_per_1m = 15.0
        self.openrouter_model = self.openrouter_cheap_model
        self.openrouter_secret_present = False
        self.openrouter_configured = False
        self.generic_default_drift_model = "ollama:deepseek-r1:7b"
        self.unavailable_confirmed_targets: set[str] = set()
        self.additional_available_models: list[dict[str, object]] = []
        self.hf_enabled = False
        self.hf_scan_body: dict[str, object] | None = None
        self.safe_mode = True
        self.mode_source = "config_default"
        self.allow_remote_fallback = False
        self.allow_remote_recommendation = False
        self.scout_advisory_only = True
        self.allow_remote_switch = False
        self.allow_install_pull = False
        self.filesystem_skill: FileSystemSkill | None = None
        self.shell_skill: ShellSkill | None = None

    def __getattr__(self, name: str) -> object:
        if name.startswith("available_") and name.endswith("_status"):
            raise AssertionError(f"{name} compat shim should not be used")
        if name.startswith("current_chat_") and name.endswith("target"):
            raise AssertionError(f"{name} compat shim should not be used")
        raise AttributeError(name)

    def current_chat_target_status(self) -> dict[str, object]:
        self.calls.append(("current_chat_target_status", None))
        return {
            "provider": self.current_provider,
            "model": self.current_model,
            "ready": self.current_ready,
            "health_status": self.current_model_health_status,
            "provider_health_status": self.current_provider_health_status,
            "configured_provider": self.default_provider,
            "configured_model": self.default_model,
            "effective_provider": self.effective_provider,
            "effective_model": self.effective_model,
            "effective_ready": self.current_ready,
            "truth_timing_ms": {"current_chat_target_status_ms": 1, "cache_hit": False},
        }

    def chat_target_truth(self) -> dict[str, object]:
        self.calls.append(("chat_target_truth", None))
        qualification_reason = (
            f"Configured default {self.current_model} on {self.current_provider} is healthy and ready."
            if self.current_ready
            else (
                f"Configured default {self.current_model} on {self.current_provider} is not currently healthy. "
                f"The best healthy target would be {self.effective_model} on {self.effective_provider}."
                if self.effective_model and self.effective_model != self.current_model
                else f"Configured default {self.current_model} on {self.current_provider} is not ready right now."
            )
        )
        degraded_reason = qualification_reason if not self.current_ready else None
        return {
            "configured_provider": self.default_provider,
            "configured_model": self.default_model,
            "configured_ready": bool(self.default_model),
            "effective_provider": self.effective_provider,
            "effective_model": self.effective_model,
            "effective_ready": bool(self.effective_model),
            "qualification_reason": qualification_reason,
            "degraded_reason": degraded_reason,
        }

    def provider_status(self, provider_id: str) -> dict[str, object]:
        self.calls.append(("provider_status", provider_id))
        provider_key = str(provider_id).strip().lower()
        if provider_key == "openrouter":
            return {
                "provider": "openrouter",
                "provider_label": "OpenRouter",
                "known": True,
                "enabled": True,
                "local": False,
                "configured": self.openrouter_configured,
                "active": self.current_provider == "openrouter",
                "secret_present": self.openrouter_secret_present,
                "health_status": (
                    self.current_provider_health_status
                    if self.current_provider == "openrouter"
                    else ("ok" if self.openrouter_configured else "unknown")
                ),
                "health_reason": "timeout while reaching OpenRouter" if self.current_provider_health_status == "down" else None,
                "model_id": self.openrouter_model if self.openrouter_configured else None,
                "model_ids": (
                    [self.openrouter_cheap_model, self.openrouter_premium_model]
                    if self.openrouter_configured
                    else []
                ),
                "current_provider": self.current_provider,
                "current_model_id": self.current_model,
            }
        return {
            "provider": "ollama",
            "provider_label": "Ollama",
            "known": True,
            "enabled": True,
            "local": True,
            "configured": True,
            "active": self.current_provider == "ollama",
            "secret_present": False,
            "health_status": self.current_provider_health_status if self.current_provider == "ollama" else "ok",
            "health_reason": "timeout while reaching Ollama" if self.current_provider_health_status == "down" else None,
            "model_id": self.current_model if self.current_provider == "ollama" else "ollama:qwen3.5:4b",
            "model_ids": ["ollama:qwen3.5:4b", "ollama:qwen3.5:8b"],
            "current_provider": self.current_provider,
            "current_model_id": self.current_model,
        }

    def providers_status(self) -> dict[str, object]:
        self.calls.append(("providers_status", None))
        rows = [self.provider_status("ollama"), self.provider_status("openrouter")]
        active = next((row for row in rows if bool(row.get("active", False))), None)
        configured = [row for row in rows if bool(row.get("configured", False))]
        return {
            "providers": rows,
            "configured_providers": [str(row.get("provider") or "") for row in configured],
            "active_provider": str((active or {}).get("provider") or "") or None,
            "active_model_id": str((active or {}).get("model_id") or "") or None,
        }

    def _inventory_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = [
            {
                "model_id": "ollama:qwen3.5:4b",
                "provider_id": "ollama",
                "local": True,
                "available": True,
                "enabled": True,
                "installed_local": True,
                "active": self.current_model == "ollama:qwen3.5:4b",
                "quality_rank": 6,
            },
            {
                "model_id": "ollama:qwen2.5:7b-instruct",
                "provider_id": "ollama",
                "local": True,
                "available": True,
                "enabled": True,
                "installed_local": True,
                "active": self.current_model == "ollama:qwen2.5:7b-instruct",
                "quality_rank": 9,
            },
            {
                "model_id": self.openrouter_cheap_model,
                "provider_id": "openrouter",
                "local": False,
                "available": True,
                "enabled": True,
                "installed_local": False,
                "active": self.current_model == self.openrouter_cheap_model,
                "quality_rank": self.openrouter_cheap_quality_rank,
                "context_window": self.openrouter_cheap_context_window,
                "expected_cost_per_1m": self.openrouter_cheap_expected_cost_per_1m,
            },
            {
                "model_id": self.openrouter_premium_model,
                "provider_id": "openrouter",
                "local": False,
                "available": True,
                "enabled": True,
                "installed_local": False,
                "active": self.current_model == self.openrouter_premium_model,
                "quality_rank": self.openrouter_premium_quality_rank,
                "context_window": self.openrouter_premium_context_window,
                "expected_cost_per_1m": self.openrouter_premium_expected_cost_per_1m,
            },
        ]
        for row in self.additional_available_models:
            if isinstance(row, dict):
                payload = dict(row)
                provider_id = str(payload.get("provider_id") or "").strip().lower()
                payload.setdefault("local", provider_id == "ollama")
                payload.setdefault("available", True)
                payload.setdefault("enabled", True)
                payload.setdefault(
                    "installed_local",
                    bool(payload.get("local", False)) and bool(payload.get("available", False)),
                )
                payload.setdefault(
                    "active",
                    str(payload.get("model_id") or "").strip() == str(self.current_model or "").strip(),
                )
                payload.setdefault("quality_rank", 0)
                rows.append(payload)
        return rows

    def model_inventory_status(self) -> dict[str, object]:
        self.calls.append(("model_inventory_status", None))
        rows = [dict(row) for row in self._inventory_rows() if isinstance(row, dict)]
        return {
            "active_provider": self.current_provider,
            "active_model": self.current_model,
            "configured_provider": self.default_provider,
            "configured_model": self.default_model,
            "known_models": rows,
            "local_installed_models": [dict(row) for row in rows if bool(row.get("local", False)) and bool(row.get("available", False))],
            "remote_registered_models": [dict(row) for row in rows if not bool(row.get("local", False))],
            "models": rows,
            "source": "fake-runtime-inventory",
            "truth_timing_ms": {"model_inventory_status_ms": 1, "cache_hit": False},
        }

    def model_readiness_status(self) -> dict[str, object]:
        self.calls.append(("model_readiness_status", None))
        rows: list[dict[str, object]] = []
        for inventory_row in self._inventory_rows():
            row = dict(inventory_row)
            provider_id = str(row.get("provider_id") or "").strip().lower()
            model_id = str(row.get("model_id") or "").strip()
            provider_snapshot = self.provider_status(provider_id)
            configured = bool(provider_snapshot.get("configured", False))
            provider_health_status = (
                self.current_provider_health_status
                if self.current_provider == provider_id
                else str(provider_snapshot.get("health_status") or "ok")
            )
            model_health_status = (
                self.current_model_health_status if self.current_model == model_id else "ok"
            )
            enabled = bool(row.get("enabled", True))
            available = bool(row.get("available", True))
            usable_now = row.get("usable_now")
            if usable_now is None:
                usable_now = bool(
                    enabled
                    and available
                    and configured
                    and provider_health_status == "ok"
                    and model_health_status == "ok"
                )
            availability_reason = str(row.get("availability_reason") or "").strip()
            availability_state = str(row.get("availability_state") or "").strip()
            if not availability_reason:
                if bool(usable_now):
                    availability_reason = "healthy and ready now"
                    availability_state = availability_state or "usable_now"
                elif not enabled:
                    availability_reason = "disabled"
                    availability_state = availability_state or "disabled"
                elif not available:
                    availability_reason = "not available in the current registry"
                    availability_state = availability_state or "unavailable"
                elif not configured:
                    availability_reason = "provider setup is still required"
                    availability_state = availability_state or "needs_setup"
                elif provider_health_status in {"down", "degraded"}:
                    availability_reason = f"provider is {provider_health_status}"
                    availability_state = availability_state or "provider_unhealthy"
                elif model_health_status in {"down", "degraded"}:
                    availability_reason = f"model health is {model_health_status}"
                    availability_state = availability_state or "model_unhealthy"
                else:
                    availability_reason = "not ready right now"
                    availability_state = availability_state or "not_ready"
            rows.append(
                {
                    **row,
                    "configured": configured,
                    "usable_now": bool(usable_now),
                    "provider_health_status": provider_health_status,
                    "model_health_status": model_health_status,
                    "availability_reason": availability_reason,
                    "availability_state": availability_state,
                }
            )
        ready_rows = [dict(row) for row in rows if bool(row.get("usable_now", False))]
        not_ready_rows = [dict(row) for row in rows if not bool(row.get("usable_now", False))]
        return {
            "active_provider": self.current_provider,
            "active_model": self.current_model,
            "configured_provider": self.default_provider,
            "configured_model": self.default_model,
            "models": rows,
            "ready_now_models": ready_rows,
            "usable_models": ready_rows,
            "other_ready_now_models": [dict(row) for row in ready_rows if not bool(row.get("active", False))],
            "other_usable_models": [dict(row) for row in ready_rows if not bool(row.get("active", False))],
            "not_ready_models": not_ready_rows,
            "source": "fake-runtime-readiness",
            "truth_timing_ms": {"model_readiness_status_ms": 1, "cache_hit": False},
        }

    def setup_status(self) -> dict[str, object]:
        self.calls.append(("setup_status", None))
        current = self.current_chat_target_status()
        target_truth = self.chat_target_truth()
        inventory = self.model_inventory_status()
        local_installed_rows = [
            dict(row)
            for row in (inventory.get("local_installed_models") if isinstance(inventory.get("local_installed_models"), list) else [])
            if isinstance(row, dict)
        ]
        other_local_rows = [
            dict(row)
            for row in local_installed_rows
            if str(row.get("model_id") or "").strip() != str(self.current_model or "")
        ]
        provider_snapshot = self.provider_status(self.current_provider) if self.current_provider else {}
        ready = bool(self.current_ready and self.current_model)
        if ready:
            setup_state = "ready"
            attention_kind = None
        elif self.current_model and self.current_provider:
            setup_state = "attention"
            if self.current_provider_health_status == "down":
                attention_kind = "provider_down"
            elif self.current_provider_health_status == "degraded":
                attention_kind = "provider_degraded"
            elif self.current_model_health_status in {"down", "degraded"}:
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
            "active_provider": self.current_provider,
            "active_model": self.current_model,
            "configured_provider": self.default_provider,
            "configured_model": self.default_model,
            "effective_provider": self.effective_provider,
            "effective_model": self.effective_model,
            "provider_health_status": self.current_provider_health_status,
            "provider_health_reason": provider_snapshot.get("health_reason"),
            "model_health_status": self.current_model_health_status,
            "qualification_reason": target_truth.get("qualification_reason"),
            "degraded_reason": target_truth.get("degraded_reason"),
            "local_installed_models": local_installed_rows,
            "other_local_models": other_local_rows,
            "source": "fake-runtime-setup",
        }

    def filesystem_list_directory(self, path: str | None, *, max_entries: int = 200) -> dict[str, object]:
        self.calls.append(("filesystem_list_directory", path))
        if self.filesystem_skill is None:
            raise AssertionError("filesystem_skill must be configured for this test")
        return self.filesystem_skill.list_directory(path, max_entries=max_entries)

    def filesystem_stat_path(self, path: str | None) -> dict[str, object]:
        self.calls.append(("filesystem_stat_path", path))
        if self.filesystem_skill is None:
            raise AssertionError("filesystem_skill must be configured for this test")
        return self.filesystem_skill.stat_path(path)

    def filesystem_read_text_file(
        self,
        path: str | None,
        *,
        max_bytes: int = 8192,
        offset: int = 0,
    ) -> dict[str, object]:
        self.calls.append(("filesystem_read_text_file", path))
        if self.filesystem_skill is None:
            raise AssertionError("filesystem_skill must be configured for this test")
        return self.filesystem_skill.read_text_file(path, max_bytes=max_bytes, offset=offset)

    def filesystem_search_filenames(
        self,
        root: str | None,
        query: str | None,
        *,
        max_results: int = 25,
        max_depth: int = 4,
    ) -> dict[str, object]:
        self.calls.append(("filesystem_search_filenames", (root, query)))
        if self.filesystem_skill is None:
            raise AssertionError("filesystem_skill must be configured for this test")
        return self.filesystem_skill.search_filenames(root, query, max_results=max_results, max_depth=max_depth)

    def filesystem_search_text(
        self,
        root: str | None,
        query: str | None,
        *,
        max_results: int = 25,
        max_files: int = 200,
        max_bytes_per_file: int = 8192,
    ) -> dict[str, object]:
        self.calls.append(("filesystem_search_text", (root, query)))
        if self.filesystem_skill is None:
            raise AssertionError("filesystem_skill must be configured for this test")
        return self.filesystem_skill.search_text(
            root,
            query,
            max_results=max_results,
            max_files=max_files,
            max_bytes_per_file=max_bytes_per_file,
        )

    def shell_execute_safe_command(
        self,
        command_name: str | None,
        *,
        subject: str | None = None,
        query: str | None = None,
        cwd: str | None = None,
        timeout_s: float = 2.0,
        max_output_chars: int = 4000,
    ) -> dict[str, object]:
        self.calls.append(("shell_execute_safe_command", (command_name, subject, query, cwd)))
        if self.shell_skill is None:
            raise AssertionError("shell_skill must be configured for this test")
        return self.shell_skill.execute_safe_command(
            command_name,
            subject=subject,
            query=query,
            cwd=cwd,
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
        )

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
    ) -> dict[str, object]:
        self.calls.append(("shell_install_package", (manager, package, scope, dry_run, cwd)))
        if self.shell_skill is None:
            raise AssertionError("shell_skill must be configured for this test")
        return self.shell_skill.install_package(
            manager=manager,
            package=package,
            scope=scope,
            dry_run=dry_run,
            cwd=cwd,
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
        )

    def shell_preview_install_package(
        self,
        *,
        manager: str | None,
        package: str | None,
        scope: str | None = None,
        dry_run: bool = False,
        cwd: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(("shell_preview_install_package", (manager, package, scope, dry_run, cwd)))
        if self.shell_skill is None:
            raise AssertionError("shell_skill must be configured for this test")
        return self.shell_skill.preview_install_package(
            manager=manager,
            package=package,
            scope=scope,
            dry_run=dry_run,
            cwd=cwd,
        )

    def shell_create_directory(self, path: str | None) -> dict[str, object]:
        self.calls.append(("shell_create_directory", path))
        if self.shell_skill is None:
            raise AssertionError("shell_skill must be configured for this test")
        return self.shell_skill.create_directory(path)

    def shell_preview_create_directory(self, path: str | None) -> dict[str, object]:
        self.calls.append(("shell_preview_create_directory", path))
        if self.shell_skill is None:
            raise AssertionError("shell_skill must be configured for this test")
        return self.shell_skill.preview_create_directory(path)

    def model_controller_policy_status(self, *, target_truth=None) -> dict[str, object]:  # type: ignore[no-untyped-def]
        self.calls.append(("model_controller_policy_status", target_truth))
        mode = "safe" if self.safe_mode else "controlled"
        return {
            "type": "model_controller_policy",
            "mode": mode,
            "mode_label": "SAFE MODE" if self.safe_mode else "Controlled Mode",
            "mode_source": self.mode_source,
            "safe_mode": self.safe_mode,
            "allow_remote_fallback": self.allow_remote_fallback,
            "allow_remote_recommendation": self.allow_remote_recommendation if not self.safe_mode else False,
            "allow_remote_switch": self.allow_remote_switch if not self.safe_mode else False,
            "allow_install_pull": self.allow_install_pull if not self.safe_mode else False,
            "scout_advisory_only": self.scout_advisory_only,
            "approval_required_actions": [
                "test_model",
                "switch_temporarily",
                "make_default",
                "acquire_model",
            ],
            "forbidden_actions": ["remote_switch", "install_download_import"] if self.safe_mode else [],
            "exact_target_preserved": True,
            "effective_provider": self.current_provider,
            "effective_model": self.current_model,
            "configured_provider": self.default_provider,
            "configured_model": self.default_model,
            "source": "fake-runtime-policy",
        }

    def model_status(self) -> dict[str, object]:
        self.calls.append(("model_status", None))
        return {}

    def runtime_status(self, kind: str = "runtime_status") -> dict[str, object]:
        self.calls.append(("runtime_status", kind))
        if kind == "telegram_status":
            return {
                "scope": "telegram",
                "configured": True,
                "state": "running",
                "summary": "Telegram is running.",
            }
        return {
            "scope": "ready",
            "ready": True,
            "runtime_mode": "READY",
            "failure_code": None,
            "provider": self.current_provider,
            "model": self.current_model,
                "summary": "Ready.",
        }

    def ready_status(self) -> dict[str, object]:
        self.calls.append(("ready_status", None))
        return {
            "ok": True,
            "ready": bool(self.current_ready),
            "phase": "ready" if self.current_ready else "degraded",
            "startup_phase": "ready" if self.current_ready else "starting",
            "runtime_mode": "READY" if self.current_ready else "DEGRADED",
            "failure_code": None,
            "summary": "Ready." if self.current_ready else "Not ready.",
            "runtime_status": {
                "runtime_mode": "READY" if self.current_ready else "DEGRADED",
                "summary": "Ready." if self.current_ready else "Not ready.",
            },
        }

    def list_managed_adapters(self) -> dict[str, object]:
        self.calls.append(("list_managed_adapters", None))
        return {
            "managed_adapters": [
                {
                    "adapter_id": "telegram",
                    "adapter_type": "telegram_polling",
                    "approved": True,
                    "enabled": True,
                    "reason": "Telegram transport is a managed adapter owned by the runtime.",
                    "requested_by": "platform",
                    "owner": "runtime",
                }
            ],
            "active_adapters": [
                {
                    "adapter_id": "telegram",
                    "adapter_type": "telegram_polling",
                    "approved": True,
                    "enabled": True,
                    "reason": "Telegram transport is a managed adapter owned by the runtime.",
                    "requested_by": "platform",
                    "owner": "runtime",
                }
            ],
            "active_adapter_ids": ["telegram"],
        }

    def get_managed_adapter_status(self, adapter_id: str | None) -> dict[str, object]:
        self.calls.append(("get_managed_adapter_status", adapter_id))
        if str(adapter_id or "").strip().lower() == "telegram":
            return {
                "found": True,
                "adapter": {
                    "adapter_id": "telegram",
                    "adapter_type": "telegram_polling",
                    "approved": True,
                    "enabled": True,
                    "reason": "Telegram transport is a managed adapter owned by the runtime.",
                    "requested_by": "platform",
                    "owner": "runtime",
                },
            }
        return {"found": False, "adapter_id": adapter_id}

    def list_background_tasks(self) -> dict[str, object]:
        self.calls.append(("list_background_tasks", None))
        return {
            "background_tasks": [
                {
                    "task_id": "runtime_scheduler",
                    "approved": True,
                    "enabled": True,
                    "health_status": "running",
                }
            ],
            "active_tasks": [
                {
                    "task_id": "runtime_scheduler",
                    "approved": True,
                    "enabled": True,
                    "health_status": "running",
                }
            ],
            "active_task_ids": ["runtime_scheduler"],
        }

    def get_background_task_status(self, task_id: str | None) -> dict[str, object]:
        self.calls.append(("get_background_task_status", task_id))
        if str(task_id or "").strip().lower() == "runtime_scheduler":
            return {
                "found": True,
                "task": {
                    "task_id": "runtime_scheduler",
                    "approved": True,
                    "enabled": True,
                    "health_status": "running",
                },
            }
        return {"found": False, "task_id": task_id}

    def list_governance_blocks(self) -> dict[str, object]:
        self.calls.append(("list_governance_blocks", None))
        return {
            "blocked_skills": [
                {
                    "skill_id": "rogue_daemon",
                    "reason": "forbidden_persistence_pattern",
                }
            ]
        }

    def list_pending_governance_requests(self) -> dict[str, object]:
        self.calls.append(("list_pending_governance_requests", None))
        return {
            "pending_skills": [
                {
                    "skill_id": "scheduled_sync",
                    "requested_execution_mode": "managed_background_task",
                    "requires_user_approval": True,
                }
            ],
            "pending_adapters": [],
            "pending_background_tasks": [],
        }

    def get_skill_governance_status(self, skill_id: str | None) -> dict[str, object]:
        self.calls.append(("get_skill_governance_status", skill_id))
        normalized = str(skill_id or "").strip().lower()
        if not normalized:
            return {"found": False, "needs_skill_id": True}
        if normalized == "scheduled_sync":
            return {
                "found": True,
                "skill": {
                    "skill_id": "scheduled_sync",
                    "requested_execution_mode": "managed_background_task",
                    "allowed": False,
                    "requires_user_approval": True,
                },
            }
        return {"found": False, "skill_id": skill_id}

    def model_policy_status(self) -> dict[str, object]:
        self.calls.append(("model_policy_status", None))
        return {
            "type": "model_policy_status",
            "policy_name": "default",
            "local_first": True,
            "tier_order": ["local", "free_remote", "cheap_remote"],
            "general_remote_cap_per_1m": 6.0,
            "cheap_remote_cap_per_1m": 0.5,
            "current_active_model": self.current_model,
            "current_active_provider": self.current_provider,
            "current_default_model": self.current_model,
            "current_default_provider": self.current_provider,
            "current_candidate": {
                "model_id": self.current_model,
                "provider_id": self.current_provider,
                "local": True,
                "tier": "local",
                "health_status": "ok",
            },
            "selected_candidate": {
                "model_id": self.current_model,
                "provider_id": self.current_provider,
                "local": True,
                "tier": "local",
                "health_status": "ok",
            },
            "recommended_candidate": {
                "model_id": self.current_model,
                "provider_id": self.current_provider,
                "local": True,
                "tier": "local",
                "health_status": "ok",
            },
            "switch_recommended": False,
            "decision_reason": "current_already_best",
            "decision_detail": "Current default already matches the best candidate in its tier.",
            "utility_delta": 0.0,
            "min_improvement": 0.08,
            "tier_candidates": {
                "local": {
                    "model_id": self.current_model,
                    "provider_id": self.current_provider,
                    "local": True,
                    "tier": "local",
                    "health_status": "ok",
                },
                "free_remote": {
                    "model_id": self.openrouter_model,
                    "provider_id": "openrouter",
                    "local": False,
                    "tier": "free_remote",
                    "health_status": "ok",
                },
            },
            "ordered_candidates": [],
            "rejected_candidates": [],
        }

    def model_policy_candidate(self, tier: str | None = None, *, status=None) -> dict[str, object]:  # type: ignore[no-untyped-def]
        self.calls.append(("model_policy_candidate", {"tier": tier, "status": status}))
        selection = status if isinstance(status, dict) else self.model_policy_status()
        if str(tier or "").strip().lower() == "free_remote":
            return {
                "type": "model_policy_candidate",
                "tier": "free_remote",
                "found": True,
                "candidate": {
                    "model_id": self.openrouter_model,
                    "provider_id": "openrouter",
                    "local": False,
                    "tier": "free_remote",
                },
                "selection": selection,
            }
        if str(tier or "").strip().lower() == "cheap_remote":
            return {
                "type": "model_policy_candidate",
                "tier": "cheap_remote",
                "found": False,
                "candidate": None,
                "cheap_remote_cap_per_1m": 0.5,
                "selection": selection,
            }
        return {
            "type": "model_policy_candidate",
            "tier": None,
            "found": True,
            "candidate": {
                "model_id": self.current_model,
                "provider_id": self.current_provider,
                "local": True,
                "tier": "local",
            },
            "switch_recommended": False,
            "decision_reason": "current_already_best",
            "decision_detail": "Current default already matches the best candidate in its tier.",
            "selection": selection,
        }

    def model_policy_provider_candidate(self, provider_id: str | None) -> dict[str, object]:
        self.calls.append(("model_policy_provider_candidate", provider_id))
        return {
            "type": "model_policy_candidate",
            "provider_id": "openrouter",
            "found": True,
            "candidate": {
                "model_id": self.openrouter_model,
                "provider_id": "openrouter",
                "local": False,
                "tier": "free_remote",
            },
            "provider_status": self.provider_status("openrouter"),
            "selection": self.model_policy_status(),
            "provider_selection": {
                "recommended_candidate": {
                    "model_id": self.openrouter_model,
                    "provider_id": "openrouter",
                    "local": False,
                    "tier": "free_remote",
                },
                "selected_candidate": {
                    "model_id": self.openrouter_model,
                    "provider_id": "openrouter",
                    "local": False,
                    "tier": "free_remote",
                },
                "rejected_candidates": [],
            },
        }

    def choose_best_local_chat_model(self, payload=None):  # type: ignore[no-untyped-def]
        self.calls.append(("choose_best_local_chat_model", payload))
        return True, {
            "ok": True,
            "candidate": {"model_id": "ollama:qwen3.5:8b"},
            "candidates": [{"model_id": "ollama:qwen3.5:8b"}],
        }

    def configure_local_chat_model(self, model_id: str):  # type: ignore[no-untyped-def]
        self.calls.append(("configure_local_chat_model", model_id))
        self.current_provider = "ollama"
        self.current_model = str(model_id)
        return True, {
            "ok": True,
            "provider": "ollama",
            "model_id": str(model_id),
            "message": f"I switched chat to your local model {model_id}.",
        }

    def configure_openrouter(self, api_key: str | None, payload=None):  # type: ignore[no-untyped-def]
        self.calls.append(("configure_openrouter", {"api_key": api_key, "payload": payload}))
        if not str(api_key or "").strip() and not self.openrouter_secret_present:
            return False, {
                "ok": False,
                "error": "api_key_required",
                "message": "Paste your OpenRouter API key and I will finish the setup.",
            }
        if str(api_key or "").strip():
            self.openrouter_secret_present = True
        self.openrouter_configured = True
        if bool((payload or {}).get("make_default")):
            self.current_provider = "openrouter"
            self.current_model = self.openrouter_model
        return True, {
            "ok": True,
            "provider": "openrouter",
            "model_id": self.openrouter_model,
            "message": "OpenRouter is ready for chat.",
        }

    def set_default_chat_model(self, model_id: str):  # type: ignore[no-untyped-def]
        self.calls.append(("set_default_chat_model", model_id))
        target_model = str(model_id)
        self.default_provider = str(target_model).split(":", 1)[0]
        self.default_model = target_model
        if not self.temporary_override_active:
            self.current_provider = self.default_provider
            self.current_model = self.default_model
        self.effective_provider = self.current_provider
        self.effective_model = self.current_model
        return True, {
            "ok": True,
            "provider": self.default_provider,
            "model_id": self.default_model,
            "message": f"Now using {self.default_model} for chat.",
        }

    def set_confirmed_chat_model_target(
        self,
        model_id: str,
        *,
        provider_id: str | None = None,
    ):  # type: ignore[no-untyped-def]
        self.calls.append(("set_confirmed_chat_model_target", {"model_id": model_id, "provider_id": provider_id}))
        if str(model_id) in self.unavailable_confirmed_targets:
            return False, {
                "ok": False,
                "error": "switch_target_unavailable",
                "message": "That model is no longer available, so I couldn't switch to it.",
            }
        applied_provider = str(provider_id or str(model_id).split(":", 1)[0]).strip().lower()
        self.default_provider = applied_provider
        self.default_model = str(model_id)
        self.current_provider = applied_provider
        self.current_model = str(model_id)
        self.effective_provider = self.current_provider
        self.effective_model = self.current_model
        self.temporary_override_active = False
        return True, {
            "ok": True,
            "provider": self.current_provider,
            "model_id": self.current_model,
            "message": f"Now using {self.current_model} for chat.",
        }

    def set_temporary_chat_model_target(
        self,
        model_id: str,
        *,
        provider_id: str | None = None,
    ):  # type: ignore[no-untyped-def]
        self.calls.append(("set_temporary_chat_model_target", {"model_id": model_id, "provider_id": provider_id}))
        applied_provider = str(provider_id or str(model_id).split(":", 1)[0]).strip().lower()
        self.current_provider = applied_provider
        self.current_model = str(model_id)
        self.effective_provider = self.current_provider
        self.effective_model = self.current_model
        self.temporary_override_active = True
        return True, {
            "ok": True,
            "provider": self.current_provider,
            "model_id": self.current_model,
            "message": f"Temporarily using {self.current_model} for chat.",
        }

    def acquire_chat_model_target(
        self,
        model_id: str,
        *,
        provider_id: str | None = None,
    ):
        self.calls.append(("acquire_chat_model_target", {"model_id": model_id, "provider_id": provider_id}))
        target_provider = str(provider_id or str(model_id).split(":", 1)[0]).strip().lower()
        target_model = str(model_id).strip()
        if self.safe_mode or not self.allow_install_pull:
            return False, {
                "ok": False,
                "error": "safe_mode_blocked",
                "error_kind": "safe_mode_blocked",
                "provider": target_provider,
                "model_id": target_model,
                "message": "SAFE MODE blocks model install/download actions right now.",
            }
        self.additional_available_models.append(
            {
                "model_id": target_model,
                "provider_id": target_provider,
                "local": target_provider == "ollama",
                "available": True,
                "enabled": True,
                "installed_local": target_provider == "ollama",
                "quality_rank": 7,
            }
        )
        return True, {
            "ok": True,
            "executed": True,
            "provider": target_provider,
            "model_id": target_model,
            "message": f"Started acquiring {target_model} through the canonical model manager.",
        }

    def restore_temporary_chat_model_target(
        self,
        model_id: str,
        *,
        provider_id: str | None = None,
    ):  # type: ignore[no-untyped-def]
        self.calls.append(("restore_temporary_chat_model_target", {"model_id": model_id, "provider_id": provider_id}))
        applied_provider = str(provider_id or str(model_id).split(":", 1)[0]).strip().lower()
        self.current_provider = applied_provider
        self.current_model = str(model_id)
        self.effective_provider = self.current_provider
        self.effective_model = self.current_model
        self.temporary_override_active = (
            self.current_provider != self.default_provider or self.current_model != self.default_model
        )
        return True, {
            "ok": True,
            "provider": self.current_provider,
            "model_id": self.current_model,
            "message": f"Now using {self.current_model} for chat.",
        }

    def test_chat_model_target(
        self,
        model_id: str,
        *,
        provider_id: str | None = None,
    ):  # type: ignore[no-untyped-def]
        self.calls.append(("test_chat_model_target", {"model_id": model_id, "provider_id": provider_id}))
        applied_provider = str(provider_id or str(model_id).split(":", 1)[0]).strip().lower()
        return True, {
            "ok": True,
            "provider": applied_provider,
            "model_id": str(model_id),
            "message": f"I tested {model_id} without switching. It looks healthy on {applied_provider}.",
        }

    def model_discovery_query(
        self,
        query: str | None = None,
        filters: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.calls.append(("model_discovery_query", {"query": query, "filters": dict(filters or {})}))
        normalized_query = str(query or "").strip().lower()
        models: list[dict[str, object]] = []
        scan_body = self.hf_scan_body if isinstance(self.hf_scan_body, dict) else None
        if self.hf_enabled and scan_body is not None:
            scan = scan_body.get("scan") if isinstance(scan_body.get("scan"), dict) else {}
            updates = scan.get("updates") if isinstance(scan, dict) and isinstance(scan.get("updates"), list) else []
            for row in updates:
                if not isinstance(row, dict):
                    continue
                repo_id = str(row.get("repo_id") or row.get("model_id") or "").strip()
                if not repo_id:
                    continue
                models.append(
                    {
                        "id": repo_id if repo_id.startswith("huggingface:") else f"huggingface:{repo_id}",
                        "provider": "huggingface",
                        "source": "huggingface",
                        "capabilities": ["chat"],
                        "local": False,
                        "installable": str(row.get("installability") or "").strip().startswith("installable"),
                        "confidence": 0.8,
                    }
                )
        if not models and any(token in normalized_query for token in ("gemma", "tiny", "smol", "small", "lightweight")):
            models.append(
                {
                    "id": "huggingface:tiny-gemma",
                    "provider": "huggingface",
                    "source": "huggingface",
                    "capabilities": ["chat"],
                    "local": False,
                    "installable": True,
                    "confidence": 0.7,
                }
            )
        return {
            "ok": True,
            "query": query,
            "message": f"Found {len(models)} model(s) across 1 source(s).",
            "models": models,
            "sources": [
                {
                    "source": "huggingface",
                    "enabled": bool(self.hf_enabled),
                    "queried": bool(self.hf_enabled or models),
                    "ok": True,
                    "count": len(models),
                }
            ],
            "debug": {"ranking": {"broadening_used": bool(models)}},
        }

    def model_scout_v2_status(self, *, task_request: dict[str, object] | None = None) -> dict[str, object]:
        self.calls.append(("model_scout_v2_status", dict(task_request) if isinstance(task_request, dict) else None))
        readiness = self.model_readiness_status()
        normalized_task = dict(task_request) if isinstance(task_request, dict) else {
            "task_type": "chat",
            "requirements": ["chat"],
            "preferred_local": True,
        }
        task_type = str(normalized_task.get("task_type") or "chat").strip().lower() or "chat"
        rows = [dict(row) for row in readiness.get("models", []) if isinstance(row, dict)]
        candidate_rows: list[dict[str, object]] = []
        for row in rows:
            model_id = str(row.get("model_id") or "")
            quality = int(row.get("quality_rank") or 0)
            local = str(row.get("provider_id") or "") == "ollama"
            expected_cost = float(row.get("expected_cost_per_1m") or 0.0)
            if local:
                tier = "local"
            elif expected_cost <= 0.0:
                tier = "free_remote"
            elif expected_cost <= 0.5:
                tier = "cheap_remote"
            else:
                tier = "remote"
            utility = float(quality)
            if not local:
                utility += 0.5
            candidate_rows.append(
                {
                    **row,
                    "tier": tier,
                    "utility": utility,
                    "context_window": int(row.get("context_window") or 32768),
                    "expected_cost_per_1m": expected_cost if not local else 0.0,
                    "auth_ok": bool(local or self.openrouter_secret_present),
                }
            )
        candidate_rows = [
            row
            for row in candidate_rows
            if bool(row.get("usable_now", False)) and bool(row.get("auth_ok", False))
        ]
        if self.safe_mode or not self.allow_remote_recommendation:
            candidate_rows = [row for row in candidate_rows if bool(row.get("local", False))]
        candidate_rows.sort(
            key=lambda row: (
                -float(row.get("utility") or 0.0),
                -int(row.get("quality_rank") or 0),
                str(row.get("model_id") or ""),
            )
        )
        current_candidate = next(
            (dict(row) for row in candidate_rows if str(row.get("model_id") or "") == str(self.current_model)),
            {
                "model_id": self.current_model,
                "provider_id": self.current_provider,
                "local": self.current_provider == "ollama",
                "tier": "local" if self.current_provider == "ollama" else "free_remote",
                "utility": 0.0,
                "quality_rank": 0,
                "context_window": 0,
                "expected_cost_per_1m": 0.0,
                "usable_now": self.current_ready,
            },
        )
        recommended = next(
            (
                dict(
                    row,
                    recommendation_reason="quality_upgrade",
                    recommendation_basis=(
                        "best_task_coding"
                        if task_type == "coding"
                        else "best_task_research"
                        if task_type == "reasoning"
                        else "best_local"
                        if bool(row.get("local", False))
                        else "best_task_chat"
                    ),
                    recommendation_explanation=(
                        "strongest available option currently visible for coding"
                        if task_type == "coding"
                        else "best available research option currently visible"
                        if task_type == "reasoning"
                        else "strongest local option currently available"
                        if bool(row.get("local", False))
                        else "strongest available option currently visible for chat"
                    ),
                )
                for row in candidate_rows
                if str(row.get("model_id") or "") != str(self.current_model)
                and int(row.get("quality_rank") or 0) >= int(current_candidate.get("quality_rank") or 0) + 2
            ),
            None,
        )
        cheap_cloud = next(
            (
                dict(
                    row,
                    role="cheap_cloud",
                    recommendation_reason="cheap_remote_value",
                    recommendation_basis="cheap_remote_value",
                    recommendation_explanation="lower-cost remote option for general use",
                )
                for row in candidate_rows
                if not bool(row.get("local", False))
                and str(row.get("tier") or "") in {"free_remote", "cheap_remote"}
            ),
            None,
        )
        premium_remote = next(
            (
                {
                    **dict(row),
                    "role": "premium_coding_cloud" if task_type == "coding" else "premium_research_cloud",
                    "recommendation_reason": "task_specialist_upgrade",
                    "recommendation_basis": (
                        "premium_coding_tier"
                        if task_type == "coding"
                        else "premium_research_tier"
                    ),
                    "recommendation_explanation": (
                        "qualifies for the premium coding tier"
                        if task_type == "coding"
                        else "meets the premium quality and large-context requirements for research"
                    ),
                }
                for row in candidate_rows
                if not bool(row.get("local", False))
                and str(row.get("tier") or "") == "remote"
                and int(row.get("quality_rank") or 0) >= 7
                and (
                    task_type != "reasoning"
                    or int(row.get("context_window") or 0) >= 131072
                )
            ),
            None,
        )
        task_recommendation = (
            dict(
                premium_remote,
                recommendation_basis="best_task_coding" if task_type == "coding" else "best_task_research",
                recommendation_explanation=(
                    "strongest available option currently visible for coding"
                    if task_type == "coding"
                    else "best available research option currently visible"
                ),
            )
            if task_type in {"coding", "reasoning"} and isinstance(premium_remote, dict)
            else recommended
        )
        role_candidates = {
            "comfortable_local_default": next(
                (dict(row) for row in candidate_rows if bool(row.get("local", False))),
                None,
            ),
            "cheap_cloud": cheap_cloud if self.allow_remote_recommendation and not self.safe_mode else None,
            "premium_coding_cloud": (
                premium_remote if task_type == "coding" else None
            ) if self.allow_remote_recommendation and not self.safe_mode else None,
            "premium_research_cloud": (
                premium_remote if task_type == "reasoning" else None
            ) if self.allow_remote_recommendation and not self.safe_mode else None,
        }

        def _comparison_for(role_key: str, candidate: dict[str, object] | None) -> dict[str, object] | None:
            if not isinstance(candidate, dict):
                return None
            model_id = str(candidate.get("model_id") or "")
            if model_id == str(self.current_model):
                return {
                    "state": "lateral",
                    "basis": "same_as_current",
                    "explanation": "already using this model",
                }
            if role_key == "cheap_cloud":
                return {
                    "state": "lateral",
                    "basis": "lower_cost_alternative",
                    "explanation": "alternative option, not a clear overall upgrade",
                }
            if role_key == "premium_coding":
                return {
                    "state": "upgrade",
                    "basis": "higher_premium_role_than_current",
                    "explanation": "upgrade for coding quality",
                }
            if role_key == "premium_research":
                return {
                    "state": "upgrade",
                    "basis": "larger_context_research_fit_than_current",
                    "explanation": "upgrade for research quality and context",
                }
            if role_key == "best_task_research":
                return {
                    "state": "upgrade",
                    "basis": "larger_context_research_fit_than_current",
                    "explanation": "upgrade for research quality and context",
                }
            if role_key == "best_local":
                return {
                    "state": "upgrade",
                    "basis": "stronger_local_than_current",
                    "explanation": "upgrade within the local options",
                }
            if role_key == "best_task_coding":
                return {
                    "state": "upgrade",
                    "basis": "stronger_task_fit_than_current",
                    "explanation": "upgrade for this task",
                }
            return {
                "state": "lateral",
                "basis": "no_meaningful_difference",
                "explanation": "alternative option, not a clear overall upgrade",
            }

        def _resolution(
            *,
            state: str,
            candidate: dict[str, object] | None = None,
            reason_code: str | None = None,
            explanation: str | None = None,
            comparison: dict[str, object] | None = None,
            advisory_actions: dict[str, object] | None = None,
        ) -> dict[str, object]:
            payload: dict[str, object] = {"state": state}
            if isinstance(candidate, dict):
                payload["model_id"] = str(candidate.get("model_id") or "")
                payload["provider_id"] = str(candidate.get("provider_id") or "")
                if str(candidate.get("recommendation_basis") or ""):
                    payload["recommendation_basis"] = str(candidate.get("recommendation_basis") or "")
            if reason_code:
                payload["reason_code"] = reason_code
            if explanation:
                payload["explanation"] = explanation
            if isinstance(comparison, dict):
                payload["comparison"] = dict(comparison)
            if isinstance(advisory_actions, dict):
                payload["advisory_actions"] = dict(advisory_actions)
            return payload

        def _action(state: str, reason_code: str | None = None) -> dict[str, object]:
            payload: dict[str, object] = {"state": state}
            if reason_code:
                payload["reason_code"] = reason_code
            if state == "available":
                payload["approval_required"] = True
            return payload

        def _advisory_actions(
            *,
            resolution_state: str,
            candidate: dict[str, object] | None = None,
            reason_code: str | None = None,
        ) -> dict[str, object]:
            action_names = ("test", "switch_temporarily", "make_default", "acquire")
            if resolution_state == "blocked_by_mode":
                block_code = reason_code or ("safe_mode_remote_block" if self.safe_mode else "remote_switch_disabled")
                return {name: _action("blocked", block_code) for name in action_names}
            if resolution_state != "selected" or not isinstance(candidate, dict):
                return {name: _action("not_applicable", "no_selected_candidate") for name in action_names}

            model_id = str(candidate.get("model_id") or "")
            local = bool(candidate.get("local", False))
            remote_switch_blocked = (not local) and (not self.allow_remote_switch)
            remote_block_code = "safe_mode_remote_block" if self.safe_mode else "remote_switch_disabled"
            install_block_code = "safe_mode_install_block" if self.safe_mode else "install_disabled_by_policy"
            if remote_switch_blocked:
                test_action = _action("blocked", remote_block_code)
                switch_action = _action("blocked", remote_block_code)
                default_action = _action("blocked", remote_block_code)
            else:
                if model_id == str(self.current_model):
                    test_action = _action("not_applicable", "already_current_model")
                    switch_action = _action("not_applicable", "already_current_model")
                else:
                    test_action = _action("available")
                    switch_action = _action("available")
                if model_id == str(self.default_model):
                    default_action = _action("not_applicable", "already_default_model")
                else:
                    default_action = _action("available")

            if str(candidate.get("acquisition_state") or "").strip().lower() in {"acquirable", "installed_not_ready", "queued", "downloading"}:
                if self.allow_install_pull:
                    acquire_action = _action("available")
                else:
                    acquire_action = _action("blocked", install_block_code)
            elif local:
                acquire_action = _action("not_applicable", "local_model_no_acquire_needed")
            else:
                acquire_action = _action("not_applicable", "already_available_remote")

            return {
                "test": test_action,
                "switch_temporarily": switch_action,
                "make_default": default_action,
                "acquire": acquire_action,
            }

        recommendation_roles = {
            "best_local": (
                _resolution(
                    state="selected",
                    candidate=role_candidates["comfortable_local_default"],
                    comparison=_comparison_for("best_local", role_candidates["comfortable_local_default"]),
                    advisory_actions=_advisory_actions(
                        resolution_state="selected",
                        candidate=role_candidates["comfortable_local_default"],
                    ),
                )
                if isinstance(role_candidates["comfortable_local_default"], dict)
                else _resolution(
                    state="unavailable",
                    reason_code="no_local_candidate",
                    explanation="no usable local model is currently available",
                    advisory_actions=_advisory_actions(
                        resolution_state="unavailable",
                        reason_code="no_local_candidate",
                    ),
                )
            ),
            "cheap_cloud": (
                _resolution(
                    state="selected",
                    candidate=cheap_cloud,
                    comparison=_comparison_for("cheap_cloud", cheap_cloud),
                    advisory_actions=_advisory_actions(
                        resolution_state="selected",
                        candidate=cheap_cloud,
                    ),
                )
                if self.allow_remote_recommendation and not self.safe_mode and isinstance(cheap_cloud, dict)
                else _resolution(
                    state="blocked_by_mode" if (self.safe_mode or not self.allow_remote_recommendation) else "no_qualifying_candidate",
                    reason_code="safe_mode_remote_block" if self.safe_mode else "no_cheap_remote_candidate",
                    explanation=(
                        "remote recommendations are not usable in this mode"
                        if self.safe_mode
                        else "no usable low-cost remote model is currently available"
                    ),
                    advisory_actions=_advisory_actions(
                        resolution_state="blocked_by_mode" if (self.safe_mode or not self.allow_remote_recommendation) else "no_qualifying_candidate",
                        reason_code="safe_mode_remote_block" if self.safe_mode else "no_cheap_remote_candidate",
                    ),
                )
            ),
            "premium_coding": (
                _resolution(
                    state="selected",
                    candidate=role_candidates["premium_coding_cloud"],
                    comparison=_comparison_for("premium_coding", role_candidates["premium_coding_cloud"]),
                    advisory_actions=_advisory_actions(
                        resolution_state="selected",
                        candidate=role_candidates["premium_coding_cloud"],
                    ),
                )
                if self.allow_remote_recommendation and not self.safe_mode and isinstance(role_candidates["premium_coding_cloud"], dict)
                else _resolution(
                    state="blocked_by_mode" if (self.safe_mode or not self.allow_remote_recommendation) else "no_qualifying_candidate",
                    reason_code="safe_mode_remote_block" if self.safe_mode else "premium_coding_threshold_unmet",
                    explanation=(
                        "remote recommendations are not usable in this mode"
                        if self.safe_mode
                        else "no remote model currently meets the required premium quality threshold"
                    ),
                    advisory_actions=_advisory_actions(
                        resolution_state="blocked_by_mode" if (self.safe_mode or not self.allow_remote_recommendation) else "no_qualifying_candidate",
                        reason_code="safe_mode_remote_block" if self.safe_mode else "premium_coding_threshold_unmet",
                    ),
                )
            ),
            "premium_research": (
                _resolution(
                    state="selected",
                    candidate=role_candidates["premium_research_cloud"],
                    comparison=_comparison_for("premium_research", role_candidates["premium_research_cloud"]),
                    advisory_actions=_advisory_actions(
                        resolution_state="selected",
                        candidate=role_candidates["premium_research_cloud"],
                    ),
                )
                if self.allow_remote_recommendation and not self.safe_mode and isinstance(role_candidates["premium_research_cloud"], dict)
                else _resolution(
                    state="blocked_by_mode" if (self.safe_mode or not self.allow_remote_recommendation) else "no_qualifying_candidate",
                    reason_code="safe_mode_remote_block" if self.safe_mode else "premium_research_threshold_unmet",
                    explanation=(
                        "remote recommendations are not usable in this mode"
                        if self.safe_mode
                        else "no remote model currently meets the required premium quality and context thresholds"
                    ),
                    advisory_actions=_advisory_actions(
                        resolution_state="blocked_by_mode" if (self.safe_mode or not self.allow_remote_recommendation) else "no_qualifying_candidate",
                        reason_code="safe_mode_remote_block" if self.safe_mode else "premium_research_threshold_unmet",
                    ),
                )
            ),
            "best_task_chat": _resolution(
                state="selected",
                candidate=current_candidate,
                comparison=_comparison_for("best_task_chat", current_candidate),
                advisory_actions=_advisory_actions(
                    resolution_state="selected",
                    candidate=current_candidate,
                ),
            ),
            "best_task_coding": _resolution(
                state="selected",
                candidate=task_recommendation if task_type == "coding" and isinstance(task_recommendation, dict) else current_candidate,
                comparison=_comparison_for(
                    "best_task_coding",
                    task_recommendation if task_type == "coding" and isinstance(task_recommendation, dict) else current_candidate,
                ),
                advisory_actions=_advisory_actions(
                    resolution_state="selected",
                    candidate=task_recommendation if task_type == "coding" and isinstance(task_recommendation, dict) else current_candidate,
                ),
            ),
            "best_task_research": _resolution(
                state="selected",
                candidate=task_recommendation if task_type == "reasoning" and isinstance(task_recommendation, dict) else current_candidate,
                comparison=_comparison_for(
                    "best_task_research",
                    task_recommendation if task_type == "reasoning" and isinstance(task_recommendation, dict) else current_candidate,
                ),
                advisory_actions=_advisory_actions(
                    resolution_state="selected",
                    candidate=task_recommendation if task_type == "reasoning" and isinstance(task_recommendation, dict) else current_candidate,
                ),
            ),
        }
        return {
            "type": "model_scout_v2",
            "active_provider": self.current_provider,
            "active_model": self.current_model,
            "current_candidate": current_candidate,
            "recommended_candidate": recommended,
            "better_candidates": [recommended] if isinstance(recommended, dict) else [],
            "candidate_rows": candidate_rows,
            "not_ready_models": [
                dict(row) for row in rows if not bool(row.get("usable_now", False))
            ],
            "policy": self.model_controller_policy_status(),
            "task_request": normalized_task,
            "role_candidates": role_candidates,
            "recommendation_roles": recommendation_roles,
            "task_recommendation": task_recommendation,
            "inventory": self.model_inventory_status(),
            "readiness": self.model_readiness_status(),
            "advisory_only": self.safe_mode,
            "selection": {
                "ordered_candidates": candidate_rows,
                "switch_recommended": bool(recommended),
                "decision_reason": "quality_upgrade" if recommended else "current_already_best",
                "decision_detail": (
                    "Candidate quality rank materially exceeds the current model."
                    if recommended
                    else "Current default already matches the best candidate in its tier."
                ),
                "min_improvement": 0.08,
            },
            "source": "fake-runtime-truth",
        }

    def model_watch_hf_status(self) -> dict[str, object]:
        self.calls.append(("model_watch_hf_status", None))
        return {
            "ok": True,
            "enabled": self.hf_enabled,
            "tracked_repos": 1 if self.hf_enabled else 0,
            "discovered_count": 0,
        }

    def model_watch_hf_scan(self, *, trigger: str = "manual", notify_proposal: bool = False, persist_proposal: bool = False):  # type: ignore[no-untyped-def]
        self.calls.append(
            (
                "model_watch_hf_scan",
                {
                    "trigger": trigger,
                    "notify_proposal": notify_proposal,
                    "persist_proposal": persist_proposal,
                },
            )
        )
        if self.hf_scan_body is not None:
            return True, dict(self.hf_scan_body)
        return True, {
            "ok": True,
            "trigger": trigger,
            "scan": {
                "ok": True,
                "enabled": self.hf_enabled,
                "updates": [],
                "discovered_count": 0,
            },
            "proposal_created": False,
            "proposal": None,
        }


class TestOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.log_path = os.path.join(self.tmpdir.name, "events.log")
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def _orchestrator(self) -> Orchestrator:
        return Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )

    def test_handle_message_no_longer_raises(self) -> None:
        orchestrator = self._orchestrator()
        response = orchestrator.handle_message("hello there", "user1")
        self.assertIsInstance(response, OrchestratorResponse)
        self.assertIn("I’m not ready to chat yet", response.text)

    def test_llm_available_routes_free_text_to_llm_chat(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="hi from llm")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        response = orchestrator.handle_message("what can you help with?", "user1")
        self.assertEqual("hi from llm", response.text)
        self.assertEqual(1, len(llm.chat_calls))
        call = llm.chat_calls[0]
        kwargs = call.get("kwargs") if isinstance(call, dict) else {}
        self.assertEqual("chat", (kwargs or {}).get("purpose"))
        self.assertNotIn("/brief", response.text.lower())
        messages = call.get("messages") if isinstance(call, dict) else []
        system_text = str((messages or [{}])[0].get("content") if messages else "")
        self.assertIn("Always identify as the local Personal Agent", system_text)

    def test_generic_chat_reports_no_prior_memory_diagnostics(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="forest reply")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )

        response = orchestrator._llm_chat(
            "user1",
            "Tell me about forests.",
            chat_context={"messages": [{"role": "user", "content": "Tell me about forests."}]},
        )

        diagnostics = response.data.get("memory_diagnostics")
        self.assertIsInstance(diagnostics, dict)
        self.assertFalse(bool(diagnostics.get("used")))
        self.assertEqual("no_prior_memory", diagnostics.get("reason"))
        self.assertEqual(0, diagnostics.get("context_chars"))
        self.assertFalse(bool(response.data.get("used_memory")))
        self.assertNotIn("Tell me about forests", json.dumps(diagnostics))

    def test_generic_chat_reports_prior_working_memory_diagnostics(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="continue reply")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        state = WorkingMemoryState()
        append_turn(state, role="user", text="We were refactoring the alerts workflow.")
        self.assertTrue(orchestrator._memory_runtime.save_working_memory_state("user1", state))

        response = orchestrator._llm_chat(
            "user1",
            "Tell me how to proceed.",
            chat_context={"messages": [{"role": "user", "content": "Tell me how to proceed."}]},
        )

        diagnostics = response.data.get("memory_diagnostics")
        self.assertIsInstance(diagnostics, dict)
        self.assertTrue(bool(diagnostics.get("used")))
        self.assertEqual("hot_thread_history", diagnostics.get("reason"))
        self.assertTrue(bool(response.data.get("used_memory")))
        sources = diagnostics.get("sources") if isinstance(diagnostics.get("sources"), dict) else {}
        self.assertTrue(bool(sources.get("working_memory")))
        working_memory = diagnostics.get("working_memory") if isinstance(diagnostics.get("working_memory"), dict) else {}
        self.assertTrue(bool(working_memory.get("had_prior_state")))
        self.assertGreaterEqual(int(working_memory.get("hot_turn_count") or 0), 2)
        self.assertNotIn("refactoring the alerts workflow", json.dumps(diagnostics))

    def test_handle_message_preserves_runtime_prepared_chat_overrides(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="blue and white")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            chat_runtime_adapter=_PreparedChatAdapter(),
        )

        response = orchestrator.handle_message(
            "What colour is a bluejay?",
            "user1",
            chat_context={
                "trace_id": "trace-bluejay",
                "source_surface": "webui",
                "payload": {"source_surface": "webui"},
                "messages": [{"role": "user", "content": "What colour is a bluejay?"}],
            },
        )

        self.assertEqual("blue and white", response.text)
        self.assertEqual(1, len(llm.chat_calls))
        kwargs = llm.chat_calls[0].get("kwargs") if isinstance(llm.chat_calls[0], dict) else {}
        self.assertEqual("ollama", (kwargs or {}).get("provider_override"))
        self.assertEqual("ollama:qwen2.5:7b-instruct", (kwargs or {}).get("model_override"))

    def test_telegram_latency_fallback_drops_default_target_pin_overrides(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
            chat_runtime_adapter=_PreparedChatAdapter(),
        )
        route_calls: list[dict[str, object]] = []

        def _fake_route_inference(**kwargs: object) -> dict[str, object]:
            route_calls.append(dict(kwargs))
            metadata = kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else {}
            if not bool(metadata.get("latency_fallback")):
                return {
                    "ok": False,
                    "text": "",
                    "provider": "ollama",
                    "model": "ollama:qwen2.5:7b-instruct",
                    "task_type": "chat",
                    "selection_reason": "timeout",
                    "fallback_used": False,
                    "error_kind": "timeout",
                    "error_class": "timeout",
                    "duration_ms": 5000,
                    "attempts": [],
                    "data": {"selection": {"selected_model": "ollama:qwen2.5:7b-instruct", "provider": "ollama"}},
                }
            return {
                "ok": True,
                "text": "Fallback reply",
                "provider": "ollama",
                "model": "ollama:qwen3.5:4b",
                "task_type": "chat",
                "selection_reason": "healthy",
                "fallback_used": False,
                "error_kind": None,
                "duration_ms": 700,
                "attempts": [],
                "data": {"selection": {"selected_model": "ollama:qwen3.5:4b", "provider": "ollama"}},
            }

        with patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference):
            response = orchestrator._llm_chat(
                "user1",
                "Tell me a joke",
                chat_context={
                    "payload": {"source_surface": "telegram"},
                    "messages": [{"role": "user", "content": "Tell me a joke"}],
                    "source_surface": "telegram",
                    "channel": "telegram",
                },
            )

        self.assertEqual("Fallback reply", response.text)
        self.assertEqual(2, len(route_calls))
        self.assertEqual("ollama:qwen2.5:7b-instruct", route_calls[0].get("model_override"))
        self.assertEqual("ollama", route_calls[0].get("provider_override"))
        self.assertIsNone(route_calls[1].get("model_override"))
        self.assertIsNone(route_calls[1].get("provider_override"))
        fallback_meta = route_calls[1].get("metadata") if isinstance(route_calls[1].get("metadata"), dict) else {}
        self.assertTrue(bool(fallback_meta.get("latency_fallback")))

    def test_generic_chat_router_failure_uses_deterministic_conversation_fallback(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="unused")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )

        with patch(
            "agent.orchestrator.route_inference",
            return_value={
                "ok": False,
                "text": "",
                "provider": None,
                "model": None,
                "task_type": "chat",
                "selection_reason": "no_suitable_model",
                "fallback_used": True,
                "error_kind": "no_suitable_model",
                "next_action": None,
                "trace_id": "orch-test",
                "data": {"selection": {"selected_model": None, "provider": None, "reason": "no_suitable_model"}},
            },
        ):
            response = orchestrator._llm_chat(
                "user1",
                "I'm testing whether you can stay coherent through a long chat.",
                chat_context={
                    "payload": {"source_surface": "telegram"},
                    "messages": [{"role": "user", "content": "I'm testing whether you can stay coherent through a long chat."}],
                    "source_surface": "telegram",
                    "channel": "telegram",
                },
            )

        self.assertIn("ask your next question", response.text.lower())
        self.assertIn("keep the thread consistent", response.text.lower())
        self.assertTrue(bool(response.data.get("ok")))
        self.assertFalse(bool(response.data.get("used_llm")))

    def test_runtime_ready_allows_normal_chat_despite_stale_onboarding_state(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="hi from llm")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_ready = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        self.db.set_user_pref(onboarding_completed_key("user1"), "false")

        response = orchestrator.handle_message("hello", "user1", chat_context={"source_surface": "webui"})

        self.assertEqual("hi from llm", response.text)
        self.assertEqual(1, len(llm.chat_calls))
        self.assertNotIn("not ready", response.text.lower())
        self.assertEqual("true", str(self.db.get_user_pref(onboarding_completed_key("user1")) or "").strip().lower())

    def test_coding_prompt_does_not_fall_into_disk_pressure_observe_path(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="here is the script")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        response = orchestrator.handle_message(
            "Write a Bash script that finds the 10 largest files under a directory and ignores .git and node_modules.",
            "user-coding",
        )

        self.assertEqual("generic_chat", response.data["route"])
        self.assertEqual(1, len(llm.chat_calls))
        self.assertIn("here is the script", response.text)
        self.assertNotIn("Disk pressure culprit", response.text)

    def test_double_check_followup_uses_recent_context_without_not_ready_gate(self) -> None:
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_ready = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
            runtime_truth_service=runtime_truth,
        )
        orchestrator._last_interpretable_result["user1"] = {
            "created_ts": 0,
            "route": "runtime_status",
            "kind": "runtime_status",
            "user_text": "why is ollama slow?",
            "response_text": "Ollama is reachable.",
            "summary": "Ollama is reachable.",
            "payload": {
                "type": "runtime_status",
                "summary": "Ollama is reachable.",
            },
            "used_tools": ["resource_report"],
        }

        with patch(
            "agent.orchestrator.resource_followup",
            return_value="fresh grounded answer",
        ) as mock_followup:
            response = orchestrator.handle_message("i dont think it is, can you double check please?", "user1")

        self.assertEqual("fresh grounded answer", response.text)
        self.assertNotIn("not ready to chat yet", response.text.lower())
        self.assertNotIn("open setup", response.text.lower())
        mock_followup.assert_called_once()

    def test_confusion_turn_clarifies_without_resetting_to_not_ready(self) -> None:
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_ready = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
            runtime_truth_service=runtime_truth,
        )
        orchestrator._last_interpretable_result["user1"] = {
            "created_ts": 0,
            "route": "runtime_status",
            "kind": "runtime_status",
            "user_text": "what is ollama doing?",
            "response_text": "Ollama is reachable.",
            "summary": "Ollama is reachable.",
            "payload": {
                "type": "runtime_status",
                "summary": "Ollama is reachable.",
            },
            "used_tools": ["resource_report"],
        }

        response = orchestrator.handle_message("wat?", "user1", chat_context={"source_surface": "webui"})

        self.assertIn("continue with that", response.text.lower())
        self.assertNotIn("not ready to chat yet", response.text.lower())

    def test_correction_turn_reassesses_without_generic_choice_prompt(self) -> None:
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_ready = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
            runtime_truth_service=runtime_truth,
        )
        orchestrator._last_interpretable_result["user1"] = {
            "created_ts": 0,
            "route": "operational_status",
            "kind": "resource_report",
            "user_text": "how is memory looking?",
            "response_text": "Memory is getting tight.",
            "summary": "Memory is getting tight.",
            "payload": {
                "type": "resource_report",
                "kind": "resource_report",
                "summary": "Memory is getting tight.",
                "memory": {
                    "total": int(62.7 * (1024**3)),
                    "used": int(12.8 * (1024**3)),
                    "available": int(49.9 * (1024**3)),
                    "free": int(49.9 * (1024**3)),
                    "used_pct": 20.4,
                    "buffers": 0,
                    "cached": 0,
                    "shared": 0,
                },
                "swap": {"total": int(8 * (1024**3)), "used": int(4 * (1024**3))},
                "cpu_count": 8,
                "loads": {"1m": 1.0, "5m": 0.8, "15m": 0.6},
            },
            "used_tools": ["resource_report"],
        }

        response = orchestrator.handle_message("that's wrong, please reassess", "user1")

        lowered = response.text.lower()
        self.assertIn("you’re right", lowered)
        self.assertIn("not under pressure", lowered)
        self.assertNotIn("getting tight", lowered)
        self.assertNotIn("do you want runtime status", lowered)
        self.assertNotIn("setup help", lowered)
        self.assertEqual("assistant_clarification", response.data["route"])
        self.assertEqual("assistant_correction", response.data["runtime_payload"]["type"])

    def test_linux_tool_suggestions_do_not_keep_windows_only_tools(self) -> None:
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_ready = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
            runtime_truth_service=runtime_truth,
        )
        response = OrchestratorResponse(
            "Use SpeedFan to check temperatures.",
            {"route": "generic_chat", "used_llm": True, "provider": "ollama", "model": "llama3"},
        )

        with patch("agent.orchestrator.platform.system", return_value="Linux"):
            guarded = orchestrator._apply_assistant_response_guard(
                user_id="user1",
                user_text="tell me how to check temperatures",
                response=response,
            )

        self.assertNotIn("SpeedFan", guarded.text)
        self.assertIn("lm-sensors", guarded.text.lower())

    def test_runtime_provider_query_uses_runtime_truth_service_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("is openrouter configured?", "user1")
        self.assertIn("OpenRouter", response.text)
        self.assertEqual("provider_status", response.data["route"])
        self.assertTrue(response.data["used_runtime_state"])
        self.assertFalse(response.data["used_llm"])
        self.assertFalse(response.data["used_memory"])
        self.assertEqual([], response.data["used_tools"])
        self.assertEqual(0, len(llm.chat_calls))
        self.assertIn(("provider_status", "openrouter"), runtime_truth.calls)
        self.assertTrue(response.data.get("skip_post_response_hooks", False))
        timing = response.data.get("orchestrator_timing_ms") if isinstance(response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))

    def test_providers_status_response_uses_runtime_truth_without_guard(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        response = orchestrator._providers_status_response()

        self.assertEqual("provider_status", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertTrue(response.data.get("skip_post_response_hooks", False))
        self.assertIn("providers", response.data.get("runtime_payload", {}))
        self.assertEqual(0, len(llm.chat_calls))

    def test_natural_providers_setup_query_uses_runtime_truth_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("What providers are set up right now?", "user1")

        self.assertEqual("provider_status", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertTrue(response.data["used_runtime_state"])
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertIn("providers", payload)
        self.assertIn(("providers_status", None), runtime_truth.calls)

    def test_runtime_status_unavailable_still_explains_live_ollama_instances(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
            runtime_truth_service=None,
        )
        live_snapshot = {
            "taken_at": "2026-04-13T12:00:00+00:00",
            "snapshot_local_date": "2026-04-13",
            "hostname": "testhost",
            "cpu_count": 8,
            "loadavg": (1.0, 0.8, 0.6),
            "mem": {
                "total": 64 * 1024**3,
                "used": 20 * 1024**3,
                "available": 44 * 1024**3,
                "free": 44 * 1024**3,
                "buffers": 0,
                "cached": 0,
                "shared": 0,
                "used_pct": 31.25,
            },
            "swap": {"total": 0, "used": 0},
            "top_cpu": [
                {"pid": 201, "name": "ollama", "cpu_ticks": 2200, "rss_bytes": 12 * 1024**3},
                {"pid": 202, "name": "ollama runner", "cpu_ticks": 700, "rss_bytes": 6 * 1024**3},
            ],
            "top_rss": [
                {"pid": 201, "name": "ollama", "cpu_ticks": 2200, "rss_bytes": 12 * 1024**3},
                {"pid": 202, "name": "ollama runner", "cpu_ticks": 700, "rss_bytes": 6 * 1024**3},
            ],
            "proc_stats": {"procs_scanned": 2, "errors_skipped": 0},
        }
        with patch.object(collector, "collect_live_snapshot", return_value=live_snapshot), patch.object(
            collector,
            "collect_live_process_index",
            return_value=[
                {"pid": 201, "name": "ollama"},
                {"pid": 202, "name": "ollama runner"},
            ],
        ):
            response = orchestrator.handle_message("why are there 2 ollama instances?", "user1")

        lowered = response.text.lower()
        self.assertNotIn("can't read a clean runtime status", lowered)
        self.assertIn("ollama", lowered)
        self.assertTrue("server" in lowered or "worker" in lowered or "helper" in lowered)

    def test_runtime_and_model_status_queries_use_runtime_truth_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            model_response = orchestrator.handle_message("what model are you using?", "user1")
            runtime_response = orchestrator.handle_message("runtime", "user1")
        self.assertEqual("model_status", model_response.data["route"])
        self.assertFalse(model_response.data["used_llm"])
        self.assertIn(("current_chat_target_status", None), runtime_truth.calls)
        self.assertNotIn(("chat_target_truth", None), runtime_truth.calls)
        self.assertNotIn(("model_controller_policy_status", None), runtime_truth.calls)
        model_payload = model_response.data.get("runtime_payload") if isinstance(model_response.data.get("runtime_payload"), dict) else {}
        self.assertIn("truth_timing_ms", model_payload)
        self.assertIn("current_chat_target_status_ms", model_payload["truth_timing_ms"])
        self.assertIn("chat is currently using", model_response.text.lower())
        self.assertEqual("runtime_status", runtime_response.data["route"])
        self.assertFalse(runtime_response.data["used_llm"])
        self.assertIn(("runtime_status", "runtime_status"), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))
        self.assertTrue(runtime_response.data.get("skip_post_response_hooks", False))
        runtime_timing = runtime_response.data.get("orchestrator_timing_ms") if isinstance(runtime_response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(runtime_timing.get("assistant_response_guard_ms", -1)))
        self.assertTrue(runtime_response.data.get("skip_post_response_hooks", False))
        timing = runtime_response.data.get("orchestrator_timing_ms") if isinstance(runtime_response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))

    def test_assistant_capabilities_mentions_pack_suggestions(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what can you help me with?", "user1")
        self.assertEqual("assistant_capabilities", response.data["route"])
        self.assertIn("Pack suggestions", response.text)
        self.assertIn("missing capability", response.text.lower())
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(0, len(llm.chat_calls))
        self.assertTrue(response.data.get("skip_post_response_hooks", False))
        timing = response.data.get("orchestrator_timing_ms") if isinstance(response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))

    def test_assistant_capabilities_brief_prompt_uses_concise_summary(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message(
                "say what you do in one sentence, but keep it natural",
                "user1",
            )
        self.assertEqual("assistant_capabilities", response.data["route"])
        self.assertNotIn("- System inspection:", response.text)
        self.assertIn("I can help inspect this system", response.text)
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertTrue(bool(payload.get("brief_prompt")))
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(0, len(llm.chat_calls))

    def test_assistant_capabilities_guided_thinking_prompt_uses_short_natural_reply(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message(
                "I need help thinking through something messy, but keep it simple.",
                "user1",
            )
        self.assertEqual("assistant_capabilities", response.data["route"])
        self.assertEqual(
            "Yes. Tell me the goal, the messy part, and any constraint, and I’ll help break it down simply.",
            response.text,
        )
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertTrue(bool(payload.get("guided_thinking_prompt")))
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(0, len(llm.chat_calls))

    def test_assistant_layer_question_explains_assistant_agent_boundary_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message(
                "what are you and what is the agent layer supposed to do?",
                "user1",
            )
        self.assertEqual("assistant_capabilities", response.data["route"])
        self.assertIn("You interact with me, the assistant.", response.text)
        self.assertIn("agent layer", response.text)
        self.assertIn("should not talk to you directly", response.text)
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertTrue(bool(payload.get("agent_layer_prompt")))
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(0, len(llm.chat_calls))

    def test_filesystem_queries_use_native_skill_without_llm_fallback(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        safe_root = os.path.join(self.tmpdir.name, "workspace")
        os.makedirs(safe_root, exist_ok=True)
        note_path = os.path.join(safe_root, "note.txt")
        with open(note_path, "w", encoding="utf-8") as handle:
            handle.write("native filesystem skill")
        runtime_truth.filesystem_skill = FileSystemSkill(
            allowed_roots=[safe_root],
            base_dir=safe_root,
            sensitive_roots=[os.path.join(safe_root, "private")],
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            list_response = orchestrator.handle_message(f'list files in "{safe_root}"', "user1")
            stat_response = orchestrator.handle_message(f'how big is "{note_path}"', "user1")
            read_response = orchestrator.handle_message(f'read "{note_path}"', "user1")
            relative_read_response = orchestrator.handle_message("read note.txt", "user1")

        self.assertEqual("action_tool", list_response.data["route"])
        self.assertEqual(["filesystem"], list_response.data["used_tools"])
        self.assertFalse(list_response.data["used_llm"])
        self.assertFalse(
            bool(
                (list_response.data.get("runtime_payload") if isinstance(list_response.data.get("runtime_payload"), dict) else {}).get(
                    "requires_confirmation"
                )
            )
        )
        self.assertIn("note.txt", list_response.text)
        self.assertIn(("filesystem_list_directory", safe_root), runtime_truth.calls)

        self.assertEqual("action_tool", stat_response.data["route"])
        self.assertEqual(["filesystem"], stat_response.data["used_tools"])
        self.assertFalse(stat_response.data["used_llm"])
        self.assertFalse(
            bool(
                (stat_response.data.get("runtime_payload") if isinstance(stat_response.data.get("runtime_payload"), dict) else {}).get(
                    "requires_confirmation"
                )
            )
        )
        self.assertIn("bytes", stat_response.text.lower())
        self.assertIn(("filesystem_stat_path", note_path), runtime_truth.calls)

        self.assertEqual("action_tool", read_response.data["route"])
        self.assertEqual(["filesystem"], read_response.data["used_tools"])
        self.assertFalse(read_response.data["used_llm"])
        self.assertFalse(
            bool(
                (read_response.data.get("runtime_payload") if isinstance(read_response.data.get("runtime_payload"), dict) else {}).get(
                    "requires_confirmation"
                )
            )
        )
        self.assertIn("native filesystem skill", read_response.text)
        self.assertIn(("filesystem_read_text_file", note_path), runtime_truth.calls)

        self.assertEqual("action_tool", relative_read_response.data["route"])
        self.assertEqual(["filesystem"], relative_read_response.data["used_tools"])
        self.assertFalse(relative_read_response.data["used_llm"])
        self.assertIn("native filesystem skill", relative_read_response.text)
        self.assertIn(("filesystem_read_text_file", "note.txt"), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))

    def test_filesystem_sensitive_paths_are_blocked_without_llm_fallback(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        safe_root = os.path.join(self.tmpdir.name, "workspace")
        private_root = os.path.join(safe_root, "private")
        os.makedirs(private_root, exist_ok=True)
        secret_path = os.path.join(private_root, "secret.txt")
        with open(secret_path, "w", encoding="utf-8") as handle:
            handle.write("top secret")
        runtime_truth.filesystem_skill = FileSystemSkill(
            allowed_roots=[safe_root],
            base_dir=safe_root,
            sensitive_roots=[private_root],
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message(f'read "{secret_path}"', "user1")

        self.assertEqual("action_tool", response.data["route"])
        self.assertEqual(["filesystem"], response.data["used_tools"])
        self.assertFalse(response.data["used_llm"])
        self.assertFalse(response.data["ok"])
        self.assertEqual("sensitive_path_blocked", response.data["error_kind"])
        self.assertIn("can't access that location", response.text.lower())
        self.assertIn("choose a different location", response.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_filesystem_search_queries_use_native_skill_without_llm_fallback(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        safe_root = os.path.join(self.tmpdir.name, "workspace")
        docs_dir = os.path.join(safe_root, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        readme_path = os.path.join(docs_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as handle:
            handle.write("hello ollama\nTODO item\n")
        runtime_truth.filesystem_skill = FileSystemSkill(
            allowed_roots=[safe_root],
            base_dir=safe_root,
            sensitive_roots=[os.path.join(safe_root, "private")],
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            filename_response = orchestrator.handle_message("find files named README.md", "user1")
            text_response = orchestrator.handle_message("search this repo for TODO", "user1")

        self.assertEqual("action_tool", filename_response.data["route"])
        self.assertEqual(["filesystem"], filename_response.data["used_tools"])
        self.assertFalse(filename_response.data["used_llm"])
        self.assertIn("README.md", filename_response.text)
        self.assertIn(("filesystem_search_filenames", (".", "README.md")), runtime_truth.calls)

        self.assertEqual("action_tool", text_response.data["route"])
        self.assertEqual(["filesystem"], text_response.data["used_tools"])
        self.assertFalse(text_response.data["used_llm"])
        self.assertIn("TODO", text_response.text)
        self.assertIn(("filesystem_search_text", (".", "TODO")), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))

    def test_shell_queries_use_native_skill_without_llm_fallback(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        safe_root = os.path.join(self.tmpdir.name, "workspace")
        os.makedirs(safe_root, exist_ok=True)
        runtime_truth.shell_skill = ShellSkill(
            allowed_roots=[safe_root],
            base_dir=safe_root,
            sensitive_roots=[os.path.join(safe_root, "private")],
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        def _fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            if argv == ["python", "--version"]:
                return subprocess.CompletedProcess(argv, 0, "Python 3.12.1\n", "")
            if argv == ["which", "pip"]:
                return subprocess.CompletedProcess(argv, 0, "/usr/bin/pip\n", "")
            if argv == ["apt-cache", "search", "--names-only", "ripgrep"]:
                return subprocess.CompletedProcess(argv, 0, "ripgrep - line-oriented search tool\n", "")
            raise AssertionError(f"unexpected argv: {argv!r}")

        with patch("agent.shell_skill.subprocess.run", side_effect=_fake_run), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("LLM should not run"),
        ):
            python_response = orchestrator.handle_message("what version of python do i have", "user1")
            pip_response = orchestrator.handle_message("where is pip installed", "user1")
            apt_response = orchestrator.handle_message("search apt for ripgrep", "user1")

        self.assertEqual("action_tool", python_response.data["route"])
        self.assertEqual(["shell"], python_response.data["used_tools"])
        self.assertFalse(python_response.data["used_llm"])
        self.assertFalse(
            bool(
                (python_response.data.get("runtime_payload") if isinstance(python_response.data.get("runtime_payload"), dict) else {}).get(
                    "requires_confirmation"
                )
            )
        )
        self.assertIn("Python 3.12.1", python_response.text)
        self.assertIn(("shell_execute_safe_command", ("python_version", None, None, None)), runtime_truth.calls)

        self.assertEqual("action_tool", pip_response.data["route"])
        self.assertEqual(["shell"], pip_response.data["used_tools"])
        self.assertFalse(pip_response.data["used_llm"])
        self.assertFalse(
            bool(
                (pip_response.data.get("runtime_payload") if isinstance(pip_response.data.get("runtime_payload"), dict) else {}).get(
                    "requires_confirmation"
                )
            )
        )
        self.assertIn("/usr/bin/pip", pip_response.text)
        self.assertIn(("shell_execute_safe_command", ("which", "pip", None, None)), runtime_truth.calls)

        self.assertEqual("action_tool", apt_response.data["route"])
        self.assertEqual(["shell"], apt_response.data["used_tools"])
        self.assertFalse(apt_response.data["used_llm"])
        self.assertFalse(
            bool(
                (apt_response.data.get("runtime_payload") if isinstance(apt_response.data.get("runtime_payload"), dict) else {}).get(
                    "requires_confirmation"
                )
            )
        )
        self.assertIn("ripgrep", apt_response.text.lower())
        self.assertIn(("shell_execute_safe_command", ("apt_search", None, "ripgrep", None)), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))

    def test_shell_blocks_unsupported_requests_without_llm_fallback(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        safe_root = os.path.join(self.tmpdir.name, "workspace")
        os.makedirs(safe_root, exist_ok=True)
        runtime_truth.shell_skill = ShellSkill(
            allowed_roots=[safe_root],
            base_dir=safe_root,
            sensitive_roots=[os.path.join(safe_root, "private")],
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("run ls; whoami", "user1")

        self.assertEqual("action_tool", response.data["route"])
        self.assertEqual(["shell"], response.data["used_tools"])
        self.assertFalse(response.data["used_llm"])
        self.assertFalse(response.data["ok"])
        self.assertEqual("shell_interpolation_blocked", response.data["error_kind"])
        self.assertIn("can't run that command here", response.text.lower())
        self.assertIn("one supported action at a time", response.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_shell_install_requires_confirmation_then_executes(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        safe_root = os.path.join(self.tmpdir.name, "workspace")
        os.makedirs(safe_root, exist_ok=True)
        runtime_truth.shell_skill = ShellSkill(
            allowed_roots=[safe_root],
            base_dir=safe_root,
            sensitive_roots=[os.path.join(safe_root, "private")],
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        def _fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            if argv == ["apt-get", "install", "-y", "ripgrep"]:
                return subprocess.CompletedProcess(argv, 0, "Installing ripgrep\n", "")
            raise AssertionError(f"unexpected argv: {argv!r}")

        with patch("agent.shell_skill.subprocess.run", side_effect=_fake_run), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("LLM should not run"),
        ):
            preview = orchestrator.handle_message("install ripgrep", "user1")
            self.assertNotIn(("shell_install_package", ("apt", "ripgrep", None, False, None)), runtime_truth.calls)
            confirm = orchestrator.handle_message("yes", "user1")

        preview_payload = preview.data.get("runtime_payload") if isinstance(preview.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", preview.data["route"])
        self.assertEqual(["shell"], preview.data["used_tools"])
        self.assertFalse(preview.data["used_llm"])
        self.assertTrue(preview_payload.get("requires_confirmation"))
        self.assertIn("install ripgrep", preview.text.lower())
        self.assertIn(("shell_preview_install_package", ("apt", "ripgrep", None, False, None)), runtime_truth.calls)
        self.assertEqual("action_tool", confirm.data["route"])
        self.assertEqual(["shell"], confirm.data["used_tools"])
        self.assertFalse(confirm.data["used_llm"])
        self.assertIn("Installing ripgrep", confirm.text)
        self.assertIn(("shell_install_package", ("apt", "ripgrep", None, False, None)), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))

    def test_shell_create_directory_requires_confirmation_then_executes(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        safe_root = os.path.join(self.tmpdir.name, "workspace")
        os.makedirs(safe_root, exist_ok=True)
        runtime_truth.shell_skill = ShellSkill(
            allowed_roots=[safe_root],
            base_dir=safe_root,
            sensitive_roots=[os.path.join(safe_root, "private")],
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            preview = orchestrator.handle_message("create a folder called logs in this repo", "user1")
            self.assertFalse(os.path.isdir(os.path.join(safe_root, "logs")))
            self.assertNotIn(("shell_create_directory", "logs"), runtime_truth.calls)
            confirm = orchestrator.handle_message("yes", "user1")

        preview_payload = preview.data.get("runtime_payload") if isinstance(preview.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", preview.data["route"])
        self.assertEqual(["shell"], preview.data["used_tools"])
        self.assertFalse(preview.data["used_llm"])
        self.assertTrue(preview_payload.get("requires_confirmation"))
        self.assertIn("create the directory", preview.text.lower())
        self.assertIn(("shell_preview_create_directory", "logs"), runtime_truth.calls)
        self.assertEqual("action_tool", confirm.data["route"])
        self.assertEqual(["shell"], confirm.data["used_tools"])
        self.assertFalse(confirm.data["used_llm"])
        self.assertTrue(os.path.isdir(os.path.join(safe_root, "logs")))
        self.assertIn(("shell_create_directory", "logs"), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))

    def test_natural_runtime_and_provider_health_queries_use_runtime_truth_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.openrouter_configured = True
        runtime_truth.openrouter_secret_present = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            runtime_response = orchestrator.handle_message("can you tell if everything is working with the agent?", "user1")
            provider_response = orchestrator.handle_message("openrouter health", "user1")

        self.assertEqual("runtime_status", runtime_response.data["route"])
        self.assertEqual("provider_status", provider_response.data["route"])
        self.assertEqual(0, len(llm.chat_calls))
        self.assertIn(("runtime_status", "runtime_status"), runtime_truth.calls)
        self.assertIn(("provider_status", "openrouter"), runtime_truth.calls)

    def test_operational_queries_bypass_llm_and_meta_choice(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )

        with patch.object(
            orchestrator,
            "_tool_handler_doctor",
            return_value={"ok": True, "user_text": "Doctor: OK", "data": {}},
        ) as doctor_mock, patch.object(
            orchestrator,
            "_handle_nl_observe",
            return_value=OrchestratorResponse("Memory used: 42%", {"summary": "Memory used: 42%"}),
        ) as observe_mock, patch.object(
            orchestrator,
            "_apply_assistant_response_guard",
            side_effect=AssertionError("post-response guard should not run"),
        ), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("LLM should not run"),
        ):
            doctor_response = orchestrator.handle_message("agent doctor", "user1")
            memory_response = orchestrator.handle_message("how much memory am I using?", "user1")

        self.assertEqual("operational_status", doctor_response.data["route"])
        self.assertEqual("operational_status", memory_response.data["route"])
        self.assertFalse(doctor_response.data["used_llm"])
        self.assertFalse(memory_response.data["used_llm"])
        self.assertIn("doctor", doctor_response.text.lower())
        self.assertIn("memory", memory_response.text.lower())
        self.assertTrue(doctor_response.data.get("skip_post_response_hooks", False))
        self.assertTrue(memory_response.data.get("skip_post_response_hooks", False))
        doctor_timing = doctor_response.data.get("orchestrator_timing_ms") if isinstance(doctor_response.data.get("orchestrator_timing_ms"), dict) else {}
        memory_timing = memory_response.data.get("orchestrator_timing_ms") if isinstance(memory_response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(doctor_timing.get("assistant_response_guard_ms", -1)))
        self.assertEqual(0, int(memory_timing.get("assistant_response_guard_ms", -1)))
        self.assertEqual(0, len(llm.chat_calls))

    def test_governance_adapter_query_uses_runtime_truth_service_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what managed adapters exist?", "user1")

        self.assertEqual("governance_status", response.data["route"])
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("governance_managed_adapters", payload.get("type"))
        self.assertIn("Telegram", response.text)
        self.assertIn(("list_managed_adapters", None), runtime_truth.calls)
        self.assertTrue(response.data.get("skip_post_response_hooks", False))
        timing = response.data.get("orchestrator_timing_ms") if isinstance(response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))
        self.assertEqual(0, len(llm.chat_calls))

    def test_governance_pending_query_uses_runtime_truth_service_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("is any skill waiting for approval?", "user1")

        self.assertEqual("governance_status", response.data["route"])
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("governance_pending", payload.get("type"))
        self.assertIn("scheduled_sync", response.text)
        self.assertIn(("list_pending_governance_requests", None), runtime_truth.calls)
        self.assertTrue(response.data.get("skip_post_response_hooks", False))
        timing = response.data.get("orchestrator_timing_ms") if isinstance(response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))
        self.assertEqual(0, len(llm.chat_calls))

    def test_governance_skill_status_query_uses_runtime_truth_service_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what execution mode does skill scheduled_sync use?", "user1")

        self.assertEqual("model_policy_status", response.data["route"])
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("model_controller_policy", payload.get("type"))
        self.assertEqual("safe", payload.get("mode"))
        self.assertIn("Mode: SAFE MODE.", response.text)
        self.assertIn(("model_controller_policy_status", None), runtime_truth.calls)
        self.assertTrue(response.data.get("skip_post_response_hooks", False))
        timing = response.data.get("orchestrator_timing_ms") if isinstance(response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))
        self.assertEqual(0, len(llm.chat_calls))

    def test_governance_execution_mode_query_uses_runtime_truth_service_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what execution mode does Telegram use?", "user1")

        self.assertEqual("model_policy_status", response.data["route"])
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("model_controller_policy", payload.get("type"))
        self.assertEqual("safe", payload.get("mode"))
        self.assertIn("Mode: SAFE MODE.", response.text)
        self.assertIn(("model_controller_policy_status", None), runtime_truth.calls)
        self.assertTrue(response.data.get("skip_post_response_hooks", False))
        timing = response.data.get("orchestrator_timing_ms") if isinstance(response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))
        self.assertEqual(0, len(llm.chat_calls))

    def test_model_policy_status_query_uses_runtime_truth_service_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what is my model selection policy?", "user1")

        self.assertEqual("model_policy_status", response.data["route"])
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("model_policy_status", payload.get("type"))
        self.assertIn("local first", response.text.lower())
        self.assertIn(("model_policy_status", None), runtime_truth.calls)
        self.assertTrue(response.data.get("skip_post_response_hooks", False))
        timing = response.data.get("orchestrator_timing_ms") if isinstance(response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))
        self.assertEqual(0, len(llm.chat_calls))

    def test_model_policy_current_choice_query_uses_runtime_truth_service_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("why are you using this model?", "user1")

        self.assertEqual("model_policy_status", response.data["route"])
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("model_policy_explanation", payload.get("type"))
        self.assertIn("current default already matches the best candidate", response.text.lower())
        self.assertIn(("model_policy_status", None), runtime_truth.calls)
        self.assertTrue(response.data.get("skip_post_response_hooks", False))
        timing = response.data.get("orchestrator_timing_ms") if isinstance(response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))
        self.assertEqual(0, len(llm.chat_calls))

    def test_model_policy_candidate_queries_use_runtime_truth_service_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            free_remote = orchestrator.handle_message("what free remote model would you choose?", "user1")
            provider_reason = orchestrator.handle_message("why didn't you switch to openrouter?", "user1")

        self.assertEqual("model_policy_status", free_remote.data["route"])
        free_payload = free_remote.data.get("runtime_payload") if isinstance(free_remote.data.get("runtime_payload"), dict) else {}
        self.assertEqual("model_policy_candidate", free_payload.get("type"))
        self.assertIn("free remote", free_remote.text.lower())
        self.assertTrue(free_remote.data.get("skip_post_response_hooks", False))
        self.assertEqual("model_policy_status", provider_reason.data["route"])
        provider_payload = provider_reason.data.get("runtime_payload") if isinstance(provider_reason.data.get("runtime_payload"), dict) else {}
        self.assertEqual("model_policy_explanation", provider_payload.get("type"))
        self.assertIn("openrouter", provider_reason.text.lower())
        self.assertTrue(provider_reason.data.get("skip_post_response_hooks", False))
        self.assertIn(("model_policy_candidate", {"tier": "free_remote", "status": None}), runtime_truth.calls)
        self.assertIn(("model_policy_provider_candidate", "openrouter"), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))

    def test_runtime_openrouter_setup_prompt_and_key_submission_use_runtime_truth(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            first = orchestrator.handle_message("help me set up openrouter", "user1")
            second = orchestrator.handle_message("sk-or-v1-testsecret1234567890", "user1")
        self.assertIn("API key", first.text)
        self.assertEqual("setup_flow", first.data["route"])
        self.assertTrue(first.data["used_runtime_state"])
        self.assertFalse(first.data["used_memory"])
        self.assertIn("OpenRouter", second.text)
        self.assertEqual("setup_flow", second.data["route"])
        self.assertTrue(second.data["used_runtime_state"])
        self.assertTrue(second.data["used_memory"])
        self.assertFalse(second.data["used_llm"])
        self.assertIn(
            (
                "configure_openrouter",
                {
                    "api_key": "sk-or-v1-testsecret1234567890",
                    "payload": {"make_default": False, "defer_model_refresh": True},
                },
            ),
            runtime_truth.calls,
        )
        self.assertEqual(0, len(llm.chat_calls))

    def test_runtime_configure_openrouter_with_stored_key_prompts_before_reuse(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.openrouter_secret_present = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            first = orchestrator.handle_message("configure openrouter", "user1")
            calls_after_first = list(runtime_truth.calls)
            second = orchestrator.handle_message("yes", "user1")

        self.assertEqual("setup_flow", first.data["route"])
        self.assertIn("stored", first.text.lower())
        self.assertIn("OpenRouter", first.text)
        self.assertTrue(first.data["used_runtime_state"])
        self.assertFalse(first.data["used_llm"])
        self.assertNotIn(
            (
                "configure_openrouter",
                {"api_key": None, "payload": {"make_default": False, "defer_model_refresh": True}},
            ),
            calls_after_first,
        )

        self.assertEqual("setup_flow", second.data["route"])
        self.assertIn("OpenRouter", second.text)
        self.assertIn(
            (
                "configure_openrouter",
                {"api_key": None, "payload": {"make_default": False, "defer_model_refresh": True}},
            ),
            runtime_truth.calls,
        )
        self.assertEqual(0, len(llm.chat_calls))

    def test_runtime_confirmed_switch_uses_exact_offered_target(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.openrouter_secret_present = True
        runtime_truth.openrouter_model = "openrouter:ai21/jamba-large-1.7"
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            first = orchestrator.handle_message("configure openrouter", "user1")
            second = orchestrator.handle_message("yes", "user1")
            third = orchestrator.handle_message("yes", "user1")

        self.assertEqual("setup_flow", first.data["route"])
        self.assertIn("stored", first.text.lower())
        self.assertEqual("setup_flow", second.data["route"])
        self.assertIn("openrouter:ai21/jamba-large-1.7", second.text)
        self.assertEqual("setup_flow", third.data["route"])
        self.assertEqual("Now using openrouter:ai21/jamba-large-1.7 for chat.", third.text)
        self.assertIn(
            (
                "set_confirmed_chat_model_target",
                {
                    "model_id": "openrouter:ai21/jamba-large-1.7",
                    "provider_id": "openrouter",
                },
            ),
            runtime_truth.calls,
        )
        self.assertNotIn(("set_default_chat_model", "openrouter:ai21/jamba-large-1.7"), runtime_truth.calls)
        self.assertEqual("openrouter:ai21/jamba-large-1.7", runtime_truth.current_model)
        self.assertEqual(0, len(llm.chat_calls))

    def test_runtime_confirmed_switch_reports_unavailable_target_without_drift(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.openrouter_secret_present = True
        runtime_truth.openrouter_model = "openrouter:ai21/jamba-large-1.7"
        runtime_truth.unavailable_confirmed_targets.add("openrouter:ai21/jamba-large-1.7")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            orchestrator.handle_message("configure openrouter", "user1")
            orchestrator.handle_message("yes", "user1")
            third = orchestrator.handle_message("yes", "user1")
            fourth = orchestrator.handle_message("yes", "user1")

        self.assertEqual("setup_flow", third.data["route"])
        self.assertFalse(third.data["ok"])
        self.assertIn("no longer available", third.text.lower())
        self.assertEqual("ollama:qwen3.5:4b", runtime_truth.current_model)
        self.assertIn("No resumable work is active", fourth.text)
        self.assertEqual(0, len(llm.chat_calls))

    def test_direct_model_switch_uses_exact_target_without_drift(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("switch to qwen2.5:7b-instruct", "user1")

        self.assertEqual("model_status", response.data["route"])
        self.assertEqual(
            "I will switch chat to ollama:qwen2.5:7b-instruct. This mutates the active chat target. Say yes to continue, or no to cancel.",
            response.text,
        )
        self.assertNotIn(("set_confirmed_chat_model_target", {"model_id": "ollama:qwen2.5:7b-instruct", "provider_id": "ollama"}), runtime_truth.calls)
        self.assertNotIn(("set_default_chat_model", "ollama:qwen2.5:7b-instruct"), runtime_truth.calls)
        self.assertEqual("ollama:qwen3.5:4b", runtime_truth.current_model)
        self.assertEqual(0, len(llm.chat_calls))

    def test_direct_model_switch_records_previous_target_for_switch_back(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            first = orchestrator.handle_message("switch to qwen2.5:7b-instruct", "user1")
            second = orchestrator.handle_message("switch back", "user1")

        self.assertEqual(
            "I will switch chat to ollama:qwen2.5:7b-instruct. This mutates the active chat target. Say yes to continue, or no to cancel.",
            first.text,
        )
        self.assertIn("do not have a recent trial model switch", second.text.lower())
        self.assertEqual("ollama:qwen3.5:4b", runtime_truth.current_model)
        self.assertNotIn(("set_confirmed_chat_model_target", {"model_id": "ollama:qwen3.5:4b", "provider_id": "ollama"}), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))

    def test_switch_to_unhealthy_model_reports_issue_and_offers_rollback(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()

        def _unhealthy_switch(model_id: str, *, provider_id: str | None = None):  # type: ignore[no-untyped-def]
            runtime_truth.calls.append(
                ("set_confirmed_chat_model_target", {"model_id": model_id, "provider_id": provider_id})
            )
            runtime_truth.current_provider = str(provider_id or str(model_id).split(":", 1)[0]).strip().lower()
            runtime_truth.current_model = str(model_id)
            runtime_truth.current_ready = False
            runtime_truth.current_provider_health_status = "down"
            runtime_truth.current_model_health_status = "down"
            return True, {
                "ok": True,
                "provider": runtime_truth.current_provider,
                "model_id": runtime_truth.current_model,
                "message": f"Now using {runtime_truth.current_model} for chat.",
            }

        runtime_truth.set_confirmed_chat_model_target = _unhealthy_switch  # type: ignore[method-assign]
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("switch to qwen2.5:7b-instruct", "user1")
            rollback = orchestrator.handle_message("switch back", "user1")

        self.assertEqual("model_status", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertIn("i will switch chat to ollama:qwen2.5:7b-instruct", response.text.lower())
        self.assertIn("say yes to continue, or no to cancel", response.text.lower())
        self.assertNotIn("isn't responding properly right now", response.text.lower())
        self.assertIn("do not have a recent trial model switch", rollback.text.lower())
        self.assertEqual("ollama:qwen3.5:4b", runtime_truth.current_model)

    def test_ollama_status_reports_health_and_reason(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_provider = "ollama"
        runtime_truth.current_model = "ollama:qwen3.5:4b"
        runtime_truth.current_ready = False
        runtime_truth.current_provider_health_status = "down"
        runtime_truth.current_model_health_status = "down"
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("ollama status", "user1")

        self.assertEqual("provider_status", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        lowered = response.text.lower()
        self.assertIn("ollama is currently down", lowered)
        self.assertIn("chat is configured for ollama:qwen3.5:4b", lowered)
        self.assertIn("timeout while reaching ollama", lowered)
        self.assertNotIn("configured, but it is not responding right now", lowered)
        self.assertEqual(0, len(llm.chat_calls))

    def test_repair_followup_reuses_recent_unhealthy_context_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_provider = "ollama"
        runtime_truth.current_model = "ollama:qwen3.5:4b"
        runtime_truth.current_ready = False
        runtime_truth.current_provider_health_status = "down"
        runtime_truth.current_model_health_status = "down"
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            first = orchestrator.handle_message("what model are you using?", "user1")
            second = orchestrator.handle_message("can you repair it?", "user1")
            third = orchestrator.handle_message("can you fix ollama?", "user1")

        self.assertEqual("model_status", first.data["route"])
        self.assertEqual("setup_flow", second.data["route"])
        self.assertEqual("interpretation_followup", third.data["route"])
        self.assertFalse(second.data["used_llm"])
        self.assertFalse(third.data["used_llm"])
        self.assertIn("ollama is currently down", second.text.lower())
        self.assertIn("reconfigure ollama", second.text.lower())
        self.assertIn("likely cause", third.text.lower())
        self.assertNotIn("chat, ask, or model check/switch", second.text.lower())
        self.assertNotIn("chat, ask, or model check/switch", third.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_repair_followup_rechecks_current_provider_truth_after_stale_unhealthy_context(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_provider = "ollama"
        runtime_truth.current_model = "ollama:qwen3.5:4b"
        runtime_truth.current_ready = False
        runtime_truth.current_provider_health_status = "down"
        runtime_truth.current_model_health_status = "down"
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            first = orchestrator.handle_message("what model are you using?", "user1")
            runtime_truth.current_ready = True
            runtime_truth.current_provider_health_status = "ok"
            runtime_truth.current_model_health_status = "ok"
            second = orchestrator.handle_message("can you repair it?", "user1")

        self.assertEqual("model_status", first.data["route"])
        self.assertEqual("setup_flow", second.data["route"])
        self.assertFalse(second.data["used_llm"])
        self.assertIn("reachable again", second.text.lower())
        self.assertIn("looks healthy now", second.text.lower())
        self.assertNotIn("chat, ask, or model check/switch", second.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_repair_followup_keeps_grounded_local_recovery_when_model_unhealthy(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_provider = "ollama"
        runtime_truth.current_model = "ollama:qwen3.5:4b"
        runtime_truth.current_ready = False
        runtime_truth.current_provider_health_status = "ok"
        runtime_truth.current_model_health_status = "down"
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        inventory_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": False,
                    "active": True,
                    "availability_reason": "model health is down",
                },
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                },
            ],
            "usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
            "other_usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
            "not_ready_models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": False,
                    "active": True,
                    "availability_reason": "model health is down",
                }
            ],
        }
        inventory_status_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [dict(row) for row in inventory_payload["models"]],
            "local_installed_models": [
                dict(row)
                for row in inventory_payload["models"]
                if bool(row.get("local", False)) and bool(row.get("available", False))
            ],
        }
        readiness_status_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [dict(row) for row in inventory_payload["models"]],
            "ready_now_models": [dict(row) for row in inventory_payload["usable_models"]],
            "other_ready_now_models": [dict(row) for row in inventory_payload["other_usable_models"]],
            "not_ready_models": [dict(row) for row in inventory_payload["not_ready_models"]],
        }

        with patch.object(runtime_truth, "model_inventory_status", return_value=inventory_status_payload), patch.object(
            runtime_truth,
            "model_readiness_status",
            return_value=readiness_status_payload,
        ), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("LLM should not run"),
        ):
            first = orchestrator.handle_message("what model are you using?", "user1")
            second = orchestrator.handle_message("Help me get this working", "user1")
            third = orchestrator.handle_message("1", "user1")
            fourth = orchestrator.handle_message("2", "user1")

        self.assertEqual("model_status", first.data["route"])
        self.assertEqual("setup_flow", second.data["route"])
        self.assertFalse(second.data["used_llm"])
        second_text = second.text.lower()
        self.assertIn("ollama is reachable", second_text)
        self.assertIn("current chat model ollama:qwen3.5:4b is not healthy right now", second_text)
        self.assertIn("1) recheck ollama:qwen3.5:4b now", second_text)
        self.assertIn("2) switch to ollama:qwen2.5:3b-instruct", second_text)
        self.assertNotIn("no chat model available", second_text)
        self.assertNotIn("start ollama locally", second_text)
        self.assertNotIn("install a local chat model", second_text)
        self.assertNotIn("chat, ask, or model check/switch", second_text)

        self.assertEqual("setup_flow", third.data["route"])
        self.assertIn("ollama is reachable", third.text.lower())
        self.assertIn("not healthy right now", third.text.lower())
        self.assertNotIn("no chat model available", third.text.lower())

        self.assertEqual("setup_flow", fourth.data["route"])
        self.assertIn("switch chat to ollama:qwen2.5:3b-instruct now", fourth.text.lower())
        self.assertNotIn("no chat model available", fourth.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_unhealthy_runtime_context_keeps_setup_and_attention_followups_in_repair_flow(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_provider = "ollama"
        runtime_truth.current_model = "ollama:qwen3.5:4b"
        runtime_truth.current_ready = False
        runtime_truth.current_provider_health_status = "ok"
        runtime_truth.current_model_health_status = "down"
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        inventory_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": False,
                    "active": True,
                    "availability_reason": "model health is down",
                },
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                },
            ],
            "usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
            "other_usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
            "not_ready_models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": False,
                    "active": True,
                    "availability_reason": "model health is down",
                }
            ],
        }
        inventory_status_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [dict(row) for row in inventory_payload["models"]],
            "local_installed_models": [
                dict(row)
                for row in inventory_payload["models"]
                if bool(row.get("local", False)) and bool(row.get("available", False))
            ],
        }
        readiness_status_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [dict(row) for row in inventory_payload["models"]],
            "ready_now_models": [dict(row) for row in inventory_payload["usable_models"]],
            "other_ready_now_models": [dict(row) for row in inventory_payload["other_usable_models"]],
            "not_ready_models": [dict(row) for row in inventory_payload["not_ready_models"]],
        }

        with patch.object(runtime_truth, "model_inventory_status", return_value=inventory_status_payload), patch.object(
            runtime_truth,
            "model_readiness_status",
            return_value=readiness_status_payload,
        ), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("LLM should not run"),
        ):
            first = orchestrator.handle_message("what model are you using?", "user1")
            second = orchestrator.handle_message("What needs attention?", "user1")
            third = orchestrator.handle_message("Check setup and explain what's wrong", "user1")

        self.assertEqual("model_status", first.data["route"])
        for response in (second, third):
            self.assertEqual("setup_flow", response.data["route"])
            self.assertFalse(response.data["used_llm"])
            lowered = response.text.lower()
            self.assertIn("ollama is reachable", lowered)
            self.assertIn("current chat model ollama:qwen3.5:4b is not healthy right now", lowered)
            self.assertIn("1) recheck ollama:qwen3.5:4b now", lowered)
            self.assertIn("2) switch to ollama:qwen2.5:3b-instruct", lowered)
            self.assertNotIn("what can i help you with", lowered)
            self.assertNotIn("what needs attention?", lowered)
            self.assertNotIn("no chat model available", lowered)
            self.assertNotIn("start ollama locally", lowered)
            self.assertNotIn("chat, ask, or model check/switch", lowered)
        self.assertEqual(0, len(llm.chat_calls))

    def test_safe_mode_containment_blocks_generic_escape_for_unhealthy_repair_followup(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="What can I help you with?")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_provider = "ollama"
        runtime_truth.current_model = "ollama:qwen3.5:4b"
        runtime_truth.current_ready = False
        runtime_truth.current_provider_health_status = "ok"
        runtime_truth.current_model_health_status = "down"
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        inventory_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": False,
                    "active": True,
                    "availability_reason": "model health is down",
                },
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                },
            ],
            "usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
            "other_usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
        }
        inventory_status_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [dict(row) for row in inventory_payload["models"]],
            "local_installed_models": [
                dict(row)
                for row in inventory_payload["models"]
                if bool(row.get("local", False)) and bool(row.get("available", False))
            ],
        }
        readiness_status_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [dict(row) for row in inventory_payload["models"]],
            "ready_now_models": [dict(row) for row in inventory_payload["usable_models"]],
            "other_ready_now_models": [dict(row) for row in inventory_payload["other_usable_models"]],
            "not_ready_models": [],
        }

        with patch.object(runtime_truth, "model_inventory_status", return_value=inventory_status_payload), patch.object(
            runtime_truth,
            "model_readiness_status",
            return_value=readiness_status_payload,
        ), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("LLM should not run"),
        ):
            first = orchestrator.handle_message("what model are you using?", "user1")
            self.assertEqual("model_status", first.data["route"])

        with patch.object(orchestrator, "_handle_runtime_truth_chat", return_value=None), patch.object(
            orchestrator,
            "_handle_action_tool_intent",
            return_value=None,
        ), patch.object(
            orchestrator,
            "_grounded_system_fallback_response",
            return_value=None,
        ), patch.object(
            orchestrator,
            "_llm_chat",
            side_effect=AssertionError("generic chat should not run for grounded repair followup"),
        ):
            response = orchestrator.handle_message("Help me get this working", "user1")

        self.assertEqual("setup_flow", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        lowered = response.text.lower()
        self.assertIn("ollama is reachable", lowered)
        self.assertIn("current chat model ollama:qwen3.5:4b is not healthy right now", lowered)
        self.assertNotIn("what can i help you with", lowered)
        self.assertNotIn("no chat model available", lowered)
        self.assertTrue(response.data.get("skip_post_response_hooks", False))
        timing = response.data.get("orchestrator_timing_ms") if isinstance(response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))

    def test_safe_mode_containment_blocks_generic_escape_for_setup_explanation(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="What can I help you with?")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_provider = "ollama"
        runtime_truth.current_model = "ollama:qwen3.5:4b"
        runtime_truth.current_ready = False
        runtime_truth.current_provider_health_status = "ok"
        runtime_truth.current_model_health_status = "down"
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        with patch.object(
            runtime_truth,
            "model_status",
            return_value={
                "current": {
                    "provider": "ollama",
                    "model_id": "ollama:qwen3.5:4b",
                },
                "llm_availability": {
                    "available": False,
                    "reason": "model_unhealthy",
                    "ollama": {"native_ok": True},
                },
            },
        ), patch.object(
            runtime_truth,
            "model_inventory_status",
            return_value={
                "active_provider": "ollama",
                "active_model": "ollama:qwen3.5:4b",
                "configured_provider": "ollama",
                "configured_model": "ollama:qwen3.5:4b",
                "models": [
                    {
                        "model_id": "ollama:qwen3.5:4b",
                        "provider_id": "ollama",
                        "local": True,
                        "available": True,
                        "installed_local": True,
                        "active": True,
                    },
                    {
                        "model_id": "ollama:qwen2.5:3b-instruct",
                        "provider_id": "ollama",
                        "local": True,
                        "available": True,
                        "installed_local": True,
                        "active": False,
                    },
                ],
                "local_installed_models": [
                    {
                        "model_id": "ollama:qwen3.5:4b",
                        "provider_id": "ollama",
                        "local": True,
                        "available": True,
                        "installed_local": True,
                        "active": True,
                    },
                    {
                        "model_id": "ollama:qwen2.5:3b-instruct",
                        "provider_id": "ollama",
                        "local": True,
                        "available": True,
                        "installed_local": True,
                        "active": False,
                    },
                ],
            },
        ), patch.object(
            runtime_truth,
            "model_readiness_status",
            return_value={
                "active_provider": "ollama",
                "active_model": "ollama:qwen3.5:4b",
                "configured_provider": "ollama",
                "configured_model": "ollama:qwen3.5:4b",
                "models": [
                    {
                        "model_id": "ollama:qwen3.5:4b",
                        "provider_id": "ollama",
                        "local": True,
                        "usable_now": False,
                        "active": True,
                        "availability_reason": "model health is down",
                    },
                    {
                        "model_id": "ollama:qwen2.5:3b-instruct",
                        "provider_id": "ollama",
                        "local": True,
                        "usable_now": True,
                        "active": False,
                        "availability_reason": "healthy and ready now",
                    },
                ],
                "ready_now_models": [
                    {
                        "model_id": "ollama:qwen2.5:3b-instruct",
                        "provider_id": "ollama",
                        "local": True,
                        "usable_now": True,
                        "active": False,
                        "availability_reason": "healthy and ready now",
                    }
                ],
                "other_ready_now_models": [
                    {
                        "model_id": "ollama:qwen2.5:3b-instruct",
                        "provider_id": "ollama",
                        "local": True,
                        "usable_now": True,
                        "active": False,
                        "availability_reason": "healthy and ready now",
                    }
                ],
                "not_ready_models": [
                    {
                        "model_id": "ollama:qwen3.5:4b",
                        "provider_id": "ollama",
                        "local": True,
                        "usable_now": False,
                        "active": True,
                        "availability_reason": "model health is down",
                    }
                ],
            },
        ), patch.object(orchestrator, "_handle_runtime_truth_chat", return_value=None), patch.object(
            orchestrator,
            "_handle_action_tool_intent",
            return_value=None,
        ), patch.object(
            orchestrator,
            "_grounded_system_fallback_response",
            return_value=None,
        ), patch.object(
            orchestrator,
            "_llm_chat",
            side_effect=AssertionError("generic chat should not run for setup explanation"),
        ):
            response = orchestrator.handle_message("Check setup and explain what's wrong", "user1")

        self.assertEqual("setup_flow", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        lowered = response.text.lower()
        self.assertIn("setup needs attention right now", lowered)
        self.assertIn("chat is configured for ollama:qwen3.5:4b on ollama", lowered)
        self.assertNotIn("what can i help you with", lowered)
        self.assertNotIn("no chat model available", lowered)

    def test_llm_chat_rechecks_safe_mode_containment_before_generic_fallback(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="What can I help you with?")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_provider = "ollama"
        runtime_truth.current_model = "ollama:qwen3.5:4b"
        runtime_truth.current_ready = False
        runtime_truth.current_provider_health_status = "ok"
        runtime_truth.current_model_health_status = "down"
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        original_containment = orchestrator._safe_mode_containment_response
        containment_calls = {"count": 0}

        def _containment_side_effect(user_id: str, text: str):  # type: ignore[no-untyped-def]
            containment_calls["count"] += 1
            if containment_calls["count"] == 1:
                return None
            return original_containment(user_id, text)

        with patch.object(
            runtime_truth,
            "model_status",
            return_value={
                "current": {
                    "provider": "ollama",
                    "model_id": "ollama:qwen3.5:4b",
                },
                "llm_availability": {
                    "available": False,
                    "reason": "model_unhealthy",
                    "ollama": {"native_ok": True},
                },
            },
        ), patch.object(
            runtime_truth,
            "model_inventory_status",
            return_value={
                "active_provider": "ollama",
                "active_model": "ollama:qwen3.5:4b",
                "configured_provider": "ollama",
                "configured_model": "ollama:qwen3.5:4b",
                "models": [
                    {
                        "model_id": "ollama:qwen3.5:4b",
                        "provider_id": "ollama",
                        "local": True,
                        "available": True,
                        "installed_local": True,
                        "active": True,
                    },
                    {
                        "model_id": "ollama:qwen2.5:3b-instruct",
                        "provider_id": "ollama",
                        "local": True,
                        "available": True,
                        "installed_local": True,
                        "active": False,
                    },
                ],
                "local_installed_models": [
                    {
                        "model_id": "ollama:qwen3.5:4b",
                        "provider_id": "ollama",
                        "local": True,
                        "available": True,
                        "installed_local": True,
                        "active": True,
                    },
                    {
                        "model_id": "ollama:qwen2.5:3b-instruct",
                        "provider_id": "ollama",
                        "local": True,
                        "available": True,
                        "installed_local": True,
                        "active": False,
                    },
                ],
            },
        ), patch.object(
            runtime_truth,
            "model_readiness_status",
            return_value={
                "active_provider": "ollama",
                "active_model": "ollama:qwen3.5:4b",
                "configured_provider": "ollama",
                "configured_model": "ollama:qwen3.5:4b",
                "models": [
                    {
                        "model_id": "ollama:qwen3.5:4b",
                        "provider_id": "ollama",
                        "local": True,
                        "usable_now": False,
                        "active": True,
                        "availability_reason": "model health is down",
                    },
                    {
                        "model_id": "ollama:qwen2.5:3b-instruct",
                        "provider_id": "ollama",
                        "local": True,
                        "usable_now": True,
                        "active": False,
                        "availability_reason": "healthy and ready now",
                    },
                ],
                "ready_now_models": [
                    {
                        "model_id": "ollama:qwen2.5:3b-instruct",
                        "provider_id": "ollama",
                        "local": True,
                        "usable_now": True,
                        "active": False,
                        "availability_reason": "healthy and ready now",
                    }
                ],
                "other_ready_now_models": [
                    {
                        "model_id": "ollama:qwen2.5:3b-instruct",
                        "provider_id": "ollama",
                        "local": True,
                        "usable_now": True,
                        "active": False,
                        "availability_reason": "healthy and ready now",
                    }
                ],
                "not_ready_models": [
                    {
                        "model_id": "ollama:qwen3.5:4b",
                        "provider_id": "ollama",
                        "local": True,
                        "usable_now": False,
                        "active": True,
                        "availability_reason": "model health is down",
                    }
                ],
            },
        ), patch.object(
            orchestrator,
            "_safe_mode_containment_response",
            side_effect=_containment_side_effect,
        ), patch.object(
            orchestrator,
            "_handle_runtime_truth_chat",
            return_value=None,
        ), patch.object(
            orchestrator,
            "_handle_action_tool_intent",
            return_value=None,
        ), patch.object(
            orchestrator,
            "_grounded_system_fallback_response",
            return_value=None,
        ), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("LLM should not run after final containment"),
        ):
            response = orchestrator.handle_message("Check setup and explain what's wrong", "user1")

        self.assertGreaterEqual(containment_calls["count"], 2)
        self.assertEqual("setup_flow", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        lowered = response.text.lower()
        self.assertIn("setup needs attention right now", lowered)
        self.assertNotIn("what can i help you with", lowered)

    def test_grounded_process_followup_uses_fresh_resource_probe(self) -> None:
        orchestrator = self._orchestrator()
        with patch.object(
            orchestrator,
            "_looks_like_grounded_system_query",
            return_value=True,
        ), patch(
            "agent.orchestrator.resource_followup",
            return_value="live-answer",
        ) as mock_followup:
            response = orchestrator._grounded_system_fallback_response(
                "user1",
                "is pid 3367 still running?",
                allow_actions=False,
            )

        self.assertIsNotNone(response)
        self.assertEqual("live-answer", response.text)
        mock_followup.assert_called_once()
        self.assertEqual("process_state", mock_followup.call_args.args[2])
        self.assertEqual("is pid 3367 still running?", mock_followup.call_args.kwargs.get("question"))

    def test_assistant_unmatched_runtime_overview_prompt_returns_grounded_status(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="I am DeepSeek.")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("system report", "user1")

        self.assertEqual("runtime_status", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertIn("ready", response.text.lower())
        self.assertNotIn("deepseek", response.text.lower())

    def test_assistant_unmatched_low_signal_prompt_returns_bounded_response(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="I am GPT and I can help.")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        with patch(
            "agent.orchestrator.route_inference",
            return_value={
                "ok": True,
                "text": "I can help with that.",
                "provider": None,
                "model": None,
                "duration_ms": 1,
                "attempts": [],
            },
        ):
            response = orchestrator.handle_message("do the thing", "user1")

        self.assertEqual("generic_chat", response.data["route"])
        self.assertTrue(response.data["used_llm"])
        self.assertIn("i can help with that", response.text.lower())
        self.assertNotIn("gpt", response.text.lower())

    def test_repair_followup_prefers_split_inventory_and_readiness_surfaces(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_provider = "ollama"
        runtime_truth.current_model = "ollama:qwen3.5:4b"
        runtime_truth.current_ready = False
        runtime_truth.current_provider_health_status = "ok"
        runtime_truth.current_model_health_status = "down"
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        inventory_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "installed_local": True,
                    "active": True,
                },
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "installed_local": True,
                    "active": False,
                },
            ],
            "local_installed_models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "installed_local": True,
                    "active": True,
                },
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "installed_local": True,
                    "active": False,
                },
            ],
        }
        readiness_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "usable_now": False,
                    "active": True,
                    "availability_reason": "model health is down",
                },
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "healthy and ready now",
                },
            ],
            "ready_now_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "healthy and ready now",
                }
            ],
            "other_ready_now_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "healthy and ready now",
                }
            ],
            "not_ready_models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "usable_now": False,
                    "active": True,
                    "availability_reason": "model health is down",
                }
            ],
        }

        with patch.object(runtime_truth, "model_inventory_status", return_value=inventory_payload), patch.object(
            runtime_truth,
            "model_readiness_status",
            return_value=readiness_payload,
        ), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("LLM should not run"),
        ):
            orchestrator.handle_message("what model are you using?", "user1")
            response = orchestrator.handle_message("Help me get this working", "user1")

        self.assertEqual("setup_flow", response.data["route"])
        self.assertIn("ollama is reachable", response.text.lower())
        self.assertIn("switch to ollama:qwen2.5:3b-instruct", response.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_setup_explanation_uses_canonical_setup_status(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        setup_payload = {
            "setup_state": "ready",
            "ready": True,
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "effective_provider": "ollama",
            "effective_model": "ollama:qwen3.5:4b",
            "provider_health_status": "ok",
            "provider_health_reason": None,
            "model_health_status": "ok",
            "local_installed_models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "installed_local": True,
                    "active": True,
                },
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "installed_local": True,
                    "active": False,
                },
            ],
            "other_local_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "installed_local": True,
                    "active": False,
                }
            ],
        }

        with patch.object(runtime_truth, "setup_status", return_value=setup_payload), patch.object(
            runtime_truth,
            "model_status",
            side_effect=AssertionError("setup explanation should use canonical setup status"),
        ), patch.object(
            runtime_truth,
            "model_inventory_status",
            side_effect=AssertionError("setup explanation should use canonical setup status"),
        ), patch.object(
            runtime_truth,
            "model_readiness_status",
            side_effect=AssertionError("setup explanation should use canonical setup status"),
        ), patch.object(
            runtime_truth,
            "provider_status",
            side_effect=AssertionError("setup explanation should use canonical setup status"),
        ):
            response = orchestrator._setup_explanation_response(used_memory=False)

        self.assertEqual("setup_flow", response.data["route"])
        self.assertIn("setup looks okay right now", response.text.lower())
        self.assertIn("ollama is reachable", response.text.lower())
        self.assertIn("ollama:qwen2.5:3b-instruct", response.text.lower())
        self.assertTrue(response.data.get("skip_post_response_hooks", False))

    def test_setup_explanation_without_chat_model_uses_canonical_no_llm_copy(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_ready = False
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

        with patch.object(
            runtime_truth,
            "setup_status",
            return_value={
                "setup_state": "unavailable",
                "active_provider": None,
                "active_model": None,
                "provider_health_status": "unknown",
                "provider_health_reason": None,
            },
        ):
            response = orchestrator._setup_explanation_response(used_memory=False)

        self.assertEqual("setup_flow", response.data["route"])
        self.assertEqual(build_no_llm_public_message(), response.text)
        self.assertNotIn("No chat model is available right now", response.text)
        self.assertNotIn("provider", response.text.lower())
        self.assertTrue(response.data.get("skip_post_response_hooks", False))

    def test_direct_model_switch_asks_specific_clarification_when_bare_name_is_ambiguous(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.additional_available_models.append(
            {
                "model_id": "openrouter:qwen2.5:7b-instruct",
                "provider_id": "openrouter",
                "available": True,
                "usable_now": True,
                "active": False,
                "quality_rank": 8,
                "availability_reason": "ready",
            }
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("switch to qwen2.5:7b-instruct", "user1")

        self.assertEqual("model_status", response.data["route"])
        self.assertIn("more than one provider", response.text.lower())
        self.assertIn("ollama:qwen2.5:7b-instruct", response.text)
        self.assertIn("openrouter:qwen2.5:7b-instruct", response.text)
        self.assertNotIn("what are you referring to", response.text.lower())
        self.assertNotIn("set_confirmed_chat_model_target", [call[0] for call in runtime_truth.calls])
        self.assertEqual(0, len(llm.chat_calls))

    def test_local_model_inventory_questions_use_grounded_local_inventory(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        prompts = (
            "what ollama models do we have downloaded?",
            "what models do we have downloaded?",
            "what local models are available?",
            "show me local models",
            "list downloaded models",
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt in prompts:
                with self.subTest(prompt=prompt):
                    runtime_truth.calls.clear()
                    with patch.object(
                        orchestrator,
                        "_apply_assistant_response_guard",
                        side_effect=AssertionError("post-response guard should not run"),
                    ):
                        response = orchestrator.handle_message(prompt, "user1")
                    self.assertEqual("model_status", response.data["route"])
                    self.assertFalse(response.data["used_llm"])
                    self.assertTrue(response.data.get("skip_post_response_hooks", False))
                    timing = response.data.get("orchestrator_timing_ms") if isinstance(response.data.get("orchestrator_timing_ms"), dict) else {}
                    self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))
                    payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
                    self.assertIn("truth_timing_ms", payload)
                    self.assertIn("canonical_model_inventory_snapshot_ms", payload["truth_timing_ms"])
                    self.assertIn("local installed chat models", response.text.lower())
                    self.assertIn("ollama:qwen3.5:4b", response.text.lower())
                    self.assertIn("ollama:qwen2.5:7b-instruct", response.text.lower())
                    self.assertNotIn("hugging face", response.text.lower())
                    self.assertNotIn("not installed yet", response.text.lower())
                    self.assertNotIn(("model_controller_policy_status", None), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))

    def test_recommend_local_model_uses_canonical_best_local_scout_path(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("recommend a local model", "user1")

        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["model_scout"], response.data["used_tools"])
        self.assertEqual("model_scout", payload.get("type"))
        self.assertIn("Best local option:", response.text)
        self.assertIn("Why: strongest local option currently available.", response.text)
        self.assertIn("Compared with current: upgrade within the local options.", response.text)
        self.assertIn(
            (
                "model_scout_v2_status",
                {"task_type": "chat", "requirements": ["chat"], "preferred_local": True},
            ),
            runtime_truth.calls,
        )
        self.assertNotIn("choose_best_local_chat_model", [call[0] for call in runtime_truth.calls])
        self.assertEqual(0, len(llm.chat_calls))

    def test_pack_capability_recommendation_offers_preview_without_auto_install(self) -> None:
        class _FakePackStore:
            def list_external_packs(self) -> list[dict[str, object]]:
                return []

            def list_external_pack_removals(self) -> list[dict[str, object]]:
                return []

        class _FakePackDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                return [
                    {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True},
                ]

            def search(self, source_id: str, query: str) -> dict[str, object]:
                _ = source_id
                if query != "voice":
                    return {"source": {}, "search": {"results": []}, "from_cache": False, "stale": False}
                return {
                    "source": {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True},
                    "search": {
                        "results": [
                            {
                                "remote_id": "local-voice",
                                "name": "Local Voice",
                                "summary": "Local speech output for this machine.",
                                "artifact_type_hint": "portable_text_skill",
                                "installable_by_current_policy": True,
                                "source_url": "/tmp/local-voice",
                            }
                        ]
                    },
                    "from_cache": False,
                    "stale": False,
                }

        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        orchestrator._pack_store = _FakePackStore()
        orchestrator._pack_registry_discovery = lambda: _FakePackDiscovery()  # type: ignore[assignment]

        response = orchestrator.handle_message("Talk to me out loud", "user1")
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["pack_capability_recommendation"], response.data["used_tools"])
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("pack_capability_recommendation", payload.get("type"))
        self.assertEqual("voice_output", payload.get("capability_required"))
        self.assertEqual("install_preview", payload.get("fallback"))
        self.assertIn("Voice output isn't installed.", response.text)
        self.assertIn("most practical option here", response.text)
        self.assertIn("install details", response.text)
        self.assertEqual(0, len(llm.chat_calls))

    def test_pack_capability_prompt_short_circuits_before_llm_without_frontdoor(self) -> None:
        class _FakePackStore:
            def list_external_packs(self) -> list[dict[str, object]]:
                return []

            def list_external_pack_removals(self) -> list[dict[str, object]]:
                return []

        class _FakePackDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                return []

            def search(self, source_id: str, query: str) -> dict[str, object]:
                _ = source_id
                _ = query
                return {"source": {}, "search": {"results": []}, "from_cache": False, "stale": False}

        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            chat_runtime_adapter=_RuntimeChatAvailableAdapter(),
        )
        orchestrator._pack_store = _FakePackStore()
        orchestrator._pack_registry_discovery = lambda: _FakePackDiscovery()  # type: ignore[assignment]

        response = orchestrator.handle_message("hey can you read something to me?", "user1")
        self.assertEqual("action_tool", response.data["route"])
        self.assertEqual(["capability_gap_planning"], response.data["used_tools"])
        self.assertIn("voice helper", response.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_pack_capability_recommendation_compares_two_good_options_without_auto_install(self) -> None:
        class _FakePackStore:
            def list_external_packs(self) -> list[dict[str, object]]:
                return []

            def list_external_pack_removals(self) -> list[dict[str, object]]:
                return []

        class _FakePackDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                return [
                    {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True},
                ]

            def search(self, source_id: str, query: str) -> dict[str, object]:
                _ = source_id
                if query != "voice":
                    return {"source": {}, "search": {"results": []}, "from_cache": False, "stale": False}
                return {
                    "source": {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True},
                    "search": {
                        "results": [
                            {
                                "remote_id": "local-voice",
                                "name": "Local Voice",
                                "summary": "Lightweight local speech output for this machine.",
                                "artifact_type_hint": "portable_text_skill",
                                "installable_by_current_policy": True,
                                "source_url": "/tmp/local-voice",
                                "tags": ["voice_output", "lightweight"],
                            },
                            {
                                "remote_id": "studio-voice",
                                "name": "Studio Voice",
                                "summary": "Full speech output with broader phrasing support.",
                                "artifact_type_hint": "experience_pack",
                                "installable_by_current_policy": True,
                                "source_url": "/tmp/studio-voice",
                                "tags": ["voice_output", "studio", "complete"],
                            },
                        ]
                    },
                    "from_cache": False,
                    "stale": False,
                }

        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        orchestrator._pack_store = _FakePackStore()
        orchestrator._pack_registry_discovery = lambda: _FakePackDiscovery()  # type: ignore[assignment]

        response = orchestrator.handle_message("Talk to me out loud", "user1")
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["pack_capability_recommendation"], response.data["used_tools"])
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("recommended_plus_alternate", payload.get("comparison_mode"))
        self.assertEqual("pack_capability_recommendation", payload.get("type"))
        self.assertIn("I found 2 packs that fit this machine.", response.text)
        self.assertIn("Local Voice looks like the lighter option.", response.text)
        self.assertIn("Studio Voice may need more resources.", response.text)
        self.assertIn("I'd start with Local Voice.", response.text)
        self.assertIn("install details for Local Voice", response.text)
        self.assertEqual(0, len(llm.chat_calls))

    def test_installed_but_disabled_pack_is_explained_without_auto_install(self) -> None:
        class _FakePackStore:
            def list_external_packs(self) -> list[dict[str, object]]:
                return [
                    {
                        "pack_id": "pack.voice.local_fast",
                        "name": "Local Voice",
                        "status": "normalized",
                        "enabled": False,
                        "normalized_path": str(Path(__file__).resolve()),
                        "canonical_pack": {
                            "display_name": "Local Voice",
                            "pack_identity": {"canonical_id": "pack.voice.local_fast"},
                            "source": {"source_id": "local", "source_type": "local"},
                            "capabilities": {
                                "summary": "Local speech output for this machine.",
                                "declared": ["voice_output"],
                            },
                        },
                        "review_envelope": {"pack_name": "Local Voice"},
                    }
                ]

            def list_external_pack_removals(self) -> list[dict[str, object]]:
                return []

        class _FakePackDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                return []

            def search(self, source_id: str, query: str) -> dict[str, object]:
                _ = source_id
                _ = query
                return {"source": {}, "search": {"results": []}, "from_cache": False, "stale": False}

        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        orchestrator._pack_store = _FakePackStore()
        orchestrator._pack_registry_discovery = lambda: _FakePackDiscovery()  # type: ignore[assignment]

        response = orchestrator.handle_message("Talk to me out loud", "user1")
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["pack_capability_recommendation"], response.data["used_tools"])
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("pack_capability_recommendation", payload.get("type"))
        self.assertEqual("voice_output", payload.get("capability_required"))
        self.assertIn("installed, but it is disabled", response.text)
        self.assertIn("not enabled as a live capability", response.text)
        self.assertIn("Enable it before using it.", response.text)
        self.assertIn("keep this in text", response.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_installed_and_healthy_pack_is_task_unconfirmed_without_auto_install(self) -> None:
        class _FakePackStore:
            def list_external_packs(self) -> list[dict[str, object]]:
                return [
                    {
                        "pack_id": "pack.voice.local_fast",
                        "name": "Local Voice",
                        "status": "normalized",
                        "enabled": True,
                        "normalized_path": str(Path(__file__).resolve()),
                        "canonical_pack": {
                            "display_name": "Local Voice",
                            "pack_identity": {"canonical_id": "pack.voice.local_fast"},
                            "source": {"source_id": "local", "source_type": "local"},
                            "capabilities": {
                                "summary": "Local speech output for this machine.",
                                "declared": ["voice_output"],
                            },
                        },
                        "review_envelope": {"pack_name": "Local Voice"},
                    }
                ]

            def list_external_pack_removals(self) -> list[dict[str, object]]:
                return []

        class _FakePackDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                return []

            def search(self, source_id: str, query: str) -> dict[str, object]:
                _ = source_id
                _ = query
                return {"source": {}, "search": {"results": []}, "from_cache": False, "stale": False}

        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        orchestrator._pack_store = _FakePackStore()
        orchestrator._pack_registry_discovery = lambda: _FakePackDiscovery()  # type: ignore[assignment]

        response = orchestrator.handle_message("Talk to me out loud", "user1")
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["pack_capability_recommendation"], response.data["used_tools"])
        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("pack_capability_recommendation", payload.get("type"))
        self.assertEqual("voice_output", payload.get("capability_required"))
        self.assertIn("installed and healthy", response.text)
        self.assertIn("can't confirm it's usable for this task yet", response.text)
        self.assertIn("task compatibility not confirmed", response.text)
        self.assertIn("Open the pack preview before relying on it.", response.text)
        self.assertEqual(0, len(llm.chat_calls))

    def test_paraphrased_voice_request_routes_to_existing_pack_flow(self) -> None:
        class _FakePackStore:
            def list_external_packs(self) -> list[dict[str, object]]:
                return []

            def list_external_pack_removals(self) -> list[dict[str, object]]:
                return []

        class _FakePackDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                return [
                    {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True},
                ]

            def search(self, source_id: str, query: str) -> dict[str, object]:
                _ = source_id
                if query not in {"voice", "speech", "audio", "tts"}:
                    return {"source": {}, "search": {"results": []}, "from_cache": False, "stale": False}
                return {
                    "source": {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True},
                    "search": {
                        "results": [
                            {
                                "remote_id": "local-voice",
                                "name": "Local Voice",
                                "summary": "Local speech output for this machine.",
                                "artifact_type_hint": "portable_text_skill",
                                "installable_by_current_policy": True,
                                "source_url": "/tmp/local-voice",
                            }
                        ]
                    },
                    "from_cache": False,
                    "stale": False,
                }

        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        orchestrator._pack_store = _FakePackStore()
        orchestrator._pack_registry_discovery = lambda: _FakePackDiscovery()  # type: ignore[assignment]

        response = orchestrator.handle_message("Can you read this page back to me in speech?", "user1")
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["pack_capability_recommendation"], response.data["used_tools"])
        self.assertIn("Voice output isn't installed.", response.text)
        self.assertIn("Local Voice", response.text)
        self.assertIn("install details", response.text.lower())
        self.assertNotIn("best fit for this machine", response.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_partial_custom_gap_proposes_a_small_helper(self) -> None:
        class _FakePackStore:
            def list_external_packs(self) -> list[dict[str, object]]:
                return []

            def list_external_pack_removals(self) -> list[dict[str, object]]:
                return []

        class _EmptyDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                return []

            def search(self, source_id: str, query: str) -> dict[str, object]:
                _ = source_id
                _ = query
                return {"source": {}, "search": {"results": []}, "from_cache": False, "stale": False}

        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        orchestrator._pack_store = _FakePackStore()
        orchestrator._pack_registry_discovery = lambda: _EmptyDiscovery()  # type: ignore[assignment]

        prompt = "Can you help me sketch an assistant that coordinates my studio light cues with my music cues during live shows?"
        response = orchestrator.handle_message(prompt, "user1")
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["capability_gap_planning"], response.data["used_tools"])
        self.assertIn("I can help in text", response.text)
        self.assertIn("small helper", response.text.lower())
        self.assertIn("simplest way to add it", response.text.lower())
        self.assertIn("sketch that with you", response.text.lower())
        self.assertNotIn("helper isn't installed", response.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_direct_custom_gap_proposes_a_narrow_new_helper(self) -> None:
        class _FakePackStore:
            def list_external_packs(self) -> list[dict[str, object]]:
                return []

            def list_external_pack_removals(self) -> list[dict[str, object]]:
                return []

        class _EmptyDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                return []

            def search(self, source_id: str, query: str) -> dict[str, object]:
                _ = source_id
                _ = query
                return {"source": {}, "search": {"results": []}, "from_cache": False, "stale": False}

        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        orchestrator._pack_store = _FakePackStore()
        orchestrator._pack_registry_discovery = lambda: _EmptyDiscovery()  # type: ignore[assignment]

        prompt = "Make an assistant that coordinates my studio light cues with my music cues during live shows."
        response = orchestrator.handle_message(prompt, "user1")
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["capability_gap_planning"], response.data["used_tools"])
        self.assertNotIn("I can help with the text side", response.text)
        self.assertIn("couldn't find a ready-made helper", response.text.lower())
        self.assertIn("simplest way to add it", response.text.lower())
        self.assertIn("sketch that with you", response.text.lower())
        self.assertNotIn("make an assistant that", response.text.lower())
        self.assertNotIn("install preview", response.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_capability_gap_yes_followup_advances_sketch_flow(self) -> None:
        class _FakePackStore:
            def list_external_packs(self) -> list[dict[str, object]]:
                return []

            def list_external_pack_removals(self) -> list[dict[str, object]]:
                return []

        class _EmptyDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                return []

            def search(self, source_id: str, query: str) -> dict[str, object]:
                _ = source_id
                _ = query
                return {"source": {}, "search": {"results": []}, "from_cache": False, "stale": False}

        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        orchestrator._pack_store = _FakePackStore()
        orchestrator._pack_registry_discovery = lambda: _EmptyDiscovery()  # type: ignore[assignment]

        first = orchestrator.handle_message("Make an assistant that coordinates my studio light cues with my music cues during live shows.", "user1")
        self.assertEqual("action_tool", first.data["route"])
        self.assertEqual(["capability_gap_planning"], first.data["used_tools"])
        self.assertIn("small helper", first.text.lower())

        second = orchestrator.handle_message("yes please", "user1")
        self.assertEqual("action_tool", second.data["route"])
        self.assertEqual(["capability_gap_planning"], second.data["used_tools"])
        self.assertIn("sketch a small", second.text.lower())
        self.assertIn("what should it read from first", second.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_discovery_unavailable_still_proposes_a_narrow_helper(self) -> None:
        class _FakePackStore:
            def list_external_packs(self) -> list[dict[str, object]]:
                return []

            def list_external_pack_removals(self) -> list[dict[str, object]]:
                return []

        class _BrokenDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                raise RuntimeError("temporary discovery failure")

        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        orchestrator._pack_store = _FakePackStore()
        orchestrator._pack_registry_discovery = lambda: _BrokenDiscovery()  # type: ignore[assignment]

        response = orchestrator.handle_message("Can you read this page back to me in speech?", "user1")
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["pack_capability_recommendation"], response.data["used_tools"])
        self.assertIn("couldn't check", response.text.lower())
        self.assertIn("voice helper", response.text.lower())
        self.assertIn("reads text aloud", response.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_task_model_recommendation_questions_use_model_scout_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        prompts = (
            ("recommend a coding model", "Best coding option", {"task_type": "coding", "requirements": ["chat"], "preferred_local": True}),
            ("recommend a research model", "Best research option", {"task_type": "reasoning", "requirements": ["chat", "long_context"], "preferred_local": True}),
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt, heading, task_request in prompts:
                with self.subTest(prompt=prompt):
                    runtime_truth.calls.clear()
                    response = orchestrator.handle_message(prompt, "user1")
                    payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
                    self.assertEqual("action_tool", response.data["route"])
                    self.assertFalse(response.data["used_llm"])
                    self.assertEqual(["model_scout"], response.data["used_tools"])
                    self.assertEqual("model_scout", payload.get("type"))
                    self.assertIn(heading, response.text)
                    self.assertNotIn("I couldn't read that from the runtime state.", response.text)
                    self.assertIn(("model_scout_v2_status", task_request), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))

    def test_should_i_switch_models_uses_controller_advisory_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("should I switch models", "user1")

        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["model_controller"], response.data["used_tools"])
        self.assertEqual("model_controller", payload.get("type"))
        self.assertIn("I would keep", response.text)
        self.assertIn("No change has been made.", response.text)
        self.assertTrue(
            any(
                name == "model_policy_candidate"
                and isinstance(payload, dict)
                and payload.get("tier") is None
                for name, payload in runtime_truth.calls
            )
        )
        self.assertTrue(
            any(name == "model_controller_policy_status" for name, _payload in runtime_truth.calls)
        )
        self.assertEqual(0, len(llm.chat_calls))

    def test_cloud_model_inventory_questions_are_cloud_scoped_and_grounded(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.openrouter_configured = True
        runtime_truth.openrouter_secret_present = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        prompts = (
            "show cloud models",
            "what cloud models are available",
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt in prompts:
                with self.subTest(prompt=prompt):
                    runtime_truth.calls.clear()
                    response = orchestrator.handle_message(prompt, "user1")
                    self.assertEqual("model_status", response.data["route"])
                    self.assertFalse(response.data["used_llm"])
                    self.assertIn("cloud models", response.text.lower())
                    self.assertIn("openrouter:", response.text.lower())
                    self.assertNotIn("ollama:", response.text.lower())
                    self.assertIn(("model_inventory_status", None), runtime_truth.calls)
                    self.assertIn(("model_readiness_status", None), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))

    def test_grounded_model_query_never_leaks_raw_cloud_identity(self) -> None:
        llm = _FakeChatLLM(
            enabled=True,
            text="I am running in an environment managed by Alibaba Cloud and I do not have real-time access.",
        )
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        inventory_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "installed_local": True,
                    "active": True,
                },
                {
                    "model_id": "ollama:qwen2.5:7b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "installed_local": True,
                    "active": False,
                },
            ],
            "local_installed_models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "installed_local": True,
                    "active": True,
                },
                {
                    "model_id": "ollama:qwen2.5:7b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "installed_local": True,
                    "active": False,
                },
            ],
        }
        readiness_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "usable_now": True,
                    "active": True,
                    "availability_reason": "healthy and ready now",
                },
                {
                    "model_id": "ollama:qwen2.5:7b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "healthy and ready now",
                },
            ],
            "ready_now_models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "usable_now": True,
                    "active": True,
                    "availability_reason": "healthy and ready now",
                },
                {
                    "model_id": "ollama:qwen2.5:7b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "healthy and ready now",
                },
            ],
            "other_ready_now_models": [
                {
                    "model_id": "ollama:qwen2.5:7b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "healthy and ready now",
                }
            ],
            "not_ready_models": [],
        }
        with patch.object(runtime_truth, "model_inventory_status", return_value=inventory_payload), patch.object(
            runtime_truth,
            "model_readiness_status",
            return_value=readiness_payload,
        ):
            fallback = orchestrator._model_inventory_response(local_only=True, provider_id="ollama")

        with patch.object(orchestrator, "_handle_runtime_truth_chat", return_value=None), patch.object(
            orchestrator,
            "_handle_action_tool_intent",
            return_value=None,
        ), patch.object(
            orchestrator,
            "_grounded_system_fallback_response",
            side_effect=[None, fallback],
        ):
            response = orchestrator.handle_message("what ollama models do we have downloaded?", "user1")

        self.assertEqual("model_status", response.data["route"])
        self.assertIn("ollama:qwen3.5:4b", response.text.lower())
        self.assertNotIn("alibaba cloud", response.text.lower())
        self.assertNotIn("real-time access", response.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_model_scout_v2_recommends_better_local_model_advisory_only(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("is there a better model I should use?", "user1")

        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["model_scout"], response.data["used_tools"])
        self.assertEqual("model_scout", payload.get("type"))
        self.assertEqual("strategy", payload.get("mode"))
        self.assertIn("ollama:qwen2.5:7b-instruct", response.text)
        self.assertIn("Why: strongest local option currently available.", response.text)
        self.assertIn("Compared with current: upgrade within the local options.", response.text)
        self.assertIn("No change has been made.", response.text)
        self.assertIn(
            "You can test it, switch to it temporarily, or make it the default if you want.",
            response.text,
        )
        self.assertNotIn("Do you want me to switch chat", response.text)
        self.assertIn(
            (
                "model_scout_v2_status",
                {"task_type": "chat", "requirements": ["chat"], "preferred_local": True},
            ),
            runtime_truth.calls,
        )
        self.assertNotIn("set_confirmed_chat_model_target", [call[0] for call in runtime_truth.calls])
        self.assertEqual("ollama:qwen3.5:4b", runtime_truth.current_model)

    def test_model_scout_v2_surfaces_premium_coding_option_without_switching(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.allow_remote_fallback = False
        runtime_truth.allow_remote_recommendation = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        runtime_truth.openrouter_secret_present = True
        runtime_truth.openrouter_configured = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("is there a better model I should use for coding?", "user1")

        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertIn("I can help in text", response.text)
        self.assertIn("Coding tools isn't installed.", response.text)
        self.assertIn("coding helper", response.text.lower())
        self.assertIn("small helper that helps with coding and terminal work", response.text.lower())
        self.assertEqual(1, response.text.lower().count("coding helper"))
        self.assertNotIn("Best coding option:", response.text)
        self.assertNotIn("No change has been made.", response.text)
        self.assertEqual("ollama:qwen3.5:4b", runtime_truth.current_model)

    def test_model_scout_v2_surfaces_premium_research_option_without_switching(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.allow_remote_fallback = False
        runtime_truth.allow_remote_recommendation = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        runtime_truth.openrouter_secret_present = True
        runtime_truth.openrouter_configured = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what better model should I use for research?", "user1")

        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertIn(f"Best research option: {runtime_truth.openrouter_premium_model}.", response.text)
        self.assertIn("Why: best available research option currently visible.", response.text)
        self.assertIn("Compared with current: upgrade for research quality and context.", response.text)
        self.assertIn("Best local option (fast, no cost): ollama:qwen2.5:7b-instruct.", response.text)
        self.assertIn(f"Cheap cloud option: {runtime_truth.openrouter_cheap_model}.", response.text)
        self.assertNotIn(f"Premium research option: {runtime_truth.openrouter_premium_model}.", response.text)
        self.assertIn("No change has been made.", response.text)
        self.assertIn("Controlled Mode", response.text)
        self.assertIn(
            "You can test it, switch to it temporarily, or make it the default if you want.",
            response.text,
        )
        self.assertEqual("ollama:qwen3.5:4b", runtime_truth.current_model)
        self.assertNotIn("set_confirmed_chat_model_target", [call[0] for call in runtime_truth.calls])

    def test_model_scout_premium_role_phrases_stay_grounded_and_advisory(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.allow_remote_fallback = False
        runtime_truth.allow_remote_recommendation = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        runtime_truth.openrouter_secret_present = True
        runtime_truth.openrouter_configured = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        prompts = (
            (
                "what premium coding model should I use?",
                "Premium coding recommendation:",
                "Why: qualifies for the premium coding tier.",
            ),
            (
                "what premium model should I use for research?",
                "Premium research recommendation:",
                "Why: meets the premium quality and large-context requirements for research.",
            ),
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt, expected_premium_line, expected_why_line in prompts:
                with self.subTest(prompt=prompt):
                    response = orchestrator.handle_message(prompt, "user1")

                    self.assertEqual("action_tool", response.data["route"])
                    self.assertFalse(response.data["used_llm"])
                    self.assertIn(expected_premium_line, response.text)
                    self.assertIn(expected_why_line, response.text)
                    self.assertIn("Compared with current:", response.text)
                    self.assertIn("No change has been made.", response.text)
                    self.assertIn("Controlled Mode", response.text)
                    self.assertNotIn("trouble reading the current runtime state", response.text.lower())
                    self.assertNotIn("set_confirmed_chat_model_target", [call[0] for call in runtime_truth.calls])

    def test_premium_research_prompt_reports_none_when_no_premium_candidate_qualifies(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.allow_remote_fallback = False
        runtime_truth.allow_remote_recommendation = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        runtime_truth.openrouter_secret_present = True
        runtime_truth.openrouter_configured = True
        runtime_truth.openrouter_cheap_quality_rank = 8
        runtime_truth.openrouter_premium_quality_rank = 6
        runtime_truth.openrouter_premium_context_window = 65536
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what premium model should I use for research?", "user1")

        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertIn("Premium research recommendation: none currently qualifies.", response.text)
        self.assertIn(
            "Reason: no remote model currently meets the required premium quality and context thresholds.",
            response.text,
        )
        self.assertNotIn("Why:", response.text)
        self.assertNotIn("Compared with current:", response.text)
        self.assertIn(f"Cheap cloud option: {runtime_truth.openrouter_cheap_model}.", response.text)
        self.assertNotIn(
            f"Premium research recommendation: {runtime_truth.openrouter_cheap_model}",
            response.text,
        )
        self.assertIn("No change has been made.", response.text)
        self.assertNotIn("set_confirmed_chat_model_target", [call[0] for call in runtime_truth.calls])

    def test_premium_coding_prompt_reports_none_when_no_premium_candidate_qualifies(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.allow_remote_fallback = False
        runtime_truth.allow_remote_recommendation = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        runtime_truth.openrouter_secret_present = True
        runtime_truth.openrouter_configured = True
        runtime_truth.openrouter_cheap_quality_rank = 8
        runtime_truth.openrouter_premium_quality_rank = 6
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what premium coding model should I use?", "user1")

        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertIn("Premium coding recommendation: none currently qualifies.", response.text)
        self.assertIn(
            "Reason: no remote model currently meets the required premium quality threshold.",
            response.text,
        )
        self.assertNotIn("Why:", response.text)
        self.assertNotIn("Compared with current:", response.text)
        self.assertIn(f"Cheap cloud option: {runtime_truth.openrouter_cheap_model}.", response.text)
        self.assertNotIn(
            f"Premium coding recommendation: {runtime_truth.openrouter_cheap_model}",
            response.text,
        )
        self.assertIn("No change has been made.", response.text)
        self.assertNotIn("set_confirmed_chat_model_target", [call[0] for call in runtime_truth.calls])

    def test_generic_better_model_prompt_can_still_use_best_available_overall_when_premium_is_unavailable(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.allow_remote_fallback = False
        runtime_truth.allow_remote_recommendation = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        runtime_truth.openrouter_secret_present = True
        runtime_truth.openrouter_configured = True
        runtime_truth.openrouter_cheap_quality_rank = 8
        runtime_truth.openrouter_premium_quality_rank = 6
        runtime_truth.openrouter_premium_context_window = 65536
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what better model should I use for research?", "user1")

        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertIn("Best research option:", response.text)
        self.assertIn("Why: best available research option currently visible.", response.text)
        self.assertIn("Compared with current:", response.text)
        self.assertNotIn("Premium research recommendation: none currently qualifies.", response.text)
        self.assertIn(f"Cheap cloud option: {runtime_truth.openrouter_cheap_model}.", response.text)
        self.assertIn("No change has been made.", response.text)
        self.assertNotIn("set_confirmed_chat_model_target", [call[0] for call in runtime_truth.calls])

    def test_model_scout_cheap_cloud_prompts_stay_grounded_and_advisory_in_controlled_mode(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.allow_remote_fallback = False
        runtime_truth.allow_remote_recommendation = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        runtime_truth.openrouter_secret_present = True
        runtime_truth.openrouter_configured = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        prompts = (
            "what cheap cloud model should I use?",
            "what low-cost cloud model should I use for coding?",
            "what budget remote model should I use?",
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt in prompts:
                with self.subTest(prompt=prompt):
                    response = orchestrator.handle_message(prompt, "user1")
                    self.assertEqual("action_tool", response.data["route"])
                    self.assertFalse(response.data["used_llm"])
                    self.assertIn("Best local option (fast, no cost):", response.text)
                    self.assertIn("Cheap cloud recommendation:", response.text)
                    self.assertIn("Why: lower-cost remote option for general use.", response.text)
                    self.assertIn("Compared with current: alternative option, not a clear overall upgrade.", response.text)
                    self.assertNotIn("Cheap cloud option:", response.text)
                    self.assertIn("Controlled Mode", response.text)
                    self.assertIn("No change has been made.", response.text)
                    self.assertNotIn("curl ", response.text.lower())
                    self.assertNotIn("ollama pull", response.text.lower())
                    self.assertEqual("ollama:qwen3.5:4b", runtime_truth.current_model)
                    self.assertNotIn("set_confirmed_chat_model_target", [call[0] for call in runtime_truth.calls])

    def test_cheap_cloud_prompt_is_recontained_to_model_scout_before_generic_chat(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.allow_remote_fallback = False
        runtime_truth.allow_remote_recommendation = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        runtime_truth.openrouter_secret_present = True
        runtime_truth.openrouter_configured = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        orchestrator._chat_runtime_adapter = _FrontdoorRuntimeAdapter()

        with patch.object(orchestrator, "_handle_runtime_truth_chat", return_value=None), patch.object(
            orchestrator,
            "_handle_action_tool_intent",
            return_value=None,
        ), patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what cheap cloud model should I use?", "user1")

        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("model_status", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual([], response.data["used_tools"])
        self.assertEqual("model_availability", payload.get("type"))
        self.assertEqual("remote", payload.get("inventory_scope"))
        self.assertIn("Cloud models available to use now:", response.text)

    def test_model_controller_policy_response_explains_controlled_mode_contract(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.allow_remote_fallback = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        runtime_truth.openrouter_secret_present = True
        runtime_truth.openrouter_configured = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what mode am i in?", "user1")

        self.assertEqual("model_policy_status", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertIn("Controlled Mode", response.text)
        self.assertIn("Status: Controlled Mode is the current baseline.", response.text)
        self.assertIn("I will not switch or install on my own", response.text)
        self.assertIn("You can return to SAFE MODE at any time.", response.text)
        self.assertIn("explicit approval", response.text.lower())

    def test_parse_control_mode_intent_maps_imperative_variants(self) -> None:
        cases = {
            "go into controlled mode": {"kind": "set_mode", "mode": "controlled"},
            "put us in controlled mode": {"kind": "set_mode", "mode": "controlled"},
            "enable controlled mode": {"kind": "set_mode", "mode": "controlled"},
            "exit controlled mode": {"kind": "set_mode", "mode": "baseline"},
            "return to baseline mode": {"kind": "set_mode", "mode": "baseline"},
            "enter safe mode": {"kind": "set_mode", "mode": "safe"},
        }

        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(expected, Orchestrator._parse_control_mode_intent(prompt))

    def test_parse_control_mode_intent_maps_informational_prompts_to_get_mode(self) -> None:
        prompts = (
            "what is controlled mode",
            "should i use controlled mode",
            "explain safe mode",
            "tell me about baseline mode",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual({"kind": "get_mode"}, Orchestrator._parse_control_mode_intent(prompt))

    def test_parse_control_mode_intent_ignores_model_recommendation_prompts(self) -> None:
        prompts = (
            "what model are you using?",
            "what models do we have downloaded?",
            "what local models are available?",
            "what better model should I use for research?",
            "what cheap cloud model should I use?",
            "what premium coding model should I use?",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIsNone(Orchestrator._parse_control_mode_intent(prompt))

    def test_switch_to_controlled_mode_executes_control_action_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        adapter = _ControlModeRuntimeAdapter(runtime_truth)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        orchestrator._chat_runtime_adapter = adapter

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("switch to controlled mode", "user1")

        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["model_controller"], response.data["used_tools"])
        self.assertEqual(
            [{"mode": "controlled", "confirm": True, "actor": "assistant"}],
            adapter.control_mode_calls,
        )
        self.assertFalse(runtime_truth.safe_mode)
        self.assertEqual("explicit_override", runtime_truth.mode_source)
        self.assertEqual("model_controller_policy", payload.get("type"))
        self.assertEqual("control_mode_set", payload.get("action"))
        self.assertIn("Mode: Controlled Mode.", response.text)
        self.assertIn(
            "Status: Controlled Mode is active because you explicitly turned it on.",
            response.text,
        )

    def test_natural_controlled_mode_variant_executes_control_action_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        adapter = _ControlModeRuntimeAdapter(runtime_truth)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        orchestrator._chat_runtime_adapter = adapter

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("go into controlled mode", "user1")

        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["model_controller"], response.data["used_tools"])
        self.assertEqual(
            [{"mode": "controlled", "confirm": True, "actor": "assistant"}],
            adapter.control_mode_calls,
        )
        self.assertIn("Mode: Controlled Mode.", response.text)

    def test_return_to_safe_mode_executes_control_action_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.mode_source = "explicit_override"
        runtime_truth.allow_remote_fallback = True
        runtime_truth.allow_remote_recommendation = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        adapter = _ControlModeRuntimeAdapter(runtime_truth)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        orchestrator._chat_runtime_adapter = adapter

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("return to safe mode", "user1")

        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["model_controller"], response.data["used_tools"])
        self.assertEqual(
            [{"mode": "safe", "confirm": True, "actor": "assistant"}],
            adapter.control_mode_calls,
        )
        self.assertTrue(runtime_truth.safe_mode)
        self.assertEqual("explicit_override", runtime_truth.mode_source)
        self.assertEqual("control_mode_set", payload.get("action"))
        self.assertIn("Mode: SAFE MODE.", response.text)
        self.assertIn(
            "Status: SAFE MODE is active because you explicitly turned it on.",
            response.text,
        )
        self.assertIn("Blocked: remote switching and install/download/import.", response.text)

    def test_informational_control_mode_prompt_does_not_execute_action(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        adapter = _ControlModeRuntimeAdapter(runtime_truth)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        orchestrator._chat_runtime_adapter = adapter

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("tell me about controlled mode", "user1")

        self.assertEqual("model_policy_status", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual([], adapter.control_mode_calls)
        self.assertTrue(runtime_truth.safe_mode)
        self.assertIn("Mode: SAFE MODE.", response.text)

    def test_ambiguous_control_mode_phrase_does_not_trigger_mode_change(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        adapter = _ControlModeRuntimeAdapter(runtime_truth)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        orchestrator._chat_runtime_adapter = adapter

        response = orchestrator._handle_action_tool_intent("user1", "controlled mode sounds useful")

        self.assertIsNone(response)
        self.assertEqual([], adapter.control_mode_calls)
        self.assertTrue(runtime_truth.safe_mode)

    def test_controlled_mode_explicit_acquisition_request_uses_model_manager_path(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.allow_remote_fallback = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        runtime_truth.additional_available_models.append(
            {
                "model_id": "ollama:qwen2.5:14b",
                "provider_id": "ollama",
                "local": True,
                "available": False,
                "enabled": True,
                "installed_local": False,
                "quality_rank": 10,
            }
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            preview = orchestrator.handle_message("install ollama:qwen2.5:14b", "user1")
            self.assertNotIn(
                (
                    "acquire_chat_model_target",
                    {"model_id": "ollama:qwen2.5:14b", "provider_id": "ollama"},
                ),
                runtime_truth.calls,
            )
            confirm = orchestrator.handle_message("yes", "user1")

        preview_payload = preview.data.get("runtime_payload") if isinstance(preview.data.get("runtime_payload"), dict) else {}
        confirm_payload = confirm.data.get("runtime_payload") if isinstance(confirm.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", preview.data["route"])
        self.assertFalse(preview.data["used_llm"])
        self.assertEqual(["model_manager"], preview.data["used_tools"])
        self.assertTrue(preview_payload.get("requires_confirmation"))
        self.assertIn("acquire ollama:qwen2.5:14b", preview.text.lower())
        self.assertEqual("action_tool", confirm.data["route"])
        self.assertFalse(confirm.data["used_llm"])
        self.assertEqual(["model_manager"], confirm.data["used_tools"])
        self.assertEqual("model_acquisition", confirm_payload.get("type"))
        self.assertIn("Started acquiring ollama:qwen2.5:14b", confirm.text)
        self.assertIn(
            (
                "acquire_chat_model_target",
                {"model_id": "ollama:qwen2.5:14b", "provider_id": "ollama"},
            ),
            runtime_truth.calls,
        )
        self.assertNotIn("set_confirmed_chat_model_target", [call[0] for call in runtime_truth.calls])

    def test_model_controller_trial_switch_and_switch_back_restore_previous_target(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            first = orchestrator.handle_message("try a better model", "user1")
            second = orchestrator.handle_message("switch temporarily", "user1")
            self.assertNotIn(
                (
                    "set_temporary_chat_model_target",
                    {"model_id": "ollama:qwen2.5:7b-instruct", "provider_id": "ollama"},
                ),
                runtime_truth.calls,
            )
            second_confirm = orchestrator.handle_message("yes", "user1")
            third = orchestrator.handle_message("switch back", "user1")
            self.assertNotIn(
                (
                    "restore_temporary_chat_model_target",
                    {"model_id": "ollama:qwen3.5:4b", "provider_id": "ollama"},
                ),
                runtime_truth.calls,
            )
            third_confirm = orchestrator.handle_message("yes", "user1")

        second_payload = second.data.get("runtime_payload") if isinstance(second.data.get("runtime_payload"), dict) else {}
        third_payload = third.data.get("runtime_payload") if isinstance(third.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", first.data["route"])
        self.assertIn("ollama:qwen2.5:7b-instruct", first.text)
        self.assertEqual("model_status", second.data["route"])
        self.assertFalse(second.data["used_llm"])
        self.assertEqual(["model_controller"], second.data["used_tools"])
        self.assertTrue(second_payload.get("requires_confirmation"))
        self.assertIn("switch chat temporarily", second.text.lower())
        self.assertEqual("model_status", second_confirm.data["route"])
        self.assertFalse(second_confirm.data["used_llm"])
        self.assertEqual(["model_controller"], second_confirm.data["used_tools"])
        self.assertEqual("Temporarily using ollama:qwen2.5:7b-instruct for chat.", second_confirm.text)
        self.assertEqual("model_status", third.data["route"])
        self.assertFalse(third.data["used_llm"])
        self.assertEqual(["model_controller"], third.data["used_tools"])
        self.assertTrue(third_payload.get("requires_confirmation"))
        self.assertIn("switch chat back", third.text.lower())
        self.assertEqual("model_status", third_confirm.data["route"])
        self.assertFalse(third_confirm.data["used_llm"])
        self.assertEqual(["model_controller"], third_confirm.data["used_tools"])
        self.assertEqual("Now using ollama:qwen3.5:4b for chat.", third_confirm.text)
        self.assertIn(
            (
                "set_temporary_chat_model_target",
                {"model_id": "ollama:qwen2.5:7b-instruct", "provider_id": "ollama"},
            ),
            runtime_truth.calls,
        )
        self.assertIn(
            (
                "restore_temporary_chat_model_target",
                {"model_id": "ollama:qwen3.5:4b", "provider_id": "ollama"},
            ),
            runtime_truth.calls,
        )

    def test_model_readiness_query_uses_deterministic_readiness_surface(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.openrouter_configured = True
        runtime_truth.openrouter_secret_present = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("what models are ready now?", "user1")

        self.assertEqual("model_status", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertIn(("model_readiness_status", None), runtime_truth.calls)
        self.assertIn("ready to use now", response.text.lower())
        self.assertIn("ollama:qwen2.5:7b-instruct", response.text.lower())

    def test_model_controller_can_test_context_model_without_adopting_it(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            first = orchestrator.handle_message("what better local models could I try?", "user1")
            second = orchestrator.handle_message("test this model without adopting it", "user1")

        self.assertEqual("action_tool", first.data["route"])
        self.assertEqual("action_tool", second.data["route"])
        self.assertIn("without switching", second.text.lower())
        self.assertIn(
            (
                "test_chat_model_target",
                {"model_id": "ollama:qwen2.5:7b-instruct", "provider_id": "ollama"},
            ),
            runtime_truth.calls,
        )
        self.assertEqual("ollama:qwen3.5:4b", runtime_truth.current_model)

    def test_make_this_the_default_uses_contextual_model_target(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            first = orchestrator.handle_message("what better local models could I try?", "user1")
            second = orchestrator.handle_message("make this the default", "user1")
            self.assertNotIn(("set_default_chat_model", "ollama:qwen2.5:7b-instruct"), runtime_truth.calls)
            confirm = orchestrator.handle_message("yes", "user1")

        second_payload = second.data.get("runtime_payload") if isinstance(second.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", first.data["route"])
        self.assertEqual("model_status", second.data["route"])
        self.assertFalse(second.data["used_llm"])
        self.assertEqual(["model_controller"], second.data["used_tools"])
        self.assertTrue(second_payload.get("requires_confirmation"))
        self.assertIn("make ollama:qwen2.5:7b-instruct the default chat model", second.text.lower())
        self.assertFalse(confirm.data["used_llm"])
        self.assertEqual(["model_controller"], confirm.data["used_tools"])
        self.assertEqual("ollama:qwen2.5:7b-instruct is now the default chat model, and chat is now using it.", confirm.text)
        self.assertIn(
            (
                "set_default_chat_model",
                "ollama:qwen2.5:7b-instruct",
            ),
            runtime_truth.calls,
        )

    def test_controlled_mode_make_default_uses_confirmed_target_setter(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.safe_mode = False
        runtime_truth.allow_remote_fallback = True
        runtime_truth.allow_remote_switch = True
        runtime_truth.allow_install_pull = True
        runtime_truth.scout_advisory_only = False
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch.object(
            runtime_truth,
            "set_default_chat_model",
            side_effect=AssertionError("controlled make default should use confirmed target setter"),
        ), patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            preview = orchestrator.handle_message("make ollama:qwen2.5:7b-instruct the default", "user1")
            confirm = orchestrator.handle_message("yes", "user1")

        preview_payload = preview.data.get("runtime_payload") if isinstance(preview.data.get("runtime_payload"), dict) else {}
        self.assertEqual("model_status", preview.data["route"])
        self.assertFalse(preview.data["used_llm"])
        self.assertEqual(["model_controller"], preview.data["used_tools"])
        self.assertTrue(preview_payload.get("requires_confirmation"))
        self.assertIn("make ollama:qwen2.5:7b-instruct the default chat model", preview.text.lower())
        self.assertFalse(confirm.data["used_llm"])
        self.assertEqual(["model_controller"], confirm.data["used_tools"])
        self.assertIn("ollama:qwen2.5:7b-instruct is now the default chat model", confirm.text)
        self.assertIn(
            (
                "set_confirmed_chat_model_target",
                {"model_id": "ollama:qwen2.5:7b-instruct", "provider_id": "ollama"},
            ),
            runtime_truth.calls,
        )

    def test_explicit_controller_test_phrase_uses_deterministic_test_path(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("test ollama:qwen2.5:7b-instruct without adopting it", "user1")

        self.assertEqual("action_tool", response.data["route"])
        self.assertIn("I tested ollama:qwen2.5:7b-instruct without switching.", response.text)
        self.assertIn("responded successfully", response.text)
        self.assertNotEqual("Hello!", response.text)
        self.assertIn(
            (
                "test_chat_model_target",
                {"model_id": "ollama:qwen2.5:7b-instruct", "provider_id": "ollama"},
            ),
            runtime_truth.calls,
        )
        self.assertEqual("ollama:qwen3.5:4b", runtime_truth.current_model)
        self.assertEqual("ollama:qwen3.5:4b", runtime_truth.default_model)

    def test_explicit_temporary_switch_does_not_mutate_default(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.additional_available_models.append(
            {
                "model_id": "ollama:deepseek-r1:7b",
                "provider_id": "ollama",
                "local": True,
                "available": True,
                "usable_now": True,
                "active": False,
                "quality_rank": 7,
                "availability_reason": "ready",
            }
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            preview = orchestrator.handle_message("switch temporarily to ollama:qwen2.5:7b-instruct", "user1")
            self.assertNotIn(
                (
                    "set_temporary_chat_model_target",
                    {"model_id": "ollama:qwen2.5:7b-instruct", "provider_id": "ollama"},
                ),
                runtime_truth.calls,
            )
            confirm = orchestrator.handle_message("yes", "user1")

        preview_payload = preview.data.get("runtime_payload") if isinstance(preview.data.get("runtime_payload"), dict) else {}
        self.assertEqual("model_status", preview.data["route"])
        self.assertFalse(preview.data["used_llm"])
        self.assertEqual(["model_controller"], preview.data["used_tools"])
        self.assertTrue(preview_payload.get("requires_confirmation"))
        self.assertIn("switch chat temporarily", preview.text.lower())
        self.assertFalse(confirm.data["used_llm"])
        self.assertEqual(["model_controller"], confirm.data["used_tools"])
        self.assertEqual("Temporarily using ollama:qwen2.5:7b-instruct for chat.", confirm.text)
        self.assertIn(
            (
                "set_temporary_chat_model_target",
                {"model_id": "ollama:qwen2.5:7b-instruct", "provider_id": "ollama"},
            ),
            runtime_truth.calls,
        )
        self.assertEqual("ollama:qwen2.5:7b-instruct", runtime_truth.current_model)
        self.assertEqual("ollama:qwen3.5:4b", runtime_truth.default_model)

    def test_explicit_make_default_distinguishes_default_from_active_target(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.additional_available_models.append(
            {
                "model_id": "ollama:deepseek-r1:7b",
                "provider_id": "ollama",
                "local": True,
                "available": True,
                "usable_now": True,
                "active": False,
                "quality_rank": 7,
                "availability_reason": "ready",
            }
        )
        runtime_truth.current_model = "ollama:qwen2.5:7b-instruct"
        runtime_truth.current_provider = "ollama"
        runtime_truth.effective_model = "ollama:qwen2.5:7b-instruct"
        runtime_truth.effective_provider = "ollama"
        runtime_truth.temporary_override_active = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            preview = orchestrator.handle_message("make ollama:deepseek-r1:7b the default", "user1")
            self.assertNotIn(("set_default_chat_model", "ollama:deepseek-r1:7b"), runtime_truth.calls)
            confirm = orchestrator.handle_message("yes", "user1")

        preview_payload = preview.data.get("runtime_payload") if isinstance(preview.data.get("runtime_payload"), dict) else {}
        self.assertEqual("model_status", preview.data["route"])
        self.assertFalse(preview.data["used_llm"])
        self.assertEqual(["model_controller"], preview.data["used_tools"])
        self.assertTrue(preview_payload.get("requires_confirmation"))
        self.assertIn("make ollama:deepseek-r1:7b the default chat model", preview.text.lower())
        self.assertFalse(confirm.data["used_llm"])
        self.assertEqual(["model_controller"], confirm.data["used_tools"])
        self.assertIn("ollama:deepseek-r1:7b is now the default chat model.", confirm.text)
        self.assertIn("Chat is still using ollama:qwen2.5:7b-instruct.", confirm.text)
        self.assertIn(("set_default_chat_model", "ollama:deepseek-r1:7b"), runtime_truth.calls)
        self.assertEqual("ollama:qwen2.5:7b-instruct", runtime_truth.current_model)
        self.assertEqual("ollama:deepseek-r1:7b", runtime_truth.default_model)

    def test_switch_better_local_model_requires_confirmation_then_executes(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            preview = orchestrator.handle_message("switch to a better local model", "user1")
            self.assertNotIn(("configure_local_chat_model", "ollama:qwen3.5:8b"), runtime_truth.calls)
            confirm = orchestrator.handle_message("yes", "user1")

        preview_payload = preview.data.get("runtime_payload") if isinstance(preview.data.get("runtime_payload"), dict) else {}
        self.assertEqual("model_status", preview.data["route"])
        self.assertFalse(preview.data["used_llm"])
        self.assertEqual(["model_controller"], preview.data["used_tools"])
        self.assertTrue(preview_payload.get("requires_confirmation"))
        self.assertIn("best available local model", preview.text.lower())
        self.assertEqual("model_status", confirm.data["route"])
        self.assertFalse(confirm.data["used_llm"])
        self.assertEqual(["model_controller"], confirm.data["used_tools"])
        self.assertIn("I switched chat to your local model ollama:qwen3.5:8b.", confirm.text)
        self.assertIn(("configure_local_chat_model", "ollama:qwen3.5:8b"), runtime_truth.calls)

    def test_model_scout_hf_discovery_is_honest_about_download_candidates(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.hf_enabled = True
        runtime_truth.hf_scan_body = {
            "ok": True,
            "trigger": "manual",
            "scan": {
                "ok": True,
                "enabled": True,
                "updates": [
                    {
                        "repo_id": "nanbeige/Nanbeige2-16B-Chat-GGUF",
                        "installability": "installable_ollama",
                    }
                ],
                "discovered_count": 1,
            },
            "proposal_created": True,
            "proposal": {
                "repo_id": "nanbeige/Nanbeige2-16B-Chat-GGUF",
                "installability": "installable_ollama",
            },
        }
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message(
                "look for new better models",
                "user1",
            )

        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["model_discovery_manager"], response.data["used_tools"])
        self.assertEqual("external_discovery", payload.get("mode"))
        self.assertIn("nanbeige/nanbeige2-16b-chat-gguf", response.text.lower())
        self.assertIn("closest matches look like", response.text.lower())
        self.assertIn(("model_discovery_query", {"query": "look for new better models", "filters": {}}), runtime_truth.calls)

    def test_model_scout_discovery_routes_huggingface_smol_model_prompt_without_status_fallback(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message(
                "can you find some smol models on Hugging Face?",
                "user1",
            )

        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual(["model_discovery_manager"], response.data["used_tools"])
        self.assertEqual("external_discovery", payload.get("mode"))
        self.assertIn("huggingface", response.text.lower())
        self.assertNotIn("runtime status", response.text.lower())
        self.assertIn(("model_discovery_query", {"query": "can you find some smol models on Hugging Face?", "filters": {}}), runtime_truth.calls)

    def test_model_scout_discovery_routes_brand_new_tiny_model_prompt_without_runtime_dead_end(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.model_discovery_query = lambda query=None, filters=None: {  # type: ignore[assignment]
            "ok": True,
            "query": query,
            "message": "Found 1 model(s) across 3 source(s).",
            "models": [
                {
                    "id": "openrouter:vendor/tiny-gemma",
                    "provider": "openrouter",
                    "source": "openrouter",
                    "capabilities": ["chat"],
                    "local": False,
                    "installable": False,
                    "confidence": 0.8,
                }
            ],
            "sources": [
                {"source": "openrouter", "enabled": True, "queried": True, "ok": True, "count": 1},
            ],
            "debug": {"source_errors": {}, "source_counts": {"openrouter": 1}},
        }
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message(
                "there is a brand new tiny Gemma 4 model, can you look into it?",
                "user1",
            )

        payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
        self.assertEqual("action_tool", response.data["route"])
        self.assertEqual(["model_discovery_manager"], response.data["used_tools"])
        self.assertEqual("external_discovery", payload.get("mode"))
        self.assertIn("openrouter:vendor/tiny-gemma", response.text.lower())
        self.assertNotIn("runtime state", response.text.lower())

    def test_fuzzy_discovery_prompts_route_to_discovery_and_stay_user_facing(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()

        def _discovery_payload(query: str | None = None, filters: dict[str, object] | None = None) -> dict[str, object]:
            _ = filters
            q = str(query or "").strip().lower()
            if "gemma" in q:
                models = [
                    {
                        "id": "huggingface:google/gemma-2-2b-it",
                        "provider": "huggingface",
                        "source": "huggingface",
                        "capabilities": ["chat"],
                        "local": False,
                        "installable": False,
                        "confidence": 0.91,
                        "match_band": "likely",
                    },
                    {
                        "id": "ollama:qwen3.5:4b",
                        "provider": "ollama",
                        "source": "ollama",
                        "capabilities": ["chat"],
                        "local": True,
                        "installable": True,
                        "confidence": 0.84,
                        "match_band": "related",
                    },
                    {
                        "id": "openrouter:google/gemma-3-4b-it",
                        "provider": "openrouter",
                        "source": "openrouter",
                        "capabilities": ["chat"],
                        "local": False,
                        "installable": False,
                        "confidence": 0.8,
                        "match_band": "likely",
                    }
                ]
                message = "Grouped discovery results."
            elif "vision" in q:
                models = [
                    {
                        "id": "ollama:llava:7b",
                        "provider": "ollama",
                        "source": "ollama",
                        "capabilities": ["vision"],
                        "local": True,
                        "installable": True,
                        "confidence": 0.88,
                        "match_band": "likely",
                    },
                    {
                        "id": "ollama:qwen3.5:4b",
                        "provider": "ollama",
                        "source": "ollama",
                        "capabilities": ["chat"],
                        "local": True,
                        "installable": True,
                        "confidence": 0.83,
                        "match_band": "related",
                    },
                    {
                        "id": "huggingface:moondream2",
                        "provider": "huggingface",
                        "source": "huggingface",
                        "capabilities": ["vision"],
                        "local": False,
                        "installable": False,
                        "confidence": 0.78,
                        "match_band": "likely",
                    }
                ]
                message = "Grouped discovery results."
            elif "newer than" in q or "qwen2.5 3b" in q:
                models = [
                    {
                        "id": "ollama:qwen3.5:4b",
                        "provider": "ollama",
                        "source": "ollama",
                        "capabilities": ["chat"],
                        "local": True,
                        "installable": True,
                        "confidence": 0.84,
                        "match_band": "likely",
                    },
                    {
                        "id": "openrouter:vendor/qwen2.5-7b",
                        "provider": "openrouter",
                        "source": "openrouter",
                        "capabilities": ["chat"],
                        "local": False,
                        "installable": False,
                        "confidence": 0.86,
                        "match_band": "related",
                    },
                    {
                        "id": "openrouter:vendor/qwen2.5-3b-instruct",
                        "provider": "openrouter",
                        "source": "openrouter",
                        "capabilities": ["chat"],
                        "local": False,
                        "installable": False,
                        "confidence": 0.75,
                        "match_band": "likely",
                    }
                ]
                message = "Grouped discovery results."
            else:
                models = [
                    {
                        "id": "ollama:qwen2.5-coder:7b",
                        "provider": "ollama",
                        "source": "ollama",
                        "capabilities": ["chat"],
                        "local": True,
                        "installable": True,
                        "confidence": 0.9,
                        "match_band": "likely",
                    },
                    {
                        "id": "ollama:deepseek-r1:7b",
                        "provider": "ollama",
                        "source": "ollama",
                        "capabilities": ["chat"],
                        "local": True,
                        "installable": True,
                        "confidence": 0.82,
                        "match_band": "related",
                    },
                    {
                        "id": "openrouter:vendor/coder-pro",
                        "provider": "openrouter",
                        "source": "openrouter",
                        "capabilities": ["chat"],
                        "local": False,
                        "installable": False,
                        "confidence": 0.88,
                        "match_band": "likely",
                    }
                ]
                message = "Grouped discovery results."
            return {
                "ok": True,
                "query": query,
                "message": message,
                "models": models,
                "sources": [
                    {"source": "huggingface", "enabled": True, "queried": True, "ok": True, "count": 1},
                    {"source": "openrouter", "enabled": True, "queried": True, "ok": True, "count": 1},
                    {"source": "ollama", "enabled": True, "queried": True, "ok": True, "count": 1},
                    {"source": "external_snapshots", "enabled": True, "queried": True, "ok": True, "count": 0},
                ],
                "debug": {"ranking": {"broadening_used": True}},
            }

        runtime_truth.model_discovery_query = _discovery_payload  # type: ignore[assignment]
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        prompts = {
            "there is a brand new tiny Gemma 4 model, can you look into it?": {
                "marker": "gemma",
                "intro": "For Gemma",
            },
            "what's a small local coding model?": {
                "marker": "coding",
                "intro": "For a small local coding model",
            },
            "is there a lightweight vision model I could run locally?": {
                "marker": "vision",
                "intro": "For a lightweight local vision model",
                "preferred": "ollama:llava:7b",
            },
            "what's newer than qwen2.5 3b for chat?": {
                "marker": "qwen",
                "intro": "For a newer chat option",
                "exclude": "qwen2.5:3b-instruct",
                "preferred": "qwen3.5:4b",
            },
        }
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt, expected in prompts.items():
                with self.subTest(prompt=prompt):
                    response = orchestrator.handle_message(prompt, "user1")
                    payload = response.data.get("runtime_payload") if isinstance(response.data.get("runtime_payload"), dict) else {}
                    first_line = str(response.text or "").splitlines()[0] if str(response.text or "").strip() else ""
                    self.assertEqual("action_tool", response.data["route"])
                    self.assertEqual(["model_discovery_manager"], response.data["used_tools"])
                    self.assertEqual("external_discovery", payload.get("mode"))
                    self.assertTrue(first_line.startswith(expected["intro"]), msg=prompt)
                    self.assertNotIn("found ", first_line.lower(), msg=prompt)
                    self.assertNotIn("model(s)", first_line.lower(), msg=prompt)
                    self.assertIn(expected["marker"], response.text.lower())
                    if "exclude" in expected:
                        self.assertNotIn(expected["exclude"], first_line.lower(), msg=prompt)
                    if prompt.startswith("what's a small local coding model") or prompt.startswith("is there a lightweight vision model"):
                        self.assertNotIn("openrouter:", first_line.lower(), msg=prompt)
                    self.assertIn("Likely family match:", response.text)
                    self.assertIn("Practical local fit:", response.text)
                    if prompt.startswith("is there a lightweight vision model"):
                        self.assertIn("llava:7b", first_line.lower(), msg=prompt)
                    if prompt.startswith("what's newer than"):
                        self.assertIn("qwen3.5:4b", first_line.lower(), msg=prompt)
                        self.assertNotIn("qwen2.5:3b-instruct", response.text.lower(), msg=prompt)
                    self.assertIn("Sources checked:", response.text)
                    self.assertNotIn("no models matched", response.text.lower())
                    self.assertNotIn("chat llm is unavailable", response.text.lower())
                    self.assertNotIn("runtime state", response.text.lower())

    def test_ram_vram_prompt_renders_hardware_first_line_before_secondary_stats(self) -> None:
        orchestrator = self._orchestrator()
        with patch("agent.orchestrator.can_run_nl_skill", return_value=(True, None)), patch(
            "agent.nl_router.select_observe_skills",
            return_value=[{"skill": "hardware_report", "function": "hardware_report"}],
        ):
            orchestrator.skills["hardware_report"].functions["hardware_report"].handler = lambda ctx, user_id=None: {
                "status": "ok",
                "text": "You have 64 GiB of RAM and 12 GiB of VRAM available right now.",
                "payload": {
                    "memory": {
                        "total_bytes": 64 * 1024**3,
                        "used_bytes": 22 * 1024**3,
                        "available_bytes": 42 * 1024**3,
                        "used_pct": 34.4,
                    },
                    "gpu": {
                        "available": True,
                        "gpus": [
                            {
                                "name": "NVIDIA RTX 4080",
                                "memory_used_mb": 4096,
                                "memory_total_mb": 12288,
                                "utilization_gpu_pct": 17.0,
                                "temperature_c": 54,
                            }
                        ],
                    },
                },
                "cards_payload": {
                    "cards": [{"title": "Hardware inventory", "lines": ["RAM: 64 GiB total", "VRAM: 12 GiB available"], "severity": "ok"}],
                    "raw_available": True,
                    "summary": "You have 64 GiB of RAM and 12 GiB of VRAM available right now.",
                    "confidence": 1.0,
                    "next_questions": ["How much memory am I using?"],
                },
            }
            response = orchestrator.handle_message("what do i have for ram and vram right now?", "user1")

        self.assertEqual("operational_status", response.data["route"])
        self.assertIn("You have 64 GiB of RAM with 42 GiB available.", response.text)
        self.assertIn("VRAM is available on NVIDIA RTX 4080", response.text)
        self.assertFalse(response.text.startswith("CPU load"))

    def test_diagnostics_capture_flow_requests_confirmation_then_returns_compact_snapshot(self) -> None:
        orchestrator = self._orchestrator()
        snapshot = {
            "trace_id": "doctor-123",
            "generated_at": "2026-04-19T12:00:00+00:00",
            "os": {
                "available": True,
                "source": "uname -a",
                "text": "Linux test-host 6.8.0-1 x86_64",
                "error_kind": None,
            },
            "network": {
                "source": "nmcli",
                "nmcli_available": True,
                "system_summary": {
                    "state": "degraded",
                    "up_interfaces": ["wlp2s0"],
                    "default_route": False,
                    "dns_configured": True,
                },
                "nmcli_rows": [
                    {"device": "wlp2s0", "type": "wifi", "state": "connected", "connection": "HomeNetwork"},
                ],
            },
            "suspend_resume": {
                "source": "journalctl -b",
                "available": True,
                "match_count": 2,
                "matches": [
                    "kernel: PM: suspend entry (deep)",
                    "kernel: network disconnected after suspend",
                ],
            },
            "summary": {
                "status": "warn",
                "notes": ["Network is not fully up in the snapshot."],
                "next_action": "Check NetworkManager and the Wi-Fi driver after suspend.",
            },
        }

        with (
            patch("agent.orchestrator.collect_diagnostics_snapshot", return_value=snapshot) as collect_call,
            patch.object(orchestrator, "_llm_chat", side_effect=AssertionError("generic chat should not be used")),
        ):
            preview = orchestrator.handle_message("My Wi-Fi drops after suspend. Can you help me figure out why?", "user-1")

        self.assertEqual(0, collect_call.call_count)
        self.assertIn("Do you want me to do that?", preview.text)
        preview_data = preview.data or {}
        self.assertEqual("diagnostics_capture", preview_data.get("route"))
        preview_payload = preview_data.get("runtime_payload") if isinstance(preview_data.get("runtime_payload"), dict) else {}
        self.assertTrue(bool(preview_payload.get("requires_confirmation")))
        self.assertFalse(bool(preview_payload.get("mutating", True)))
        self.assertNotIn("uname", preview.text.lower())
        self.assertNotIn("journalctl", preview.text.lower())

        with patch("agent.orchestrator.collect_diagnostics_snapshot", return_value=snapshot) as collect_call:
            confirmed = orchestrator.handle_message("yes", "user-1")

        self.assertEqual(1, collect_call.call_count)
        self.assertEqual("diagnostics_capture", confirmed.data.get("route"))
        payload = confirmed.data.get("runtime_payload") if isinstance(confirmed.data.get("runtime_payload"), dict) else {}
        self.assertEqual("collect_diagnostics", payload.get("kind"))
        cards_payload = payload.get("cards_payload") if isinstance(payload.get("cards_payload"), dict) else {}
        self.assertEqual("Diagnostics snapshot", ((cards_payload.get("cards") or [{}])[0] or {}).get("title"))
        self.assertIn("Diagnostics snapshot", confirmed.text)
        self.assertIn("OS/kernel:", confirmed.text)
        self.assertIn("Network:", confirmed.text)
        self.assertIn("Suspend/resume matches:", confirmed.text)
        self.assertIn("Check NetworkManager and the Wi-Fi driver after suspend.", confirmed.text)
        self.assertIn("diagnostics_snapshot", payload)
        self.assertNotIn("Run:", confirmed.text)

    def test_bluetooth_diagnostics_capture_flow_requests_confirmation_then_returns_compact_snapshot(self) -> None:
        orchestrator = self._orchestrator()
        snapshot = {
            "preset": "bluetooth_audio",
            "bluetooth": {
                "service": {
                    "available": True,
                    "source": "systemctl status bluetooth",
                    "active_state": "active",
                    "loaded_state": "loaded",
                },
                "controller": {
                    "available": True,
                    "source": "bluetoothctl show",
                    "address": "AA:BB:CC:DD:EE:FF",
                    "controller": "test-host",
                    "alias": "test-host",
                    "powered": True,
                    "discoverable": False,
                    "pairable": True,
                    "discovering": False,
                },
                "devices": {
                    "source": "bluetoothctl paired-devices",
                    "available": True,
                    "paired_count": 1,
                    "connected_count": 0,
                    "devices": [
                        {
                            "address": "11:22:33:44:55:66",
                            "name": "Headphones",
                            "connected": False,
                            "paired": True,
                            "trusted": True,
                        }
                    ],
                },
                "logs": {
                    "source": "journalctl -u bluetooth",
                    "available": True,
                    "match_count": 2,
                    "matches": [
                        "bluetoothd[123]: profiles/audio: disconnect",
                        "bluetoothd[123]: reconnect failed",
                    ],
                },
            },
            "summary": {
                "status": "warn",
                "notes": ["Recent Bluetooth logs contain failure markers."],
                "next_action": "Re-test the Bluetooth device and check whether it reconnects after suspend.",
            },
        }

        with (
            patch("agent.orchestrator.collect_bluetooth_audio_diagnostics_snapshot", return_value=snapshot) as collect_call,
            patch("agent.orchestrator.collect_diagnostics_snapshot", side_effect=AssertionError("generic diagnostics should not be used")),
            patch.object(orchestrator, "_llm_chat", side_effect=AssertionError("generic chat should not be used")),
        ):
            preview = orchestrator.handle_message("My Bluetooth headphones disconnect after sleep. Can you help me figure out why?", "user-2")

        self.assertEqual(0, collect_call.call_count)
        self.assertIn("Do you want me to do that?", preview.text)
        preview_data = preview.data or {}
        self.assertEqual("diagnostics_capture", preview_data.get("route"))
        preview_payload = preview_data.get("runtime_payload") if isinstance(preview_data.get("runtime_payload"), dict) else {}
        self.assertEqual("bluetooth_audio", preview_payload.get("capture_scope"))
        self.assertEqual("collect_bluetooth_audio_diagnostics", preview_payload.get("kind"))
        self.assertNotIn("systemctl status bluetooth", preview.text.lower())

        with patch("agent.orchestrator.collect_bluetooth_audio_diagnostics_snapshot", return_value=snapshot) as collect_call:
            confirmed = orchestrator.handle_message("yes", "user-2")

        self.assertEqual(1, collect_call.call_count)
        self.assertEqual("diagnostics_capture", confirmed.data.get("route"))
        payload = confirmed.data.get("runtime_payload") if isinstance(confirmed.data.get("runtime_payload"), dict) else {}
        self.assertEqual("collect_bluetooth_audio_diagnostics", payload.get("kind"))
        cards_payload = payload.get("cards_payload") if isinstance(payload.get("cards_payload"), dict) else {}
        self.assertEqual("Bluetooth/audio diagnostics", ((cards_payload.get("cards") or [{}])[0] or {}).get("title"))
        self.assertIn("Bluetooth/audio diagnostics", confirmed.text)
        self.assertIn("Service:", confirmed.text)
        self.assertIn("Controller:", confirmed.text)
        self.assertIn("Devices:", confirmed.text)
        self.assertIn("Logs:", confirmed.text)
        self.assertIn("diagnostics_snapshot", payload)
        self.assertNotIn("journalctl", confirmed.text.lower())

    def test_storage_diagnostics_capture_flow_requests_confirmation_then_returns_compact_snapshot(self) -> None:
        orchestrator = self._orchestrator()
        snapshot = {
            "preset": "storage_disk",
            "storage": {
                "filesystems": {
                    "source": "df -hT --local --output=source,fstype,size,used,avail,pcent,target",
                    "available": True,
                    "rows": [
                        {
                            "device": "/dev/nvme0n1p2",
                            "mountpoint": "/",
                            "fstype": "ext4",
                            "size": "100G",
                            "used": "96G",
                            "avail": "4G",
                            "used_pct": 96.0,
                        },
                        {
                            "device": "/dev/nvme0n1p3",
                            "mountpoint": "/home",
                            "fstype": "ext4",
                            "size": "200G",
                            "used": "120G",
                            "avail": "80G",
                            "used_pct": 60.0,
                        },
                    ],
                },
                "consumers": {
                    "source": "du -x -B1 -d 1 <candidate paths>",
                    "available": True,
                    "match_count": 2,
                    "entries": [
                        {"path": "/home/user/Downloads", "size_bytes": 8 * 1024**3, "base_path": "/home/user"},
                        {"path": "/var/cache", "size_bytes": 4 * 1024**3, "base_path": "/var"},
                    ],
                },
                "logs": {
                    "source": "journalctl -b",
                    "available": True,
                    "match_count": 2,
                    "matches": [
                        "app[123]: write failed: No space left on device",
                        "app[123]: retry failed",
                    ],
                },
            },
            "summary": {
                "status": "critical",
                "notes": ["High filesystem usage on / (96.0%)."],
                "next_action": "Free space on the fullest filesystem and retry the save.",
            },
        }

        with (
            patch("agent.orchestrator.collect_storage_disk_diagnostics_snapshot", return_value=snapshot) as collect_call,
            patch("agent.orchestrator.collect_diagnostics_snapshot", side_effect=AssertionError("generic diagnostics should not be used")),
            patch.object(orchestrator, "_llm_chat", side_effect=AssertionError("generic chat should not be used")),
        ):
            preview = orchestrator.handle_message("My disk is full and I can't save files. Can you help me figure out why?", "user-3")

        self.assertEqual(0, collect_call.call_count)
        self.assertIn("Do you want me to do that?", preview.text)
        preview_data = preview.data or {}
        self.assertEqual("diagnostics_capture", preview_data.get("route"))
        preview_payload = preview_data.get("runtime_payload") if isinstance(preview_data.get("runtime_payload"), dict) else {}
        self.assertEqual("storage_disk", preview_payload.get("capture_scope"))
        self.assertEqual("collect_storage_disk_diagnostics", preview_payload.get("kind"))
        self.assertNotIn("df -hT", preview.text.lower())

        with patch("agent.orchestrator.collect_storage_disk_diagnostics_snapshot", return_value=snapshot) as collect_call:
            confirmed = orchestrator.handle_message("yes", "user-3")

        self.assertEqual(1, collect_call.call_count)
        self.assertEqual("diagnostics_capture", confirmed.data.get("route"))
        payload = confirmed.data.get("runtime_payload") if isinstance(confirmed.data.get("runtime_payload"), dict) else {}
        self.assertEqual("collect_storage_disk_diagnostics", payload.get("kind"))
        cards_payload = payload.get("cards_payload") if isinstance(payload.get("cards_payload"), dict) else {}
        self.assertEqual("Storage/disk diagnostics", ((cards_payload.get("cards") or [{}])[0] or {}).get("title"))
        self.assertIn("Storage/disk diagnostics", confirmed.text)
        self.assertIn("Filesystem:", confirmed.text)
        self.assertIn("Mount/device:", confirmed.text)
        self.assertIn("Consumers:", confirmed.text)
        self.assertIn("Log matches:", confirmed.text)
        self.assertIn("diagnostics_snapshot", payload)
        self.assertNotIn("journalctl", confirmed.text.lower())

    def test_printer_diagnostics_capture_flow_requests_confirmation_then_returns_compact_snapshot(self) -> None:
        orchestrator = self._orchestrator()
        snapshot = {
            "preset": "printer_cups",
            "printer": {
                "service": {
                    "available": True,
                    "source": "systemctl status cups",
                    "active_state": "active",
                    "active_detail": "active (running) since Fri 2026-04-19 12:00:00 UTC; 1min ago",
                    "loaded_state": "loaded",
                    "loaded_detail": "loaded (/usr/lib/systemd/system/cups.service; enabled; preset: enabled)",
                },
                "printers": {
                    "source": "lpstat -p -d",
                    "available": True,
                    "default_printer": "HP_LaserJet",
                    "printer_count": 2,
                    "rows": [
                        {
                            "name": "HP_LaserJet",
                            "state": "idle",
                            "enabled": True,
                            "accepting": True,
                            "details": "is idle. enabled since Fri 19 Apr 2026 12:00:00 PM UTC",
                        },
                        {
                            "name": "Brother_Office",
                            "state": "offline",
                            "enabled": False,
                            "accepting": False,
                            "details": "is offline. disabled since Fri 19 Apr 2026 11:30:00 AM UTC",
                        },
                    ],
                },
                "jobs": {
                    "source": "lpstat -o",
                    "available": True,
                    "match_count": 2,
                    "rows": [
                        {"queue": "HP_LaserJet", "job_id": "42", "owner": "user", "text": "HP_LaserJet-42 user 1234 Fri 19 Apr 2026 12:01:00 PM UTC"},
                        {"queue": "Brother_Office", "job_id": "43", "owner": "user", "text": "Brother_Office-43 user 4321 Fri 19 Apr 2026 12:02:00 PM UTC"},
                    ],
                },
                "logs": {
                    "source": "journalctl -u cups",
                    "available": True,
                    "match_count": 2,
                    "matches": [
                        "cups[123]: printer HP_LaserJet resumed",
                        "cups[123]: filter failed for job 42",
                    ],
                },
            },
            "summary": {
                "status": "warn",
                "notes": ["Print queue shows 2 job(s)."],
                "next_action": "Clear or re-submit the print queue, then retry the job.",
            },
        }

        with (
            patch("agent.orchestrator.collect_printer_cups_diagnostics_snapshot", return_value=snapshot) as collect_call,
            patch("agent.orchestrator.collect_diagnostics_snapshot", side_effect=AssertionError("generic diagnostics should not be used")),
            patch.object(orchestrator, "_llm_chat", side_effect=AssertionError("generic chat should not be used")),
        ):
            preview = orchestrator.handle_message("My printer is offline and print jobs are stuck. Can you help me figure out why?", "user-4")

        self.assertEqual(0, collect_call.call_count)
        self.assertIn("Do you want me to do that?", preview.text)
        preview_data = preview.data or {}
        self.assertEqual("diagnostics_capture", preview_data.get("route"))
        preview_payload = preview_data.get("runtime_payload") if isinstance(preview_data.get("runtime_payload"), dict) else {}
        self.assertEqual("printer_cups", preview_payload.get("capture_scope"))
        self.assertEqual("collect_printer_cups_diagnostics", preview_payload.get("kind"))
        self.assertNotIn("lpstat -p -d", preview.text.lower())

        with patch("agent.orchestrator.collect_printer_cups_diagnostics_snapshot", return_value=snapshot) as collect_call:
            confirmed = orchestrator.handle_message("yes", "user-4")

        self.assertEqual(1, collect_call.call_count)
        self.assertEqual("diagnostics_capture", confirmed.data.get("route"))
        payload = confirmed.data.get("runtime_payload") if isinstance(confirmed.data.get("runtime_payload"), dict) else {}
        self.assertEqual("collect_printer_cups_diagnostics", payload.get("kind"))
        cards_payload = payload.get("cards_payload") if isinstance(payload.get("cards_payload"), dict) else {}
        self.assertEqual("Printer/CUPS diagnostics", ((cards_payload.get("cards") or [{}])[0] or {}).get("title"))
        self.assertIn("Printer/CUPS diagnostics", confirmed.text)
        self.assertIn("Printers:", confirmed.text)
        self.assertIn("Jobs:", confirmed.text)
        self.assertIn("Log matches:", confirmed.text)
        self.assertIn("diagnostics_snapshot", payload)
        self.assertNotIn("journalctl", confirmed.text.lower())

    def test_confirmation_variants_are_accepted_for_pending_diagnostics(self) -> None:
        orchestrator = self._orchestrator()
        snapshot = {
            "preset": "printer_cups",
            "printer": {
                "service": {"available": True, "source": "systemctl status cups", "active_state": "active", "loaded_state": "loaded"},
                "printers": {"source": "lpstat -p -d", "available": True, "default_printer": "HP_LaserJet", "printer_count": 1, "rows": []},
                "jobs": {"source": "lpstat -o", "available": True, "match_count": 0, "rows": []},
                "logs": {"source": "journalctl -u cups", "available": True, "match_count": 0, "matches": []},
            },
            "summary": {"status": "warn", "notes": ["Printer snapshot ready."], "next_action": "Try printing again."},
        }

        for index, confirmation in enumerate(("yes do it", "sure go ahead", "please do it"), start=1):
            user_id = f"user-confirm-{index}"
            with patch("agent.orchestrator.collect_printer_cups_diagnostics_snapshot", return_value=snapshot):
                preview = orchestrator.handle_message("printer stuck again, jobs wont print", user_id)
                self.assertEqual("diagnostics_capture", preview.data.get("route"))
                confirmed = orchestrator.handle_message(confirmation, user_id)

            self.assertEqual("diagnostics_capture", confirmed.data.get("route"))
            self.assertIn("Printer/CUPS diagnostics", confirmed.text)
            self.assertNotIn("I’m not sure. What exactly are you referring to?", confirmed.text)

    def test_vague_system_trouble_prompts_for_clarification_instead_of_broad_observe(self) -> None:
        orchestrator = self._orchestrator()

        response = orchestrator.handle_message("my pc is acting weird", "user-vague")

        self.assertEqual("assistant_clarification", response.data.get("route"))
        self.assertIn("What symptom should I focus on", response.text)
        self.assertNotIn("Hardware inventory", response.text)

    def test_generic_device_fallback_capture_flow_requests_confirmation_then_returns_compact_snapshot(self) -> None:
        orchestrator = self._orchestrator()
        snapshot = {
            "preset": "generic_device_fallback",
            "device": {
                "os": {
                    "available": True,
                    "source": "uname -a",
                    "text": "Linux test-host 6.8.0-1 x86_64",
                    "error_kind": None,
                },
                "presence": {
                    "usb": {
                        "available": True,
                        "source": "lsusb",
                        "match_count": 2,
                        "lines": [
                            "Bus 001 Device 002: ID 046d:0825 Logitech, Inc. Webcam C270",
                            "Bus 001 Device 003: ID 0bda:58f4 Realtek Semiconductor Corp.",
                        ],
                        "error_kind": None,
                    },
                    "pci": {
                        "available": True,
                        "source": "lspci -nn",
                        "match_count": 2,
                        "lines": [
                            "00:02.0 VGA compatible controller [0300]: Intel Corporation Device [8086:46a6]",
                            "00:14.0 USB controller [0c03]: Intel Corporation Device [8086:7aa8]",
                        ],
                        "error_kind": None,
                    },
                },
                "logs": {
                    "journal": {
                        "available": True,
                        "source": "journalctl -b -p warning",
                        "match_count": 2,
                        "matches": [
                            "kernel: usb 1-1: device descriptor read/64, error -71",
                            "kernel: webcam: not detected after resume",
                        ],
                        "error_kind": None,
                    },
                    "dmesg": {
                        "available": True,
                        "source": "dmesg --ctime --level=err,warn",
                        "match_count": 2,
                        "matches": [
                            "usb 1-1: reset full-speed USB device number 2 using xhci_hcd",
                            "webcam: firmware failed to load",
                        ],
                        "error_kind": None,
                    },
                },
            },
            "summary": {
                "status": "warn",
                "notes": ["Recent system logs contain device or driver failure markers."],
                "next_action": "Retry the device action and check whether it appears after reconnecting or rebooting.",
            },
        }

        with (
            patch("agent.orchestrator.collect_generic_device_fallback_diagnostics_snapshot", return_value=snapshot) as collect_call,
            patch("agent.orchestrator.collect_diagnostics_snapshot", side_effect=AssertionError("generic diagnostics should not be used")),
            patch.object(orchestrator, "_llm_chat", side_effect=AssertionError("generic chat should not be used")),
        ):
            preview = orchestrator.handle_message("My webcam isn't detected after sleep. Can you help me figure out why?", "user-4")

        self.assertEqual(0, collect_call.call_count)
        self.assertIn("Want me to do that?", preview.text)
        preview_data = preview.data or {}
        self.assertEqual("diagnostics_capture", preview_data.get("route"))
        preview_payload = preview_data.get("runtime_payload") if isinstance(preview_data.get("runtime_payload"), dict) else {}
        self.assertEqual("generic_device_fallback", preview_payload.get("capture_scope"))
        self.assertEqual("collect_generic_device_fallback_diagnostics", preview_payload.get("kind"))
        self.assertNotIn("lsusb", preview.text.lower())

        with patch("agent.orchestrator.collect_generic_device_fallback_diagnostics_snapshot", return_value=snapshot) as collect_call:
            confirmed = orchestrator.handle_message("yes", "user-4")

        self.assertEqual(1, collect_call.call_count)
        self.assertEqual("diagnostics_capture", confirmed.data.get("route"))
        payload = confirmed.data.get("runtime_payload") if isinstance(confirmed.data.get("runtime_payload"), dict) else {}
        self.assertEqual("collect_generic_device_fallback_diagnostics", payload.get("kind"))
        cards_payload = payload.get("cards_payload") if isinstance(payload.get("cards_payload"), dict) else {}
        self.assertEqual("General device diagnostics", ((cards_payload.get("cards") or [{}])[0] or {}).get("title"))
        self.assertIn("General device diagnostics", confirmed.text)
        self.assertIn("USB presence:", confirmed.text)
        self.assertIn("PCI presence:", confirmed.text)
        self.assertIn("Logs:", confirmed.text)
        self.assertIn("diagnostics_snapshot", payload)

    def test_generic_device_fallback_logs_gap_event_with_trace_and_prefix(self) -> None:
        orchestrator = self._orchestrator()
        with patch("agent.orchestrator.nl_route", return_value={"intent": "DIAGNOSTICS_CAPTURE_GENERIC_DEVICE_FALLBACK_REQUEST"}), patch.object(
            orchestrator,
            "_diagnostics_capture_preview_response",
            return_value=OrchestratorResponse(
                "preview",
                {"route": "diagnostics_capture", "skip_post_response_hooks": True},
            ),
        ) as preview_mock, patch.object(orchestrator, "_record_runtime_event") as record_mock:
            response = orchestrator.handle_message(
                "My webcam is not detected after sleep and I need help figuring out why.",
                "user-4",
                chat_context={"trace_id": "trace-123", "source_surface": "api"},
            )

        self.assertEqual("preview", response.text)
        preview_mock.assert_called_once()
        record_mock.assert_called_once()
        event_name, = record_mock.call_args.args
        self.assertEqual("diagnostics_fallback", event_name)
        fields = record_mock.call_args.kwargs
        self.assertEqual("diagnostics_fallback", fields.get("route"))
        self.assertEqual("diagnostics_capture_generic_device_fallback_request", fields.get("intent_label"))
        self.assertEqual("trace-123", fields.get("trace_id"))
        self.assertTrue(str(fields.get("text_prefix") or "").startswith("My webcam is not detected after sleep"))
        self.assertLessEqual(len(str(fields.get("text_prefix") or "")), 83)

    def test_resource_summary_acknowledges_monitor_mismatch_without_fabricating_zeroes(self) -> None:
        orchestrator = self._orchestrator()
        summary, followups = orchestrator._domain_summary_and_followups(  # type: ignore[attr-defined]
            "resource_governor",
            "resource_report",
            {
                "payload": {
                    "source": "live",
                    "loads": {"1m": 0.42},
                    "memory": {
                        "used": int(21.3 * 1024**3),
                        "total": int(67.4 * 1024**3),
                        "available": int(46.1 * 1024**3),
                        "free": int(12.4 * 1024**3),
                    },
                    "memory_note": "MemAvailable includes reclaimable cache/buffers/shared memory: cached 8G, buffers 512M, shared 256M",
                }
            },
            "actual system monitor reading: 21.3 GB / 31.6% steady use",
        )
        self.assertIn("wrong or incomplete", summary.lower())
        self.assertIn("21.3 GiB", summary)
        self.assertNotIn("0.0%", summary)
        self.assertIn("memory delta", " ".join(followups).lower())

    def test_simple_ram_question_is_concise_and_skips_deep_process_breakdown(self) -> None:
        orchestrator = self._orchestrator()
        live = {
            "taken_at": "2026-04-21T15:03:00-06:00",
            "hostname": "host-a",
            "loadavg": (1.23, 0.99, 0.75),
            "mem": {
                "total": int(67.4 * 1024**3),
                "used": int(21.3 * 1024**3),
                "available": int(46.1 * 1024**3),
                "free": int(12.4 * 1024**3),
                "buffers": int(512 * 1024**2),
                "cached": int(8 * 1024**3),
                "shared": int(256 * 1024**2),
                "used_pct": 31.6,
            },
            "swap": {"total": 0, "used": 0},
            "top_cpu": [{"pid": 11, "name": "python", "cpu_ticks": 42, "rss_bytes": int(2.8 * 1024**3)}],
            "top_rss": [
                {"pid": 22, "name": "chrome", "cpu_ticks": 18, "rss_bytes": int(4.2 * 1024**3)},
                {"pid": 11, "name": "python", "cpu_ticks": 42, "rss_bytes": int(2.8 * 1024**3)},
            ],
            "proc_stats": {"procs_scanned": 128, "errors_skipped": 2},
        }

        with patch.object(collector, "collect_live_snapshot", return_value=live):
            response = orchestrator.handle_message("how much RAM am I using right now?", "tester")

        self.assertIn("You’re using", response.text)
        self.assertIn("21.3 GiB of RAM", response.text)
        self.assertNotIn("Likely cause:", response.text)
        self.assertNotIn("Top CPU processes", response.text)

    def test_ram_diagnostic_question_keeps_full_explanation(self) -> None:
        orchestrator = self._orchestrator()
        live = {
            "taken_at": "2026-04-21T15:03:00-06:00",
            "hostname": "host-a",
            "loadavg": (1.23, 0.99, 0.75),
            "mem": {
                "total": int(67.4 * 1024**3),
                "used": int(21.3 * 1024**3),
                "available": int(46.1 * 1024**3),
                "free": int(12.4 * 1024**3),
                "buffers": int(512 * 1024**2),
                "cached": int(8 * 1024**3),
                "shared": int(256 * 1024**2),
                "used_pct": 31.6,
            },
            "swap": {"total": 0, "used": 0},
            "top_cpu": [{"pid": 11, "name": "python", "cpu_ticks": 42, "rss_bytes": int(2.8 * 1024**3)}],
            "top_rss": [
                {"pid": 22, "name": "chrome", "cpu_ticks": 18, "rss_bytes": int(4.2 * 1024**3)},
                {"pid": 11, "name": "python", "cpu_ticks": 42, "rss_bytes": int(2.8 * 1024**3)},
            ],
            "proc_stats": {"procs_scanned": 128, "errors_skipped": 2},
        }

        with patch.object(collector, "collect_live_snapshot", return_value=live):
            response = orchestrator.handle_message("what is using my RAM?", "tester")

        self.assertIn("Likely cause:", response.text)
        self.assertIn("Top CPU processes", response.text)
        self.assertNotIn("You’re using", response.text)

    def test_runtime_configure_ollama_routes_to_setup_flow_without_llm(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_provider = "openrouter"
        runtime_truth.current_model = runtime_truth.openrouter_model
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("configure ollama", "user1")
        self.assertEqual("setup_flow", response.data["route"])
        self.assertFalse(response.data["used_llm"])
        self.assertIn(("choose_best_local_chat_model", {"refresh": True}), runtime_truth.calls)
        self.assertIn(("configure_local_chat_model", "ollama:qwen3.5:8b"), runtime_truth.calls)
        self.assertIn("ollama", response.text.lower())
        self.assertEqual(0, len(llm.chat_calls))

    def test_product_specific_guard_without_runtime_service_blocks_llm_guessing(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message("is openrouter configured?", "user1")
        self.assertEqual("I can't read a clean runtime status yet.", response.text)
        self.assertEqual("provider_status", response.data["route"])
        self.assertTrue(response.data["used_runtime_state"])
        self.assertFalse(response.data["used_llm"])
        self.assertEqual([], llm.chat_calls)

    def test_current_model_response_qualifies_down_provider_state(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="should not run")
        runtime_truth = _FakeRuntimeTruthService()
        runtime_truth.current_provider = "openrouter"
        runtime_truth.current_model = runtime_truth.openrouter_model
        runtime_truth.effective_provider = "ollama"
        runtime_truth.effective_model = "ollama:qwen3.5:4b"
        runtime_truth.current_ready = False
        runtime_truth.current_provider_health_status = "down"
        runtime_truth.current_model_health_status = "down"
        runtime_truth.openrouter_configured = True
        runtime_truth.openrouter_secret_present = True
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            runtime_truth_service=runtime_truth,
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            with patch.object(
                orchestrator,
                "_apply_assistant_response_guard",
                side_effect=AssertionError("post-response guard should not run"),
            ):
                response = orchestrator.handle_message("what model are you using right now?", "user1")

        self.assertEqual("model_status", response.data["route"])
        self.assertIn("configured to use", response.text.lower())
        self.assertIn("best healthy target would be", response.text.lower())
        self.assertIn("ollama:qwen3.5:4b", response.text)
        self.assertNotIn(("chat_target_truth", None), runtime_truth.calls)
        self.assertEqual(0, len(llm.chat_calls))
        self.assertTrue(response.data.get("skip_post_response_hooks", False))
        timing = response.data.get("orchestrator_timing_ms") if isinstance(response.data.get("orchestrator_timing_ms"), dict) else {}
        self.assertEqual(0, int(timing.get("assistant_response_guard_ms", -1)))

    def test_skill_context_exposes_bound_canonical_route_only(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="hi from llm")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        ctx = orchestrator._context()
        self.assertIn("route_inference", ctx)
        self.assertNotIn("llm_router", ctx)

        result = ctx["route_inference"](
            messages=[{"role": "user", "content": "rewrite this"}],
            user_text="rewrite this",
            task_hint="rewrite this",
            purpose="chat",
        )
        self.assertTrue(result.get("ok"))
        self.assertEqual("hi from llm", result.get("text"))
        self.assertEqual(1, len(llm.chat_calls))

    def test_no_llm_available_returns_bootstrap_chat_setup(self) -> None:
        llm = _FakeChatLLM(enabled=False)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        response = orchestrator.handle_message("hello", "user1")
        self.assertIn("I’m not ready to chat yet", response.text)
        self.assertIn("Open Setup", response.text)
        self.assertEqual([], llm.chat_calls)

    def test_llm_chat_uses_runtime_adapter_chat_availability_when_client_disabled(self) -> None:
        llm = _FakeChatLLM(enabled=False, text="unused")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            chat_runtime_adapter=_RuntimeChatAvailableAdapter(),
        )
        with patch(
            "agent.orchestrator.route_inference",
            return_value={
                "ok": True,
                "text": "hi from llm",
                "provider": "ollama",
                "model": "llama3",
                "task_type": "chat",
                "selection_reason": "healthy",
                "fallback_used": False,
                "error_kind": None,
                "next_action": None,
                "trace_id": "orch-test",
                "data": {"selection": {"selected_model": "llama3", "provider": "ollama", "reason": "healthy"}},
            },
        ) as route_mock:
            response = orchestrator._llm_chat("user1", "what can you help with?")

        self.assertEqual("hi from llm", response.text)
        route_mock.assert_called_once()
        self.assertEqual([], llm.chat_calls)

    def test_llm_chat_run_directive_executes_internal_brief(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="[[RUN:/brief]]")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        expected = OrchestratorResponse("BRIEF_OUTPUT")
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._llm_chat("user1", "what changed on my pc")
        self.assertEqual("BRIEF_OUTPUT", response.text)
        run_mock.assert_called_once_with("/brief", "user1")

    def test_llm_chat_embedded_run_directive_executes_internal_health(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="Sure — [[RUN:/health]]")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        expected = OrchestratorResponse("HEALTH_OUTPUT")
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._llm_chat("user1", "is my system running ok")
        self.assertEqual("HEALTH_OUTPUT", response.text)
        run_mock.assert_called_once_with("/health", "user1")

    def test_llm_chat_heuristic_fallback_executes_internal_health(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="I can check that for you.")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        expected = OrchestratorResponse("HEALTH_OUTPUT")
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._llm_chat("user1", "show me the stats")
        self.assertEqual("HEALTH_OUTPUT", response.text)
        run_mock.assert_called_once_with("/health", "user1")

    def test_llm_chat_heuristic_health_phrase_executes_internal_health(self) -> None:
        llm = _RaisingChatLLM(enabled=True)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        expected = OrchestratorResponse("HEALTH_OUTPUT")
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._llm_chat("user1", "how is the bot health")
        self.assertEqual("HEALTH_OUTPUT", response.text)
        run_mock.assert_called_once_with("/health", "user1")
        self.assertEqual(0, llm.chat_calls)

    def test_llm_chat_without_run_directive_returns_llm_text(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="Regular chat answer")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        with patch.object(orchestrator, "_handle_message_impl") as run_mock:
            response = orchestrator._llm_chat("user1", "hello")
        self.assertEqual("Regular chat answer", response.text)
        run_mock.assert_not_called()

    def test_llm_chat_delegates_to_inference_router_for_chat_execution(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="Code answer")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        with patch(
            "agent.orchestrator.route_inference",
            return_value={
                "ok": True,
                "text": "Code answer",
                "provider": "ollama",
                "model": "ollama:qwen2.5:7b-instruct",
                "task_type": "coding",
                "selection_reason": "healthy+approved+local_first+task=coding",
                "fallback_used": False,
                "error_kind": None,
                "next_action": None,
                "trace_id": "orch-test",
                "data": {
                    "task_request": {
                        "task_type": "coding",
                        "requirements": ["chat"],
                        "preferred_local": True,
                    },
                    "selection": {
                        "selected_model": "ollama:qwen2.5:7b-instruct",
                        "provider": "ollama",
                        "reason": "healthy+approved+local_first+task=coding",
                        "fallbacks": ["ollama:qwen2.5:3b-instruct"],
                    },
                },
            },
        ) as route_mock:
            response = orchestrator._llm_chat("user1", "debug this python traceback")
        self.assertEqual("Code answer", response.text)
        self.assertEqual([], llm.chat_calls)
        route_kwargs = route_mock.call_args.kwargs
        self.assertEqual("chat", route_kwargs.get("purpose"))
        self.assertEqual("debug this python traceback", route_kwargs.get("user_text"))
        self.assertEqual("debug this python traceback", route_kwargs.get("task_hint"))
        messages = route_kwargs.get("messages") or []
        self.assertIn("Always identify as the local Personal Agent", str(messages[0].get("content") or ""))
        self.assertIn("Avoid generic filler", str(messages[0].get("content") or ""))

    def test_llm_chat_includes_selective_memory_context_when_relevant(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="Sure — let's continue.")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        orchestrator.db.set_preference("response_style", "concise")
        orchestrator.db.add_open_loop("finish the transcript harness", "2026-03-20")
        orchestrator._memory_runtime.set_current_topic("user1", topic="safe mode stabilization")
        orchestrator._memory_runtime.record_user_request("user1", "finish the safe mode transcript harness")

        with patch(
            "agent.orchestrator.route_inference",
            return_value={
                "ok": True,
                "text": "Sure — let's continue.",
                "provider": "ollama",
                "model": "ollama:qwen3.5:4b",
                "task_type": "chat",
                "selection_reason": "healthy+approved+local_first+task=chat",
                "fallback_used": False,
                "error_kind": None,
                "next_action": None,
                "trace_id": "orch-test",
                "data": {"selection": {"selected_model": "ollama:qwen3.5:4b", "provider": "ollama", "reason": "healthy"}},
            },
        ) as route_mock:
            response = orchestrator._llm_chat("user1", "Can you help me continue with the next step?")

        self.assertEqual("Sure — let's continue.", response.text)
        route_kwargs = route_mock.call_args.kwargs
        messages = route_kwargs.get("messages") or []
        system_text = str((messages or [{}])[0].get("content") if messages else "")
        self.assertIn("Relevant remembered context", system_text)
        self.assertIn("Current topic: safe mode stabilization", system_text)
        self.assertIn("Preferences: you prefer concise replies", system_text)
        self.assertIn("Open loops: finish the transcript harness", system_text)

    def test_llm_chat_compacts_working_memory_before_prompt_assembly(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="Continuing now.")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        state = WorkingMemoryState()
        for index in range(8):
            role = "assistant" if index % 2 else "user"
            append_turn(
                state,
                role=role,  # type: ignore[arg-type]
                text=(f"Older implementation exchange {index}. " + "token " * 160).strip(),
            )
        orchestrator._memory_runtime.save_working_memory_state("user1", state)

        with patch(
            "agent.orchestrator.route_inference",
            return_value={
                "ok": True,
                "text": "Continuing now.",
                "provider": "ollama",
                "model": "ollama:qwen3.5:4b",
                "task_type": "chat",
                "selection_reason": "healthy+approved+local_first+task=chat",
                "fallback_used": False,
                "error_kind": None,
                "next_action": None,
                "trace_id": "orch-test",
                "data": {"selection": {"selected_model": "ollama:qwen3.5:4b", "provider": "ollama", "reason": "healthy"}},
            },
        ) as route_mock:
            response = orchestrator._llm_chat(
                "user1",
                "Please continue the implementation.",
                chat_context={
                    "payload": {"min_context_tokens": 4096},
                    "messages": [{"role": "user", "content": "Please continue the implementation."}],
                    "thread_id": "thread-a",
                },
            )

        self.assertEqual("Continuing now.", response.text)
        route_kwargs = route_mock.call_args.kwargs
        messages = route_kwargs.get("messages") or []
        system_text = str((messages or [{}])[0].get("content") if messages else "")
        self.assertIn("Working memory summaries", system_text)
        hot_messages = messages[1:]
        self.assertEqual("assistant", hot_messages[-2]["role"])
        self.assertEqual("user", hot_messages[-1]["role"])
        self.assertEqual("Please continue the implementation.", hot_messages[-1]["content"])
        older_texts = [str(row.get("content") or "") for row in hot_messages]
        self.assertNotIn("Older implementation exchange 0.", " ".join(older_texts))

    def test_llm_chat_returns_install_plan_guidance_when_no_model_selected(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="unused")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        with patch(
            "agent.orchestrator.route_inference",
            return_value={
                "ok": False,
                "text": build_no_llm_public_message(),
                "provider": None,
                "model": None,
                "task_type": "chat",
                "selection_reason": "no_suitable_model",
                "fallback_used": True,
                "error_kind": "no_suitable_model",
                "next_action": "Run: python -m agent llm_install --model ollama:qwen2.5:3b-instruct --approve",
                "trace_id": "orch-test",
                "data": {
                    "task_request": {
                        "task_type": "chat",
                        "requirements": ["chat"],
                        "preferred_local": True,
                    },
                    "selection": {
                        "selected_model": None,
                        "provider": None,
                        "reason": "no_suitable_model",
                        "fallbacks": [],
                    },
                    "plan": {
                        "needed": True,
                        "approved": True,
                        "plan": [{"action": "ollama.pull_model", "model": "qwen2.5:3b-instruct"}],
                        "next_action": "Run: python -m agent llm_install --model ollama:qwen2.5:3b-instruct --approve",
                    },
                },
            },
        ):
            response = orchestrator._llm_chat("user1", "tell me a joke")
        self.assertEqual(build_no_llm_public_message(), response.text)
        self.assertNotIn("No suitable local-first model is ready", response.text)
        self.assertNotIn("Run: python -m agent llm_install --model ollama:qwen2.5:3b-instruct --approve", response.text)
        self.assertEqual([], llm.chat_calls)

    def test_llm_chat_keeps_junk_command_suffix_unchanged_when_no_trigger(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="Regular answer /brief /status /help")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        with patch.object(orchestrator, "_handle_message_impl") as run_mock:
            response = orchestrator._llm_chat("user1", "hello there")
        self.assertEqual("Regular answer /brief /status /help", response.text)
        run_mock.assert_not_called()

    def test_llm_chat_sanitizes_untrusted_vendor_identity_claim(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="I am created by Anthropic.")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        with patch.object(orchestrator, "_handle_message_impl") as run_mock:
            response = orchestrator._llm_chat("user1", "hello")
        self.assertIn("Personal Agent", response.text)
        self.assertNotIn("created by Anthropic", response.text)
        run_mock.assert_not_called()

    def test_llm_chat_sanitizes_model_self_identity_claim(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="I am DeepSeek, and I can help with that.")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        response = orchestrator._llm_chat("user1", "hello")
        self.assertIn("Personal Agent", response.text)
        self.assertNotIn("I am DeepSeek", response.text)
        self.assertNotIn("deepseek", response.text.lower())

    def test_llm_chat_sanitizes_external_host_origin_claim(self) -> None:
        llm = _FakeChatLLM(
            enabled=True,
            text="I am running in an environment managed by Alibaba Cloud and can help with that.",
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        response = orchestrator._llm_chat("user1", "who are you?")
        self.assertIn("Personal Agent", response.text)
        self.assertNotIn("alibaba cloud", response.text.lower())
        self.assertNotIn("managed by", response.text.lower())

    def test_llm_chat_sanitizes_generic_company_origin_claim(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="I was created by Example Company to help you.")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        response = orchestrator._llm_chat("user1", "who are you?")
        self.assertIn("your Personal Agent", response.text)
        self.assertNotIn("example company", response.text.lower())
        self.assertNotIn("created by", response.text.lower())

    def test_llm_chat_sanitizes_vendor_identity_even_when_provider_matches(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="placeholder")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        llm.chat = lambda *_args, **_kwargs: {  # type: ignore[assignment]
            "ok": True,
            "text": "I am created by Anthropic.",
            "provider": "anthropic",
            "model": "claude-3.5-sonnet",
        }
        response = orchestrator._llm_chat("user1", "hello")
        self.assertIn("Personal Agent", response.text)
        self.assertNotIn("created by Anthropic", response.text)

    def test_llm_chat_uses_configured_names_without_inventing_persona(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="I am created by Anthropic.")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        orchestrator.db.set_preference("assistant_name", "Nova")
        orchestrator.db.set_preference("user_name", "Casey")

        with patch(
            "agent.orchestrator.route_inference",
            return_value={
                "ok": True,
                "text": "I am created by Anthropic.",
                "provider": "anthropic",
                "model": "claude-3.5-sonnet",
                "task_type": "chat",
                "selection_reason": "healthy",
                "fallback_used": False,
                "error_kind": None,
                "next_action": None,
                "trace_id": "orch-test",
                "data": {"selection": {"selected_model": "claude-3.5-sonnet", "provider": "anthropic", "reason": "healthy"}},
            },
        ) as route_mock:
            response = orchestrator._llm_chat("user1", "who are you?")

        self.assertIn("Nova, your Personal Agent", response.text)
        self.assertNotIn("Anthropic", response.text)
        system_text = str((route_mock.call_args.kwargs.get("messages") or [{}])[0].get("content") or "")
        self.assertIn("You may refer to the user as Casey sparingly", system_text)

    def test_llm_chat_does_not_invent_name_when_unset(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="I am created by OpenAI.")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        response = orchestrator._llm_chat("user1", "who are you?")
        self.assertIn("your Personal Agent", response.text)
        self.assertNotIn("Nova", response.text)

    def test_llm_chat_json_tool_request_executes_internal_status(self) -> None:
        llm = _FakeChatLLM(
            enabled=True,
            text='{"tool":"status","args":{},"reason":"status_check","read_only":true,"confidence":0.9}',
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        expected = OrchestratorResponse("STATUS_OUTPUT")
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._llm_chat("user1", "show me status")
        self.assertEqual("STATUS_OUTPUT", response.text)
        run_mock.assert_called_once_with("/status", "user1")

    def test_llm_chat_unsupported_tool_request_returns_deterministic_refusal(self) -> None:
        llm = _FakeChatLLM(
            enabled=True,
            text='{"tool":"shutdown_everything","args":{},"reason":"bad","read_only":false,"confidence":1.0}',
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        response = orchestrator._llm_chat("user1", "please do something unsupported")
        self.assertIn("failure_code: tool_unsupported", response.text)
        self.assertIn("component: orchestrator.tool_executor", response.text)
        self.assertIn("next_action:", response.text)

    def test_execute_tool_request_allows_read_only_in_degraded(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        expected = OrchestratorResponse("STATUS_OUTPUT")
        request = normalize_tool_request({"tool": "status", "args": {}, "reason": "degraded_check"})
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._execute_tool_request(
                tool_request=request,
                user_id="user1",
                surface="orchestrator",
                runtime_mode="DEGRADED",
            )
        self.assertEqual("STATUS_OUTPUT", response.text)
        run_mock.assert_called_once_with("/status", "user1")

    def test_llm_chat_exception_with_stats_uses_health_fallback(self) -> None:
        llm = _RaisingChatLLM(enabled=True)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        expected = OrchestratorResponse("HEALTH_OUTPUT")
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._llm_chat("user1", "show me the stats")
        self.assertEqual("HEALTH_OUTPUT", response.text)
        run_mock.assert_called_once_with("/health", "user1")
        self.assertEqual(0, llm.chat_calls)

    def test_llm_chat_exception_without_heuristic_returns_friendly_fallback(self) -> None:
        llm = _RaisingChatLLM(enabled=True)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        with patch.object(orchestrator, "_handle_message_impl") as run_mock:
            response = orchestrator._llm_chat("user1", "tell me a joke")
        self.assertIn("I’m not ready to chat yet", response.text)
        self.assertNotIn("Chat LLM is unavailable.", response.text)
        self.assertNotIn("Try /brief", response.text)
        run_mock.assert_not_called()
        self.assertEqual(1, llm.chat_calls)

    def test_assistant_frontdoor_translates_raw_llm_unavailable_error(self) -> None:
        llm = _RaisingChatLLM(enabled=True)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        response = orchestrator.handle_message("please summarize the current task", "user1")
        self.assertIn("I’m not ready to chat yet", response.text)
        self.assertNotIn("Something went wrong while answering that", response.text)
        self.assertNotIn("Chat LLM is unavailable.", response.text)
        self.assertNotIn("python -m agent setup", response.text)
        self.assertEqual(1, llm.chat_calls)

    def test_handle_message_short_joke_prompt_short_circuits_before_llm(self) -> None:
        llm = _RaisingChatLLM(enabled=True)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        response = orchestrator.handle_message("actually, give me a one-line joke", "user1")
        self.assertIn("good backlog", response.text)
        self.assertEqual("joke", response.data.get("route"))
        self.assertTrue(bool(response.data.get("ok", False)))
        self.assertEqual(0, llm.chat_calls)

    def test_assistant_frontdoor_translates_structured_internal_tool_error(self) -> None:
        llm = _FakeChatLLM(
            enabled=True,
            text='{"tool":"shutdown_everything","args":{},"reason":"bad","read_only":false,"confidence":1.0}',
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )
        response = orchestrator.handle_message("please do something unsupported", "user1")
        self.assertIn("Something went wrong while handling that request", response.text)
        self.assertNotIn("failure_code:", response.text)
        self.assertNotIn("trace_id:", response.text)
        self.assertNotIn("component:", response.text)

    def test_followup_yes_is_ambiguous_with_multiple_pending_items(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._memory_runtime.add_pending_item(
            "user1",
            {
                "pending_id": "p1",
                "kind": "clarification",
                "origin_tool": "ask_query",
                "question": "Pick one",
                "options": ["a", "b"],
                "thread_id": "thread-a",
                "created_at": 1,
                "expires_at": 9999999999,
                "status": "WAITING_FOR_USER",
            },
        )
        orchestrator._memory_runtime.add_pending_item(
            "user1",
            {
                "pending_id": "p2",
                "kind": "clarification",
                "origin_tool": "ask_query",
                "question": "Pick one",
                "options": ["a", "b"],
                "thread_id": "thread-a",
                "created_at": 2,
                "expires_at": 9999999999,
                "status": "WAITING_FOR_USER",
            },
        )
        response = orchestrator.handle_message("yes", "user1")
        self.assertIn("failure_code: followup_ambiguous", response.text)
        self.assertIn("trace_id:", response.text)
        self.assertIn("next_action:", response.text)

    def test_followup_yes_without_pending_returns_no_resumable_error(self) -> None:
        orchestrator = self._orchestrator()
        response = orchestrator.handle_message("yes", "user1")
        self.assertIn("failure_code: no_resumable_work", response.text)
        self.assertIn("trace_id:", response.text)
        self.assertIn("next_action:", response.text)

    def test_followup_yes_with_expired_pending_returns_expired_error(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._memory_runtime.add_pending_item(
            "user1",
            {
                "pending_id": "p-exp",
                "kind": "followup",
                "origin_tool": "compare_now",
                "question": "Run compare?",
                "options": ["yes", "no"],
                "thread_id": "thread-a",
                "created_at": 1,
                "expires_at": 2,
                "status": "READY_TO_RESUME",
            },
        )
        response = orchestrator.handle_message("yes", "user1")
        self.assertIn("failure_code: pending_expired", response.text)
        self.assertIn("trace_id:", response.text)

    def test_memory_command_returns_deterministic_summary(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._memory_runtime.set_current_topic("user1", topic="setup ollama")
        orchestrator._memory_runtime.add_pending_item(
            "user1",
            {
                "pending_id": "p1",
                "kind": "clarification",
                "origin_tool": "setup",
                "question": "Which model size?",
                "options": ["small", "medium"],
                "thread_id": "thread-a",
                "created_at": 1,
                "expires_at": 9999999999,
                "status": "WAITING_FOR_USER",
            },
        )
        response = orchestrator.handle_message("/memory", "user1")
        self.assertIn("Memory summary (thread thread-a):", response.text)
        self.assertIn("Pending items: 1", response.text)
        self.assertIn("Resumable: yes", response.text)

    def test_plan_day_seeded_topic_can_be_resumed_with_working_context_prompt(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")

        first = orchestrator.handle_message("help me plan my day", "user1")
        second = orchestrator.handle_message("what are we doing?", "user1")

        self.assertIn("Today priorities", first.text)
        self.assertEqual("plan_day", first.data.get("route"))
        self.assertFalse(bool(first.data.get("generic_fallback_used")))
        self.assertFalse(bool(first.data.get("generic_fallback_allowed")))
        self.assertIn("We were working on today plan.", second.text)
        self.assertIn("I can pick up today plan from there.", second.text)

    def test_viability_gate_probe_seeds_working_context_for_resume_prompts(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")

        first = orchestrator.handle_message("we are testing the assistant viability gate", "user1")
        second = orchestrator.handle_message("what are we doing?", "user1")

        self.assertIn("assistant_viability_gate", first.text)
        self.assertIn("We were working on assistant viability gate.", second.text)

    def test_long_chat_coherence_probe_uses_deterministic_assistant_reply(self) -> None:
        orchestrator = self._orchestrator()

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            response = orchestrator.handle_message(
                "I'm testing whether you can stay coherent through a long chat.",
                "user1",
            )

        self.assertIn("ask your next question", response.text.lower())
        self.assertIn("keep the thread consistent", response.text.lower())
        self.assertEqual("generic_chat", response.data.get("route"))
        self.assertFalse(bool(response.data.get("used_llm")))

    def test_generic_testing_statement_seeds_working_context_with_fast_ack(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")

        first = orchestrator.handle_message("we are testing the barrage hardening task", "user1")
        second = orchestrator.handle_message("what are we doing?", "user1")

        self.assertEqual("agent_memory", first.data.get("route"))
        self.assertIn("Got it. We're working on the barrage hardening task.", first.text)
        self.assertIn("We were working on the barrage hardening task.", second.text)
        self.assertIn("I can pick up the barrage hardening task from there.", second.text)

    def test_selective_chat_memory_context_uses_rewind_prompts(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._memory_runtime.set_current_topic("user1", topic="assistant viability gate")

        context = orchestrator._selective_chat_memory_context(  # noqa: SLF001
            "user1",
            "No, go back and explain the larger task.",
        )

        self.assertIn("Current topic: assistant viability gate", context)
        self.assertIn("assistant viability gate", context)

    def test_memory_overview_uses_rewind_prompts_for_working_context(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._memory_runtime.set_current_topic("user1", topic="assistant viability gate")

        response = orchestrator._assistant_memory_overview_response(  # noqa: SLF001
            "user1",
            query_text="No, go back and explain the larger task.",
        )

        self.assertIn("We were working on assistant viability gate.", response.text)
        self.assertIn("I can pick up assistant viability gate from there.", response.text)

    def test_correction_response_uses_current_topic_when_available(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._memory_runtime.set_current_topic("user1", topic="assistant viability gate")
        orchestrator._last_interpretable_result["user1"] = {  # noqa: SLF001
            "created_ts": int(time.time()),
            "route": "generic_chat",
            "kind": "generic_chat",
            "summary": "keep the answer short",
            "payload": {"summary": "keep the answer short"},
        }

        response = orchestrator._assistant_correction_response(  # noqa: SLF001
            user_id="user1",
            used_memory=True,
            reason="correction_turn",
        )

        self.assertIn("We were working on assistant viability gate.", response.text)
        self.assertIn("If you want, I can pick it back up from there.", response.text)

    def test_correction_prompt_includes_rewind_phrases(self) -> None:
        self.assertTrue(Orchestrator._looks_like_correction_prompt("No, go back and explain the larger task."))
        self.assertTrue(Orchestrator._looks_like_correction_prompt("If you had to continue from here, what would you do next?"))

    def test_working_context_rewind_prompt_detector_matches_long_session_turns(self) -> None:
        self.assertTrue(
            Orchestrator._looks_like_working_context_rewind_prompt("No, go back and explain the larger task.")
        )
        self.assertTrue(
            Orchestrator._looks_like_working_context_rewind_prompt("If you had to continue from here, what would you do next?")
        )
        self.assertTrue(
            Orchestrator._looks_like_working_context_rewind_prompt("What should we do next?")
        )
        self.assertTrue(
            Orchestrator._looks_like_working_context_rewind_prompt("Summarize where we left this in one sentence.")
        )

    def test_memory_overview_uses_actionable_next_step_for_working_context(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._memory_runtime.set_current_topic("user1", topic="today_plan")

        response = orchestrator._assistant_memory_overview_response(  # noqa: SLF001
            "user1",
            query_text="What should we do next?",
        )

        self.assertIn("We were working on today plan.", response.text)
        self.assertIn("Best next step: continue with today plan.", response.text)

    def test_memory_overview_uses_natural_last_request_phrase(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._memory_runtime.set_current_topic("user1", topic="today_plan")
        orchestrator._memory_runtime.record_user_request("user1", "what are we doing?")

        response = orchestrator._assistant_memory_overview_response(  # noqa: SLF001
            "user1",
            query_text="what are we doing?",
        )

        self.assertIn("Last thing you asked me to do: what are we doing?.", response.text)
        self.assertNotIn("Your last concrete request was:", response.text)

    def test_summarize_the_work_prefers_working_context_over_correction(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._memory_runtime.set_current_topic("user1", topic="assistant viability gate")
        orchestrator._last_interpretable_result["user1"] = {  # noqa: SLF001
            "created_ts": int(time.time()),
            "route": "runtime_status",
            "kind": "runtime_status",
            "summary": "Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
            "payload": {"summary": "Ready. Using ollama / ollama:qwen2.5:7b-instruct."},
        }

        response = orchestrator._assistant_unmatched_input_response(
            "user1",
            "Okay, now summarize the work in one sentence.",
        )

        assert response is not None
        self.assertEqual("agent_memory", response.data.get("route"))
        self.assertIn("We were working on assistant viability gate.", response.text)
        self.assertNotIn("You’re right, I should reassess that.", response.text)

    def test_context_refusal_detector_matches_generic_dead_end_replies(self) -> None:
        self.assertTrue(
            Orchestrator._looks_like_context_refusal_reply(
                "I do not have access to information outside of our current conversation, and am unable to provide that."
            )
        )

    def test_memory_command_does_not_overwrite_last_meaningful_action(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._memory_runtime.record_agent_action("user1", "Ran /brief", action_kind="brief")

        first = orchestrator.handle_message("/memory", "user1")
        second = orchestrator.handle_message("/memory", "user1")

        self.assertIn("Last action: Ran /brief", first.text)
        self.assertIn("Last action: Ran /brief", second.text)
        self.assertNotIn("Last action: Memory summary", second.text)

    def test_resume_without_pending_keeps_memory_summary_stable(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._memory_runtime.record_agent_action("user1", "Ran /brief", action_kind="brief")

        orchestrator.handle_message("/resume", "user1")
        summary_after_first = orchestrator.handle_message("/memory", "user1")
        orchestrator.handle_message("/resume", "user1")
        summary_after_second = orchestrator.handle_message("/memory", "user1")

        self.assertIn("Last action: Ran /brief", summary_after_first.text)
        self.assertIn("Last action: Ran /brief", summary_after_second.text)
        self.assertEqual(summary_after_first.text, summary_after_second.text)

    def test_memory_command_surfaces_degraded_continuity_state_plainly(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator.db.set_user_pref("memory_runtime:user1:thread_state", "{bad json")

        response = orchestrator.handle_message("/memory", "user1")

        self.assertIn("Continuity memory is degraded right now", response.text)
        self.assertIn("corrupted entry", response.text)
        self.assertEqual("{bad json", orchestrator.db.get_user_pref("memory_runtime:user1:thread_state"))

    def test_compare_followup_yes_resumes_pending_compare(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator._set_active_thread_id_for_user("user1", "thread-a")
        orchestrator._store_pending_compare("user1", "what if we lower background apps")
        with patch("agent.orchestrator.compare_now_to_what_if", return_value="COMPARE_OUTPUT"):
            response = orchestrator.handle_message("yes", "user1")
        self.assertEqual("COMPARE_OUTPUT", response.text)

    def test_knowledge_query_cache_and_cta(self) -> None:
        orchestrator = self._orchestrator()
        response = orchestrator.handle_message("what changed this week", "user1")
        self.assertIn("Want my opinion", response.text)
        entry = orchestrator._knowledge_cache.get_recent("user1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.facts_hash, facts_hash(entry.facts))

    def test_opinion_followup_uses_cached_facts(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator.handle_message("what changed this week", "user1")
        entry = orchestrator._knowledge_cache.get_recent("user1")
        response = orchestrator.handle_message("opinion", "user1")
        self.assertIn("source", response.data.get("data", {}))
        self.assertEqual(response.data["data"]["facts_hash"], entry.facts_hash)

    def test_greeting_then_affirmation_stays_in_bootstrap_when_no_llm(self) -> None:
        orch = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )

        def _insert_system_facts(snapshot_id: str, taken_at: str) -> None:
            facts = {
                "schema": {"name": "system_facts", "version": 1},
                "snapshot": {
                    "snapshot_id": snapshot_id,
                    "taken_at": taken_at,
                    "timezone": "UTC",
                    "collector": {
                        "agent_version": "0.6.0",
                        "hostname": "host",
                        "boot_id": "boot",
                        "uptime_s": 1,
                        "collection_duration_ms": 1,
                        "partial": False,
                        "errors": [],
                    },
                    "provenance": {"sources": []},
                },
                "os": {"kernel": {"release": "6.0.0", "arch": "x86_64"}},
                "cpu": {"load": {"load_1m": 0.1, "load_5m": 0.1, "load_15m": 0.1}},
                "memory": {
                    "ram_bytes": {
                        "total": 16 * 1024**3,
                        "used": 2 * 1024**3,
                        "free": 0,
                        "available": 14 * 1024**3,
                        "buffers": 0,
                        "cached": 0,
                    },
                    "swap_bytes": {"total": 0, "free": 0, "used": 0},
                    "pressure": {
                        "psi_supported": False,
                        "memory_some_avg10": None,
                        "io_some_avg10": None,
                        "cpu_some_avg10": None,
                    },
                },
                "filesystems": {
                    "mounts": [
                        {
                            "mountpoint": "/",
                            "device": "/dev/sda1",
                            "fstype": "ext4",
                            "total_bytes": 100 * 1024**3,
                            "used_bytes": 60 * 1024**3,
                            "avail_bytes": 40 * 1024**3,
                            "used_pct": 60.0,
                            "inodes": {"total": None, "used": None, "avail": None, "used_pct": None},
                        }
                    ]
                },
                "process_summary": {"top_cpu": [], "top_mem": []},
                "integrity": {"content_hash_sha256": "0" * 64, "signed": False, "signature": None},
            }
            facts_json = json.dumps(facts, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            self.db.insert_system_facts_snapshot(
                id=snapshot_id,
                user_id="user1",
                taken_at=taken_at,
                boot_id="boot",
                schema_version=1,
                facts_json=facts_json,
                content_hash_sha256="0" * 64,
                partial=False,
                errors_json="[]",
            )

        def observe_handler(ctx, user_id=None):
            _insert_system_facts("snap-1", "2026-02-06T00:00:00+00:00")
            return {"text": "Snapshot taken", "payload": {}}

        orch.skills["observe_now"].functions["observe_now"].handler = observe_handler

        first = orch.handle_message("hello", "user1")
        self.assertIn("i’m not ready to chat yet", first.text.lower())

        second = orch.handle_message("yes please", "user1")
        self.assertIn("i’m not ready to chat yet", second.text.lower())

    def test_done_invalid_id(self) -> None:
        orch = self._orchestrator()
        response = orch.handle_message("/done abc", "user1")
        self.assertIsInstance(response.data, dict)
        self.assertEqual("Usage: /done <id>", response.data["cards"][0]["lines"][0])

    def test_done_nonexistent_id(self) -> None:
        orch = self._orchestrator()
        response = orch.handle_message("/done 9999", "user1")
        self.assertIsInstance(response.data, dict)
        self.assertEqual("Task not found: 9999", response.data["cards"][0]["lines"][0])

    def test_done_marks_done_then_reports_already_done(self) -> None:
        orch = self._orchestrator()
        task_id = self.db.add_task(None, "Write report", 30, 4)

        first = orch.handle_message(f"/done {task_id}", "user1")
        self.assertIsInstance(first.data, dict)
        self.assertEqual(f"Done: [{task_id}] Write report", first.data["cards"][0]["lines"][0])
        task = self.db.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual("done", task["status"])

        second = orch.handle_message(f"/done {task_id}", "user1")
        self.assertIsInstance(second.data, dict)
        self.assertEqual(f"Already done: [{task_id}] Write report", second.data["cards"][0]["lines"][0])

    def test_build_epistemic_context_scopes_memory_to_active_thread(self) -> None:
        orch = self._orchestrator()
        response = OrchestratorResponse(
            "ok",
            {
                "thread_id": "thread-a",
                "thread_label": "Focus",
                "audit_ref": "audit-7",
                "memory_items": [
                    {"ref": "mem:local-a", "thread_id": "thread-a", "relevant": True},
                    {"ref": "mem:global-style", "scope": "global", "relevant": True},
                    {"ref": "mem:other-b", "thread_id": "thread-b", "relevant": True},
                ],
            },
        )
        ctx = orch._build_epistemic_context("user1", response)
        self.assertEqual("thread-a", ctx.active_thread_id)
        self.assertEqual("Focus", ctx.thread_label)
        self.assertEqual(("mem:global-style", "mem:local-a"), ctx.in_scope_memory)
        self.assertEqual(("mem:global-style", "mem:local-a"), ctx.in_scope_memory_ids)
        self.assertEqual(("mem:other-b",), ctx.out_of_scope_memory)
        self.assertTrue(ctx.out_of_scope_relevant_memory)
        self.assertEqual(("audit-7",), ctx.tool_event_ids)
        self.assertEqual(("thread-a:u:1",), ctx.recent_turn_ids)

    def test_starting_new_thread_resets_turn_count(self) -> None:
        orch = self._orchestrator()
        orch._apply_epistemic_layer("user1", "hello", OrchestratorResponse("hello back", {"thread_id": "thread-a"}))

        same_thread_ctx = orch._build_epistemic_context("user1", OrchestratorResponse("ok", {"thread_id": "thread-a"}))
        self.assertGreaterEqual(same_thread_ctx.thread_turn_count, 2)

        new_thread_ctx = orch._build_epistemic_context("user1", OrchestratorResponse("ok", {"thread_id": "thread-b"}))
        self.assertEqual(0, new_thread_ctx.thread_turn_count)

    def test_epistemic_turn_activity_logs_include_thread_id(self) -> None:
        orch = self._orchestrator()
        orch.handle_message("hello there", "user1")
        rows = self.db.activity_log_list_recent("epistemic_turn", limit=2)
        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            payload = row.get("payload") or {}
            self.assertEqual("user1", payload.get("user_id"))
            self.assertTrue(str(payload.get("thread_id") or "").strip())
            self.assertTrue(str(payload.get("turn_id") or "").strip())
            self.assertIn(payload.get("role"), {"user", "assistant"})

    def test_build_epistemic_candidate_populates_missing_provenance(self) -> None:
        orch = self._orchestrator()
        response = OrchestratorResponse(
            "ok",
            {
                "thread_id": "thread-a",
                "audit_ref": "audit-3",
                "memory_items": [{"id": 11, "ref": "mem:11", "thread_id": "thread-a", "relevant": True}],
                "epistemic_candidate_json": json.dumps(
                    {
                        "kind": "answer",
                        "final_answer": "Confirmed.",
                        "clarifying_question": None,
                        "claims": [
                            {"text": "From user", "support": "user", "ref": None},
                            {"text": "From memory", "support": "memory", "ref": "mem:11"},
                            {"text": "From tool", "support": "tool", "ref": None},
                        ],
                        "assumptions": [],
                        "unresolved_refs": [],
                        "thread_refs": [],
                    },
                    ensure_ascii=True,
                ),
            },
        )
        ctx = orch._build_epistemic_context("user1", response)
        candidate = orch._build_epistemic_candidate(response, ctx)
        self.assertFalse(isinstance(candidate, str))
        assert not isinstance(candidate, str)
        self.assertEqual("thread-a:u:1", candidate.claims[0].user_turn_id)
        self.assertEqual("11", candidate.claims[1].memory_id)
        self.assertEqual("audit-3", candidate.claims[2].tool_event_id)


if __name__ == "__main__":
    unittest.main()
