from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.llm.model_discovery_manager import ModelDiscoveryManager
from agent.orchestrator import Orchestrator
from agent.setup_chat_flow import classify_runtime_chat_route
from agent.telegram_bridge import handle_telegram_text
from memory.db import MemoryDB


def _first_line(text: str) -> str:
    return str(text or "").strip().splitlines()[0] if str(text or "").strip() else ""


class _BehavioralRuntimeTruth:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []
        self.current_provider = "ollama"
        self.current_model = "ollama:qwen2.5:7b-instruct"
        self.default_provider = "ollama"
        self.default_model = "ollama:qwen2.5:7b-instruct"
        self.effective_provider = "ollama"
        self.effective_model = "ollama:qwen2.5:7b-instruct"
        self.ollama_configured = True
        self.openrouter_configured = False
        self.openrouter_secret_present = False
        self.openrouter_health_status = "unknown"
        self.openrouter_health_reason = None
        self.safe_mode = True
        self.allow_remote_recommendation = False
        self.allow_remote_switch = False
        self.allow_install_pull = False
        self.discovery_payload: dict[str, object] = {
            "ok": True,
            "query": None,
            "message": "Found 2 model(s) across 3 source(s), but some sources failed. Source errors: huggingface: hf timeout.",
            "models": [
                {
                    "id": "openrouter:vendor/tiny-gemma",
                    "provider": "openrouter",
                    "source": "openrouter",
                    "capabilities": ["chat"],
                    "local": False,
                    "installable": False,
                    "confidence": 0.78,
                },
                {
                    "id": "ollama:qwen2.5:7b-instruct",
                    "provider": "ollama",
                    "source": "ollama",
                    "capabilities": ["chat"],
                    "local": True,
                    "installable": True,
                    "confidence": 0.93,
                },
            ],
            "sources": [
                {
                    "source": "huggingface",
                    "enabled": True,
                    "queried": True,
                    "ok": False,
                    "count": 0,
                    "error_kind": "fetch_failed",
                    "error": "hf timeout",
                },
                {
                    "source": "openrouter",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 1,
                    "error_kind": None,
                    "error": None,
                },
                {
                    "source": "ollama",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 1,
                    "error_kind": None,
                    "error": None,
                },
                {
                    "source": "external_snapshots",
                    "enabled": False,
                    "queried": False,
                    "ok": True,
                    "count": 0,
                    "error_kind": "not_requested",
                    "error": None,
                },
            ],
            "debug": {
                "source_registry": ["huggingface", "openrouter", "ollama", "external_snapshots"],
                "source_errors": {
                    "huggingface": {"error_kind": "fetch_failed", "error": "hf timeout"}
                },
                "source_counts": {"huggingface": 0, "openrouter": 1, "ollama": 1},
                "matched_count": 2,
            },
        }
        self._openrouter_ready_message = "Paste your OpenRouter API key and I will finish the setup."
        self._openrouter_ready = False
        self._current_ready = True
        self._current_health_status = "ok"
        self._provider_health_status = "ok"

    def current_chat_target_status(self) -> dict[str, object]:
        self.calls.append(("current_chat_target_status", None))
        return {
            "provider": self.current_provider,
            "model": self.current_model,
            "ready": self._current_ready,
            "health_status": self._current_health_status,
            "provider_health_status": self._provider_health_status,
        }

    def chat_target_truth(self) -> dict[str, object]:
        self.calls.append(("chat_target_truth", None))
        return {
            "configured_provider": self.default_provider,
            "configured_model": self.default_model,
            "effective_provider": self.effective_provider,
            "effective_model": self.effective_model,
            "qualification_reason": "Configured default is healthy and ready.",
            "degraded_reason": None,
        }

    def provider_status(self, provider_id: str) -> dict[str, object]:
        self.calls.append(("provider_status", provider_id))
        provider_key = str(provider_id or "").strip().lower()
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
                "health_status": self.openrouter_health_status if self.openrouter_configured else "unknown",
                "health_reason": self.openrouter_health_reason,
                "model_id": "openrouter:openai/gpt-4o-mini" if self.openrouter_configured else None,
                "model_ids": ["openrouter:openai/gpt-4o-mini", "openrouter:anthropic/claude-sonnet-4"],
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
            "health_status": self._current_health_status if self.current_provider == "ollama" else "ok",
            "health_reason": None,
            "model_id": self.current_model if self.current_provider == "ollama" else "ollama:qwen2.5:7b-instruct",
            "model_ids": ["ollama:qwen2.5:7b-instruct", "ollama:qwen3.5:4b"],
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

    def runtime_status(self, kind: str = "runtime_status") -> dict[str, object]:
        self.calls.append(("runtime_status", kind))
        if str(kind).strip().lower() == "telegram_status":
            return {
                "scope": "telegram",
                "configured": True,
                "state": "ready",
                "summary": "Telegram is ready.",
            }
        return {
            "scope": "ready",
            "ready": True,
            "runtime_mode": "READY",
            "failure_code": None,
            "provider": self.effective_provider,
            "model": self.effective_model,
            "configured_provider": self.default_provider,
            "configured_model": self.default_model,
            "qualification_reason": "The runtime is healthy and ready.",
            "summary": "The runtime is healthy and ready.",
        }

    def model_discovery_query(self, query: str | None = None, filters: dict[str, object] | None = None) -> dict[str, object]:
        self.calls.append(("model_discovery_query", {"query": query, "filters": dict(filters or {})}))
        payload = dict(self.discovery_payload)
        payload["query"] = query
        return payload

    def model_controller_policy_status(self) -> dict[str, object]:
        self.calls.append(("model_controller_policy_status", None))
        return {
            "mode": "safe",
            "mode_label": "SAFE MODE",
            "mode_source": "config_default",
            "allow_remote_recommendation": self.allow_remote_recommendation,
            "allow_remote_fallback": self.allow_remote_recommendation,
            "allow_install_pull": self.allow_install_pull,
        }

    def model_policy_status(self) -> dict[str, object]:
        self.calls.append(("model_policy_status", None))
        current = {
            "model_id": self.current_model,
            "provider_id": self.current_provider,
            "local": self.current_provider == "ollama",
        }
        recommendation = {
            "model_id": "openrouter:vendor/coder-pro",
            "provider_id": "openrouter",
            "local": False,
        }
        return {
            "cheap_remote_cap_per_1m": 1.0,
            "general_remote_cap_per_1m": 5.0,
            "current_candidate": current,
            "selected_candidate": current,
            "recommended_candidate": recommendation,
            "switch_recommended": False,
            "decision_detail": "the current default is already good enough",
            "tier_candidates": {"free_remote": {"model_id": "openrouter:vendor/free-chat", "provider_id": "openrouter"}},
        }

    def model_policy_provider_candidate(self, provider_id: str | None) -> dict[str, object]:
        self.calls.append(("model_policy_provider_candidate", provider_id))
        provider_key = str(provider_id or "").strip().lower() or None
        candidate = None
        if provider_key == "openrouter" and self.openrouter_configured:
            candidate = {"model_id": "openrouter:openai/gpt-4o-mini", "provider_id": "openrouter"}
        return {
            "provider_status": self.provider_status(provider_key or "openrouter"),
            "selection": self.model_policy_status(),
            "provider_selection": {"rejected_candidates": []},
            "candidate": candidate,
        }

    def setup_status(self) -> dict[str, object]:
        self.calls.append(("setup_status", None))
        if self.openrouter_configured:
            return {
                "setup_state": "ready",
                "attention_kind": None,
                "ready": True,
                "active_provider": self.current_provider,
                "active_model": self.current_model,
                "configured_provider": self.default_provider,
                "configured_model": self.default_model,
                "effective_provider": self.effective_provider,
                "effective_model": self.effective_model,
                "provider_health_status": "ok",
                "provider_health_reason": None,
                "model_health_status": "ok",
                "qualification_reason": "Configured default is healthy and ready.",
                "degraded_reason": None,
                "local_installed_models": [],
                "other_local_models": [],
                "source": "fake",
            }
        return {
            "setup_state": "unavailable",
            "attention_kind": None,
            "ready": False,
            "active_provider": None,
            "active_model": None,
            "configured_provider": None,
            "configured_model": None,
            "effective_provider": None,
            "effective_model": None,
            "provider_health_status": "unknown",
            "provider_health_reason": None,
            "model_health_status": "unknown",
            "qualification_reason": None,
            "degraded_reason": None,
            "local_installed_models": [],
            "other_local_models": [],
            "source": "fake",
        }

    def configure_openrouter(self, api_key: str | None, options: dict[str, object] | None = None) -> tuple[bool, dict[str, object]]:
        self.calls.append(("configure_openrouter", {"api_key": api_key, "options": dict(options or {})}))
        if not str(api_key or "").strip():
            return False, {
                "ok": False,
                "error": "missing_api_key",
                "error_kind": "missing_api_key",
                "message": self._openrouter_ready_message,
            }
        self.openrouter_configured = True
        self.openrouter_secret_present = True
        self.openrouter_health_status = "ok"
        self.current_provider = "openrouter"
        self.current_model = "openrouter:openai/gpt-4o-mini"
        self.effective_provider = "openrouter"
        self.effective_model = self.current_model
        self._current_ready = True
        return True, {
            "ok": True,
            "provider": "openrouter",
            "model_id": self.current_model,
            "message": "OpenRouter is ready.",
        }

    def set_default_chat_model(self, model_id: str, provider_id: str | None = None) -> tuple[bool, dict[str, object]]:
        self.calls.append(("set_default_chat_model", {"model_id": model_id, "provider_id": provider_id}))
        self.current_model = str(model_id or "").strip() or self.current_model
        if provider_id:
            self.current_provider = str(provider_id).strip().lower() or self.current_provider
        else:
            self.current_provider = self.current_model.split(":", 1)[0] if ":" in self.current_model else self.current_provider
        self.effective_provider = self.current_provider
        self.effective_model = self.current_model
        return True, {
            "ok": True,
            "provider": self.current_provider,
            "model_id": self.current_model,
            "message": f"Now using {self.current_model} for chat.",
        }

    def set_confirmed_chat_model_target(self, model_id: str, provider_id: str | None = None) -> tuple[bool, dict[str, object]]:
        self.calls.append(("set_confirmed_chat_model_target", {"model_id": model_id, "provider_id": provider_id}))
        return self.set_default_chat_model(model_id, provider_id)

    def set_temporary_chat_model_target(self, model_id: str, provider_id: str | None = None) -> tuple[bool, dict[str, object]]:
        self.calls.append(("set_temporary_chat_model_target", {"model_id": model_id, "provider_id": provider_id}))
        self.current_model = str(model_id or "").strip() or self.current_model
        if provider_id:
            self.current_provider = str(provider_id).strip().lower() or self.current_provider
        else:
            self.current_provider = self.current_model.split(":", 1)[0] if ":" in self.current_model else self.current_provider
        return True, {
            "ok": True,
            "provider": self.current_provider,
            "model_id": self.current_model,
            "message": f"Temporarily switched to {self.current_model}.",
        }

    def acquire_chat_model_target(self, model_id: str, provider_id: str | None = None) -> tuple[bool, dict[str, object]]:
        self.calls.append(("acquire_chat_model_target", {"model_id": model_id, "provider_id": provider_id}))
        return True, {
            "ok": True,
            "provider": str(provider_id or "").strip().lower() or "ollama",
            "model_id": str(model_id or "").strip(),
            "message": f"Started acquiring {model_id} through the canonical model manager.",
        }

    def get_skill_governance_status(self, skill_id: str | None) -> dict[str, object]:
        self.calls.append(("get_skill_governance_status", skill_id))
        normalized = str(skill_id or "").strip().lower()
        if not normalized:
            return {"needs_skill_id": True}
        return {
            "skill": {
                "skill_id": normalized,
                "requested_execution_mode": "in_process",
                "allowed": normalized != "experimental_pack",
                "requires_user_approval": normalized == "experimental_pack",
            },
            "needs_skill_id": False,
        }

    def list_managed_adapters(self) -> dict[str, object]:
        self.calls.append(("list_managed_adapters", None))
        return {
            "managed_adapters": [
                {
                    "adapter_id": "telegram",
                    "approved": True,
                    "enabled": True,
                }
            ],
            "active_adapters": [{"adapter_id": "telegram"}],
        }

    def list_background_tasks(self) -> dict[str, object]:
        self.calls.append(("list_background_tasks", None))
        return {"background_tasks": [], "active_tasks": []}

    def list_governance_blocks(self) -> dict[str, object]:
        self.calls.append(("list_governance_blocks", None))
        return {"blocked_skills": []}

    def list_pending_governance_requests(self) -> dict[str, object]:
        self.calls.append(("list_pending_governance_requests", None))
        return {"pending_skills": [], "pending_adapters": [], "pending_background_tasks": []}

    def get_managed_adapter_status(self, adapter_id: str | None) -> dict[str, object]:
        self.calls.append(("get_managed_adapter_status", adapter_id))
        normalized = str(adapter_id or "").strip().lower()
        if normalized == "telegram":
            return {
                "adapter": {
                    "adapter_id": "telegram",
                    "reason": "Telegram bridge is enabled for chat delivery.",
                    "requested_by": "runtime",
                    "owner": "agent",
                }
            }
        return {"adapter": {}}


class TestBehavioralEvalBattery(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.log_path = os.path.join(self.tmpdir.name, "events.log")
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))
        self.schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        self.db = MemoryDB(self.db_path)
        self.db.init_schema(self.schema_path)
        self.truth = _BehavioralRuntimeTruth()
        self.orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
            runtime_truth_service=self.truth,
        )
        self.orchestrator._memory_runtime.set_current_topic("user1", topic="behavioral eval battery")
        self.orchestrator._memory_runtime.record_user_request("user1", "build a behavioral eval battery")
        self.orchestrator._memory_runtime.record_agent_action(
            "user1",
            "assembled transcript-style eval scenarios",
            action_kind="evaluation",
        )
        self.db.set_preference("response_style", "concise")
        self.db.set_preference("show_confidence", "on")
        self.db.add_open_loop("finish behavioral eval battery", due_date="2026-04-09", priority=2)

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def _chat(self, text: str, *, user_id: str = "user1") -> object:
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            return self.orchestrator.handle_message(text, user_id)

    def _telegram(self, text: str, *, chat_id: str = "42") -> dict[str, object]:
        def _proxy(payload: dict[str, object]) -> dict[str, object]:
            message = str((payload.get("messages") or [{}])[0].get("content") or "")
            user_id = str(payload.get("user_id") or f"telegram:{chat_id}")
            response = self._chat(message, user_id=user_id)
            return {
                "ok": True,
                "assistant": {"content": str(getattr(response, "text", ""))},
                "meta": dict(getattr(response, "data", {}) or {}),
            }

        return handle_telegram_text(
            text=text,
            chat_id=chat_id,
            trace_id="trace-test",
            runtime=None,
            orchestrator=self.orchestrator,
            fetch_local_api_chat_json=_proxy,
        )

    def test_system_and_hardware_transcripts_answer_first(self) -> None:
        ram_prompt = "what do i have for ram and vram right now?"
        self.assertEqual("operational_status", classify_runtime_chat_route(ram_prompt).get("route"))
        self.assertEqual("operational_observe", classify_runtime_chat_route(ram_prompt).get("kind"))

        with patch("agent.orchestrator.can_run_nl_skill", return_value=(True, None)), patch(
            "agent.nl_router.select_observe_skills",
            return_value=[
                {"skill": "hardware_report", "function": "hardware_report"},
                {"skill": "resource_governor", "function": "resource_report"},
            ],
        ):
            self.orchestrator.skills["hardware_report"].functions["hardware_report"].handler = lambda ctx, user_id=None: {
                "status": "ok",
                "text": "You have 64 GiB of RAM and 12 GiB of VRAM available right now.",
                "payload": {
                    "cpu_model": "AMD Ryzen 9 7900X",
                    "memory": {"total_bytes": 64 * 1024 * 1024 * 1024, "used_bytes": 22 * 1024 * 1024 * 1024, "available_bytes": 42 * 1024 * 1024 * 1024},
                    "gpu": {"available": True, "gpus": [{"name": "NVIDIA RTX 4080", "memory_total_mb": 12288, "memory_used_mb": 4096, "utilization_gpu_pct": 17.0, "temperature_c": 54}]},
                    "disk": [],
                    "services": {},
                    "network": {},
                },
                "cards_payload": {
                    "cards": [{"title": "Hardware inventory", "lines": ["CPU: AMD Ryzen 9 7900X", "RAM: 64 GiB total", "VRAM: 12 GiB available"], "severity": "ok"}],
                    "raw_available": True,
                    "summary": "You have 64 GiB of RAM and 12 GiB of VRAM available right now.",
                    "confidence": 1.0,
                    "next_questions": ["What is using memory right now?", "Is anything eating my CPU?"],
                },
            }
            self.orchestrator.skills["resource_governor"].functions["resource_report"].handler = lambda ctx, user_id=None: {
                "status": "ok",
                "text": "CPU load 1m 0.42; memory 41.0% used",
                "payload": {"loads": {"1m": 0.42}, "memory": {"used": 22, "total": 64}},
                "cards_payload": {
                    "cards": [{"title": "Live machine stats", "lines": ["CPU load 1m=0.42", "Memory in use: 41%"], "severity": "ok"}],
                    "raw_available": True,
                    "summary": "CPU load 1m 0.42; memory 41.0% used",
                    "confidence": 1.0,
                    "next_questions": ["What is using memory right now?"],
                },
            }
            ram_response = self._chat(ram_prompt)

        ram_first_line = _first_line(str(getattr(ram_response, "text", "")))
        self.assertTrue(
            ram_first_line.startswith("You have 64 GiB of RAM and 12 GiB of VRAM available right now."),
            ram_first_line,
        )
        self.assertNotIn("System health", ram_first_line)
        self.assertNotIn("{", ram_first_line)
        self.assertNotIn("PID", ram_first_line)
        self.assertNotIn("process", ram_first_line.lower())
        ram_lines = str(getattr(ram_response, "text", "")).splitlines()
        self.assertGreaterEqual(len(ram_lines), 4, msg=str(getattr(ram_response, "text", "")))
        self.assertEqual("", ram_lines[1], msg=str(getattr(ram_response, "text", "")))
        self.assertEqual("*Hardware inventory*", ram_lines[2], msg=str(getattr(ram_response, "text", "")))
        self.assertIn("Follow-ups:", str(getattr(ram_response, "text", "")))

        cpu_prompt = "is anything eating my cpu?"
        self.assertEqual("operational_status", classify_runtime_chat_route(cpu_prompt).get("route"))
        with patch("agent.orchestrator.can_run_nl_skill", return_value=(True, None)), patch(
            "agent.nl_router.select_observe_skills",
            return_value=[{"skill": "resource_governor", "function": "resource_report"}],
        ):
            self.orchestrator.skills["resource_governor"].functions["resource_report"].handler = lambda ctx, user_id=None: {
                "status": "ok",
                "text": "CPU load 1m 0.42; memory 41.0% used",
                "payload": {"loads": {"1m": 0.42}, "memory": {"used": 22, "total": 64}},
                "cards_payload": {
                    "cards": [{"title": "Live machine stats", "lines": ["CPU load 1m=0.42", "Memory in use: 41%"], "severity": "ok"}],
                    "raw_available": True,
                    "summary": "CPU load 1m 0.42; memory 41.0% used",
                    "confidence": 1.0,
                    "next_questions": ["What is using memory right now?"],
                },
            }
            cpu_response = self._chat(cpu_prompt)

        cpu_first_line = _first_line(str(getattr(cpu_response, "text", "")))
        self.assertTrue(cpu_first_line.startswith("CPU load 1m 0.42; memory 34.4% used"), cpu_first_line)
        self.assertNotIn("System health", cpu_first_line)
        self.assertNotIn("PID", cpu_first_line)
        self.assertNotIn("process", cpu_first_line.lower())

        no_vram_prompt = "what do i have for ram and vram right now?"
        with patch("agent.orchestrator.can_run_nl_skill", return_value=(True, None)), patch(
            "agent.nl_router.select_observe_skills",
            return_value=[{"skill": "hardware_report", "function": "hardware_report"}],
        ):
            self.orchestrator.skills["hardware_report"].functions["hardware_report"].handler = lambda ctx, user_id=None: {
                "status": "ok",
                "text": "You have 64 GiB of RAM. VRAM is unavailable right now.",
                "payload": {
                    "cpu_model": "AMD Ryzen 9 7900X",
                    "memory": {"total_bytes": 64 * 1024 * 1024 * 1024, "used_bytes": 22 * 1024 * 1024 * 1024, "available_bytes": 42 * 1024 * 1024 * 1024},
                    "gpu": {"available": False, "gpus": []},
                    "disk": [],
                    "services": {},
                    "network": {},
                },
                "cards_payload": {
                    "cards": [{"title": "Hardware inventory", "lines": ["CPU: AMD Ryzen 9 7900X", "RAM: 64 GiB total", "VRAM: unavailable"], "severity": "warn"}],
                    "raw_available": True,
                    "summary": "You have 64 GiB of RAM. VRAM is unavailable right now.",
                    "confidence": 1.0,
                    "next_questions": ["What is using memory right now?"],
                },
            }
            no_vram_response = self._chat(no_vram_prompt)

        no_vram_first_line = _first_line(str(getattr(no_vram_response, "text", "")))
        self.assertTrue(
            no_vram_first_line.startswith("You have 64 GiB of RAM. VRAM is unavailable right now."),
            no_vram_first_line,
        )
        self.assertIn("VRAM is unavailable right now", no_vram_first_line)
        self.assertNotIn("PID", no_vram_first_line)
        self.assertNotIn("process", no_vram_first_line.lower())

    def test_memory_transcripts_summarize_saved_context(self) -> None:
        memory_prompt = "what do you remember about me?"
        self.assertEqual("agent_memory", classify_runtime_chat_route(memory_prompt).get("route"))
        response = self._chat(memory_prompt)
        first_line = _first_line(str(getattr(response, "text", "")))
        self.assertIn("Here is the useful memory I have right now.", first_line)
        self.assertIn("you prefer concise replies", str(getattr(response, "text", "")).lower())
        self.assertIn("open loops i am tracking", str(getattr(response, "text", "")).lower())

        working_prompt = "what are we working on?"
        response = self._chat(working_prompt)
        text = str(getattr(response, "text", "")).lower()
        self.assertIn("behavioral eval battery", text)
        self.assertIn("pick up from that context", text)

    def test_model_discovery_transcripts_continue_on_partial_failure(self) -> None:
        discovery_prompt = "there is a brand new tiny Gemma 4 model, can you look into it?"
        self.assertEqual("action_tool", classify_runtime_chat_route(discovery_prompt).get("route"))
        self.assertEqual("model_scout_discovery", classify_runtime_chat_route(discovery_prompt).get("kind"))
        response = self._chat(discovery_prompt)
        text = str(getattr(response, "text", ""))
        self.assertIn("Found 2 model(s) across 3 source(s), but some sources failed.", text)
        self.assertIn("Top matches:", text)
        self.assertEqual("model_discovery_manager", response.data["used_tools"][0])
        self.assertIn("huggingface: hf timeout", text)

        self.truth.discovery_payload = {
            "ok": False,
            "query": None,
            "message": "No discovery sources are enabled. Enable Hugging Face/OpenRouter/Ollama in the registry or point AGENT_MODEL_WATCH_CATALOG_PATH at a snapshot.",
            "models": [],
            "sources": [],
            "debug": {
                "source_registry": ["huggingface", "openrouter", "ollama", "external_snapshots"],
                "source_errors": {},
                "source_counts": {},
                "matched_count": 0,
            },
        }
        disabled_prompt = "look for information on a new model from hugging face"
        self.assertEqual("action_tool", classify_runtime_chat_route(disabled_prompt).get("route"))
        response = self._chat(disabled_prompt)
        text = str(getattr(response, "text", ""))
        self.assertIn("No discovery sources are enabled.", text)
        self.assertIn("Enable Hugging Face/OpenRouter/Ollama", text)

    def test_controlled_mode_and_mutating_flows_require_approval(self) -> None:
        switch_prompt = "switch to ollama:qwen3.5:4b"
        self.assertEqual("model_status", classify_runtime_chat_route(switch_prompt).get("route"))
        self.assertEqual("set_default_model", classify_runtime_chat_route(switch_prompt).get("kind"))
        before = self.truth.current_model
        preview = self._chat(switch_prompt)
        preview_text = str(getattr(preview, "text", ""))
        self.assertIn("Reply yes to proceed or no to cancel.", preview_text)
        self.assertEqual(before, self.truth.current_model)

        confirm = self._chat("yes")
        confirm_text = str(getattr(confirm, "text", ""))
        self.assertIn("Now using ollama:qwen3.5:4b for chat.", confirm_text)
        self.assertEqual("ollama:qwen3.5:4b", self.truth.current_model)

        acquire_prompt = "install ollama:qwen2.5:14b"
        self.assertEqual("action_tool", classify_runtime_chat_route(acquire_prompt).get("route"))
        self.assertEqual("model_acquisition_request", classify_runtime_chat_route(acquire_prompt).get("kind"))
        preview = self._chat(acquire_prompt, user_id="user2")
        preview_text = str(getattr(preview, "text", ""))
        self.assertIn("I need one exact model before I can acquire it.", preview_text)
        self.assertEqual(0, sum(1 for call in self.truth.calls if call[0] == "acquire_chat_model_target"))

        policy_prompt = "what would you need my approval for?"
        self.assertEqual("model_policy_status", classify_runtime_chat_route(policy_prompt).get("route"))
        policy_response = self._chat(policy_prompt)
        policy_text = str(getattr(policy_response, "text", ""))
        self.assertIn("Approval: I still need your approval before I test a model, switch temporarily, make a default change, or switch back.", policy_text)
        self.assertIn("Blocked: remote switching and install/download/import.", policy_text)

    def test_setup_and_runtime_transcripts_are_actionable(self) -> None:
        setup_prompt = "configure openrouter"
        self.assertEqual("setup_flow", classify_runtime_chat_route(setup_prompt).get("route"))
        setup_response = self._chat(setup_prompt)
        setup_text = str(getattr(setup_response, "text", ""))
        self.assertIn("Paste your OpenRouter API key", setup_text)
        self.assertIn("finish the setup", setup_text)

        provider_prompt = "is openrouter configured?"
        self.assertEqual("provider_status", classify_runtime_chat_route(provider_prompt).get("route"))
        provider_response = self._chat(provider_prompt)
        provider_text = str(getattr(provider_response, "text", ""))
        self.assertIn("OpenRouter", provider_text)
        self.assertIn("not set up yet", provider_text.lower())
        self.assertIn("api key", provider_text.lower())

        runtime_prompt = "what is the runtime status?"
        self.assertEqual("runtime_status", classify_runtime_chat_route(runtime_prompt).get("route"))
        runtime_response = self._chat(runtime_prompt)
        runtime_text = str(getattr(runtime_response, "text", ""))
        self.assertIn("The runtime is healthy and ready.", runtime_text)
        self.assertNotIn("{", _first_line(runtime_text))

    def test_telegram_bridge_uses_the_same_user_visible_answer_shape(self) -> None:
        ram_prompt = "what do i have for ram and vram right now?"
        with patch("agent.orchestrator.can_run_nl_skill", return_value=(True, None)), patch(
            "agent.nl_router.select_observe_skills",
            return_value=[
                {"skill": "hardware_report", "function": "hardware_report"},
                {"skill": "resource_governor", "function": "resource_report"},
            ],
        ):
            self.orchestrator.skills["hardware_report"].functions["hardware_report"].handler = lambda ctx, user_id=None: {
                "status": "ok",
                "text": "You have 64 GiB of RAM and 12 GiB of VRAM available right now.",
                "payload": {
                    "cpu_model": "AMD Ryzen 9 7900X",
                    "memory": {"total_bytes": 64 * 1024 * 1024 * 1024, "used_bytes": 22 * 1024 * 1024 * 1024, "available_bytes": 42 * 1024 * 1024 * 1024},
                    "gpu": {"available": True, "gpus": [{"name": "NVIDIA RTX 4080", "memory_total_mb": 12288, "memory_used_mb": 4096, "utilization_gpu_pct": 17.0, "temperature_c": 54}]},
                    "disk": [],
                    "services": {},
                    "network": {},
                },
                "cards_payload": {
                    "cards": [{"title": "Hardware inventory", "lines": ["CPU: AMD Ryzen 9 7900X", "RAM: 64 GiB total", "VRAM: 12 GiB available"], "severity": "ok"}],
                    "raw_available": True,
                    "summary": "You have 64 GiB of RAM and 12 GiB of VRAM available right now.",
                    "confidence": 1.0,
                    "next_questions": ["What is using memory right now?", "Is anything eating my CPU?"],
                },
            }
            self.orchestrator.skills["resource_governor"].functions["resource_report"].handler = lambda ctx, user_id=None: {
                "status": "ok",
                "text": "CPU load 1m 0.42; memory 41.0% used",
                "payload": {"loads": {"1m": 0.42}, "memory": {"used": 22, "total": 64}},
                "cards_payload": {
                    "cards": [{"title": "Live machine stats", "lines": ["CPU load 1m=0.42", "Memory in use: 41%"], "severity": "ok"}],
                    "raw_available": True,
                    "summary": "CPU load 1m 0.42; memory 41.0% used",
                    "confidence": 1.0,
                    "next_questions": ["What is using memory right now?"],
                },
            }
            chat_response = self._chat(ram_prompt)
            telegram_payload = self._telegram(ram_prompt, chat_id="42")

        self.assertTrue(bool(telegram_payload.get("handled", False)))
        self.assertEqual("operational_status", telegram_payload.get("route"))
        self.assertEqual(_first_line(str(getattr(chat_response, "text", ""))), _first_line(str(telegram_payload.get("text") or "")))
        self.assertEqual(chat_response.data.get("used_tools"), telegram_payload.get("used_tools"))
        self.assertEqual(chat_response.data.get("route"), telegram_payload.get("route"))
        self.assertIn("VRAM", str(telegram_payload.get("text") or ""))
        self.assertNotIn("{", _first_line(str(telegram_payload.get("text") or "")))


if __name__ == "__main__":
    unittest.main()
