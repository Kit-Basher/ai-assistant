from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import load_config
from agent.orchestrator import OrchestratorResponse


def _config(registry_path: str, db_path: str):
    cfg = load_config(require_telegram_token=False)
    return replace(
        cfg,
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        llm_registry_path=registry_path,
        llm_health_state_path=os.path.join(os.path.dirname(db_path), "llm_health_state.json"),
        llm_automation_enabled=False,
        telegram_enabled=False,
    )


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.status_code = 0
        self.content_type = ""
        self.body = b""

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        self.status_code = status
        self.content_type = "application/json"
        self.body = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        cache_control: str | None = None,
    ) -> None:
        _ = cache_control
        self.status_code = status
        self.content_type = content_type
        self.body = body


class TestRuntimeEvents(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _runtime(self) -> AgentRuntime:
        return AgentRuntime(_config(self.registry_path, self.db_path))

    @staticmethod
    def _telegram_payload() -> dict[str, object]:
        return {
            "ok": True,
            "enabled": False,
            "configured": False,
            "token_source": "none",
            "state": "disabled_optional",
            "effective_state": "disabled_optional",
            "config_source": "default",
            "config_source_path": None,
            "service_installed": False,
            "service_active": False,
            "service_enabled": False,
            "lock_present": False,
            "lock_live": False,
            "lock_stale": False,
            "lock_path": None,
            "lock_pid": None,
            "next_action": "No action needed.",
        }

    @staticmethod
    def _seed_ok_health(runtime: AgentRuntime, *, provider_id: str, model_id: str) -> None:
        provider_key = str(provider_id).strip().lower()
        runtime._health_monitor.state["providers"] = {  # type: ignore[attr-defined]
            provider_key: {"status": "ok", "last_checked_at": 123}
        }
        runtime._health_monitor.state["models"] = {  # type: ignore[attr-defined]
            model_id: {"provider_id": provider_key, "status": "ok", "last_checked_at": 123}
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

    def test_runtime_phase_change_emits_event(self) -> None:
        runtime = self._runtime()
        runtime.add_provider_model(
            "ollama",
            {"model": "qwen2.5:7b-instruct", "capabilities": ["chat"], "available": True},
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._seed_ok_health(runtime, provider_id="ollama", model_id="ollama:qwen2.5:7b-instruct")
        runtime.startup_phase = "warming"
        with runtime._startup_warmup_lock:
            runtime._startup_warmup_remaining = ["router_reload"]
        with patch.object(runtime, "telegram_status", return_value=self._telegram_payload()):
            runtime.ready_status()
            runtime.startup_phase = "ready"
            with runtime._startup_warmup_lock:
                runtime._startup_warmup_remaining = []
            runtime.ready_status()
        events = runtime.runtime_event_history(limit=10)["events"]
        phase_events = [row for row in events if row.get("event") == "runtime_phase_change"]
        self.assertTrue(any(row.get("phase_to") == "warmup" for row in phase_events))
        self.assertTrue(any(row.get("phase_from") == "warmup" and row.get("phase_to") in {"recovering", "ready"} for row in phase_events))

    def test_provider_switch_and_default_model_change_emit_events(self) -> None:
        runtime = self._runtime()
        runtime.add_provider_model(
            "ollama",
            {"model": "qwen2.5:7b-instruct", "capabilities": ["chat"], "available": True},
        )
        runtime.add_provider(
            {
                "id": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "enabled": True,
            }
        )
        runtime.add_provider_model(
            "openrouter",
            {"model": "openai/gpt-4o-mini", "capabilities": ["chat"], "available": True},
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        runtime.set_default_chat_model("openrouter:openai/gpt-4o-mini")
        events = runtime.runtime_event_history(limit=10)["events"]
        self.assertTrue(
            any(
                row.get("event") == "provider_switch"
                and row.get("old_provider") == "ollama"
                and row.get("new_provider") == "openrouter"
                for row in events
            )
        )
        self.assertTrue(
            any(
                row.get("event") == "default_model_change"
                and row.get("new_model") == "openrouter:openai/gpt-4o-mini"
                for row in events
            )
        )

    def test_chat_request_emits_start_and_end_events(self) -> None:
        runtime = self._runtime()
        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            runtime,
            "orchestrator",
        ) as orchestrator_mock:
            orchestrator_mock.return_value.handle_message.return_value = OrchestratorResponse(
                "hello",
                {
                    "ok": True,
                    "route": "generic_chat",
                    "used_runtime_state": True,
                    "used_llm": False,
                    "used_memory": False,
                    "used_tools": [],
                    "provider": "ollama",
                    "model": "ollama:qwen2.5:7b-instruct",
                },
            )
            ok, _body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "source_surface": "api",
                }
            )
        self.assertTrue(ok)
        events = runtime.runtime_event_history(limit=10)["events"]
        start_event = next(row for row in events if row.get("event") == "chat_request_start")
        end_event = next(row for row in events if row.get("event") == "chat_request_end")
        self.assertEqual("api", start_event.get("source"))
        self.assertEqual(start_event.get("request_id"), end_event.get("request_id"))
        self.assertEqual("ok", end_event.get("result"))
        self.assertEqual("ollama:qwen2.5:7b-instruct", end_event.get("model_selected"))

    def test_runtime_history_endpoint_returns_recent_events(self) -> None:
        runtime = self._runtime()
        runtime._runtime_events.log_runtime_event("custom_event", detail="value")
        handler = _HandlerForTest(runtime, "/runtime/history")
        handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(200, handler.status_code)
        self.assertTrue(bool(payload["ok"]))
        self.assertEqual("custom_event", payload["events"][-1]["event"])

    def test_runtime_event_history_caps_at_max_entries(self) -> None:
        runtime = self._runtime()
        for index in range(150):
            runtime._runtime_events.log_runtime_event("custom_event", sequence=index)
        events = runtime.runtime_event_history(limit=200)["events"]
        self.assertEqual(100, len(events))
        self.assertEqual(50, events[0]["sequence"])
        self.assertEqual(149, events[-1]["sequence"])

    def test_provider_health_transition_event_is_recorded(self) -> None:
        runtime = self._runtime()
        runtime._health_monitor.state["providers"] = {  # type: ignore[attr-defined]
            "ollama": {"status": "down", "last_checked_at": 100}
        }
        runtime._health_monitor.state["models"] = {}  # type: ignore[attr-defined]
        runtime._record_authoritative_provider_success("ollama", None)  # type: ignore[attr-defined]
        events = runtime.runtime_event_history(limit=10)["events"]
        self.assertTrue(
            any(
                row.get("event") == "provider_health_transition"
                and row.get("provider") == "ollama"
                and row.get("old_status") == "down"
                and row.get("new_status") == "ok"
                for row in events
            )
        )


if __name__ == "__main__":
    unittest.main()
