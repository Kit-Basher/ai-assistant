from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import load_config


def _config(registry_path: str, db_path: str, **overrides: object):
    cfg = load_config(require_telegram_token=False)
    base = replace(
        cfg,
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        llm_registry_path=registry_path,
        llm_automation_enabled=False,
        telegram_enabled=False,
    )
    return replace(base, **overrides)


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


class TestUIStateEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    @staticmethod
    def _seed_ready_runtime(runtime: AgentRuntime) -> None:
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:3b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:3b-instruct")
        runtime.startup_phase = "ready"
        runtime._health_monitor.state["providers"] = {  # type: ignore[attr-defined]
            "ollama": {"status": "ok", "last_checked_at": 123}
        }
        runtime._health_monitor.state["models"] = {  # type: ignore[attr-defined]
            "ollama:qwen2.5:3b-instruct": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123}
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

    @staticmethod
    def _seed_ui_truth_sources(runtime: AgentRuntime) -> None:
        runtime.ready_status = lambda: {  # type: ignore[method-assign]
            "ok": True,
            "runtime_mode": "READY",
            "runtime_status": {
                "runtime_mode": "READY",
                "summary": "Ready. Using ollama / ollama:qwen2.5:3b-instruct.",
                "next_action": "No action needed.",
                "failure_code": None,
            },
            "message": "Ready. Using ollama / ollama:qwen2.5:3b-instruct.",
            "safe_mode_target": {"enabled": False, "configured_valid": True},
            "blocked": {"blocked": False, "kind": None, "reason": None, "message": None},
        }  # type: ignore[assignment]

        class _TruthServiceStub:
            @staticmethod
            def current_chat_target_status() -> dict[str, object]:
                return {
                    "effective_provider": "ollama",
                    "effective_model": "ollama:qwen2.5:3b-instruct",
                    "provider_health_status": "ok",
                    "health_status": "ok",
                }

        runtime.runtime_truth_service = lambda: _TruthServiceStub()  # type: ignore[method-assign]
        runtime.get_defaults = lambda: {  # type: ignore[method-assign]
            "default_provider": "ollama",
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "routing_mode": "auto",
            "allow_remote_fallback": True,
        }
        runtime.llm_control_mode_status = lambda: {  # type: ignore[method-assign]
            "mode": "safe",
            "mode_label": "SAFE MODE",
            "mode_source": "config_default",
            "allow_remote_switch": True,
            "allow_install_pull": True,
            "forbidden_actions": [],
            "approval_required_actions": [],
        }

    def test_ui_state_returns_compact_truthful_payload(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        self._seed_ready_runtime(runtime)
        self._seed_ui_truth_sources(runtime)
        memory_db = runtime._ensure_memory_db()
        memory_db.set_preference("response_style", "concise")
        memory_db.set_preference("show_confidence", "off")

        payload = runtime.ui_state()

        self.assertTrue(payload["ok"])
        self.assertEqual("ready", payload["runtime"]["status"])
        self.assertEqual("Ready. Using ollama / ollama:qwen2.5:3b-instruct.", payload["runtime"]["summary"])
        self.assertEqual("No action needed.", payload["runtime"]["next_action"])
        self.assertEqual("ollama", payload["model"]["provider"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", payload["model"]["model"])
        self.assertEqual("ollama / ollama:qwen2.5:3b-instruct", payload["model"]["path"])
        self.assertEqual("auto", payload["model"]["routing_mode"])
        self.assertEqual("up", payload["model"]["health"])
        self.assertIsNone(payload["conversation"]["topic"])
        self.assertIsNone(payload["conversation"]["recent_request"])
        self.assertIsNone(payload["conversation"]["open_loop"])
        self.assertFalse(payload["action"]["pending_approval"])
        self.assertIsNone(payload["action"]["blocked_reason"])
        self.assertIsNone(payload["action"]["last_action"])
        self.assertEqual("concise", payload["signals"]["response_style"])
        self.assertFalse(payload["signals"]["confidence_visible"])
        self.assertNotIn("Agent is", payload["runtime"]["summary"])
        self.assertNotIn("OpenAI", payload["runtime"]["summary"])

    def test_state_route_serves_ui_state_payload(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        self._seed_ready_runtime(runtime)
        self._seed_ui_truth_sources(runtime)

        handler = _HandlerForTest(runtime, "/state")
        handler.do_GET()

        self.assertEqual(200, handler.status_code)
        self.assertEqual("application/json", handler.content_type)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual("ready", payload["runtime"]["status"])
        self.assertEqual("ollama", payload["model"]["provider"])
        self.assertEqual("Ready. Using ollama / ollama:qwen2.5:3b-instruct.", payload["runtime"]["summary"])
        self.assertIsNone(payload["conversation"]["topic"])
        self.assertFalse(payload["action"]["pending_approval"])


if __name__ == "__main__":
    unittest.main()
