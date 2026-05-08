from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent.api_server import APIServerHandler, AgentRuntime
from tests.test_assistant_behavior_release_gate import _config


class _ChatHandlerForTest(APIServerHandler):
    def __init__(self, runtime: AgentRuntime, payload: dict[str, Any]) -> None:
        self.runtime = runtime
        self.path = "/chat"
        self.headers = {"Content-Length": "0"}
        self._payload = dict(payload)
        self.status_code = 0
        self.body = b""

    def _read_json(self) -> dict[str, Any]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:  # type: ignore[override]
        self.status_code = status
        self.body = json.dumps(payload, ensure_ascii=True).encode("utf-8")


class TestModelSwitchSemantics(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        self.runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                skills_path=str(Path(__file__).resolve().parents[1] / "skills"),
            )
        )
        for model in ("qwen2.5:7b-instruct", "qwen3.6:35b-a3b", "deepseek-r1:7b"):
            self.runtime.add_provider_model(
                "ollama",
                {
                    "model": model,
                    "capabilities": ["chat"],
                    "quality_rank": 5,
                    "available": True,
                    "max_context_tokens": 32768,
                },
            )
        self.runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:qwen2.5:7b-instruct",
                "allow_remote_fallback": False,
            }
        )
        self.runtime._health_monitor.state = {
            "providers": {"ollama": {"status": "ok", "last_checked_at": 123}},
            "models": {
                "ollama:qwen2.5:7b-instruct": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "ollama:qwen3.6:35b-a3b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "ollama:deepseek-r1:7b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
            },
        }
        self.runtime._router.set_external_health_state(self.runtime._health_monitor.state)  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _chat(self, message: str, *, user_id: str = "web:user", thread_id: str = "thread:a") -> dict[str, Any]:
        payload = {
            "messages": [{"role": "user", "content": message}],
            "source_surface": "webui",
            "user_id": user_id,
            "thread_id": thread_id,
        }
        handler = _ChatHandlerForTest(self.runtime, payload)
        handler.do_POST()
        self.assertEqual(200, handler.status_code, handler.body.decode("utf-8", errors="replace"))
        body = json.loads(handler.body.decode("utf-8"))
        self.assertIsInstance(body, dict)
        return body

    @staticmethod
    def _text(payload: dict[str, Any]) -> str:
        assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
        return str(assistant.get("content") or payload.get("message") or "").strip()

    def test_temporary_switch_is_thread_scoped_and_does_not_change_stored_default(self) -> None:
        preview = self._chat("use ollama:qwen3.6:35b-a3b for this chat session only")
        preview_text = self._text(preview)
        self.assertIn("This does not change your default model", preview_text)
        self.assertNotIn("Default model updated", preview_text)

        confirmed = self._chat("yes")
        confirmed_text = self._text(confirmed)
        self.assertIn("Temporary chat model switched", confirmed_text)
        self.assertIn("This does not change your default model", confirmed_text)
        self.assertNotIn("Default model updated", confirmed_text)

        status = self.runtime.llm_status()
        self.assertEqual("ollama:qwen2.5:7b-instruct", status.get("stored_chat_model"))
        self.assertEqual("ollama:qwen3.6:35b-a3b", status.get("temporary_effective_model"))
        self.assertTrue(bool(status.get("temporary_override_active")))

        active_thread = self._chat("what model am i using", thread_id="thread:a")
        self.assertIn("ollama:qwen3.6:35b-a3b", self._text(active_thread))

        fresh_thread = self._chat("what model am i using", thread_id="thread:fresh")
        self.assertIn("ollama:qwen2.5:7b-instruct", self._text(fresh_thread))
        self.assertNotIn("ollama:qwen3.6:35b-a3b", self._text(fresh_thread))

    def test_make_default_changes_stored_chat_model_only_after_confirmation(self) -> None:
        preview = self._chat("make ollama:deepseek-r1:7b the default", thread_id="thread:default")
        preview_text = self._text(preview)
        self.assertIn("I will make ollama:deepseek-r1:7b the default chat model.", preview_text)
        self.assertEqual("ollama:qwen2.5:7b-instruct", self.runtime.llm_status().get("stored_chat_model"))

        confirmed = self._chat("yes", thread_id="thread:default")
        confirmed_text = self._text(confirmed)
        self.assertIn("Default chat model updated", confirmed_text)
        self.assertIn("Previous default: ollama:qwen2.5:7b-instruct", confirmed_text)
        self.assertIn("New default: ollama:deepseek-r1:7b", confirmed_text)
        self.assertEqual("ollama:deepseek-r1:7b", self.runtime.llm_status().get("stored_chat_model"))


if __name__ == "__main__":
    unittest.main()
