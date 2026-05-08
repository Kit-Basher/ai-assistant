from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent.api_server import AgentRuntime
from tests.test_assistant_behavior_release_gate import _MemoryHandlerForTest, _assistant_text, _config


class TestChatBehaviorAudit(unittest.TestCase):
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

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _post_chat(self, prompt: str) -> tuple[int, dict[str, Any], str]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "session_id": "behavior-audit-session",
            "thread_id": "behavior-audit-thread",
            "source_surface": "webui",
            "purpose": "chat",
            "task_type": "chat",
            "trace_id": f"behavior-audit-{prompt[:8]}",
        }
        handler = _MemoryHandlerForTest(self.runtime, "/chat", payload)
        handler.do_POST()
        raw = handler.body.decode("utf-8", errors="replace")
        body = json.loads(raw) if raw.strip().startswith("{") else {}
        if not isinstance(body, dict):
            body = {}
        return int(handler.status_code), body, _assistant_text(body)

    def _assert_grounded_reply(self, prompt: str) -> tuple[int, dict[str, Any], str]:
        status, body, text = self._post_chat(prompt)
        self.assertIn(status, {200, 400}, prompt)
        self.assertTrue(text.strip(), prompt)
        self.assertEqual(text, str(body.get("message") or "").strip(), prompt)
        lowered = text.lower()
        for forbidden in (
            "runtime_payload",
            "selection_policy",
            "trace_id:",
            "source_surface:",
            "thread_id:",
            "read-only guard",
            "nl path refused",
        ):
            self.assertNotIn(forbidden, lowered, prompt)
        return status, body, text

    def test_dumb_user_prompts_stay_on_grounded_chat_path(self) -> None:
        expectations = {
            "what model am i using": ("model_status", ("model", "configured")),
            "is memory on": ("agent_memory", ("memory", "separate from system ram")),
            "why arent you working": ("runtime_status", ("chat target", "runtime")),
            "what can you do": ("assistant_capabilities", ("system inspection", "local memory")),
            "install a skill that lets you browse": ("action_tool", ("browser", "preview")),
            "fix yourself": ("runtime_status", ("diagnostics", "code changes")),
            "use the best local model": ("model_status", ("local", "model")),
            "do you remember what we were doing": ("agent_memory", ("saved", "runtime context")),
            "my computer is slow": ("operational_status", ("ram", "load")),
            "open the app": ("assistant_capabilities", ("personal agent", "127.0.0.1")),
        }

        for prompt, (expected_route, required_phrases) in expectations.items():
            with self.subTest(prompt=prompt):
                _status, body, text = self._assert_grounded_reply(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertEqual(expected_route, meta.get("route"))
                lowered = text.lower()
                for phrase in required_phrases:
                    self.assertIn(phrase, lowered)
                if prompt == "install a skill that lets you browse":
                    self.assertNotIn("which model do you want me to acquire", lowered)


if __name__ == "__main__":
    unittest.main()
