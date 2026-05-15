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

    def test_direct_questions_do_not_fall_into_stale_clarification_context(self) -> None:
        cases = {
            "is the local API healthy": ("runtime_status", ("ready", "chat")),
            "are you actually connected to a model right now": ("runtime_status", ("ready", "chat")),
            "how do i open the web UI": ("assistant_capabilities", ("127.0.0.1", "personal agent")),
            "is setup complete": ("setup_flow", ("setup", "chat")),
            "help me set this up": ("setup_flow", ("setup", "chat")),
            "where were we before": ("agent_memory", ("runtime context", "saved")),
            "this feels broken, what is wrong": ("runtime_status", ("chat target", "runtime")),
        }

        for prompt, (expected_route, required_phrases) in cases.items():
            with self.subTest(prompt=prompt):
                self._post_chat("what model am i using")
                _status, body, text = self._assert_grounded_reply(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertEqual(expected_route, meta.get("route"))
                self.assertNotEqual("assistant_clarification", meta.get("route"))
                self.assertNotIn("i was following", text.lower())
                lowered = text.lower()
                for phrase in required_phrases:
                    self.assertIn(phrase, lowered)

    def test_browser_skill_requests_use_pack_preview_not_apt_or_stale_followup(self) -> None:
        prompts = (
            "install a skill that lets you browse",
            "can you add browser capabilities",
            "what skills can you install for web research",
            "add a capability for reading webpages",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self._post_chat("my computer is slow")
                _status, body, text = self._assert_grounded_reply(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertEqual("action_tool", meta.get("route"))
                self.assertIn("pack_capability_recommendation", meta.get("used_tools") or [])
                lowered = text.lower()
                self.assertIn("browser", lowered)
                self.assertIn("preview", lowered)
                self.assertNotIn("apt-get", lowered)
                self.assertNotIn("install a using", lowered)
                self.assertNotIn("which model do you want me to acquire", lowered)
                self.assertNotIn("likely cause:", lowered)

    def test_open_chat_prompts_after_operational_status_do_not_reuse_stale_context(self) -> None:
        prompts = (
            "help me plan the next hour",
            "explain this project in plain english",
            "give me a concise checklist for testing this app",
            "what should I ask you next",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self._post_chat("my computer is slow")
                _status, body, text = self._assert_grounded_reply(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertNotEqual("assistant_clarification", meta.get("route"))
                self.assertNotEqual("interpretation_followup", meta.get("route"))
                self.assertNotIn("i was following", text.lower())
                self.assertNotIn("likely cause:", text.lower())
                if prompt == "what should I ask you next":
                    self.assertEqual("assistant_capabilities", meta.get("route"))

    def test_resource_prompts_after_operational_status_route_to_operational_status(self) -> None:
        prompts = (
            "is something eating resources",
            "what is using resources",
            "what is eating memory",
            "what is eating cpu",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self._post_chat("my computer is slow")
                _status, body, text = self._assert_grounded_reply(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertEqual("operational_status", meta.get("route"))
                self.assertNotEqual("assistant_clarification", meta.get("route"))
                self.assertNotIn("i was following", text.lower())


if __name__ == "__main__":
    unittest.main()
