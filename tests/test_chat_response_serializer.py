from __future__ import annotations

import unittest

from agent.chat_response_serializer import serialize_orchestrator_chat_response
from agent.orchestrator import OrchestratorResponse


class TestChatResponseSerializer(unittest.TestCase):
    def test_serializes_generic_chat_envelope_and_provenance(self) -> None:
        response = OrchestratorResponse(
            "Hello from the orchestrator.",
            {
                "route": "generic_chat",
                "route_reason": "generic_chat",
                "used_runtime_state": True,
                "used_llm": True,
                "used_memory": True,
                "used_tools": ["search"],
                "ok": True,
                "provider": "ollama",
                "model": "ollama:qwen3.5:4b",
                "fallback_used": False,
                "attempts": [{"provider": "ollama", "model": "ollama:qwen3.5:4b"}],
                "duration_ms": 12,
            },
        )

        serialized = serialize_orchestrator_chat_response(
            response,
            source_surface="api",
            user_id="api:session-1",
            thread_id="thread-1",
            autopilot_meta={"status": "idle"},
        )

        self.assertTrue(serialized.ok)
        self.assertEqual("generic_chat", serialized.route)
        self.assertTrue(serialized.generic_fallback_allowed)
        self.assertIsNone(serialized.generic_fallback_reason)
        body = serialized.body
        self.assertTrue(body["ok"])
        self.assertEqual("Hello from the orchestrator.", body["assistant"]["content"])
        self.assertEqual("Hello from the orchestrator.", body["message"])
        meta = body["meta"]
        self.assertEqual("ollama", meta["provider"])
        self.assertEqual("ollama:qwen3.5:4b", meta["model"])
        self.assertEqual("generic_chat", meta["route"])
        self.assertEqual("generic_chat", meta["route_reason"])
        self.assertEqual("api", meta["source_surface"])
        self.assertFalse(meta["fallback_used"])
        self.assertTrue(meta["generic_fallback_used"])
        self.assertTrue(meta["generic_fallback_allowed"])
        self.assertTrue(meta["used_runtime_state"])
        self.assertTrue(meta["used_llm"])
        self.assertTrue(meta["used_memory"])
        self.assertEqual(["search"], meta["used_tools"])
        self.assertEqual({"status": "idle"}, meta["autopilot"])
        self.assertEqual("thread-1", meta["thread_id"])
        self.assertEqual("api:session-1", meta["user_id"])
        self.assertNotIn("setup", body)

    def test_serializes_runtime_payload_into_setup_shape(self) -> None:
        response = OrchestratorResponse(
            "OpenRouter is configured.",
            {
                "route": "provider_status",
                "used_runtime_state": True,
                "used_llm": False,
                "used_memory": False,
                "used_tools": [],
                "ok": True,
                "next_question": "Do you want me to switch to it now?",
                "runtime_payload": {
                    "type": "provider_status",
                    "provider": "openrouter",
                    "configured": True,
                    "healthy": True,
                    "model_id": "openrouter:openai/gpt-4o-mini",
                },
                "selection_policy": {
                    "mode": "prefer_local",
                    "reason": "provider_status_check",
                },
            },
        )

        serialized = serialize_orchestrator_chat_response(
            response,
            source_surface="api",
            user_id="api:session-2",
            thread_id="thread-2",
            autopilot_meta={"status": "off"},
        )

        self.assertTrue(serialized.ok)
        self.assertEqual("provider_status", serialized.route)
        self.assertFalse(serialized.generic_fallback_allowed)
        body = serialized.body
        self.assertEqual("OpenRouter is configured.", body["assistant"]["content"])
        self.assertEqual("provider_status", body["setup"]["type"])
        self.assertEqual("openrouter", body["setup"]["provider"])
        self.assertTrue(body["setup"]["configured"])
        self.assertEqual("openrouter:openai/gpt-4o-mini", body["setup"]["model_id"])
        self.assertEqual("Do you want me to switch to it now?", body["next_question"])
        meta = body["meta"]
        self.assertIsNone(meta["provider"])
        self.assertIsNone(meta["model"])
        self.assertEqual("provider_status", meta["setup_type"])
        self.assertFalse(meta["used_llm"])
        self.assertTrue(meta["used_runtime_state"])
        self.assertEqual(
            {
                "mode": "prefer_local",
                "reason": "provider_status_check",
            },
            meta["selection_policy"],
        )


if __name__ == "__main__":
    unittest.main()
