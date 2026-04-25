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
                "source_surface": "api",
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
                "selection_policy": {"mode": "internal"},
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
        self.assertFalse(meta["fallback_used"])
        self.assertTrue(meta["generic_fallback_used"])
        self.assertTrue(meta["generic_fallback_allowed"])
        self.assertTrue(meta["used_runtime_state"])
        self.assertTrue(meta["used_llm"])
        self.assertTrue(meta["used_memory"])
        self.assertEqual(["search"], meta["used_tools"])
        self.assertNotIn("route_reason", meta)
        self.assertNotIn("source_surface", meta)
        self.assertNotIn("selection_policy", meta)
        self.assertNotIn("autopilot", meta)
        self.assertNotIn("thread_id", meta)
        self.assertNotIn("user_id", meta)
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
        self.assertFalse(meta["used_llm"])
        self.assertTrue(meta["used_runtime_state"])
        self.assertNotIn("setup_type", meta)
        self.assertNotIn("runtime_state_failure_reason", meta)
        self.assertNotIn("selection_policy", meta)

    def test_serializes_cards_payload_for_compact_cards(self) -> None:
        response = OrchestratorResponse(
            "*Diagnostics snapshot*\n\n- OS/kernel: Linux\n- Next action: Check NetworkManager",
            {
                "route": "diagnostics_capture",
                "used_runtime_state": False,
                "used_llm": False,
                "used_memory": False,
                "used_tools": ["doctor"],
                "ok": True,
                "runtime_payload": {
                    "type": "diagnostics_capture",
                    "kind": "collect_diagnostics",
                    "status": "warn",
                },
                "cards_payload": {
                    "cards": [
                        {
                            "title": "Diagnostics snapshot",
                            "lines": ["OS/kernel: Linux", "Next action: Check NetworkManager"],
                            "severity": "warn",
                        }
                    ],
                    "raw_available": False,
                    "summary": "Network is not fully up.",
                    "confidence": 1.0,
                    "next_questions": [],
                },
            },
        )

        serialized = serialize_orchestrator_chat_response(
            response,
            source_surface="api",
            user_id="api:session-4",
            thread_id="thread-4",
            autopilot_meta={"status": "off"},
        )

        self.assertEqual("diagnostics_capture", serialized.route)
        self.assertEqual(
            ["doctor"],
            serialized.body["meta"]["used_tools"],
        )
        self.assertIn("*Diagnostics snapshot*", serialized.body["assistant"]["content"])
        self.assertIn("cards_payload", serialized.body)
        self.assertEqual("Diagnostics snapshot", serialized.body["cards_payload"]["cards"][0]["title"])

    def test_serializes_debug_metadata_only_when_requested(self) -> None:
        response = OrchestratorResponse(
            "OpenRouter is configured.",
            {
                "route": "provider_status",
                "route_reason": "provider_status",
                "source_surface": "api",
                "used_runtime_state": True,
                "used_llm": False,
                "used_memory": False,
                "used_tools": [],
                "ok": True,
                "provider": "openrouter",
                "model": "openrouter:openai/gpt-4o-mini",
                "fallback_used": False,
                "attempts": [],
                "duration_ms": 4,
                "runtime_payload": {
                    "type": "provider_status",
                    "reason": "provider_status_check",
                    "provider": "openrouter",
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
            user_id="api:session-3",
            thread_id="thread-3",
            autopilot_meta={"status": "off"},
            include_debug=True,
        )

        meta = serialized.body["meta"]
        self.assertEqual("api", meta["source_surface"])
        self.assertEqual("thread-3", meta["thread_id"])
        self.assertEqual("api:session-3", meta["user_id"])
        self.assertEqual("provider_status", meta["setup_type"])
        self.assertEqual("provider_status_check", meta["runtime_state_failure_reason"])
        self.assertEqual({"mode": "prefer_local", "reason": "provider_status_check"}, meta["selection_policy"])

    def test_non_mapping_response_data_falls_back_to_empty_metadata(self) -> None:
        response = OrchestratorResponse("Still helpful.", [("route", "generic_chat")])  # type: ignore[arg-type]

        serialized = serialize_orchestrator_chat_response(
            response,
            source_surface="telegram",
            user_id="telegram:user-1",
            thread_id="thread-5",
            autopilot_meta={"status": "off"},
        )

        self.assertTrue(serialized.ok)
        self.assertEqual("generic_chat", serialized.route)
        self.assertEqual("Still helpful.", serialized.body["assistant"]["content"])
        meta = serialized.body["meta"]
        self.assertEqual("generic_chat", meta["route"])
        self.assertFalse(meta["used_llm"])


if __name__ == "__main__":
    unittest.main()
