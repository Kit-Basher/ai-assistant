from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from agent.llm.inference_router import route_inference
from agent.public_chat import build_no_llm_public_message


class _FakeChatLLM:
    def __init__(self) -> None:
        self.chat_calls: list[dict[str, object]] = []
        self.config = types.SimpleNamespace(log_path="/tmp/router.log", llm_routing_mode="auto")
        self.registry = types.SimpleNamespace(defaults=types.SimpleNamespace(allow_remote_fallback=True))

    def enabled(self) -> bool:
        return True

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.chat_calls.append(
            {
                "messages": messages,
                "kwargs": kwargs,
            }
        )
        return {
            "ok": True,
            "text": "Code answer",
            "provider": kwargs.get("provider_override"),
            "model": kwargs.get("model_override"),
            "duration_ms": 3,
        }


class TestInferenceRouter(unittest.TestCase):
    def test_route_inference_chat_uses_selector_result_and_normalizes_output(self) -> None:
        llm = _FakeChatLLM()
        with patch(
            "agent.llm.inference_router.build_model_inventory",
            return_value=[
                {
                    "id": "ollama:qwen2.5:7b-instruct",
                    "provider": "ollama",
                    "local": True,
                    "available": True,
                    "healthy": True,
                    "approved": True,
                    "capabilities": ["chat"],
                }
            ],
        ), patch(
            "agent.llm.inference_router.select_model_for_task",
            return_value={
                "selected_model": "ollama:qwen2.5:7b-instruct",
                "provider": "ollama",
                "reason": "healthy+approved+local_first+task=coding",
                "fallbacks": ["ollama:qwen2.5:3b-instruct"],
                "trace_id": "orch-test",
            },
        ):
            result = route_inference(
                llm_client=llm,
                messages=[{"role": "user", "content": "debug this python traceback"}],
                user_text="debug this python traceback",
                task_hint="debug this python traceback",
                purpose="chat",
                trace_id="orch-test",
            )
        self.assertTrue(bool(result.get("ok")))
        self.assertEqual("Code answer", result.get("text"))
        self.assertEqual("coding", result.get("task_type"))
        self.assertEqual("healthy+approved+local_first+task=coding", result.get("selection_reason"))
        self.assertEqual("ollama", result.get("provider"))
        self.assertEqual("ollama:qwen2.5:7b-instruct", result.get("model"))
        self.assertEqual(1, len(llm.chat_calls))
        kwargs = llm.chat_calls[0]["kwargs"] if isinstance(llm.chat_calls[0], dict) else {}
        self.assertEqual("coding", kwargs.get("task_type"))
        self.assertEqual("ollama", kwargs.get("provider_override"))
        self.assertEqual("ollama:qwen2.5:7b-instruct", kwargs.get("model_override"))

    def test_route_inference_chat_returns_normalized_no_model_guidance(self) -> None:
        llm = _FakeChatLLM()
        with patch("agent.llm.inference_router.build_model_inventory", return_value=[]), patch(
            "agent.llm.inference_router.select_model_for_task",
            return_value={
                "selected_model": None,
                "provider": None,
                "reason": "no_suitable_model",
                "fallbacks": [],
                "trace_id": "orch-test",
            },
        ), patch(
            "agent.llm.inference_router.build_install_plan",
            return_value={
                "needed": True,
                "approved": True,
                "plan": [{"action": "ollama.pull_model", "model": "qwen2.5:3b-instruct"}],
                "next_action": "Run: python -m agent llm_install --model ollama:qwen2.5:3b-instruct --approve",
            },
        ):
            result = route_inference(
                llm_client=llm,
                messages=[{"role": "user", "content": "tell me a joke"}],
                user_text="tell me a joke",
                task_hint="tell me a joke",
                purpose="chat",
                trace_id="orch-test",
            )
        self.assertFalse(bool(result.get("ok")))
        self.assertEqual("no_suitable_model", result.get("error_kind"))
        self.assertEqual("I’m not ready to chat yet. Open Setup to finish getting me ready.", result.get("text"))
        self.assertEqual(
            "Run: python -m agent llm_install --model ollama:qwen2.5:3b-instruct --approve",
            result.get("next_action"),
        )
        self.assertEqual([], llm.chat_calls)
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        self.assertIn("selection", data)
        self.assertIn("plan", data)

    def test_route_inference_chat_uses_ready_aware_no_llm_guidance_when_runtime_ready(self) -> None:
        llm = _FakeChatLLM()
        with patch(
            "agent.llm.inference_router.build_model_inventory",
            return_value=[],
        ), patch(
            "agent.llm.inference_router.select_model_for_task",
            return_value={
                "selected_model": None,
                "provider": None,
                "reason": "no_suitable_model",
                "fallbacks": [],
                "trace_id": "orch-test",
            },
        ), patch(
            "agent.llm.inference_router.build_install_plan",
            return_value={
                "needed": True,
                "approved": True,
                "plan": [{"action": "ollama.pull_model", "model": "qwen2.5:3b-instruct"}],
                "next_action": "Run: python -m agent llm_install --model ollama:qwen2.5:3b-instruct --approve",
            },
        ):
            result = route_inference(
                llm_client=llm,
                messages=[{"role": "user", "content": "tell me a joke"}],
                user_text="tell me a joke",
                task_hint="tell me a joke",
                purpose="chat",
                trace_id="orch-test",
                metadata={"runtime_ready": True},
            )
        self.assertFalse(bool(result.get("ok")))
        self.assertNotEqual(build_no_llm_public_message(), result.get("text"))
        self.assertIn("runtime is ready", str(result.get("text") or "").lower())

    def test_route_inference_emits_selection_and_provider_timing_events(self) -> None:
        llm = _FakeChatLLM()
        events: list[tuple[str, dict[str, object]]] = []

        def _capture(log_path: str | None, event_name: str, payload: dict[str, object]) -> None:
            _ = log_path
            events.append((event_name, dict(payload)))

        with patch(
            "agent.llm.inference_router.build_model_inventory",
            return_value=[
                {
                    "id": "ollama:qwen2.5:3b-instruct",
                    "provider": "ollama",
                    "local": True,
                    "available": True,
                    "healthy": True,
                    "approved": True,
                    "capabilities": ["chat"],
                }
            ],
        ), patch(
            "agent.llm.inference_router.select_model_for_task",
            return_value={
                "selected_model": "ollama:qwen2.5:3b-instruct",
                "provider": "ollama",
                "reason": "healthy+approved+local_first+task=chat",
                "fallbacks": [],
                "trace_id": "orch-test",
            },
        ), patch("agent.llm.inference_router.log_event", side_effect=_capture):
            result = route_inference(
                llm_client=llm,
                messages=[{"role": "user", "content": "hello"}],
                user_text="hello",
                task_hint="hello",
                purpose="chat",
                trace_id="orch-test",
            )

        self.assertTrue(bool(result.get("ok")))
        event_names = [name for name, _payload in events]
        self.assertIn("llm_routing_selection", event_names)
        self.assertIn("llm_provider_request_start", event_names)
        self.assertIn("llm_provider_request_end", event_names)
        selection_payload = next(payload for name, payload in events if name == "llm_routing_selection")
        provider_start_payload = next(payload for name, payload in events if name == "llm_provider_request_start")
        provider_end_payload = next(payload for name, payload in events if name == "llm_provider_request_end")
        self.assertEqual("ollama:qwen2.5:3b-instruct", selection_payload.get("selected_model"))
        self.assertIsInstance(selection_payload.get("selection_ms"), int)
        self.assertEqual("ollama", provider_start_payload.get("provider"))
        self.assertEqual("ollama:qwen2.5:3b-instruct", provider_start_payload.get("model"))
        self.assertEqual("ollama", provider_end_payload.get("provider"))
        self.assertEqual("ollama:qwen2.5:3b-instruct", provider_end_payload.get("model"))
        self.assertIsInstance(provider_end_payload.get("duration_ms"), int)


if __name__ == "__main__":
    unittest.main()
