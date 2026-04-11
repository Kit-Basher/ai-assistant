from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from agent.llm.inference_router import route_inference


class _FakeChatLLM:
    def __init__(self) -> None:
        self.chat_calls: list[dict[str, object]] = []
        self.config = object()
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


if __name__ == "__main__":
    unittest.main()
