from __future__ import annotations

import unittest

from agent.api_server import APIServerHandler
from agent.fallback_ladder import run_with_fallback
from agent.response_envelope import validate_envelope


class TestFlawlessResponseContract(unittest.TestCase):
    def test_envelope_validation_rejects_empty_message(self) -> None:
        with self.assertRaises(ValueError):
            validate_envelope(
                {
                    "ok": True,
                    "intent": "test",
                    "confidence": 0.5,
                    "did_work": True,
                    "message": "",
                    "next_question": None,
                    "actions": [],
                    "errors": [],
                    "trace_id": "trace-1",
                }
            )

    def test_next_question_single_question_mark(self) -> None:
        with self.assertRaises(ValueError):
            validate_envelope(
                {
                    "ok": True,
                    "intent": "test",
                    "confidence": 0.5,
                    "did_work": False,
                    "message": "Need clarification.",
                    "next_question": "One? Two?",
                    "actions": [],
                    "errors": [],
                    "trace_id": "trace-1",
                }
            )

        envelope = validate_envelope(
            {
                "ok": True,
                "intent": "test",
                "confidence": 0.5,
                "did_work": False,
                "message": "Need clarification.",
                "next_question": "Which model should I use?",
                "actions": [],
                "errors": [],
                "trace_id": "trace-2",
            }
        )
        self.assertEqual("Which model should I use?", envelope["next_question"])

    def test_fallback_ladder_turns_exception_into_failure_envelope(self) -> None:
        def _explode() -> dict[str, object]:
            raise RuntimeError("boom")

        envelope = run_with_fallback(
            fn=_explode,
            context={"intent": "test.intent", "trace_id": "trace-3", "actions": []},
        )
        self.assertFalse(envelope["ok"])
        self.assertFalse(envelope["did_work"])
        self.assertEqual("test.intent", envelope["intent"])
        self.assertIn("RuntimeError", envelope["errors"])
        self.assertTrue(envelope["message"].strip())

    def test_fallback_ladder_never_returns_empty_message(self) -> None:
        def _invalid() -> dict[str, object]:
            return {
                "ok": True,
                "intent": "test.intent",
                "confidence": 0.5,
                "did_work": True,
                "message": "   ",
                "next_question": None,
                "actions": [],
                "errors": [],
                "trace_id": "trace-4",
            }

        envelope = run_with_fallback(
            fn=_invalid,
            context={"intent": "test.intent", "trace_id": "trace-4", "actions": []},
        )
        self.assertFalse(envelope["ok"])
        self.assertTrue(envelope["message"].strip())

    def test_api_server_does_not_raise_on_handler_exception(self) -> None:
        class _HandlerForTest(APIServerHandler):
            def __init__(self) -> None:
                self.path = "/chat"
                self.headers = {}
                self.status_code = 0
                self.payload: dict[str, object] = {}

            def _path_parts(self) -> tuple[str, list[str]]:  # type: ignore[override]
                raise RuntimeError("forced-handler-error")

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.payload = payload

        handler = _HandlerForTest()
        handler.do_POST()

        self.assertEqual(500, handler.status_code)
        self.assertEqual(False, handler.payload.get("ok"))
        self.assertTrue(str(handler.payload.get("message") or "").strip())


if __name__ == "__main__":
    unittest.main()
