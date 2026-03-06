from __future__ import annotations

import unittest

from agent.tool_executor import ToolExecutor


class TestToolExecutor(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def _emit(self, event: str, payload: dict[str, object]) -> None:
        self.events.append((event, payload))

    def test_read_only_tool_allowed_in_degraded_mode(self) -> None:
        def _status_handler(request: dict[str, object], user_id: str) -> dict[str, object]:
            self.assertEqual("status", request.get("tool"))
            self.assertEqual("u1", user_id)
            return {"ok": True, "user_text": "STATUS_OK", "data": {"state": "ok"}}

        executor = ToolExecutor(
            handlers={"status": _status_handler},
            emit_log=self._emit,
            component="test.executor",
        )
        result = executor.execute(
            request={"tool": "status", "reason": "check"},
            user_id="u1",
            surface="telegram",
            runtime_mode="DEGRADED",
            enable_writes=False,
            safe_mode=True,
        )
        self.assertTrue(result["ok"])
        self.assertEqual("status", result["tool"])
        self.assertEqual("STATUS_OK", result["user_text"])
        self.assertEqual({"state": "ok"}, result["data"])
        self.assertTrue(str(result.get("trace_id") or "").startswith("tool-"))
        self.assertIsNone(result.get("error_code"))
        self.assertEqual(["tool.request", "tool.decision", "tool.execute", "tool.result"], [e[0] for e in self.events])

    def test_write_tool_blocked_when_writes_disabled(self) -> None:
        executor = ToolExecutor(
            handlers={"observe_now": lambda _req, _user: {"ok": True, "user_text": "UNUSED"}},
            emit_log=self._emit,
            component="test.executor",
        )
        result = executor.execute(
            request={"tool": "observe_now", "args": {}},
            user_id="u1",
            surface="api",
            runtime_mode="READY",
            enable_writes=False,
            safe_mode=False,
        )
        self.assertFalse(result["ok"])
        self.assertEqual("observe_now", result["tool"])
        self.assertEqual("writes_disabled", result["error_code"])
        self.assertIn("read-only", str(result.get("next_action") or "").lower())
        self.assertTrue(str(result.get("trace_id") or ""))

    def test_invalid_request_returns_deterministic_error_shape(self) -> None:
        executor = ToolExecutor(
            handlers={"status": lambda _req, _user: {"ok": True, "user_text": "STATUS"}},
            emit_log=self._emit,
            component="test.executor",
        )
        result = executor.execute(
            request={"tool": "not_allowed"},
            user_id="u1",
            surface="cli",
            runtime_mode="READY",
            enable_writes=True,
            safe_mode=False,
            trace_id="trace-fixed",
        )
        self.assertFalse(result["ok"])
        self.assertEqual("trace-fixed", result["trace_id"])
        self.assertEqual("tool_unsupported", result["error_code"])
        self.assertEqual("test.executor", result["component"])
        self.assertTrue(str(result.get("user_text") or "").startswith("Tool request rejected"))
        self.assertTrue(str(result.get("next_action") or ""))


if __name__ == "__main__":
    unittest.main()
