from __future__ import annotations

import unittest

from agent.tool_executor import ToolExecutor


class TestToolExecutorConsistency(unittest.TestCase):
    def test_same_read_only_request_has_aligned_semantics_across_surfaces(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []

        def _emit(event: str, payload: dict[str, object]) -> None:
            events.append((event, payload))

        executor = ToolExecutor(
            handlers={"status": lambda _req, _user: {"ok": True, "user_text": "STATUS_OK", "data": {"status": "ok"}}},
            emit_log=_emit,
            component="test.executor",
        )

        results = []
        for surface in ("telegram", "cli", "api"):
            results.append(
                executor.execute(
                    request={"tool": "status", "args": {}, "reason": "consistency_check"},
                    user_id="u1",
                    surface=surface,
                    runtime_mode="READY",
                    enable_writes=False,
                    safe_mode=True,
                    trace_id=f"trace-{surface}",
                )
            )

        for result in results:
            self.assertTrue(result["ok"])
            self.assertEqual("status", result["tool"])
            self.assertEqual("STATUS_OK", result["user_text"])
            self.assertEqual({"status": "ok"}, result["data"])
            self.assertIsNone(result["error_code"])
            self.assertIsNone(result["next_action"])

        decision_events = [payload for event, payload in events if event == "tool.decision"]
        self.assertEqual(3, len(decision_events))
        for payload in decision_events:
            self.assertTrue(bool(payload.get("allowed")))
            self.assertEqual("READY", payload.get("runtime_mode"))


if __name__ == "__main__":
    unittest.main()
