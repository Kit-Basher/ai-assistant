from __future__ import annotations

import unittest

from agent.error_response_ux import deterministic_error_message


class TestErrorMessageUX(unittest.TestCase):
    def test_deterministic_error_message_shape(self) -> None:
        text = deterministic_error_message(
            title="❌ LLM provider unavailable",
            trace_id="abcd1234",
            component="llm_client",
            next_action="run `agent doctor`",
        )
        lines = text.splitlines()
        self.assertEqual("❌ LLM provider unavailable", lines[0])
        self.assertEqual("trace_id: abcd1234", lines[1])
        self.assertEqual("component: llm_client", lines[2])
        self.assertEqual("next_action: run `agent doctor`", lines[3])


if __name__ == "__main__":
    unittest.main()

