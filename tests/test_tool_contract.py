from __future__ import annotations

import unittest

from agent.tool_contract import (
    normalize_tool_request,
    supported_tools,
    tool_request_to_public_summary,
    validate_tool_request,
)


class TestToolContract(unittest.TestCase):
    def test_supported_tools_allowlist_is_stable(self) -> None:
        self.assertEqual(
            ("brief", "doctor", "health", "observe_now", "observe_system_health", "status"),
            supported_tools(),
        )

    def test_normalize_tool_request_clamps_confidence_and_sorts_args(self) -> None:
        normalized = normalize_tool_request(
            {
                "tool": "STATUS",
                "args": {"z": 1, "a": 2},
                "reason": "  check now  ",
                "confidence": 2.0,
            }
        )
        self.assertEqual("status", normalized["tool"])
        self.assertEqual(["a", "z"], list(normalized["args"].keys()))
        self.assertEqual(1.0, normalized["confidence"])
        self.assertTrue(normalized["read_only"])

    def test_validate_rejects_unsupported_tool(self) -> None:
        ok, normalized, error_code = validate_tool_request({"tool": "shutdown_all"})
        self.assertFalse(ok)
        self.assertEqual("shutdown_all", normalized["tool"])
        self.assertEqual("tool_unsupported", error_code)

    def test_validate_enforces_expected_read_only_setting(self) -> None:
        ok, normalized, error_code = validate_tool_request(
            {"tool": "observe_now", "read_only": True, "args": {}}
        )
        self.assertTrue(ok)
        self.assertIsNone(error_code)
        self.assertFalse(normalized["read_only"])

    def test_public_summary_is_deterministic(self) -> None:
        summary = tool_request_to_public_summary(
            {"tool": "health", "args": {"x": 1}, "reason": "check", "confidence": 0.3333}
        )
        self.assertEqual(
            "tool=health read_only=true confidence=0.33 args=x reason=check",
            summary,
        )


if __name__ == "__main__":
    unittest.main()
