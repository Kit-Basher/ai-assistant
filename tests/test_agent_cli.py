from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent import cli


class TestAgentCLI(unittest.TestCase):
    def test_doctor_subcommand_forwards_args(self) -> None:
        with patch("agent.cli.doctor_main", return_value=0) as doctor_main:
            code = cli.main(["doctor", "--json"])
        self.assertEqual(0, code)
        doctor_main.assert_called_once_with(["--json"])

    def test_status_subcommand_uses_ready_payload(self) -> None:
        payload = {
            "ok": True,
            "ready": True,
            "message": "Ready.",
            "telegram": {"state": "running"},
        }
        output = io.StringIO()
        with patch("agent.cli._http_json", return_value=(True, payload)), redirect_stdout(output):
            code = cli.main(["status"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("Agent is ready.", text)
        self.assertIn("telegram: running", text)
        self.assertIn("message: Ready.", text)

    def test_health_subcommand_failure_returns_structured_error(self) -> None:
        output = io.StringIO()
        with patch("agent.cli._http_json", return_value=(False, "boom")), redirect_stdout(output):
            code = cli.main(["health"])
        self.assertEqual(1, code)
        text = output.getvalue()
        self.assertIn("trace_id:", text)
        self.assertIn("component: agent.cli.health", text)
        self.assertIn("next_action: run `agent doctor`", text)

    def test_logs_subcommand_tails_last_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agent.log"
            path.write_text("\n".join(f"line-{idx}" for idx in range(1, 11)) + "\n", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = cli.main(["logs", "--path", str(path), "--lines", "3"])
        self.assertEqual(0, code)
        text = output.getvalue().strip()
        self.assertIn("Showing last 3 lines from", text)
        self.assertTrue(text.endswith("line-8\nline-9\nline-10"))

    def test_version_subcommand_prints_version_and_commit(self) -> None:
        output = io.StringIO()
        with patch("agent.cli._resolve_git_commit", return_value="abc1234"), redirect_stdout(output):
            code = cli.main(["version"])
        self.assertEqual(0, code)
        text = output.getvalue().strip()
        self.assertIn("version=", text)
        self.assertIn("commit=abc1234", text)


if __name__ == "__main__":
    unittest.main()
