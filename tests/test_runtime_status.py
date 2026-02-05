import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from agent.diagnostics import CommandResult
from agent.runtime_status import build_runtime_status_report
from memory.db import MemoryDB


class FakeRunner:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, args, _timeout):
        key = tuple(args)
        return self.mapping.get(
            key,
            CommandResult(
                args=list(args),
                stdout="",
                stderr="",
                returncode=0,
                error=None,
                permission_denied=False,
                not_available=False,
            ),
        )


class TestRuntimeStatusReport(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_report_structure_and_redaction(self) -> None:
        runner = FakeRunner(
            {
                ("systemctl", "show", "personal-agent", "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "Result", "-p", "ExecMainStatus", "-p", "ExecMainCode", "-p", "ExecMainExitTimestamp"): CommandResult(
                    args=[
                        "systemctl",
                        "show",
                        "personal-agent",
                        "-p",
                        "ActiveState",
                        "-p",
                        "SubState",
                        "-p",
                        "MainPID",
                        "-p",
                        "Result",
                        "-p",
                        "ExecMainStatus",
                        "-p",
                        "ExecMainCode",
                        "-p",
                        "ExecMainExitTimestamp",
                    ],
                    stdout="ActiveState=active\nMainPID=123\nResult=success\nExecMainStatus=0\nExecMainCode=exited\nExecMainExitTimestamp=n/a\n",
                    stderr="",
                    returncode=0,
                    error=None,
                    permission_denied=False,
                    not_available=False,
                ),
                ("ps", "-eo", "pid,ppid,args"): CommandResult(
                    args=["ps", "-eo", "pid,ppid,args"],
                    stdout="PID PPID COMMAND\n123 1 python -m telegram_adapter.bot\n",
                    stderr="",
                    returncode=0,
                    error=None,
                    permission_denied=False,
                    not_available=False,
                ),
                ("journalctl", "-u", "personal-agent", "-n", "50", "--no-pager"): CommandResult(
                    args=["journalctl", "-u", "personal-agent", "-n", "50", "--no-pager"],
                    stdout="line one\nTOKEN=12345:ABCDEF1234567890123456\nOPENAI_API_KEY=sk-test\n",
                    stderr="",
                    returncode=0,
                    error=None,
                    permission_denied=False,
                    not_available=False,
                ),
            }
        )
        report = build_runtime_status_report(self.db, run_command_fn=runner)
        order = [
            "1. Service State",
            "2. Process State",
            "3. Recent Logs",
            "4. Database State",
            "5. Conflicts Detected",
        ]
        indices = [report.index(section) for section in order]
        self.assertEqual(indices, sorted(indices))
        self.assertNotIn("12345:ABCDEF1234567890123456", report)
        self.assertNotIn("sk-test", report)
        self.assertIn("[REDACTED]", report)

    def test_db_counts(self) -> None:
        now = datetime(2026, 2, 4, 12, 0, 0, tzinfo=timezone.utc)
        past_ts = (now - timedelta(minutes=10)).isoformat()
        self.db.add_reminder(past_ts, "alpha")
        self.db.add_reminder(past_ts, "beta")
        self.db._conn.execute("UPDATE reminders SET status = 'sent' WHERE text = 'alpha'")
        self.db._conn.execute("UPDATE reminders SET status = 'failed' WHERE text = 'beta'")
        self.db._conn.execute(
            "INSERT INTO audit_log (created_at, user_id, action_type, action_id, status, details_json, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now.isoformat(), "user", "action", "1", "failed", "{}", None),
        )
        self.db._conn.execute(
            "INSERT INTO audit_log (created_at, user_id, action_type, action_id, status, details_json, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ((now - timedelta(days=2)).isoformat(), "user", "action", "2", "failed", "{}", None),
        )
        self.db._conn.commit()
        report = build_runtime_status_report(self.db, run_command_fn=FakeRunner({}), now=now)
        self.assertIn("reminders: total=2, pending=0, sent=1, failed=1", report)
        self.assertIn("audit_log last_24h: 1", report)

    def test_permission_denied_degrades(self) -> None:
        runner = FakeRunner(
            {
                ("systemctl", "show", "personal-agent", "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "Result", "-p", "ExecMainStatus", "-p", "ExecMainCode", "-p", "ExecMainExitTimestamp"): CommandResult(
                    args=[],
                    stdout="",
                    stderr="permission denied",
                    returncode=1,
                    error=None,
                    permission_denied=True,
                    not_available=True,
                ),
                ("ps", "-eo", "pid,ppid,args"): CommandResult(
                    args=[],
                    stdout="",
                    stderr="permission denied",
                    returncode=1,
                    error=None,
                    permission_denied=True,
                    not_available=True,
                ),
                ("journalctl", "-u", "personal-agent", "-n", "50", "--no-pager"): CommandResult(
                    args=[],
                    stdout="",
                    stderr="permission denied",
                    returncode=1,
                    error=None,
                    permission_denied=True,
                    not_available=True,
                ),
            }
        )
        report = build_runtime_status_report(self.db, run_command_fn=runner)
        self.assertIn("status: not available (permission)", report)
        self.assertIn("Recent Logs\n- not available (permission)", report)


if __name__ == "__main__":
    unittest.main()
