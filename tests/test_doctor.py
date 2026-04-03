import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.doctor import (
    _check_db,
    _check_daily_brief_timer,
    _check_version,
    _check_systemd_units,
    expected_schema_from_version,
)


class _Proc:
    def __init__(self, returncode: int, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


class TestDoctor(unittest.TestCase):
    def test_expected_schema_from_version(self) -> None:
        self.assertEqual(2, expected_schema_from_version("0.2.0"))
        self.assertIsNone(expected_schema_from_version("bad"))

    def test_check_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "x.db")
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '2')")
            conn.commit()
            conn.close()
            result = _check_db(db_path)
            self.assertTrue(result.ok)

    def test_check_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            version_path = os.path.join(tmpdir, "VERSION")
            Path(version_path).write_text("0.2.0\n", encoding="utf-8")
            ok_result = _check_version(version_path, 2)
            self.assertTrue(ok_result.ok)
            bad_result = _check_version(version_path, 3)
            self.assertFalse(bad_result.ok)

    def test_systemd_units_skip_when_unavailable(self) -> None:
        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            if args[0][:2] == ["systemctl", "--version"]:
                return _Proc(1)
            return _Proc(1)

        result = _check_systemd_units(fake_run, "/nonexistent/doctor.db")
        self.assertTrue(result.ok)
        self.assertIn("skipped", result.message)

    def test_systemd_timer_active_service_inactive_with_execmainstatus_ok(self) -> None:
        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            cmd = args[0]
            if cmd[:2] == ["systemctl", "--version"]:
                return _Proc(0, "systemd 253")
            if cmd[:2] == ["systemctl", "cat"]:
                return _Proc(0, "unit")
            if cmd[:2] == ["systemctl", "is-enabled"]:
                if cmd[2].endswith(".timer"):
                    return _Proc(0, "enabled\n")
                return _Proc(0, "disabled\n")
            if cmd[:2] == ["systemctl", "is-active"]:
                if cmd[2].endswith(".timer"):
                    return _Proc(0, "active\n")
                return _Proc(0, "inactive\n")
            if cmd[:3] == ["systemctl", "show", "personal-agent-observe.service"]:
                return _Proc(0, "0\n")
            return _Proc(1)

        result = _check_systemd_units(fake_run, "/nonexistent/doctor.db")
        self.assertTrue(result.ok)

    def test_systemd_fails_when_timer_not_active(self) -> None:
        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            cmd = args[0]
            if cmd[:2] == ["systemctl", "--version"]:
                return _Proc(0, "systemd 253")
            if cmd[:2] == ["systemctl", "cat"]:
                return _Proc(0, "unit")
            if cmd[:2] == ["systemctl", "is-enabled"]:
                return _Proc(0, "enabled\n")
            if cmd[:2] == ["systemctl", "is-active"]:
                if cmd[2].endswith(".timer"):
                    return _Proc(0, "inactive\n")
                return _Proc(0, "inactive\n")
            return _Proc(1)

        result = _check_systemd_units(fake_run, "/nonexistent/doctor.db")
        self.assertFalse(result.ok)

    def test_systemd_missing_unit_fails_in_strict_mode(self) -> None:
        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            cmd = args[0]
            if cmd[:2] == ["systemctl", "--version"]:
                return _Proc(0, "systemd 253")
            if cmd[:2] == ["systemctl", "cat"]:
                if cmd[2] == "personal-agent-observe.service":
                    return _Proc(1)
                return _Proc(0, "unit")
            return _Proc(1)

        with patch.dict(os.environ, {"AGENT_DOCTOR_REQUIRE_SYSTEMD_UNITS": "1"}, clear=False):
            result = _check_systemd_units(fake_run, "/nonexistent/doctor.db")
        self.assertFalse(result.ok)

    def test_daily_brief_timer_passes_when_active(self) -> None:
        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            cmd = args[0]
            if cmd[:2] == ["systemctl", "--version"]:
                return _Proc(0, "systemd 253")
            if cmd[:2] == ["systemctl", "cat"]:
                return _Proc(0, "unit")
            if cmd[:2] == ["systemctl", "is-enabled"]:
                if cmd[2].endswith(".timer"):
                    return _Proc(0, "enabled\n")
                return _Proc(0, "disabled\n")
            if cmd[:2] == ["systemctl", "is-active"]:
                if cmd[2].endswith(".timer"):
                    return _Proc(0, "active\n")
                return _Proc(0, "inactive\n")
            if cmd[:3] == ["systemctl", "show", "personal-agent-daily-brief.service"]:
                return _Proc(0, "0\n")
            return _Proc(1)

        result = _check_daily_brief_timer(fake_run)
        self.assertTrue(result.ok)

    def test_daily_brief_timer_skips_when_missing_by_default(self) -> None:
        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            cmd = args[0]
            if cmd[:2] == ["systemctl", "--version"]:
                return _Proc(0, "systemd 253")
            if cmd[:2] == ["systemctl", "cat"]:
                if cmd[2] == "personal-agent-daily-brief.service":
                    return _Proc(0, "unit")
                return _Proc(1)
            return _Proc(1)

        result = _check_daily_brief_timer(fake_run)
        self.assertTrue(result.ok)
        self.assertIn("skipped", result.message)

    def test_daily_brief_timer_fails_when_missing_in_strict_mode(self) -> None:
        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            cmd = args[0]
            if cmd[:2] == ["systemctl", "--version"]:
                return _Proc(0, "systemd 253")
            if cmd[:2] == ["systemctl", "cat"]:
                if cmd[2] == "personal-agent-daily-brief.service":
                    return _Proc(0, "unit")
                return _Proc(1)
            return _Proc(1)

        with patch.dict(os.environ, {"AGENT_DOCTOR_REQUIRE_SYSTEMD_UNITS": "1"}, clear=False):
            result = _check_daily_brief_timer(fake_run)
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
