from __future__ import annotations

import os
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent.doctor import (
    DoctorCheck,
    _telegram_enabled_for_doctor,
    _doctor_checks,
    _check_llm_availability,
    _check_secret_store_path,
    _check_telegram_dropin,
    _check_telegram_token,
    _check_write_mode_safe,
    main,
    run_doctor_report,
)


class TestDoctorCLI(unittest.TestCase):
    def test_run_doctor_report_json_shape_and_order(self) -> None:
        checks = [
            DoctorCheck("a", "OK", "alpha"),
            DoctorCheck("b", "WARN", "beta", next_action="Do beta"),
            DoctorCheck("c", "FAIL", "gamma", next_action="Do gamma"),
        ]
        with patch("agent.doctor._doctor_checks", return_value=checks):
            report = run_doctor_report(now_epoch=1_700_000_000)
        payload = report.to_dict()
        self.assertEqual(
            [
                "trace_id",
                "generated_at",
                "summary_status",
                "checks",
                "next_action",
                "fixes_applied",
                "support_bundle_path",
            ],
            list(payload.keys()),
        )
        self.assertTrue(str(payload["trace_id"]).startswith("doctor-1700000000-"))
        self.assertEqual(["a", "b", "c"], [item["check_id"] for item in payload["checks"]])
        self.assertEqual("FAIL", payload["summary_status"])
        self.assertEqual("Do gamma", payload["next_action"])

    def test_secret_store_missing_returns_warn_with_action(self) -> None:
        with patch.dict(os.environ, {"AGENT_SECRET_STORE_PATH": "/tmp/does-not-exist.secrets"}, clear=False):
            check = _check_secret_store_path()
        self.assertEqual("WARN", check.status)
        self.assertIn("secret_store missing", check.detail_short)
        self.assertTrue(bool(check.next_action))

    def test_telegram_dropin_missing_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agent.doctor.Path.home", return_value=Path(tmpdir)):
                check = _check_telegram_dropin()
        self.assertEqual("WARN", check.status)
        self.assertIn("missing drop-in", check.detail_short)
        self.assertEqual("Run: python -m agent doctor --fix", check.next_action)

    def test_llm_unavailable_returns_fail_with_next_action(self) -> None:
        payload = {
            "default_provider": "ollama",
            "default_model": "ollama:qwen2.5:3b-instruct",
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "allow_remote_fallback": False,
            "active_provider_health": {"status": "down"},
            "active_model_health": {"status": "down"},
        }
        with patch("agent.doctor._api_get_json", return_value=(True, payload)):
            check = _check_llm_availability("http://127.0.0.1:8765")
        self.assertEqual("FAIL", check.status)
        self.assertTrue(bool(check.next_action))

    def test_enable_writes_off_is_safe_mode_pass(self) -> None:
        with patch.dict(os.environ, {"ENABLE_WRITES": "0"}, clear=False):
            check = _check_write_mode_safe()
        self.assertEqual("OK", check.status)
        self.assertIn("read-only safe mode", check.detail_short)

    def test_telegram_token_is_redacted(self) -> None:
        token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ12345"
        with patch("agent.doctor.SecretStore.get_secret", return_value=token):
            check = _check_telegram_token(online=False)
        self.assertEqual("OK", check.status)
        self.assertNotIn(token, check.detail_short)
        self.assertIn("...", check.detail_short)

    def test_main_json_output_is_valid_object(self) -> None:
        checks = [DoctorCheck("x", "OK", "ok")]
        output = io.StringIO()
        with patch("agent.doctor._doctor_checks", return_value=checks), redirect_stdout(output):
            code = main(["--json"])
        self.assertEqual(0, code)
        parsed = json.loads(output.getvalue())
        self.assertIsInstance(parsed, dict)
        self.assertIn("trace_id", parsed)
        self.assertIn("checks", parsed)

    def test_doctor_checks_skip_telegram_failures_when_optional_disabled(self) -> None:
        ok = DoctorCheck("ok", "OK", "ok")

        def _systemd_stub(unit: str, check_id: str) -> DoctorCheck:
            _ = unit
            return DoctorCheck(check_id=check_id, status="OK", detail_short="ok")

        with (
            patch("agent.doctor._telegram_enabled_for_doctor", return_value=False),
            patch("agent.doctor._check_python_runtime", return_value=ok),
            patch("agent.doctor._check_repo_readable", return_value=ok),
            patch("agent.doctor._check_secret_store_path", return_value=ok),
            patch("agent.doctor._check_required_dirs", return_value=ok),
            patch("agent.doctor._check_write_mode_safe", return_value=ok),
            patch("agent.doctor._check_systemd_service", side_effect=_systemd_stub),
            patch("agent.doctor._check_llm_availability", return_value=ok),
            patch("agent.doctor._check_logging_to_stdout", return_value=ok),
        ):
            checks = _doctor_checks(repo_root=Path("."), online=False, api_base_url="http://127.0.0.1:8765")
        rows = {row.check_id: row for row in checks}
        self.assertEqual("OK", rows["telegram.dropin"].status)
        self.assertIn("optional", rows["telegram.dropin"].detail_short)
        self.assertEqual("OK", rows["systemd.telegram_service"].status)
        self.assertEqual("OK", rows["process.telegram_pollers"].status)
        self.assertEqual("OK", rows["telegram.token"].status)

    def test_telegram_enabled_for_doctor_uses_live_runtime_state(self) -> None:
        with patch(
            "agent.doctor.get_telegram_runtime_state",
            return_value={"enabled": True},
        ):
            self.assertTrue(_telegram_enabled_for_doctor())
        with patch(
            "agent.doctor.get_telegram_runtime_state",
            return_value={"enabled": False},
        ):
            self.assertFalse(_telegram_enabled_for_doctor())


if __name__ == "__main__":
    unittest.main()
