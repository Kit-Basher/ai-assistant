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
    DoctorReport,
    _telegram_enabled_for_doctor,
    _doctor_checks,
    _check_llm_availability,
    _check_secret_store_path,
    _check_telegram_dropin,
    _check_telegram_token,
    _check_write_mode_safe,
    _render_text_report,
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

    def test_llm_availability_prefers_ready_embedded_llm_payload(self) -> None:
        def _fetch(url: str, timeout_seconds: float = 0.8) -> tuple[bool, dict[str, object] | str]:
            _ = timeout_seconds
            if url.endswith("/ready"):
                return True, {
                    "ok": True,
                    "llm": {
                        "default_provider": "ollama",
                        "resolved_default_model": "ollama:qwen2.5:3b-instruct",
                        "allow_remote_fallback": False,
                        "active_provider_health": {"status": "ok"},
                        "active_model_health": {"status": "ok"},
                    },
                }
            if url.endswith("/llm/status"):
                raise AssertionError("_check_llm_availability should use /ready llm before /llm/status")
            raise AssertionError(url)

        with patch("agent.doctor._api_get_json", side_effect=_fetch):
            check = _check_llm_availability("http://127.0.0.1:8765")

        self.assertEqual("OK", check.status)
        self.assertIn("provider=ollama", check.detail_short)

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

    def test_collect_diagnostics_writes_redacted_bundle_with_recovery_manifest(self) -> None:
        token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ12345"
        openai_key = "sk-abc12345678901234567890"
        checks = [
            DoctorCheck(
                "telegram.token",
                "WARN",
                f"token={token}",
                next_action=f"Visit https://example.invalid/reset?token={token}",
            )
        ]

        def _fetch(url: str, timeout_seconds: float = 0.8) -> tuple[bool, dict[str, object] | str]:
            _ = timeout_seconds
            if url.endswith("/health"):
                return True, {"ok": True, "authorization": f"Bearer {openai_key}", "query": f"?token={token}"}
            return False, "unavailable"

        with (
            patch("agent.doctor._doctor_checks", return_value=checks),
            patch("agent.doctor._api_get_json", side_effect=_fetch),
            patch(
                "agent.doctor.run_startup_checks",
                side_effect=[
                    {"status": "PASS", "checks": [], "next_action": None},
                    {"status": "WARN", "checks": [], "next_action": "Run: python -m agent doctor"},
                ],
            ),
            patch("agent.doctor.load_config", side_effect=RuntimeError("config missing")),
        ):
            report = run_doctor_report(now_epoch=1_700_000_000, collect_diagnostics=True)

        self.assertTrue(str(report.support_bundle_path or "").strip())
        bundle_dir = Path(str(report.support_bundle_path))
        bundle_path = bundle_dir / "doctor_support_bundle.json"
        summary_path = bundle_dir / "SUMMARY.txt"
        self.assertTrue(bundle_path.is_file())
        self.assertTrue(summary_path.is_file())

        raw = bundle_path.read_text(encoding="utf-8")
        self.assertNotIn(token, raw)
        self.assertNotIn(openai_key, raw)
        self.assertIn("[REDACTED]", raw)
        payload = json.loads(raw)
        self.assertIn("recovery", payload)
        self.assertIn("backup_targets", payload["recovery"])
        self.assertIn("collect_diagnostics", payload["recovery"]["canonical_commands"])

    def test_main_collect_diagnostics_json_returns_bundle_path(self) -> None:
        checks = [DoctorCheck("x", "OK", "ok")]
        output = io.StringIO()
        with patch("agent.doctor._doctor_checks", return_value=checks), redirect_stdout(output):
            code = main(["--json", "--collect-diagnostics"])
        self.assertEqual(0, code)
        parsed = json.loads(output.getvalue())
        self.assertTrue(str(parsed.get("support_bundle_path") or "").strip())
        self.assertTrue((Path(str(parsed["support_bundle_path"])) / "doctor_support_bundle.json").is_file())

    def test_fix_still_generates_support_bundle_path(self) -> None:
        checks = [DoctorCheck("x", "OK", "ok")]
        with (
            patch("agent.doctor._doctor_checks", return_value=checks),
            patch("agent.doctor._apply_safe_fixes", return_value=[]),
        ):
            report = run_doctor_report(now_epoch=1_700_000_000, fix=True)
        self.assertTrue(str(report.support_bundle_path or "").strip())
        self.assertTrue((Path(str(report.support_bundle_path)) / "doctor_support_bundle.json").is_file())

    def test_render_text_report_shows_per_check_next_steps(self) -> None:
        custom = DoctorReport(
            trace_id="doctor-1700000000-1",
            generated_at="2023-11-14T22:13:20+00:00",
            summary_status="WARN",
            checks=[
                DoctorCheck("env.repo", "OK", "repo ok"),
                DoctorCheck("systemd.api_service", "WARN", "service inactive", next_action="Run: systemctl --user restart personal-agent-api.service"),
            ],
            next_action="Run: systemctl --user restart personal-agent-api.service",
            fixes_applied=[],
            support_bundle_path=None,
        )
        rendered = _render_text_report(custom)
        self.assertIn("systemd.api_service", rendered)
        self.assertIn("next: Run: systemctl --user restart personal-agent-api.service", rendered)

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
