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
    collect_bluetooth_audio_diagnostics_snapshot,
    collect_diagnostics_snapshot,
    collect_generic_device_fallback_diagnostics_snapshot,
    collect_printer_cups_diagnostics_snapshot,
    collect_storage_disk_diagnostics_snapshot,
    render_bluetooth_audio_diagnostics_snapshot,
    render_diagnostics_snapshot,
    render_generic_device_fallback_diagnostics_snapshot,
    render_printer_cups_diagnostics_snapshot,
    render_storage_disk_diagnostics_snapshot,
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

    def test_collect_diagnostics_snapshot_is_compact_and_redacted(self) -> None:
        uname = "Linux test-host 6.8.0-1 x86_64"
        nmcli = "wlp2s0:wifi:connected:HomeNetwork\nlo:loopback:connected:lo"
        journal = (
            "Apr 19 12:00:00 kernel: PM: suspend entry (deep)\n"
            "Apr 19 12:00:02 kernel: network disconnected after suspend\n"
            "Apr 19 12:00:05 kernel: PM: resume from suspend\n"
        )

        def _run_command(args, timeout_s=2.0):  # type: ignore[no-untyped-def]
            if args[:2] == ["uname", "-a"]:
                return type("R", (), {"stdout": uname, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:2] == ["nmcli", "-t"]:
                return type("R", (), {"stdout": nmcli, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:1] == ["journalctl"]:
                return type("R", (), {"stdout": journal, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            raise AssertionError(f"unexpected command: {args}")

        with patch("agent.doctor.collect_system_health", return_value={"network": {"state": "up", "up_interfaces": ["wlp2s0"], "default_route": True, "dns_configured": True}}):
            snapshot = collect_diagnostics_snapshot(run_command_fn=_run_command)

        self.assertEqual("Linux test-host 6.8.0-1 x86_64", snapshot["os"]["text"])
        self.assertEqual("nmcli", snapshot["network"]["source"])
        self.assertEqual(2, len(snapshot["network"]["nmcli_rows"]))
        self.assertEqual(3, snapshot["suspend_resume"]["match_count"])
        self.assertIn("Suspend/resume logs contain failure markers.", snapshot["summary"]["assessment"])
        text = render_diagnostics_snapshot(snapshot)
        self.assertIn("Diagnostics snapshot", text)
        self.assertIn("OS/kernel:", text)
        self.assertIn("Suspend/resume matches:", text)
        self.assertNotIn("Run: uname", text)
        self.assertNotIn("journalctl", text)

    def test_collect_bluetooth_audio_diagnostics_snapshot_is_compact_and_structured(self) -> None:
        service = (
            "bluetooth.service - Bluetooth service\n"
            "   Loaded: loaded (/usr/lib/systemd/system/bluetooth.service; enabled; preset: enabled)\n"
            "   Active: active (running) since Fri 2026-04-19 12:00:00 UTC; 1min ago\n"
        )
        controller = (
            "Controller AA:BB:CC:DD:EE:FF test-host [default]\n"
            "        Alias: test-host\n"
            "        Powered: yes\n"
            "        Discoverable: no\n"
            "        Pairable: yes\n"
            "        Discovering: no\n"
        )
        devices = "Device 11:22:33:44:55:66 Headphones\nDevice AA:BB:CC:DD:EE:FF Speaker"
        info_1 = (
            "Device 11:22:33:44:55:66 (public)\n"
            "        Name: Headphones\n"
            "        Paired: yes\n"
            "        Connected: no\n"
            "        Trusted: yes\n"
        )
        info_2 = (
            "Device AA:BB:CC:DD:EE:FF (public)\n"
            "        Name: Speaker\n"
            "        Paired: yes\n"
            "        Connected: yes\n"
            "        Trusted: yes\n"
        )
        journal = (
            "Apr 19 12:00:00 kernel: bluetoothd[123]: profiles/audio: disconnect\n"
            "Apr 19 12:00:02 kernel: bluetoothd[123]: reconnect failed\n"
        )

        def _run_command(args, timeout_s=2.0):  # type: ignore[no-untyped-def]
            if args[:3] == ["systemctl", "status", "bluetooth"]:
                return type("R", (), {"stdout": service, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:2] == ["bluetoothctl", "show"]:
                return type("R", (), {"stdout": controller, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:2] == ["bluetoothctl", "paired-devices"]:
                return type("R", (), {"stdout": devices, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:2] == ["bluetoothctl", "info"] and args[2] == "11:22:33:44:55:66":
                return type("R", (), {"stdout": info_1, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:2] == ["bluetoothctl", "info"] and args[2] == "AA:BB:CC:DD:EE:FF":
                return type("R", (), {"stdout": info_2, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:1] == ["journalctl"]:
                return type("R", (), {"stdout": journal, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            raise AssertionError(f"unexpected command: {args}")

        snapshot = collect_bluetooth_audio_diagnostics_snapshot(run_command_fn=_run_command)
        self.assertEqual("bluetooth_audio", snapshot["preset"])
        self.assertEqual("active", snapshot["bluetooth"]["service"]["active_state"])
        self.assertTrue(snapshot["bluetooth"]["controller"]["powered"])
        self.assertEqual(2, snapshot["bluetooth"]["devices"]["paired_count"])
        self.assertEqual(1, snapshot["bluetooth"]["devices"]["connected_count"])
        self.assertEqual(2, snapshot["bluetooth"]["logs"]["match_count"])
        self.assertIn("Recent Bluetooth logs contain failure markers.", snapshot["summary"]["assessment"])
        text = render_bluetooth_audio_diagnostics_snapshot(snapshot)
        self.assertIn("Bluetooth/audio diagnostics", text)
        self.assertIn("Service: active=active", text)
        self.assertIn("Controller: address=AA:BB:CC:DD:EE:FF", text)
        self.assertIn("Devices: paired=2; connected=1", text)
        self.assertIn("Logs: matches=2", text)
        self.assertNotIn("systemctl status bluetooth", text)
        self.assertNotIn("journalctl", text)

    def test_collect_storage_disk_diagnostics_snapshot_is_compact_and_structured(self) -> None:
        df_output = (
            "Filesystem     Type  Size  Used Avail Use% Mounted on\n"
            "/dev/nvme0n1p2 ext4  100G   96G    4G  96% /\n"
            "/dev/nvme0n1p3 ext4  200G  120G   80G  60% /home\n"
            "/dev/nvme0n1p4 ext4   50G   45G    5G  90% /var\n"
        )
        journal = (
            "Apr 19 12:00:00 kernel: app[123]: write failed: No space left on device\n"
            "Apr 19 12:00:01 kernel: app[123]: disk full while saving\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            home_dir = Path(tmpdir) / "home"
            (home_dir / "Downloads").mkdir(parents=True, exist_ok=True)
            (home_dir / "Videos").mkdir(parents=True, exist_ok=True)

            def _run_command(args, timeout_s=2.0):  # type: ignore[no-untyped-def]
                if args[:1] == ["df"]:
                    return type("R", (), {"stdout": df_output, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
                if args[:1] == ["du"] and args[-1] == str(home_dir):
                    return type(
                        "R",
                        (),
                        {
                            "stdout": f"1024 {home_dir / 'Downloads'}\n2048 {home_dir / 'Videos'}\n3072 {home_dir}\n",
                            "permission_denied": False,
                            "not_available": False,
                            "error": None,
                            "returncode": 0,
                        },
                    )()
                if args[:1] == ["du"] and args[-1] == "/var":
                    return type(
                        "R",
                        (),
                        {
                            "stdout": "4096 /var/cache\n2048 /var/log\n6144 /var\n",
                            "permission_denied": False,
                            "not_available": False,
                            "error": None,
                            "returncode": 0,
                        },
                    )()
                if args[:1] == ["du"] and args[-1] == "/tmp":
                    return type(
                        "R",
                        (),
                        {
                            "stdout": "512 /tmp/session\n1024 /tmp\n",
                            "permission_denied": False,
                            "not_available": False,
                            "error": None,
                            "returncode": 0,
                        },
                    )()
                if args[:1] == ["journalctl"]:
                    return type("R", (), {"stdout": journal, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
                raise AssertionError(f"unexpected command: {args}")

            with patch("agent.doctor.Path.home", return_value=home_dir):
                snapshot = collect_storage_disk_diagnostics_snapshot(run_command_fn=_run_command)

        self.assertEqual("storage_disk", snapshot["preset"])
        self.assertEqual(3, len(snapshot["storage"]["filesystems"]["rows"]))
        self.assertEqual(5, snapshot["storage"]["consumers"]["match_count"])
        self.assertEqual(2, snapshot["storage"]["logs"]["match_count"])
        self.assertIn("Recent logs contain disk-full or write-failure markers.", snapshot["summary"]["assessment"])
        text = render_storage_disk_diagnostics_snapshot(snapshot)
        self.assertIn("Storage/disk diagnostics", text)
        self.assertIn("Filesystem:", text)
        self.assertIn("Mount/device:", text)
        self.assertIn("Consumers:", text)
        self.assertIn("Log matches: 2", text)
        self.assertNotIn("journalctl", text)

    def test_collect_printer_cups_diagnostics_snapshot_is_compact_and_structured(self) -> None:
        service = (
            "cups.service - CUPS Scheduler\n"
            "   Loaded: loaded (/usr/lib/systemd/system/cups.service; enabled; preset: enabled)\n"
            "   Active: active (running) since Fri 2026-04-19 12:00:00 UTC; 1min ago\n"
        )
        printers = (
            "printer HP_LaserJet is idle. enabled since Fri 19 Apr 2026 12:00:00 PM UTC\n"
            "printer Brother_Office is offline. disabled since Fri 19 Apr 2026 11:30:00 AM UTC\n"
            "system default destination: HP_LaserJet\n"
        )
        jobs = (
            "HP_LaserJet-42 user 1234 Fri 19 Apr 2026 12:01:00 PM UTC\n"
            "Brother_Office-43 user 4321 Fri 19 Apr 2026 12:02:00 PM UTC\n"
        )
        journal = (
            "Apr 19 12:00:00 host cups[123]: printer HP_LaserJet resumed\n"
            "Apr 19 12:00:02 host cups[123]: filter failed for job 42\n"
        )

        def _run_command(args, timeout_s=2.0):  # type: ignore[no-untyped-def]
            if args[:3] == ["systemctl", "status", "cups"]:
                return type("R", (), {"stdout": service, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:2] == ["lpstat", "-p"]:
                return type("R", (), {"stdout": printers, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:2] == ["lpstat", "-o"]:
                return type("R", (), {"stdout": jobs, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:1] == ["journalctl"]:
                return type("R", (), {"stdout": journal, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            raise AssertionError(f"unexpected command: {args}")

        snapshot = collect_printer_cups_diagnostics_snapshot(run_command_fn=_run_command)
        self.assertEqual("printer_cups", snapshot["preset"])
        self.assertEqual("active", snapshot["printer"]["service"]["active_state"])
        self.assertEqual("HP_LaserJet", snapshot["printer"]["printers"]["default_printer"])
        self.assertEqual(2, snapshot["printer"]["printers"]["printer_count"])
        self.assertEqual(2, snapshot["printer"]["jobs"]["match_count"])
        self.assertEqual(2, snapshot["printer"]["logs"]["match_count"])
        self.assertIn("Recent CUPS logs contain failure markers.", snapshot["summary"]["assessment"])
        text = render_printer_cups_diagnostics_snapshot(snapshot)
        self.assertIn("Printer/CUPS diagnostics", text)
        self.assertIn("Service: active=active", text)
        self.assertIn("Printers: default=HP_LaserJet", text)
        self.assertIn("Jobs: count=2", text)
        self.assertIn("Log matches: 2", text)
        self.assertNotIn("journalctl", text)

    def test_collect_generic_device_fallback_snapshot_is_compact_and_structured(self) -> None:
        uname = "Linux test-host 6.8.0-1 x86_64"
        usb = (
            "Bus 001 Device 002: ID 046d:0825 Logitech, Inc. Webcam C270\n"
            "Bus 001 Device 003: ID 0bda:58f4 Realtek Semiconductor Corp.\n"
        )
        pci = (
            "00:02.0 VGA compatible controller [0300]: Intel Corporation Device [8086:46a6]\n"
            "00:14.0 USB controller [0c03]: Intel Corporation Device [8086:7aa8]\n"
        )
        journal = (
            "Apr 19 12:00:00 host kernel: usb 1-1: device descriptor read/64, error -71\n"
            "Apr 19 12:00:01 host kernel: webcam: not detected after resume\n"
        )
        dmesg = (
            "[  10.000000] usb 1-1: reset full-speed USB device number 2 using xhci_hcd\n"
            "[  10.100000] webcam: firmware failed to load\n"
        )

        def _run_command(args, timeout_s=2.0):  # type: ignore[no-untyped-def]
            if args[:2] == ["uname", "-a"]:
                return type("R", (), {"stdout": uname, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:1] == ["lsusb"]:
                return type("R", (), {"stdout": usb, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:1] == ["lspci"]:
                return type("R", (), {"stdout": pci, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:1] == ["journalctl"]:
                return type("R", (), {"stdout": journal, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            if args[:1] == ["dmesg"]:
                return type("R", (), {"stdout": dmesg, "permission_denied": False, "not_available": False, "error": None, "returncode": 0})()
            raise AssertionError(f"unexpected command: {args}")

        snapshot = collect_generic_device_fallback_diagnostics_snapshot(run_command_fn=_run_command)
        self.assertEqual("generic_device_fallback", snapshot["preset"])
        self.assertEqual("Linux test-host 6.8.0-1 x86_64", snapshot["device"]["os"]["text"])
        self.assertEqual(2, snapshot["device"]["presence"]["usb"]["match_count"])
        self.assertEqual(2, snapshot["device"]["presence"]["pci"]["match_count"])
        self.assertEqual(2, snapshot["device"]["logs"]["journal"]["match_count"])
        self.assertEqual(2, snapshot["device"]["logs"]["dmesg"]["match_count"])
        self.assertIn("device or driver failure markers", snapshot["summary"]["assessment"])
        text = render_generic_device_fallback_diagnostics_snapshot(snapshot)
        self.assertIn("General device diagnostics", text)
        self.assertIn("OS/kernel:", text)
        self.assertIn("USB presence:", text)
        self.assertIn("PCI presence:", text)
        self.assertIn("Logs: journal=2; dmesg=2", text)
        self.assertNotIn("journalctl", text)

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
