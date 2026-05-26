from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.secret_store import SecretStore
from agent.telegram_runtime_state import (
    clear_stale_telegram_locks,
    get_telegram_runtime_state,
    is_personal_agent_telegram_dropin_path,
    read_telegram_enablement,
    telegram_control_env,
    telegram_dropin_path,
    telegram_lock_paths,
    write_telegram_enablement,
    write_telegram_enablement_managed,
)


def _completed(args: list[str], returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


class TestTelegramRuntimeState(unittest.TestCase):
    def _run_stub(self, *, installed: bool, active: bool, enabled: bool):
        def _runner(args, **_kwargs):  # type: ignore[no-untyped-def]
            cmd = list(args)
            if cmd[-2:] == ["cat", "personal-agent-telegram.service"]:
                return _completed(cmd, 0 if installed else 1)
            if cmd[-2:] == ["is-active", "personal-agent-telegram.service"]:
                return _completed(cmd, 0 if active else 3, "active\n" if active else "inactive\n")
            if cmd[-2:] == ["is-enabled", "personal-agent-telegram.service"]:
                return _completed(cmd, 0 if enabled else 1, "enabled\n" if enabled else "disabled\n")
            raise AssertionError(f"unexpected command: {cmd}")

        return _runner

    def test_disabled_state_reports_disabled_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            state = get_telegram_runtime_state(
                home=home,
                env={},
                run=self._run_stub(installed=True, active=False, enabled=False),
            )
        self.assertFalse(bool(state["enabled"]))
        self.assertEqual("default", state["config_source"])
        self.assertEqual("disabled_optional", state["effective_state"])

    def test_enabled_token_missing_reports_misconfigured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            write_telegram_enablement(True, home=home, env={})
            state = get_telegram_runtime_state(
                home=home,
                env={},
                run=self._run_stub(installed=True, active=False, enabled=True),
            )
        self.assertTrue(bool(state["enabled"]))
        self.assertFalse(bool(state["token_configured"]))
        self.assertEqual("enabled_misconfigured", state["effective_state"])

    def test_enabled_active_service_reports_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            secret_path = home / ".local" / "share" / "personal-agent" / "secrets.enc.json"
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", "123456:abcdef")
            write_telegram_enablement(True, home=home, env={})
            state = get_telegram_runtime_state(
                home=home,
                env={"AGENT_SECRET_STORE_PATH": str(secret_path)},
                run=self._run_stub(installed=True, active=True, enabled=True),
            )
        self.assertTrue(bool(state["token_configured"]))
        self.assertEqual("enabled_running", state["effective_state"])

    def test_stale_lock_reports_blocked_and_can_be_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            secret_path = home / ".local" / "share" / "personal-agent" / "secrets.enc.json"
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", "123456:abcdef")
            write_telegram_enablement(True, home=home, env={})
            lock_path = telegram_lock_paths("123456:abcdef", home=home, env={})[0]
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text("999999\n", encoding="utf-8")
            state = get_telegram_runtime_state(
                home=home,
                env={"AGENT_SECRET_STORE_PATH": str(secret_path)},
                run=self._run_stub(installed=True, active=False, enabled=True),
            )
            removed = clear_stale_telegram_locks(
                "123456:abcdef",
                home=home,
                env={"AGENT_SECRET_STORE_PATH": str(secret_path)},
            )
        self.assertTrue(bool(state["lock_present"]))
        self.assertTrue(bool(state["lock_stale"]))
        self.assertEqual("enabled_blocked_by_lock", state["effective_state"])
        self.assertEqual([str(lock_path)], removed)

    def test_operator_env_ignores_shell_enabled_override_and_uses_dropin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            write_telegram_enablement(False, home=home, env={})
            enablement = read_telegram_enablement(
                home=home,
                env=telegram_control_env({"TELEGRAM_ENABLED": "1"}),
            )
        self.assertFalse(bool(enablement["enabled"]))
        self.assertEqual("config", enablement["config_source"])

    def test_managed_enablement_write_journals_and_verifies_owned_dropin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)

            ok, response = write_telegram_enablement_managed(True, home=home, env={})

            self.assertTrue(ok)
            self.assertTrue(response["ok"])
            self.assertTrue(telegram_dropin_path(home=home).is_file())
            self.assertTrue(bool(read_telegram_enablement(home=home, env={})["enabled"]))
            journal = response.get("managed_action_journal", {})
            self.assertEqual("telegram_enablement_config", journal.get("action_type"))
            self.assertTrue(journal.get("planned_steps"))
            self.assertTrue(journal.get("executed_steps"))
            self.assertTrue(journal.get("changed_resources"))
            self.assertTrue(journal.get("verification_result", {}).get("ok"))
            self.assertFalse(journal.get("rollback_result", {}).get("attempted"))

    def test_managed_enablement_rollback_restores_owned_dropin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            path = telegram_dropin_path(home=home)
            path.parent.mkdir(parents=True, exist_ok=True)
            original = "[Service]\nEnvironment=TELEGRAM_ENABLED=0\n"
            path.write_text(original, encoding="utf-8")

            with patch.object(Path, "read_text", side_effect=[original, "[Service]\n"]):
                ok, response = write_telegram_enablement_managed(True, home=home, env={})

            self.assertFalse(ok)
            self.assertEqual(original, path.read_text(encoding="utf-8"))
            self.assertEqual("telegram_dropin_write_verification_failed", response["error_kind"])
            journal = response.get("managed_action_journal", {})
            rollback_steps = journal.get("rollback_steps", [])
            self.assertTrue(any(step.get("name") == "restore_telegram_dropin" and step.get("status") == "ok" for step in rollback_steps))

    def test_telegram_dropin_path_validator_rejects_unrelated_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            unrelated = home / ".config" / "systemd" / "user" / "unrelated.service.d" / "override.conf"

            self.assertFalse(is_personal_agent_telegram_dropin_path(unrelated, home=home))
            self.assertTrue(is_personal_agent_telegram_dropin_path(telegram_dropin_path(home=home), home=home))

    def test_managed_enablement_write_path_does_not_shell_out(self) -> None:
        import inspect

        source = inspect.getsource(write_telegram_enablement_managed)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("shell=True", source)


if __name__ == "__main__":
    unittest.main()
