from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.actions.persistent_journal import PersistentManagedActionJournalStore
from agent.secret_store import SecretStore
from agent.telegram_runtime_state import (
    clear_stale_telegram_locks,
    get_telegram_runtime_state,
    is_approved_telegram_systemctl_user_action,
    is_personal_agent_telegram_dropin_path,
    manage_telegram_service_state,
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
    @staticmethod
    def _journal_store(home: Path) -> PersistentManagedActionJournalStore:
        return PersistentManagedActionJournalStore(home / "managed_actions.db")

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

    def _service_action_runner(self, *, active_after_restart: bool = True, active_after_stop: bool = False):
        state = {"active": False, "enabled": True}
        calls: list[list[str]] = []

        def _runner(args, **_kwargs):  # type: ignore[no-untyped-def]
            cmd = list(args)
            calls.append(cmd)
            if cmd == ["systemctl", "--user", "--version"]:
                return _completed(cmd, 0, "systemd 255\n")
            if cmd[-2:] == ["cat", "personal-agent-telegram.service"]:
                return _completed(cmd, 0)
            if cmd[-2:] == ["is-active", "personal-agent-telegram.service"]:
                return _completed(cmd, 0 if state["active"] else 3, "active\n" if state["active"] else "inactive\n")
            if cmd[-2:] == ["is-enabled", "personal-agent-telegram.service"]:
                return _completed(cmd, 0 if state["enabled"] else 1, "enabled\n" if state["enabled"] else "disabled\n")
            if cmd[-1:] == ["daemon-reload"]:
                return _completed(cmd, 0)
            if cmd[-2:] == ["restart", "personal-agent-telegram.service"]:
                state["active"] = bool(active_after_restart)
                return _completed(cmd, 0 if active_after_restart else 1)
            if cmd[-2:] == ["stop", "personal-agent-telegram.service"]:
                state["active"] = bool(active_after_stop)
                return _completed(cmd, 0 if not active_after_stop else 1)
            raise AssertionError(f"unexpected command: {cmd}")

        return _runner, calls

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
        self.assertEqual("DISABLED_OPTIONAL", state["telegram_health_level"])
        self.assertEqual("No action needed.", state["next_action"])

    def test_disabled_with_saved_token_remains_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            secret_path = home / ".local" / "share" / "personal-agent" / "secrets.enc.json"
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", "123456:abcdef")
            state = get_telegram_runtime_state(
                home=home,
                env={"AGENT_SECRET_STORE_PATH": str(secret_path)},
                run=self._run_stub(installed=True, active=False, enabled=False),
            )
        self.assertTrue(bool(state["token_configured"]))
        self.assertEqual("disabled_optional", state["effective_state"])
        self.assertEqual("DISABLED_OPTIONAL", state["telegram_health_level"])
        self.assertEqual("No action needed.", state["next_action"])

    def test_enabled_with_saved_token_but_stopped_is_degraded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            secret_path = home / ".local" / "share" / "personal-agent" / "secrets.enc.json"
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", "123456:abcdef")
            write_telegram_enablement(True, home=home, env={})
            state = get_telegram_runtime_state(
                home=home,
                env={"AGENT_SECRET_STORE_PATH": str(secret_path)},
                run=self._run_stub(installed=True, active=False, enabled=True),
            )
        self.assertEqual("enabled_stopped", state["effective_state"])
        self.assertEqual("DEGRADED", state["telegram_health_level"])

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

    def test_live_lock_is_not_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            secret_path = home / ".local" / "share" / "personal-agent" / "secrets.enc.json"
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", "123456:abcdef")
            write_telegram_enablement(True, home=home, env={})
            lock_path = telegram_lock_paths("123456:abcdef", home=home, env={})[0]
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

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
            lock_still_exists = lock_path.exists()

        self.assertTrue(bool(state["lock_present"]))
        self.assertTrue(bool(state["lock_live"]))
        self.assertFalse(bool(state["lock_stale"]))
        self.assertEqual([], removed)
        self.assertTrue(lock_still_exists)

    def test_duplicate_pollers_are_reported_without_token_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            secret_path = home / ".local" / "share" / "personal-agent" / "secrets.enc.json"
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            token = "123456:abcdef"
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", token)
            write_telegram_enablement(True, home=home, env={})

            def _runner(args, **_kwargs):  # type: ignore[no-untyped-def]
                cmd = list(args)
                if cmd == ["ps", "-eo", "pid,args"]:
                    return _completed(
                        cmd,
                        0,
                        f" 101 python -m agent.telegram_adapter.bot --token={token}\n"
                        " 202 /usr/bin/python -m agent.telegram_adapter.bot\n",
                    )
                if cmd[-2:] == ["cat", "personal-agent-telegram.service"]:
                    return _completed(cmd, 0)
                if cmd[-2:] == ["is-active", "personal-agent-telegram.service"]:
                    return _completed(cmd, 0, "active\n")
                if cmd[-2:] == ["is-enabled", "personal-agent-telegram.service"]:
                    return _completed(cmd, 0, "enabled\n")
                raise AssertionError(f"unexpected command: {cmd}")

            state = get_telegram_runtime_state(
                home=home,
                env={"AGENT_SECRET_STORE_PATH": str(secret_path)},
                run=_runner,
            )

        self.assertTrue(bool(state["duplicate_pollers"]))
        self.assertEqual(2, state["poller_count"])
        self.assertEqual("enabled_duplicate_pollers", state["effective_state"])
        self.assertIn("Stop duplicate Telegram pollers", state["next_action"])
        self.assertNotIn(token, json.dumps(state, sort_keys=True))
        self.assertIn("[REDACTED", json.dumps(state, sort_keys=True))

    def test_poller_inspection_failure_is_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            secret_path = home / ".local" / "share" / "personal-agent" / "secrets.enc.json"
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", "123456:abcdef")
            write_telegram_enablement(True, home=home, env={})

            def _runner(args, **_kwargs):  # type: ignore[no-untyped-def]
                cmd = list(args)
                if cmd == ["ps", "-eo", "pid,args"]:
                    raise RuntimeError("ps unavailable")
                if cmd[-2:] == ["cat", "personal-agent-telegram.service"]:
                    return _completed(cmd, 0)
                if cmd[-2:] == ["is-active", "personal-agent-telegram.service"]:
                    return _completed(cmd, 0, "active\n")
                if cmd[-2:] == ["is-enabled", "personal-agent-telegram.service"]:
                    return _completed(cmd, 0, "enabled\n")
                raise AssertionError(f"unexpected command: {cmd}")

            state = get_telegram_runtime_state(
                home=home,
                env={"AGENT_SECRET_STORE_PATH": str(secret_path)},
                run=_runner,
            )

        self.assertFalse(bool(state["poller_inspection_available"]))
        self.assertFalse(bool(state["duplicate_pollers"]))
        self.assertEqual("enabled_running", state["effective_state"])

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
            store = self._journal_store(home)

            ok, response = write_telegram_enablement_managed(
                True,
                home=home,
                env={},
                managed_action_journal_store=store,
            )

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
            persisted = store.get(journal["action_id"])
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertEqual("verified", persisted["status"])
            self.assertNotIn(str(home), json.dumps(persisted, sort_keys=True))

    def test_managed_enablement_rollback_restores_owned_dropin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            store = self._journal_store(home)
            path = telegram_dropin_path(home=home)
            path.parent.mkdir(parents=True, exist_ok=True)
            original = "[Service]\nEnvironment=TELEGRAM_ENABLED=0\n"
            path.write_text(original, encoding="utf-8")

            with patch.object(Path, "read_text", side_effect=[original, "[Service]\n"]):
                ok, response = write_telegram_enablement_managed(
                    True,
                    home=home,
                    env={},
                    managed_action_journal_store=store,
                )

            self.assertFalse(ok)
            self.assertEqual(original, path.read_text(encoding="utf-8"))
            self.assertEqual("telegram_dropin_write_verification_failed", response["error_kind"])
            journal = response.get("managed_action_journal", {})
            rollback_steps = journal.get("rollback_steps", [])
            self.assertTrue(any(step.get("name") == "restore_telegram_dropin" and step.get("status") == "ok" for step in rollback_steps))
            persisted = store.get(journal["action_id"])
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertEqual("rolled_back", persisted["status"])
            self.assertNotIn(str(home), json.dumps(persisted, sort_keys=True))

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

    def test_manage_telegram_service_enable_journals_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            store = self._journal_store(home)
            secret_path = home / ".local" / "share" / "personal-agent" / "secrets.enc.json"
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", "123456:abcdef")
            runner, calls = self._service_action_runner(active_after_restart=True)

            ok, response = manage_telegram_service_state(
                True,
                home=home,
                env={"AGENT_SECRET_STORE_PATH": str(secret_path)},
                run=runner,
                managed_action_journal_store=store,
            )

            self.assertTrue(ok)
            self.assertTrue(response["ok"])
            commands = [call[-2:] for call in calls]
            self.assertIn(["restart", "personal-agent-telegram.service"], commands)
            journal = response.get("managed_action_journal", {})
            self.assertEqual("telegram_service_enable", journal.get("action_type"))
            self.assertTrue(journal.get("verification_result", {}).get("ok"))
            self.assertFalse(journal.get("rollback_result", {}).get("attempted"))
            persisted = store.get(journal["action_id"])
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertEqual("verified", persisted["status"])
            self.assertNotIn(str(home), json.dumps(persisted, sort_keys=True))
            self.assertEqual([], store.incomplete())

    def test_manage_telegram_service_verification_failure_rolls_back_dropin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            store = self._journal_store(home)
            secret_path = home / ".local" / "share" / "personal-agent" / "secrets.enc.json"
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", "123456:abcdef")
            original = "[Service]\nEnvironment=TELEGRAM_ENABLED=0\n"
            path = telegram_dropin_path(home=home)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(original, encoding="utf-8")
            runner, calls = self._service_action_runner(active_after_restart=False)

            ok, response = manage_telegram_service_state(
                True,
                home=home,
                env={"AGENT_SECRET_STORE_PATH": str(secret_path)},
                run=runner,
                managed_action_journal_store=store,
            )

            self.assertFalse(ok)
            self.assertEqual(original, path.read_text(encoding="utf-8"))
            self.assertEqual("telegram_service_restart_failed", response["error_kind"])
            self.assertIn("restored previous Telegram service config", response["message"])
            commands = [call[-2:] for call in calls]
            self.assertIn(["restart", "personal-agent-telegram.service"], commands)
            self.assertIn(["stop", "personal-agent-telegram.service"], commands)
            journal = response.get("managed_action_journal", {})
            self.assertTrue(journal.get("rollback_result", {}).get("attempted"))
            self.assertTrue(
                any(step.get("name") == "restore_telegram_dropin" for step in journal.get("rollback_steps", []))
            )
            persisted = store.get(journal["action_id"])
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertEqual("rolled_back", persisted["status"])
            self.assertNotIn(str(home), json.dumps(persisted, sort_keys=True))

    def test_manage_telegram_service_rollback_failure_persists_recovery_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            store = self._journal_store(home)
            secret_path = home / ".local" / "share" / "personal-agent" / "secrets.enc.json"
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", "123456:abcdef")
            runner, _calls = self._service_action_runner(active_after_restart=False)

            with patch("agent.telegram_runtime_state._rollback_telegram_dropin", return_value=False):
                ok, response = manage_telegram_service_state(
                    True,
                    home=home,
                    env={"AGENT_SECRET_STORE_PATH": str(secret_path)},
                    run=runner,
                    managed_action_journal_store=store,
                )

            self.assertFalse(ok)
            journal = response.get("managed_action_journal", {})
            persisted = store.get(journal["action_id"])
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertEqual("recovery_needed", persisted["status"])
            self.assertTrue(persisted["recovery_needed"])
            self.assertNotIn(str(home), json.dumps(persisted, sort_keys=True))

    def test_manage_telegram_service_disable_stops_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            store = self._journal_store(home)
            write_telegram_enablement(True, home=home, env={})
            runner, calls = self._service_action_runner(active_after_stop=False)

            ok, response = manage_telegram_service_state(
                False,
                home=home,
                env={},
                run=runner,
                managed_action_journal_store=store,
            )

            self.assertTrue(ok)
            commands = [call[-2:] for call in calls]
            self.assertIn(["stop", "personal-agent-telegram.service"], commands)
            self.assertFalse(bool(response["state"]["enabled"]))
            self.assertFalse(bool(response["state"]["service_active"]))
            journal = response.get("managed_action_journal", {})
            persisted = store.get(journal["action_id"])
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertEqual("verified", persisted["status"])

    def test_approved_telegram_systemctl_actions_reject_arbitrary_service(self) -> None:
        self.assertTrue(is_approved_telegram_systemctl_user_action(["restart", "personal-agent-telegram.service"]))
        self.assertFalse(is_approved_telegram_systemctl_user_action(["restart", "unrelated.service"]))
        self.assertFalse(is_approved_telegram_systemctl_user_action(["enable", "personal-agent-telegram.service"]))

    def test_manage_telegram_service_does_not_require_online_getme(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            store = self._journal_store(home)
            secret_path = home / ".local" / "share" / "personal-agent" / "secrets.enc.json"
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", "123456:abcdef")
            runner, _calls = self._service_action_runner(active_after_restart=True)

            ok, response = manage_telegram_service_state(
                True,
                home=home,
                env={"AGENT_SECRET_STORE_PATH": str(secret_path)},
                run=runner,
                managed_action_journal_store=store,
            )

        self.assertTrue(ok)
        self.assertFalse(bool(response["managed_action_journal"]["verification_result"].get("online_getme_required")))


if __name__ == "__main__":
    unittest.main()
