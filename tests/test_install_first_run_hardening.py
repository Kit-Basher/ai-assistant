from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config import load_config, resolved_default_db_path, resolved_default_log_path
from agent.doctor import DoctorReport, _apply_safe_fixes, _check_required_dirs
from agent.secret_store import SecretStore
from agent.startup_checks import run_startup_checks


class TestInstallFirstRunHardening(unittest.TestCase):
    def test_resolved_default_paths_use_canonical_location_for_fresh_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            canonical_db = Path(tmpdir) / "state" / "agent.db"
            canonical_log = Path(tmpdir) / "state" / "agent.jsonl"
            legacy_db = Path(tmpdir) / "repo" / "memory" / "agent.db"
            legacy_log = Path(tmpdir) / "repo" / "logs" / "agent.jsonl"
            with patch("agent.config.canonical_db_path", return_value=canonical_db), patch(
                "agent.config.canonical_log_path",
                return_value=canonical_log,
            ), patch(
                "agent.config.legacy_repo_db_path",
                return_value=legacy_db,
            ), patch(
                "agent.config.legacy_repo_log_path",
                return_value=legacy_log,
            ):
                self.assertEqual(str(canonical_db), resolved_default_db_path())
                self.assertEqual(str(canonical_log), resolved_default_log_path())

    def test_resolved_default_paths_keep_existing_legacy_state_until_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            canonical_db = Path(tmpdir) / "state" / "agent.db"
            canonical_log = Path(tmpdir) / "state" / "agent.jsonl"
            legacy_db = Path(tmpdir) / "repo" / "memory" / "agent.db"
            legacy_log = Path(tmpdir) / "repo" / "logs" / "agent.jsonl"
            legacy_db.parent.mkdir(parents=True, exist_ok=True)
            legacy_log.parent.mkdir(parents=True, exist_ok=True)
            legacy_db.write_text("legacy", encoding="utf-8")
            legacy_log.write_text("legacy-log", encoding="utf-8")
            with patch("agent.config.canonical_db_path", return_value=canonical_db), patch(
                "agent.config.canonical_log_path",
                return_value=canonical_log,
            ), patch(
                "agent.config.legacy_repo_db_path",
                return_value=legacy_db,
            ), patch(
                "agent.config.legacy_repo_log_path",
                return_value=legacy_log,
            ):
                self.assertEqual(str(legacy_db), resolved_default_db_path())
                self.assertEqual(str(legacy_log), resolved_default_log_path())

    def test_load_config_uses_canonical_runtime_paths_for_fresh_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            canonical_db = Path(tmpdir) / "state" / "agent.db"
            canonical_log = Path(tmpdir) / "state" / "agent.jsonl"
            with patch("agent.config.resolved_default_db_path", return_value=str(canonical_db)), patch(
                "agent.config.resolved_default_log_path",
                return_value=str(canonical_log),
            ), patch.dict(
                os.environ,
                {"LLM_PROVIDER": "none", "TELEGRAM_ENABLED": "0"},
                clear=False,
            ):
                os.environ.pop("AGENT_DB_PATH", None)
                os.environ.pop("AGENT_LOG_PATH", None)
                cfg = load_config()
        self.assertEqual(str(canonical_db), cfg.db_path)
        self.assertEqual(str(canonical_log), cfg.log_path)

    def test_startup_checks_fail_closed_when_config_load_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            (home / ".local" / "share" / "personal-agent").mkdir(parents=True, exist_ok=True)
            (home / ".config" / "personal-agent").mkdir(parents=True, exist_ok=True)
            (home / ".config" / "systemd" / "user").mkdir(parents=True, exist_ok=True)
            with patch.dict(
                os.environ,
                {
                    "HOME": str(home),
                    "LLM_PROVIDER": "bad-provider",
                    "AGENT_SECRET_STORE_PATH": str(home / ".local" / "share" / "personal-agent" / "missing.enc.json"),
                },
                clear=False,
            ):
                report = run_startup_checks(service="api", config=None)
        self.assertEqual("FAIL", str(report.get("status")))
        self.assertEqual("config_load_failed", str(report.get("failure_code")))

    def test_startup_checks_fail_when_registry_json_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            state_dir = home / ".local" / "share" / "personal-agent"
            config_dir = home / ".config" / "personal-agent"
            systemd_dir = home / ".config" / "systemd" / "user"
            state_dir.mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(parents=True, exist_ok=True)
            systemd_dir.mkdir(parents=True, exist_ok=True)
            secret_path = state_dir / "secrets.enc.json"
            SecretStore(path=str(secret_path)).set_secret("telegram:bot_token", "1234567:abcdefghijklmnopqrstuvwxyz_123456")
            registry_path = Path(tmpdir) / "registry.json"
            registry_path.write_text("{not-json", encoding="utf-8")
            config = type(
                "Cfg",
                (),
                {
                    "llm_registry_path": str(registry_path),
                    "llm_provider": "ollama",
                    "ollama_base_url": "http://127.0.0.1:11434",
                    "ollama_host": "http://127.0.0.1:11434",
                    "telegram_enabled": False,
                },
            )()
            with patch.dict(
                os.environ,
                {"HOME": str(home), "AGENT_SECRET_STORE_PATH": str(secret_path)},
                clear=False,
            ):
                report = run_startup_checks(service="api", config=config)
        self.assertEqual("FAIL", str(report.get("status")))
        checks = report.get("checks") if isinstance(report.get("checks"), list) else []
        self.assertTrue(any(isinstance(row, dict) and row.get("failure_code") == "registry_invalid_json" for row in checks))

    def test_startup_checks_fail_when_secret_store_is_corrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            state_dir = home / ".local" / "share" / "personal-agent"
            config_dir = home / ".config" / "personal-agent"
            systemd_dir = home / ".config" / "systemd" / "user"
            state_dir.mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(parents=True, exist_ok=True)
            systemd_dir.mkdir(parents=True, exist_ok=True)
            secret_path = state_dir / "secrets.enc.json"
            secret_path.write_text("{bad-json", encoding="utf-8")
            registry_path = Path(tmpdir) / "registry.json"
            registry_path.write_text(json.dumps({"providers": {}, "models": {}}), encoding="utf-8")
            config = type(
                "Cfg",
                (),
                {
                    "llm_registry_path": str(registry_path),
                    "llm_provider": "ollama",
                    "ollama_base_url": "http://127.0.0.1:11434",
                    "ollama_host": "http://127.0.0.1:11434",
                    "telegram_enabled": False,
                },
            )()
            with patch.dict(
                os.environ,
                {"HOME": str(home), "AGENT_SECRET_STORE_PATH": str(secret_path)},
                clear=False,
            ):
                report = run_startup_checks(service="api", config=config)
        self.assertEqual("FAIL", str(report.get("status")))
        checks = report.get("checks") if isinstance(report.get("checks"), list) else []
        self.assertTrue(any(isinstance(row, dict) and row.get("failure_code") == "secret_store_decrypt_failed" for row in checks))

    def test_doctor_required_dirs_warn_when_install_dirs_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                check = _check_required_dirs()
        self.assertEqual("WARN", check.status)
        self.assertIn(".local/share/personal-agent", check.detail_short)
        self.assertEqual("Run: python -m agent doctor --fix", check.next_action)

    def test_doctor_fix_copies_legacy_runtime_storage_into_canonical_state_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            repo_root = Path(tmpdir) / "repo"
            (repo_root / "memory").mkdir(parents=True, exist_ok=True)
            (repo_root / "logs").mkdir(parents=True, exist_ok=True)
            legacy_db = repo_root / "memory" / "agent.db"
            legacy_log = repo_root / "logs" / "agent.jsonl"
            legacy_db.write_text("legacy-db", encoding="utf-8")
            legacy_log.write_text("legacy-log", encoding="utf-8")
            report = DoctorReport(
                trace_id="doctor-test",
                generated_at="2026-04-02T00:00:00+00:00",
                summary_status="WARN",
                checks=[],
                next_action="Run: python -m agent doctor --fix",
                fixes_applied=[],
                support_bundle_path=None,
            )
            with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                changes, _bundle = _apply_safe_fixes(report, repo_root=repo_root)
                migrated_db = home / ".local" / "share" / "personal-agent" / "agent.db"
                migrated_log = home / ".local" / "share" / "personal-agent" / "agent.jsonl"
                self.assertTrue(migrated_db.is_file())
                self.assertTrue(migrated_log.is_file())
                self.assertEqual("legacy-db", migrated_db.read_text(encoding="utf-8"))
                self.assertEqual("legacy-log", migrated_log.read_text(encoding="utf-8"))
                self.assertTrue(any(str(row).startswith("copied_legacy_db:") for row in changes))
                self.assertTrue(any(str(row).startswith("copied_legacy_log:") for row in changes))

    def test_legacy_root_scripts_fail_closed_and_point_to_canonical_docs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for script_name in ("install.sh", "uninstall.sh", "doctor.sh"):
            proc = subprocess.run(
                ["bash", str(repo_root / script_name)],
                check=False,
                capture_output=True,
                text=True,
                cwd=str(repo_root),
            )
            self.assertEqual(1, proc.returncode, msg=script_name)
            self.assertIn("no longer supported", proc.stdout.lower(), msg=script_name)
            self.assertIn("docs/operator/SETUP.md".lower(), proc.stdout.lower(), msg=script_name)


if __name__ == "__main__":
    unittest.main()
