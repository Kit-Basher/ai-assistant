from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.secret_store import SecretStore
from agent.startup_checks import run_startup_checks


class TestStartupChecks(unittest.TestCase):
    def _config(self, registry_path: str, *, telegram_enabled: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            llm_registry_path=registry_path,
            llm_provider="ollama",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_host="http://127.0.0.1:11434",
            telegram_enabled=telegram_enabled,
        )

    def test_warn_when_secret_store_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            (home / ".local" / "share" / "personal-agent").mkdir(parents=True, exist_ok=True)
            (home / ".config" / "systemd" / "user").mkdir(parents=True, exist_ok=True)
            registry_path = Path(tmpdir) / "registry.json"
            registry_path.write_text(json.dumps({"providers": {}, "models": {}}), encoding="utf-8")
            with patch("agent.startup_checks.Path.home", return_value=home):
                with patch.dict(os.environ, {"AGENT_SECRET_STORE_PATH": str(Path(tmpdir) / "missing.enc.json")}, clear=False):
                    report = run_startup_checks(service="api", config=self._config(str(registry_path)))
        self.assertEqual("WARN", str(report.get("status")))
        checks = report.get("checks") if isinstance(report.get("checks"), list) else []
        self.assertTrue(any(isinstance(row, dict) and row.get("failure_code") == "secret_store_missing" for row in checks))

    def test_fail_when_telegram_token_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            (home / ".local" / "share" / "personal-agent").mkdir(parents=True, exist_ok=True)
            (home / ".config" / "systemd" / "user").mkdir(parents=True, exist_ok=True)
            secret_path = Path(tmpdir) / "secrets.enc.json"
            store = SecretStore(path=str(secret_path))
            store.set_secret("dummy", "1")
            registry_path = Path(tmpdir) / "registry.json"
            registry_path.write_text(json.dumps({"providers": {}, "models": {}}), encoding="utf-8")
            with patch("agent.startup_checks.Path.home", return_value=home):
                with patch.dict(os.environ, {"AGENT_SECRET_STORE_PATH": str(secret_path)}, clear=False):
                    report = run_startup_checks(
                        service="telegram",
                        config=self._config(str(registry_path), telegram_enabled=True),
                        token=None,
                    )
        self.assertEqual("FAIL", str(report.get("status")))
        self.assertEqual("telegram_token_missing", str(report.get("failure_code")))
        self.assertIn("telegram:bot_token", str(report.get("next_action")))

    def test_pass_when_telegram_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            (home / ".local" / "share" / "personal-agent").mkdir(parents=True, exist_ok=True)
            (home / ".config" / "systemd" / "user").mkdir(parents=True, exist_ok=True)
            secret_path = Path(tmpdir) / "secrets.enc.json"
            store = SecretStore(path=str(secret_path))
            store.set_secret("dummy", "1")
            registry_path = Path(tmpdir) / "registry.json"
            registry_path.write_text(json.dumps({"providers": {}, "models": {}}), encoding="utf-8")
            with patch("agent.startup_checks.Path.home", return_value=home):
                with patch.dict(os.environ, {"AGENT_SECRET_STORE_PATH": str(secret_path)}, clear=False):
                    report = run_startup_checks(
                        service="telegram",
                        config=self._config(str(registry_path), telegram_enabled=False),
                        token=None,
                    )
        self.assertEqual("PASS", str(report.get("status")))
        checks = report.get("checks") if isinstance(report.get("checks"), list) else []
        enabled_rows = [
            row for row in checks if isinstance(row, dict) and str(row.get("check_id")) == "telegram.enabled"
        ]
        self.assertTrue(enabled_rows)
        self.assertIn("disabled (optional)", str(enabled_rows[0].get("message") or ""))

    def test_pass_for_api_when_requirements_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            (home / ".local" / "share" / "personal-agent").mkdir(parents=True, exist_ok=True)
            (home / ".config" / "systemd" / "user").mkdir(parents=True, exist_ok=True)
            secret_path = Path(tmpdir) / "secrets.enc.json"
            store = SecretStore(path=str(secret_path))
            store.set_secret("telegram:bot_token", "1234567:abcdefghijklmnopqrstuvwxyz_123456")
            registry_path = Path(tmpdir) / "registry.json"
            registry_path.write_text(json.dumps({"providers": {}, "models": {}}), encoding="utf-8")
            with patch("agent.startup_checks.Path.home", return_value=home):
                with patch.dict(os.environ, {"AGENT_SECRET_STORE_PATH": str(secret_path)}, clear=False):
                    report = run_startup_checks(service="api", config=self._config(str(registry_path)))
        self.assertEqual("PASS", str(report.get("status")))
        self.assertTrue(str(report.get("trace_id")).startswith("startup-api-"))


if __name__ == "__main__":
    unittest.main()
