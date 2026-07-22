from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRecoveryInstallAudit(unittest.TestCase):
    def test_safe_mode_is_the_clean_environment_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = Path(tmpdir) / "llm_registry.json"
            registry.write_text("{}\n", encoding="utf-8")
            env = {
                "LLM_REGISTRY_PATH": str(registry),
                "AGENT_DB_PATH": str(Path(tmpdir) / "agent.db"),
                "AGENT_LOG_PATH": str(Path(tmpdir) / "agent.jsonl"),
            }
            with patch.dict(os.environ, env, clear=True):
                config = load_config(require_telegram_token=False)
            self.assertTrue(config.safe_mode_enabled)
            self.assertFalse(config.telegram_enabled)

    def test_production_service_surfaces_pin_safe_mode_without_overriding_telegram(self) -> None:
        paths = (
            REPO_ROOT / "systemd" / "personal-agent-api.service",
            REPO_ROOT / "packaging" / "debian" / "personal-agent-api.service.in",
            REPO_ROOT / "packaging" / "release_bundle" / "install.sh",
        )
        for path in paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertIn("AGENT_SAFE_MODE=1", text)
                self.assertNotIn("TELEGRAM_ENABLED=0", text)
                self.assertNotIn("personal-agent-telegram.service", text)

    def test_embedded_telegram_dependency_includes_job_queue(self) -> None:
        metadata = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('python-telegram-bot[job-queue]>=22.6', metadata)

    def test_shipped_units_use_canonical_mutable_state_roots(self) -> None:
        service_paths = (
            REPO_ROOT / "systemd" / "personal-agent-api.service",
            REPO_ROOT / "systemd" / "personal-agent-api-dev.service",
            REPO_ROOT / "packaging" / "debian" / "personal-agent-api.service.in",
        )
        for path in service_paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertIn(".local/share/personal-agent/agent.db", text)
                self.assertIn(".local/share/personal-agent/llm_registry.json", text)
                self.assertNotIn("personal-agent/memory/agent.db", text)
                self.assertNotIn("%h/personal-agent/llm_registry.json", text)

    def test_artifact_builders_require_fresh_webui_manifest(self) -> None:
        for relative in ("scripts/build_release_bundle.sh", "scripts/build_deb.sh"):
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            with self.subTest(path=relative):
                self.assertIn("webui_build_manifest.py", text)
                self.assertIn("verify --repo-root", text)

    def test_install_surface_contract_is_unambiguous(self) -> None:
        stable = (REPO_ROOT / "scripts" / "install_local.sh").read_text(encoding="utf-8")
        dev = (REPO_ROOT / "scripts" / "install_dev.sh").read_text(encoding="utf-8")
        self.assertIn("127.0.0.1:8765", stable)
        self.assertNotIn("127.0.0.1:18765", stable)
        self.assertIn("127.0.0.1:18765", dev)
        self.assertIn("personal-agent-api-dev.service", dev)


if __name__ == "__main__":
    unittest.main()
