from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.recovery_install_audit import audit


class TestRecoveryInstallAuditScript(unittest.TestCase):
    def _state(self, root: Path) -> Path:
        state = root / "state"
        state.mkdir()
        connection = sqlite3.connect(state / "agent.db")
        connection.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO schema_meta VALUES ('schema_version', '2')")
        connection.commit()
        connection.close()
        (state / "llm_registry.json").write_text("{}\n", encoding="utf-8")
        return state

    def test_audit_passes_canonical_state_and_preserved_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = self._state(root)
            config = root / "config"
            repo = root / "repo"
            artifact = root / "recovery.tar.gz"
            config.mkdir()
            repo.mkdir()
            artifact.write_bytes(b"preserved")
            checks = audit(repo_root=repo, state_root=state, config_root=config, expected_artifacts=[artifact])
            self.assertFalse([check for check in checks if check.status == "FAIL"])

    def test_audit_warns_without_deleting_legacy_repo_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = self._state(root)
            config = root / "config"
            repo = root / "repo"
            config.mkdir()
            (repo / "memory").mkdir(parents=True)
            legacy = repo / "memory" / "agent.db"
            legacy.write_bytes(b"legacy")
            checks = audit(repo_root=repo, state_root=state, config_root=config, expected_artifacts=[])
            self.assertTrue(any(check.status == "WARN" and "memory/agent.db" in check.name for check in checks))
            self.assertTrue(legacy.exists())


if __name__ == "__main__":
    unittest.main()
