from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import backup_restore_proof


class TestBackupRestoreProof(unittest.TestCase):
    def _fixture_backup(self, root: Path, *, app_version: str | None = None) -> tuple[Path, Path]:
        source_home = root / "source-home"
        backup_restore_proof._write_fixture_state(source_home)  # noqa: SLF001
        archive = root / "backup.tar.gz"
        kwargs = {"app_version": app_version} if app_version is not None else {}
        backup_restore_proof.create_backup(source_home, archive, **kwargs)
        return source_home, archive

    def test_valid_backup_archive_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _source, archive = self._fixture_backup(Path(tmp))

            result = backup_restore_proof.validate_backup(archive)

            self.assertTrue(result.ok, result.error)
            self.assertIn(".local/share/personal-agent/agent.db", result.files)
            self.assertIn(".local/share/personal-agent/secrets.enc.json", result.sensitive_files)

    def test_dry_run_restore_succeeds_without_secret_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _source, archive = self._fixture_backup(Path(tmp))

            result = backup_restore_proof.dry_run_restore(archive)
            rendered = json.dumps(result, sort_keys=True)

            self.assertTrue(result["ok"], result)
            self.assertFalse(result["mutated"])
            self.assertNotIn("SUPER_SECRET_TOKEN_VALUE", rendered)
            self.assertIn("<redacted>", rendered)

    def test_restore_into_temp_state_preserves_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _source, archive = self._fixture_backup(root)
            target = root / "target-home"

            result = backup_restore_proof.restore_to_temp_state(archive, target)

            self.assertTrue(result["ok"], result)
            self.assertFalse(result["mutated_live_state"])
            self.assertTrue((target / ".config/personal-agent/config.json").is_file())
            self.assertTrue((target / ".local/share/personal-agent/agent.db").is_file())
            self.assertTrue((target / ".local/share/personal-agent/secrets.enc.json").is_file())

    def test_corrupt_backup_fails_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "corrupt.tar.gz"
            archive.write_bytes(b"not a tar")

            result = backup_restore_proof.validate_backup(archive)

            self.assertFalse(result.ok)
            self.assertEqual("corrupt_backup", result.error)

    def test_version_mismatch_is_refused_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _source, archive = self._fixture_backup(Path(tmp), app_version="0.0.0-mismatch")

            result = backup_restore_proof.validate_backup(
                archive,
                expected_app_version=backup_restore_proof.APP_VERSION,
                strict_version=True,
            )

            self.assertFalse(result.ok)
            self.assertEqual("version_mismatch", result.error)

    def test_end_to_end_proof_passes(self) -> None:
        ok, rows = backup_restore_proof.run_proof()

        self.assertTrue(ok, rows)


if __name__ == "__main__":
    unittest.main()
