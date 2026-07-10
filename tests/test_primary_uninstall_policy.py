from __future__ import annotations

from datetime import timedelta
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from agent.primary_uninstall_policy import (
    PRIMARY_UNINSTALL_MAX_DAYS,
    build_policy_context,
    build_primary_uninstall_marker_payload,
    diagnose_primary_uninstall_host_policy,
    disable_primary_uninstall_marker,
    enable_primary_uninstall_marker,
    payload_sha256,
    repair_primary_uninstall_host_policy_permissions,
    utc_now,
    validate_primary_uninstall_marker,
)


class PrimaryUninstallPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.state = self.root / "state"
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.ctx = build_policy_context(
            state_root=self.state,
            repository_path=self.repo,
            create_identity=True,
        )

    def _write_marker(self, payload: dict) -> None:
        self.ctx.host_lifecycle_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.ctx.host_lifecycle_root, 0o700)
        self.ctx.marker_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(self.ctx.marker_path, 0o600)

    def _valid_payload(self) -> dict:
        return build_primary_uninstall_marker_payload(self.ctx, expires_in_days=30)

    def test_valid_marker_enables(self) -> None:
        self._write_marker(self._valid_payload())
        status = validate_primary_uninstall_marker(self.ctx)
        self.assertTrue(status.enabled)
        self.assertEqual("enabled", status.reason)
        self.assertTrue(status.fingerprint)

    def test_schema_and_integrity_fail_closed(self) -> None:
        payload = self._valid_payload()
        payload["schema_version"] = 99
        payload["integrity"]["payload_sha256"] = payload_sha256(payload)
        self._write_marker(payload)
        self.assertEqual("marker_schema_version_unsupported", validate_primary_uninstall_marker(self.ctx).reason)

        payload = self._valid_payload()
        payload["repository_path"] = str(self.root / "other")
        self._write_marker(payload)
        self.assertEqual("marker_integrity_mismatch", validate_primary_uninstall_marker(self.ctx).reason)

    def test_binding_mismatches_fail_closed(self) -> None:
        for field, value, reason in (
            ("installation_id", "other", "marker_installation_id_mismatch"),
            ("repository_path", str(self.root / "copy"), "marker_repository_path_mismatch"),
            ("primary_service", "other.service", "marker_primary_service_mismatch"),
            ("uid", -1, "marker_uid_mismatch"),
        ):
            payload = self._valid_payload()
            payload[field] = value
            payload["integrity"]["payload_sha256"] = payload_sha256(payload)
            self._write_marker(payload)
            self.assertEqual(reason, validate_primary_uninstall_marker(self.ctx).reason)

    def test_expired_bad_permissions_symlink_and_duplicate_json_fail_closed(self) -> None:
        payload = self._valid_payload()
        payload["created_at"] = (utc_now() - timedelta(days=2)).isoformat()
        payload["expires_at"] = (utc_now() - timedelta(days=1)).isoformat()
        payload["integrity"]["payload_sha256"] = payload_sha256(payload)
        self._write_marker(payload)
        self.assertEqual("marker_expired", validate_primary_uninstall_marker(self.ctx).reason)

        payload = self._valid_payload()
        self._write_marker(payload)
        os.chmod(self.ctx.marker_path, 0o644)
        self.assertEqual("marker_permissions_too_broad", validate_primary_uninstall_marker(self.ctx).reason)

        self.ctx.marker_path.unlink()
        target = self.ctx.host_lifecycle_root / "target.json"
        target.write_text("{}", encoding="utf-8")
        self.ctx.marker_path.symlink_to(target)
        self.assertEqual("marker_symlink_rejected", validate_primary_uninstall_marker(self.ctx).reason)

        self.ctx.marker_path.unlink()
        self.ctx.marker_path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
        os.chmod(self.ctx.marker_path, 0o600)
        self.assertTrue(validate_primary_uninstall_marker(self.ctx).reason.startswith("duplicate_json_key"))

    def test_enable_disable_idempotent_and_expiry_bounds(self) -> None:
        status = enable_primary_uninstall_marker(expires_in_days=1, context=self.ctx)
        self.assertTrue(status.enabled)
        disabled = disable_primary_uninstall_marker(self.ctx)
        self.assertFalse(disabled.enabled)
        self.assertEqual("marker_missing", disabled.reason)
        again = disable_primary_uninstall_marker(self.ctx)
        self.assertEqual("marker_missing", again.reason)
        with self.assertRaisesRegex(ValueError, "expiry_exceeds_maximum"):
            build_primary_uninstall_marker_payload(self.ctx, expires_in_days=PRIMARY_UNINSTALL_MAX_DAYS + 1)

    def test_host_policy_diagnose_and_repair_permissions(self) -> None:
        self.ctx.host_lifecycle_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.ctx.host_lifecycle_root, 0o775)
        diagnostic = diagnose_primary_uninstall_host_policy(self.ctx)
        self.assertEqual("0o775", diagnostic.mode)
        self.assertTrue(diagnostic.repair_available)
        self.assertEqual("chmod_0700_available", diagnostic.repair_reason)

        result = repair_primary_uninstall_host_policy_permissions(self.ctx)
        self.assertTrue(result["ok"])
        self.assertTrue(result["changed"])
        after = diagnose_primary_uninstall_host_policy(self.ctx)
        self.assertEqual("0o700", after.mode)
        self.assertFalse(self.ctx.marker_path.exists())

    def test_host_policy_repair_refuses_symlink_root(self) -> None:
        target = self.root / "target"
        target.mkdir()
        self.ctx.host_lifecycle_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.ctx.host_lifecycle_root)
        self.ctx.host_lifecycle_root.symlink_to(target)
        diagnostic = diagnose_primary_uninstall_host_policy(self.ctx)
        self.assertTrue(diagnostic.is_symlink)
        self.assertFalse(diagnostic.repair_available)
        result = repair_primary_uninstall_host_policy_permissions(self.ctx)
        self.assertFalse(result["ok"])
        self.assertFalse(result["changed"])


if __name__ == "__main__":
    unittest.main()
