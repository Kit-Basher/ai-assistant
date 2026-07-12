from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agent.skill_pack_permissions import (
    SkillGrantStore,
    SkillPackInvocationBroker,
    SkillPermissionError,
    build_skill_identity,
    diff_skill_permissions,
    validate_skill_manifest,
)


def _manifest(*, permissions: list[str], version: str = "1.0.0") -> dict[str, object]:
    return {
        "schema_version": 1,
        "skill_pack_id": "example.skill",
        "publisher_id": "example.publisher",
        "name": "Example Skill",
        "version": version,
        "entrypoints": [],
        "declared_permissions": permissions,
        "read_only_surfaces": [],
        "network_domains": [],
        "filesystem_roots": [],
        "provider_accounts": [],
        "background_tasks": [],
        "configuration_schema": {},
    }


class TestSkillPackPermissions(unittest.TestCase):
    def test_manifest_rejects_unknown_wildcard_and_duplicate_permissions(self) -> None:
        with self.assertRaisesRegex(SkillPermissionError, "unknown_permission"):
            validate_skill_manifest(_manifest(permissions=["invoke.shell.execute"]))
        with self.assertRaisesRegex(SkillPermissionError, "wildcard_permission_denied"):
            validate_skill_manifest(_manifest(permissions=["invoke.*"]))
        with self.assertRaisesRegex(SkillPermissionError, "duplicate_permission"):
            validate_skill_manifest(_manifest(permissions=["read.notifications.inspect", "read.notifications.inspect"]))

    def test_manifest_rejects_broad_roots_and_wildcard_domains(self) -> None:
        with self.assertRaisesRegex(SkillPermissionError, "broad_filesystem_root_denied"):
            validate_skill_manifest({**_manifest(permissions=["read.notifications.inspect"]), "filesystem_roots": ["/"]})
        with self.assertRaisesRegex(SkillPermissionError, "wildcard_domain_denied"):
            validate_skill_manifest({**_manifest(permissions=["read.notifications.inspect"]), "network_domains": ["*.example.com"]})

    def test_grant_binds_identity_version_and_scope(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            manifest = _manifest(permissions=["invoke.files.create"])
            identity = build_skill_identity(manifest, install_source="fixture", install_path=str(tmp / "skill"))
            store = SkillGrantStore(tmp / "grants.json")
            root = tmp / "allowed"
            root.mkdir()
            store.create_grant(identity=identity, permission_id="invoke.files.create", target_scope={"root": str(root)}, granted_by="local_operator_cli")
            allowed, reason = store.effective_grant(identity=identity, permission_id="invoke.files.create", target={"target_path": str(root / "x.txt")})
            self.assertIsNotNone(allowed)
            self.assertEqual("allowed", reason)
            denied, reason = store.effective_grant(identity=identity, permission_id="invoke.files.create", target={"target_path": str(tmp / "outside.txt")})
            self.assertIsNone(denied)
            self.assertEqual("target_scope_denied", reason)
            changed_identity = build_skill_identity(_manifest(permissions=["invoke.files.create"], version="2.0.0"), install_source="fixture", install_path=str(tmp / "skill"))
            missing, reason = store.effective_grant(identity=changed_identity, permission_id="invoke.files.create", target={"target_path": str(root / "x.txt")})
            self.assertIsNone(missing)
            self.assertEqual("grant_missing", reason)

    def test_broker_denies_undeclared_and_ungranted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            manifest = _manifest(permissions=["read.notifications.inspect"])
            identity = build_skill_identity(manifest, install_source="fixture", install_path=str(tmp / "skill"))
            broker = SkillPackInvocationBroker(grant_store=SkillGrantStore(tmp / "grants.json"))
            undeclared = broker.inspect(identity=identity, manifest=manifest, permission_id="read.git.inspect", target={})
            self.assertFalse(undeclared["ok"])
            self.assertEqual("permission_not_declared", undeclared["reason_code"])
            ungranted = broker.inspect(identity=identity, manifest=manifest, permission_id="read.notifications.inspect", target={})
            self.assertFalse(ungranted["ok"])
            self.assertEqual("grant_missing", ungranted["reason_code"])

    def test_permission_diff_marks_new_request(self) -> None:
        diff = diff_skill_permissions(
            _manifest(permissions=["read.notifications.inspect"]),
            _manifest(permissions=["read.notifications.inspect", "invoke.notification.local.send"], version="1.1.0"),
        )
        self.assertEqual(["invoke.notification.local.send"], diff["newly_requested"])
        self.assertEqual(["read.notifications.inspect"], diff["unchanged"])


if __name__ == "__main__":
    unittest.main()
