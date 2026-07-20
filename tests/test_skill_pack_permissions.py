from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from unittest.mock import patch

from agent.skill_pack_permissions import (
    SkillGrantStore,
    SkillPackInvocationBroker,
    SkillPermissionError,
    build_skill_identity,
    diff_skill_permissions,
    validate_skill_manifest,
)
from agent.executor_registry import ExecutorRegistry, ExecutorSpec


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

    def test_mutation_requires_scoped_single_use_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            root = tmp / "allowed"
            root.mkdir()
            manifest = _manifest(permissions=["invoke.files.create"])
            identity = build_skill_identity(manifest, install_source="fixture", install_path=str(tmp / "skill"))
            store = SkillGrantStore(tmp / "grants.json")
            store.create_grant(
                identity=identity,
                permission_id="invoke.files.create",
                target_scope={"root": str(root)},
                granted_by="local_operator_cli",
            )
            calls: list[dict[str, object]] = []

            def _fixture_executor(_plan: dict[str, object], action: dict[str, object]) -> dict[str, object]:
                calls.append(action)
                return {"ok": True, "mutated": True, "resources_touched": [str(root / "x.txt")]}

            registry = ExecutorRegistry(tmp / "journal.jsonl")
            registry.register(ExecutorSpec(
                executor_id="operator.file.create.v1",
                action_type="operator.file.create",
                status="enabled",
                run=_fixture_executor,
                capability_id="files.create",
            ))
            registry.freeze()
            broker = SkillPackInvocationBroker(grant_store=store, executor_registry=registry)
            target = {"target_path": str(root / "x.txt")}
            action = {"target_path": str(root / "x.txt"), "content": "hello"}
            preview = broker.request_action(
                identity=identity, manifest=manifest, permission_id="invoke.files.create",
                target=target, action_payload=action, actor_id="alice", thread_id="thread-a", session_id="session-a",
            )
            self.assertEqual("confirmation_required", preview["status"])
            self.assertEqual([], calls)

            changed = broker.confirm_action(
                plan_id=preview["plan_id"], confirmation_id="confirm-wrong", identity=identity,
                manifest=manifest, permission_id="invoke.files.create", target=target,
                action_payload={**action, "content": "changed"}, actor_id="alice", thread_id="thread-a", session_id="session-a",
            )
            self.assertEqual("skill_invocation_scope_changed", changed["reason_code"])
            self.assertEqual([], calls)

            applied = broker.confirm_action(
                plan_id=preview["plan_id"], confirmation_id="confirm-one", identity=identity,
                manifest=manifest, permission_id="invoke.files.create", target=target,
                action_payload=action, actor_id="alice", thread_id="thread-a", session_id="session-a",
            )
            self.assertTrue(applied["ok"])
            self.assertEqual(1, len(calls))
            replay = broker.confirm_action(
                plan_id=preview["plan_id"], confirmation_id="confirm-two", identity=identity,
                manifest=manifest, permission_id="invoke.files.create", target=target,
                action_payload=action, actor_id="alice", thread_id="thread-a", session_id="session-a",
            )
            self.assertEqual("skill_invocation_plan_not_pending", replay["reason_code"])
            self.assertEqual(1, len(calls))

    def test_mutation_confirmation_revalidates_pack_and_grant(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            root = tmp / "allowed"
            root.mkdir()
            manifest = _manifest(permissions=["invoke.files.create"])
            identity = build_skill_identity(manifest, install_source="fixture", install_path=str(tmp / "skill"))
            store = SkillGrantStore(tmp / "grants.json")
            grant = store.create_grant(identity=identity, permission_id="invoke.files.create", target_scope={"root": str(root)}, granted_by="local_operator_cli")
            registry = ExecutorRegistry(tmp / "journal.jsonl")
            registry.register(ExecutorSpec(executor_id="operator.file.create.v1", action_type="operator.file.create", status="enabled", capability_id="files.create", run=lambda _p, _a: {"ok": True, "mutated": True}))
            registry.freeze()
            broker = SkillPackInvocationBroker(grant_store=store, executor_registry=registry)
            target = {"target_path": str(root / "x.txt")}
            preview = broker.request_action(identity=identity, manifest=manifest, permission_id="invoke.files.create", target=target, action_payload={"content": "x"})
            store.revoke_grant(grant["grant_id"])
            blocked = broker.confirm_action(
                plan_id=preview["plan_id"], confirmation_id="confirm", identity=identity,
                manifest=manifest, permission_id="invoke.files.create", target=target,
                action_payload={"content": "x"},
            )
            self.assertEqual("grant_missing", blocked["reason_code"])

    def test_confirmation_rejects_cross_scope_and_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            root = tmp / "allowed"
            root.mkdir()
            manifest = _manifest(permissions=["invoke.files.create"])
            identity = build_skill_identity(manifest, install_source="fixture", install_path=str(tmp / "skill"))
            store = SkillGrantStore(tmp / "grants.json")
            store.create_grant(identity=identity, permission_id="invoke.files.create", target_scope={"root": str(root)}, granted_by="local_operator_cli")
            registry = ExecutorRegistry(tmp / "journal.jsonl")
            registry.register(ExecutorSpec(executor_id="operator.file.create.v1", action_type="operator.file.create", status="enabled", capability_id="files.create", run=lambda _p, _a: {"ok": True, "mutated": True}))
            registry.freeze()
            broker = SkillPackInvocationBroker(grant_store=store, executor_registry=registry)
            target = {"target_path": str(root / "x.txt")}
            action = {"content": "x"}
            preview = broker.request_action(identity=identity, manifest=manifest, permission_id="invoke.files.create", target=target, action_payload=action, actor_id="alice", thread_id="a", session_id="s")
            crossed = broker.confirm_action(plan_id=preview["plan_id"], confirmation_id="x", identity=identity, manifest=manifest, permission_id="invoke.files.create", target=target, action_payload=action, actor_id="alice", thread_id="b", session_id="s")
            self.assertEqual("skill_invocation_thread_id_mismatch", crossed["reason_code"])

            with patch("agent.skill_pack_permissions.time.time", return_value=1):
                expired = broker.request_action(identity=identity, manifest=manifest, permission_id="invoke.files.create", target=target, action_payload=action)
            result = broker.confirm_action(plan_id=expired["plan_id"], confirmation_id="expired", identity=identity, manifest=manifest, permission_id="invoke.files.create", target=target, action_payload=action)
            self.assertEqual("mutation_confirmation_expired", result["error_code"])

    def test_concurrent_double_confirmation_executes_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            root = tmp / "allowed"
            root.mkdir()
            manifest = _manifest(permissions=["invoke.files.create"])
            identity = build_skill_identity(manifest, install_source="fixture", install_path=str(tmp / "skill"))
            store = SkillGrantStore(tmp / "grants.json")
            store.create_grant(identity=identity, permission_id="invoke.files.create", target_scope={"root": str(root)}, granted_by="local_operator_cli")
            calls: list[int] = []
            registry = ExecutorRegistry(tmp / "journal.jsonl")
            registry.register(ExecutorSpec(executor_id="operator.file.create.v1", action_type="operator.file.create", status="enabled", capability_id="files.create", run=lambda _p, _a: (calls.append(1) or {"ok": True, "mutated": True})))
            registry.freeze()
            broker = SkillPackInvocationBroker(grant_store=store, executor_registry=registry)
            target = {"target_path": str(root / "x.txt")}
            action = {"content": "x"}
            preview = broker.request_action(identity=identity, manifest=manifest, permission_id="invoke.files.create", target=target, action_payload=action)

            def confirm(token: str) -> dict[str, object]:
                return broker.confirm_action(plan_id=preview["plan_id"], confirmation_id=token, identity=identity, manifest=manifest, permission_id="invoke.files.create", target=target, action_payload=action)

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(confirm, ("one", "two")))
            self.assertEqual(1, sum(bool(row.get("ok")) for row in results))
            self.assertEqual(1, len(calls))


if __name__ == "__main__":
    unittest.main()
