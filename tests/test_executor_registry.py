from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.host_lifecycle import (
    HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION,
    HOST_LIFECYCLE_RUNNER_VERSION,
    _validate_proof_service_name,
    _validate_update_service_name,
    attach_approved_hash,
    load_and_validate_operation,
    write_json_atomic,
)
from agent.version import read_git_commit
from agent.executor_registry import (
    BACKUP_SCHEMA_VERSION,
    BACKUP_MAX_FILE_BYTES,
    BACKUP_MAX_TOTAL_BYTES,
    EXECUTOR_JOURNAL_MAX_RECORD_BYTES,
    RESTORE_V1_CAPABILITY,
    SUPPORT_BUNDLE_SCHEMA_VERSION,
    ExecutorRegistry,
    ExecutorPartialFailure,
    ExecutorSpec,
    create_additive_backup,
    create_redacted_support_bundle,
    execute_cleanup,
    execute_uninstall_v1,
    execute_update_v1,
    restore_backup_v1,
    support_bundle_redact,
)


def _plan(**overrides):
    payload = {
        "plan_id": "confirm-test",
        "action_type": "operator.support_bundle",
        "target": "support bundle",
        "risk_level": "low",
        "executor_status": "enabled",
    }
    payload.update(overrides)
    return payload


def _write_release(path: Path, commit: str) -> None:
    (path / "agent").mkdir(parents=True)
    (path / "agent" / "BUILD_INFO.json").write_text(
        json.dumps({"git_commit": commit, "version": "fixture"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _host_lifecycle_record(path: Path) -> dict:
    root = path.parent
    return attach_approved_hash(
        {
            "schema_version": HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION,
            "runner_version": HOST_LIFECYCLE_RUNNER_VERSION,
            "operation_id": "host-test",
            "operation_type": "update",
            "plan_id": "host-test",
            "created_at": "2026-07-08T00:00:00+00:00",
            "fixture_mode": "strict",
            "state_root": str(root / "state"),
            "runtime_root": str(root / "runtime"),
            "releases_root": str(root / "runtime/releases"),
            "current_link": str(root / "runtime/current"),
            "staged_source_path": str(root / "source-release-b"),
            "target_release_id": "release-b",
            "current_runtime_commit": "commit-a",
            "target_commit": "commit-b",
            "operation_state_path": str(root / "state/host_lifecycle/operations/host-test/state.json"),
            "receipt_path": str(root / "state/host_lifecycle/operations/host-test/receipt.json"),
        }
    )


def _uninstall_fixture(root: Path) -> tuple[dict, dict]:
    fixture = root / "fixture-install"
    runtime = fixture / "personal-agent/runtime"
    releases = runtime / "releases"
    state = fixture / "personal-agent"
    config = fixture / "config/systemd/user"
    launchers = fixture / "applications"
    icons = fixture / "icons"
    repo = fixture / "repo"
    backups = state / "backups"
    secrets = state / "secrets.enc.json"
    release = releases / "0.2.0"
    for path in (release, state, config, launchers, icons, repo, backups):
        path.mkdir(parents=True, exist_ok=True)
    (runtime / "current").symlink_to(release)
    (config / "personal-agent-api.service").write_text("[Service]\n", encoding="utf-8")
    (config / "personal-agent-telegram.service").write_text("[Service]\n", encoding="utf-8")
    (launchers / "personal-agent.desktop").write_text("[Desktop Entry]\n", encoding="utf-8")
    (icons / "personal-agent.svg").write_text("<svg />\n", encoding="utf-8")
    (runtime / "install-manifest.json").write_text("{}\n", encoding="utf-8")
    (state / "agent.db").write_text("memory\n", encoding="utf-8")
    secrets.write_text('{"token":"secret"}\n', encoding="utf-8")
    (backups / "existing").mkdir()
    (repo / "README.md").write_text("repo\n", encoding="utf-8")
    removable_roots = [str(runtime), str(config), str(launchers), str(icons)]
    removable = [
        {"id": "release", "class": "runtime release", "path": str(release), "owned": True, "expected_type": "directory"},
        {"id": "current", "class": "runtime symlink", "path": str(runtime / "current"), "owned": True, "expected_type": "symlink"},
        {"id": "manifest", "class": "install metadata", "path": str(runtime / "install-manifest.json"), "owned": True, "expected_type": "file"},
        {"id": "api-service", "class": "service unit", "path": str(config / "personal-agent-api.service"), "owned": True, "expected_type": "file"},
        {"id": "telegram-service", "class": "service unit", "path": str(config / "personal-agent-telegram.service"), "owned": True, "expected_type": "file"},
        {"id": "desktop-entry", "class": "desktop entry", "path": str(launchers / "personal-agent.desktop"), "owned": True, "expected_type": "file"},
        {"id": "desktop-icon", "class": "desktop icon", "path": str(icons / "personal-agent.svg"), "owned": True, "expected_type": "file"},
    ]
    preserved = [
        {"id": "state", "class": "state root", "path": str(state)},
        {"id": "secret-store", "class": "secret store", "path": str(secrets)},
        {"id": "backups", "class": "backup root", "path": str(backups)},
        {"id": "repo", "class": "repository", "path": str(repo)},
    ]
    snapshot = {
        "fixture_marker": True,
        "fixture_root": str(fixture),
        "mode": "preserve_data",
        "state_root": str(state),
        "backup_root": str(backups),
        "receipt_root": str(state / "uninstall_receipts"),
        "runtime_commit": "fixture",
        "removable_roots": removable_roots,
        "removable_resources": removable,
        "preserved_resources": preserved,
    }
    action = {
        "pending_id": "confirm-test",
        "operation_id": "uninstall-test",
        "uninstall_mode": "preserve_data",
        "uninstall_execution_mode": "fixture_preserve_data",
        "state_root": str(state),
        "backup_root": str(backups),
        "receipt_root": str(state / "uninstall_receipts"),
        "target_snapshot": snapshot,
    }
    return snapshot, action


class ExecutorRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.journal_path = Path(self.tmpdir.name) / "executor_journal.jsonl"

    def test_registry_lookup_by_action_type(self) -> None:
        registry = ExecutorRegistry(self.journal_path)
        spec = ExecutorSpec(executor_id="test.executor", action_type="operator.support_bundle", status="enabled")
        registry.register(spec)
        self.assertEqual(spec, registry.lookup("operator.support_bundle"))

    def test_host_lifecycle_record_validates_and_rejects_tamper(self) -> None:
        path = Path(self.tmpdir.name) / "host" / "operation.json"
        record = _host_lifecycle_record(path)
        write_json_atomic(path, record)
        _, loaded = load_and_validate_operation(path, expected_type="update")
        self.assertEqual("host-test", loaded["operation_id"])

        record["target_commit"] = "evil"
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "operation_record_tampered"):
            load_and_validate_operation(path, expected_type="update")

    def test_host_lifecycle_record_rejects_command_field(self) -> None:
        path = Path(self.tmpdir.name) / "host-command" / "operation.json"
        record = _host_lifecycle_record(path)
        record["command"] = "rm -rf /"
        record = attach_approved_hash(record)
        write_json_atomic(path, record)
        with self.assertRaisesRegex(ValueError, "arbitrary_command_field_rejected"):
            load_and_validate_operation(path, expected_type="update")

    def test_host_lifecycle_proof_service_name_allowlist(self) -> None:
        self.assertEqual(
            "personal-agent-active-host-proof-api.service",
            _validate_proof_service_name("personal-agent-active-host-proof-api.service"),
        )
        with self.assertRaisesRegex(ValueError, "host_lifecycle_service_not_allowlisted"):
            _validate_proof_service_name("personal-agent-api.service")
        with self.assertRaisesRegex(ValueError, "host_lifecycle_service_invalid"):
            _validate_proof_service_name("personal-agent-active-host-proof-api\n.service")

    def test_host_lifecycle_primary_update_service_allowlist(self) -> None:
        self.assertEqual(
            "personal-agent-api.service",
            _validate_update_service_name("personal-agent-api.service", fixture_mode="primary_update_proof"),
        )
        with self.assertRaisesRegex(ValueError, "host_lifecycle_service_not_allowlisted"):
            _validate_update_service_name("personal-agent-telegram.service", fixture_mode="primary_update_proof")
        with self.assertRaisesRegex(ValueError, "host_lifecycle_service_not_allowlisted"):
            _validate_update_service_name("personal-agent-api.service", fixture_mode="active_host_proof")

    def test_git_commit_override_is_explicit(self) -> None:
        with patch.dict(os.environ, {"PERSONAL_AGENT_GIT_COMMIT_OVERRIDE": "active-host-A"}):
            self.assertEqual("active-host-A", read_git_commit(repo_root=Path(self.tmpdir.name)))

    def test_preview_only_refusal_is_journaled(self) -> None:
        registry = ExecutorRegistry(self.journal_path)
        result = registry.execute_confirmed_plan(
            plan=_plan(action_type="operator.cleanup", executor_status="preview_only"),
            action={"pending_id": "confirm-test"},
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.mutated)
        self.assertEqual("executor_not_enabled", result.error_code)
        self.assertTrue(result.journal_id)
        recent = registry.journal.recent()
        self.assertEqual(1, len(recent))
        self.assertEqual("executor_not_enabled", recent[0]["result"]["error_code"])

    def test_unavailable_refusal_is_journaled(self) -> None:
        registry = ExecutorRegistry(self.journal_path)
        result = registry.execute_confirmed_plan(
            plan=_plan(executor_status="unavailable"),
            action={"pending_id": "confirm-test"},
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.mutated)
        self.assertEqual("executor_unavailable", result.error_code)
        self.assertTrue(result.journal_id)

    def test_enabled_executor_result_schema(self) -> None:
        registry = ExecutorRegistry(self.journal_path)

        def _run(plan, action):
            return {
                "ok": True,
                "mutated": True,
                "resources_touched": ["/tmp/example"],
                "rollback_available": True,
                "rollback_hint": "remove /tmp/example",
                "user_message": "created example",
            }

        registry.register(
            ExecutorSpec(
                executor_id="test.enabled",
                action_type="operator.support_bundle",
                status="enabled",
                run=_run,
            )
        )
        result = registry.execute_confirmed_plan(plan=_plan(), action={"pending_id": "confirm-test"})
        payload = result.to_dict()
        for key in (
            "ok",
            "mutated",
            "executor_id",
            "plan_id",
            "action_type",
            "target",
            "started_at",
            "finished_at",
            "resources_touched",
            "journal_id",
            "rollback_available",
            "rollback_hint",
            "error_code",
            "user_message",
        ):
            self.assertIn(key, payload)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["mutated"])
        self.assertEqual("test.enabled", payload["executor_id"])

    def test_update_executor_fixture_promotes_staged_release(self) -> None:
        root = Path(self.tmpdir.name)
        runtime = root / "runtime"
        releases = runtime / "releases"
        state = root / "state"
        release_a = releases / "release-a"
        source_b = root / "source-release-b"
        releases.mkdir(parents=True)
        _write_release(release_a, "commit-a")
        _write_release(source_b, "commit-b")
        (runtime / "current").symlink_to(release_a)
        plan = _plan(action_type="operator.update", target="Personal Agent update")
        action = {
            "pending_id": "confirm-test",
            "update_mode": "fixture_staged_release",
            "state_root": str(state),
            "runtime_root": str(runtime),
            "releases_root": str(releases),
            "current_link": str(runtime / "current"),
            "staged_source_path": str(source_b),
            "target_release_id": "release-b",
            "expected_current_commit": "commit-a",
            "target_commit": "commit-b",
            "preview_target_commit": "commit-b",
            "working_tree_clean": True,
        }

        result = execute_update_v1(plan, action)

        self.assertTrue(result["ok"])
        self.assertTrue(result["mutated"])
        self.assertEqual("completed_verified", result["details"]["status"])
        self.assertEqual((runtime / "current").resolve(), releases / "release-b")
        self.assertTrue((state / "update_checkpoints" / "confirm-test" / "manifest.json").is_file())

    def test_update_executor_fixture_rolls_back_failed_verification(self) -> None:
        root = Path(self.tmpdir.name)
        runtime = root / "runtime"
        releases = runtime / "releases"
        state = root / "state"
        release_a = releases / "release-a"
        source_b = root / "source-release-b"
        releases.mkdir(parents=True)
        _write_release(release_a, "commit-a")
        _write_release(source_b, "commit-b")
        (runtime / "current").symlink_to(release_a)
        plan = _plan(action_type="operator.update", target="Personal Agent update")
        action = {
            "pending_id": "confirm-test",
            "update_mode": "fixture_staged_release",
            "state_root": str(state),
            "runtime_root": str(runtime),
            "releases_root": str(releases),
            "current_link": str(runtime / "current"),
            "staged_source_path": str(source_b),
            "target_release_id": "release-b",
            "expected_current_commit": "commit-a",
            "target_commit": "commit-b",
            "preview_target_commit": "commit-b",
            "working_tree_clean": True,
            "force_post_promotion_failure": True,
        }

        result = execute_update_v1(plan, action)

        self.assertFalse(result["ok"])
        self.assertTrue(result["mutated"])
        self.assertEqual("update_failed_rolled_back", result["details"]["status"])
        self.assertEqual((runtime / "current").resolve(), release_a)

    def test_update_executor_primary_handoff_returns_in_progress(self) -> None:
        root = Path(self.tmpdir.name)
        fake_home = root / "home"
        state = fake_home / ".local/share/personal-agent"
        runtime = state / "runtime"
        releases = runtime / "releases"
        release_a = releases / "release-a"
        source_b = state / "host_lifecycle/staged_sources/source-b"
        releases.mkdir(parents=True)
        _write_release(release_a, "commit-a")
        _write_release(source_b, "commit-a")
        (runtime / "current").symlink_to(release_a)
        marker = state / "host_lifecycle/primary_update_enablement.marker"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok\n", encoding="utf-8")
        plan = _plan(action_type="operator.update", target="Personal Agent update")
        action = {
            "pending_id": "confirm-test",
            "operation_id": "primary-update-test",
            "update_mode": "primary_staged_release",
            "state_root": str(state),
            "runtime_root": str(runtime),
            "releases_root": str(releases),
            "current_link": str(runtime / "current"),
            "staged_source_path": str(source_b),
            "target_release_id": "release-b",
            "expected_current_commit": "commit-a",
            "target_commit": "commit-a",
            "preview_target_commit": "commit-a",
            "working_tree_clean": True,
            "api_service_name": "personal-agent-api.service",
            "verify_base_url": "http://127.0.0.1:8765",
            "proof_marker_path": str(marker),
            "expires_at": 1783556574,
        }

        with patch("agent.executor_registry.Path.home", return_value=fake_home), patch(
            "agent.executor_registry._launch_host_lifecycle_runner_systemd",
            return_value={"ok": True, "unit": "personal-agent-host-lifecycle-update-primary-update-test.service", "returncode": 0},
        ) as launch:
            result = execute_update_v1(plan, action)

        self.assertTrue(result["ok"])
        self.assertTrue(result["mutated"])
        self.assertEqual("in_progress", result["details"]["status"])
        launch.assert_called_once()
        record_path = state / "host_lifecycle/operations/primary-update-test/operation.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual("primary_update_proof", record["fixture_mode"])
        self.assertEqual("2026-07-09T00:22:54+00:00", record["expires_at"])
        self.assertEqual("personal-agent-api.service", record["api_service_name"])
        self.assertEqual("http://127.0.0.1:8765", record["verify_base_url"])

    def test_update_executor_primary_handoff_rejects_path_escape(self) -> None:
        root = Path(self.tmpdir.name)
        fake_home = root / "home"
        state = fake_home / ".local/share/personal-agent"
        runtime = state / "runtime"
        releases = runtime / "releases"
        release_a = releases / "release-a"
        source_b = root / "outside-source"
        releases.mkdir(parents=True)
        _write_release(release_a, "commit-a")
        _write_release(source_b, "commit-a")
        (runtime / "current").symlink_to(release_a)
        marker = state / "host_lifecycle/primary_update_enablement.marker"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok\n", encoding="utf-8")
        plan = _plan(action_type="operator.update", target="Personal Agent update")

        with patch("agent.executor_registry.Path.home", return_value=fake_home):
            result = execute_update_v1(
                plan,
                {
                    "pending_id": "confirm-test",
                    "operation_id": "primary-update-test",
                    "update_mode": "primary_staged_release",
                    "state_root": str(state),
                    "runtime_root": str(runtime),
                    "releases_root": str(releases),
                    "current_link": str(runtime / "current"),
                    "staged_source_path": str(source_b),
                    "target_release_id": "release-b",
                    "expected_current_commit": "commit-a",
                    "target_commit": "commit-a",
                    "preview_target_commit": "commit-a",
                    "working_tree_clean": True,
                    "api_service_name": "personal-agent-api.service",
                    "verify_base_url": "http://127.0.0.1:8765",
                    "proof_marker_path": str(marker),
                },
            )
        self.assertFalse(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertEqual("update_primary_staged_source_escape", result["error_code"])

    def test_update_executor_refuses_dirty_tree_and_target_drift(self) -> None:
        plan = _plan(action_type="operator.update", target="Personal Agent update")
        dirty = execute_update_v1(
            plan,
            {
                "pending_id": "confirm-test",
                "update_mode": "fixture_staged_release",
                "working_tree_clean": False,
                "dirty_files": ["M agent/orchestrator.py"],
                "target_commit": "commit-b",
                "preview_target_commit": "commit-b",
            },
        )
        self.assertFalse(dirty["ok"])
        self.assertFalse(dirty["mutated"])
        self.assertEqual("update_dirty_working_tree", dirty["error_code"])

        drift = execute_update_v1(
            plan,
            {
                "pending_id": "confirm-test",
                "update_mode": "fixture_staged_release",
                "working_tree_clean": True,
                "target_commit": "commit-b",
                "preview_target_commit": "commit-old",
            },
        )
        self.assertFalse(drift["ok"])
        self.assertFalse(drift["mutated"])
        self.assertEqual("update_target_changed_since_preview", drift["error_code"])

    def test_uninstall_executor_fixture_removes_runtime_and_preserves_data(self) -> None:
        root = Path(self.tmpdir.name)
        snapshot, action = _uninstall_fixture(root)
        plan = _plan(action_type="operator.uninstall", target="Personal Agent uninstall", risk_level="high")
        result = execute_uninstall_v1(plan, action)

        fixture = Path(snapshot["fixture_root"])
        self.assertTrue(result["ok"])
        self.assertTrue(result["mutated"])
        self.assertEqual("completed_verified", result["details"]["status"])
        self.assertFalse((fixture / "personal-agent/runtime/releases/0.2.0").exists())
        self.assertFalse((fixture / "config/systemd/user/personal-agent-api.service").exists())
        self.assertTrue((fixture / "personal-agent/agent.db").exists())
        self.assertTrue((fixture / "personal-agent/secrets.enc.json").exists())
        self.assertTrue((fixture / "repo/README.md").exists())
        self.assertTrue(Path(result["details"]["final_backup_path"]).is_dir())
        self.assertTrue(Path(result["details"]["receipt_path"]).is_file())

    def test_uninstall_executor_blocks_live_and_snapshot_drift(self) -> None:
        plan = _plan(action_type="operator.uninstall", target="Personal Agent uninstall", risk_level="high")
        live = execute_uninstall_v1(plan, {"pending_id": "confirm-test", "uninstall_execution_mode": "live_guarded"})
        self.assertFalse(live["ok"])
        self.assertFalse(live["mutated"])
        self.assertEqual("uninstall_live_execution_not_enabled", live["error_code"])

        snapshot, action = _uninstall_fixture(Path(self.tmpdir.name))
        action["target_snapshot_hash"] = "not-the-real-hash"
        drift = execute_uninstall_v1(plan, action)
        self.assertFalse(drift["ok"])
        self.assertFalse(drift["mutated"])
        self.assertEqual("uninstall_target_changed_since_preview", drift["error_code"])

    def test_uninstall_executor_rejects_symlink_escape(self) -> None:
        root = Path(self.tmpdir.name)
        snapshot, action = _uninstall_fixture(root)
        fixture = Path(snapshot["fixture_root"])
        escape = fixture / "personal-agent/runtime/escape"
        escape.symlink_to(Path("/tmp"))
        snapshot["removable_resources"].append(
            {"id": "escape", "class": "runtime symlink", "path": str(escape), "owned": True, "expected_type": "symlink"}
        )
        plan = _plan(action_type="operator.uninstall", target="Personal Agent uninstall", risk_level="high")
        result = execute_uninstall_v1(plan, action)
        self.assertFalse(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertEqual("resource_symlink_escape", result["error_code"])

    def test_journal_redacts_secrets(self) -> None:
        registry = ExecutorRegistry(self.journal_path)
        result = registry.execute_confirmed_plan(
            plan=_plan(confirmation_token="confirm-secret-token", secret_key="ultrasecret"),
            action={"pending_id": "confirm-test", "confirmation_token": "confirm-secret-token", "api_key": "sk-test"},
        )
        self.assertFalse(result.ok)
        text = self.journal_path.read_text(encoding="utf-8")
        self.assertNotIn("confirm-secret-token", text)
        self.assertNotIn("ultrasecret", text)
        self.assertNotIn("sk-test", text)
        self.assertIn("[REDACTED]", text)

    def test_recent_journal_reads_bounded_tail_and_skips_huge_lines(self) -> None:
        huge = json.dumps({"journal_id": "huge", "payload": "x" * (128 * 1024)})
        small = json.dumps({"journal_id": "small", "result": {"ok": True}})
        self.journal_path.write_text(huge + "\n" + small + "\n", encoding="utf-8")
        journal = ExecutorRegistry(self.journal_path).journal
        recent = journal.recent(limit=20, max_tail_bytes=256 * 1024, max_line_bytes=64 * 1024)
        self.assertEqual(["small"], [entry.get("journal_id") for entry in recent])

    def test_exact_plan_id_binding_before_execution(self) -> None:
        registry = ExecutorRegistry(self.journal_path)
        called = False

        def _run(plan, action):
            nonlocal called
            called = True
            return {"ok": True, "mutated": True}

        registry.register(ExecutorSpec(executor_id="test.enabled", action_type="operator.support_bundle", status="enabled", run=_run))
        result = registry.execute_confirmed_plan(plan=_plan(plan_id="confirm-a"), action={"pending_id": "confirm-b"})
        self.assertFalse(result.ok)
        self.assertFalse(result.mutated)
        self.assertEqual("plan_id_mismatch", result.error_code)
        self.assertFalse(called)

    def test_executor_exception_reports_no_verified_mutation(self) -> None:
        registry = ExecutorRegistry(self.journal_path)

        def _run(plan, action):
            raise RuntimeError("boom")

        registry.register(ExecutorSpec(executor_id="test.boom", action_type="operator.support_bundle", status="enabled", run=_run))
        result = registry.execute_confirmed_plan(plan=_plan(), action={"pending_id": "confirm-test"})
        self.assertFalse(result.ok)
        self.assertFalse(result.mutated)
        self.assertEqual("executor_exception_before_verified_mutation", result.error_code)
        self.assertEqual("RuntimeError", result.details.get("exception"))

    def test_executor_partial_failure_records_artifact_and_rollback_scope(self) -> None:
        registry = ExecutorRegistry(self.journal_path)

        def _run(plan, action):
            raise ExecutorPartialFailure(
                "created partial artifact",
                resources_touched=["/tmp/personal-agent-partial/summary.json"],
                rollback_hint="Remove only /tmp/personal-agent-partial.",
                details={"artifact_path": "/tmp/personal-agent-partial"},
            )

        registry.register(ExecutorSpec(executor_id="test.partial", action_type="operator.support_bundle", status="enabled", run=_run))
        result = registry.execute_confirmed_plan(plan=_plan(), action={"pending_id": "confirm-test"})
        self.assertFalse(result.ok)
        self.assertFalse(result.mutated)
        self.assertEqual("executor_partial_failure", result.error_code)
        self.assertEqual(["/tmp/personal-agent-partial/summary.json"], result.resources_touched)
        self.assertIn("Remove only", result.rollback_hint)
        self.assertEqual("/tmp/personal-agent-partial", result.details.get("artifact_path"))

    def test_cleanup_executor_deletes_only_previewed_owned_support_artifact(self) -> None:
        tmp_root = Path(self.tmpdir.name) / "tmp"
        tmp_root.mkdir()
        support = tmp_root / "personal-agent-support-old"
        support.mkdir()
        (support / "summary.json").write_text("{}", encoding="utf-8")
        preview = {
            "candidates": [
                {
                    "path": str(support),
                    "canonical_path": str(support.resolve()),
                    "classification": "old support bundle artifact",
                    "safe_to_delete_later": True,
                    "size_bytes": sum(path.lstat().st_size for path in support.rglob("*")),
                    "file_count": 1,
                }
            ],
            "protected": [],
        }

        with patch("tempfile.gettempdir", return_value=str(tmp_root)):
            result = execute_cleanup(_plan(action_type="operator.cleanup"), {"cleanup_preview": preview})

        self.assertTrue(result["ok"])
        self.assertTrue(result["mutated"])
        self.assertFalse(support.exists())
        self.assertEqual("operator.cleanup.v1", result["executor_id"])
        self.assertEqual("completed", result["details"]["status"])

    def test_cleanup_executor_skips_candidate_changed_after_preview(self) -> None:
        tmp_root = Path(self.tmpdir.name) / "tmp"
        tmp_root.mkdir()
        support = tmp_root / "personal-agent-support-old"
        support.mkdir()
        (support / "summary.json").write_text("{}", encoding="utf-8")
        preview = {
            "candidates": [
                {
                    "path": str(support),
                    "canonical_path": str(support.resolve()),
                    "classification": "old support bundle artifact",
                    "safe_to_delete_later": True,
                    "size_bytes": 1,
                    "file_count": 1,
                }
            ],
            "protected": [],
        }

        with patch("tempfile.gettempdir", return_value=str(tmp_root)):
            result = execute_cleanup(_plan(action_type="operator.cleanup"), {"cleanup_preview": preview})

        self.assertTrue(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertTrue(support.exists())
        self.assertEqual("no_op", result["details"]["status"])
        self.assertEqual("cleanup_candidate_changed_after_preview", result["details"]["skipped"][0]["reason"])

    def test_cleanup_executor_rejects_symlink_candidate(self) -> None:
        tmp_root = Path(self.tmpdir.name) / "tmp"
        tmp_root.mkdir()
        support = tmp_root / "personal-agent-support-old"
        support.mkdir()
        outside = Path(self.tmpdir.name) / "outside"
        outside.mkdir()
        (support / "escape").symlink_to(outside)
        preview = {
            "candidates": [
                {
                    "path": str(support),
                    "canonical_path": str(support.resolve()),
                    "classification": "old support bundle artifact",
                    "safe_to_delete_later": True,
                    "size_bytes": sum(path.lstat().st_size for path in support.rglob("*")),
                    "file_count": 1,
                }
            ],
            "protected": [],
        }

        with patch("tempfile.gettempdir", return_value=str(tmp_root)):
            result = execute_cleanup(_plan(action_type="operator.cleanup"), {"cleanup_preview": preview})

        self.assertTrue(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertTrue(support.exists())
        self.assertEqual("cleanup_symlink_protected", result["details"]["skipped"][0]["reason"])

    def _restore_fixture_db(self, root: Path, *, initial_value: str | None = None) -> Path:
        db_path = root / "agent.db"
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE preferences (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
            if initial_value is not None:
                conn.execute(
                    "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
                    ("system_resource_baseline_v1", initial_value, "before"),
                )
            conn.commit()
        finally:
            conn.close()
        return db_path

    def _restore_fixture_backup(self, root: Path, *, value: str = "restored") -> Path:
        backup = root / "backups" / "personal-agent-backup-fixture"
        backup.mkdir(parents=True)
        files = {
            "backup_summary.json": {"backup_schema_version": "backup.v1"},
            "diagnostics_summary.json": {"ok": True},
            "executor_registry_journal_summary.json": {"entries": []},
            "memory_anchors_summary.json": {"mode": "summary_only_raw_memory_text_excluded"},
            "pack_metadata_summary.json": {"raw_pack_text": "excluded"},
            "preferences_summary.json": {
                "mode": "allowlisted_restore_export",
                "preferences": [{"key": "system_resource_baseline_v1", "value": value, "restore_supported": True}],
            },
            "runtime_config_summary.json": {"version": {"git_commit": "abc123"}},
            "state_database_summary.json": {"mode": "summary_only_raw_database_excluded"},
            "support_bundle_style_summary.json": {"redaction": "same redaction helper as Support Bundle v2"},
        }
        file_sizes: dict[str, int] = {}
        for name, payload in files.items():
            text = json.dumps(payload, sort_keys=True) + "\n"
            (backup / name).write_text(text, encoding="utf-8")
            file_sizes[name] = len(text.encode("utf-8"))
        included = sorted([*files.keys(), "manifest.json"])
        manifest = {
            "backup_schema_version": "backup.v1",
            "created_at": "2026-07-02T00:00:00+00:00",
            "runtime_commit": "abc123",
            "runtime_instance": "stable",
            "included_files": included,
            "excluded_files": ["raw secret-store files", "raw logs and full support bundles", "arbitrary home directory files"],
            "file_sizes": file_sizes,
            "total_size_bytes": sum(file_sizes.values()),
            "restore_status": "dry_run_only",
            "live_restore": "restore_not_enabled",
        }
        manifest_text = json.dumps(manifest, sort_keys=True) + "\n"
        (backup / "manifest.json").write_text(manifest_text, encoding="utf-8")
        return backup

    def test_restore_backup_v1_applies_allowlisted_preferences_with_snapshot(self) -> None:
        root = Path(self.tmpdir.name) / "state"
        root.mkdir()
        db_path = self._restore_fixture_db(root, initial_value="old")
        backup = self._restore_fixture_backup(root, value="new")
        result = restore_backup_v1(
            _plan(action_type="operator.restore"),
            {
                "pending_id": "confirm-test",
                "state_root": str(root),
                "db_path": str(db_path),
                "backup_root": str(root / "backups"),
                "restore_backup_path": str(backup),
            },
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["mutated"])
        self.assertEqual("operator.restore.v1", result["executor_id"])
        self.assertEqual("completed_verified", result["details"]["status"])
        self.assertTrue(Path(result["details"]["snapshot_path"]).joinpath("manifest.json").exists())
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT value FROM preferences WHERE key = ?", ("system_resource_baseline_v1",)).fetchone()
            self.assertEqual("new", row[0])
        finally:
            conn.close()

    def test_restore_backup_v1_noops_when_state_matches(self) -> None:
        root = Path(self.tmpdir.name) / "state-noop"
        root.mkdir()
        db_path = self._restore_fixture_db(root, initial_value="same")
        backup = self._restore_fixture_backup(root, value="same")
        result = restore_backup_v1(
            _plan(action_type="operator.restore"),
            {
                "pending_id": "confirm-test",
                "state_root": str(root),
                "db_path": str(db_path),
                "backup_root": str(root / "backups"),
                "restore_backup_path": str(backup),
            },
        )
        self.assertTrue(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertEqual("no_op_already_matches", result["details"]["status"])

    def test_restore_backup_v1_blocks_changed_since_preview(self) -> None:
        root = Path(self.tmpdir.name) / "state-changed"
        root.mkdir()
        db_path = self._restore_fixture_db(root, initial_value="old")
        backup = self._restore_fixture_backup(root, value="new")
        result = restore_backup_v1(
            _plan(action_type="operator.restore"),
            {
                "pending_id": "confirm-test",
                "state_root": str(root),
                "db_path": str(db_path),
                "backup_root": str(root / "backups"),
                "restore_backup_path": str(backup),
                "restore_fingerprint": "different",
            },
        )
        self.assertFalse(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertEqual("backup_changed_since_preview", result["error_code"])

    def test_restore_backup_v1_rolls_back_on_post_apply_verification_failure(self) -> None:
        root = Path(self.tmpdir.name) / "state-rollback"
        root.mkdir()
        db_path = self._restore_fixture_db(root, initial_value="old")
        backup = self._restore_fixture_backup(root, value="new")
        import agent.executor_registry as registry_module

        real_current = registry_module._current_preferences
        calls = 0

        def _current_with_bad_verification(path, keys):
            nonlocal calls
            calls += 1
            if calls == 2:
                return {"system_resource_baseline_v1": "wrong-after-apply"}
            return real_current(path, keys)

        with patch("agent.executor_registry._current_preferences", side_effect=_current_with_bad_verification):
            result = restore_backup_v1(
                _plan(action_type="operator.restore"),
                {
                    "pending_id": "confirm-test",
                    "state_root": str(root),
                    "db_path": str(db_path),
                    "backup_root": str(root / "backups"),
                    "restore_backup_path": str(backup),
                },
            )
        self.assertFalse(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertEqual("restore_failed_rolled_back", result["error_code"])
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT value FROM preferences WHERE key = ?", ("system_resource_baseline_v1",)).fetchone()
            self.assertEqual("old", row[0])
        finally:
            conn.close()

    def test_malformed_executor_result_returns_structured_failure(self) -> None:
        registry = ExecutorRegistry(self.journal_path)

        def _run(plan, action):
            return "not a result"

        registry.register(ExecutorSpec(executor_id="test.malformed", action_type="operator.support_bundle", status="enabled", run=_run))
        result = registry.execute_confirmed_plan(plan=_plan(), action={"pending_id": "confirm-test"})
        self.assertFalse(result.ok)
        self.assertFalse(result.mutated)
        self.assertEqual("executor_exception_before_verified_mutation", result.error_code)
        self.assertEqual("AttributeError", result.details.get("exception"))

    def test_journal_append_compacts_oversized_nested_records_and_redacts(self) -> None:
        registry = ExecutorRegistry(self.journal_path)
        huge = "diagnostic-payload-" * 10_000
        secret = "Bearer " + ("secret-token-" * 100)
        journal_id = registry.journal.append(
            {
                "event": "executor_result",
                "plan": {
                    "plan_id": "confirm-huge",
                    "action_type": "operator.backup",
                    "target": "backup",
                    "confirmation_token": "confirm-secret",
                    "fields": {f"k{i}": huge for i in range(80)},
                },
                "action": {"pending_id": "confirm-huge", "api_key": "sk-secret", "nested": [{"token": secret, "notes": huge} for _ in range(80)]},
                "result": {
                    "journal_id": "executor-huge",
                    "plan_id": "confirm-huge",
                    "action_type": "operator.backup",
                    "target": "backup",
                    "executor_id": "operator.backup.v1",
                    "ok": True,
                    "mutated": True,
                    "resources_touched": [huge for _ in range(80)],
                },
            }
        )
        text = self.journal_path.read_text(encoding="utf-8")
        self.assertIn(journal_id, text)
        self.assertLess(len(text.encode("utf-8")), EXECUTOR_JOURNAL_MAX_RECORD_BYTES)
        self.assertIn('"compacted": true', text)
        self.assertNotIn("confirm-secret", text)
        self.assertNotIn("sk-secret", text)
        self.assertNotIn("secret-token-secret-token", text)

    def test_support_bundle_v2_manifest_and_expected_files(self) -> None:
        action = {
            "pending_id": "confirm-test",
            "diagnostics": {
                "version": {"git_commit": "abc123", "runtime_instance": "stable"},
                "ready": {"ready": True, "runtime_mode": "READY", "state_label": "Ready"},
                "state": {"ok": True, "ready": True, "search": {"available": True}},
                "search_status": {"enabled": True, "available": True, "base_url": "http://127.0.0.1:8888"},
                "telegram_status": {"configured": True, "token": "123456:telegram-secret-token", "state": "stopped"},
                "packs_state": {"ok": True, "counts": {"enabled": 1}},
                "checkout_commit": "abc123",
                "git": {"status_clean": True},
            },
            "executor_journal_recent": [{"confirmation_token": "confirm-secret", "api_key": "sk-secret"}],
        }

        result = create_redacted_support_bundle(_plan(), action)
        artifact = Path(result["details"]["artifact_path"])
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))

        self.assertTrue(result["mutated"])
        self.assertEqual(SUPPORT_BUNDLE_SCHEMA_VERSION, manifest["bundle_schema_version"])
        self.assertEqual("abc123", manifest["runtime_commit"])
        self.assertEqual("abc123", manifest["checkout_commit"])
        expected = {
            "manifest.json",
            "doctor_summary.json",
            "version.json",
            "ready.json",
            "state_summary.json",
            "search_status.json",
            "telegram_status.json",
            "packs_state_summary.json",
            "executor_registry_journal_summary.json",
            "readiness_proof_summary.json",
            "git_runtime_freshness.json",
            "support_summary.json",
        }
        self.assertTrue(expected.issubset(set(manifest["included_files"])))
        for name in expected:
            self.assertTrue((artifact / name).is_file(), name)

    def test_support_bundle_redacts_sensitive_values(self) -> None:
        payload = {
            "telegram_token": "123456:telegram-secret-token",
            "api_key": "sk-test-secret",
            "password": "correct horse battery staple",
            "authorization": "Bearer abc.def.ghi",
            "server.secret_key": "ultrasecretkey",
            "confirmation_token": "confirm-secret",
            "config_source_path": str(Path.home() / ".config/personal-agent/private/path.json"),
            "safe": "plain",
        }
        redacted = support_bundle_redact(payload)
        text = json.dumps(redacted, sort_keys=True)
        for secret in (
            "123456:telegram-secret-token",
            "sk-test-secret",
            "correct horse battery staple",
            "abc.def.ghi",
            "ultrasecretkey",
            "confirm-secret",
        ):
            self.assertNotIn(secret, text)
        self.assertIn("[REDACTED]", text)
        self.assertIn("plain", text)

    def test_support_bundle_output_does_not_write_raw_secrets(self) -> None:
        result = create_redacted_support_bundle(
            _plan(),
            {
                "pending_id": "confirm-test",
                "diagnostics": {
                    "telegram_status": {"token": "123456:telegram-secret-token"},
                    "ready": {"message": "Bearer abc.def.ghi"},
                    "search_status": {"api_key": "sk-test-secret"},
                },
                "executor_journal_recent": [{"password": "letmein"}],
            },
        )
        artifact = Path(result["details"]["artifact_path"])
        combined = "\n".join(path.read_text(encoding="utf-8") for path in artifact.glob("*.json"))
        self.assertNotIn("123456:telegram-secret-token", combined)
        self.assertNotIn("abc.def.ghi", combined)
        self.assertNotIn("sk-test-secret", combined)
        self.assertNotIn("letmein", combined)

    def test_support_bundle_summarizes_recursive_executor_journal(self) -> None:
        huge_payload = "x" * (BACKUP_MAX_FILE_BYTES + 1024)
        result = create_redacted_support_bundle(
            _plan(),
            {
                "pending_id": "confirm-test",
                "diagnostics": {"version": {"git_commit": "abc123", "runtime_instance": "stable"}},
                "executor_journal_recent": [
                    {
                        "journal_id": "executor-old",
                        "event": "executor_result",
                        "plan": {"plan_id": "old-plan", "action_type": "operator.backup", "target": "backup"},
                        "action": {"executor_journal_recent": [{"large": huge_payload}]},
                        "result": {
                            "journal_id": "executor-old",
                            "plan_id": "old-plan",
                            "action_type": "operator.backup",
                            "target": "backup",
                            "executor_id": "operator.backup.v1",
                            "ok": True,
                            "mutated": True,
                            "resources_touched": [huge_payload],
                        },
                    }
                ],
            },
        )
        artifact = Path(result["details"]["artifact_path"])
        journal_summary = (artifact / "executor_registry_journal_summary.json").read_text(encoding="utf-8")
        self.assertNotIn(huge_payload[:1000], journal_summary)
        self.assertIn('"resources_touched_count": 1', journal_summary)
        self.assertLess((artifact / "executor_registry_journal_summary.json").stat().st_size, BACKUP_MAX_FILE_BYTES)

    def test_support_bundle_failure_before_manifest_records_partial_artifacts(self) -> None:
        calls = 0

        def _fail_after_first(path, payload):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise OSError("disk full")
            path.write_text("{}", encoding="utf-8")
            return 2

        with patch("agent.executor_registry._write_support_json", side_effect=_fail_after_first):
            result = create_redacted_support_bundle(
                _plan(),
                {
                    "pending_id": "confirm-test",
                    "diagnostics": {"version": {"git_commit": "abc123", "runtime_instance": "stable"}},
                },
            )
        artifact = Path(result["details"]["artifact_path"])
        self.assertFalse(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertEqual("support_bundle_v2_failed_before_final_manifest", result["error_code"])
        self.assertTrue(result["resources_touched"])
        self.assertFalse((artifact / "manifest.json").exists())
        self.assertIn("partial support bundle directory", result["rollback_hint"])

    def test_backup_v1_manifest_and_expected_files(self) -> None:
        backup_root = Path(self.tmpdir.name) / "backups"
        result = create_additive_backup(
            _plan(action_type="operator.backup", target="backup assistant"),
            {
                "pending_id": "confirm-test",
                "backup_root": str(backup_root),
                "diagnostics": {
                    "version": {"git_commit": "abc123", "runtime_instance": "stable"},
                    "ready": {"ready": True, "runtime_mode": "READY", "state_label": "Ready"},
                    "state": {"ok": True, "ready": True},
                    "search_status": {"enabled": True, "available": True},
                    "telegram_status": {"configured": True, "token": "123456:telegram-secret-token"},
                    "packs_state": {"ok": True, "counts": {"enabled": 1}},
                },
                "backup_sources": {
                    "state_database": {"path": str(Path.home() / ".local/share/personal-agent/memory.db"), "size_bytes": 123},
                    "preferences": {"count": 2},
                    "memory": {"summary_count": 1},
                },
                "executor_journal_recent": [{"api_key": "sk-secret"}],
            },
        )
        artifact = Path(result["details"]["artifact_path"])
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        expected = {
            "manifest.json",
            "state_database_summary.json",
            "preferences_summary.json",
            "memory_anchors_summary.json",
            "pack_metadata_summary.json",
            "runtime_config_summary.json",
            "executor_registry_journal_summary.json",
            "diagnostics_summary.json",
            "support_bundle_style_summary.json",
            "backup_summary.json",
        }
        self.assertTrue(result["mutated"])
        self.assertEqual("operator.backup.v1", result["executor_id"])
        self.assertEqual(BACKUP_SCHEMA_VERSION, manifest["backup_schema_version"])
        self.assertEqual("abc123", manifest["runtime_commit"])
        self.assertEqual(RESTORE_V1_CAPABILITY, manifest["restore_status"])
        self.assertEqual(RESTORE_V1_CAPABILITY, manifest["live_restore"])
        self.assertLess(manifest["total_size_bytes"], BACKUP_MAX_TOTAL_BYTES)
        self.assertEqual(BACKUP_MAX_FILE_BYTES, manifest["size_caps"]["max_file_bytes"])
        self.assertTrue(expected.issubset(set(manifest["included_files"])))
        for name in expected:
            self.assertTrue((artifact / name).is_file(), name)

    def test_backup_v1_redacts_and_excludes_raw_secret_material(self) -> None:
        result = create_additive_backup(
            _plan(action_type="operator.backup"),
            {
                "pending_id": "confirm-test",
                "backup_root": str(Path(self.tmpdir.name) / "backups"),
                "diagnostics": {
                    "telegram_status": {"token": "123456:telegram-secret-token"},
                    "search_status": {"api_key": "sk-test-secret"},
                    "ready": {"message": "Bearer abc.def.ghi"},
                },
                "backup_sources": {"state_database": {"path": str(Path.home() / ".local/share/personal-agent/secrets.enc.json")}},
                "executor_journal_recent": [{"password": "letmein"}],
            },
        )
        artifact = Path(result["details"]["artifact_path"])
        combined = "\n".join(path.read_text(encoding="utf-8") for path in artifact.glob("*.json"))
        self.assertNotIn("123456:telegram-secret-token", combined)
        self.assertNotIn("abc.def.ghi", combined)
        self.assertNotIn("sk-test-secret", combined)
        self.assertNotIn("letmein", combined)
        self.assertIn("raw secret-store files", combined)
        self.assertIn("summary_only_raw_database_excluded", combined)
        self.assertIn(RESTORE_V1_CAPABILITY, combined)

    def test_backup_v1_summarizes_recursive_executor_journal(self) -> None:
        huge_payload = "x" * (BACKUP_MAX_FILE_BYTES + 1024)
        result = create_additive_backup(
            _plan(action_type="operator.backup"),
            {
                "pending_id": "confirm-test",
                "backup_root": str(Path(self.tmpdir.name) / "backups"),
                "diagnostics": {"version": {"git_commit": "abc123", "runtime_instance": "stable"}},
                "executor_journal_recent": [
                    {
                        "journal_id": "executor-old",
                        "event": "executor_result",
                        "plan": {"plan_id": "old-plan", "action_type": "operator.backup", "target": "backup"},
                        "action": {"executor_journal_recent": [{"large": huge_payload}]},
                        "result": {
                            "journal_id": "executor-old",
                            "plan_id": "old-plan",
                            "action_type": "operator.backup",
                            "target": "backup",
                            "executor_id": "operator.backup.v1",
                            "ok": True,
                            "mutated": True,
                            "resources_touched": [huge_payload],
                        },
                    }
                ],
            },
        )
        artifact = Path(result["details"]["artifact_path"])
        journal_summary = (artifact / "executor_registry_journal_summary.json").read_text(encoding="utf-8")
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn(huge_payload[:1000], journal_summary)
        self.assertIn('"resources_touched_count": 1', journal_summary)
        self.assertLess((artifact / "executor_registry_journal_summary.json").stat().st_size, BACKUP_MAX_FILE_BYTES)
        self.assertLess(sum(path.stat().st_size for path in artifact.glob("*.json")), BACKUP_MAX_TOTAL_BYTES)
        self.assertGreater(manifest["file_sizes"]["executor_registry_journal_summary.json"], 0)

    def test_backup_failure_before_manifest_records_partial_artifacts(self) -> None:
        backup_root = Path(self.tmpdir.name) / "backups"
        calls = 0

        def _fail_after_first(path, payload):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise OSError("disk full")
            path.write_text("{}", encoding="utf-8")
            return 2

        with patch("agent.executor_registry._write_backup_json", side_effect=_fail_after_first):
            result = create_additive_backup(
                _plan(action_type="operator.backup"),
                {
                    "pending_id": "confirm-test",
                    "backup_root": str(backup_root),
                    "diagnostics": {"version": {"git_commit": "abc123", "runtime_instance": "stable"}},
                },
            )
        artifact = Path(result["details"]["artifact_path"])
        self.assertFalse(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertEqual("backup_v1_failed_before_final_manifest", result["error_code"])
        self.assertTrue(result["resources_touched"])
        self.assertFalse((artifact / "manifest.json").exists())
        self.assertIn("partial backup directory", result["rollback_hint"])

    def test_backup_v1_refuses_unapproved_backup_root(self) -> None:
        with self.assertRaises(ValueError):
            create_additive_backup(
                _plan(action_type="operator.backup"),
                {"pending_id": "confirm-test", "backup_root": "/etc/personal-agent-backups"},
            )


if __name__ == "__main__":
    unittest.main()
