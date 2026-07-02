from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.executor_registry import (
    BACKUP_SCHEMA_VERSION,
    BACKUP_MAX_FILE_BYTES,
    BACKUP_MAX_TOTAL_BYTES,
    EXECUTOR_JOURNAL_MAX_RECORD_BYTES,
    SUPPORT_BUNDLE_SCHEMA_VERSION,
    ExecutorRegistry,
    ExecutorPartialFailure,
    ExecutorSpec,
    create_additive_backup,
    create_redacted_support_bundle,
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
        self.assertEqual("dry_run_only", manifest["restore_status"])
        self.assertEqual("restore_not_enabled", manifest["live_restore"])
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
        self.assertIn("restore_not_enabled", combined)

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
