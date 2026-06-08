from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.actions.managed_action_recovery import ManagedActionJournal
from agent.actions.persistent_journal import PersistentManagedActionJournalStore, redact_journal_value


class TestPersistentManagedActionJournal(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = PersistentManagedActionJournalStore(Path(self.tmpdir.name) / "managed_actions.db")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_upsert_recent_and_incomplete_round_trip(self) -> None:
        journal = ManagedActionJournal(
            action_type="provider_api_key_config",
            target="openai",
            action_id="managed-action-test",
        )
        journal.plan_step("preflight_provider", resource="openai")
        journal.record_changed_resource("provider_secret", "openai", rollback_supported=True)

        running = self.store.upsert(journal, status="running", recovery_hint="Confirm before retrying provider setup.")

        self.assertEqual("managed-action-test", running["action_id"])
        self.assertEqual("running", running["status"])
        self.assertTrue(self.store.incomplete())
        self.assertEqual("provider_api_key_config", self.store.recent(limit=1)[0]["action_type"])

        journal.mark_verification(ok=True, provider="openai")
        verified = self.store.upsert(journal, status="verified")

        self.assertEqual("verified", verified["status"])
        self.assertFalse(verified["recovery_needed"])
        self.assertEqual([], self.store.incomplete())
        self.assertTrue(verified["verification_result"]["ok"])

    def test_recovery_needed_status_is_read_only_surface(self) -> None:
        journal = ManagedActionJournal(
            action_type="telegram_enablement_config",
            target="/home/c/.config/systemd/user/personal-agent-telegram.service.d/override.conf",
            action_id="managed-action-telegram",
        )
        journal.record_created_resource("systemd_dropin", "personal-agent-telegram", path="/home/c/private/path")

        saved = self.store.upsert(
            journal,
            status="recovery_needed",
            recovery_hint="Show status and ask for confirmation before cleanup.",
        )

        incomplete = self.store.incomplete()
        self.assertEqual(1, len(incomplete))
        self.assertTrue(saved["recovery_needed"])
        self.assertEqual("Show status and ask for confirmation before cleanup.", incomplete[0]["recovery_hint"])
        self.assertEqual("recovery_needed", incomplete[0]["status"])

    def test_redaction_masks_secrets_raw_content_and_long_strings(self) -> None:
        journal = ManagedActionJournal(
            action_type="semantic_memory.ingest",
            target="private memory: my token is sk-secret-value and path /home/c/private/history.json",
            action_id="managed-action-redact",
        )
        journal.record_step(
            "write_source",
            ok=True,
            api_key="sk-super-secret",
            raw_text="hostile imported instructions and private memory",
            content="do not archive this",
            summary="x" * 400,
        )

        row = self.store.upsert(journal, status="running")
        payload = json.dumps(row, sort_keys=True)

        self.assertNotIn("sk-super-secret", payload)
        self.assertNotIn("private memory", payload)
        self.assertNotIn("/home/c/private/history.json", payload)
        self.assertNotIn("hostile imported instructions", payload)
        self.assertNotIn("do not archive this", payload)
        self.assertIn("***redacted***", payload)
        self.assertIn("[truncated]", payload)

    def test_unknown_status_is_rejected(self) -> None:
        journal = ManagedActionJournal(action_type="test", target="target")

        with self.assertRaises(ValueError):
            self.store.upsert(journal, status="done")

    def test_redact_helper_recurses_without_archiving_sensitive_values(self) -> None:
        redacted = redact_journal_value(
            {
                "secret_name": "openai",
                "details": {"memory_value": "private profile text", "count": 2},
                "messages": [{"body": "raw notification text"}],
                "deleted_keys": ["show_summary"],
                "deleted_key_hashes": ["09fede995ea382ddefc07e4e7c5fa9d2dd7dbde0a4b24b0fa9656f2f44600d99"],
            }
        )

        self.assertEqual("***redacted***", redacted["secret_name"])
        self.assertEqual("***redacted***", redacted["details"]["memory_value"])
        self.assertEqual(2, redacted["details"]["count"])
        self.assertEqual("***redacted***", redacted["messages"][0]["body"])
        self.assertEqual("***redacted***", redacted["deleted_keys"])
        self.assertEqual(
            ["09fede995ea382ddefc07e4e7c5fa9d2dd7dbde0a4b24b0fa9656f2f44600d99"],
            redacted["deleted_key_hashes"],
        )


if __name__ == "__main__":
    unittest.main()
