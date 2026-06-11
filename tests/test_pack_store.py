from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.actions.persistent_journal import PersistentManagedActionJournalStore
from agent.packs.external_ingestion import ExternalPackIngestor
from agent.packs.store import PackStore


class TestPackStoreExternalPackRedaction(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.storage_root = self.root / "external_packs"
        self.journal_store = PersistentManagedActionJournalStore(self.root / "managed_actions.db")
        self.store = PackStore(str(self.root / "packs.db"), journal_store=self.journal_store)
        self.ingestor = ExternalPackIngestor(str(self.storage_root))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_removed_pack_tombstone_does_not_store_full_skill_text(self) -> None:
        source = self.root / "source"
        source.mkdir()
        hostile_marker = "HOSTILE_MARKER_DO_NOT_STORE"
        (source / "SKILL.md").write_text(
            f"# Removed Pack\n\nUse this guidance. {hostile_marker}\n",
            encoding="utf-8",
        )
        result, review = self.ingestor.ingest_from_path(str(source))
        row = self.store.record_external_pack(
            canonical_pack=result.pack.to_dict(),
            classification=result.classification,
            status=result.status,
            risk_report=result.risk_report.to_dict(),
            review_envelope=review.to_dict(),
            quarantine_path=result.quarantine_path,
            normalized_path=result.normalized_path,
        )
        pack_id = str(row["pack_id"])

        removed = self.store.remove_external_pack(pack_id, removed_by="test", reason="unit_test")

        assert removed is not None
        tombstone = self.store.get_external_pack_removal(pack_id)
        assert tombstone is not None
        tombstone_text = json.dumps(tombstone, ensure_ascii=True, sort_keys=True)
        self.assertNotIn(hostile_marker, tombstone_text)
        self.assertIsNone(tombstone.get("skill_text"))
        review_envelope = tombstone.get("review_envelope") if isinstance(tombstone.get("review_envelope"), dict) else {}
        audit = review_envelope.get("removed_skill_text") if isinstance(review_envelope.get("removed_skill_text"), dict) else {}
        self.assertTrue(audit.get("sha256"))
        self.assertFalse(bool(audit.get("stored", True)))
        persisted_journals = json.dumps(self.journal_store.recent(limit=10), ensure_ascii=True, sort_keys=True)
        self.assertNotIn(hostile_marker, persisted_journals)

    def _record_basic_external_pack(self) -> dict[str, object]:
        index = len(list(self.root.glob("source-basic-*")))
        source = self.root / f"source-basic-{index}"
        source.mkdir()
        (source / "SKILL.md").write_text(f"# Basic Pack {index}\n\nUse as untrusted guidance.\n", encoding="utf-8")
        result, review = self.ingestor.ingest_from_path(str(source))
        return self.store.record_external_pack(
            canonical_pack=result.pack.to_dict(),
            classification=result.classification,
            status=result.status,
            risk_report=result.risk_report.to_dict(),
            review_envelope=review.to_dict(),
            quarantine_path=result.quarantine_path,
            normalized_path=result.normalized_path,
        )

    def test_external_pack_import_record_is_journaled_and_verified(self) -> None:
        row = self._record_basic_external_pack()

        journal = row.get("managed_action_journal") if isinstance(row.get("managed_action_journal"), dict) else {}
        self.assertEqual("external_pack_import_record", journal.get("action_type"))
        self.assertTrue(journal.get("verification_result", {}).get("ok"))
        self.assertFalse(journal.get("rollback_result", {}).get("attempted"))
        persisted = self.journal_store.get(str(journal.get("action_id") or ""))
        assert persisted is not None
        self.assertEqual("verified", persisted.get("status"))
        self.assertEqual("external_pack_import_record", persisted.get("action_type"))

    def test_pack_removal_journals_and_verifies_tombstone_state(self) -> None:
        row = self._record_basic_external_pack()
        pack_id = str(row["pack_id"])

        removed = self.store.remove_external_pack(pack_id, removed_by="test", reason="unit_test")

        assert removed is not None
        journal = removed.get("managed_action_journal") if isinstance(removed.get("managed_action_journal"), dict) else {}
        self.assertTrue(removed.get("metadata_update_ok"))
        self.assertEqual("external_pack_removal", journal.get("action_type"))
        self.assertTrue(journal.get("verification_result", {}).get("ok"))
        self.assertTrue(journal.get("verification_result", {}).get("tombstone_present"))
        self.assertFalse(journal.get("verification_result", {}).get("usable"))
        self.assertIsNone(self.store.get_external_pack(pack_id))
        self.assertIsNotNone(self.store.get_external_pack_removal(pack_id))
        persisted = self.journal_store.get(str(journal.get("action_id") or ""))
        assert persisted is not None
        self.assertEqual("verified", persisted.get("status"))

    def test_failed_pack_removal_verification_restores_previous_metadata(self) -> None:
        row = self._record_basic_external_pack()
        pack_id = str(row["pack_id"])
        current = self.store.get_external_pack(pack_id)
        assert current is not None

        with patch.object(self.store, "get_external_pack", side_effect=[current, current, current]):
            failed = self.store.remove_external_pack(pack_id, removed_by="test", reason="unit_test")

        assert failed is not None
        journal = failed.get("managed_action_journal") if isinstance(failed.get("managed_action_journal"), dict) else {}
        self.assertFalse(failed.get("metadata_update_ok"))
        self.assertEqual("external_pack_removal_verification_failed", failed.get("error_kind"))
        self.assertTrue(journal.get("rollback_result", {}).get("attempted"))
        self.assertTrue(journal.get("rollback_result", {}).get("ok"))
        persisted = self.journal_store.get(str(journal.get("action_id") or ""))
        assert persisted is not None
        self.assertEqual("rolled_back", persisted.get("status"))
        restored = self.store.get_external_pack(pack_id)
        assert restored is not None
        self.assertEqual(pack_id, restored.get("pack_id"))

    def test_pack_removal_does_not_touch_unrelated_pack_records(self) -> None:
        first = self._record_basic_external_pack()
        second = self._record_basic_external_pack()
        first_id = str(first["pack_id"])
        second_id = str(second["pack_id"])

        removed = self.store.remove_external_pack(first_id, removed_by="test", reason="unit_test")

        assert removed is not None
        self.assertIsNone(self.store.get_external_pack(first_id))
        unrelated = self.store.get_external_pack(second_id)
        assert unrelated is not None
        self.assertEqual(second_id, unrelated.get("pack_id"))

    def test_removed_pack_cannot_be_enabled(self) -> None:
        row = self._record_basic_external_pack()
        pack_id = str(row["pack_id"])
        self.store.set_external_pack_review_status(pack_id, local_review_status="approved", approve_current_hash=True)

        removed = self.store.remove_external_pack(pack_id, removed_by="test", reason="unit_test")
        enabled = self.store.set_external_pack_enabled(pack_id, enabled=True)

        assert removed is not None
        self.assertIsNone(enabled)
        self.assertIsNone(self.store.get_external_pack(pack_id))

    def test_pack_removal_does_not_execute_external_pack_code(self) -> None:
        row = self._record_basic_external_pack()
        pack_id = str(row["pack_id"])

        with patch("subprocess.run") as run:
            removed = self.store.remove_external_pack(pack_id, removed_by="test", reason="unit_test")

        assert removed is not None
        run.assert_not_called()

    def test_review_approval_journals_and_verifies(self) -> None:
        row = self._record_basic_external_pack()
        pack_id = str(row["pack_id"])

        updated = self.store.set_external_pack_review_status(
            pack_id,
            local_review_status="approved",
            approve_current_hash=True,
        )

        assert updated is not None
        journal = updated.get("managed_action_journal") if isinstance(updated.get("managed_action_journal"), dict) else {}
        self.assertTrue(updated.get("metadata_update_ok"))
        self.assertEqual("external_pack_review_status", journal.get("action_type"))
        self.assertTrue(journal.get("verification_result", {}).get("ok"))
        canonical = updated.get("canonical_pack") if isinstance(updated.get("canonical_pack"), dict) else {}
        trust = canonical.get("trust_anchor") if isinstance(canonical.get("trust_anchor"), dict) else {}
        self.assertEqual("approved", trust.get("local_review_status"))
        persisted = self.journal_store.get(str(journal.get("action_id") or ""))
        assert persisted is not None
        self.assertEqual("verified", persisted.get("status"))

    def test_review_approval_verification_failure_restores_previous_state(self) -> None:
        row = self._record_basic_external_pack()
        pack_id = str(row["pack_id"])
        current = self.store.get_external_pack(pack_id)
        assert current is not None
        bad_after = copy.deepcopy(current)

        with patch.object(self.store, "get_external_pack", side_effect=[current, bad_after, current]):
            failed = self.store.set_external_pack_review_status(
                pack_id,
                local_review_status="approved",
                approve_current_hash=True,
            )

        assert failed is not None
        journal = failed.get("managed_action_journal") if isinstance(failed.get("managed_action_journal"), dict) else {}
        self.assertFalse(failed.get("metadata_update_ok"))
        self.assertEqual("pack_review_state_verification_failed", failed.get("error_kind"))
        self.assertTrue(journal.get("rollback_result", {}).get("attempted"))
        persisted = self.journal_store.get(str(journal.get("action_id") or ""))
        assert persisted is not None
        self.assertEqual("rolled_back", persisted.get("status"))
        restored = self.store.get_external_pack(pack_id)
        assert restored is not None
        canonical = restored.get("canonical_pack") if isinstance(restored.get("canonical_pack"), dict) else {}
        trust = canonical.get("trust_anchor") if isinstance(canonical.get("trust_anchor"), dict) else {}
        self.assertNotEqual("approved", trust.get("local_review_status"))

    def test_enablement_journals_and_verifies(self) -> None:
        row = self._record_basic_external_pack()
        pack_id = str(row["pack_id"])
        approved = self.store.set_external_pack_review_status(
            pack_id,
            local_review_status="approved",
            approve_current_hash=True,
        )
        assert approved is not None

        updated = self.store.set_external_pack_enabled(pack_id, enabled=True)

        assert updated is not None
        journal = updated.get("managed_action_journal") if isinstance(updated.get("managed_action_journal"), dict) else {}
        self.assertTrue(updated.get("metadata_update_ok"))
        self.assertEqual("external_pack_enablement", journal.get("action_type"))
        self.assertTrue(journal.get("verification_result", {}).get("ok"))
        canonical = updated.get("canonical_pack") if isinstance(updated.get("canonical_pack"), dict) else {}
        runtime = canonical.get("runtime") if isinstance(canonical.get("runtime"), dict) else {}
        self.assertTrue(runtime.get("enabled"))
        persisted = self.journal_store.get(str(journal.get("action_id") or ""))
        assert persisted is not None
        self.assertEqual("verified", persisted.get("status"))

    def test_enablement_verification_failure_restores_previous_state(self) -> None:
        row = self._record_basic_external_pack()
        pack_id = str(row["pack_id"])
        current = self.store.get_external_pack(pack_id)
        assert current is not None
        bad_after = copy.deepcopy(current)

        with patch.object(self.store, "get_external_pack", side_effect=[current, bad_after, current]):
            failed = self.store.set_external_pack_enabled(pack_id, enabled=True)

        assert failed is not None
        journal = failed.get("managed_action_journal") if isinstance(failed.get("managed_action_journal"), dict) else {}
        self.assertFalse(failed.get("metadata_update_ok"))
        self.assertEqual("pack_enablement_verification_failed", failed.get("error_kind"))
        self.assertTrue(journal.get("rollback_result", {}).get("attempted"))
        persisted = self.journal_store.get(str(journal.get("action_id") or ""))
        assert persisted is not None
        self.assertEqual("rolled_back", persisted.get("status"))
        restored = self.store.get_external_pack(pack_id)
        assert restored is not None
        canonical = restored.get("canonical_pack") if isinstance(restored.get("canonical_pack"), dict) else {}
        runtime = canonical.get("runtime") if isinstance(canonical.get("runtime"), dict) else {}
        self.assertFalse(runtime.get("enabled", False))


if __name__ == "__main__":
    unittest.main()
