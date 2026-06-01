from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.packs.external_ingestion import ExternalPackIngestor
from agent.packs.store import PackStore


class TestPackStoreExternalPackRedaction(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.storage_root = self.root / "external_packs"
        self.store = PackStore(str(self.root / "packs.db"))
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

    def _record_basic_external_pack(self) -> dict[str, object]:
        source = self.root / "source-basic"
        source.mkdir()
        (source / "SKILL.md").write_text("# Basic Pack\n\nUse as untrusted guidance.\n", encoding="utf-8")
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
        restored = self.store.get_external_pack(pack_id)
        assert restored is not None
        canonical = restored.get("canonical_pack") if isinstance(restored.get("canonical_pack"), dict) else {}
        runtime = canonical.get("runtime") if isinstance(canonical.get("runtime"), dict) else {}
        self.assertFalse(runtime.get("enabled", False))


if __name__ == "__main__":
    unittest.main()
