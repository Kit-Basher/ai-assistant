from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.packs.registry_discovery import REGISTRY_KIND_GENERIC_API, REGISTRY_KIND_GITHUB_INDEX, PackRegistryDiscoveryService
from agent.packs.source_approval import SourceApprovalController
from agent.packs.source_leads import SourceLead
from agent.packs.store import PackStore


class TestPackSourceApproval(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.store = PackStore(str(self.root / "packs.db"))
        self.discovery = PackRegistryDiscoveryService(
            pack_store=self.store,
            storage_root=str(self.root / "external_packs"),
        )
        self.controller = SourceApprovalController(pack_registry_discovery=self.discovery)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_github_repo_lead_preview_is_allowed_but_untrusted(self) -> None:
        preview = self.controller.preview(
            SourceLead(title="Repo lead", url="https://github.com/acme/pack", suspected_source_kind="github_repo")
        )

        self.assertTrue(preview.ok)
        self.assertEqual("github_repo", preview.source_kind)
        self.assertEqual(REGISTRY_KIND_GITHUB_INDEX, preview.registry_kind)
        self.assertTrue(preview.untrusted)
        self.assertTrue(preview.explicit_user_trust_required)
        self.assertTrue(preview.content_remains_hostile)
        self.assertIn("does not make the content safe", preview.user_message.lower())
        self.assertIn("no pages were fetched", preview.user_message.lower())

    def test_github_archive_lead_preview_is_allowed_but_untrusted(self) -> None:
        preview = self.controller.preview(
            SourceLead(
                title="Archive lead",
                url="https://github.com/acme/pack/archive/refs/heads/main.zip",
                suspected_source_kind="github_archive",
            )
        )

        self.assertTrue(preview.ok)
        self.assertEqual("github_archive", preview.source_kind)
        self.assertEqual(REGISTRY_KIND_GENERIC_API, preview.registry_kind)
        self.assertTrue(preview.untrusted)
        self.assertTrue(preview.fetch_allowed_after_approval)

    def test_generic_web_result_cannot_be_approved_for_direct_fetch(self) -> None:
        preview = self.controller.preview(
            SourceLead(title="Article", url="https://example.com/post", suspected_source_kind="generic_web_result")
        )

        self.assertFalse(preview.ok)
        self.assertEqual("generic_web_result_not_directly_fetchable", preview.blocked_reason)
        self.assertIn("manual source configuration", preview.user_message.lower())

        result = self.controller.approve(preview)
        self.assertFalse(result.ok)
        self.assertFalse(result.approved)
        self.assertEqual([], self.store.list_external_packs())

    def test_token_query_params_are_redacted_in_approval_record(self) -> None:
        preview = self.controller.preview(
            SourceLead(
                title="Secret archive",
                url="https://example.com/pack.zip?token=secret&api_key=abc&ok=yes",
                suspected_source_kind="generic_archive_url",
            )
        )
        self.assertTrue(preview.ok)
        self.assertNotIn("secret", preview.url)
        self.assertNotIn("abc", preview.url)

        result = self.controller.approve(preview)

        self.assertTrue(result.ok)
        source = self.discovery.get_catalog_source(str(result.source_id))
        persisted = source.get("persisted_source") or {}
        self.assertIn("token=%5BREDACTED%5D", str(persisted.get("base_url") or ""))
        self.assertNotIn("secret", str(persisted))
        self.assertNotIn("abc", str(persisted))

    def test_source_approval_writes_trusted_source_policy_record_only(self) -> None:
        preview = self.controller.preview(
            SourceLead(title="Pack", url="https://github.com/acme/pack", suspected_source_kind="github_repo")
        )
        result = self.controller.approve(preview)

        self.assertTrue(result.ok)
        self.assertTrue(result.approved)
        self.assertFalse(result.did_fetch)
        self.assertFalse(result.did_import)
        self.assertFalse(result.did_install)
        policy = self.discovery.get_source_policy(str(result.source_id))
        persisted_override = policy.get("persisted_override") or {}
        effective_policy = policy.get("effective_policy") or {}
        self.assertTrue(effective_policy.get("allowed_by_policy"))
        self.assertTrue(persisted_override.get("allowlisted"))
        self.assertTrue(persisted_override.get("approved_by_user"))
        self.assertIn("content_remains_hostile=true", str(persisted_override.get("notes") or ""))
        self.assertEqual([], self.store.list_external_packs())
        journal = result.managed_action_journal
        self.assertEqual("pack_source_approval", journal.get("action_type"))
        self.assertTrue(journal.get("verification_result", {}).get("ok"))
        self.assertFalse(journal.get("rollback_result", {}).get("attempted"))

    def test_approval_result_says_content_remains_hostile(self) -> None:
        preview = self.controller.preview(
            SourceLead(title="Pack", url="https://example.com/pack.zip", suspected_source_kind="generic_archive_url")
        )
        result = self.controller.approve(preview)

        self.assertIn("source content remains hostile", result.user_message.lower())
        self.assertIn("no pack was fetched", result.user_message.lower())
        self.assertIn("next safe step", result.user_message.lower())

    def test_repeated_approval_does_not_fetch_or_import(self) -> None:
        preview = self.controller.preview(
            SourceLead(title="Pack", url="https://example.com/pack.zip", suspected_source_kind="generic_archive_url")
        )

        first = self.controller.approve(preview)
        second = self.controller.approve(preview)

        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertFalse(first.did_fetch or first.did_import or first.did_install)
        self.assertFalse(second.did_fetch or second.did_import or second.did_install)
        self.assertEqual([], self.store.list_external_packs())


if __name__ == "__main__":
    unittest.main()
