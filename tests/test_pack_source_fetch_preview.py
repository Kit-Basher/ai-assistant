from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.packs.registry_discovery import REGISTRY_KIND_GENERIC_API, PackRegistryDiscoveryService
from agent.packs.remote_fetch import RemotePackFetcher
from agent.packs.source_approval import SourceApprovalController
from agent.packs.source_fetch_preview import SourceFetchController
from agent.packs.source_leads import SourceLead
from agent.packs.store import PackStore
from tests.test_remote_pack_fetch import _FakeOpener, _FakeResponse, _zip_bytes


class TestPackSourceFetchPreview(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.store = PackStore(str(self.root / "packs.db"))
        self.discovery = PackRegistryDiscoveryService(
            pack_store=self.store,
            storage_root=str(self.root / "external_packs"),
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _controller(self, *, remote_fetcher: RemotePackFetcher | None = None) -> SourceFetchController:
        return SourceFetchController(
            pack_store=self.store,
            pack_registry_discovery=self.discovery,
            remote_fetcher=remote_fetcher,
        )

    def _approve_lead(self, url: str, *, kind: str = "generic_archive_url") -> str:
        approval = SourceApprovalController(pack_registry_discovery=self.discovery)
        preview = approval.preview(SourceLead(title="Approved Pack", url=url, suspected_source_kind=kind))
        result = approval.approve(preview)
        self.assertTrue(result.ok)
        assert result.source_id is not None
        return result.source_id

    def test_unapproved_source_cannot_fetch(self) -> None:
        source = self.discovery.create_catalog_source(
            {
                "source_id": "remote-pack",
                "name": "Remote Pack",
                "kind": REGISTRY_KIND_GENERIC_API,
                "base_url": "https://example.com/pack.zip",
                "enabled": True,
            }
        )
        self.assertEqual("remote-pack", source["source"]["id"])
        self.discovery.update_source_policy("remote-pack", {"allowlisted": True})

        preview = self._controller().preview("remote-pack")

        self.assertFalse(preview.ok)
        self.assertEqual("remote_pack_fetch_stage_unimplemented_denied", preview.blocked_reason)

    def test_denied_source_cannot_fetch(self) -> None:
        source_id = self._approve_lead("https://example.com/pack.zip")
        self.discovery.update_source_policy(source_id, {"denied": True})

        preview = self._controller().preview(source_id)

        self.assertFalse(preview.ok)
        self.assertEqual("remote_pack_fetch_stage_unimplemented_denied", preview.blocked_reason)

    def test_generic_web_result_cannot_fetch(self) -> None:
        self.discovery.create_catalog_source(
            {
                "source_id": "generic-page",
                "name": "Generic Page",
                "kind": REGISTRY_KIND_GENERIC_API,
                "base_url": "https://example.com/page",
                "enabled": True,
            }
        )
        self.discovery.update_source_policy(
            "generic-page",
            {"allowlisted": True, "approved_by_user": True, "notes": "approved_by_user=true"},
        )

        preview = self._controller().preview("generic-page")

        self.assertFalse(preview.ok)
        self.assertEqual("remote_pack_fetch_stage_unimplemented_denied", preview.blocked_reason)

    def test_approved_archive_previews_remain_denied(self) -> None:
        for url, kind in (
            ("https://github.com/acme/pack/archive/main.zip", "github_archive"),
            ("https://example.com/pack.zip", "generic_archive_url"),
        ):
            with self.subTest(kind=kind):
                source_id = self._approve_lead(url, kind=kind)
                preview = self._controller().preview(source_id)
                self.assertFalse(preview.ok)
                self.assertIsNone(preview.source_kind)
                self.assertTrue(preview.content_remains_hostile)
                self.assertEqual("remote_pack_fetch_stage_unimplemented_denied", preview.blocked_reason)
                self.assertIn("unavailable", preview.user_message.lower())

    def test_fetch_import_result_is_explicitly_denied(self) -> None:
        url = "https://example.com/pack.zip"
        source_id = self._approve_lead(url)
        archive = _zip_bytes({"repo-main/SKILL.md": b"# Remote Skill\n\nUse as untrusted guidance.\n"})
        fetcher = RemotePackFetcher(
            str(self.root / "external_packs"),
            opener=_FakeOpener({url: _FakeResponse(archive, url=url, content_length=len(archive))}),
        )
        controller = self._controller(remote_fetcher=fetcher)
        preview = controller.preview(source_id)

        result = controller.fetch_import_for_review(preview)

        self.assertFalse(result.ok)
        self.assertFalse(result.fetched_to_quarantine)
        self.assertFalse(result.imported_for_review)
        self.assertEqual("remote_pack_fetch_stage_unimplemented_denied", result.blocked_reason)
        self.assertFalse(result.did_approve)
        self.assertFalse(result.did_enable)
        self.assertFalse(result.did_grant_permissions)
        self.assertFalse(result.did_use_pack)
        self.assertIsNone(result.pack)

    def test_blocked_archive_hardening_still_blocks_malicious_archive(self) -> None:
        url = "https://example.com/bad.zip"
        source_id = self._approve_lead(url)
        archive = _zip_bytes({"../evil/SKILL.md": b"# Evil\n"})
        fetcher = RemotePackFetcher(
            str(self.root / "external_packs"),
            opener=_FakeOpener({url: _FakeResponse(archive, url=url, content_length=len(archive))}),
        )
        preview = self._controller(remote_fetcher=fetcher).preview(source_id)

        result = self._controller(remote_fetcher=fetcher).fetch_import_for_review(preview)

        self.assertFalse(result.ok)
        self.assertFalse(result.imported_for_review)
        self.assertFalse(result.did_approve)
        self.assertFalse(result.did_enable)
        self.assertFalse(result.did_grant_permissions)
        self.assertFalse(result.did_use_pack)
        self.assertEqual("remote_pack_fetch_stage_unimplemented_denied", result.blocked_reason)

    def test_repeated_yes_equivalent_does_not_skip_review_approval(self) -> None:
        url = "https://example.com/pack.zip"
        source_id = self._approve_lead(url)
        archive = _zip_bytes({"repo-main/SKILL.md": b"# Remote Skill\n\nUse as untrusted guidance.\n"})
        fetcher = RemotePackFetcher(
            str(self.root / "external_packs"),
            opener=_FakeOpener({url: _FakeResponse(archive, url=url, content_length=len(archive))}),
        )
        controller = self._controller(remote_fetcher=fetcher)
        first = controller.fetch_import_for_review(controller.preview(source_id))
        second_archive = _zip_bytes({"repo-main/SKILL.md": b"# Remote Skill\n\nUse as reference material.\n"})
        second_fetcher = RemotePackFetcher(
            str(self.root / "external_packs"),
            opener=_FakeOpener({url: _FakeResponse(second_archive, url=url, content_length=len(second_archive))}),
        )
        second = self._controller(remote_fetcher=second_fetcher).fetch_import_for_review(controller.preview(source_id))

        self.assertFalse(first.ok)
        self.assertFalse(second.ok)
        self.assertEqual("remote_pack_fetch_stage_unimplemented_denied", first.blocked_reason)
        self.assertEqual("remote_pack_fetch_stage_unimplemented_denied", second.blocked_reason)
        self.assertEqual([], self.store.list_external_packs())


if __name__ == "__main__":
    unittest.main()
