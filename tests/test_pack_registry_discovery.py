from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from agent.packs.external_ingestion import ExternalPackIngestor
from agent.packs.registry_discovery import PackRegistryDiscoveryService, RegistrySourcePolicyError
from agent.packs.remote_fetch import RemotePackFetcher, RemotePackSource
from agent.packs.store import PackStore


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, data in files.items():
            handle.writestr(name, data)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, body: bytes, *, url: str) -> None:
        self._body = io.BytesIO(body)
        self._url = url
        self.headers = {"Content-Length": str(len(body))}

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _CountingOpener:
    def __init__(self, mapping: dict[str, _FakeResponse], *, fail: bool = False) -> None:
        self.mapping = mapping
        self.fail = fail
        self.calls = 0

    def open(self, request, timeout: int = 15):  # noqa: ANN001
        _ = timeout
        self.calls += 1
        if self.fail:
            raise RuntimeError("fetch failed")
        url = getattr(request, "full_url", str(request))
        response = self.mapping.get(url)
        if response is None:
            raise RuntimeError(f"unexpected url: {url}")
        return response


class TestPackRegistryDiscovery(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.storage_root = self.root / "external_packs"
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.store = PackStore(str(self.root / "packs.db"))
        self.sources_path = self.storage_root / "registry_sources.json"
        self.policy_path = self.storage_root / "registry_source_policy.json"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_sources(self, rows: list[dict[str, object]]) -> None:
        self.sources_path.write_text(json.dumps({"sources": rows}, ensure_ascii=True), encoding="utf-8")

    def _write_policy(self, payload: dict[str, object]) -> None:
        self.policy_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _record_remote_pack(self, source_url: str, files: dict[str, bytes], *, ref: str = "main") -> dict[str, object]:
        archive = _zip_bytes(files)
        fetcher = RemotePackFetcher(
            str(self.storage_root),
            opener=_CountingOpener({source_url: _FakeResponse(archive, url=source_url)}),
        )
        ingestor = ExternalPackIngestor(str(self.storage_root), remote_fetcher=fetcher)
        result, review = ingestor.ingest_from_remote_source(
            RemotePackSource(kind="github_archive", url=source_url, ref=ref)
        )
        return self.store.record_external_pack(
            canonical_pack=result.pack.to_dict(),
            classification=result.classification,
            status=result.status,
            risk_report=result.risk_report.to_dict(),
            review_envelope=review.to_dict(),
            quarantine_path=result.quarantine_path,
            normalized_path=result.normalized_path,
        )

    def test_list_and_search_results_are_normalized(self) -> None:
        source_url = "https://example.com/catalog.json"
        catalog = {
            "packs": [
                {
                    "id": "docs-skill",
                    "name": "Docs Skill",
                    "summary": "Summarizes docs safely.",
                    "author": "Example",
                    "source_url": "https://github.com/example/docs-skill",
                    "source_kind_hint": "github_repo",
                    "latest_ref_hint": "main",
                    "has_skill_md": True,
                    "tags": ["docs", "markdown"],
                },
                {
                    "id": "plugin-pack",
                    "name": "Plugin Pack",
                    "summary": "Runs a native plugin.",
                    "author": "Example",
                    "source_url": "https://github.com/example/plugin-pack",
                    "requires_execution": True,
                    "tags": ["plugin"],
                },
            ]
        }
        self._write_sources(
            [
                {
                    "id": "generic-registry",
                    "kind": "generic_registry_api",
                    "name": "Generic Registry",
                    "base_url": source_url,
                    "enabled": True,
                }
            ]
        )
        opener = _CountingOpener({source_url: _FakeResponse(json.dumps(catalog).encode("utf-8"), url=source_url)})
        service = PackRegistryDiscoveryService(
            pack_store=self.store,
            storage_root=str(self.storage_root),
            sources_path=str(self.sources_path),
            opener=opener,
        )

        sources = service.list_sources()
        self.assertEqual(1, len(sources))
        self.assertEqual("generic-registry", sources[0]["id"])

        listing_payload = service.list_packs("generic-registry")
        self.assertEqual(2, listing_payload["count"])
        portable = next(row for row in listing_payload["packs"] if row["remote_id"] == "docs-skill")
        native = next(row for row in listing_payload["packs"] if row["remote_id"] == "plugin-pack")
        self.assertEqual("portable_text_skill", portable["artifact_type_hint"])
        self.assertIn("portable_text_skill", portable["badges"])
        self.assertIn("installable_by_current_policy", portable["badges"])
        self.assertEqual("native_code_pack", native["artifact_type_hint"])
        self.assertIn("blocked_by_current_policy", native["badges"])

        search_payload = service.search("generic-registry", "plugin")
        self.assertEqual(1, search_payload["search"]["count"])
        self.assertEqual("plugin-pack", search_payload["search"]["results"][0]["remote_id"])
        self.assertTrue(sources[0]["allowed_by_policy"])
        self.assertEqual(300, sources[0]["cache_ttl_seconds"])

    def test_preview_uses_existing_local_pack_and_compare_hints(self) -> None:
        remote_url = "https://github.com/example/docs-skill"
        local_row = self._record_remote_pack(
            remote_url,
            {"repo-main/SKILL.md": b"# Docs Skill\n\nVersion one.\n"},
        )
        updated_row = self.store.set_external_pack_review_status(
            str(local_row["canonical_id"]),
            local_review_status="approved",
            approve_current_hash=True,
        )
        assert updated_row is not None
        catalog_path = self.storage_root / "registry_catalog.json"
        catalog_path.write_text(
            json.dumps(
                {
                    "packs": [
                        {
                            "id": "docs-skill",
                            "name": "Docs Skill",
                            "summary": "Summarizes docs safely.",
                            "source_url": remote_url,
                            "source_kind_hint": "github_repo",
                            "latest_ref_hint": "next",
                            "has_skill_md": True,
                        }
                    ]
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        self._write_sources(
            [
                {
                    "id": "local-registry",
                    "kind": "local_catalog",
                    "name": "Local Registry",
                    "base_url": str(catalog_path),
                    "enabled": True,
                }
            ]
        )
        service = PackRegistryDiscoveryService(
            pack_store=self.store,
            storage_root=str(self.storage_root),
            sources_path=str(self.sources_path),
        )

        before_count = len(self.store.list_external_packs())
        preview_payload = service.preview("local-registry", "docs-skill")
        after_count = len(self.store.list_external_packs())
        self.assertEqual(before_count, after_count)

        preview = preview_payload["preview"]
        self.assertEqual("portable_text_skill", preview["artifact_type_hint"])
        self.assertTrue(preview["related_local_pack"])
        self.assertEqual(str(local_row["canonical_id"]), preview["related_local_pack"]["canonical_id"])
        self.assertTrue(preview["compare_hint"]["available"])
        self.assertTrue(preview["compare_hint"]["likely_changed_upstream"])
        self.assertIn("previously_imported", preview["badges"])
        self.assertIn("compare_available", preview["badges"])
        self.assertIn("reviewed_locally", preview["badges"])
        self.assertIn("approval_tied_to_content", preview["badges"])
        self.assertIn("changed_upstream", preview["badges"])
        self.assertIn("identity is tied to content", preview["summary"])

    def test_policy_controls_allowlist_denied_disabled_and_kind_restriction(self) -> None:
        catalog_path = self.storage_root / "registry_catalog.json"
        catalog_path.write_text(
            json.dumps(
                {
                    "packs": [
                        {
                            "id": "docs-skill",
                            "name": "Docs Skill",
                            "summary": "Summarizes docs safely.",
                            "source_url": "https://github.com/example/docs-skill",
                            "has_skill_md": True,
                        }
                    ]
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        self._write_sources(
            [
                {
                    "id": "local-registry",
                    "kind": "local_catalog",
                    "name": "Local Registry",
                    "base_url": str(catalog_path),
                    "enabled": True,
                },
                {
                    "id": "disabled-registry",
                    "kind": "local_catalog",
                    "name": "Disabled Registry",
                    "base_url": str(catalog_path),
                    "enabled": False,
                },
                {
                    "id": "generic-registry",
                    "kind": "generic_registry_api",
                    "name": "Generic Registry",
                    "base_url": "https://example.com/catalog.json",
                    "enabled": True,
                },
            ]
        )
        self._write_policy(
            {
                "defaults": {
                    "allowlisted": False,
                    "allowed_source_kinds": ["local_catalog"],
                    "cache_ttl_seconds": 300,
                    "max_results": 25,
                },
                "overrides": [
                    {"source_id": "local-registry", "allowlisted": True},
                    {"source_id": "generic-registry", "allowlisted": True, "denied": True},
                ],
            }
        )
        service = PackRegistryDiscoveryService(
            pack_store=self.store,
            storage_root=str(self.storage_root),
            sources_path=str(self.sources_path),
            policy_path=str(self.policy_path),
        )

        sources = {row["id"]: row for row in service.list_sources()}
        self.assertTrue(sources["local-registry"]["allowed_by_policy"])
        self.assertFalse(sources["disabled-registry"]["allowed_by_policy"])
        self.assertEqual("source_disabled", sources["disabled-registry"]["blocked_reason"])
        self.assertFalse(sources["generic-registry"]["allowed_by_policy"])
        self.assertEqual("source_denied_by_policy", sources["generic-registry"]["blocked_reason"])

        allowed = service.list_packs("local-registry")
        self.assertEqual(1, allowed["count"])

        with self.assertRaises(RegistrySourcePolicyError) as disabled_exc:
            service.list_packs("disabled-registry")
        self.assertEqual("source_disabled", disabled_exc.exception.policy.blocked_reason)

        with self.assertRaises(RegistrySourcePolicyError) as denied_exc:
            service.search("generic-registry", "docs")
        self.assertEqual("source_denied_by_policy", denied_exc.exception.policy.blocked_reason)

        self._write_policy(
            {
                "defaults": {
                    "allowlisted": True,
                    "allowed_source_kinds": ["local_catalog"],
                }
            }
        )
        service_kind_blocked = PackRegistryDiscoveryService(
            pack_store=self.store,
            storage_root=str(self.storage_root),
            sources_path=str(self.sources_path),
            policy_path=str(self.policy_path),
        )
        with self.assertRaises(RegistrySourcePolicyError) as kind_exc:
            service_kind_blocked.preview("generic-registry", "docs-skill")
        self.assertEqual("source_kind_not_allowed", kind_exc.exception.policy.blocked_reason)

    def test_per_source_ttl_override_is_respected(self) -> None:
        source_url = "https://example.com/catalog.json"
        catalog = {
            "packs": [
                {
                    "id": "docs-skill",
                    "name": "Docs Skill",
                    "summary": "Summarizes docs safely.",
                    "source_url": "https://github.com/example/docs-skill",
                    "has_skill_md": True,
                }
            ]
        }
        self._write_sources(
            [
                {
                    "id": "generic-registry",
                    "kind": "generic_registry_api",
                    "name": "Generic Registry",
                    "base_url": source_url,
                    "enabled": True,
                }
            ]
        )
        self._write_policy(
            {
                "overrides": [
                    {
                        "source_id": "generic-registry",
                        "cache_ttl_seconds": 9,
                        "max_results": 7,
                    }
                ]
            }
        )
        opener = _CountingOpener({source_url: _FakeResponse(json.dumps(catalog).encode("utf-8"), url=source_url)})
        service = PackRegistryDiscoveryService(
            pack_store=self.store,
            storage_root=str(self.storage_root),
            sources_path=str(self.sources_path),
            policy_path=str(self.policy_path),
            opener=opener,
        )

        payload = service.list_packs("generic-registry")
        self.assertEqual(1, payload["count"])
        cached = self.store.get_registry_source_cache("generic-registry")
        assert cached is not None
        self.assertEqual(9, int(cached["expires_at"]) - int(cached["fetched_at"]))

    def test_cache_is_reused_and_stale_cache_degrades_gracefully(self) -> None:
        source_url = "https://example.com/catalog.json"
        catalog = {
            "packs": [
                {
                    "id": "docs-skill",
                    "name": "Docs Skill",
                    "summary": "Summarizes docs safely.",
                    "source_url": "https://github.com/example/docs-skill",
                    "has_skill_md": True,
                }
            ]
        }
        self._write_sources(
            [
                {
                    "id": "generic-registry",
                    "kind": "generic_registry_api",
                    "name": "Generic Registry",
                    "base_url": source_url,
                    "enabled": True,
                }
            ]
        )
        opener = _CountingOpener({source_url: _FakeResponse(json.dumps(catalog).encode("utf-8"), url=source_url)})
        service = PackRegistryDiscoveryService(
            pack_store=self.store,
            storage_root=str(self.storage_root),
            sources_path=str(self.sources_path),
            opener=opener,
        )

        first = service.list_packs("generic-registry")
        self.assertFalse(first["from_cache"])
        self.assertEqual(1, opener.calls)

        second = service.list_packs("generic-registry")
        self.assertTrue(second["from_cache"])
        self.assertEqual(1, opener.calls)

        self.store._conn.execute(
            "UPDATE external_pack_registry_cache SET expires_at = 0 WHERE source_id = ?",
            ("generic-registry",),
        )
        self.store._conn.commit()

        failing_service = PackRegistryDiscoveryService(
            pack_store=self.store,
            storage_root=str(self.storage_root),
            sources_path=str(self.sources_path),
            opener=_CountingOpener({}, fail=True),
        )
        stale = failing_service.list_packs("generic-registry")
        self.assertTrue(stale["from_cache"])
        self.assertTrue(stale["stale"])
        self.assertEqual(1, stale["count"])


if __name__ == "__main__":
    unittest.main()
