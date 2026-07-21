from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent.api_server import AgentRuntime
from agent.packs.acquisition import AcquisitionRequest, PackAcquisitionCoordinator
from agent.packs.external_ingestion import ExternalPackIngestor
from agent.packs.lifecycle import PackLifecycleService
from agent.packs.managed_adapter_invocation import OP_DESCRIBE_CAPABILITY, OP_DRY_RUN
from agent.packs.managed_adapters import ADAPTER_LOCAL_FILE_IMPORT
from agent.packs.registry_discovery import PackRegistryDiscoveryService
from agent.packs.store import PackStore
from agent.search.safe_web_search import SafeWebSearchClient, SafeWebSearchConfig
from tests.test_safe_web_search import _FakeOpener
from tests.test_assistant_behavior_release_gate import _MemoryHandlerForTest, _assistant_text, _config
from tests.test_remote_pack_fetch import _FakeOpener as _RemoteFakeOpener, _FakeResponse, _zip_bytes


def _adapter_spec() -> dict[str, Any]:
    return {
        "kind": ADAPTER_LOCAL_FILE_IMPORT,
        "purpose": "Import one user-selected local file.",
        "allowed_extensions": [".json"],
        "max_file_size_mb": 1,
        "path_policy": "user_selected_file_only",
        "stores_local_index": False,
        "network_allowed": False,
    }


class TestPackAcquisitionCoordinator(unittest.TestCase):
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

    def _coordinator(
        self,
        discovery: Any | None = None,
        *,
        web_search_handler: Any | None = None,
        search_status_handler: Any | None = None,
    ) -> PackAcquisitionCoordinator:
        return PackAcquisitionCoordinator(
            pack_store=self.store,
            pack_registry_discovery=discovery or self.discovery,
            web_search_handler=web_search_handler,
            search_status_handler=search_status_handler,
        )

    def _record_pack(self, *, with_adapter: bool = True) -> dict[str, Any]:
        source = self.root / "pack_source"
        source.mkdir(exist_ok=True)
        (source / "SKILL.md").write_text("# Local Import Pack\n\nUse as untrusted guidance only.\n", encoding="utf-8")
        ingestor = ExternalPackIngestor(str(self.root / "external_packs"))
        result, review = ingestor.ingest_from_path(str(source), source_origin="test", created_by="test")
        canonical = result.pack.to_dict()
        if with_adapter:
            canonical["managed_adapters"] = [_adapter_spec()]
            canonical["runtime"] = {"managed_adapters": [_adapter_spec()]}
            canonical["permissions"] = {"managed_adapters": [_adapter_spec()], "granted": []}
        return self.store.record_external_pack(
            canonical_pack=canonical,
            classification=result.classification,
            status=result.status,
            risk_report=result.risk_report.to_dict(),
            review_envelope=review.to_dict(),
            quarantine_path=result.quarantine_path,
            normalized_path=result.normalized_path,
        )

    def test_missing_capability_searches_approved_starter_catalog_and_asks_preview_only(self) -> None:
        result = self._coordinator().acquire(AcquisitionRequest(text="install a skill that lets you browse"))

        assert result is not None
        self.assertEqual("browser_automation_planning", result.detected_capability)
        self.assertEqual("trusted_catalog_candidate", result.source_status)
        self.assertEqual("discovered", result.lifecycle_state)
        self.assertEqual("preview", result.next_step.action if result.next_step else None)
        self.assertTrue(result.requires_confirmation)
        self.assertIn("not installed or usable", result.user_message.lower())
        self.assertNotIn("installed and ready", result.user_message.lower())

    def test_untrusted_remote_source_is_blocked_at_trust_gate(self) -> None:
        storage = self.root / "remote_external_packs"
        storage.mkdir(parents=True)
        sources = storage / "registry_sources.json"
        sources.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "id": "remote-registry",
                            "kind": "generic_registry_api",
                            "name": "Remote Registry",
                            "base_url": "https://example.com/catalog.json",
                            "enabled": True,
                        }
                    ]
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        discovery = PackRegistryDiscoveryService(
            pack_store=self.store,
            storage_root=str(storage),
            sources_path=str(sources),
            policy_path=str(storage / "missing_policy.json"),
        )

        result = self._coordinator(discovery).acquire(AcquisitionRequest(text="install a skill that lets you browse"))

        assert result is not None
        self.assertEqual("source_trust_required", result.source_status)
        self.assertEqual("source_trust_required", result.blocked_reason)
        self.assertIn("source approval/trust is required", result.user_message.lower())
        self.assertNotIn("github is safe", result.user_message.lower())

    def test_no_candidate_offers_scaffold_preview(self) -> None:
        result = self._coordinator().acquire(AcquisitionRequest(text="read my email"))

        assert result is not None
        self.assertEqual("email_access", result.detected_capability)
        self.assertEqual("no_candidate_scaffold_available", result.source_status)
        self.assertEqual("missing", result.lifecycle_state)
        self.assertEqual("scaffold_preview", result.next_step.action if result.next_step else None)
        self.assertIsInstance(result.scaffold_preview, dict)
        self.assertFalse(result.scaffold_preview["creates_files"])
        self.assertFalse(result.scaffold_preview["executes_code"])

    def test_no_trusted_candidate_with_search_disabled_uses_scaffold_fallback(self) -> None:
        result = self._coordinator(
            search_status_handler=lambda: {"available": False, "enabled": False, "reason": "search_disabled"}
        ).acquire(AcquisitionRequest(text="add a capability for analyzing plant watering", requested_capability="plant_watering_analysis"))

        assert result is not None
        self.assertEqual("no_candidate_scaffold_available", result.source_status)
        self.assertEqual("scaffold_preview", result.next_step.action if result.next_step else None)
        self.assertFalse(result.source_leads)
        self.assertEqual("search_disabled", (result.search_status or {}).get("reason"))

    def test_no_trusted_candidate_with_search_available_shows_untrusted_source_leads(self) -> None:
        calls: list[dict[str, Any]] = []

        def search(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
            calls.append(dict(payload))
            return True, {
                "ok": True,
                "results": [
                    {
                        "title": "Possible pack",
                        "url": "https://github.com/example/plant-pack",
                        "snippet": "Search metadata only.",
                        "engine": "mock",
                    }
                ],
            }

        result = self._coordinator(
            web_search_handler=search,
            search_status_handler=lambda: {"available": True, "enabled": True, "provider": "searxng"},
        ).acquire(AcquisitionRequest(text="add a capability for analyzing plant watering", requested_capability="plant_watering_analysis"))

        assert result is not None
        self.assertEqual("untrusted_source_leads", result.source_status)
        self.assertEqual("source_approval_preview", result.next_step.action if result.next_step else None)
        self.assertEqual("source_approval_required", result.blocked_reason)
        self.assertEqual(1, len(result.source_leads))
        self.assertEqual("github_repo", result.source_leads[0]["suspected_source_kind"])
        self.assertIn("not trusted", result.user_message.lower())
        self.assertIn("cannot be fetched/imported", result.user_message.lower())
        self.assertEqual(1, len(calls))
        self.assertEqual([], self.store.list_external_packs())

    def test_trusted_catalog_candidates_take_priority_over_search_leads(self) -> None:
        calls: list[dict[str, Any]] = []
        result = self._coordinator(
            web_search_handler=lambda payload: (calls.append(dict(payload)) or (True, {"ok": True, "results": []})),
            search_status_handler=lambda: {"available": True, "enabled": True, "provider": "searxng"},
        ).acquire(AcquisitionRequest(text="install a skill that lets you browse"))

        assert result is not None
        self.assertEqual("trusted_catalog_candidate", result.source_status)
        self.assertEqual([], calls)

    def test_yes_continues_one_acquisition_gate_only(self) -> None:
        first = self._coordinator().acquire(AcquisitionRequest(text="install a skill that lets you browse"))
        assert first is not None
        preview_payload = {"preview": {"listing": first.candidate_pack or {}, "summary": "Preview only."}}
        coordinator = PackAcquisitionCoordinator(
            pack_store=self.store,
            pack_registry_discovery=self.discovery,
            action_handlers={"preview": lambda _ctx: {"ok": True, "text": "Preview shown.", **preview_payload}},
        )

        continued = coordinator.continue_step(first.lifecycle, context=first.next_step.pending_context if first.next_step else {})

        self.assertEqual("previewed", continued.lifecycle_state)
        self.assertEqual("import_for_review", continued.next_step.action if continued.next_step else None)
        self.assertFalse(bool(continued.lifecycle.get("usable")))

    def test_repeated_yes_does_not_skip_approval_enable_permission_gates(self) -> None:
        row = self._record_pack()
        lifecycle = PackLifecycleService().evaluate(imported_pack=row, permission_grants=[]).to_dict()
        coordinator = PackAcquisitionCoordinator(
            pack_store=self.store,
            pack_registry_discovery=self.discovery,
            action_handlers={"review_approve": lambda _ctx: {"ok": True, "text": "Approved.", "pack": self.store.set_external_pack_review_status(str(row["pack_id"]), local_review_status="approved", approve_current_hash=True)}},
        )

        approved = coordinator.continue_step(lifecycle, action="review_approve")
        repeated = coordinator.continue_step(approved.lifecycle, action="review_approve")

        self.assertEqual("approved", approved.lifecycle_state)
        self.assertFalse(bool(approved.lifecycle.get("usable")))
        self.assertEqual("enable", approved.next_step.action if approved.next_step else None)
        self.assertEqual("approved", repeated.lifecycle_state)
        self.assertEqual("action_refused", repeated.blocked_reason)

    def test_imported_approved_enabled_and_usable_states_are_truthful(self) -> None:
        row = self._record_pack()
        service = PackLifecycleService()
        imported = service.evaluate(imported_pack=row, permission_grants=[]).to_dict()
        self.assertEqual("imported_for_review", imported["state"])
        self.assertFalse(imported["usable"])

        approved_row = self.store.set_external_pack_review_status(str(row["pack_id"]), local_review_status="approved", approve_current_hash=True)
        approved = service.evaluate(imported_pack=approved_row, permission_grants=[]).to_dict()
        self.assertEqual("approved", approved["state"])

        enabled_row = self.store.set_external_pack_enabled(str(row["pack_id"]), enabled=True)
        enabled = service.evaluate(imported_pack=enabled_row, permission_grants=[]).to_dict()
        self.assertEqual("needs_permission", enabled["state"])

        grant = {
            "pack_id": str(row["pack_id"]),
            "adapter_kind": ADAPTER_LOCAL_FILE_IMPORT,
            "state": "granted",
            "granted_path": str(self.root / "sample.json"),
            "path_metadata": {"extension": ".json", "size_bytes": 2, "is_file": True},
        }
        usable = service.evaluate(imported_pack=enabled_row, permission_grants=[grant]).to_dict()
        self.assertTrue(usable["usable"])
        self.assertEqual("usable", usable["state"])

        coordinator = self._coordinator()
        described = coordinator.use_if_usable(
            lifecycle=usable,
            pack=enabled_row,
            adapter_declarations=[_adapter_spec()],
            permission_grants=[grant],
            operation=OP_DESCRIBE_CAPABILITY,
        )
        self.assertIsNone(described.blocked_reason)
        self.assertIn("managed adapter", described.user_message.lower())

        unsupported = coordinator.use_if_usable(
            lifecycle=usable,
            pack=enabled_row,
            adapter_declarations=[_adapter_spec()],
            permission_grants=[grant],
            operation="parse_private_contents",
        )
        self.assertEqual("operation_unsupported", unsupported.blocked_reason)
        self.assertIn("not implemented yet", unsupported.user_message.lower())
        self.assertIn("no_arbitrary_code", described.safe_actions)


class TestPackAcquisitionOrchestratorRegression(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        self.runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                skills_path=str(Path(__file__).resolve().parents[1] / "skills"),
            )
        )

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _post_chat(self, prompt: str) -> tuple[dict[str, Any], str]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "session_id": "pack-acquisition-session",
            "thread_id": "pack-acquisition-thread",
            "source_surface": "webui",
            "purpose": "chat",
            "task_type": "chat",
        }
        handler = _MemoryHandlerForTest(self.runtime, "/chat", payload)
        handler.do_POST()
        body = json.loads(handler.body.decode("utf-8", errors="replace"))
        return body, _assistant_text(body)

    def _import_pack_for_review_via_source_lead(self, *, url: str = "https://example.com/pack.zip") -> tuple[dict[str, Any], str]:
        self.runtime.config = replace(
            self.runtime.config,
            search_enabled=True,
            search_provider="searxng",
            searxng_base_url="http://127.0.0.1:8888",
        )
        self.runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=_FakeOpener({"results": [{"title": "Lead", "url": url}]}),
        )
        archive = _zip_bytes({"repo-main/SKILL.md": b"# Remote Skill\n\nUse as untrusted guidance.\n"})
        self._post_chat("add a capability for analyzing plant watering")
        self._post_chat("yes")
        self._post_chat("yes")
        self._post_chat("yes")
        with mock.patch(
            "agent.packs.external_ingestion.RemotePackFetcher",
            return_value=__import__("agent.packs.remote_fetch", fromlist=["RemotePackFetcher"]).RemotePackFetcher(
                self.runtime.pack_store.external_storage_root(),
                opener=_RemoteFakeOpener({url: _FakeResponse(archive, url=url, content_length=len(archive))}),
            ),
        ):
            return self._post_chat("yes")

    def test_browser_skill_requests_use_acquisition_coordinator(self) -> None:
        for prompt in ("install a skill that lets you browse", "add capability for reading webpages"):
            with self.subTest(prompt=prompt):
                body, text = self._post_chat(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertEqual("action_tool", meta.get("route"))
                self.assertIn("pack_acquisition", meta.get("used_tools") or [])
                self.assertIn("next safe step", text.lower())
                self.assertIn("preview", text.lower())

    def test_generic_non_youtube_capability_uses_same_acquisition_path(self) -> None:
        body, text = self._post_chat("add a capability for making qr codes")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("pack_acquisition", meta.get("used_tools") or [])
        self.assertIn("next safe step", text.lower())

    def test_operational_context_does_not_hijack_acquisition_flow(self) -> None:
        self._post_chat("my computer is slow")
        body, text = self._post_chat("install a skill that lets you browse")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("pack_acquisition", meta.get("used_tools") or [])
        self.assertNotIn("likely cause:", text.lower())

    def test_no_catalog_capability_with_mocked_search_returns_untrusted_leads(self) -> None:
        self.runtime.config = replace(
            self.runtime.config,
            search_enabled=True,
            search_provider="searxng",
            searxng_base_url="http://127.0.0.1:8888",
        )
        self.runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=_FakeOpener(
                {
                    "results": [
                        {
                            "title": "Possible external pack",
                            "url": "https://github.com/example/custom-pack",
                            "content": "Untrusted search metadata.",
                            "engine": "mock",
                        }
                    ]
                }
            ),
        )
        body, text = self._post_chat("add a capability for analyzing plant watering")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        setup = body.get("setup") if isinstance(body.get("setup"), dict) else {}
        acquisition = setup.get("acquisition") if isinstance(setup.get("acquisition"), dict) else {}

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("pack_acquisition", meta.get("used_tools") or [])
        self.assertEqual("untrusted_source_leads", acquisition.get("source_status"))
        self.assertIn("untrusted source leads", text.lower())
        self.assertIn("cannot be fetched/imported", text.lower())
        self.assertIn("source approval", text.lower())
        self.assertFalse(self.runtime.pack_store.list_external_packs())

    def test_yes_after_source_leads_does_not_fetch_or_import(self) -> None:
        self.runtime.config = replace(
            self.runtime.config,
            search_enabled=True,
            search_provider="searxng",
            searxng_base_url="http://127.0.0.1:8888",
        )
        self.runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=_FakeOpener({"results": [{"title": "Lead", "url": "https://example.com/pack.zip"}]}),
        )
        self._post_chat("add a capability for analyzing plant watering")
        _body, text = self._post_chat("yes")

        self.assertIn("source approval preview", text.lower())
        self.assertIn("source is still untrusted", text.lower())
        self.assertIn("no pages were fetched", text.lower())
        self.assertIn("legacy assistant approval flow is read-only", text.lower())
        self.assertIn("remote pack acquisition remains unavailable", text.lower())
        self.assertFalse(self.runtime.pack_store.list_external_packs())

    def test_second_yes_after_source_approval_preview_records_approval_only(self) -> None:
        self.runtime.config = replace(
            self.runtime.config,
            search_enabled=True,
            search_provider="searxng",
            searxng_base_url="http://127.0.0.1:8888",
        )
        self.runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=_FakeOpener({"results": [{"title": "Lead", "url": "https://example.com/pack.zip"}]}),
        )
        self._post_chat("add a capability for analyzing plant watering")
        self._post_chat("yes")
        body, text = self._post_chat("yes")
        setup = body.get("setup") if isinstance(body.get("setup"), dict) else {}
        result = setup.get("result") if isinstance(setup.get("result"), dict) else {}

        self.assertIn("source approval", text.lower())
        self.assertIn("no pack was fetched", text.lower())
        self.assertIn("preview/fetch into quarantine", text.lower())
        self.assertFalse(result.get("did_fetch"))
        self.assertFalse(result.get("did_import"))
        self.assertFalse(result.get("did_install"))
        self.assertFalse(self.runtime.pack_store.list_external_packs())

    def test_source_approval_then_fetch_preview_then_imports_for_review_only(self) -> None:
        url = "https://example.com/pack.zip"
        self.runtime.config = replace(
            self.runtime.config,
            search_enabled=True,
            search_provider="searxng",
            searxng_base_url="http://127.0.0.1:8888",
        )
        self.runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=_FakeOpener({"results": [{"title": "Lead", "url": url}]}),
        )
        archive = _zip_bytes({"repo-main/SKILL.md": b"# Remote Skill\n\nUse as untrusted guidance.\n"})

        self._post_chat("add a capability for analyzing plant watering")
        self._post_chat("yes")
        self._post_chat("yes")
        _preview_body, preview_text = self._post_chat("yes")
        self.assertIn("fetch preview", preview_text.lower())
        self.assertIn("content remains hostile", preview_text.lower())
        self.assertFalse(self.runtime.pack_store.list_external_packs())

        with mock.patch(
            "agent.packs.external_ingestion.RemotePackFetcher",
            return_value=__import__("agent.packs.remote_fetch", fromlist=["RemotePackFetcher"]).RemotePackFetcher(
                self.runtime.pack_store.external_storage_root(),
                opener=_RemoteFakeOpener({url: _FakeResponse(archive, url=url, content_length=len(archive))}),
            ),
        ):
            body, text = self._post_chat("yes")
        setup = body.get("setup") if isinstance(body.get("setup"), dict) else {}
        result = setup.get("result") if isinstance(setup.get("result"), dict) else {}

        self.assertIn("imported", text.lower())
        self.assertIn("for review only", text.lower())
        self.assertIn("no pack was approved", text.lower())
        self.assertIn("not approved", text.lower())
        self.assertIn("not enabled", text.lower())
        self.assertIn("no permissions granted", text.lower())
        self.assertIn("not usable yet", text.lower())
        self.assertIn("next safe step: review/approval", text.lower())
        self.assertTrue(result.get("imported_for_review"))
        self.assertIsInstance(setup.get("review_state"), dict)
        self.assertFalse(result.get("did_approve"))
        self.assertFalse(result.get("did_enable"))
        self.assertFalse(result.get("did_grant_permissions"))
        self.assertFalse(result.get("did_use_pack"))
        packs = self.runtime.pack_store.list_external_packs()
        self.assertEqual(1, len(packs))
        canonical = packs[0].get("canonical_pack") if isinstance(packs[0].get("canonical_pack"), dict) else {}
        trust = canonical.get("trust_anchor") if isinstance(canonical.get("trust_anchor"), dict) else {}
        self.assertEqual("unreviewed", trust.get("local_review_status"))

    def test_review_state_prompt_after_import_reports_not_usable(self) -> None:
        self._import_pack_for_review_via_source_lead()

        body, text = self._post_chat("is that pack safe yet")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        setup = body.get("setup") if isinstance(body.get("setup"), dict) else {}

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("external_pack_review_state", meta.get("used_tools") or [])
        self.assertIn("not approved", text.lower())
        self.assertIn("not enabled", text.lower())
        self.assertIn("not usable yet", text.lower())
        self.assertIn("next safe step: review/approval", text.lower())
        self.assertFalse(setup.get("did_approve"))
        self.assertFalse(setup.get("did_enable"))
        self.assertFalse(setup.get("did_grant_permissions"))
        self.assertFalse(setup.get("did_use_pack"))

    def test_can_i_use_pack_now_after_import_names_review_gate(self) -> None:
        self._import_pack_for_review_via_source_lead()

        _body, text = self._post_chat("can I use that pack now")

        self.assertIn("not usable yet", text.lower())
        self.assertIn("not approved", text.lower())
        self.assertIn("review/approval", text.lower())

    def test_review_approval_preview_then_confirm_records_approval_only(self) -> None:
        self._import_pack_for_review_via_source_lead()

        preview_body, preview_text = self._post_chat("yes")
        preview_meta = preview_body.get("meta") if isinstance(preview_body.get("meta"), dict) else {}
        preview_payload = preview_body.get("setup") if isinstance(preview_body.get("setup"), dict) else {}

        self.assertIn("pack_lifecycle_action", preview_meta.get("used_tools") or [])
        self.assertEqual("review_approve_preview", preview_payload.get("action"))
        self.assertIn("review preview", preview_text.lower())
        self.assertIn("does not turn the skill on", preview_text.lower())
        self.assertIn("grant file permission", preview_text.lower())
        self.assertIn("run code", preview_text.lower())
        self.assertIn("not enabled", preview_text.lower())
        self.assertIn("no permissions granted", preview_text.lower())
        self.assertFalse(preview_payload.get("did_approve"))
        self.assertFalse(preview_payload.get("did_enable"))
        self.assertFalse(preview_payload.get("did_grant_permissions"))
        self.assertFalse(preview_payload.get("did_use_pack"))
        packs = self.runtime.pack_store.list_external_packs()
        canonical = packs[0].get("canonical_pack") if isinstance(packs[0].get("canonical_pack"), dict) else {}
        trust = canonical.get("trust_anchor") if isinstance(canonical.get("trust_anchor"), dict) else {}
        self.assertEqual("unreviewed", trust.get("local_review_status"))

        approve_body, approve_text = self._post_chat("yes")
        approve_payload = approve_body.get("setup") if isinstance(approve_body.get("setup"), dict) else {}

        self.assertEqual("review_approve", approve_payload.get("action"))
        self.assertIn("draft approved", approve_text.lower())
        self.assertIn("still not turned on", approve_text.lower())
        self.assertIn("no permissions were granted", approve_text.lower())
        self.assertIn("no code ran", approve_text.lower())
        self.assertTrue(approve_payload.get("did_approve"))
        self.assertFalse(approve_payload.get("did_enable"))
        self.assertFalse(approve_payload.get("did_grant_permissions"))
        self.assertFalse(approve_payload.get("did_use_pack"))
        self.assertFalse(approve_payload.get("enabled"))
        self.assertFalse(approve_payload.get("usable"))
        packs = self.runtime.pack_store.list_external_packs()
        canonical = packs[0].get("canonical_pack") if isinstance(packs[0].get("canonical_pack"), dict) else {}
        trust = canonical.get("trust_anchor") if isinstance(canonical.get("trust_anchor"), dict) else {}
        self.assertEqual("approved", trust.get("local_review_status"))

        enable_preview_body, enable_preview_text = self._post_chat("yes")
        enable_preview_payload = enable_preview_body.get("setup") if isinstance(enable_preview_body.get("setup"), dict) else {}
        self.assertEqual("enable_preview", enable_preview_payload.get("action"))
        self.assertIn("turn-on preview", enable_preview_text.lower())
        self.assertIn("will not grant file permission", enable_preview_text.lower())
        self.assertIn("run code", enable_preview_text.lower())
        self.assertIn("use the skill", enable_preview_text.lower())
        self.assertFalse(enable_preview_payload.get("did_enable"))
        self.assertFalse(enable_preview_payload.get("did_grant_permissions"))
        self.assertFalse(enable_preview_payload.get("did_use_pack"))
        packs = self.runtime.pack_store.list_external_packs()
        canonical = packs[0].get("canonical_pack") if isinstance(packs[0].get("canonical_pack"), dict) else {}
        runtime = canonical.get("runtime") if isinstance(canonical.get("runtime"), dict) else {}
        self.assertFalse(runtime.get("enabled"))

        enable_body, enable_text = self._post_chat("yes")
        enable_payload = enable_body.get("setup") if isinstance(enable_body.get("setup"), dict) else {}
        self.assertEqual("enable", enable_payload.get("action"))
        self.assertIn("is turned on", enable_text.lower())
        self.assertIn("no permissions were granted", enable_text.lower())
        self.assertIn("no adapter ran", enable_text.lower())
        self.assertIn("did not use the skill", enable_text.lower())
        self.assertTrue(enable_payload.get("did_enable"))
        self.assertFalse(enable_payload.get("did_grant_permissions"))
        self.assertFalse(enable_payload.get("did_use_pack"))
        self.assertTrue(enable_payload.get("usable"))
        packs = self.runtime.pack_store.list_external_packs()
        canonical = packs[0].get("canonical_pack") if isinstance(packs[0].get("canonical_pack"), dict) else {}
        runtime = canonical.get("runtime") if isinstance(canonical.get("runtime"), dict) else {}
        self.assertTrue(runtime.get("enabled"))

        repeat_body, repeat_text = self._post_chat("yes")
        repeat_meta = repeat_body.get("meta") if isinstance(repeat_body.get("meta"), dict) else {}
        self.assertEqual("assistant_clarification", repeat_meta.get("route"))
        self.assertIn("current action", repeat_text.lower())

    def test_can_i_use_it_now_after_review_approval_names_enable_gate(self) -> None:
        self._import_pack_for_review_via_source_lead()
        self._post_chat("yes")
        self._post_chat("yes")

        _body, text = self._post_chat("can I use it now")

        self.assertIn("not enabled", text.lower())
        self.assertIn("not usable", text.lower())
        self.assertIn("enable", text.lower())

    def test_direct_prompts_after_enablement_report_lifecycle_truth(self) -> None:
        self._import_pack_for_review_via_source_lead()
        self._post_chat("yes")
        self._post_chat("yes")
        self._post_chat("yes")
        self._post_chat("yes")

        _body, use_text = self._post_chat("can I use it now?")
        self.assertIn("enabled: true", use_text.lower())
        self.assertIn("usable", use_text.lower())
        self.assertIn("usable now", use_text.lower())

        _body, permission_text = self._post_chat("did you grant permissions?")
        self.assertIn("no permissions granted", permission_text.lower())
        self.assertIn("usable", permission_text.lower())

    def test_permission_preview_confirm_records_grant_only(self) -> None:
        self._post_chat("Look through my YouTube history and find the video about neurons differentiating during animal infancy.")
        self._post_chat("yes")
        self._post_chat("yes")
        self._post_chat("yes")
        self._post_chat("yes")
        self._post_chat("yes")
        enabled_body, enabled_text = self._post_chat("yes")
        enabled_payload = enabled_body.get("setup") if isinstance(enabled_body.get("setup"), dict) else {}
        self.assertEqual("enable", enabled_payload.get("action"))
        self.assertEqual("needs_permission", (enabled_payload.get("lifecycle") or {}).get("state"))
        self.assertIn("next safe step", enabled_text.lower())

        permission_body, permission_text = self._post_chat("yes")
        permission_payload = permission_body.get("setup") if isinstance(permission_body.get("setup"), dict) else {}
        self.assertEqual("managed_adapter_permission_request", permission_payload.get("type"))
        self.assertIn("needs your permission to use one selected file", permission_text.lower())
        self.assertIn("scope: user_selected_file_only", permission_text.lower())
        self.assertIn(".json", permission_text.lower())
        self.assertIn("will not run code", permission_text.lower())
        self.assertIn("use the skill", permission_text.lower())
        self.assertIn("read the file yet", permission_text.lower())
        self.assertFalse(permission_payload.get("did_grant_permissions"))
        self.assertFalse(permission_payload.get("did_invoke_adapter"))
        self.assertFalse(permission_payload.get("did_use_pack"))

        selected_path = Path(self.runtime.pack_store.external_storage_root()) / "watch-history.json"
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text('{"private": "history contents"}\n', encoding="utf-8")
        with mock.patch("pathlib.Path.read_text", side_effect=AssertionError("permission preview should not read files")):
            path_body, path_text = self._post_chat(f"use {selected_path}")
        path_payload = path_body.get("setup") if isinstance(path_body.get("setup"), dict) else {}
        self.assertEqual("managed_adapter_permission_preview", path_payload.get("type"))
        self.assertIn("<redacted-local-history-path>/watch-history.json", path_text)
        self.assertIn("will not run code", path_text.lower())
        self.assertIn("read the file", path_text.lower())
        self.assertIn("use the skill", path_text.lower())

        grant_body, grant_text = self._post_chat("yes")
        grant_payload = grant_body.get("setup") if isinstance(grant_body.get("setup"), dict) else {}
        self.assertEqual("managed_adapter_permission_grant", grant_payload.get("type"))
        self.assertIn("selected-file permission recorded", grant_text.lower())
        self.assertIn("i did not read or parse the file", grant_text.lower())
        self.assertIn("run code", grant_text.lower())
        self.assertIn("run an adapter", grant_text.lower())
        self.assertTrue(grant_payload.get("did_grant_permissions"))
        self.assertFalse(grant_payload.get("did_invoke_adapter"))
        self.assertFalse(grant_payload.get("did_use_pack"))
        self.assertFalse(grant_payload.get("executes_code"))
        self.assertFalse(grant_payload.get("reads_file"))
        self.assertTrue(grant_payload.get("usable"))
        self.assertEqual("usable", (grant_payload.get("lifecycle") or {}).get("state"))
        self.assertNotIn("history contents", str(grant_payload))

        repeat_body, repeat_text = self._post_chat("yes")
        repeat_meta = repeat_body.get("meta") if isinstance(repeat_body.get("meta"), dict) else {}
        self.assertEqual("assistant_clarification", repeat_meta.get("route"))
        self.assertIn("current action", repeat_text.lower())

        _body, use_text = self._post_chat("use it now")
        self.assertIn("usable", use_text.lower())
        self.assertIn("did not invoke or use", use_text.lower())
        self.assertIn("specific input", use_text.lower())

        packs = self.runtime.pack_store.list_external_packs()
        canonical = packs[0].get("canonical_pack") if packs and isinstance(packs[0].get("canonical_pack"), dict) else {}
        pack_name = str(canonical.get("display_name") or canonical.get("name") or "YouTube History Search")
        preview_body, preview_text = self._post_chat(f"dry run {pack_name}")
        preview_payload = preview_body.get("setup") if isinstance(preview_body.get("setup"), dict) else {}
        self.assertEqual("managed_adapter_invocation_preview", preview_payload.get("type"))
        self.assertIn("safe check preview", preview_text.lower())
        self.assertIn("built-in safe adapter", preview_text.lower())
        self.assertIn("reads file contents: no", preview_text.lower())
        self.assertFalse(preview_payload.get("did_invoke_adapter"))
        self.assertFalse(preview_payload.get("did_use_pack"))

        original_read_text = Path.read_text

        def _guard_read_text(path_obj: Path, *args: Any, **kwargs: Any) -> str:
            if Path(path_obj) == selected_path:
                raise AssertionError("dry-run should not read private file contents")
            return original_read_text(path_obj, *args, **kwargs)

        with mock.patch("pathlib.Path.read_text", new=_guard_read_text):
            invoke_body, invoke_text = self._post_chat("yes")
        invoke_payload = invoke_body.get("setup") if isinstance(invoke_body.get("setup"), dict) else {}
        invocation = invoke_payload.get("invocation") if isinstance(invoke_payload.get("invocation"), dict) else {}
        self.assertEqual("managed_adapter_invocation", invoke_payload.get("type"))
        self.assertEqual(OP_DRY_RUN, invocation.get("operation"))
        self.assertTrue(invocation.get("ok"))
        self.assertTrue(invoke_payload.get("did_invoke_adapter"))
        self.assertFalse(invoke_payload.get("did_use_pack"))
        self.assertFalse(invoke_payload.get("reads_file_contents"))
        self.assertFalse(invoke_payload.get("writes_data"))
        self.assertFalse(invoke_payload.get("executes_code"))
        self.assertFalse(invoke_payload.get("task_completed"))
        self.assertIn("external code executed: no", invoke_text.lower())
        self.assertNotIn("history contents", str(invoke_payload))

        repeat_body, repeat_text = self._post_chat("yes")
        repeat_meta = repeat_body.get("meta") if isinstance(repeat_body.get("meta"), dict) else {}
        self.assertEqual("assistant_clarification", repeat_meta.get("route"))
        self.assertIn("current action", repeat_text.lower())

        blocked_body, blocked_text = self._post_chat(f"read {pack_name} file now")
        blocked_payload = blocked_body.get("setup") if isinstance(blocked_body.get("setup"), dict) else {}
        self.assertEqual("managed_adapter_invocation_blocked", blocked_payload.get("type"))
        self.assertIn("does not yet have a safe content-read/search operation", blocked_text)
        self.assertFalse(blocked_payload.get("did_invoke_adapter"))
        self.assertFalse(blocked_payload.get("did_use_pack"))

    def test_no_after_fetch_preview_cancels_without_fetch(self) -> None:
        self.runtime.config = replace(
            self.runtime.config,
            search_enabled=True,
            search_provider="searxng",
            searxng_base_url="http://127.0.0.1:8888",
        )
        self.runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=_FakeOpener({"results": [{"title": "Lead", "url": "https://example.com/pack.zip"}]}),
        )
        self._post_chat("add a capability for analyzing plant watering")
        self._post_chat("yes")
        self._post_chat("yes")
        self._post_chat("yes")
        _body, text = self._post_chat("no")

        self.assertIn("did not fetch", text.lower())
        self.assertIn("no content was downloaded", text.lower())
        self.assertFalse(self.runtime.pack_store.list_external_packs())

    def test_no_after_read_only_source_preview_has_no_mutation_to_cancel(self) -> None:
        self.runtime.config = replace(
            self.runtime.config,
            search_enabled=True,
            search_provider="searxng",
            searxng_base_url="http://127.0.0.1:8888",
        )
        self.runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=_FakeOpener({"results": [{"title": "Lead", "url": "https://example.com/pack.zip"}]}),
        )
        self._post_chat("add a capability for analyzing plant watering")
        self._post_chat("yes")
        _body, text = self._post_chat("no")

        self.assertIn("don’t have a current action", text.lower())
        self.assertFalse(self.runtime.pack_store.list_external_packs())


if __name__ == "__main__":
    unittest.main()
