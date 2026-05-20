from __future__ import annotations

import json
import os
import tempfile
import unittest
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
        self.assertIn("not implemented", text.lower())
        self.assertIn("did not fetch", text.lower())
        self.assertFalse(self.runtime.pack_store.list_external_packs())


if __name__ == "__main__":
    unittest.main()
