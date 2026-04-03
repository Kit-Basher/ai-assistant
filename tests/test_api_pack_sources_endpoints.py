from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.packs.remote_fetch import RemotePackFetcher


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


class _FakeOpener:
    def __init__(self, mapping: dict[str, _FakeResponse]) -> None:
        self.mapping = mapping

    def open(self, request, timeout: int = 15):  # noqa: ANN001
        _ = timeout
        url = getattr(request, "full_url", str(request))
        response = self.mapping.get(url)
        if response is None:
            raise RuntimeError(f"unexpected url: {url}")
        return response


def _config(registry_path: str, db_path: str, skills_path: str) -> Config:
    return Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path=skills_path,
        ollama_host="http://127.0.0.1:11434",
        ollama_model="llama3",
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=15,
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url="http://127.0.0.1:11434",
        anthropic_api_key=None,
        llm_selector="single",
        llm_broker_policy_path=None,
        llm_allow_remote=True,
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=registry_path,
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=os.path.join(os.path.dirname(db_path), "usage_stats.json"),
        llm_health_state_path=os.path.join(os.path.dirname(db_path), "llm_health_state.json"),
        llm_automation_enabled=False,
        model_scout_state_path=os.path.join(os.path.dirname(db_path), "model_scout_state.json"),
        autopilot_notify_store_path=os.path.join(os.path.dirname(db_path), "llm_notifications.json"),
    )


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object] | None = None) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.status_code = 0
        self.content_type = ""
        self.body = b""
        self._payload = payload or {}

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        self.status_code = status
        self.content_type = "application/json"
        self.body = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        cache_control: str | None = None,
    ) -> None:
        _ = cache_control
        self.status_code = status
        self.content_type = content_type
        self.body = body

    def _read_json(self) -> dict[str, object]:
        return self._payload


class TestAPIPackSourceEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.skills_path = os.path.join(self.tmpdir.name, "skills")
        os.makedirs(self.skills_path, exist_ok=True)
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        self.runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.skills_path))
        self.storage_root = self.runtime.pack_store.external_storage_root()
        os.makedirs(self.storage_root, exist_ok=True)
        self.catalog_path = os.path.join(self.storage_root, "registry_catalog.json")
        self.sources_path = os.path.join(self.storage_root, "registry_sources.json")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _write_source_catalog(self, *, source_id: str = "local-registry") -> None:
        with open(self.catalog_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "packs": [
                        {
                            "id": "docs-skill",
                            "name": "Docs Skill",
                            "summary": "Summarizes docs safely.",
                            "author": "Example",
                            "source_url": "https://github.com/example/docs-skill/archive/main.zip",
                            "source_kind_hint": "github_archive",
                            "latest_ref_hint": "main",
                            "has_skill_md": True,
                            "tags": ["docs"],
                        }
                    ]
                },
                handle,
                ensure_ascii=True,
            )
        with open(self.sources_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "sources": [
                        {
                            "id": source_id,
                            "kind": "local_catalog",
                            "name": "Local Registry",
                            "base_url": self.catalog_path,
                            "enabled": True,
                        }
                    ]
                },
                handle,
                ensure_ascii=True,
            )

    def _policy_path(self) -> str:
        return os.path.join(self.storage_root, "registry_source_policy.json")

    def _sources_file_path(self) -> str:
        return self.sources_path

    def _read_json_file(self, path: str) -> dict[str, object]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _audit_rows(self) -> list[dict[str, object]]:
        audit_path = os.environ["AGENT_AUDIT_LOG_PATH"]
        if not os.path.exists(audit_path):
            return []
        with open(audit_path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle.read().splitlines() if line.strip()]

    def test_pack_source_list_search_preview_and_install_handoff(self) -> None:
        remote_url = "https://github.com/example/docs-skill/archive/main.zip"
        with open(self.catalog_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "packs": [
                        {
                            "id": "docs-skill",
                            "name": "Docs Skill",
                            "summary": "Summarizes docs safely.",
                            "author": "Example",
                            "source_url": remote_url,
                            "source_kind_hint": "github_archive",
                            "latest_ref_hint": "main",
                            "has_skill_md": True,
                            "tags": ["docs"],
                        }
                    ]
                },
                handle,
                ensure_ascii=True,
            )
        with open(self.sources_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "sources": [
                        {
                            "id": "local-registry",
                            "kind": "local_catalog",
                            "name": "Local Registry",
                            "base_url": self.catalog_path,
                            "enabled": True,
                        }
                    ]
                },
                handle,
                ensure_ascii=True,
            )

        sources_handler = _HandlerForTest(self.runtime, "/pack_sources")
        sources_handler.do_GET()
        sources_payload = json.loads(sources_handler.body.decode("utf-8"))
        self.assertEqual(200, sources_handler.status_code)
        self.assertEqual("local-registry", sources_payload["sources"][0]["id"])

        list_handler = _HandlerForTest(self.runtime, "/pack_sources/local-registry/packs")
        list_handler.do_GET()
        list_payload = json.loads(list_handler.body.decode("utf-8"))
        self.assertEqual(200, list_handler.status_code)
        self.assertEqual(1, list_payload["count"])
        self.assertEqual("docs-skill", list_payload["packs"][0]["remote_id"])

        search_handler = _HandlerForTest(self.runtime, "/pack_sources/local-registry/search?q=docs")
        search_handler.do_GET()
        search_payload = json.loads(search_handler.body.decode("utf-8"))
        self.assertEqual(200, search_handler.status_code)
        self.assertEqual(1, search_payload["search"]["count"])

        before_preview_count = len(self.runtime.pack_store.list_external_packs())
        preview_handler = _HandlerForTest(self.runtime, "/pack_sources/local-registry/packs/docs-skill/preview")
        preview_handler.do_GET()
        preview_payload = json.loads(preview_handler.body.decode("utf-8"))
        after_preview_count = len(self.runtime.pack_store.list_external_packs())
        self.assertEqual(200, preview_handler.status_code)
        self.assertEqual(before_preview_count, after_preview_count)
        self.assertFalse(preview_payload["preview"]["fetched"])
        self.assertIn("Nothing has been fetched yet.", preview_payload["preview"]["source_hints"])
        handoff = preview_payload["preview"]["install_handoff"]
        self.assertEqual(remote_url, handoff["source"])
        self.assertEqual("github_archive", handoff["source_kind"])

        archive = _zip_bytes(
            {
                "repo-main/SKILL.md": b"# Docs Skill\n\nUse the repository notes.\n",
                "repo-main/references/guide.md": b"# Guide\n\nHelpful docs.\n",
            }
        )
        fake_fetcher = RemotePackFetcher(
            self.runtime.pack_store.external_storage_root(),
            opener=_FakeOpener({remote_url: _FakeResponse(archive, url=remote_url)}),
        )
        with mock.patch("agent.packs.external_ingestion.RemotePackFetcher", return_value=fake_fetcher):
            install_handler = _HandlerForTest(self.runtime, "/packs/install", handoff)
            install_handler.do_POST()
            install_payload = json.loads(install_handler.body.decode("utf-8"))

        self.assertEqual(200, install_handler.status_code)
        self.assertTrue(install_payload["ok"])
        self.assertEqual("portable_text_skill", install_payload["normalization_result"]["classification"])
        self.assertEqual("normalized", install_payload["normalization_result"]["status"])
        self.assertEqual(1, len(self.runtime.pack_store.list_external_packs()))

    def test_pack_install_partial_safe_import_returns_plain_actionable_message(self) -> None:
        pack_dir = os.path.join(self.tmpdir.name, "scripted_pack")
        os.makedirs(os.path.join(pack_dir, "scripts"), exist_ok=True)
        with open(os.path.join(pack_dir, "SKILL.md"), "w", encoding="utf-8") as handle:
            handle.write("# Scripted Pack\n\nUse the notes.\n")
        with open(os.path.join(pack_dir, "scripts", "install.sh"), "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\necho hi\n")

        install_handler = _HandlerForTest(self.runtime, "/packs/install", {"source": pack_dir})
        install_handler.do_POST()
        install_payload = json.loads(install_handler.body.decode("utf-8"))

        self.assertEqual(200, install_handler.status_code)
        self.assertTrue(install_payload["ok"])
        self.assertEqual("partial_safe_import", install_payload["normalization_result"]["status"])
        self.assertIn("only imported the safe parts", str(install_payload["message"] or "").lower())
        self.assertIn("unsafe", str(install_payload["why"] or "").lower())
        self.assertTrue(str(install_payload["next_action"] or "").strip())

    def test_pack_source_policy_blocks_list_search_and_preview_consistently(self) -> None:
        remote_url = "https://github.com/example/docs-skill/archive/main.zip"
        with open(self.catalog_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "packs": [
                        {
                            "id": "docs-skill",
                            "name": "Docs Skill",
                            "summary": "Summarizes docs safely.",
                            "source_url": remote_url,
                            "source_kind_hint": "github_archive",
                            "latest_ref_hint": "main",
                            "has_skill_md": True,
                        }
                    ]
                },
                handle,
                ensure_ascii=True,
            )
        with open(self.sources_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "sources": [
                        {
                            "id": "local-registry",
                            "kind": "local_catalog",
                            "name": "Local Registry",
                            "base_url": self.catalog_path,
                            "enabled": True,
                        }
                    ]
                },
                handle,
                ensure_ascii=True,
            )
        with open(os.path.join(self.storage_root, "registry_source_policy.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "defaults": {
                        "allowlisted": False,
                    }
                },
                handle,
                ensure_ascii=True,
            )

        sources_handler = _HandlerForTest(self.runtime, "/pack_sources")
        sources_handler.do_GET()
        sources_payload = json.loads(sources_handler.body.decode("utf-8"))
        self.assertEqual(200, sources_handler.status_code)
        self.assertEqual("local-registry", sources_payload["sources"][0]["id"])
        self.assertFalse(sources_payload["sources"][0]["allowed_by_policy"])
        self.assertEqual("source_not_allowlisted", sources_payload["sources"][0]["blocked_reason"])

        for path in (
            "/pack_sources/local-registry/packs",
            "/pack_sources/local-registry/search?q=docs",
            "/pack_sources/local-registry/packs/docs-skill/preview",
        ):
            handler = _HandlerForTest(self.runtime, path)
            handler.do_GET()
            payload = json.loads(handler.body.decode("utf-8"))
            self.assertEqual(400, handler.status_code)
            self.assertEqual("pack_source_blocked_by_policy", payload["error"])
            self.assertEqual("blocked_by_policy", payload["error_kind"])
            self.assertEqual("local-registry", payload["policy"]["source_id"])
            self.assertIn("discovery policy blocks", str(payload.get("why") or "").lower())
            self.assertTrue(str(payload.get("next_action") or "").strip())

    def test_pack_source_policy_api_reads_and_updates_global_defaults(self) -> None:
        self._write_source_catalog()

        get_handler = _HandlerForTest(self.runtime, "/pack_sources/policy")
        get_handler.do_GET()
        get_payload = json.loads(get_handler.body.decode("utf-8"))
        self.assertEqual(200, get_handler.status_code)
        self.assertTrue(get_payload["ok"])
        self.assertTrue(get_payload["normalized_policy"]["defaults"]["allowlisted"])
        self.assertEqual(str(self._policy_path()), get_payload["path"])

        put_handler = _HandlerForTest(
            self.runtime,
            "/pack_sources/policy",
            {
                "allowlisted": False,
                "cache_ttl_seconds": 9,
                "max_results": 7,
                "notes": "operator tightened defaults",
            },
        )
        put_handler.do_PUT()
        put_payload = json.loads(put_handler.body.decode("utf-8"))
        self.assertEqual(200, put_handler.status_code)
        self.assertFalse(put_payload["normalized_policy"]["defaults"]["allowlisted"])
        self.assertEqual(9, put_payload["normalized_policy"]["defaults"]["cache_ttl_seconds"])
        self.assertEqual(7, put_payload["normalized_policy"]["defaults"]["max_results"])
        last_change = put_payload["persisted_policy"]["meta"]["last_change"]
        self.assertEqual("defaults", last_change["scope"])
        self.assertEqual(
            ["allowlisted", "cache_ttl_seconds", "max_results", "notes"],
            last_change["changed_fields"],
        )

        list_handler = _HandlerForTest(self.runtime, "/pack_sources/local-registry/packs")
        list_handler.do_GET()
        list_payload = json.loads(list_handler.body.decode("utf-8"))
        self.assertEqual(400, list_handler.status_code)
        self.assertEqual("pack_source_blocked_by_policy", list_payload["error"])
        self.assertEqual("source_not_allowlisted", list_payload["policy"]["blocked_reason"])

        audit_rows = self._audit_rows()
        self.assertTrue(audit_rows)
        self.assertEqual("packs.discovery_policy.update", audit_rows[-1]["action"])
        self.assertEqual("success", audit_rows[-1]["outcome"])

    def test_pack_source_policy_api_reads_and_updates_per_source_override(self) -> None:
        self._write_source_catalog()
        with open(self._policy_path(), "w", encoding="utf-8") as handle:
            json.dump({"defaults": {"allowlisted": False}}, handle, ensure_ascii=True)

        get_handler = _HandlerForTest(self.runtime, "/pack_sources/local-registry/policy")
        get_handler.do_GET()
        get_payload = json.loads(get_handler.body.decode("utf-8"))
        self.assertEqual(200, get_handler.status_code)
        self.assertEqual("local-registry", get_payload["source"]["id"])
        self.assertFalse(get_payload["effective_policy"]["allowlisted"])

        put_handler = _HandlerForTest(
            self.runtime,
            "/pack_sources/local-registry/policy",
            {
                "allowlisted": True,
                "cache_ttl_seconds": 5,
                "max_results": 3,
                "notes": "allow this one source",
            },
        )
        put_handler.do_PUT()
        put_payload = json.loads(put_handler.body.decode("utf-8"))
        self.assertEqual(200, put_handler.status_code)
        self.assertEqual("local-registry", put_payload["effective_policy"]["source_id"])
        self.assertTrue(put_payload["effective_policy"]["allowlisted"])
        self.assertEqual(5, put_payload["effective_policy"]["cache_ttl_seconds"])
        self.assertEqual(3, put_payload["effective_policy"]["max_results"])
        self.assertEqual("source:local-registry", put_payload["meta"]["last_change"]["scope"])

        list_handler = _HandlerForTest(self.runtime, "/pack_sources/local-registry/packs")
        list_handler.do_GET()
        list_payload = json.loads(list_handler.body.decode("utf-8"))
        self.assertEqual(200, list_handler.status_code)
        self.assertTrue(list_payload["ok"])
        self.assertEqual(1, list_payload["count"])
        self.assertTrue(list_payload["policy"]["allowlisted"])

    def test_pack_source_policy_api_rejects_invalid_writes_without_partial_persist(self) -> None:
        self._write_source_catalog()
        with open(self._policy_path(), "w", encoding="utf-8") as handle:
            json.dump({"defaults": {"allowlisted": True, "cache_ttl_seconds": 11}}, handle, ensure_ascii=True)
        with open(self._policy_path(), "r", encoding="utf-8") as handle:
            before_text = handle.read()

        cases = [
            ("/pack_sources/policy", {"unexpected": True}, "unknown_policy_fields"),
            ("/pack_sources/policy", {"cache_ttl_seconds": -1}, "invalid_cache_ttl_seconds"),
            ("/pack_sources/policy", {"max_results": 0}, "invalid_max_results"),
            ("/pack_sources/policy", {"allowed_source_kinds": ["not_a_kind"]}, "invalid_allowed_source_kinds"),
            ("/pack_sources/missing/policy", {"allowlisted": True}, "pack_source_not_found"),
        ]
        for path, payload, expected_error in cases:
            handler = _HandlerForTest(self.runtime, path, payload)
            handler.do_PUT()
            body = json.loads(handler.body.decode("utf-8"))
            self.assertEqual(400, handler.status_code)
            self.assertEqual(expected_error, body["error"])

        with open(self._policy_path(), "r", encoding="utf-8") as handle:
            after_text = handle.read()
        self.assertEqual(before_text, after_text)

        list_handler = _HandlerForTest(self.runtime, "/pack_sources/local-registry/packs")
        list_handler.do_GET()
        list_payload = json.loads(list_handler.body.decode("utf-8"))
        self.assertEqual(200, list_handler.status_code)
        self.assertEqual(1, list_payload["count"])

    def test_pack_source_policy_endpoints_are_loopback_only(self) -> None:
        self._write_source_catalog()

        get_handler = _HandlerForTest(self.runtime, "/pack_sources/policy")
        get_handler.client_address = ("203.0.113.20", 4242)
        get_handler.do_GET()
        get_payload = json.loads(get_handler.body.decode("utf-8"))
        self.assertEqual(403, get_handler.status_code)
        self.assertEqual("forbidden", get_payload["error"])

        put_handler = _HandlerForTest(
            self.runtime,
            "/pack_sources/local-registry/policy",
            {"allowlisted": False},
        )
        put_handler.client_address = ("203.0.113.20", 4242)
        put_handler.do_PUT()
        put_payload = json.loads(put_handler.body.decode("utf-8"))
        self.assertEqual(403, put_handler.status_code)
        self.assertEqual("forbidden", put_payload["error"])

    def test_pack_source_catalog_api_crud_and_audit(self) -> None:
        create_handler = _HandlerForTest(
            self.runtime,
            "/pack_sources/catalog",
            {
                "source_id": "generic-registry",
                "name": "Generic Registry",
                "kind": "generic_registry_api",
                "base_url": "https://example.com/catalog.json",
                "enabled": True,
                "supports_search": True,
                "supports_preview": True,
                "supports_compare_hint": True,
                "notes": "operator managed",
            },
        )
        create_handler.do_POST()
        create_payload = json.loads(create_handler.body.decode("utf-8"))
        self.assertEqual(200, create_handler.status_code)
        self.assertEqual("generic-registry", create_payload["source"]["id"])
        self.assertEqual("create", create_payload["meta"]["last_change"]["operation"])

        catalog_handler = _HandlerForTest(self.runtime, "/pack_sources/catalog")
        catalog_handler.do_GET()
        catalog_payload = json.loads(catalog_handler.body.decode("utf-8"))
        self.assertEqual(200, catalog_handler.status_code)
        self.assertEqual(1, len(catalog_payload["normalized_sources"]))
        self.assertEqual("generic-registry", catalog_payload["normalized_sources"][0]["source"]["id"])

        pack_sources_handler = _HandlerForTest(self.runtime, "/pack_sources")
        pack_sources_handler.do_GET()
        pack_sources_payload = json.loads(pack_sources_handler.body.decode("utf-8"))
        self.assertEqual(200, pack_sources_handler.status_code)
        self.assertEqual("generic-registry", pack_sources_payload["sources"][0]["id"])
        self.assertTrue(pack_sources_payload["sources"][0]["queryable"])

        update_handler = _HandlerForTest(
            self.runtime,
            "/pack_sources/catalog/generic-registry",
            {
                "enabled": False,
                "notes": "disabled for maintenance",
            },
        )
        update_handler.do_PUT()
        update_payload = json.loads(update_handler.body.decode("utf-8"))
        self.assertEqual(200, update_handler.status_code)
        self.assertFalse(update_payload["source"]["enabled"])
        self.assertEqual("update", update_payload["meta"]["last_change"]["operation"])

        after_update_handler = _HandlerForTest(self.runtime, "/pack_sources")
        after_update_handler.do_GET()
        after_update_payload = json.loads(after_update_handler.body.decode("utf-8"))
        self.assertEqual(200, after_update_handler.status_code)
        self.assertFalse(after_update_payload["sources"][0]["queryable"])
        self.assertEqual("source_disabled", after_update_payload["sources"][0]["blocked_reason"])

        delete_handler = _HandlerForTest(self.runtime, "/pack_sources/catalog/generic-registry")
        delete_handler.do_DELETE()
        delete_payload = json.loads(delete_handler.body.decode("utf-8"))
        self.assertEqual(200, delete_handler.status_code)
        self.assertEqual("generic-registry", delete_payload["deleted_source_id"])
        self.assertEqual("delete", delete_payload["persisted_catalog"]["meta"]["last_change"]["operation"])

        final_sources_handler = _HandlerForTest(self.runtime, "/pack_sources")
        final_sources_handler.do_GET()
        final_sources_payload = json.loads(final_sources_handler.body.decode("utf-8"))
        self.assertEqual(200, final_sources_handler.status_code)
        self.assertEqual([], final_sources_payload["sources"])

        sources_file = self._read_json_file(self._sources_file_path())
        self.assertEqual([], sources_file["sources"])
        self.assertEqual("delete", sources_file["meta"]["last_change"]["operation"])

        audit_rows = self._audit_rows()
        actions = [row["action"] for row in audit_rows]
        self.assertIn("packs.discovery_catalog.create", actions)
        self.assertIn("packs.discovery_catalog.update", actions)
        self.assertIn("packs.discovery_catalog.delete", actions)

    def test_pack_source_catalog_api_rejects_invalid_writes_without_persisting(self) -> None:
        create_handler = _HandlerForTest(
            self.runtime,
            "/pack_sources/catalog",
            {
                "source_id": "generic-registry",
                "name": "Generic Registry",
                "kind": "generic_registry_api",
                "base_url": "https://example.com/catalog.json",
            },
        )
        create_handler.do_POST()
        self.assertEqual(200, create_handler.status_code)
        before_payload = self._read_json_file(self._sources_file_path())

        cases = [
            ("/pack_sources/catalog", {"source_id": "generic-registry", "name": "Dup", "kind": "generic_registry_api", "base_url": "https://example.com/other.json"}, "duplicate_source_id", "POST"),
            ("/pack_sources/catalog", {"source_id": "bad-kind", "name": "Bad Kind", "kind": "nope", "base_url": "https://example.com/catalog.json"}, "invalid_source_kind", "POST"),
            ("/pack_sources/catalog", {"source_id": "bad-url", "name": "Bad Url", "kind": "generic_registry_api", "base_url": "http://example.com/catalog.json"}, "invalid_base_url", "POST"),
            ("/pack_sources/catalog", {"source_id": "unknown-field", "name": "Unknown", "kind": "generic_registry_api", "base_url": "https://example.com/catalog.json", "extra": True}, "unknown_source_fields", "POST"),
            ("/pack_sources/catalog/generic-registry", {"source_id": "renamed"}, "source_id_rename_not_supported", "PUT"),
        ]
        for path, payload, expected_error, method in cases:
            handler = _HandlerForTest(self.runtime, path, payload)
            if method == "POST":
                handler.do_POST()
            else:
                handler.do_PUT()
            body = json.loads(handler.body.decode("utf-8"))
            self.assertEqual(400, handler.status_code)
            self.assertEqual(expected_error, body["error"])

        after_payload = self._read_json_file(self._sources_file_path())
        self.assertEqual(before_payload, after_payload)

    def test_pack_source_catalog_api_policy_interaction_and_loopback_enforcement(self) -> None:
        with open(self._policy_path(), "w", encoding="utf-8") as handle:
            json.dump({"defaults": {"allowlisted": False}}, handle, ensure_ascii=True)

        create_handler = _HandlerForTest(
            self.runtime,
            "/pack_sources/catalog",
            {
                "source_id": "generic-registry",
                "name": "Generic Registry",
                "kind": "generic_registry_api",
                "base_url": "https://example.com/catalog.json",
            },
        )
        create_handler.do_POST()
        self.assertEqual(200, create_handler.status_code)

        pack_sources_handler = _HandlerForTest(self.runtime, "/pack_sources")
        pack_sources_handler.do_GET()
        pack_sources_payload = json.loads(pack_sources_handler.body.decode("utf-8"))
        self.assertEqual(200, pack_sources_handler.status_code)
        self.assertFalse(pack_sources_payload["sources"][0]["queryable"])
        self.assertEqual("source_not_allowlisted", pack_sources_payload["sources"][0]["blocked_reason"])

        get_handler = _HandlerForTest(self.runtime, "/pack_sources/catalog")
        get_handler.client_address = ("203.0.113.20", 4242)
        get_handler.do_GET()
        get_payload = json.loads(get_handler.body.decode("utf-8"))
        self.assertEqual(403, get_handler.status_code)
        self.assertEqual("forbidden", get_payload["error"])

        post_handler = _HandlerForTest(
            self.runtime,
            "/pack_sources/catalog",
            {
                "source_id": "second-registry",
                "name": "Second Registry",
                "kind": "generic_registry_api",
                "base_url": "https://example.com/second.json",
            },
        )
        post_handler.client_address = ("203.0.113.20", 4242)
        post_handler.do_POST()
        post_payload = json.loads(post_handler.body.decode("utf-8"))
        self.assertEqual(403, post_handler.status_code)
        self.assertEqual("forbidden", post_payload["error"])


if __name__ == "__main__":
    unittest.main()
