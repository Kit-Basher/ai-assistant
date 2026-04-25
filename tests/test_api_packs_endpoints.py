from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import io
import zipfile
from pathlib import Path

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
        url = getattr(request, "full_url", str(request))
        response = self.mapping.get(url)
        if response is None:
            raise RuntimeError(f"unexpected url: {url}")
        return response


def _config(registry_path: str, db_path: str, skills_path: str) -> Config:
    base = Config(
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
    return base


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


class TestAPIPacksEndpoints(unittest.TestCase):
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

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_install_and_list_portable_text_pack_endpoints(self) -> None:
        pack_dir = os.path.join(self.tmpdir.name, "pack_one")
        os.makedirs(pack_dir, exist_ok=True)
        with open(os.path.join(pack_dir, "SKILL.md"), "w", encoding="utf-8") as handle:
            handle.write(
                "---\n"
                "id: portable-pack\n"
                "name: Portable Pack\n"
                "version: 0.1.0\n"
                "description: safe text skill\n"
                "---\n"
                "# Portable Pack\n\n"
                "Use the reference notes only.\n"
            )
        os.makedirs(os.path.join(pack_dir, "references"), exist_ok=True)
        with open(os.path.join(pack_dir, "references", "guide.md"), "w", encoding="utf-8") as handle:
            handle.write("# Guide\n\nHelpful notes.\n")

        install_handler = _HandlerForTest(
            self.runtime,
            "/packs/install",
            {"source": pack_dir},
        )
        install_handler.do_POST()
        install_payload = json.loads(install_handler.body.decode("utf-8"))
        self.assertEqual(200, install_handler.status_code)
        self.assertTrue(install_payload["ok"])
        self.assertEqual(install_payload["pack"]["canonical_id"], install_payload["pack"]["pack_id"])
        self.assertEqual(
            "portable-pack",
            install_payload["pack"]["canonical_pack"]["audit"]["declared_id"],
        )
        self.assertFalse(install_payload["requires_approval"])
        self.assertTrue(install_payload["review_required"])
        self.assertEqual("portable_text_skill", install_payload["normalization_result"]["classification"])
        self.assertEqual("normalized", install_payload["normalization_result"]["status"])
        self.assertTrue(install_payload["pack"]["non_executable"])
        self.assertTrue(install_payload["pack"]["normalized_path"])
        self.assertEqual("Portable Pack", install_payload["review"]["pack_name"])
        self.assertIn("metadata/normalization.json", install_payload["why"])

        list_handler = _HandlerForTest(self.runtime, "/packs")
        list_handler.do_GET()
        list_payload = json.loads(list_handler.body.decode("utf-8"))
        self.assertEqual(200, list_handler.status_code)
        self.assertTrue(list_payload["ok"])
        pack_ids = [row["pack_id"] for row in list_payload["packs"]]
        self.assertIn(install_payload["pack"]["pack_id"], pack_ids)

    def test_install_native_code_pack_is_blocked_but_recorded(self) -> None:
        pack_dir = os.path.join(self.tmpdir.name, "plugin_pack")
        os.makedirs(pack_dir, exist_ok=True)
        with open(os.path.join(pack_dir, "package.json"), "w", encoding="utf-8") as handle:
            json.dump({"name": "plugin-pack", "version": "1.0.0"}, handle, ensure_ascii=True)
        with open(os.path.join(pack_dir, "handler.js"), "w", encoding="utf-8") as handle:
            handle.write("export function run() { return true; }\n")

        install_handler = _HandlerForTest(self.runtime, "/packs/install", {"source": pack_dir})
        install_handler.do_POST()
        install_payload = json.loads(install_handler.body.decode("utf-8"))

        self.assertEqual(200, install_handler.status_code)
        self.assertTrue(install_payload["ok"])
        self.assertEqual("native_code_pack", install_payload["normalization_result"]["classification"])
        self.assertEqual("blocked", install_payload["normalization_result"]["status"])
        self.assertIn(
            "native_code_pack_requires_execution",
            install_payload["normalization_result"]["blocked_reasons"],
        )
        self.assertTrue(install_payload["review_required"])
        self.assertTrue(install_payload["pack"]["non_executable"])
        self.assertIn("blocked", str(install_payload["message"] or "").lower())
        self.assertIn("current safe import policy", str(install_payload["why"] or "").lower())
        self.assertIn("metadata/normalization.json", install_payload["why"])
        self.assertIn("quarantined", str(install_payload["message"] or "").lower())
        self.assertTrue(str(install_payload["next_action"] or "").strip())

    def test_remote_install_persists_and_returns_provenance(self) -> None:
        archive = _zip_bytes(
            {
                "repo-main/SKILL.md": b"# Remote Skill\n\nUse the repository notes.\n",
                "repo-main/references/guide.md": b"# Guide\n\nReference text.\n",
            }
        )
        remote_url = "https://github.com/example/repo/archive/main.zip"
        fake_fetcher = RemotePackFetcher(
            self.runtime.pack_store.external_storage_root(),
            opener=_FakeOpener({remote_url: _FakeResponse(archive, url=remote_url)}),
        )
        with mock.patch("agent.packs.external_ingestion.RemotePackFetcher", return_value=fake_fetcher):
            install_handler = _HandlerForTest(
                self.runtime,
                "/packs/install",
                {"source": remote_url, "source_kind": "github_archive", "ref": "main"},
            )
            install_handler.do_POST()
            install_payload = json.loads(install_handler.body.decode("utf-8"))

        self.assertEqual(200, install_handler.status_code)
        self.assertTrue(install_payload["ok"])
        self.assertEqual("portable_text_skill", install_payload["normalization_result"]["classification"])
        self.assertEqual("normalized", install_payload["normalization_result"]["status"])
        self.assertEqual("github_archive", install_payload["pack"]["source"]["origin"])
        self.assertEqual(remote_url, install_payload["pack"]["source"]["url"])
        self.assertEqual("main", install_payload["pack"]["source"]["ref"])
        self.assertTrue(install_payload["pack"]["source"]["archive_sha256"])
        self.assertTrue(install_payload["pack"]["quarantine_path"])

    def test_remote_install_blocks_generic_urls_without_trusted_source(self) -> None:
        with mock.patch("agent.api_server.ExternalPackIngestor", side_effect=AssertionError("ingestor should not be constructed")):
            install_handler = _HandlerForTest(
                self.runtime,
                "/packs/install",
                {"source": "https://example.com/skill-pack.zip", "source_kind": "generic_archive_url"},
            )
            install_handler.do_POST()
            install_payload = json.loads(install_handler.body.decode("utf-8"))

        self.assertEqual(400, install_handler.status_code)
        self.assertFalse(install_payload["ok"])
        self.assertEqual("source_trust_required", install_payload["error"])
        self.assertIn("trusted source", str(install_payload["message"] or "").lower())
        self.assertIn("/pack_sources", str(install_payload["next_question"] or ""))

    def test_pack_state_endpoint_reports_installed_available_and_blocked_packs(self) -> None:
        def _seed_pack(*, pack_id: str, name: str, status: str, content_hash: str, source_key: str, capabilities: list[str], enabled: bool = False) -> None:
            normalized_path = Path(self.tmpdir.name) / f"{pack_id}.normalized"
            if status != "blocked":
                normalized_path.mkdir(parents=True, exist_ok=True)
            self.runtime.pack_store.record_external_pack(
                canonical_pack={
                    "id": pack_id,
                    "name": name,
                    "version": "1.0.0",
                    "type": "skill",
                    "source": {
                        "origin": "local_registry",
                        "display_name": "Local Registry",
                        "kind": "local_catalog",
                        "url": "file:///tmp/local-registry",
                    },
                    "pack_identity": {
                        "canonical_id": pack_id,
                        "content_hash": content_hash,
                        "source_key": source_key,
                    },
                    "capabilities": {
                        "summary": ", ".join(capabilities),
                        "declared": capabilities,
                        "inferred": [],
                    },
                    "trust": {"level": "review_required", "flags": []},
                },
                classification="portable_text_skill" if status != "blocked" else "unknown_pack",
                status=status,
                risk_report={"level": "low", "score": 0.1, "flags": [], "blocked_reason": "missing GPU acceleration" if status == "blocked" else None},
                review_envelope={
                    "pack_name": name,
                    "review_required": status != "normalized",
                    "safe_options": ["Review it", "Ignore it"],
                    "summary": f"{name} review summary",
                },
                quarantine_path=None,
                normalized_path=str(normalized_path) if status != "blocked" else None,
            )
            if enabled:
                self.runtime.pack_store.set_enabled(pack_id, True)

        _seed_pack(
            pack_id="pack.voice.local_fast",
            name="Local Voice",
            status="normalized",
            content_hash="hash-voice",
            source_key="voice",
            capabilities=["voice_output"],
            enabled=True,
        )
        _seed_pack(
            pack_id="pack.avatar.basic",
            name="Basic Avatar",
            status="partial_safe_import",
            content_hash="hash-avatar",
            source_key="avatar",
            capabilities=["avatar_visual"],
        )
        _seed_pack(
            pack_id="pack.camera.robot",
            name="Robot Camera",
            status="blocked",
            content_hash="hash-camera",
            source_key="camera",
            capabilities=["camera_feed"],
        )

        class _FakePackDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                return [
                    {"id": "local", "name": "Local Registry", "kind": "local_catalog", "enabled": True, "allowed_by_policy": True}
                ]

            def list_packs(self, source_id: str) -> dict[str, object]:
                assert source_id == "local"
                return {
                    "source": {"id": "local", "name": "Local Registry", "kind": "local_catalog", "enabled": True},
                    "policy": {"enabled": True, "allowlisted": True},
                    "packs": [
                        {
                            "remote_id": "robot-camera",
                            "name": "Robot Camera",
                            "summary": "Robot camera feed integration.",
                            "artifact_type_hint": "native_code_pack",
                            "installable_by_current_policy": False,
                            "install_block_reason_if_known": "missing GPU acceleration",
                            "source_kind_hint": "github_archive",
                            "latest_ref_hint": "main",
                            "source_url": "https://example.invalid/robot-camera",
                            "tags": ["camera_feed"],
                            "badges": ["blocked"],
                        },
                        {
                            "remote_id": "text-speech",
                            "name": "Local Voice",
                            "summary": "Local speech output for this machine.",
                            "artifact_type_hint": "portable_text_skill",
                            "installable_by_current_policy": True,
                            "source_kind_hint": "github_archive",
                            "latest_ref_hint": "main",
                            "source_url": "https://example.invalid/local-voice",
                            "tags": ["voice_output"],
                            "badges": ["ready"],
                        },
                    ],
                    "from_cache": True,
                    "stale": False,
                    "fetched_at": "2026-04-09T00:00:00Z",
                }

        self.runtime._pack_registry_discovery = lambda: _FakePackDiscovery()  # type: ignore[assignment]

        handler = _HandlerForTest(self.runtime, "/packs/state")
        handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))

        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["read_only"])
        self.assertEqual(
            {
                "total": 5,
                "installed": 3,
                "enabled": 0,
                "healthy": 1,
                "machine_usable": 1,
                "task_unconfirmed": 1,
                "usable": 0,
                "blocked": 2,
                "available": 1,
            },
            payload["summary"],
        )
        self.assertEqual("Ready", payload["state_label"])
        self.assertIsNone(payload["recovery"])
        self.assertEqual(3, len(payload["packs"]))
        self.assertEqual(2, len(payload["available_packs"]))
        installed_states = {row["state"]: row for row in payload["packs"]}
        self.assertEqual("Installed · Healthy", installed_states["installed_healthy"]["state_label"])
        self.assertEqual("Installed and healthy, but task usability is not confirmed.", installed_states["installed_healthy"]["status_note"])
        self.assertIsNone(installed_states["installed_healthy"]["enabled"])
        self.assertEqual("unknown", installed_states["installed_healthy"]["normalized_state"]["activation_state"])
        self.assertTrue(installed_states["installed_healthy"]["healthy"])
        self.assertTrue(installed_states["installed_healthy"]["machine_usable"])
        self.assertFalse(installed_states["installed_healthy"]["usable"])
        self.assertEqual("Installed · Limited", installed_states["installed_limited"]["state_label"])
        self.assertIn("compatibility is not fully confirmed", installed_states["installed_limited"]["status_note"])
        self.assertIsNone(installed_states["installed_limited"]["enabled"])
        self.assertFalse(installed_states["installed_limited"]["healthy"])
        self.assertFalse(installed_states["installed_limited"]["machine_usable"])
        self.assertFalse(installed_states["installed_limited"]["usable"])
        self.assertEqual("Installed · Blocked", installed_states["installed_blocked"]["state_label"])
        self.assertIn("blocked during import", installed_states["installed_blocked"]["status_note"])
        self.assertIsNone(installed_states["installed_blocked"]["enabled"])
        self.assertFalse(installed_states["installed_blocked"]["healthy"])
        self.assertFalse(installed_states["installed_blocked"]["machine_usable"])
        self.assertFalse(installed_states["installed_blocked"]["usable"])

        available_states = {row["state"]: row for row in payload["available_packs"]}
        self.assertIn("Available", available_states["available"]["state_label"])
        self.assertIn("Available to preview", available_states["available"]["status_note"])
        self.assertFalse(available_states["available"]["installed"])
        self.assertFalse(available_states["available"]["enabled"])
        self.assertEqual("previewable", available_states["available"]["normalized_state"]["discovery_state"])
        self.assertEqual("installable", available_states["available"]["normalized_state"]["install_state"])
        self.assertEqual("unknown", available_states["available"]["normalized_state"]["health_state"])
        self.assertEqual("unconfirmed", available_states["available"]["normalized_state"]["compatibility_state"])
        self.assertEqual("unknown", available_states["available"]["normalized_state"]["usability_state"])
        self.assertFalse(available_states["available"]["normalized_state"]["machine_usable"])
        self.assertEqual("Blocked", available_states["blocked"]["state_label"])
        self.assertIn("missing GPU acceleration", available_states["blocked"]["blocker"])
        self.assertFalse(available_states["blocked"]["usable"])
        self.assertEqual("blocked", available_states["blocked"]["normalized_state"]["compatibility_state"])
        self.assertEqual("failing", available_states["blocked"]["normalized_state"]["health_state"])
        self.assertEqual("unusable", available_states["blocked"]["normalized_state"]["usability_state"])

    def test_pack_state_endpoint_handles_empty_state_without_guessing(self) -> None:
        class _EmptyPackDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                return []

            def list_packs(self, source_id: str) -> dict[str, object]:
                raise AssertionError(f"unexpected source lookup: {source_id}")

        self.runtime._pack_registry_discovery = lambda: _EmptyPackDiscovery()  # type: ignore[assignment]

        handler = _HandlerForTest(self.runtime, "/packs/state")
        handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))

        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["read_only"])
        self.assertEqual(
            {
                "total": 0,
                "installed": 0,
                "enabled": 0,
                "healthy": 0,
                "machine_usable": 0,
                "task_unconfirmed": 0,
                "usable": 0,
                "blocked": 0,
                "available": 0,
            },
            payload["summary"],
        )
        self.assertEqual("Ready", payload["state_label"])
        self.assertIsNone(payload["recovery"])
        self.assertEqual([], payload["packs"])
        self.assertEqual([], payload["available_packs"])
        self.assertEqual([], payload["source_warnings"])

    def test_external_pack_is_usable_through_chat_and_removed_cleanly(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures" / "reference_packs" / "anthropic_clean_skill" / "source"
        source_dir = Path(self.tmpdir.name) / "clean_source"
        shutil.copytree(fixture_root, source_dir)

        install_handler = _HandlerForTest(self.runtime, "/packs/install", {"source": str(source_dir)})
        install_handler.do_POST()
        install_payload = json.loads(install_handler.body.decode("utf-8"))
        self.assertEqual(200, install_handler.status_code)
        self.assertTrue(install_payload["ok"])

        canonical_id = str(install_payload["pack"]["canonical_id"] or "").strip()
        self.assertTrue(canonical_id)
        normalized_path = Path(str(install_payload["pack"]["normalized_path"] or "").strip())
        quarantine_path = Path(str(install_payload["pack"]["quarantine_path"] or "").strip())
        self.assertTrue(normalized_path.is_dir())
        self.assertTrue(quarantine_path.exists())

        fake_llm = mock.Mock()
        fake_llm.enabled.return_value = True
        fake_llm.chat.return_value = {"ok": True, "text": "LLM reply", "provider": "ollama", "model": "llama3"}
        self.runtime.router = fake_llm
        orchestrator = self.runtime.orchestrator()
        orchestrator.llm_client = fake_llm
        with mock.patch.object(orchestrator, "_llm_chat_available", return_value=True):
            response = orchestrator.handle_message("What is Repo Helper for?", "user-a")
        self.assertIn("available as an imported pack", response.text.lower())
        self.assertIn("repo helper", response.text.lower())
        self.assertNotIn("safe imported text pack", response.text.lower())
        self.assertNotIn("example prompts:", response.text.lower())
        self.assertNotIn("inputs:", response.text.lower())
        self.assertNotIn("behavior:", response.text.lower())

        orchestrator._pack_store._conn.execute("BEGIN")
        orchestrator._pack_store.list_external_packs()

        delete_handler = _HandlerForTest(self.runtime, f"/packs/{canonical_id}")
        delete_handler.do_DELETE()
        delete_payload = json.loads(delete_handler.body.decode("utf-8"))
        self.assertEqual(200, delete_handler.status_code)
        self.assertTrue(delete_payload["ok"])

        list_handler = _HandlerForTest(self.runtime, "/packs")
        list_handler.do_GET()
        list_payload = json.loads(list_handler.body.decode("utf-8"))
        pack_ids = [row["pack_id"] for row in list_payload["packs"]]
        self.assertNotIn(canonical_id, pack_ids)
        self.assertFalse(normalized_path.exists())
        self.assertFalse(quarantine_path.exists())

        with mock.patch.object(orchestrator, "_llm_chat_available", return_value=True):
            after_removal = orchestrator.handle_message("What is Repo Helper for?", "user-a")
        self.assertIsNotNone(orchestrator._external_pack_knowledge_response("user-a", "What is Repo Helper for?"))
        self.assertIn("removed", after_removal.text.lower())
        self.assertIn("reinstall", after_removal.text.lower())
        self.assertNotIn("safe imported text pack", after_removal.text.lower())
        self.assertNotIn("example prompts:", after_removal.text.lower())
        self.assertNotIn("inputs:", after_removal.text.lower())
        self.assertNotIn("behavior:", after_removal.text.lower())
        self.assertIn("quarantined and normalized", install_payload["message"].lower())
        self.assertIn("metadata/normalization.json", install_payload["why"])

    def test_remote_install_deduplicates_identical_content_across_sources(self) -> None:
        archive = _zip_bytes(
            {
                "repo-main/SKILL.md": b"# Remote Skill\n\nUse the repository notes.\n",
                "repo-main/references/guide.md": b"# Guide\n\nReference text.\n",
            }
        )
        first_url = "https://github.com/example/repo/archive/main.zip"
        second_url = "https://github.com/another-owner/repo/archive/main.zip"
        first_fetcher = RemotePackFetcher(
            self.runtime.pack_store.external_storage_root(),
            opener=_FakeOpener(
                {
                    first_url: _FakeResponse(archive, url=first_url),
                    second_url: _FakeResponse(archive, url=second_url),
                }
            ),
        )
        with mock.patch("agent.packs.external_ingestion.RemotePackFetcher", return_value=first_fetcher):
            first_handler = _HandlerForTest(
                self.runtime,
                "/packs/install",
                {"source": first_url, "source_kind": "github_archive", "ref": "main"},
            )
            first_handler.do_POST()
            first_payload = json.loads(first_handler.body.decode("utf-8"))
            second_handler = _HandlerForTest(
                self.runtime,
                "/packs/install",
                {"source": second_url, "source_kind": "github_archive", "ref": "main"},
            )
            second_handler.do_POST()
            second_payload = json.loads(second_handler.body.decode("utf-8"))

        self.assertEqual(200, first_handler.status_code)
        self.assertEqual(200, second_handler.status_code)
        self.assertTrue(first_payload["ok"])
        self.assertTrue(second_payload["ok"])
        self.assertEqual(first_payload["pack"]["canonical_id"], second_payload["pack"]["canonical_id"])
        self.assertEqual(1, len(self.runtime.pack_store.list_external_packs()))

    def test_remote_install_returns_changed_upstream_review_when_source_mutates(self) -> None:
        remote_url = "https://github.com/example/repo/archive/main.zip"
        first_archive = _zip_bytes(
            {
                "repo-main/SKILL.md": b"# Remote Skill\n\nUse the first instructions.\n",
                "repo-main/references/original.md": b"# Original\n\nKeep this reference.\n",
            }
        )
        second_archive = _zip_bytes(
            {
                "repo-main/SKILL.md": b"# Remote Skill\n\nUse the changed instructions.\n",
                "repo-main/references/updated.md": b"# Updated\n\nNew reference.\n",
            }
        )

        first_fetcher = RemotePackFetcher(
            self.runtime.pack_store.external_storage_root(),
            opener=_FakeOpener({remote_url: _FakeResponse(first_archive, url=remote_url)}),
        )
        with mock.patch("agent.packs.external_ingestion.RemotePackFetcher", return_value=first_fetcher):
            first_handler = _HandlerForTest(
                self.runtime,
                "/packs/install",
                {"source": remote_url, "source_kind": "github_archive", "ref": "main"},
            )
            first_handler.do_POST()
            first_payload = json.loads(first_handler.body.decode("utf-8"))

        second_fetcher = RemotePackFetcher(
            self.runtime.pack_store.external_storage_root(),
            opener=_FakeOpener({remote_url: _FakeResponse(second_archive, url=remote_url)}),
        )
        with mock.patch("agent.packs.external_ingestion.RemotePackFetcher", return_value=second_fetcher):
            second_handler = _HandlerForTest(
                self.runtime,
                "/packs/install",
                {"source": remote_url, "source_kind": "github_archive", "ref": "main"},
            )
            second_handler.do_POST()
            second_payload = json.loads(second_handler.body.decode("utf-8"))

        self.assertEqual(200, first_handler.status_code)
        self.assertEqual(200, second_handler.status_code)
        self.assertNotEqual(first_payload["pack"]["canonical_id"], second_payload["pack"]["canonical_id"])
        self.assertIn("upstream_content_changed", second_payload["pack"]["risk_flags"])
        self.assertIn("changed since the last time it was seen", second_payload["review"]["summary"])
        self.assertEqual(
            first_payload["pack"]["canonical_id"],
            second_payload["review"]["previous_version"]["canonical_id"],
        )
        self.assertIn("SKILL.md", second_payload["review"]["change_summary"]["changed_instructions"])


if __name__ == "__main__":
    unittest.main()
