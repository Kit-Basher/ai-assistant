from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import io
import zipfile

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
        self.assertTrue(install_payload["pack"]["normalized_path"])
        self.assertIn("I fetched a snapshot", install_payload["review"]["summary"])
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
