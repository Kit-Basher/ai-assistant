from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.packs.external_ingestion import ExternalPackIngestor
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
    def __init__(self, runtime_obj: AgentRuntime, path: str) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.status_code = 0
        self.content_type = ""
        self.body = b""

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
        return {}


class TestPackCompare(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.store = PackStore(str(self.root / "packs.db"))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _record_local_pack(self, relative_dir: str, files: dict[str, bytes]) -> dict[str, object]:
        source = self.root / relative_dir
        source.mkdir(parents=True, exist_ok=True)
        for rel_path, data in files.items():
            target = source / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        ingestor = ExternalPackIngestor(str(self.root / "storage"))
        result, review = ingestor.ingest_from_path(str(source))
        return self.store.record_external_pack(
            canonical_pack=result.pack.to_dict(),
            classification=result.classification,
            status=result.status,
            risk_report=result.risk_report.to_dict(),
            review_envelope=review.to_dict(),
            quarantine_path=result.quarantine_path,
            normalized_path=result.normalized_path,
        )

    def _record_remote_pack(self, source_url: str, files: dict[str, bytes], *, ref: str = "main") -> dict[str, object]:
        archive = _zip_bytes(files)
        fetcher = RemotePackFetcher(
            str(self.root / "storage"),
            opener=_FakeOpener({source_url: _FakeResponse(archive, url=source_url)}),
        )
        ingestor = ExternalPackIngestor(str(self.root / "storage"), remote_fetcher=fetcher)
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

    def test_instruction_only_change_reports_instruction_diff(self) -> None:
        previous = self._record_local_pack(
            "instruction-v1",
            {"SKILL.md": b"# Skill\n\nUse the first instructions.\n"},
        )
        current = self._record_local_pack(
            "instruction-v2",
            {"SKILL.md": b"# Skill\n\nUse the changed instructions.\n"},
        )

        diff = self.store.get_or_build_external_pack_diff(str(previous["canonical_id"]), str(current["canonical_id"]))
        assert diff is not None
        self.assertTrue(diff["change_summary"]["instructions_changed"])
        self.assertFalse(diff["change_summary"]["assets_changed"])
        self.assertIn("The instructions changed.", diff["summary"])

    def test_assets_only_change_reports_asset_diff(self) -> None:
        previous = self._record_local_pack(
            "asset-v1",
            {
                "SKILL.md": b"# Skill\n\nSame instructions.\n",
                "assets/icon.png": b"PNGDATA1",
            },
        )
        current = self._record_local_pack(
            "asset-v2",
            {
                "SKILL.md": b"# Skill\n\nSame instructions.\n",
                "assets/icon.png": b"PNGDATA2",
            },
        )

        diff = self.store.get_or_build_external_pack_diff(str(previous["canonical_id"]), str(current["canonical_id"]))
        assert diff is not None
        self.assertTrue(diff["change_summary"]["assets_changed"])
        self.assertFalse(diff["change_summary"]["instructions_changed"])
        self.assertIn("Only images or other static assets changed.", diff["summary"])

    def test_executable_file_added_is_flagged(self) -> None:
        source_url = "https://github.com/example/exec-skill/archive/main.zip"
        previous = self._record_remote_pack(
            source_url,
            {"repo-main/SKILL.md": b"# Skill\n\nUse the docs only.\n"},
        )
        current = self._record_remote_pack(
            source_url,
            {
                "repo-main/SKILL.md": b"# Skill\n\nUse the docs only.\n",
                "repo-main/scripts/install.sh": b"#!/bin/sh\necho nope\n",
            },
        )

        diff = self.store.get_or_build_external_pack_diff(str(previous["canonical_id"]), str(current["canonical_id"]))
        assert diff is not None
        self.assertIn("executable_content_added", diff["flags"])
        self.assertTrue(diff["change_summary"]["executable_content_added"])
        self.assertIn("risk increased", diff["summary"])

    def test_install_instructions_added_are_flagged(self) -> None:
        previous = self._record_local_pack(
            "install-v1",
            {"SKILL.md": b"# Skill\n\nRead the notes only.\n"},
        )
        current = self._record_local_pack(
            "install-v2",
            {"SKILL.md": b"# Skill\n\nRun `pip install coolpkg` before use.\n"},
        )

        diff = self.store.get_or_build_external_pack_diff(str(previous["canonical_id"]), str(current["canonical_id"]))
        assert diff is not None
        self.assertIn("install_instructions_added", diff["flags"])
        self.assertIn("higher risk", diff["summary"])

    def test_normalized_content_unchanged_when_only_source_changes(self) -> None:
        first = self._record_remote_pack(
            "https://github.com/example/skill-one/archive/main.zip",
            {"repo-main/SKILL.md": b"# Skill\n\nShared content.\n"},
        )
        second = self._record_remote_pack(
            "https://github.com/another-owner/skill-two/archive/main.zip",
            {"repo-main/SKILL.md": b"# Skill\n\nShared content.\n"},
        )

        self.assertEqual(first["canonical_id"], second["canonical_id"])
        diff = self.store.get_or_build_external_pack_diff(str(first["canonical_id"]), str(second["canonical_id"]))
        assert diff is not None
        self.assertIn("normalized_content_unchanged", diff["flags"])
        self.assertIn("raw_source_only_changed", diff["flags"])
        self.assertIn("normalized content is unchanged", diff["summary"].lower())

    def test_history_chain_and_compare_endpoints_return_stable_structured_output(self) -> None:
        remote_url = "https://github.com/example/repo/archive/main.zip"
        previous = self._record_remote_pack(
            remote_url,
            {
                "repo-main/SKILL.md": b"# Skill\n\nVersion one.\n",
                "repo-main/references/guide.md": b"# Guide\n\nOriginal guide.\n",
            },
        )
        current = self._record_remote_pack(
            remote_url,
            {
                "repo-main/SKILL.md": b"# Skill\n\nVersion two with `pip install thing`.\n",
                "repo-main/references/guide.md": b"# Guide\n\nChanged guide.\n",
            },
        )

        registry_path = os.path.join(self.tmpdir.name, "registry.json")
        db_path = os.path.join(self.tmpdir.name, "agent.db")
        skills_path = os.path.join(self.tmpdir.name, "skills")
        os.makedirs(skills_path, exist_ok=True)
        env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        try:
            runtime = AgentRuntime(_config(registry_path, db_path, skills_path))
            runtime.pack_store = self.store

            detail_handler = _HandlerForTest(runtime, f"/packs/{current['canonical_id']}")
            detail_handler.do_GET()
            detail_payload = json.loads(detail_handler.body.decode("utf-8"))
            self.assertEqual(200, detail_handler.status_code)
            self.assertEqual(current["canonical_id"], detail_payload["pack"]["canonical_id"])

            history_handler = _HandlerForTest(runtime, f"/packs/{current['canonical_id']}/history")
            history_handler.do_GET()
            history_payload = json.loads(history_handler.body.decode("utf-8"))
            self.assertEqual(200, history_handler.status_code)
            self.assertEqual(2, history_payload["version_count"])

            compare_path = f"/packs/compare?from={previous['canonical_id']}&to={current['canonical_id']}"
            compare_handler_one = _HandlerForTest(runtime, compare_path)
            compare_handler_one.do_GET()
            compare_payload_one = json.loads(compare_handler_one.body.decode("utf-8"))
            self.assertEqual(200, compare_handler_one.status_code)
            self.assertIn("install_instructions_added", compare_payload_one["compare"]["flags"])
            self.assertIn("higher risk", compare_payload_one["compare"]["summary"])
            self.assertIn("previous", compare_payload_one["compare"])
            self.assertIn("current", compare_payload_one["compare"])
            self.assertTrue(compare_payload_one["compare"]["entries"])

            compare_handler_two = _HandlerForTest(runtime, compare_path)
            compare_handler_two.do_GET()
            compare_payload_two = json.loads(compare_handler_two.body.decode("utf-8"))
            self.assertEqual(compare_payload_one["compare"], compare_payload_two["compare"])
        finally:
            os.environ.clear()
            os.environ.update(env_backup)


if __name__ == "__main__":
    unittest.main()
