from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.api_server import APIServerHandler, AgentRuntime, MAX_JSON_REQUEST_BYTES
from agent.config import Config
from agent.packs.capability_recommendation import (
    recommend_packs_for_capability,
    render_pack_capability_response,
)
from agent.packs.state_truth import normalize_installed_pack_truth
from agent.packs.store import PackStore


def _config(db_path: str, registry_path: str, **overrides: object) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path="/tmp/skills",
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
    return base.__class__(**{**base.__dict__, **overrides})


class _PostHandler(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object]) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {"Content-Length": "0"}
        self._payload = dict(payload)
        self.status_code = 0
        self.response_payload: dict[str, object] = {}

    def _read_json(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))


class _RawPostHandler(APIServerHandler):
    def __init__(
        self,
        runtime_obj: AgentRuntime,
        path: str,
        *,
        raw_body: bytes,
        content_type: str,
        content_length: int | None = None,
    ) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {
            "Content-Length": str(len(raw_body) if content_length is None else content_length),
            "Content-Type": content_type,
        }
        self.rfile = io.BytesIO(raw_body)
        self.status_code = 0
        self.response_payload: dict[str, object] = {}

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))


class _RawPutHandler(_RawPostHandler):
    def do(self) -> None:
        self.do_PUT()


class TestAdversarialRequests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.env_backup)
        self.tmpdir.cleanup()

    def _runtime(self) -> AgentRuntime:
        return AgentRuntime(_config(self.db_path, self.registry_path))

    def test_invalid_content_length_returns_structured_bad_request(self) -> None:
        runtime = self._runtime()
        handler = _RawPostHandler(
            runtime,
            "/packs/enable",
            raw_body=b"",
            content_type="application/json",
            content_length=-1,
        )
        handler.do_POST()

        self.assertEqual(400, handler.status_code)
        self.assertEqual(False, handler.response_payload.get("ok"))
        self.assertEqual("bad_request", handler.response_payload.get("error_kind"))
        self.assertIn("Content-Length", str(handler.response_payload.get("message") or ""))

    def test_invalid_json_body_returns_structured_bad_request(self) -> None:
        runtime = self._runtime()
        handler = _RawPostHandler(
            runtime,
            "/packs/enable",
            raw_body=b"{",
            content_type="application/json",
        )
        handler.do_POST()

        self.assertEqual(400, handler.status_code)
        self.assertEqual("bad_request", handler.response_payload.get("error_kind"))
        self.assertIn("valid JSON", str(handler.response_payload.get("next_question") or ""))

    def test_missing_json_content_type_returns_structured_bad_request(self) -> None:
        runtime = self._runtime()
        handler = _RawPostHandler(
            runtime,
            "/packs/enable",
            raw_body=b"{}",
            content_type="text/plain",
        )
        handler.do_POST()

        self.assertEqual(400, handler.status_code)
        self.assertEqual("bad_request", handler.response_payload.get("error_kind"))
        self.assertIn("Content-Type", str(handler.response_payload.get("next_question") or ""))

    def test_oversized_json_body_returns_structured_error(self) -> None:
        runtime = self._runtime()
        handler = _RawPostHandler(
            runtime,
            "/packs/enable",
            raw_body=b"",
            content_type="application/json",
            content_length=MAX_JSON_REQUEST_BYTES + 1,
        )
        handler.do_POST()

        self.assertEqual(413, handler.status_code)
        self.assertEqual("bad_request", handler.response_payload.get("error_kind"))
        self.assertIn("too large", str(handler.response_payload.get("message") or "").lower())

    def test_enable_rejects_non_boolean_enabled_value(self) -> None:
        runtime = self._runtime()
        handler = _PostHandler(runtime, "/packs/enable", {"pack_id": "pack.one", "enabled": "false"})
        handler.do_POST()

        self.assertEqual(400, handler.status_code)
        self.assertEqual("bad_request", handler.response_payload.get("error_kind"))
        self.assertIn("boolean", str(handler.response_payload.get("message") or "").lower())

    def test_pack_state_handles_startup_source_listing_failure(self) -> None:
        runtime = self._runtime()

        class _Discovery:
            def list_sources(self) -> list[dict[str, object]]:
                raise sqlite3.OperationalError("database is locked")

        runtime._pack_registry_discovery = lambda: _Discovery()  # type: ignore[method-assign]

        payload = runtime.packs_state()
        self.assertEqual(True, payload.get("ok"))
        self.assertTrue(isinstance(payload.get("source_warnings"), list))
        self.assertTrue(any("pack_sources_busy" in str(row.get("kind") or "") for row in payload.get("source_warnings", [])))

    def test_pack_source_listing_handles_startup_source_listing_failure(self) -> None:
        runtime = self._runtime()

        class _Discovery:
            def list_sources(self) -> list[dict[str, object]]:
                raise sqlite3.OperationalError("database is busy")

        runtime._pack_registry_discovery = lambda: _Discovery()  # type: ignore[method-assign]

        payload = runtime.list_pack_sources()
        self.assertEqual(True, payload.get("ok"))
        self.assertEqual([], payload.get("sources"))
        self.assertTrue(isinstance(payload.get("warnings"), list))
        self.assertTrue(any("pack_sources_unavailable" in str(row.get("kind") or "") for row in payload.get("warnings", [])))

    def test_missing_normalized_files_degrade_to_blocked_state(self) -> None:
        store = PackStore(self.db_path)
        missing_dir = Path(self.tmpdir.name) / "missing-normalized"
        row = store.record_external_pack(
            canonical_pack={
                "id": "pack.example",
                "name": "Example Pack",
                "version": "1.0.0",
                "type": "skill",
                "capabilities": {"declared": ["voice_output"]},
                "pack_identity": {"canonical_id": "pack.example", "content_hash": "hash"},
            },
            classification="portable_text_skill",
            status="normalized",
            risk_report={"level": "low", "score": 0.1, "flags": []},
            review_envelope={"review_required": True, "why_risk": []},
            quarantine_path=str(self.tmpdir.name),
            normalized_path=str(missing_dir),
        )
        normalized = normalize_installed_pack_truth(row)
        self.assertEqual("installed_blocked", normalized.get("state_key"))
        self.assertEqual("normalized files are missing", normalized.get("blocker"))
        self.assertEqual(False, normalized.get("task_usable"))
        self.assertEqual(False, normalized.get("machine_usable"))

    def test_missing_normalized_metadata_degrades_to_unknown_state(self) -> None:
        normalized = normalize_installed_pack_truth(
            {
                "status": "normalized",
                "enabled": True,
                "canonical_pack": {
                    "capabilities": {"declared": ["voice_output"]},
                },
            }
        )
        self.assertEqual("installed_unknown", normalized.get("state_key"))
        self.assertEqual("normalized path not recorded", normalized.get("blocker"))
        self.assertEqual("Installed · Unknown", normalized.get("state_label"))

    def test_corrupted_permissions_json_degrades_safely(self) -> None:
        store = PackStore(self.db_path)
        with store._conn:  # type: ignore[attr-defined]
            store._conn.execute(
                """
                INSERT INTO skill_packs (
                    pack_id, version, trust, manifest_path, permissions_json, permissions_hash,
                    approved_permissions_hash, enabled, installed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "pack.corrupt",
                    "1.0.0",
                    "portable",
                    None,
                    "{not json}",
                    "hash",
                    None,
                    0,
                    1,
                    1,
                ),
            )
        pack = store.get_pack("pack.corrupt")
        self.assertIsNotNone(pack)
        self.assertEqual([], (pack or {}).get("permissions", {}).get("ifaces"))

    def test_db_write_retry_eventually_succeeds(self) -> None:
        store = PackStore(self.db_path)
        attempts = {"count": 0}

        def flaky_write() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        result = store._write_with_retry(flaky_write)  # type: ignore[attr-defined]
        self.assertEqual("ok", result)
        self.assertEqual(3, attempts["count"])

    def test_recommendation_degrades_when_discovery_is_unavailable(self) -> None:
        store = PackStore(self.db_path)

        class _Discovery:
            def list_sources(self) -> list[dict[str, object]]:
                raise sqlite3.OperationalError("database is busy")

            def search(self, *_args, **_kwargs):  # pragma: no cover - not reached
                raise AssertionError("search should not be called")

        result = recommend_packs_for_capability(
            "talk to me out loud",
            pack_store=store,
            pack_registry_discovery=_Discovery(),
        )
        self.assertIsNotNone(result)
        self.assertEqual("missing", result.get("status"))
        self.assertTrue(isinstance(result.get("source_errors"), list))
        rendered = render_pack_capability_response(result)
        self.assertIn("I can keep responding in text", rendered)

    def test_enable_before_install_returns_truthful_not_found(self) -> None:
        runtime = self._runtime()
        ok, payload = runtime.packs_enable({"pack_id": "missing.pack", "enabled": True})
        self.assertFalse(ok)
        self.assertEqual("bad_request", payload.get("error_kind"))
        self.assertIn("pack not found", str(payload.get("message") or "").lower())

    def test_repeated_remove_is_safe_and_truthful(self) -> None:
        runtime = self._runtime()
        quarantine_dir = Path(self.tmpdir.name) / "quarantine-remove"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        runtime.pack_store.record_external_pack(
            canonical_pack={
                "id": "pack.remove.me",
                "name": "Remove Me",
                "version": "1.0.0",
                "type": "skill",
                "pack_identity": {"canonical_id": "pack.remove.me", "content_hash": "hash"},
            },
            classification="portable_text_skill",
            status="normalized",
            risk_report={"level": "low", "score": 0.1, "flags": []},
            review_envelope={"review_required": True, "why_risk": []},
            quarantine_path=str(quarantine_dir),
            normalized_path=str(Path(__file__).resolve()),
        )
        first_ok, first_payload = runtime.delete_external_pack("pack.remove.me")
        second_ok, second_payload = runtime.delete_external_pack("pack.remove.me")
        self.assertTrue(first_ok)
        self.assertEqual(True, first_payload.get("ok"))
        self.assertTrue(second_ok)
        self.assertEqual(True, second_payload.get("ok"))
        self.assertTrue(bool(second_payload.get("already_removed")))
        self.assertIn("already completed", str(second_payload.get("message") or "").lower())
