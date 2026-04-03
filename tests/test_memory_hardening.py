from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.memory_runtime import MemoryRuntime
from agent.working_memory import WorkingMemoryState, append_turn, default_budget, manage_working_memory
from agent.memory_v2.storage import SQLiteMemoryStore
from agent.semantic_memory.storage import SQLiteSemanticStore
from agent.semantic_memory.types import SemanticSourceKind
from memory.db import MemoryDB


def _config(registry_path: str, db_path: str, **overrides: object) -> Config:
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


def _schema_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))


class _GetHandler(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, *, loopback: bool = True) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {"Content-Length": "0"}
        self.status_code = 0
        self.response_payload: dict[str, object] = {}
        self._loopback = loopback

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

    def _request_is_loopback(self) -> bool:  # type: ignore[override]
        return self._loopback


class _PostHandler(APIServerHandler):
    def __init__(
        self,
        runtime_obj: AgentRuntime,
        path: str,
        payload: dict[str, object],
        *,
        loopback: bool = True,
    ) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {"Content-Length": "0"}
        self.status_code = 0
        self.response_payload: dict[str, object] = {}
        self._payload = dict(payload)
        self._loopback = loopback

    def _read_json(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

    def _request_is_loopback(self) -> bool:  # type: ignore[override]
        return self._loopback


class TestMemoryHardening(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _init_memory_db(self) -> MemoryDB:
        db = MemoryDB(self.db_path)
        db.init_schema(_schema_path())
        return db

    def test_memory_runtime_inspect_reports_corrupt_json_and_reset_clears_keys(self) -> None:
        db = self._init_memory_db()
        try:
            db.set_user_pref("memory_runtime:user1:thread_state", "{bad json")
            db.set_user_pref("memory_runtime:user1:pending_items", "{\"oops\":true}")
            db.set_user_pref("memory_runtime:user1:last_meaningful_user_request", "hello")
            runtime = MemoryRuntime(db)

            inspect = runtime.inspect_user_state("user1")
            self.assertFalse(bool(inspect.get("healthy")))
            corrupt_entries = inspect.get("corrupt_entries") if isinstance(inspect.get("corrupt_entries"), list) else []
            self.assertEqual(2, len(corrupt_entries))

            reset = runtime.reset_all_state()
            self.assertEqual(3, reset.get("deleted_count"))
            self.assertEqual([], runtime.inspect_all_state().get("users"))
        finally:
            db.close()

    def test_memory_status_reports_disabled_components_and_fresh_state(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        handler = _GetHandler(runtime, "/memory/status")
        handler.do_GET()

        self.assertEqual(200, handler.status_code)
        payload = handler.response_payload
        self.assertTrue(bool(payload.get("ok")))
        continuity = payload.get("continuity") if isinstance(payload.get("continuity"), dict) else {}
        self.assertTrue(bool(continuity.get("enabled")))
        self.assertTrue(bool(continuity.get("healthy")))
        self.assertEqual(0, continuity.get("user_count"))
        memory_v2 = payload.get("memory_v2") if isinstance(payload.get("memory_v2"), dict) else {}
        self.assertFalse(bool(memory_v2.get("enabled")))
        semantic = payload.get("semantic") if isinstance(payload.get("semantic"), dict) else {}
        self.assertFalse(bool(semantic.get("enabled")))

    def test_memory_status_includes_working_memory_summary(self) -> None:
        db = self._init_memory_db()
        try:
            runtime_memory = MemoryRuntime(db)
            state = WorkingMemoryState()
            append_turn(state, role="user", text="Please keep replies concise.")
            append_turn(state, role="assistant", text="I will keep replies concise.")
            runtime_memory.save_working_memory_state("user1", state)
        finally:
            db.close()

        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        ok, payload = runtime.memory_status()
        self.assertTrue(ok)
        working_memory = payload.get("working_memory") if isinstance(payload.get("working_memory"), dict) else {}
        self.assertEqual(2, working_memory.get("hot_turn_count"))
        self.assertEqual(1, working_memory.get("tracked_user_count"))

    def test_memory_status_includes_working_memory_debug_metadata(self) -> None:
        db = self._init_memory_db()
        try:
            runtime_memory = MemoryRuntime(db)
            state = WorkingMemoryState()
            for index in range(8):
                role = "assistant" if index % 2 else "user"
                append_turn(
                    state,
                    role=role,  # type: ignore[arg-type]
                    text=(f"Working memory debug turn {index}. " + "token " * 160).strip(),
                )
            manage_working_memory(
                state,
                budget=default_budget(4096),
                user_id="user1",
                thread_id="thread-a",
            )
            runtime_memory.save_working_memory_state("user1", state)
        finally:
            db.close()

        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        ok, payload = runtime.memory_status()
        self.assertTrue(ok)
        working_memory = payload.get("working_memory") if isinstance(payload.get("working_memory"), dict) else {}
        debug = working_memory.get("debug") if isinstance(working_memory.get("debug"), dict) else {}
        self.assertEqual("summarized_raw_chunk", working_memory.get("last_compaction_action"))
        self.assertEqual("summarized_raw_chunk", debug.get("last_action"))
        self.assertEqual("soft_threshold_exceeded", debug.get("reason"))
        self.assertEqual("soft", debug.get("threshold_crossed"))
        self.assertIn("counts_before", debug)
        self.assertIn("counts_after", debug)

    def test_memory_status_exposes_revision_conflict_for_working_memory(self) -> None:
        db = self._init_memory_db()
        try:
            runtime_a = MemoryRuntime(db)
            runtime_b = MemoryRuntime(db)
            state_a, issue_a = runtime_a.load_working_memory_state("user1")
            state_b, issue_b = runtime_b.load_working_memory_state("user1")
            self.assertIsNone(issue_a)
            self.assertIsNone(issue_b)
            append_turn(state_a, role="user", text="runtime a saved this")
            self.assertTrue(runtime_a.save_working_memory_state("user1", state_a))
            append_turn(state_b, role="user", text="runtime b stale write")
            self.assertFalse(runtime_b.save_working_memory_state("user1", state_b))
        finally:
            db.close()

        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        ok, payload = runtime.memory_status()
        self.assertTrue(ok)
        continuity = payload.get("continuity") if isinstance(payload.get("continuity"), dict) else {}
        users = continuity.get("users") if isinstance(continuity.get("users"), list) else []
        user_row = next(
            (
                row
                for row in users
                if isinstance(row, dict) and str(row.get("user_id") or "").strip() == "user1"
            ),
            {},
        )
        working_memory = user_row.get("working_memory") if isinstance(user_row.get("working_memory"), dict) else {}
        persistence = user_row.get("persistence") if isinstance(user_row.get("persistence"), dict) else {}
        last_conflict = persistence.get("last_conflict") if isinstance(persistence.get("last_conflict"), dict) else {}
        last_attempted_write = (
            persistence.get("last_attempted_write")
            if isinstance(persistence.get("last_attempted_write"), dict)
            else {}
        )
        last_successful_write = (
            persistence.get("last_successful_write")
            if isinstance(persistence.get("last_successful_write"), dict)
            else {}
        )
        revisions = persistence.get("current_revisions") if isinstance(persistence.get("current_revisions"), dict) else {}
        self.assertEqual(1, working_memory.get("revision"))
        self.assertEqual("revision_conflict", last_attempted_write.get("status"))
        self.assertEqual("ok", last_successful_write.get("status"))
        self.assertEqual("stale_write_conflict", last_conflict.get("reason"))
        self.assertTrue(bool(persistence.get("active_conflict")))
        self.assertEqual(1, revisions.get("working_memory_state"))

    def test_memory_status_clears_active_conflict_after_later_successful_recovery(self) -> None:
        db = self._init_memory_db()
        try:
            runtime_a = MemoryRuntime(db)
            runtime_b = MemoryRuntime(db)
            runtime_c = MemoryRuntime(db)
            state_a, _ = runtime_a.load_working_memory_state("user1")
            state_b, _ = runtime_b.load_working_memory_state("user1")
            append_turn(state_a, role="user", text="runtime a saved this first")
            self.assertTrue(runtime_a.save_working_memory_state("user1", state_a))
            append_turn(state_b, role="user", text="runtime b stale write")
            self.assertFalse(runtime_b.save_working_memory_state("user1", state_b))
            state_c, issue_c = runtime_c.load_working_memory_state("user1")
            self.assertIsNone(issue_c)
            append_turn(state_c, role="assistant", text="runtime c recovered save")
            self.assertTrue(runtime_c.save_working_memory_state("user1", state_c))
        finally:
            db.close()

        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        ok, payload = runtime.memory_status()
        self.assertTrue(ok)
        continuity = payload.get("continuity") if isinstance(payload.get("continuity"), dict) else {}
        users = continuity.get("users") if isinstance(continuity.get("users"), list) else []
        user_row = next(
            (
                row
                for row in users
                if isinstance(row, dict) and str(row.get("user_id") or "").strip() == "user1"
            ),
            {},
        )
        persistence = user_row.get("persistence") if isinstance(user_row.get("persistence"), dict) else {}
        last_attempted_write = (
            persistence.get("last_attempted_write")
            if isinstance(persistence.get("last_attempted_write"), dict)
            else {}
        )
        last_successful_write = (
            persistence.get("last_successful_write")
            if isinstance(persistence.get("last_successful_write"), dict)
            else {}
        )
        last_conflict = persistence.get("last_conflict") if isinstance(persistence.get("last_conflict"), dict) else {}
        self.assertEqual("ok", last_attempted_write.get("status"))
        self.assertEqual("ok", last_successful_write.get("status"))
        self.assertEqual("stale_write_conflict", last_conflict.get("reason"))
        self.assertFalse(bool(persistence.get("active_conflict")))
        self.assertLess(
            int(last_conflict.get("stored_revision") or 0),
            int(last_successful_write.get("stored_revision") or 0),
        )

    def test_memory_status_is_loopback_only(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        handler = _GetHandler(runtime, "/memory/status", loopback=False)
        handler.do_GET()
        self.assertEqual(403, handler.status_code)
        self.assertEqual("forbidden", handler.response_payload.get("error"))

    def test_memory_status_detects_corrupt_continuity_state(self) -> None:
        db = self._init_memory_db()
        try:
            db.set_user_pref("memory_runtime:user1:thread_state", "{bad json")
            db.set_user_pref("memory_runtime:user1:pending_items", "[]")
        finally:
            db.close()
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        ok, payload = runtime.memory_status()
        self.assertTrue(ok)
        continuity = payload.get("continuity") if isinstance(payload.get("continuity"), dict) else {}
        self.assertFalse(bool(continuity.get("healthy")))
        self.assertEqual("continuity_memory_corrupt", continuity.get("reason"))
        self.assertEqual(1, continuity.get("degraded_user_count"))

    def test_memory_reset_preview_then_confirm_clears_selected_memory_state(self) -> None:
        db = self._init_memory_db()
        try:
            db.set_user_pref("memory_runtime:user1:thread_state", "{\"thread_id\":\"thread-a\"}")
        finally:
            db.close()
        store = SQLiteMemoryStore(self.db_path)
        store.append_episodic_event(text="remember this", tags={}, source_kind="chat", source_ref="thread-a")
        store.set_state("memory_v2.bootstrap_completed", "1")
        semantic_store = SQLiteSemanticStore(self.db_path)
        semantic_store.upsert_source(
            source_id="SS-1",
            source_kind=SemanticSourceKind.DOCUMENT,
            source_ref="doc:one",
            scope="global",
            content_hash="hash-1",
            status="ready",
            created_at=1,
            updated_at=1,
        )
        semantic_store.replace_chunks(
            source_id="SS-1",
            chunks=[
                {
                    "id": "SC-1",
                    "chunk_index": 0,
                    "text": "doc chunk",
                    "chunk_hash": "chunk-hash",
                    "char_start": 0,
                    "char_end": 8,
                    "created_at": 1,
                    "updated_at": 1,
                    "metadata": {},
                }
            ],
        )
        semantic_store.upsert_vector(
            chunk_id="SC-1",
            embed_provider="fake",
            embed_model="fake-embed",
            embedding_dim=3,
            vector=(1.0, 0.0, 0.0),
            created_at=1,
            updated_at=1,
        )
        semantic_store.set_index_state(
            scope="global",
            embed_provider="fake",
            embed_model="fake-embed",
            embedding_dim=3,
            status="ready",
            source_count=1,
            chunk_count=1,
            vector_count=1,
            updated_at=1,
            details={},
        )

        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                memory_v2_enabled=True,
                semantic_memory_enabled=True,
            )
        )
        preview_handler = _PostHandler(
            runtime,
            "/memory/reset",
            {"components": ["continuity", "memory_v2", "semantic"]},
        )
        preview_handler.do_POST()
        self.assertEqual(200, preview_handler.status_code)
        self.assertTrue(bool(preview_handler.response_payload.get("requires_confirmation")))
        self.assertEqual("preview", preview_handler.response_payload.get("action"))

        status_before = runtime.memory_status()[1]
        self.assertEqual(1, ((status_before.get("continuity") or {}).get("user_count") or 0))
        self.assertEqual(1, (((status_before.get("memory_v2") or {}).get("counts") or {}).get("episodic_events_count") or 0))
        self.assertEqual(
            1,
            ((((status_before.get("semantic") or {}).get("tables") or {}).get("tables") or {}).get("semantic_sources") or {}).get("row_count"),
        )

        confirm_handler = _PostHandler(
            runtime,
            "/memory/reset",
            {"components": ["continuity", "memory_v2", "semantic"], "confirm": True},
        )
        confirm_handler.do_POST()
        self.assertEqual(200, confirm_handler.status_code)
        self.assertEqual("reset", confirm_handler.response_payload.get("action"))
        deleted = confirm_handler.response_payload.get("deleted") if isinstance(confirm_handler.response_payload.get("deleted"), dict) else {}
        self.assertEqual(1, ((deleted.get("continuity") or {}).get("user_pref_keys_deleted") or 0))
        self.assertEqual(1, ((deleted.get("memory_v2") or {}).get("memory_events_deleted") or 0))
        self.assertEqual(1, ((deleted.get("semantic") or {}).get("semantic_sources_deleted") or 0))

        status_after = runtime.memory_status()[1]
        self.assertEqual(0, ((status_after.get("continuity") or {}).get("user_count") or 0))
        self.assertEqual(0, (((status_after.get("memory_v2") or {}).get("counts") or {}).get("episodic_events_count") or 0))
        self.assertEqual(
            0,
            ((((status_after.get("semantic") or {}).get("tables") or {}).get("tables") or {}).get("semantic_sources") or {}).get("row_count"),
        )

    def test_memory_reset_rejects_invalid_component_without_persisting(self) -> None:
        db = self._init_memory_db()
        try:
            db.set_user_pref("memory_runtime:user1:thread_state", "{\"thread_id\":\"thread-a\"}")
        finally:
            db.close()
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        handler = _PostHandler(runtime, "/memory/reset", {"components": ["bogus"], "confirm": True})
        handler.do_POST()
        self.assertEqual(400, handler.status_code)
        self.assertEqual("bad_request", handler.response_payload.get("error_kind"))

        continuity = runtime.memory_status()[1].get("continuity")
        self.assertEqual(1, (continuity or {}).get("user_count"))

    def test_memory_v2_selection_failure_degrades_clearly_without_breaking_chat(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, memory_v2_enabled=True))
        self.assertIsNotNone(runtime._memory_v2_store)

        class _FakeOrchestrator:
            def handle_message(self, text: str, *, user_id: str, chat_context: dict[str, object] | None = None):
                _ = text
                _ = user_id
                _ = chat_context
                from agent.orchestrator import OrchestratorResponse

                return OrchestratorResponse(
                    "ok",
                    {
                        "route": "generic_chat",
                        "used_runtime_state": True,
                        "used_llm": True,
                        "used_memory": False,
                        "used_tools": [],
                        "ok": True,
                        "provider": "test",
                        "model": "test:model",
                        "fallback_used": False,
                        "attempts": [],
                        "duration_ms": 1,
                    },
                )

        with patch.object(runtime, "orchestrator", return_value=_FakeOrchestrator()), patch(
            "agent.api_server.select_memory",
            side_effect=RuntimeError("boom"),
        ):
            handler = _PostHandler(
                runtime,
                "/chat",
                {"messages": [{"role": "user", "content": "hello"}]},
            )
            handler.do_POST()

        self.assertEqual(200, handler.status_code)
        self.assertTrue(bool(handler.response_payload.get("ok")))
        envelope = handler.response_payload.get("envelope") if isinstance(handler.response_payload.get("envelope"), dict) else {}
        memory = envelope.get("memory") if isinstance(envelope.get("memory"), dict) else {}
        debug = memory.get("debug") if isinstance(memory.get("debug"), dict) else {}
        memory_error = debug.get("memory_v2_error") if isinstance(debug.get("memory_v2_error"), dict) else {}
        self.assertEqual("RuntimeError", memory_error.get("error"))
        status_payload = runtime.memory_status()[1]
        memory_v2 = status_payload.get("memory_v2") if isinstance(status_payload.get("memory_v2"), dict) else {}
        last_error = memory_v2.get("last_error") if isinstance(memory_v2.get("last_error"), dict) else {}
        self.assertEqual("select", last_error.get("operation"))

    def test_memory_migration_path_initializes_optional_tables_cleanly(self) -> None:
        db = self._init_memory_db()
        db.close()

        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                memory_v2_enabled=True,
            )
        )
        ok, payload = runtime.memory_status()
        self.assertTrue(ok)
        memory_v2 = payload.get("memory_v2") if isinstance(payload.get("memory_v2"), dict) else {}
        self.assertTrue(bool(memory_v2.get("available")))
        tables = ((memory_v2.get("tables") or {}).get("tables") or {}) if isinstance(memory_v2.get("tables"), dict) else {}
        self.assertTrue(bool((tables.get("memory_items") or {}).get("exists")))
        self.assertGreaterEqual(((memory_v2.get("counts") or {}).get("memory_items_count") or 0), 0)


if __name__ == "__main__":
    unittest.main()
