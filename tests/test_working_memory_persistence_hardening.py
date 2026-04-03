from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.memory_runtime import MemoryRuntime
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
        memory_v2_enabled=False,
        semantic_memory_enabled=False,
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _schema_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))


def _large_text(prefix: str, *, extra: str = "token", count: int = 160) -> str:
    return f"{prefix} " + " ".join([extra] * count)


class TestWorkingMemoryPersistenceHardening(unittest.TestCase):
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

    def _working_key(self, user_id: str) -> str:
        db = self._init_memory_db()
        try:
            runtime = MemoryRuntime(db)
            return runtime._working_memory_key(user_id)  # noqa: SLF001
        finally:
            db.close()

    def _write_working_memory_blob(self, user_id: str, payload: Any) -> None:
        db = self._init_memory_db()
        try:
            key = MemoryRuntime(db)._working_memory_key(user_id)  # noqa: SLF001
            raw = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=True, sort_keys=True)
            db.set_user_pref(key, raw)
        finally:
            db.close()

    def _read_working_memory_blob(self, user_id: str) -> str | None:
        db = self._init_memory_db()
        try:
            key = MemoryRuntime(db)._working_memory_key(user_id)  # noqa: SLF001
            raw = db.get_user_pref(key)
            return str(raw) if raw is not None else None
        finally:
            db.close()

    def _stubbed_chat(self, runtime: AgentRuntime, *, user_id: str, thread_id: str, text: str, min_context_tokens: int = 4096) -> tuple[bool, dict[str, Any]]:
        with patch(
            "agent.orchestrator.route_inference",
            return_value={
                "ok": True,
                "text": "Acknowledged. Continuing.",
                "provider": "ollama",
                "model": "ollama:qwen3.5:4b",
                "task_type": "chat",
                "selection_reason": "healthy+approved+local_first+task=chat",
                "fallback_used": False,
                "error_kind": None,
                "next_action": None,
                "trace_id": "wm-persistence-hardening",
                "data": {"selection": {"selected_model": "ollama:qwen3.5:4b", "provider": "ollama", "reason": "healthy"}},
            },
        ):
            return runtime.chat(
                {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "messages": [{"role": "user", "content": text}],
                    "min_context_tokens": min_context_tokens,
                }
            )

    def _failure_context(
        self,
        *,
        runtime: AgentRuntime,
        user_id: str,
        payload: Any,
        extra: str,
    ) -> str:
        status = runtime.memory_status()[1]
        continuity = status.get("continuity") if isinstance(status.get("continuity"), dict) else {}
        users = continuity.get("users") if isinstance(continuity.get("users"), list) else []
        user_row = next(
            (
                row
                for row in users
                if isinstance(row, dict) and str(row.get("user_id") or "").strip() == user_id
            ),
            {},
        )
        working_memory = user_row.get("working_memory") if isinstance(user_row.get("working_memory"), dict) else {}
        return (
            f"{extra}\n"
            f"payload={json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) if not isinstance(payload, str) else payload}\n"
            f"continuity={json.dumps(continuity, ensure_ascii=True, sort_keys=True, indent=2)}\n"
            f"user_working_memory={json.dumps(working_memory, ensure_ascii=True, sort_keys=True, indent=2)}"
        )

    def test_legacy_working_memory_shape_restores_and_compacts(self) -> None:
        user_id = "legacy-user"
        thread_id = "legacy-user:thread"
        legacy_payload = {
            "hot_turns": [
                {
                    "turn_id": f"legacy-turn-{index}",
                    "role": "assistant" if index % 2 else "user",
                    "text": _large_text(f"Legacy replay exchange {index}"),
                    "created_at": f"2026-04-01T00:00:{index:02d}+00:00",
                }
                for index in range(6)
            ],
            "warm_summaries": [
                {
                    "block_id": "legacy-summary-1",
                    "source_turn_ids": ["legacy-turn-0", "legacy-turn-1", "legacy-turn-2"],
                    "start_turn_id": "legacy-turn-0",
                    "end_turn_id": "legacy-turn-2",
                    "compression_level": 1,
                    "topic": "Legacy working memory",
                    "facts": ["Semantic memory is the durable store."],
                    "decisions": ["Working-memory summaries stay structured JSON."],
                    "open_threads": ["Replay validation still unfinished."],
                    "user_preferences": ["Prefer concise answers."],
                    "artifacts": ["PROJECT_STATUS.md"],
                    "tool_results": [],
                    "created_at": "2026-04-01T00:01:00+00:00",
                }
            ],
            "last_compaction_at": "2026-04-01T00:02:00+00:00",
            "last_compaction_action": "summarized_raw_chunk",
        }
        db = self._init_memory_db()
        try:
            runtime_memory = MemoryRuntime(db)
            runtime_memory.set_thread_state(user_id, thread_id=thread_id, current_topic="legacy replay")
            db.set_user_pref(runtime_memory._working_memory_key(user_id), json.dumps(legacy_payload, ensure_ascii=True))  # noqa: SLF001
        finally:
            db.close()

        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        restored_state, issue = runtime.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(issue)
        self.assertEqual([], restored_state.semantic_ingest_hashes)
        self.assertEqual({}, restored_state.last_compaction_debug)
        self.assertEqual("raw", restored_state.warm_summaries[0].derived_from)
        self.assertEqual(1, restored_state.warm_summaries[0].generation_count)
        self.assertEqual([], restored_state.warm_summaries[0].child_block_ids)
        self.assertEqual([], restored_state.warm_summaries[0].child_compression_levels)

        ok, body = self._stubbed_chat(
            runtime,
            user_id=user_id,
            thread_id=thread_id,
            text="Continue the legacy replay validation work.",
            min_context_tokens=3072,
        )
        self.assertTrue(ok, msg=self._failure_context(runtime=runtime, user_id=user_id, payload=legacy_payload, extra=str(body)))
        self.assertTrue(bool(body.get("ok")))

        status_ok, status = runtime.memory_status()
        self.assertTrue(status_ok)
        continuity = status.get("continuity") if isinstance(status.get("continuity"), dict) else {}
        self.assertTrue(bool(continuity.get("healthy")))
        state_after, issue_after = runtime.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(issue_after)
        self.assertGreater(len(state_after.warm_summaries) + len(state_after.cold_state_blocks), 0)
        self.assertTrue(bool(state_after.last_compaction_debug))

    def test_partial_working_memory_shape_restores_with_defaults(self) -> None:
        user_id = "partial-user"
        thread_id = "partial-user:thread"
        partial_payload = {
            "hot_turns": [
                {
                    "turn_id": "partial-turn-1",
                    "role": "user",
                    "text": "I prefer concise answers.",
                    "created_at": "2026-04-01T00:00:00+00:00",
                },
                {
                    "turn_id": "partial-turn-2",
                    "role": "assistant",
                    "text": "I will keep them concise.",
                    "created_at": "2026-04-01T00:00:05+00:00",
                },
            ],
        }
        db = self._init_memory_db()
        try:
            runtime_memory = MemoryRuntime(db)
            runtime_memory.set_thread_state(user_id, thread_id=thread_id, current_topic="partial restore")
            db.set_user_pref(runtime_memory._working_memory_key(user_id), json.dumps(partial_payload, ensure_ascii=True))  # noqa: SLF001
        finally:
            db.close()

        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        state, issue = runtime.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(issue)
        self.assertEqual(2, len(state.hot_turns))
        self.assertEqual([], state.warm_summaries)
        self.assertEqual([], state.cold_state_blocks)
        self.assertEqual(set(), state.pinned_turn_ids)
        self.assertEqual({}, state.last_compaction_debug)
        self.assertEqual([], state.semantic_ingest_hashes)

        ok, body = self._stubbed_chat(
            runtime,
            user_id=user_id,
            thread_id=thread_id,
            text="Continue normally after the partial restore.",
            min_context_tokens=4096,
        )
        self.assertTrue(ok, msg=self._failure_context(runtime=runtime, user_id=user_id, payload=partial_payload, extra=str(body)))
        self.assertTrue(bool(body.get("ok")))

        status_ok, status = runtime.memory_status()
        self.assertTrue(status_ok)
        continuity = status.get("continuity") if isinstance(status.get("continuity"), dict) else {}
        self.assertTrue(bool(continuity.get("healthy")))

    def test_malformed_working_memory_payload_fails_closed_and_preserves_other_continuity(self) -> None:
        user_id = "corrupt-user"
        thread_id = "corrupt-user:thread"
        malformed_payload = {
            "hot_turns": [
                {
                    "turn_id": "bad-turn-1",
                    "role": "user",
                    "text": "hello",
                    "token_count": "junk",
                }
            ],
            "warm_summaries": "not-a-list",
        }
        db = self._init_memory_db()
        try:
            runtime_memory = MemoryRuntime(db)
            runtime_memory.set_thread_state(user_id, thread_id=thread_id, current_topic="safe degraded path")
            runtime_memory.add_pending_item(
                user_id,
                {
                    "thread_id": thread_id,
                    "kind": "confirm",
                    "question": "Still pending",
                    "created_at": 100,
                    "expires_at": 9999999999,
                },
            )
            runtime_memory.record_user_request(user_id, "remember this request")
            db.set_user_pref(runtime_memory._working_memory_key(user_id), json.dumps(malformed_payload, ensure_ascii=True))  # noqa: SLF001
        finally:
            db.close()

        original_blob = self._read_working_memory_blob(user_id)
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        state, issue = runtime.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNotNone(issue)
        self.assertEqual("invalid_payload", issue.get("status"))
        self.assertEqual(0, len(state.hot_turns))

        status_ok, status = runtime.memory_status()
        self.assertTrue(status_ok)
        continuity = status.get("continuity") if isinstance(status.get("continuity"), dict) else {}
        self.assertFalse(bool(continuity.get("healthy")))
        self.assertEqual("continuity_memory_corrupt", continuity.get("reason"))
        users = continuity.get("users") if isinstance(continuity.get("users"), list) else []
        user_row = next(
            (
                row
                for row in users
                if isinstance(row, dict) and str(row.get("user_id") or "").strip() == user_id
            ),
            {},
        )
        working_memory = user_row.get("working_memory") if isinstance(user_row.get("working_memory"), dict) else {}
        self.assertEqual("invalid_payload", working_memory.get("status"))

        ok, body = self._stubbed_chat(
            runtime,
            user_id=user_id,
            thread_id=thread_id,
            text="Continue safely even though working-memory restore failed.",
            min_context_tokens=4096,
        )
        self.assertTrue(ok, msg=self._failure_context(runtime=runtime, user_id=user_id, payload=malformed_payload, extra=str(body)))
        self.assertTrue(bool(body.get("ok")))

        restored_runtime = runtime.orchestrator()._memory_runtime  # noqa: SLF001
        self.assertEqual("safe degraded path", restored_runtime.get_thread_state(user_id).get("current_topic"))
        self.assertEqual(1, len(restored_runtime.list_pending_items(user_id, thread_id=thread_id, include_expired=True)))
        self.assertEqual(
            "Continue safely even though working-memory restore failed.",
            runtime._ensure_memory_db().get_user_pref(f"memory_runtime:{user_id}:last_meaningful_user_request"),  # noqa: SLF001
        )
        self.assertEqual(original_blob, self._read_working_memory_blob(user_id))

    def test_recoverable_inconsistent_state_normalizes_deterministically(self) -> None:
        user_id = "inconsistent-user"
        thread_id = "inconsistent-user:thread"
        inconsistent_payload = {
            "warm_summaries": [
                {
                    "block_id": "summary-weird",
                    "start_turn_id": "turn-a",
                    "end_turn_id": "turn-b",
                    "source_turn_ids": [],
                    "token_count": -5,
                    "compression_level": 99,
                    "topic": "inconsistent",
                    "facts": ["PROJECT_STATUS.md is the handover/status file."],
                    "decisions": ["Semantic memory is the durable store."],
                    "open_threads": ["Replay validation still unfinished."],
                    "user_preferences": [],
                    "artifacts": [],
                    "tool_results": [],
                    "created_at": "2026-04-01T00:05:00+00:00",
                    "child_block_ids": ["child-a"],
                    "child_compression_levels": [],
                    "derived_from": "mystery",
                    "generation_count": 0,
                }
            ],
            "thresholds": {"soft": 100, "hard": 120, "panic": 140},
        }
        self._write_working_memory_blob(user_id, inconsistent_payload)
        db = self._init_memory_db()
        try:
            MemoryRuntime(db).set_thread_state(user_id, thread_id=thread_id, current_topic="inconsistent restore")
        finally:
            db.close()

        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        state, issue = runtime.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(issue)
        block = state.warm_summaries[0]
        self.assertEqual(3, block.compression_level)
        self.assertEqual("summary_merge", block.derived_from)
        self.assertEqual(["turn-a", "turn-b"], block.source_turn_ids)
        self.assertGreater(block.token_count, 0)
        self.assertEqual(3, block.generation_count)

        ok, body = self._stubbed_chat(
            runtime,
            user_id=user_id,
            thread_id=thread_id,
            text="Continue after normalizing the inconsistent restore state.",
            min_context_tokens=3072,
        )
        self.assertTrue(ok, msg=self._failure_context(runtime=runtime, user_id=user_id, payload=inconsistent_payload, extra=str(body)))
        self.assertTrue(bool(body.get("ok")))
        self.assertTrue(bool(runtime.memory_status()[1].get("continuity", {}).get("healthy")))

    def test_top_level_scalar_working_memory_state_stays_fail_closed_across_repeated_chat(self) -> None:
        user_id = "scalar-user"
        thread_id = "scalar-user:thread"
        db = self._init_memory_db()
        try:
            runtime_memory = MemoryRuntime(db)
            runtime_memory.set_thread_state(user_id, thread_id=thread_id, current_topic="scalar corruption")
            db.set_user_pref(runtime_memory._working_memory_key(user_id), "42")  # noqa: SLF001
        finally:
            db.close()

        original_blob = self._read_working_memory_blob(user_id)
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        state, issue = runtime.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNotNone(issue)
        self.assertEqual("invalid_type", issue.get("status"))
        self.assertEqual(0, len(state.hot_turns))

        for prompt in (
            "Continue safely after the first corruption detection.",
            "Continue safely after the second corruption detection.",
        ):
            ok, body = self._stubbed_chat(
                runtime,
                user_id=user_id,
                thread_id=thread_id,
                text=prompt,
                min_context_tokens=4096,
            )
            self.assertTrue(ok, msg=self._failure_context(runtime=runtime, user_id=user_id, payload="42", extra=str(body)))
            self.assertTrue(bool(body.get("ok")))

        status = runtime.memory_status()[1]
        continuity = status.get("continuity") if isinstance(status.get("continuity"), dict) else {}
        self.assertFalse(bool(continuity.get("healthy")))
        self.assertEqual("continuity_memory_corrupt", continuity.get("reason"))
        self.assertEqual(original_blob, self._read_working_memory_blob(user_id))


if __name__ == "__main__":
    unittest.main()
