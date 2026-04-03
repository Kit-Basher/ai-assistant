from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.working_memory import (
    WorkingMemoryState,
    append_turn,
    build_working_memory_summary,
    default_budget,
    manage_working_memory,
    working_memory_state_to_dict,
)


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


def _large_text(prefix: str, *, extra: str = "token", count: int = 160) -> str:
    return f"{prefix} " + " ".join([extra] * count)


class TestWorkingMemoryConcurrency(unittest.TestCase):
    """Validate the current concurrency model honestly.

    Current intended behavior is effectively:
    - SQLite serializes writes
    - working-memory state is persisted as a full-record replace per user key
    - writes are revision-aware compare-and-swap operations
    - stale cross-runtime writes are rejected explicitly
    - no merge-on-write exists across runtimes
    """

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

    def _runtime(self) -> AgentRuntime:
        return AgentRuntime(_config(self.registry_path, self.db_path))

    def _state_text(self, state: WorkingMemoryState) -> str:
        parts: list[str] = []
        for turn in state.hot_turns:
            parts.append(turn.text)
        for block in [*state.warm_summaries, *state.cold_state_blocks]:
            parts.append(block.topic)
            parts.extend(block.facts)
            parts.extend(block.decisions)
            parts.extend(block.open_threads)
            parts.extend(block.user_preferences)
            parts.extend(block.artifacts)
            parts.extend(block.tool_results)
        return "\n".join(part for part in parts if str(part).strip())

    def _assert_state_sane(self, state: WorkingMemoryState, *, note: str) -> None:
        summary = build_working_memory_summary(state)
        for turn in state.hot_turns:
            self.assertTrue(bool(str(turn.turn_id or "").strip()), msg=note)
            self.assertGreaterEqual(int(turn.token_count), 0, msg=note)
            self.assertIn(turn.role, {"system", "user", "assistant", "tool"}, msg=note)
        hot_ids = {turn.turn_id for turn in state.hot_turns}
        self.assertTrue(state.pinned_turn_ids.issubset(hot_ids), msg=note)
        for block in [*state.warm_summaries, *state.cold_state_blocks]:
            self.assertTrue(bool(block.source_turn_ids), msg=note)
            self.assertIn(block.derived_from, {"raw", "summary_merge"}, msg=note)
            self.assertGreaterEqual(int(block.generation_count), 1, msg=note)
            self.assertGreaterEqual(int(block.token_count), 0, msg=note)
            self.assertLessEqual(int(block.compression_level), 3, msg=note)
            self.assertGreaterEqual(int(block.compression_level), 1, msg=note)
            self.assertTrue(all(isinstance(item, str) and item for item in block.child_block_ids), msg=note)
            self.assertTrue(all(1 <= int(level) <= 3 for level in block.child_compression_levels), msg=note)
        self.assertIsInstance(state.last_compaction_debug, dict, msg=note)
        self.assertTrue(all(isinstance(item, str) and item for item in state.semantic_ingest_hashes), msg=note)
        self.assertGreaterEqual(int(summary.get("total_tokens") or 0), 0, msg=note)

    def _make_state(
        self,
        marker: str,
        *,
        count: int = 8,
        budget_tokens: int | None = None,
        pin_first: bool = False,
        durable: bool = False,
    ) -> WorkingMemoryState:
        state = WorkingMemoryState()
        ingested: list[dict[str, Any]] = []
        for index in range(count):
            append_turn(
                state,
                role="assistant" if index % 2 else "user",
                text=_large_text(f"{marker} exchange {index}"),
                pinned=pin_first and index == 0,
            )
        if budget_tokens is not None:
            manage_working_memory(
                state,
                budget=default_budget(budget_tokens),
                user_id="concurrency-user",
                thread_id="concurrency-thread",
                durable_ingestor=ingested.append if durable else None,
            )
        return state

    def _stubbed_chat(
        self,
        runtime: AgentRuntime,
        *,
        user_id: str,
        thread_id: str,
        text: str,
        min_context_tokens: int,
    ) -> tuple[bool, dict[str, Any]]:
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
                "trace_id": "working-memory-concurrency",
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

    def _captured_chat_context(
        self,
        runtime: AgentRuntime,
        *,
        user_id: str,
        thread_id: str,
        text: str,
        min_context_tokens: int,
    ) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def _fake_route_inference(**kwargs: Any) -> dict[str, Any]:
            captured["messages"] = json.loads(json.dumps(list(kwargs.get("messages") or []), ensure_ascii=True))
            captured["memory_context_text"] = str(kwargs.get("memory_context_text") or "")
            return {
                "ok": True,
                "text": "Captured.",
                "provider": "ollama",
                "model": "ollama:qwen3.5:4b",
                "task_type": "chat",
                "selection_reason": "healthy+approved+local_first+task=chat",
                "fallback_used": False,
                "error_kind": None,
                "next_action": None,
                "trace_id": "working-memory-concurrency-capture",
                "data": {"selection": {"selected_model": "ollama:qwen3.5:4b", "provider": "ollama", "reason": "healthy"}},
            }

        with patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference):
            ok, body = runtime.chat(
                {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "messages": [{"role": "user", "content": text}],
                    "min_context_tokens": min_context_tokens,
                }
            )
        self.assertTrue(ok, msg=f"captured chat failed: {body}")
        self.assertTrue(bool(body.get("ok")), msg=f"captured chat body not ok: {body}")
        return captured

    def _failure_context(
        self,
        *,
        runtime: AgentRuntime,
        user_id: str,
        operations: list[str],
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
        return (
            f"{extra}\n"
            f"operations={json.dumps(operations, ensure_ascii=True, indent=2)}\n"
            f"continuity={json.dumps(continuity, ensure_ascii=True, sort_keys=True, indent=2)}\n"
            f"user_row={json.dumps(user_row, ensure_ascii=True, sort_keys=True, indent=2)}"
        )

    def test_stale_runtime_interleave_rejects_overwrite_and_keeps_newer_state(self) -> None:
        user_id = "concurrency-user"
        thread_id = "concurrency-thread"
        operations: list[str] = []
        runtime_a = self._runtime()
        runtime_b = self._runtime()
        runtime_c = self._runtime()

        runtime_a.orchestrator()._memory_runtime.set_thread_state(user_id, thread_id=thread_id, current_topic="stale overwrite")  # noqa: SLF001
        state_a, issue_a = runtime_a.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        state_b, issue_b = runtime_b.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(issue_a)
        self.assertIsNone(issue_b)

        state_a = self._make_state("runtime-a", count=6, budget_tokens=4096)
        self.assertTrue(runtime_a.orchestrator()._memory_runtime.save_working_memory_state(user_id, state_a))  # noqa: SLF001
        operations.append("runtime A saved compacted stale-view state")

        state_b = self._make_state("runtime-b", count=4, budget_tokens=None)
        self.assertFalse(runtime_b.orchestrator()._memory_runtime.save_working_memory_state(user_id, state_b))  # noqa: SLF001
        operations.append("runtime B stale save was rejected")

        final_state, final_issue = runtime_c.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(final_issue)
        self._assert_state_sane(
            final_state,
            note=self._failure_context(runtime=runtime_c, user_id=user_id, operations=operations, extra="final state malformed after stale overwrite"),
        )
        self.assertEqual(working_memory_state_to_dict(state_a), working_memory_state_to_dict(final_state))
        flattened = self._state_text(final_state)
        self.assertIn("runtime-a exchange", flattened)
        self.assertNotIn("runtime-b exchange", flattened)
        persistence = runtime_b.orchestrator()._memory_runtime.inspect_user_state(user_id).get("persistence")  # noqa: SLF001
        last_attempted_write = (
            persistence.get("last_attempted_write")
            if isinstance(persistence, dict) and isinstance(persistence.get("last_attempted_write"), dict)
            else {}
        )
        self.assertEqual("revision_conflict", last_attempted_write.get("status"))

    def test_compaction_interleave_with_stale_raw_write_stays_structurally_sane(self) -> None:
        user_id = "compaction-user"
        thread_id = "compaction-thread"
        operations: list[str] = []
        runtime_seed = self._runtime()
        runtime_a = self._runtime()
        runtime_b = self._runtime()
        runtime_status = self._runtime()

        runtime_seed.orchestrator()._memory_runtime.set_thread_state(user_id, thread_id=thread_id, current_topic="compaction interleave")  # noqa: SLF001
        seed_state = self._make_state("seed-base", count=10, budget_tokens=None)
        runtime_seed.orchestrator()._memory_runtime.save_working_memory_state(user_id, seed_state)  # noqa: SLF001

        state_a, issue_a = runtime_a.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        state_b, issue_b = runtime_b.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(issue_a)
        self.assertIsNone(issue_b)

        manage_working_memory(
            state_a,
            budget=default_budget(4096),
            user_id=user_id,
            thread_id=thread_id,
            durable_ingestor=[].append,  # type: ignore[arg-type]
        )
        self.assertTrue(runtime_a.orchestrator()._memory_runtime.save_working_memory_state(user_id, state_a))  # noqa: SLF001
        operations.append("runtime A compacted and saved summary-heavy state")

        append_turn(state_b, role="user", text=_large_text("stale raw tail from runtime B"))
        self.assertFalse(runtime_b.orchestrator()._memory_runtime.save_working_memory_state(user_id, state_b))  # noqa: SLF001
        operations.append("runtime B stale raw-heavy save was rejected")

        final_state, final_issue = runtime_status.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(final_issue)
        self._assert_state_sane(
            final_state,
            note=self._failure_context(runtime=runtime_status, user_id=user_id, operations=operations, extra="final state malformed after compaction interleave"),
        )
        self.assertEqual(working_memory_state_to_dict(state_a), working_memory_state_to_dict(final_state))
        flattened = self._state_text(final_state).lower()
        self.assertNotIn("stale raw tail from runtime b", flattened)

        status_ok, status = runtime_status.memory_status()
        self.assertTrue(status_ok)
        continuity = status.get("continuity") if isinstance(status.get("continuity"), dict) else {}
        self.assertTrue(bool(continuity.get("healthy")))

    def test_rapid_burst_chat_writes_remain_coherent(self) -> None:
        user_id = "burst-user"
        thread_id = "burst-thread"
        runtime = self._runtime()
        last_prompt = ""
        for index in range(32):
            last_prompt = _large_text(f"rapid burst turn {index}", count=110 if index < 18 else 150)
            ok, body = self._stubbed_chat(
                runtime,
                user_id=user_id,
                thread_id=thread_id,
                text=last_prompt,
                min_context_tokens=8192 if index < 16 else 4096,
            )
            self.assertTrue(ok, msg=f"chat failed at burst turn {index}: {body}")
            self.assertTrue(bool(body.get("ok")), msg=f"chat body not ok at burst turn {index}: {body}")

        final_state, issue = runtime.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(issue)
        self._assert_state_sane(final_state, note="rapid burst produced malformed state")
        self.assertGreater(len(final_state.warm_summaries) + len(final_state.cold_state_blocks), 0)
        self.assertGreaterEqual(len(final_state.hot_turns), 2)
        self.assertEqual(last_prompt, final_state.hot_turns[-2].text)
        self.assertEqual("Acknowledged. Continuing.", final_state.hot_turns[-1].text)
        summary = build_working_memory_summary(final_state)
        self.assertEqual(summary.get("hot_turn_count"), len(final_state.hot_turns))
        self.assertTrue(bool(summary.get("debug")))

    def test_dedupe_pins_and_debug_survive_interleave_structurally(self) -> None:
        user_id = "metadata-user"
        thread_id = "metadata-thread"
        operations: list[str] = []
        runtime_seed = self._runtime()
        runtime_a = self._runtime()
        runtime_b = self._runtime()
        runtime_check = self._runtime()

        runtime_seed.orchestrator()._memory_runtime.set_thread_state(user_id, thread_id=thread_id, current_topic="metadata interleave")  # noqa: SLF001
        base_state = self._make_state("metadata-base", count=8, budget_tokens=4096, pin_first=True, durable=True)
        self.assertTrue(runtime_seed.orchestrator()._memory_runtime.save_working_memory_state(user_id, base_state))  # noqa: SLF001

        state_a, issue_a = runtime_a.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        state_b, issue_b = runtime_b.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(issue_a)
        self.assertIsNone(issue_b)

        append_turn(state_a, role="assistant", text="runtime A extended metadata state")
        manage_working_memory(
            state_a,
            budget=default_budget(4096),
            user_id=user_id,
            thread_id=thread_id,
            durable_ingestor=[].append,  # type: ignore[arg-type]
        )
        self.assertTrue(runtime_a.orchestrator()._memory_runtime.save_working_memory_state(user_id, state_a))  # noqa: SLF001
        operations.append("runtime A saved metadata-rich state")

        append_turn(state_b, role="assistant", text="runtime B stale metadata overwrite")
        self.assertFalse(runtime_b.orchestrator()._memory_runtime.save_working_memory_state(user_id, state_b))  # noqa: SLF001
        operations.append("runtime B stale metadata write was rejected")

        final_state, final_issue = runtime_check.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(final_issue)
        self._assert_state_sane(
            final_state,
            note=self._failure_context(runtime=runtime_check, user_id=user_id, operations=operations, extra="metadata interleave produced malformed state"),
        )
        self.assertEqual(working_memory_state_to_dict(state_a), working_memory_state_to_dict(final_state))
        self.assertEqual("metadata interleave", runtime_check.orchestrator()._memory_runtime.get_thread_state(user_id).get("current_topic"))  # noqa: SLF001
        summary = build_working_memory_summary(final_state)
        self.assertEqual(summary.get("semantic_dedupe_marker_count"), len(final_state.semantic_ingest_hashes))
        debug = summary.get("debug")
        self.assertTrue(debug is None or isinstance(debug, dict))

    def test_memory_status_remains_usable_after_interleaving(self) -> None:
        user_id = "status-user"
        thread_id = "status-thread"
        runtime_a = self._runtime()
        runtime_b = self._runtime()
        runtime_status = self._runtime()

        runtime_a.orchestrator()._memory_runtime.set_thread_state(user_id, thread_id=thread_id, current_topic="status interleave")  # noqa: SLF001
        runtime_a.orchestrator()._memory_runtime.add_pending_item(  # noqa: SLF001
            user_id,
            {
                "thread_id": thread_id,
                "kind": "confirm",
                "question": "keep this pending",
                "created_at": 100,
                "expires_at": 9999999999,
            },
        )
        state_b, issue_b = runtime_b.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(issue_b)
        state_a = self._make_state("status-a", count=9, budget_tokens=4096, durable=True)
        self.assertTrue(runtime_a.orchestrator()._memory_runtime.save_working_memory_state(user_id, state_a))  # noqa: SLF001
        append_turn(state_b, role="user", text="status runtime B follow-up")
        self.assertFalse(runtime_b.orchestrator()._memory_runtime.save_working_memory_state(user_id, state_b))  # noqa: SLF001

        ok, payload = runtime_status.memory_status()
        self.assertTrue(ok)
        continuity = payload.get("continuity") if isinstance(payload.get("continuity"), dict) else {}
        self.assertTrue(bool(continuity.get("healthy")))
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
        self.assertTrue(bool(working_memory.get("healthy")))
        self.assertIn(working_memory.get("status"), {"ok", "missing"})
        summary = working_memory.get("summary") if isinstance(working_memory.get("summary"), dict) else {}
        self.assertGreaterEqual(int(summary.get("hot_turn_count") or 0), 0)
        self.assertGreaterEqual(int(summary.get("total_tokens") or 0), 0)
        self.assertEqual("status interleave", runtime_status.orchestrator()._memory_runtime.get_thread_state(user_id).get("current_topic"))  # noqa: SLF001
        self.assertEqual(1, len(runtime_status.orchestrator()._memory_runtime.list_pending_items(user_id, thread_id=thread_id, include_expired=True)))  # noqa: SLF001
        persistence = user_row.get("persistence") if isinstance(user_row.get("persistence"), dict) else {}
        last_conflict = persistence.get("last_conflict") if isinstance(persistence.get("last_conflict"), dict) else {}
        self.assertEqual("stale_write_conflict", last_conflict.get("reason"))
        self.assertTrue(bool(persistence.get("active_conflict")))

    def test_newest_compacted_state_drives_prompt_after_stale_write_rejection(self) -> None:
        user_id = "grounded-user"
        thread_id = "grounded-thread"
        runtime_seed = self._runtime()
        runtime_a = self._runtime()
        runtime_b = self._runtime()
        runtime_fresh = self._runtime()

        runtime_seed.orchestrator()._memory_runtime.set_thread_state(  # noqa: SLF001
            user_id,
            thread_id=thread_id,
            current_topic="compacted concurrency grounding",
        )
        seed_state = self._make_state("grounded-base", count=10, budget_tokens=None)
        self.assertTrue(runtime_seed.orchestrator()._memory_runtime.save_working_memory_state(user_id, seed_state))  # noqa: SLF001

        state_a, issue_a = runtime_a.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        state_b, issue_b = runtime_b.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(issue_a)
        self.assertIsNone(issue_b)

        manage_working_memory(
            state_a,
            budget=default_budget(4096),
            user_id=user_id,
            thread_id=thread_id,
            durable_ingestor=[].append,  # type: ignore[arg-type]
        )
        self.assertTrue(runtime_a.orchestrator()._memory_runtime.save_working_memory_state(user_id, state_a))  # noqa: SLF001
        append_turn(state_b, role="user", text=_large_text("stale raw overwrite should never reach inference"))
        self.assertFalse(runtime_b.orchestrator()._memory_runtime.save_working_memory_state(user_id, state_b))  # noqa: SLF001

        final_state, final_issue = runtime_fresh.orchestrator()._memory_runtime.load_working_memory_state(user_id)  # noqa: SLF001
        self.assertIsNone(final_issue)
        self.assertGreater(len(final_state.warm_summaries) + len(final_state.cold_state_blocks), 0)

        captured = self._captured_chat_context(
            runtime_fresh,
            user_id=user_id,
            thread_id=thread_id,
            text="What are we continuing right now?",
            min_context_tokens=4096,
        )
        effective_context = "\n\n".join(
            [
                *(str(row.get("content") or "") for row in captured.get("messages") or [] if isinstance(row, dict)),
                str(captured.get("memory_context_text") or ""),
            ]
        ).lower()
        self.assertIn("working memory summaries:", effective_context)
        self.assertNotIn("stale raw overwrite should never reach inference", effective_context)

        status_payload = runtime_fresh.memory_status()[1]
        continuity = status_payload.get("continuity") if isinstance(status_payload.get("continuity"), dict) else {}
        users = continuity.get("users") if isinstance(continuity.get("users"), list) else []
        user_row = next(
            (
                row
                for row in users
                if isinstance(row, dict) and str(row.get("user_id") or "").strip() == user_id
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
        self.assertEqual("working_memory_state", last_conflict.get("kind"))
        self.assertFalse(bool(persistence.get("active_conflict")))


if __name__ == "__main__":
    unittest.main()
