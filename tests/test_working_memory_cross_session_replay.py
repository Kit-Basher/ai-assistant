from __future__ import annotations

import json
import os
import tempfile
import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config


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


def _large_text(prefix: str, *, extra: str = "token", count: int = 120) -> str:
    return f"{prefix} " + " ".join([extra] * count)


class _FakeSemanticMemoryService:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    class _Status:
        enabled = True
        configured = True
        healthy = True
        reason = None
        target = {"provider": "test", "model": "none"}
        index_state = {"state": "ready"}

    def ingest_conversation_text(self, *, source_ref: str, text: str, scope: str, thread_id: str | None, pinned: bool, metadata: dict[str, Any]) -> None:
        self.events.append(
            {
                "kind": "conversation",
                "source_ref": source_ref,
                "text": text,
                "scope": scope,
                "thread_id": thread_id,
                "pinned": pinned,
                "metadata": dict(metadata),
            }
        )

    def status(self) -> _Status:
        return self._Status()

    def report(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "configured": True,
            "healthy": True,
            "reason": None,
            "target": {"provider": "test", "model": "none"},
            "index_state": {"state": "ready"},
            "counts": {"events": len(self.events)},
            "all_counts": {"events": len(self.events)},
            "source_kinds_enabled": ["conversation", "note"],
            "recovery": {
                "state": "ok",
                "recoverable": False,
                "recommended_action": None,
                "needs_reindex": False,
            },
            "summary": "fake semantic memory service ready",
        }

    def ingest_note_text(self, *, source_ref: str, text: str, scope: str, pinned: bool, metadata: dict[str, Any]) -> None:
        self.events.append(
            {
                "kind": "note",
                "source_ref": source_ref,
                "text": text,
                "scope": scope,
                "pinned": pinned,
                "metadata": dict(metadata),
            }
        )


class TestWorkingMemoryCrossSessionReplay(unittest.TestCase):
    USER_ID = "replay-user"
    THREAD_ID = "replay-user:thread"

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

    @staticmethod
    def _effective_context(messages: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            str(row.get("content") or "").strip()
            for row in messages
            if isinstance(row, dict) and str(row.get("content") or "").strip()
        )

    @staticmethod
    def _context_excerpt(text: str, *, limit: int = 2200) -> str:
        cleaned = str(text or "").strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit] + "\n...[truncated]..."

    def _failure_context(
        self,
        *,
        runtime: AgentRuntime,
        query: str,
        captured_contexts: dict[str, dict[str, Any]],
        pre_restore_turns: list[str],
        post_restore_turns: list[str],
        extra: str,
    ) -> str:
        memory_status = runtime.memory_status()[1]
        working_memory = memory_status.get("working_memory") if isinstance(memory_status.get("working_memory"), dict) else {}
        debug = working_memory.get("debug") if isinstance(working_memory.get("debug"), dict) else {}
        captured = captured_contexts.get(query) if isinstance(captured_contexts.get(query), dict) else {}
        context_text = str(captured.get("effective_context") or "")
        return (
            f"{extra}\n"
            f"query={query}\n"
            f"pre_restore_turns={json.dumps(pre_restore_turns[-6:], ensure_ascii=True, indent=2)}\n"
            f"post_restore_turns={json.dumps(post_restore_turns[-6:], ensure_ascii=True, indent=2)}\n"
            f"working_memory_summary={json.dumps(working_memory, ensure_ascii=True, sort_keys=True, indent=2)}\n"
            f"last_debug={json.dumps(debug, ensure_ascii=True, sort_keys=True, indent=2)}\n"
            f"context_excerpt={self._context_excerpt(context_text)}"
        )

    def _scripted_prompt(self, index: int) -> str:
        if index == 0:
            return "Please remember this preference: I prefer concise answers."
        if index == 1:
            return "Also remember: do not use giant bullet lists."
        if index == 2:
            return "Decision: semantic memory is the durable long-term store."
        if index == 3:
            return "Decision: working-memory summaries must stay structured JSON."
        if index == 4:
            return "Open task: we still need cross-session replay validation. Behavioral validation remains unfinished."
        if index == 5:
            return "Fact: PROJECT_STATUS.md is the current-state handover/status file for this repo."
        if index == 6:
            return "Fact: the repo root is /home/c/personal-agent."
        if index % 31 == 8:
            return _large_text(
                f"Please inspect this verbose runtime transcript burst {index} and keep going after it.",
                extra="trace",
                count=170,
            )
        if index % 19 == 5:
            return "Continue the cross-session replay validation task and keep the unfinished follow-up visible."
        if index % 17 == 9:
            return "Keep answers concise and do not use giant bullet lists like I asked earlier."
        if index % 23 == 11:
            return "Reminder: PROJECT_STATUS.md is the current-state handover/status file for this repo."
        if index % 13 == 4:
            return _large_text(
                f"Filler planning turn {index}: continue the implementation and preserve earlier decisions coherently."
            )
        if index % 11 == 3:
            return "What is the next step for the cross-session replay validation effort?"
        return _large_text(
            f"Regular filler turn {index}: keep making progress without losing earlier facts."
        )

    @staticmethod
    def _attach_fake_semantic_service(runtime: AgentRuntime) -> _FakeSemanticMemoryService:
        service = _FakeSemanticMemoryService()
        runtime._semantic_memory_service = service  # type: ignore[assignment]  # noqa: SLF001
        runtime.orchestrator()._semantic_memory_service = service  # type: ignore[assignment]  # noqa: SLF001
        return service

    def test_cross_session_replay_preserves_effective_context_after_restore(self) -> None:
        runtime_before = AgentRuntime(_config(self.registry_path, self.db_path))
        service_before = self._attach_fake_semantic_service(runtime_before)
        pre_restore_turns: list[str] = []
        post_restore_turns: list[str] = []
        capture_queries = [
            "Earlier in this conversation, what answer style did I ask for?",
            "Earlier in this conversation, what did we choose as the durable long-term store?",
            "Earlier in this conversation, what task was still unfinished?",
            "Earlier in this conversation, what format did we require for summaries?",
            "Earlier in this conversation, what repo root path did we mention?",
        ]
        captured_contexts: dict[str, dict[str, Any]] = {}

        def _fake_route_inference(**kwargs: Any) -> dict[str, Any]:
            user_text = str(kwargs.get("user_text") or "").strip()
            messages = deepcopy(list(kwargs.get("messages") or []))
            effective_context = self._effective_context(messages)
            if user_text in capture_queries:
                captured_contexts[user_text] = {
                    "messages": messages,
                    "effective_context": effective_context,
                }
            if "verbose runtime transcript burst" in user_text:
                text = _large_text(
                    f"Verbose tool-like output for {user_text}",
                    extra="log",
                    count=220,
                )
            elif user_text in capture_queries:
                text = "Checking restored remembered context now."
            else:
                text = "Acknowledged. Continuing."
            return {
                "ok": True,
                "text": text,
                "provider": "ollama",
                "model": "ollama:qwen3.5:4b",
                "task_type": "chat",
                "selection_reason": "healthy+approved+local_first+task=chat",
                "fallback_used": False,
                "error_kind": None,
                "next_action": None,
                "trace_id": "cross-session-replay-test",
                "data": {"selection": {"selected_model": "ollama:qwen3.5:4b", "provider": "ollama", "reason": "healthy"}},
            }

        with patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference):
            for index in range(145):
                prompt = self._scripted_prompt(index)
                pre_restore_turns.append(prompt)
                budget = 8192 if index < 90 else 4096
                ok, body = runtime_before.chat(
                    {
                        "user_id": self.USER_ID,
                        "thread_id": self.THREAD_ID,
                        "messages": [{"role": "user", "content": prompt}],
                        "min_context_tokens": budget,
                    }
                )
                self.assertTrue(ok, msg=f"pre-restore chat failed at turn {index}: {body}")
                self.assertTrue(bool(body.get("ok")), msg=f"pre-restore chat body not ok at turn {index}: {body}")

            persisted_status_ok, persisted_status = runtime_before.memory_status()
            self.assertTrue(persisted_status_ok)
            persisted_working = persisted_status.get("working_memory") if isinstance(persisted_status.get("working_memory"), dict) else {}
            self.assertGreater(
                int(persisted_working.get("warm_summary_count") or 0) + int(persisted_working.get("cold_block_count") or 0),
                0,
            )

            memory_runtime_before = runtime_before.orchestrator()._memory_runtime  # noqa: SLF001
            persisted_state, persisted_issue = memory_runtime_before.load_working_memory_state(self.USER_ID)
            self.assertIsNone(persisted_issue)
            self.assertGreater(len(persisted_state.warm_summaries) + len(persisted_state.cold_state_blocks), 0)
            self.assertGreater(len(persisted_state.semantic_ingest_hashes), 0)

            runtime_after = AgentRuntime(_config(self.registry_path, self.db_path))
            service_after = self._attach_fake_semantic_service(runtime_after)
            memory_runtime_after = runtime_after.orchestrator()._memory_runtime  # noqa: SLF001
            restored_state, restored_issue = memory_runtime_after.load_working_memory_state(self.USER_ID)
            self.assertIsNone(
                restored_issue,
                msg=self._failure_context(
                    runtime=runtime_after,
                    query="restore",
                    captured_contexts=captured_contexts,
                    pre_restore_turns=pre_restore_turns,
                    post_restore_turns=post_restore_turns,
                    extra=f"restored working-memory state is corrupt: {restored_issue}",
                ),
            )
            self.assertGreater(len(restored_state.warm_summaries) + len(restored_state.cold_state_blocks), 0)
            self.assertEqual(persisted_state.semantic_ingest_hashes, restored_state.semantic_ingest_hashes)
            for block in [*restored_state.warm_summaries, *restored_state.cold_state_blocks]:
                self.assertTrue(bool(block.source_turn_ids))
                self.assertIn(block.derived_from, {"raw", "summary_merge"})
                self.assertLessEqual(int(block.compression_level), 3)
                self.assertGreaterEqual(int(block.generation_count), 1)

            restored_status_ok, restored_status = runtime_after.memory_status()
            self.assertTrue(restored_status_ok)
            restored_working = restored_status.get("working_memory") if isinstance(restored_status.get("working_memory"), dict) else {}
            self.assertTrue(bool(restored_working.get("debug")))
            self.assertGreaterEqual(int(restored_working.get("tracked_user_count") or 0), 1)

            for index in range(145, 169):
                prompt = self._scripted_prompt(index)
                post_restore_turns.append(prompt)
                ok, body = runtime_after.chat(
                    {
                        "user_id": self.USER_ID,
                        "thread_id": self.THREAD_ID,
                        "messages": [{"role": "user", "content": prompt}],
                        "min_context_tokens": 3072,
                    }
                )
                self.assertTrue(ok, msg=f"post-restore chat failed at turn {index}: {body}")
                self.assertTrue(bool(body.get("ok")), msg=f"post-restore chat body not ok at turn {index}: {body}")

            self.assertGreaterEqual(len(service_before.events), 1)
            self.assertGreaterEqual(len(service_after.events), 1)

            for query in capture_queries:
                post_restore_turns.append(query)
                ok, body = runtime_after.chat(
                    {
                        "user_id": self.USER_ID,
                        "thread_id": self.THREAD_ID,
                        "messages": [{"role": "user", "content": query}],
                        "min_context_tokens": 3072,
                    }
                )
                self.assertTrue(ok, msg=f"post-restore replay query failed for {query!r}: {body}")
                self.assertTrue(bool(body.get("ok")), msg=f"post-restore replay body not ok for {query!r}: {body}")

        for query in capture_queries:
            self.assertIn(
                query,
                captured_contexts,
                msg=self._failure_context(
                    runtime=runtime_after,
                    query=query,
                    captured_contexts=captured_contexts,
                    pre_restore_turns=pre_restore_turns,
                    post_restore_turns=post_restore_turns,
                    extra="post-restore replay query was not captured at the inference boundary",
                ),
            )

        preference_context = str(captured_contexts[capture_queries[0]].get("effective_context") or "").lower()
        self.assertIn(
            "prefer concise answers",
            preference_context,
            msg=self._failure_context(
                runtime=runtime_after,
                query=capture_queries[0],
                captured_contexts=captured_contexts,
                pre_restore_turns=pre_restore_turns,
                post_restore_turns=post_restore_turns,
                extra="post-restore preference recall disappeared from the effective inference context",
            ),
        )
        self.assertIn(
            "do not use giant bullet lists",
            preference_context,
            msg=self._failure_context(
                runtime=runtime_after,
                query=capture_queries[0],
                captured_contexts=captured_contexts,
                pre_restore_turns=pre_restore_turns,
                post_restore_turns=post_restore_turns,
                extra="post-restore bullet-list preference disappeared from the effective inference context",
            ),
        )

        durable_store_context = str(captured_contexts[capture_queries[1]].get("effective_context") or "").lower()
        self.assertTrue(
            "semantic memory" in durable_store_context and "durable long-term store" in durable_store_context,
            msg=self._failure_context(
                runtime=runtime_after,
                query=capture_queries[1],
                captured_contexts=captured_contexts,
                pre_restore_turns=pre_restore_turns,
                post_restore_turns=post_restore_turns,
                extra="post-restore durable-store decision is missing from the effective inference context",
            ),
        )
        self.assertNotIn(
            "working-memory summaries are the durable long-term store",
            durable_store_context,
            msg=self._failure_context(
                runtime=runtime_after,
                query=capture_queries[1],
                captured_contexts=captured_contexts,
                pre_restore_turns=pre_restore_turns,
                post_restore_turns=post_restore_turns,
                extra="post-restore durable-store decision drifted into a contradictory summaries-as-store claim",
            ),
        )

        open_thread_context = str(captured_contexts[capture_queries[2]].get("effective_context") or "").lower()
        self.assertTrue(
            "cross-session replay validation" in open_thread_context or "behavioral validation remains unfinished" in open_thread_context,
            msg=self._failure_context(
                runtime=runtime_after,
                query=capture_queries[2],
                captured_contexts=captured_contexts,
                pre_restore_turns=pre_restore_turns,
                post_restore_turns=post_restore_turns,
                extra="post-restore unfinished task disappeared from the effective inference context",
            ),
        )

        summary_policy_context = str(captured_contexts[capture_queries[3]].get("effective_context") or "").lower()
        self.assertIn(
            "structured json",
            summary_policy_context,
            msg=self._failure_context(
                runtime=runtime_after,
                query=capture_queries[3],
                captured_contexts=captured_contexts,
                pre_restore_turns=pre_restore_turns,
                post_restore_turns=post_restore_turns,
                extra="post-restore structured-summary rule disappeared from the effective inference context",
            ),
        )

        factual_context = str(captured_contexts[capture_queries[4]].get("effective_context") or "").lower()
        combined_context = "\n".join(
            str((captured_contexts.get(query) or {}).get("effective_context") or "").lower()
            for query in capture_queries
        )
        self.assertIn(
            "/home/c/personal-agent",
            factual_context,
            msg=self._failure_context(
                runtime=runtime_after,
                query=capture_queries[4],
                captured_contexts=captured_contexts,
                pre_restore_turns=pre_restore_turns,
                post_restore_turns=post_restore_turns,
                extra="post-restore repo-root fact disappeared from the effective inference context",
            ),
        )
        self.assertIn(
            "project_status.md",
            combined_context,
            msg=self._failure_context(
                runtime=runtime_after,
                query=capture_queries[4],
                captured_contexts=captured_contexts,
                pre_restore_turns=pre_restore_turns,
                post_restore_turns=post_restore_turns,
                extra="post-restore handover/status file fact disappeared from the effective inference context",
            ),
        )
        self.assertTrue(
            "handover" in combined_context or "status file" in combined_context,
            msg=self._failure_context(
                runtime=runtime_after,
                query=capture_queries[4],
                captured_contexts=captured_contexts,
                pre_restore_turns=pre_restore_turns,
                post_restore_turns=post_restore_turns,
                extra="post-restore handover/status file meaning drifted out of the effective inference context",
            ),
        )


if __name__ == "__main__":
    unittest.main()
