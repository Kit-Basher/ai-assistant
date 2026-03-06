import os
import tempfile
import unittest
import json
from unittest.mock import patch

from agent.knowledge_cache import facts_hash
from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class _FakeChatLLM:
    def __init__(self, *, enabled: bool, text: str = "LLM reply") -> None:
        self._enabled = bool(enabled)
        self._text = text
        self.chat_calls: list[dict[str, object]] = []

    def enabled(self) -> bool:
        return self._enabled

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.chat_calls.append(
            {
                "messages": messages,
                "kwargs": kwargs,
            }
        )
        return {"ok": True, "text": self._text, "provider": "ollama", "model": "llama3"}

    def intent_from_text(self, text: str) -> dict[str, object] | None:
        raise AssertionError(f"intent_from_text should not be called: {text}")


class _RaisingChatLLM:
    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self.chat_calls = 0

    def enabled(self) -> bool:
        return self._enabled

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        _ = messages
        _ = kwargs
        self.chat_calls += 1
        raise RuntimeError("llm chat failure")


class TestOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.log_path = os.path.join(self.tmpdir.name, "events.log")
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def _orchestrator(self) -> Orchestrator:
        return Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )

    def test_handle_message_no_longer_raises(self) -> None:
        orchestrator = self._orchestrator()
        response = orchestrator.handle_message("hello there", "user1")
        self.assertIsInstance(response, OrchestratorResponse)
        self.assertIn("No chat model available", response.text)

    def test_llm_available_routes_free_text_to_llm_chat(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="hi from llm")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        response = orchestrator.handle_message("hello", "user1")
        self.assertEqual("hi from llm", response.text)
        self.assertEqual(1, len(llm.chat_calls))
        call = llm.chat_calls[0]
        kwargs = call.get("kwargs") if isinstance(call, dict) else {}
        self.assertEqual("chat", (kwargs or {}).get("purpose"))
        self.assertNotIn("/brief", response.text.lower())
        messages = call.get("messages") if isinstance(call, dict) else []
        system_text = str((messages or [{}])[0].get("content") if messages else "")
        self.assertIn("Never say you were created by Anthropic/OpenAI", system_text)

    def test_no_llm_available_returns_bootstrap_chat_setup(self) -> None:
        llm = _FakeChatLLM(enabled=False)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        response = orchestrator.handle_message("hello", "user1")
        self.assertIn("No chat model available", response.text)
        self.assertIn("Start Ollama", response.text)
        self.assertEqual([], llm.chat_calls)

    def test_llm_chat_run_directive_executes_internal_brief(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="[[RUN:/brief]]")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        expected = OrchestratorResponse("BRIEF_OUTPUT")
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._llm_chat("user1", "what changed on my pc")
        self.assertEqual("BRIEF_OUTPUT", response.text)
        run_mock.assert_called_once_with("/brief", "user1")

    def test_llm_chat_embedded_run_directive_executes_internal_health(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="Sure — [[RUN:/health]]")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        expected = OrchestratorResponse("HEALTH_OUTPUT")
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._llm_chat("user1", "is my system running ok")
        self.assertEqual("HEALTH_OUTPUT", response.text)
        run_mock.assert_called_once_with("/health", "user1")

    def test_llm_chat_heuristic_fallback_executes_internal_health(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="I can check that for you.")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        expected = OrchestratorResponse("HEALTH_OUTPUT")
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._llm_chat("user1", "show me the stats")
        self.assertEqual("HEALTH_OUTPUT", response.text)
        run_mock.assert_called_once_with("/health", "user1")

    def test_llm_chat_heuristic_health_phrase_executes_internal_health(self) -> None:
        llm = _RaisingChatLLM(enabled=True)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        expected = OrchestratorResponse("HEALTH_OUTPUT")
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._llm_chat("user1", "how is the bot health")
        self.assertEqual("HEALTH_OUTPUT", response.text)
        run_mock.assert_called_once_with("/health", "user1")
        self.assertEqual(0, llm.chat_calls)

    def test_llm_chat_without_run_directive_returns_llm_text(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="Regular chat answer")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        with patch.object(orchestrator, "_handle_message_impl") as run_mock:
            response = orchestrator._llm_chat("user1", "hello")
        self.assertEqual("Regular chat answer", response.text)
        run_mock.assert_not_called()

    def test_llm_chat_keeps_junk_command_suffix_unchanged_when_no_trigger(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="Regular answer /brief /status /help")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        with patch.object(orchestrator, "_handle_message_impl") as run_mock:
            response = orchestrator._llm_chat("user1", "hello there")
        self.assertEqual("Regular answer /brief /status /help", response.text)
        run_mock.assert_not_called()

    def test_llm_chat_sanitizes_untrusted_vendor_identity_claim(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="I am created by Anthropic.")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        with patch.object(orchestrator, "_handle_message_impl") as run_mock:
            response = orchestrator._llm_chat("user1", "hello")
        self.assertIn("Personal Agent", response.text)
        self.assertNotIn("created by Anthropic", response.text)
        run_mock.assert_not_called()

    def test_llm_chat_keeps_vendor_identity_when_provider_matches(self) -> None:
        llm = _FakeChatLLM(enabled=True, text="placeholder")
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        llm.chat = lambda *_args, **_kwargs: {  # type: ignore[assignment]
            "ok": True,
            "text": "I am created by Anthropic.",
            "provider": "anthropic",
            "model": "claude-3.5-sonnet",
        }
        response = orchestrator._llm_chat("user1", "hello")
        self.assertIn("created by Anthropic", response.text)

    def test_llm_chat_exception_with_stats_uses_health_fallback(self) -> None:
        llm = _RaisingChatLLM(enabled=True)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        expected = OrchestratorResponse("HEALTH_OUTPUT")
        with patch.object(orchestrator, "_handle_message_impl", return_value=expected) as run_mock:
            response = orchestrator._llm_chat("user1", "show me the stats")
        self.assertEqual("HEALTH_OUTPUT", response.text)
        run_mock.assert_called_once_with("/health", "user1")
        self.assertEqual(0, llm.chat_calls)

    def test_llm_chat_exception_without_heuristic_returns_friendly_fallback(self) -> None:
        llm = _RaisingChatLLM(enabled=True)
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        with patch.object(orchestrator, "_handle_message_impl") as run_mock:
            response = orchestrator._llm_chat("user1", "tell me a joke")
        self.assertIn("Agent is starting or degraded.", response.text)
        self.assertIn("Next:", response.text)
        self.assertNotIn("Try /brief", response.text)
        run_mock.assert_not_called()
        self.assertEqual(1, llm.chat_calls)

    def test_knowledge_query_cache_and_cta(self) -> None:
        orchestrator = self._orchestrator()
        response = orchestrator.handle_message("what changed this week", "user1")
        self.assertIn("Want my opinion", response.text)
        entry = orchestrator._knowledge_cache.get_recent("user1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.facts_hash, facts_hash(entry.facts))

    def test_opinion_followup_uses_cached_facts(self) -> None:
        orchestrator = self._orchestrator()
        orchestrator.handle_message("what changed this week", "user1")
        entry = orchestrator._knowledge_cache.get_recent("user1")
        response = orchestrator.handle_message("opinion", "user1")
        self.assertIn("source", response.data.get("data", {}))
        self.assertEqual(response.data["data"]["facts_hash"], entry.facts_hash)

    def test_greeting_then_affirmation_stays_in_bootstrap_when_no_llm(self) -> None:
        orch = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )

        def _insert_system_facts(snapshot_id: str, taken_at: str) -> None:
            facts = {
                "schema": {"name": "system_facts", "version": 1},
                "snapshot": {
                    "snapshot_id": snapshot_id,
                    "taken_at": taken_at,
                    "timezone": "UTC",
                    "collector": {
                        "agent_version": "0.6.0",
                        "hostname": "host",
                        "boot_id": "boot",
                        "uptime_s": 1,
                        "collection_duration_ms": 1,
                        "partial": False,
                        "errors": [],
                    },
                    "provenance": {"sources": []},
                },
                "os": {"kernel": {"release": "6.0.0", "arch": "x86_64"}},
                "cpu": {"load": {"load_1m": 0.1, "load_5m": 0.1, "load_15m": 0.1}},
                "memory": {
                    "ram_bytes": {
                        "total": 16 * 1024**3,
                        "used": 2 * 1024**3,
                        "free": 0,
                        "available": 14 * 1024**3,
                        "buffers": 0,
                        "cached": 0,
                    },
                    "swap_bytes": {"total": 0, "free": 0, "used": 0},
                    "pressure": {
                        "psi_supported": False,
                        "memory_some_avg10": None,
                        "io_some_avg10": None,
                        "cpu_some_avg10": None,
                    },
                },
                "filesystems": {
                    "mounts": [
                        {
                            "mountpoint": "/",
                            "device": "/dev/sda1",
                            "fstype": "ext4",
                            "total_bytes": 100 * 1024**3,
                            "used_bytes": 60 * 1024**3,
                            "avail_bytes": 40 * 1024**3,
                            "used_pct": 60.0,
                            "inodes": {"total": None, "used": None, "avail": None, "used_pct": None},
                        }
                    ]
                },
                "process_summary": {"top_cpu": [], "top_mem": []},
                "integrity": {"content_hash_sha256": "0" * 64, "signed": False, "signature": None},
            }
            facts_json = json.dumps(facts, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            self.db.insert_system_facts_snapshot(
                id=snapshot_id,
                user_id="user1",
                taken_at=taken_at,
                boot_id="boot",
                schema_version=1,
                facts_json=facts_json,
                content_hash_sha256="0" * 64,
                partial=False,
                errors_json="[]",
            )

        def observe_handler(ctx, user_id=None):
            _insert_system_facts("snap-1", "2026-02-06T00:00:00+00:00")
            return {"text": "Snapshot taken", "payload": {}}

        orch.skills["observe_now"].functions["observe_now"].handler = observe_handler

        first = orch.handle_message("hello", "user1")
        self.assertIn("no chat model available", first.text.lower())

        second = orch.handle_message("yes please", "user1")
        self.assertIn("no chat model available", second.text.lower())

    def test_done_invalid_id(self) -> None:
        orch = self._orchestrator()
        response = orch.handle_message("/done abc", "user1")
        self.assertIsInstance(response.data, dict)
        self.assertEqual("Usage: /done <id>", response.data["cards"][0]["lines"][0])

    def test_done_nonexistent_id(self) -> None:
        orch = self._orchestrator()
        response = orch.handle_message("/done 9999", "user1")
        self.assertIsInstance(response.data, dict)
        self.assertEqual("Task not found: 9999", response.data["cards"][0]["lines"][0])

    def test_done_marks_done_then_reports_already_done(self) -> None:
        orch = self._orchestrator()
        task_id = self.db.add_task(None, "Write report", 30, 4)

        first = orch.handle_message(f"/done {task_id}", "user1")
        self.assertIsInstance(first.data, dict)
        self.assertEqual(f"Done: [{task_id}] Write report", first.data["cards"][0]["lines"][0])
        task = self.db.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual("done", task["status"])

        second = orch.handle_message(f"/done {task_id}", "user1")
        self.assertIsInstance(second.data, dict)
        self.assertEqual(f"Already done: [{task_id}] Write report", second.data["cards"][0]["lines"][0])

    def test_build_epistemic_context_scopes_memory_to_active_thread(self) -> None:
        orch = self._orchestrator()
        response = OrchestratorResponse(
            "ok",
            {
                "thread_id": "thread-a",
                "thread_label": "Focus",
                "audit_ref": "audit-7",
                "memory_items": [
                    {"ref": "mem:local-a", "thread_id": "thread-a", "relevant": True},
                    {"ref": "mem:global-style", "scope": "global", "relevant": True},
                    {"ref": "mem:other-b", "thread_id": "thread-b", "relevant": True},
                ],
            },
        )
        ctx = orch._build_epistemic_context("user1", response)
        self.assertEqual("thread-a", ctx.active_thread_id)
        self.assertEqual("Focus", ctx.thread_label)
        self.assertEqual(("mem:global-style", "mem:local-a"), ctx.in_scope_memory)
        self.assertEqual(("mem:global-style", "mem:local-a"), ctx.in_scope_memory_ids)
        self.assertEqual(("mem:other-b",), ctx.out_of_scope_memory)
        self.assertTrue(ctx.out_of_scope_relevant_memory)
        self.assertEqual(("audit-7",), ctx.tool_event_ids)
        self.assertEqual(("thread-a:u:1",), ctx.recent_turn_ids)

    def test_starting_new_thread_resets_turn_count(self) -> None:
        orch = self._orchestrator()
        orch._apply_epistemic_layer("user1", "hello", OrchestratorResponse("hello back", {"thread_id": "thread-a"}))

        same_thread_ctx = orch._build_epistemic_context("user1", OrchestratorResponse("ok", {"thread_id": "thread-a"}))
        self.assertGreaterEqual(same_thread_ctx.thread_turn_count, 2)

        new_thread_ctx = orch._build_epistemic_context("user1", OrchestratorResponse("ok", {"thread_id": "thread-b"}))
        self.assertEqual(0, new_thread_ctx.thread_turn_count)

    def test_epistemic_turn_activity_logs_include_thread_id(self) -> None:
        orch = self._orchestrator()
        orch.handle_message("hello there", "user1")
        rows = self.db.activity_log_list_recent("epistemic_turn", limit=2)
        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            payload = row.get("payload") or {}
            self.assertEqual("user1", payload.get("user_id"))
            self.assertTrue(str(payload.get("thread_id") or "").strip())
            self.assertTrue(str(payload.get("turn_id") or "").strip())
            self.assertIn(payload.get("role"), {"user", "assistant"})

    def test_build_epistemic_candidate_populates_missing_provenance(self) -> None:
        orch = self._orchestrator()
        response = OrchestratorResponse(
            "ok",
            {
                "thread_id": "thread-a",
                "audit_ref": "audit-3",
                "memory_items": [{"id": 11, "ref": "mem:11", "thread_id": "thread-a", "relevant": True}],
                "epistemic_candidate_json": json.dumps(
                    {
                        "kind": "answer",
                        "final_answer": "Confirmed.",
                        "clarifying_question": None,
                        "claims": [
                            {"text": "From user", "support": "user", "ref": None},
                            {"text": "From memory", "support": "memory", "ref": "mem:11"},
                            {"text": "From tool", "support": "tool", "ref": None},
                        ],
                        "assumptions": [],
                        "unresolved_refs": [],
                        "thread_refs": [],
                    },
                    ensure_ascii=True,
                ),
            },
        )
        ctx = orch._build_epistemic_context("user1", response)
        candidate = orch._build_epistemic_candidate(response, ctx)
        self.assertFalse(isinstance(candidate, str))
        assert not isinstance(candidate, str)
        self.assertEqual("thread-a:u:1", candidate.claims[0].user_turn_id)
        self.assertEqual("11", candidate.claims[1].memory_id)
        self.assertEqual("audit-3", candidate.claims[2].tool_event_id)


if __name__ == "__main__":
    unittest.main()
