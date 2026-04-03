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


class TestWorkingMemoryBehavioralReplay(unittest.TestCase):
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
        recent_prompts: list[str],
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
            f"recent_prompts={json.dumps(recent_prompts[-8:], ensure_ascii=True, indent=2)}\n"
            f"working_memory_summary={json.dumps(working_memory, ensure_ascii=True, sort_keys=True, indent=2)}\n"
            f"last_debug={json.dumps(debug, ensure_ascii=True, sort_keys=True, indent=2)}\n"
            f"context_excerpt={self._context_excerpt(context_text)}"
        )

    def _scripted_prompt(self, index: int) -> str:
        if index == 0:
            return "Please remember this preference: I prefer concise answers."
        if index == 1:
            return "Also remember: do not give me giant bullet lists."
        if index == 2:
            return "Decision: use semantic memory as the durable long-term store."
        if index == 3:
            return "Decision: working-memory summaries must stay structured JSON."
        if index == 4:
            return "Open task: we still need to build the replay harness. The remaining follow-up is behavioral validation."
        if index == 5:
            return "Fact: PROJECT_STATUS.md is the current-state handover file for this repo."
        if index == 6:
            return "Fact: the repo root is /home/c/personal-agent."
        if index % 29 == 8:
            return _large_text(
                f"Please inspect this verbose runtime transcript burst {index} and keep going after it.",
                extra="trace",
                count=160,
            )
        if index % 19 == 5:
            return "Continue the replay harness task we discussed earlier and keep the remaining follow-up visible."
        if index % 17 == 9:
            return "Keep answers concise and do not give me giant bullet lists like I asked earlier."
        if index % 13 == 4:
            return _large_text(
                f"Filler planning turn {index}: continue the implementation and preserve context coherently.",
            )
        if index % 11 == 3:
            return "What is the next step for the working-memory replay effort?"
        return _large_text(
            f"Regular filler turn {index}: keep making progress without losing earlier decisions."
        )

    def test_behavioral_replay_preserves_effective_recall_context(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        capture_queries = [
            "Earlier in this conversation, what answer style did I ask for?",
            "Earlier in this conversation, what did we choose as the durable long-term store?",
            "Earlier in this conversation, what task was still unfinished?",
            "Earlier in this conversation, what format did we require for summaries?",
            "Earlier in this conversation, which repo file did we call the current-state handover file?",
        ]
        captured_contexts: dict[str, dict[str, Any]] = {}
        recent_prompts: list[str] = []

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
                text = "Checking remembered context now."
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
                "trace_id": "behavioral-replay-test",
                "data": {"selection": {"selected_model": "ollama:qwen3.5:4b", "provider": "ollama", "reason": "healthy"}},
            }

        with patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference):
            for index in range(160):
                prompt = self._scripted_prompt(index)
                recent_prompts.append(prompt)
                budget = 8192 if index < 90 else 4096
                ok, body = runtime.chat(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "min_context_tokens": budget,
                    }
                )
                self.assertTrue(ok, msg=f"chat failed at scripted turn {index}: {body}")
                self.assertTrue(bool(body.get("ok")), msg=f"chat body not ok at scripted turn {index}: {body}")

            for query in capture_queries:
                recent_prompts.append(query)
                ok, body = runtime.chat(
                    {
                        "messages": [{"role": "user", "content": query}],
                        "min_context_tokens": 4096,
                    }
                )
                self.assertTrue(ok, msg=f"chat failed for replay query {query!r}: {body}")
                self.assertTrue(bool(body.get("ok")), msg=f"chat body not ok for replay query {query!r}: {body}")

        for query in capture_queries:
            self.assertIn(
                query,
                captured_contexts,
                msg=self._failure_context(
                    runtime=runtime,
                    query=query,
                    captured_contexts=captured_contexts,
                    recent_prompts=recent_prompts,
                    extra="replay query was not captured at the inference boundary",
                ),
            )

        preference_context = str(captured_contexts[capture_queries[0]].get("effective_context") or "").lower()
        self.assertIn(
            "prefer concise answers",
            preference_context,
            msg=self._failure_context(
                runtime=runtime,
                query=capture_queries[0],
                captured_contexts=captured_contexts,
                recent_prompts=recent_prompts,
                extra="answer-style preference disappeared from effective replay context",
            ),
        )
        self.assertIn(
            "do not give me giant bullet lists",
            preference_context,
            msg=self._failure_context(
                runtime=runtime,
                query=capture_queries[0],
                captured_contexts=captured_contexts,
                recent_prompts=recent_prompts,
                extra="bullet-list preference disappeared from effective replay context",
            ),
        )

        semantic_context = str(captured_contexts[capture_queries[1]].get("effective_context") or "").lower()
        self.assertTrue(
            "semantic memory" in semantic_context and "durable long-term store" in semantic_context,
            msg=self._failure_context(
                runtime=runtime,
                query=capture_queries[1],
                captured_contexts=captured_contexts,
                recent_prompts=recent_prompts,
                extra="semantic-memory design decision is not supported by the effective replay context",
            ),
        )
        self.assertNotIn(
            "working-memory summaries are the durable long-term store",
            semantic_context,
            msg=self._failure_context(
                runtime=runtime,
                query=capture_queries[1],
                captured_contexts=captured_contexts,
                recent_prompts=recent_prompts,
                extra="semantic-memory decision drifted into a contradictory durable-store claim",
            ),
        )

        open_thread_context = str(captured_contexts[capture_queries[2]].get("effective_context") or "").lower()
        self.assertTrue(
            "replay harness" in open_thread_context or "behavioral validation" in open_thread_context,
            msg=self._failure_context(
                runtime=runtime,
                query=capture_queries[2],
                captured_contexts=captured_contexts,
                recent_prompts=recent_prompts,
                extra="unfinished replay task is missing from the effective replay context",
            ),
        )

        summary_policy_context = str(captured_contexts[capture_queries[3]].get("effective_context") or "").lower()
        self.assertIn(
            "structured json",
            summary_policy_context,
            msg=self._failure_context(
                runtime=runtime,
                query=capture_queries[3],
                captured_contexts=captured_contexts,
                recent_prompts=recent_prompts,
                extra="structured-summary policy is missing from the effective replay context",
            ),
        )

        factual_context = str(captured_contexts[capture_queries[4]].get("effective_context") or "").lower()
        self.assertIn(
            "project_status.md",
            factual_context,
            msg=self._failure_context(
                runtime=runtime,
                query=capture_queries[4],
                captured_contexts=captured_contexts,
                recent_prompts=recent_prompts,
                extra="handover-file fact is missing from the effective replay context",
            ),
        )
        self.assertIn(
            "handover",
            factual_context,
            msg=self._failure_context(
                runtime=runtime,
                query=capture_queries[4],
                captured_contexts=captured_contexts,
                recent_prompts=recent_prompts,
                extra="handover-file meaning drifted out of the effective replay context",
            ),
        )

        memory_ok, memory_status = runtime.memory_status()
        self.assertTrue(memory_ok)
        working_memory = memory_status.get("working_memory") if isinstance(memory_status.get("working_memory"), dict) else {}
        self.assertGreater(int(working_memory.get("warm_summary_count") or 0) + int(working_memory.get("cold_block_count") or 0), 0)
        debug = working_memory.get("debug") if isinstance(working_memory.get("debug"), dict) else {}
        self.assertTrue(bool(debug))


if __name__ == "__main__":
    unittest.main()
