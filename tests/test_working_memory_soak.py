from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections import Counter

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.working_memory import (
    ContextBudget,
    WorkingMemoryState,
    append_turn,
    build_hot_messages,
    build_working_memory_context_text,
    build_working_memory_summary,
    manage_working_memory,
    normalize_working_memory_state,
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
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _large_text(prefix: str, *, extra: str = "token", count: int = 120) -> str:
    return f"{prefix} " + " ".join([extra] * count)


class TestWorkingMemorySoak(unittest.TestCase):
    NUM_TURNS = 260

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
    def _normalized_ingest_signature(payload: dict[str, object]) -> str:
        normalized = {
            "source_ref": str(payload.get("source_ref") or "").strip(),
            "raw_text": " ".join(str(payload.get("raw_text") or "").split()).strip().lower(),
            "text": " ".join(str(payload.get("text") or "").split()).strip().lower(),
            "payload": payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
        }
        return json.dumps(normalized, ensure_ascii=True, sort_keys=True)

    @staticmethod
    def _flatten_summary_text(state: WorkingMemoryState, ingested: list[dict[str, object]]) -> str:
        parts: list[str] = []
        for turn in state.hot_turns:
            parts.append(str(turn.text or ""))
        for block in [*state.warm_summaries, *state.cold_state_blocks]:
            parts.extend(block.facts)
            parts.extend(block.decisions)
            parts.extend(block.open_threads)
            parts.extend(block.user_preferences)
            parts.extend(block.artifacts)
            parts.extend(block.tool_results)
        for row in ingested:
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            for key in ("facts", "decisions", "open_threads", "user_preferences", "artifacts", "tool_results"):
                values = payload.get(key) if isinstance(payload.get(key), list) else []
                parts.extend(str(item) for item in values)
        return "\n".join(part for part in parts if str(part).strip())

    def _failure_context(
        self,
        *,
        recent_turns: list[dict[str, object]],
        state: WorkingMemoryState,
        extra: str,
    ) -> str:
        summary = build_working_memory_summary(state)
        return (
            f"{extra}\n"
            f"recent_turns={json.dumps(recent_turns[-10:], ensure_ascii=True, indent=2)}\n"
            f"working_memory_summary={json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2)}\n"
            f"last_debug={json.dumps(state.last_compaction_debug, ensure_ascii=True, sort_keys=True, indent=2)}"
        )

    def _generate_turn(self, index: int) -> dict[str, object]:
        if index in {0, 61, 123, 187, 241}:
            return {
                "role": "user",
                "text": _large_text(
                    "We still need to implement the dynamic working-memory panic trim ordering for feature X. "
                    "Next step: preserve the current coding task and keep the prompt coherent."
                ),
                "pinned": index == 0,
            }
        if index in {6, 73, 149, 211}:
            return {
                "role": "assistant",
                "text": _large_text(
                    "We decided to keep semantic memory as the durable store and preserve chronology in structured summaries."
                ),
                "pinned": False,
            }
        if index in {12, 88, 167, 229}:
            return {
                "role": "user",
                "text": _large_text(
                    "I prefer concise answers and direct responses. Please keep the wording practical and compact."
                ),
                "pinned": False,
            }
        if index in {19, 97, 174, 248}:
            return {
                "role": "user",
                "text": _large_text(
                    "Continue the feature we discussed earlier. We still need the summary drift safeguards and the panic trim fix."
                ),
                "pinned": False,
            }
        if index in {33, 111, 203}:
            return {
                "role": "assistant",
                "text": _large_text(
                    "The project uses a local-first architecture with deterministic runtime truth and bounded native actions."
                ),
                "pinned": False,
            }
        if index in {45, 46, 47, 158, 159, 160, 236, 237, 238}:
            return {
                "role": "tool",
                "text": _large_text(
                    f"DEBUG LOG {index}: repeated verbose tool output with duplicated lines, trace noise, and low-signal stdout.",
                    extra="log",
                    count=220,
                ),
                "forced_token_count": 1100,
                "pinned": False,
            }
        if index % 23 == 7:
            return {
                "role": "user",
                "text": _large_text(
                    "Open thread: we still need to verify the recency shield and avoid losing unresolved tasks."
                ),
                "pinned": False,
            }
        if index % 17 == 9:
            return {
                "role": "assistant",
                "text": _large_text(
                    "I will keep the working-memory summaries structured, bounded, and easy to inspect in status."
                ),
                "pinned": False,
            }
        if index % 13 == 5:
            return {
                "role": "user",
                "text": _large_text(
                    "Random note: PROJECT_STATUS.md should stay factual, and the repo path is /home/c/personal-agent."
                ),
                "pinned": False,
            }
        if index % 9 == 4:
            return {
                "role": "assistant",
                "text": _large_text("Low-value chatter: acknowledged, continuing now, keeping momentum."),
                "pinned": False,
            }
        return {
            "role": "user" if index % 2 == 0 else "assistant",
            "text": _large_text(
                f"Filler exchange {index}: continue normally, keep context coherent, and avoid losing earlier decisions."
            ),
            "pinned": False,
        }

    def test_working_memory_soak_preserves_continuity_and_bounds(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime_memory = runtime.orchestrator()._memory_runtime  # noqa: SLF001
        state = WorkingMemoryState()
        early_budget = ContextBudget(max_context_tokens=8192)
        late_budget = ContextBudget(max_context_tokens=4096)
        semantic_ingests: list[dict[str, object]] = []
        action_history: list[str] = []
        status_actions_seen: list[str] = []
        token_history: list[int] = []
        recent_turns: list[dict[str, object]] = []
        pinned_turn_id: str | None = None
        first_recent_user_turn_id: str | None = None
        aged_user_turn_id: str | None = None
        aged_user_turn_checked = False
        first_merge_index: int | None = None
        first_panic_index: int | None = None
        last_status_debug: dict[str, object] = {}
        saw_status_panic_event = False

        for index in range(self.NUM_TURNS):
            generated = self._generate_turn(index)
            role = str(generated.get("role") or "user")
            text = str(generated.get("text") or "")
            pinned = bool(generated.get("pinned", False))
            append_turn(state, role=role, text=text, pinned=pinned)  # type: ignore[arg-type]
            if "forced_token_count" in generated:
                state.hot_turns[-1].token_count = int(generated.get("forced_token_count") or state.hot_turns[-1].token_count)
            current_turn_id = state.hot_turns[-1].turn_id
            recent_turns.append(
                {
                    "index": index,
                    "role": role,
                    "turn_id": current_turn_id,
                    "text": text[:160],
                }
            )
            if pinned and pinned_turn_id is None:
                pinned_turn_id = current_turn_id
            if role == "user" and first_recent_user_turn_id is None:
                first_recent_user_turn_id = current_turn_id
            if role == "user" and aged_user_turn_id is None and index >= 12:
                aged_user_turn_id = current_turn_id

            budget = early_budget if index < 150 else late_budget
            manage_working_memory(
                state,
                budget=budget,
                user_id="soak-user",
                thread_id="thread-soak",
                durable_ingestor=semantic_ingests.append,
            )
            runtime_memory.save_working_memory_state("soak-user", state)
            loaded_state, issue = runtime_memory.load_working_memory_state("soak-user")
            self.assertIsNone(
                issue,
                msg=self._failure_context(
                    recent_turns=recent_turns,
                    state=state,
                    extra=f"working memory load failed at turn {index}: {issue}",
                ),
            )
            state = loaded_state
            self.assertEqual(
                state.last_token_usage.get("total_tokens"),
                build_working_memory_summary(state).get("total_tokens"),
                msg=self._failure_context(
                    recent_turns=recent_turns,
                    state=state,
                    extra=f"token summary mismatch at turn {index}",
                ),
            )
            for block in [*state.warm_summaries, *state.cold_state_blocks]:
                self.assertTrue(
                    bool(block.source_turn_ids),
                    msg=self._failure_context(
                        recent_turns=recent_turns,
                        state=state,
                        extra=f"empty source_turn_ids at turn {index}",
                    ),
                )
                self.assertLessEqual(
                    int(block.compression_level),
                    3,
                    msg=self._failure_context(
                        recent_turns=recent_turns,
                        state=state,
                        extra=f"compression level exceeded bounds at turn {index}",
                    ),
                )
                self.assertGreaterEqual(
                    int(block.generation_count),
                    1,
                    msg=self._failure_context(
                        recent_turns=recent_turns,
                        state=state,
                        extra=f"invalid generation count at turn {index}",
                    ),
                )
                self.assertIn(
                    block.derived_from,
                    {"raw", "summary_merge"},
                    msg=self._failure_context(
                        recent_turns=recent_turns,
                        state=state,
                        extra=f"invalid summary provenance at turn {index}",
                    ),
                )
            normalized = normalize_working_memory_state(working_memory_state_to_dict(state))
            self.assertEqual(
                build_working_memory_summary(state).get("total_tokens"),
                build_working_memory_summary(normalized).get("total_tokens"),
                msg=self._failure_context(
                    recent_turns=recent_turns,
                    state=state,
                    extra=f"normalization changed state unexpectedly at turn {index}",
                ),
            )

            hot_turn_ids = {turn.turn_id for turn in state.hot_turns}
            if role == "user":
                self.assertIn(
                    current_turn_id,
                    hot_turn_ids,
                    msg=self._failure_context(
                        recent_turns=recent_turns,
                        state=state,
                        extra=f"recent user turn compacted too early at turn {index}",
                    ),
                )
            if aged_user_turn_id and not aged_user_turn_checked and index >= 80:
                summarized_turn_ids = {
                    turn_id
                    for block in [*state.warm_summaries, *state.cold_state_blocks]
                    for turn_id in block.source_turn_ids
                }
                if aged_user_turn_id not in hot_turn_ids:
                    self.assertIn(
                        aged_user_turn_id,
                        summarized_turn_ids,
                        msg=self._failure_context(
                            recent_turns=recent_turns,
                            state=state,
                            extra=f"aged user turn was lost instead of summarized by turn {index}",
                        ),
                    )
                    aged_user_turn_checked = True

            prompt_context = build_working_memory_context_text(
                state,
                current_query=text,
                extra_context_text="Retrieved long-term memory: deterministic local runtime.",
            )
            hot_messages = build_hot_messages(state)
            self.assertTrue(
                bool(hot_messages),
                msg=self._failure_context(
                    recent_turns=recent_turns,
                    state=state,
                    extra=f"prompt became empty at turn {index}",
                ),
            )
            self.assertLess(
                len(prompt_context),
                25000,
                msg=self._failure_context(
                    recent_turns=recent_turns,
                    state=state,
                    extra=f"prompt context grew unexpectedly at turn {index}",
                ),
            )

            if state.last_compaction_action:
                action_history.append(state.last_compaction_action)
                if first_merge_index is None and state.last_compaction_action.startswith("merged_summaries_"):
                    first_merge_index = index
                if first_panic_index is None and state.last_compaction_action == "emergency_trim":
                    first_panic_index = index
            token_history.append(int(state.last_token_usage.get("total_tokens") or 0))

            if index % 40 == 0 or state.last_compaction_action == "emergency_trim":
                ok, payload = runtime.memory_status()
                self.assertTrue(
                    ok,
                    msg=self._failure_context(
                        recent_turns=recent_turns,
                        state=state,
                        extra=f"/memory/status failed at turn {index}",
                    ),
                )
                working_memory = payload.get("working_memory") if isinstance(payload.get("working_memory"), dict) else {}
                debug = working_memory.get("debug") if isinstance(working_memory.get("debug"), dict) else {}
                last_status_debug = debug
                if debug:
                    action = str(debug.get("last_action") or "").strip()
                    if action:
                        status_actions_seen.append(action)
                if str(state.last_compaction_action or "").strip() == "emergency_trim":
                    saw_status_panic_event = True
                    self.assertTrue(
                        bool(debug.get("panic_trim")),
                        msg=self._failure_context(
                            recent_turns=recent_turns,
                            state=state,
                            extra=f"panic trim not visible in status at turn {index}",
                        ),
                    )

        flattened = self._flatten_summary_text(state, semantic_ingests).lower()
        self.assertIn(
            "concise answers",
            flattened,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="user preference fact drifted or disappeared",
            ),
        )
        self.assertTrue(
            "working-memory summaries structured" in flattened or "easy to inspect in status" in flattened,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="working-memory design thread drifted or disappeared",
            ),
        )
        self.assertIn(
            "panic trim",
            flattened,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="active coding thread disappeared",
            ),
        )
        self.assertIn(
            "local-first architecture",
            flattened,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="system design fact disappeared",
            ),
        )

        unique_signatures = {self._normalized_ingest_signature(payload) for payload in semantic_ingests}
        self.assertEqual(
            len(unique_signatures),
            len(semantic_ingests),
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="semantic ingest duplicated identical payloads",
            ),
        )
        self.assertEqual(
            len(state.semantic_ingest_hashes),
            len(set(state.semantic_ingest_hashes)),
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="semantic dedupe markers contain duplicates",
            ),
        )
        self.assertGreater(
            len(semantic_ingests),
            0,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="semantic ingestion never happened",
            ),
        )

        self.assertTrue(
            aged_user_turn_checked,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="recency shield never aged out into summaries",
            ),
        )
        self.assertGreater(
            len(state.warm_summaries) + len(state.cold_state_blocks),
            0,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="compaction never produced summaries",
            ),
        )
        self.assertIsNotNone(
            first_merge_index,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="summary merges never occurred",
            ),
        )
        self.assertGreater(
            state.emergency_trim_count,
            0,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="panic trim never triggered",
            ),
        )
        self.assertIsNotNone(
            first_panic_index,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="panic-trim action never recorded",
            ),
        )
        self.assertLess(
            int(first_merge_index or 0),
            int(first_panic_index or 10**9),
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="panic trim occurred before any summary merge",
            ),
        )
        self.assertIsNotNone(
            pinned_turn_id,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="pinned task was never created",
            ),
        )
        self.assertIn(
            str(pinned_turn_id),
            {turn.turn_id for turn in state.hot_turns},
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="pinned task was lost during soak",
            ),
        )
        self.assertIn(
            "emergency_trim",
            set(status_actions_seen),
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="status surface never reported panic trim",
            ),
        )
        self.assertTrue(
            saw_status_panic_event,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="status surface never exposed a panic-trim debug event",
            ),
        )
        self.assertTrue(
            bool(last_status_debug),
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="status surface never exposed compaction debug metadata",
            ),
        )
        self.assertLessEqual(
            max([block.compression_level for block in [*state.warm_summaries, *state.cold_state_blocks]] or [0]),
            3,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="compression level exceeded max bound",
            ),
        )
        self.assertGreater(
            Counter(action_history)["summarized_raw_chunk"],
            0,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="raw chunk summarization never happened",
            ),
        )
        self.assertTrue(
            any(action.startswith("merged_summaries_") for action in action_history),
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="summary merging never happened",
            ),
        )
        self.assertGreater(
            max(token_history) - min(token_history),
            200,
            msg=self._failure_context(
                recent_turns=recent_turns,
                state=state,
                extra="token usage never varied enough to exercise compaction",
            ),
        )


if __name__ == "__main__":
    unittest.main()
