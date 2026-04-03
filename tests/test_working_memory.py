from __future__ import annotations

import unittest

import agent.working_memory as working_memory


def _large_text(prefix: str, *, extra: str = "token") -> str:
    return f"{prefix} " + " ".join([extra] * 160)


def _chunk_turns(label: str, *, count: int = 6) -> list[working_memory.Turn]:
    chunk_state = working_memory.WorkingMemoryState()
    for index in range(count):
        role = "assistant" if index % 2 else "user"
        working_memory.append_turn(
            chunk_state,
            role=role,  # type: ignore[arg-type]
            text=_large_text(f"{label} exchange {index}"),
        )
    return list(chunk_state.hot_turns)


class TestWorkingMemory(unittest.TestCase):
    def test_no_compaction_under_threshold(self) -> None:
        state = working_memory.WorkingMemoryState()
        working_memory.append_turn(state, role="user", text="hello")
        working_memory.append_turn(state, role="assistant", text="hi")

        working_memory.manage_working_memory(
            state,
            budget=working_memory.default_budget(32768),
            user_id="user1",
            thread_id="thread-a",
        )

        self.assertEqual(2, len(state.hot_turns))
        self.assertEqual([], state.warm_summaries)
        self.assertEqual([], state.cold_state_blocks)
        self.assertIsNone(state.last_compaction_action)

    def test_oldest_unpinned_chunk_is_summarized_and_durable_memory_is_extracted(self) -> None:
        state = working_memory.WorkingMemoryState()
        ingested: list[dict[str, object]] = []
        turns = [
            ("user", _large_text("I prefer concise answers and we still need to fix the failing prompt-order test.")),
            ("assistant", _large_text("We decided to keep the existing semantic memory service as the durable store.")),
            ("user", _large_text("Next step is to implement the working memory compactor.")),
            ("assistant", _large_text("I will add structured summaries and preserve chronology.")),
            ("user", _large_text("Please keep the output deterministic and compact.")),
            ("assistant", _large_text("I will avoid prose summaries and keep strict JSON blocks.")),
            ("user", _large_text("The repo path is /home/c/personal-agent and PROJECT_STATUS.md must be updated.")),
            ("assistant", _large_text("Tool result: semantic memory hooks already exist in agent/semantic_memory/service.py.")),
        ]
        for role, text in turns:
            working_memory.append_turn(state, role=role, text=text)  # type: ignore[arg-type]

        working_memory.manage_working_memory(
            state,
            budget=working_memory.default_budget(4096),
            user_id="user1",
            thread_id="thread-a",
            durable_ingestor=ingested.append,
        )

        self.assertGreaterEqual(len(state.warm_summaries), 1)
        self.assertEqual("summarized_raw_chunk", state.last_compaction_action)
        self.assertGreaterEqual(len(ingested), 1)
        durable_payload = ingested[0].get("payload") if isinstance(ingested[0].get("payload"), dict) else {}
        self.assertTrue(durable_payload.get("user_preferences"))
        self.assertTrue(durable_payload.get("open_threads"))
        debug = state.last_compaction_debug
        self.assertEqual("summarized_raw_chunk", debug.get("last_action"))
        self.assertEqual("soft_threshold_exceeded", debug.get("reason"))
        self.assertTrue(bool(debug.get("semantic_extracted")))
        summarized_turn_ids = {
            turn_id
            for block in state.warm_summaries
            for turn_id in block.source_turn_ids
        }
        self.assertIn(state.warm_summaries[0].start_turn_id, summarized_turn_ids)
        self.assertNotIn(state.warm_summaries[0].start_turn_id, {turn.turn_id for turn in state.hot_turns})

    def test_pinned_turns_are_preserved_during_compaction(self) -> None:
        state = working_memory.WorkingMemoryState()
        working_memory.append_turn(
            state,
            role="user",
            text=_large_text("Pinned bug context: do not compact this active investigation yet."),
            pinned=True,
        )
        pinned_turn_id = state.hot_turns[0].turn_id
        for index in range(9):
            role = "assistant" if index % 2 else "user"
            working_memory.append_turn(
                state,
                role=role,  # type: ignore[arg-type]
                text=_large_text(f"Older exchange {index}: keep compacting everything except the pinned task."),
            )

        working_memory.manage_working_memory(
            state,
            budget=working_memory.default_budget(4096),
            user_id="user1",
            thread_id="thread-a",
        )

        self.assertIn(pinned_turn_id, {turn.turn_id for turn in state.hot_turns})
        summarized_turn_ids = {
            turn_id
            for block in state.warm_summaries
            for turn_id in block.source_turn_ids
        }
        self.assertNotIn(pinned_turn_id, summarized_turn_ids)

    def test_summary_merge_occurs_after_raw_chunks_are_exhausted(self) -> None:
        first = working_memory.summarize_turn_chunk(_chunk_turns("first"), compression_level=1)
        second = working_memory.summarize_turn_chunk(_chunk_turns("second"), compression_level=1)
        first.token_count = 500
        second.token_count = 500
        state = working_memory.WorkingMemoryState(
            hot_turns=[],
            warm_summaries=[first, second],
        )

        working_memory.manage_working_memory(
            state,
            budget=working_memory.default_budget(4096),
            user_id="user1",
            thread_id="thread-a",
        )

        self.assertEqual(1, len(state.warm_summaries))
        self.assertEqual(2, state.warm_summaries[0].compression_level)
        self.assertEqual("merged_summaries_l1_to_l2", state.last_compaction_action)
        self.assertEqual("summary_merge", state.warm_summaries[0].derived_from)
        self.assertEqual(2, state.warm_summaries[0].generation_count)

    def test_emergency_trim_preserves_current_task_and_prompt_order(self) -> None:
        state = working_memory.WorkingMemoryState()
        cold = working_memory.summarize_turn_chunk(
            [
                working_memory.Turn(
                    turn_id="t-cold-1",
                    role="user",
                    text="User prefers concise answers.",
                    token_count=30,
                    created_at="2026-04-01T00:00:00+00:00",
                ),
                working_memory.Turn(
                    turn_id="t-cold-2",
                    role="assistant",
                    text="We decided to keep semantic memory as the durable store.",
                    token_count=40,
                    created_at="2026-04-01T00:00:05+00:00",
                ),
            ],
            compression_level=3,
        )
        warm = working_memory.summarize_turn_chunk(
            [
                working_memory.Turn(
                    turn_id="t-warm-1",
                    role="user",
                    text="Next step is to implement the prompt compactor.",
                    token_count=40,
                    created_at="2026-04-01T00:01:00+00:00",
                ),
                working_memory.Turn(
                    turn_id="t-warm-2",
                    role="assistant",
                    text="I will preserve chronology and use structured JSON summaries.",
                    token_count=40,
                    created_at="2026-04-01T00:01:05+00:00",
                ),
            ],
            compression_level=1,
        )
        state.cold_state_blocks.append(cold)
        state.warm_summaries.append(warm)
        for index in range(3):
            working_memory.append_turn(
                state,
                role="tool",
                text=_large_text(f"Verbose tool log {index}", extra="log"),
            )
            state.hot_turns[-1].token_count = 900
        working_memory.append_turn(state, role="assistant", text="I checked the latest failing tests.")
        working_memory.append_turn(state, role="user", text="Please keep working on the dynamic context degradation patch.")
        current_task_id = state.hot_turns[-1].turn_id

        working_memory.emergency_trim(
            state,
            budget=working_memory.default_budget(4096),
        )

        self.assertIn(current_task_id, {turn.turn_id for turn in state.hot_turns})
        context_text = working_memory.build_working_memory_context_text(
            state,
            current_query="dynamic context degradation patch",
            extra_context_text="Retrieved long-term memory:\nPROJECT_STATUS.md tracks release state.",
        )
        self.assertLess(
            context_text.index("Relevant cold state blocks"),
            context_text.index("Working memory summaries"),
        )
        hot_messages = working_memory.build_hot_messages(state)
        self.assertEqual("assistant", hot_messages[-2]["role"])
        self.assertEqual("user", hot_messages[-1]["role"])

    def test_summary_metadata_tracks_raw_provenance(self) -> None:
        block = working_memory.summarize_turn_chunk(_chunk_turns("provenance"), compression_level=1)
        self.assertEqual("raw", block.derived_from)
        self.assertEqual(1, block.generation_count)
        self.assertEqual([], block.child_block_ids)
        self.assertEqual([], block.child_compression_levels)
        self.assertGreater(len(block.source_turn_ids), 1)

    def test_summary_level_never_exceeds_limit(self) -> None:
        first = working_memory.summarize_turn_chunk(_chunk_turns("level-a"), compression_level=2)
        second = working_memory.summarize_turn_chunk(_chunk_turns("level-b"), compression_level=2)
        state = working_memory.WorkingMemoryState(warm_summaries=[first, second])

        working_memory.merge_summaries(state, from_level=2, to_level=99)

        self.assertEqual([], state.warm_summaries)
        self.assertEqual(1, len(state.cold_state_blocks))
        self.assertLessEqual(state.cold_state_blocks[0].compression_level, 3)
        self.assertEqual(3, state.cold_state_blocks[0].compression_level)

    def test_merge_prefers_raw_regeneration_when_source_turns_still_available(self) -> None:
        state = working_memory.WorkingMemoryState()
        for turn in _chunk_turns("regen"):
            state.hot_turns.append(turn)
        first = working_memory.summarize_turn_chunk(state.hot_turns[:3], compression_level=1)
        second = working_memory.summarize_turn_chunk(state.hot_turns[3:6], compression_level=1)
        state.warm_summaries = [first, second]

        working_memory.merge_summaries(state, from_level=1, to_level=2)

        merged = state.warm_summaries[0]
        self.assertEqual("raw", merged.derived_from)
        self.assertEqual(1, merged.generation_count)
        debug = state.last_compaction_debug
        self.assertTrue(bool(debug.get("regenerated_from_raw")))

    def test_recency_shield_keeps_last_user_turns_without_permanent_pins(self) -> None:
        state = working_memory.WorkingMemoryState()
        for index in range(10):
            role = "assistant" if index % 2 else "user"
            working_memory.append_turn(
                state,
                role=role,  # type: ignore[arg-type]
                text=_large_text(f"Recency shield exchange {index}"),
            )
        protected_user_ids = [turn.turn_id for turn in state.hot_turns if turn.role == "user"][-2:]

        working_memory.manage_working_memory(
            state,
            budget=working_memory.default_budget(4096),
            user_id="user1",
            thread_id="thread-a",
        )

        remaining_turn_ids = {turn.turn_id for turn in state.hot_turns}
        self.assertTrue(set(protected_user_ids).issubset(remaining_turn_ids))
        self.assertEqual(set(), state.pinned_turn_ids)
        pinned_remaining = {turn.turn_id for turn in state.hot_turns if turn.pinned}
        self.assertEqual(set(), pinned_remaining)
        summarized_turn_ids = {
            turn_id
            for block in state.warm_summaries
            for turn_id in block.source_turn_ids
        }
        self.assertTrue(bool(summarized_turn_ids))
        self.assertFalse(set(protected_user_ids).intersection(summarized_turn_ids))

    def test_semantic_dedupe_skips_reingest_of_same_material(self) -> None:
        messages = [
            {"role": "user", "content": _large_text("User prefers concise answers and still needs the dynamic context patch.")},
            {"role": "assistant", "content": _large_text("We decided to keep the existing semantic memory service.")},
            {"role": "user", "content": _large_text("Please keep the compactor deterministic.")},
            {"role": "assistant", "content": _large_text("I will preserve chronology and structured summaries.")},
            {"role": "user", "content": _large_text("Next step is to harden panic trim ordering.")},
            {"role": "assistant", "content": _large_text("I will trim low-value tool chatter first.")},
            {"role": "user", "content": _large_text("Remember the repo path /home/c/personal-agent.")},
            {"role": "assistant", "content": _large_text("Tool result: semantic memory hooks already exist.")},
        ]
        ingested: list[dict[str, object]] = []

        first_state = working_memory.rebuild_state_from_messages(messages)
        working_memory.manage_working_memory(
            first_state,
            budget=working_memory.default_budget(4096),
            user_id="user1",
            thread_id="thread-a",
            durable_ingestor=ingested.append,
        )
        second_state = working_memory.rebuild_state_from_messages(messages, previous_state=first_state)
        working_memory.manage_working_memory(
            second_state,
            budget=working_memory.default_budget(4096),
            user_id="user1",
            thread_id="thread-a",
            durable_ingestor=ingested.append,
        )

        self.assertEqual(1, len(ingested))
        self.assertEqual(
            len(second_state.semantic_ingest_hashes),
            len(set(second_state.semantic_ingest_hashes)),
        )
        debug = second_state.last_compaction_debug
        self.assertTrue(bool(debug.get("semantic_dedupe_skipped")))

    def test_semantic_dedupe_allows_changed_material(self) -> None:
        base_messages = [
            {"role": "user", "content": _large_text("User prefers concise answers and still needs the patch.")},
            {"role": "assistant", "content": _large_text("We decided to preserve chronology.")},
            {"role": "user", "content": _large_text("Please keep the compactor deterministic.")},
            {"role": "assistant", "content": _large_text("I will keep summaries structured.")},
            {"role": "user", "content": _large_text("Next step is to harden panic trim ordering.")},
            {"role": "assistant", "content": _large_text("I will trim low-value tool chatter first.")},
            {"role": "user", "content": _large_text("Remember the repo path /home/c/personal-agent.")},
            {"role": "assistant", "content": _large_text("Tool result: semantic memory hooks already exist.")},
        ]
        ingested: list[dict[str, object]] = []

        first_state = working_memory.rebuild_state_from_messages(base_messages)
        working_memory.manage_working_memory(
            first_state,
            budget=working_memory.default_budget(4096),
            user_id="user1",
            thread_id="thread-a",
            durable_ingestor=ingested.append,
        )
        changed_messages = list(base_messages)
        changed_messages[0] = {
            "role": "user",
            "content": _large_text("User prefers concise answers, pinned-task continuity, and still needs the patch."),
        }
        second_state = working_memory.rebuild_state_from_messages(changed_messages, previous_state=first_state)
        working_memory.manage_working_memory(
            second_state,
            budget=working_memory.default_budget(4096),
            user_id="user1",
            thread_id="thread-a",
            durable_ingestor=ingested.append,
        )

        self.assertEqual(2, len(ingested))

    def test_panic_trim_keeps_more_valuable_context_than_tool_logs(self) -> None:
        state = working_memory.WorkingMemoryState()
        cold = working_memory.summarize_turn_chunk(_chunk_turns("old-cold"), compression_level=3)
        warm = working_memory.summarize_turn_chunk(_chunk_turns("old-warm"), compression_level=2)
        cold.token_count = 80
        warm.token_count = 80
        state.cold_state_blocks.append(cold)
        state.warm_summaries.append(warm)
        pinned_text = _large_text("Pinned investigation context must survive.", extra="pin")
        working_memory.append_turn(state, role="assistant", text=pinned_text, pinned=True)
        pinned_turn_id = state.hot_turns[-1].turn_id
        for index in range(3):
            working_memory.append_turn(
                state,
                role="tool",
                text=_large_text(f"Verbose tool log {index}", extra="log"),
            )
            state.hot_turns[-1].token_count = 900
        working_memory.append_turn(state, role="assistant", text=_large_text("Older assistant note that should survive if logs are enough."))
        assistant_turn_id = state.hot_turns[-1].turn_id
        working_memory.append_turn(state, role="user", text="Keep the current task alive.")
        shielded_user_ids = [turn.turn_id for turn in state.hot_turns if turn.role == "user"][-1:]

        working_memory.emergency_trim(state, budget=working_memory.default_budget(4096))

        remaining_turn_ids = {turn.turn_id for turn in state.hot_turns}
        self.assertIn(pinned_turn_id, remaining_turn_ids)
        self.assertTrue(set(shielded_user_ids).issubset(remaining_turn_ids))
        self.assertIn(assistant_turn_id, remaining_turn_ids)
        self.assertEqual(1, len(state.warm_summaries))
        self.assertEqual(1, len(state.cold_state_blocks))
        self.assertLess(len([turn for turn in state.hot_turns if turn.role == "tool"]), 3)
        self.assertEqual("emergency_trim", state.last_compaction_action)
        self.assertTrue(bool(state.last_compaction_debug.get("panic_trim")))

    def test_long_session_compaction_stays_bounded_and_progresses(self) -> None:
        state = working_memory.WorkingMemoryState()
        ingested: list[dict[str, object]] = []
        first_user_turn_id = None
        for round_index in range(8):
            working_memory.append_turn(
                state,
                role="user",
                text=_large_text(f"Long session user turn {round_index}: keep working on the degradation patch."),
            )
            if first_user_turn_id is None:
                first_user_turn_id = state.hot_turns[-1].turn_id
            working_memory.append_turn(
                state,
                role="assistant",
                text=_large_text(f"Long session assistant turn {round_index}: preserving chronology and durable facts."),
            )
            working_memory.manage_working_memory(
                state,
                budget=working_memory.default_budget(4096),
                user_id="user1",
                thread_id="thread-a",
                durable_ingestor=ingested.append,
            )

        self.assertIsNotNone(first_user_turn_id)
        self.assertNotIn(first_user_turn_id, {turn.turn_id for turn in state.hot_turns})
        summarized_turn_ids = {
            turn_id
            for block in [*state.warm_summaries, *state.cold_state_blocks]
            for turn_id in block.source_turn_ids
        }
        self.assertIn(first_user_turn_id, summarized_turn_ids)
        self.assertGreater(len(ingested), 0)
        self.assertEqual(
            len(state.semantic_ingest_hashes),
            len(set(state.semantic_ingest_hashes)),
        )
        self.assertLessEqual(
            max(
                [block.compression_level for block in [*state.warm_summaries, *state.cold_state_blocks]]
                or [0]
            ),
            3,
        )
        self.assertEqual(set(), state.pinned_turn_ids)
        self.assertGreater(len(state.warm_summaries) + len(state.cold_state_blocks), 0)
        self.assertGreater(len(state.hot_turns), 0)


if __name__ == "__main__":
    unittest.main()
