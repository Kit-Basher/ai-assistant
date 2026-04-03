from __future__ import annotations

import tempfile
import unittest

from agent.memory_v2.inject import with_built_context
from agent.memory_v2.retrieval import select_memory
from agent.memory_v2.storage import SQLiteMemoryStore
from agent.memory_v2.types import MemoryItem, MemoryLevel, MemoryQuery


class TestMemoryV2Retrieval(unittest.TestCase):
    def test_recall_relevant_ignore_irrelevant_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/memory.db"
            store = SQLiteMemoryStore(db_path)

            store.append_episodic_event(
                event_id="E-100",
                text="capsule operator checks confusion",
                created_at=1_700_000_000,
                tags={"project": "personal-agent", "topic": "operator_checks"},
                source_kind="audit",
                source_ref="/llm/support/remediate/execute",
            )
            store.upsert_memory_item(
                MemoryItem(
                    id="S-100",
                    level=MemoryLevel.SEMANTIC,
                    text="Project uses determinism-first and never-raise API boundary",
                    created_at=1_700_000_100,
                    updated_at=1_700_000_100,
                    tags={"project": "personal-agent", "topic": "standards"},
                    source_kind="doc",
                    source_ref="PROJECT_STATUS.md",
                    pinned=True,
                )
            )
            store.upsert_memory_item(
                MemoryItem(
                    id="P-100",
                    level=MemoryLevel.PROCEDURAL,
                    text="How to run operator checks checklist: verify health, inspect logs, confirm routing.",
                    created_at=1_700_000_200,
                    updated_at=1_700_000_200,
                    tags={"project": "personal-agent", "topic": "operator_checks"},
                    source_kind="procedure",
                    source_ref="docs/procedures/operator_checks.md",
                    pinned=False,
                )
            )
            store.upsert_memory_item(
                MemoryItem(
                    id="P-999",
                    level=MemoryLevel.PROCEDURAL,
                    text="pokemon badge enamel pins",
                    created_at=1_700_000_300,
                    updated_at=1_700_000_300,
                    tags={"project": "other"},
                    source_kind="note",
                    source_ref="misc",
                    pinned=False,
                )
            )

            query = MemoryQuery(
                text="operator checks again",
                tags={"project": "personal-agent"},
                now_ts=1_700_100_000,
                limit_per_level={
                    MemoryLevel.EPISODIC.value: 3,
                    MemoryLevel.SEMANTIC.value: 3,
                    MemoryLevel.PROCEDURAL.value: 3,
                },
            )
            first = with_built_context(select_memory(query, store))
            second = with_built_context(select_memory(query, store))

            episodic_ids = [item.id for item in first.items_by_level[MemoryLevel.EPISODIC]]
            procedural_ids = [item.id for item in first.items_by_level[MemoryLevel.PROCEDURAL]]
            semantic_ids = [item.id for item in first.items_by_level[MemoryLevel.SEMANTIC]]

            self.assertIn("E-100", episodic_ids)
            self.assertIn("P-100", procedural_ids)
            self.assertNotIn("P-999", procedural_ids)
            self.assertNotIn("P-999", semantic_ids)

            self.assertIn('"capsule operator checks confusion"', first.merged_context_text)
            self.assertIn('"How to run operator checks checklist: verify health, inspect logs, confirm routing."', first.merged_context_text)
            self.assertNotIn("pokemon badge enamel pins", first.merged_context_text)

            first_selected = first.debug.get("selected_ids")
            second_selected = second.debug.get("selected_ids")
            self.assertEqual(first_selected, second_selected)
            self.assertEqual(first.merged_context_text, second.merged_context_text)


if __name__ == "__main__":
    unittest.main()
