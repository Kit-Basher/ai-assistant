from __future__ import annotations

import unittest

from agent.memory_contract import (
    PENDING_STATUS_WAITING_FOR_USER,
    build_memory_summary,
    deterministic_memory_snapshot,
    normalize_pending_item,
    normalize_thread_state,
)


class TestMemoryContract(unittest.TestCase):
    def test_normalize_thread_state_sets_defaults_deterministically(self) -> None:
        normalized = normalize_thread_state(
            {"current_topic": "model setup", "runtime_mode": "ready"},
            user_id="u1",
            default_thread_id="user:u1",
            now_ts=123,
        )
        self.assertEqual("user:u1", normalized["thread_id"])
        self.assertEqual("u1", normalized["user_id"])
        self.assertEqual("model setup", normalized["current_topic"])
        self.assertEqual("READY", normalized["runtime_mode"])
        self.assertEqual(123, normalized["updated_at"])
        self.assertEqual("active", normalized["status"])

    def test_normalize_pending_item_enforces_kind_status_and_keys(self) -> None:
        normalized = normalize_pending_item(
            {
                "pending_id": "p1",
                "kind": "unknown_kind",
                "origin_tool": "compare_now",
                "question": "Continue?",
                "options": ["yes", "no", ""],
                "created_at": 10,
                "expires_at": 20,
                "thread_id": "thread-a",
                "status": "bad",
            },
            default_thread_id="thread-a",
            now_ts=15,
        )
        self.assertEqual("p1", normalized["pending_id"])
        self.assertEqual("followup", normalized["kind"])
        self.assertEqual(PENDING_STATUS_WAITING_FOR_USER, normalized["status"])
        self.assertEqual(["yes", "no"], normalized["options"])

    def test_build_memory_summary_is_predictable(self) -> None:
        summary = build_memory_summary(
            thread_state={
                "thread_id": "thread-a",
                "user_id": "u1",
                "current_topic": "health checks",
                "last_tool": "health",
                "runtime_mode": "READY",
                "updated_at": 100,
                "status": "active",
            },
            pending_items=[
                {
                    "pending_id": "p1",
                    "kind": "followup",
                    "origin_tool": "compare_now",
                    "question": "Run compare?",
                    "options": ["yes", "no"],
                    "created_at": 90,
                    "expires_at": 200,
                    "thread_id": "thread-a",
                    "status": "READY_TO_RESUME",
                }
            ],
            last_meaningful_user_request="what changed on my pc",
            last_agent_action="Ran /brief",
            now_ts=100,
        )
        self.assertEqual("health checks", summary["current_topic"])
        self.assertEqual(1, summary["pending_count"])
        self.assertTrue(summary["resumable"])

    def test_deterministic_memory_snapshot_orders_pending_items(self) -> None:
        snapshot = deterministic_memory_snapshot(
            thread_state={
                "thread_id": "thread-a",
                "user_id": "u1",
                "current_topic": "topic",
                "last_tool": "status",
                "runtime_mode": "READY",
                "updated_at": 100,
                "status": "active",
            },
            pending_items=[
                {
                    "pending_id": "p2",
                    "kind": "followup",
                    "question": "q2",
                    "options": [],
                    "created_at": 20,
                    "expires_at": 200,
                    "thread_id": "thread-a",
                    "status": "READY_TO_RESUME",
                },
                {
                    "pending_id": "p1",
                    "kind": "followup",
                    "question": "q1",
                    "options": [],
                    "created_at": 10,
                    "expires_at": 200,
                    "thread_id": "thread-a",
                    "status": "READY_TO_RESUME",
                },
            ],
            last_meaningful_user_request="r",
            last_agent_action="a",
            now_ts=100,
        )
        pending = snapshot["pending_items"]
        self.assertEqual(["p1", "p2"], [row["pending_id"] for row in pending])


if __name__ == "__main__":
    unittest.main()
