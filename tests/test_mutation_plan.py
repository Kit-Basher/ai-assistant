from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from agent.mutation_plan import (
    MUTATION_PLAN_SCHEMA_VERSION,
    MUTATION_PLAN_STATUS_CANCELLED,
    MUTATION_PLAN_STATUS_EXPIRED,
    MutationPlanStore,
    build_mutation_plan,
    mutation_plan_fingerprint,
    validate_mutation_plan,
)


class MutationPlanTests(unittest.TestCase):
    def test_valid_plan_and_stable_fingerprint(self) -> None:
        plan = build_mutation_plan(
            plan_id="plan-1",
            capability_id="system.package.install",
            executor_id="operator.package.install.v1",
            expires_at_epoch=int(time.time()) + 600,
            thread_id="thread-a",
            session_id="session-a",
            actor_id="user-a",
            target_snapshot={"package": "htop"},
            mutation_inventory=[{"package": "htop", "effect": "install"}],
            recovery={"rollback_supported": False},
        )
        validate_mutation_plan(plan)
        self.assertEqual(MUTATION_PLAN_SCHEMA_VERSION, plan["schema_version"])
        self.assertEqual(plan["plan_fingerprint"], mutation_plan_fingerprint({**plan, "display_text": "ignored"}))

    def test_invalid_schema_and_executor_rejected(self) -> None:
        plan = build_mutation_plan(
            plan_id="plan-1",
            capability_id="system.package.install",
            executor_id="operator.package.install.v1",
            expires_at_epoch=int(time.time()) + 600,
            target_snapshot={"package": "htop"},
        )
        with self.assertRaisesRegex(ValueError, "schema_version"):
            validate_mutation_plan({**plan, "schema_version": 999})
        with self.assertRaisesRegex(ValueError, "executor_id"):
            validate_mutation_plan({**plan, "executor_id": "bad executor"})

    def test_fingerprint_sensitive_to_target_not_display_text(self) -> None:
        plan = build_mutation_plan(
            plan_id="plan-1",
            capability_id="system.package.install",
            executor_id="operator.package.install.v1",
            expires_at_epoch=int(time.time()) + 600,
            target_snapshot={"package": "htop"},
        )
        unchanged = mutation_plan_fingerprint({**plan, "display_text": "Install htop please"})
        changed = mutation_plan_fingerprint({**plan, "target_snapshot": {"package": "jq"}})
        self.assertEqual(plan["plan_fingerprint"], unchanged)
        self.assertNotEqual(plan["plan_fingerprint"], changed)

    def test_plan_store_save_cancel_expire_and_reject_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MutationPlanStore(Path(tmp) / "plans.json")
            plan = build_mutation_plan(
                plan_id="plan-1",
                capability_id="system.package.install",
                executor_id="operator.package.install.v1",
                expires_at_epoch=int(time.time()) + 1,
                target_snapshot={"package": "htop"},
            )
            store.save(plan)
            self.assertEqual(plan["plan_fingerprint"], store.load("plan-1")["plan_fingerprint"])
            cancelled = store.cancel("plan-1")
            self.assertEqual(MUTATION_PLAN_STATUS_CANCELLED, cancelled["status"])
            reused = build_mutation_plan(
                plan_id="plan-1",
                capability_id="system.package.install",
                executor_id="operator.package.install.v1",
                expires_at_epoch=int(time.time()) + 600,
                target_snapshot={"package": "jq"},
            )
            with self.assertRaisesRegex(ValueError, "id_reuse"):
                store.save(reused)

            expired = build_mutation_plan(
                plan_id="plan-2",
                capability_id="system.package.install",
                executor_id="operator.package.install.v1",
                expires_at_epoch=int(time.time()) - 1,
                target_snapshot={"package": "jq"},
            )
            store.save(expired)
            store.prune()
            self.assertEqual(MUTATION_PLAN_STATUS_EXPIRED, store.load("plan-2")["status"])


if __name__ == "__main__":
    unittest.main()
