from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.executor_registry import ExecutorRegistry, ExecutorSpec


def _plan(**overrides):
    payload = {
        "plan_id": "confirm-test",
        "action_type": "operator.support_bundle",
        "target": "support bundle",
        "risk_level": "low",
        "executor_status": "enabled",
    }
    payload.update(overrides)
    return payload


class ExecutorRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.journal_path = Path(self.tmpdir.name) / "executor_journal.jsonl"

    def test_registry_lookup_by_action_type(self) -> None:
        registry = ExecutorRegistry(self.journal_path)
        spec = ExecutorSpec(executor_id="test.executor", action_type="operator.support_bundle", status="enabled")
        registry.register(spec)
        self.assertEqual(spec, registry.lookup("operator.support_bundle"))

    def test_preview_only_refusal_is_journaled(self) -> None:
        registry = ExecutorRegistry(self.journal_path)
        result = registry.execute_confirmed_plan(
            plan=_plan(action_type="operator.cleanup", executor_status="preview_only"),
            action={"pending_id": "confirm-test"},
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.mutated)
        self.assertEqual("executor_not_enabled", result.error_code)
        self.assertTrue(result.journal_id)
        recent = registry.journal.recent()
        self.assertEqual(1, len(recent))
        self.assertEqual("executor_not_enabled", recent[0]["result"]["error_code"])

    def test_unavailable_refusal_is_journaled(self) -> None:
        registry = ExecutorRegistry(self.journal_path)
        result = registry.execute_confirmed_plan(
            plan=_plan(executor_status="unavailable"),
            action={"pending_id": "confirm-test"},
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.mutated)
        self.assertEqual("executor_unavailable", result.error_code)
        self.assertTrue(result.journal_id)

    def test_enabled_executor_result_schema(self) -> None:
        registry = ExecutorRegistry(self.journal_path)

        def _run(plan, action):
            return {
                "ok": True,
                "mutated": True,
                "resources_touched": ["/tmp/example"],
                "rollback_available": True,
                "rollback_hint": "remove /tmp/example",
                "user_message": "created example",
            }

        registry.register(
            ExecutorSpec(
                executor_id="test.enabled",
                action_type="operator.support_bundle",
                status="enabled",
                run=_run,
            )
        )
        result = registry.execute_confirmed_plan(plan=_plan(), action={"pending_id": "confirm-test"})
        payload = result.to_dict()
        for key in (
            "ok",
            "mutated",
            "executor_id",
            "plan_id",
            "action_type",
            "target",
            "started_at",
            "finished_at",
            "resources_touched",
            "journal_id",
            "rollback_available",
            "rollback_hint",
            "error_code",
            "user_message",
        ):
            self.assertIn(key, payload)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["mutated"])
        self.assertEqual("test.enabled", payload["executor_id"])

    def test_journal_redacts_secrets(self) -> None:
        registry = ExecutorRegistry(self.journal_path)
        result = registry.execute_confirmed_plan(
            plan=_plan(confirmation_token="confirm-secret-token", secret_key="ultrasecret"),
            action={"pending_id": "confirm-test", "confirmation_token": "confirm-secret-token", "api_key": "sk-test"},
        )
        self.assertFalse(result.ok)
        text = self.journal_path.read_text(encoding="utf-8")
        self.assertNotIn("confirm-secret-token", text)
        self.assertNotIn("ultrasecret", text)
        self.assertNotIn("sk-test", text)
        self.assertIn("[REDACTED]", text)

    def test_exact_plan_id_binding_before_execution(self) -> None:
        registry = ExecutorRegistry(self.journal_path)
        called = False

        def _run(plan, action):
            nonlocal called
            called = True
            return {"ok": True, "mutated": True}

        registry.register(ExecutorSpec(executor_id="test.enabled", action_type="operator.support_bundle", status="enabled", run=_run))
        result = registry.execute_confirmed_plan(plan=_plan(plan_id="confirm-a"), action={"pending_id": "confirm-b"})
        self.assertFalse(result.ok)
        self.assertFalse(result.mutated)
        self.assertEqual("plan_id_mismatch", result.error_code)
        self.assertFalse(called)

    def test_executor_exception_reports_no_verified_mutation(self) -> None:
        registry = ExecutorRegistry(self.journal_path)

        def _run(plan, action):
            raise RuntimeError("boom")

        registry.register(ExecutorSpec(executor_id="test.boom", action_type="operator.support_bundle", status="enabled", run=_run))
        result = registry.execute_confirmed_plan(plan=_plan(), action={"pending_id": "confirm-test"})
        self.assertFalse(result.ok)
        self.assertFalse(result.mutated)
        self.assertEqual("executor_exception_before_verified_mutation", result.error_code)
        self.assertEqual("RuntimeError", result.details.get("exception"))


if __name__ == "__main__":
    unittest.main()

