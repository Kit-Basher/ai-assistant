from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

from agent.api_server import AgentRuntime
from agent.policy import (
    build_mutator_plan,
    classify_operation,
    mutator_confirmation_required_payload,
    validate_mutator_apply,
)
from tests.test_api_packs_endpoints import _config
from tests.test_managed_local_services import _FakeSearchOpener


class TestPlanModePolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.skills_path = str(Path(__file__).resolve().parents[1] / "skills")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _runtime(self) -> AgentRuntime:
        config = replace(
            _config(self.registry_path, self.db_path, self.skills_path),
            search_enabled=False,
            search_provider="searxng",
            searxng_base_url=None,
        )
        return AgentRuntime(config)

    def test_unknown_operations_default_to_mutating(self) -> None:
        decision = classify_operation("do_magic")

        self.assertEqual("mutating", decision.classification)
        self.assertTrue(decision.requires_plan)
        self.assertTrue(decision.requires_confirmation)
        self.assertFalse(decision.allowed_without_confirmation)
        self.assertEqual("unknown_operations_default_to_mutating", decision.reason)

    def test_read_status_preview_search_are_allowed_without_confirmation(self) -> None:
        for action_type in ("safe_web_search.status", "managed_local_service.setup_plan", "external_pack.preview", "list"):
            decision = classify_operation(action_type)
            self.assertEqual("read_only", decision.classification, action_type)
            self.assertTrue(decision.allowed_without_confirmation, action_type)
            self.assertFalse(decision.requires_confirmation, action_type)

    def test_mutator_without_confirmation_returns_block_payload(self) -> None:
        payload = mutator_confirmation_required_payload("external_pack.install")

        self.assertFalse(payload["ok"])
        self.assertEqual("confirmation_required", payload["error"])
        self.assertEqual("mutating", payload["classification"])
        self.assertTrue(payload["requires_plan"])
        self.assertTrue(payload["requires_confirmation"])

    def test_mutator_plan_requires_expected_shape_and_token(self) -> None:
        plan = build_mutator_plan(
            action_type="external_pack.enable",
            resources={"created": [], "changed": ["pack:abc"], "deleted": []},
            rollback_scope="restore previous enablement metadata",
            rollback_supported=True,
            confirmation_token="confirm-token",
            expires_at=9999999999,
            plan_id="plan-1",
        )

        ok, error = validate_mutator_apply(
            plan,
            expected_action_type="external_pack.enable",
            confirmation_token="confirm-token",
            now=1,
        )
        self.assertTrue(ok)
        self.assertIsNone(error)
        tampered = dict(plan)
        tampered["action_type"] = "external_pack.install"
        ok, error = validate_mutator_apply(
            tampered,
            expected_action_type="external_pack.enable",
            confirmation_token="confirm-token",
            now=1,
        )
        self.assertFalse(ok)
        self.assertEqual("plan_action_type_mismatch", error)

    def test_managed_searxng_setup_plan_includes_policy_mutator_plan(self) -> None:
        runtime = self._runtime()
        body = runtime.search_setup_plan({"base_url": "http://127.0.0.1:8888"})

        plan = body["plan"]
        mutation_plan = plan["mutation_plan"]
        self.assertEqual("managed_local_service.setup_apply", mutation_plan["action_type"])
        self.assertEqual("mutating", mutation_plan["classification"])
        self.assertEqual(plan["confirmation_token"], mutation_plan["confirmation_token"])
        self.assertTrue(mutation_plan["rollback_supported"])
        self.assertIn("runtime_search_config", mutation_plan["resources"]["changed"])

    def test_managed_searxng_apply_rejects_tampered_policy_plan_before_executor(self) -> None:
        runtime = self._runtime()
        runtime._managed_local_service_executor = unittest.mock.Mock()  # noqa: SLF001
        plan_payload = runtime.search_setup_plan({"base_url": "http://127.0.0.1:8888"})
        plan = plan_payload["plan"]
        stored = runtime._search_setup_confirmations[plan["plan_id"]]  # noqa: SLF001
        stored["plan"]["mutation_plan"]["action_type"] = "external_pack.install"

        result = runtime.apply_search_setup({"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]})

        self.assertFalse(result["ok"])
        self.assertEqual("plan_action_type_mismatch", result["error"])
        runtime._managed_local_service_executor.execute_from_pending.assert_not_called()  # type: ignore[union-attr]

    def test_managed_searxng_apply_journals_executed_steps_and_changed_resources(self) -> None:
        runtime = self._runtime()
        plan_payload = runtime.search_setup_plan({"base_url": "http://127.0.0.1:8888"})
        plan = plan_payload["plan"]

        with unittest.mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeSearchOpener({"results": []})):
            result = runtime.apply_search_setup({"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]})

        self.assertTrue(result["ok"])
        journal = result["managed_action_journal"]
        self.assertTrue(journal.get("executed_steps"))
        changed = journal.get("changed_resources")
        self.assertTrue(changed)
        self.assertIn("runtime_search_config", str(changed))
        self.assertIn("search", str(changed))

    def test_pack_lifecycle_mutators_are_centrally_classified(self) -> None:
        runtime = self._runtime()

        for action_type in ("external_pack.install", "external_pack.enable", "external_pack.grant"):
            payload = runtime.plan_mode_policy(action_type)
            self.assertEqual("mutating", payload["classification"], action_type)
            self.assertTrue(payload["requires_plan"], action_type)
            self.assertTrue(payload["requires_confirmation"], action_type)


if __name__ == "__main__":
    unittest.main()
